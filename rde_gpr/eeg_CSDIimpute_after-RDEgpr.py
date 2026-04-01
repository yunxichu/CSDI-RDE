# -*- coding: utf-8 -*-
"""
EEG 预测脚本（RDE-GPR）

支持两种模式：
1. 空间集成（无延迟）：同时刻不同通道组合
2. 延迟嵌入（推荐）：不同时刻的通道延迟组合

示例（延迟嵌入）：
python rde_gpr/eeg_CSDIimpute_after-RDEgpr.py \
  --imputed_path ./save/eeg_imputed_random_ratio0.5_seed42_20260331_131907/eeg_imputed.npy \
  --ground_path ./data/eeg/eeg_ground.npy \
  --horizon_steps 24 \
  --L 4 --s 50 --trainlength 100 --n_jobs 2 \
  --use_delay_embedding \
  --max_delay 20 \
  --missing_ratio 0.5 \
  --missing_mode random \
  --target_indices "0,1,2"
"""

import os
import sys
import json
import time
import random
import argparse
import datetime
import itertools
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize


def set_global_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_json_dump(obj, path):
    def convert(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        if isinstance(o, list):
            return [convert(v) for v in o]
        return o
    with open(path, "w") as f:
        json.dump(convert(obj), f, indent=4)


def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan}
    y_true_m = y_true[mask]
    y_pred_m = y_pred[mask]
    rmse = np.sqrt(np.mean((y_true_m - y_pred_m) ** 2))
    mae = np.mean(np.abs(y_true_m - y_pred_m))
    return {"rmse": rmse, "mae": mae}


class GaussianProcessRegressor:
    def __init__(self, kernel='rbf', noise=1e-6):
        self.kernel = kernel
        self.noise = float(noise)
        self.X_train = None
        self.y_train = None
        self.L = None
        self.alpha = None
        self.params = None
        self.mu_X, self.sigma_X = None, None
        self.mu_y, self.sigma_y = None, None

    def _rbf_kernel(self, X1, X2, sigma_f, l):
        sqdist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return (sigma_f**2) * np.exp(-sqdist / (2.0 * (l**2)))

    def _kernel_matrix(self, X1, X2):
        sigma_f, l, sigma_n = self.params
        K = self._rbf_kernel(X1, X2, sigma_f, l)
        if X1 is X2:
            K += ((sigma_n**2) + self.noise) * np.eye(X1.shape[0])
        return K

    def fit(self, X_train, y_train, init_params=(1.0, 1.0, 0.1), optimize=False):
        X_train, self.mu_X, self.sigma_X = self._normalize(X_train)
        y_train, self.mu_y, self.sigma_y = self._normalize(y_train)

        self.X_train = X_train
        self.y_train = y_train

        self.params = self._optimize_hyperparams(init_params) if optimize else np.array(init_params, dtype=np.float64)

        K = self._kernel_matrix(X_train, X_train)
        try:
            self.L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            K += self.noise * np.eye(K.shape[0])
            self.L = cholesky(K, lower=True)

        self.alpha = solve_triangular(self.L.T, solve_triangular(self.L, y_train, lower=True))

    def predict(self, X_test, return_std=False):
        X_test = (X_test - self.mu_X) / self.sigma_X
        K_star = self._kernel_matrix(self.X_train, X_test)
        y_mean = K_star.T @ self.alpha
        y_mean = y_mean * self.sigma_y + self.mu_y

        if not return_std:
            return y_mean

        v = solve_triangular(self.L, K_star, lower=True)
        K_starstar = self._kernel_matrix(X_test, X_test)
        y_cov = K_starstar - v.T @ v

        diag = np.diag(y_cov)
        diag = np.maximum(diag, 0.0)
        y_std = np.sqrt(diag) * self.sigma_y
        return y_mean, y_std

    def _optimize_hyperparams(self, init_params):
        def nll(params):
            sigma_f, l, sigma_n = params
            K = self._rbf_kernel(self.X_train, self.X_train, sigma_f, l) + ((sigma_n**2) + 1e-5) * np.eye(len(self.X_train))
            try:
                L = cholesky(K, lower=True)
            except np.linalg.LinAlgError:
                return np.inf
            alpha = solve_triangular(L.T, solve_triangular(L, self.y_train, lower=True))
            return 0.5 * self.y_train.T @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(self.y_train) * np.log(2*np.pi)

        bounds = [(1e-5, 1e2), (1e-5, 1e2), (1e-5, 1e2)]
        res = minimize(nll, init_params, method='L-BFGS-B', bounds=bounds)
        return res.x

    @staticmethod
    def _normalize(X):
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma == 0, 1.0, sigma)
        return (X - mu) / sigma, mu, sigma


def _sample_delay_combos(total_dims, tau_max, n_combos, rng, M):
    dims = np.arange(total_dims)
    taus = np.arange(1, tau_max + 1)
    dd, tt = np.meshgrid(dims, taus, indexing='ij')
    candidates = np.stack([dd.ravel(), tt.ravel()], axis=1)

    n_cand = len(candidates)
    M_use = min(M, n_cand)

    combos = []
    for _ in range(n_combos):
        chosen_idx = rng.choice(n_cand, size=M_use, replace=False)
        combos.append(candidates[chosen_idx])
    return combos, M_use


def _build_feature_with_delay(seq, combo, t):
    feat = np.empty(len(combo))
    for k, (dim, delay) in enumerate(combo):
        idx = t - int(delay)
        feat[k] = seq[idx, int(dim)] if idx >= 0 else np.nan
    return feat


def _parallel_predict_one_comb_delay(args):
    comb, traindata, target_idx, steps_ahead, optimize_hyp, max_delay = args
    try:
        trainlength = len(traindata)
        if trainlength - steps_ahead <= max_delay + 1:
            return np.nan, np.nan, 2

        max_delay_used = int(comb[:, 1].max())
        t_min = max_delay_used
        t_max = trainlength - steps_ahead - 1

        if t_min > t_max or t_max - t_min < 5:
            return np.nan, np.nan, 2

        ts = np.arange(t_min, t_max + 1)
        X = np.array([_build_feature_with_delay(traindata, comb, t) for t in ts])
        y = traindata[ts + steps_ahead, target_idx]

        valid = ~np.any(np.isnan(X), axis=1)
        X, y = X[valid], y[valid]

        if len(y) < max(5, len(comb) + 1):
            return np.nan, np.nan, 2
        if np.std(y) < 1e-8:
            return np.nan, np.nan, 2

        x_test = _build_feature_with_delay(traindata, comb, trainlength - steps_ahead)
        if np.any(np.isnan(x_test)):
            return np.nan, np.nan, 2
        x_test = x_test.reshape(1, -1)

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        combined_X = np.vstack([X, x_test])
        combined_X_scaled = scaler_X.fit_transform(combined_X)
        X_scaled = combined_X_scaled[:-1]
        x_test_scaled = combined_X_scaled[-1:]

        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        gp = GaussianProcessRegressor(noise=1e-6)
        gp.fit(X_scaled, y_scaled, init_params=(1.0, 1.0, 0.1), optimize=optimize_hyp)

        pred_scaled, std_scaled = gp.predict(x_test_scaled, return_std=True)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        std = std_scaled[0] * scaler_y.scale_[0]

        return float(pred), float(std), 0
    except Exception:
        return np.nan, np.nan, 1


def rdegpr_predict_with_delay(traindata, target_idx, L, s, steps_ahead, pool, rng, max_delay, optimize_hyp=True):
    D = traindata.shape[1]
    n_combos = min(int(s), 500)
    combos, M_use = _sample_delay_combos(D, max_delay, n_combos, rng, L)

    args_list = [(comb, traindata, target_idx, steps_ahead, optimize_hyp, max_delay) for comb in combos]
    results = pool.map(_parallel_predict_one_comb_delay, args_list)

    pred_values = np.array([r[0] for r in results], dtype=np.float64)
    pred_stds = np.array([r[1] for r in results], dtype=np.float64)

    valid = ~np.isnan(pred_values)
    valid_preds = pred_values[valid]
    valid_stds = pred_stds[valid]

    if len(valid_preds) == 0:
        return np.nan, np.nan, {"ok_count": 0, "fail_count": len(combos)}
    if len(valid_preds) == 1:
        return float(valid_preds[0]), float(valid_stds[0]), {"ok_count": 1, "fail_count": len(combos)-1}

    inter_var = np.var(valid_preds)
    intra_var = np.mean(valid_stds ** 2)
    final_std = float(np.sqrt(inter_var + intra_var))

    if len(valid_preds) >= 5:
        try:
            kde = gaussian_kde(valid_preds)
            xi = np.linspace(valid_preds.min(), valid_preds.max(), 500)
            density = kde(xi)
            final_pred = float(np.sum(xi * density) / np.sum(density))
        except Exception:
            final_pred = float(np.mean(valid_preds))
    else:
        final_pred = float(np.mean(valid_preds))

    return final_pred, final_std, {"ok_count": int(valid.sum()), "fail_count": int((~valid).sum())}


def _parallel_predict_one_comb(comb, traindata, target_idx, steps_ahead=1, optimize_hyp=True):
    try:
        trainlength = len(traindata)
        if trainlength - steps_ahead <= 2:
            return np.nan, np.nan, 2

        X = traindata[:trainlength - steps_ahead, list(comb)]
        y = traindata[steps_ahead:trainlength, target_idx]
        x_test = traindata[trainlength - steps_ahead, list(comb)].reshape(1, -1)

        if np.isnan(X).any() or np.isnan(y).any() or np.isnan(x_test).any():
            return np.nan, np.nan, 2
        if np.isclose(np.std(y), 0.0):
            return np.nan, np.nan, 2

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        combined_X = np.vstack([X, x_test])
        combined_X_scaled = scaler_X.fit_transform(combined_X)
        X_scaled = combined_X_scaled[:-1]
        x_test_scaled = combined_X_scaled[-1:]

        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        gp = GaussianProcessRegressor(noise=1e-6)
        gp.fit(X_scaled, y_scaled, init_params=(1.0, 0.1, 0.1), optimize=optimize_hyp)

        pred_scaled, std_scaled = gp.predict(x_test_scaled, return_std=True)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        return float(pred), float(std_scaled[0]), 0
    except Exception:
        return np.nan, np.nan, 1


def rdegpr_predict_next_for_target(traindata, target_idx, L, s, steps_ahead, pool, rng, optimize_hyp=True):
    D = traindata.shape[1]
    n_combs = min(int(s), 1000)
    selected = [tuple(rng.choice(D, size=int(L), replace=False)) for _ in range(n_combs)]

    results = pool.map(
        partial(_parallel_predict_one_comb,
                traindata=traindata,
                target_idx=int(target_idx),
                steps_ahead=int(steps_ahead),
                optimize_hyp=optimize_hyp),
        selected
    )

    pred_values = np.array([r[0] for r in results], dtype=np.float64)
    pred_stds = np.array([r[1] for r in results], dtype=np.float64)
    status = np.array([r[2] for r in results], dtype=np.int32)

    valid = ~np.isnan(pred_values)
    valid_preds = pred_values[valid]

    if len(valid_preds) == 0:
        return np.nan, np.nan, {"ok_count": 0, "fail_count": len(selected)}
    if len(valid_preds) == 1:
        return float(valid_preds[0]), 0.0, {"ok_count": 1, "fail_count": len(selected)-1}

    try:
        kde = gaussian_kde(valid_preds)
        xi = np.linspace(valid_preds.min(), valid_preds.max(), 1000)
        density = kde(xi)
        final_pred = float(np.sum(xi * density) / np.sum(density))
        final_std = float(np.std(valid_preds))
    except Exception:
        final_pred = float(np.mean(valid_preds))
        final_std = float(np.std(valid_preds))

    return final_pred, final_std, {"ok_count": int(valid.sum()), "fail_count": int((~valid).sum())}


def visualize_prediction_results(y_true, y_pred, history, missing_positions, out_dir,
                                  plot_dim=0, history_timesteps=72, missing_mode="random", missing_ratio=0.1,
                                  hist_full=None, total_len=None):
    n_dim = y_true.shape[1]
    horizon = len(y_true)

    d = min(plot_dim, n_dim - 1)

    fig, ax = plt.subplots(figsize=(18, 7))

    hist_plot_len = min(history_timesteps, len(history))
    hist_index = np.arange(-hist_plot_len, 0)
    fut_index = np.arange(0, horizon)

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Forecast start')

    ax.plot(fut_index, y_true[:, d], 'g-', linewidth=2, label=f"True future")
    ax.plot(fut_index, y_pred[:, d], 'r--', linewidth=2, label=f"Predicted")

    ax.set_xlabel("Time Step (0 = forecast start)", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(f"EEG Forecast - {missing_mode} missing {missing_ratio:.0%} (Channel {d})", fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    m = compute_metrics(y_true[:, d], y_pred[:, d])
    if not np.isnan(m['rmse']):
        ax.text(0.02, 0.95, f'Forecast RMSE={m["rmse"]:.2f}, MAE={m["mae"]:.2f}',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'prediction_detail_dim{d}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存预测详情图: {os.path.join(out_dir, f'prediction_detail_dim{d}.png')}")

    n_show = min(6, n_dim)
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 3 * n_show))
    if n_show == 1:
        axes = [axes]

    for i in range(n_show):
        ax = axes[i]
        ax.plot(y_true[:, i], 'g-', linewidth=1.5, label='True')
        ax.plot(y_pred[:, i], 'r--', linewidth=1.5, label='Pred')
        ax.set_title(f'Channel {i}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        m = compute_metrics(y_true[:, i], y_pred[:, i])
        if not np.isnan(m['rmse']):
            ax.text(0.02, 0.95, f'RMSE={m["rmse"]:.2f}', transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'prediction_multi_dim.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存多维度预测图: {os.path.join(out_dir, 'prediction_multi_dim.png')}")

    rmse_list = []
    for j in range(n_dim):
        m = compute_metrics(y_true[:, j], y_pred[:, j])
        rmse_list.append({'dim': j, 'rmse': m['rmse'], 'mae': m['mae']})

    rmse_df = pd.DataFrame(rmse_list)

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(rmse_df['dim'], rmse_df['rmse'], color='steelblue', alpha=0.7)
    ax.set_xlabel('Channel')
    ax.set_ylabel('RMSE')
    ax.set_title('Prediction RMSE per Channel')
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}',
               ha='center', va='bottom', fontsize=8, rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'prediction_rmse_per_dim.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存RMSE柱状图: {os.path.join(out_dir, 'prediction_rmse_per_dim.png')}")

    fig, ax = plt.subplots(figsize=(14, 6))
    error = np.abs(y_true - y_pred).T
    im = ax.imshow(error, aspect='auto', cmap='Reds')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Channel')
    ax.set_title('Prediction Absolute Error Heatmap')
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'prediction_error_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存误差热力图: {os.path.join(out_dir, 'prediction_error_heatmap.png')}")


def main():
    parser = argparse.ArgumentParser(description="EEG RDE-GPR Prediction")

    parser.add_argument("--imputed_path", type=str, required=True,
                        help="CSDI补值后的数据路径 (.npy)")
    parser.add_argument("--missing_positions_path", type=str, default="")
    parser.add_argument("--impute_meta_path", type=str, default="",
                        help="补值脚本生成的impute_meta.json路径")
    parser.add_argument("--ground_path", type=str, default="./data/eeg/eeg_ground.npy",
                        help="EEG原始完整数据")

    parser.add_argument("--total_len", type=int, default=0,
                        help="总数据长度")
    parser.add_argument("--horizon_steps", type=int, default=24)
    parser.add_argument("--history_timesteps", type=int, default=100)

    parser.add_argument("--missing_ratio", type=float, default=0.5)
    parser.add_argument("--missing_mode", type=str, default="random")

    parser.add_argument("--target_indices", type=str, default="", help="如 '0,1,2'；为空=全维")
    parser.add_argument("--plot_dim", type=int, default=0)
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--multi_step", action="store_true", help="多步预测（每次预测多步）")
    parser.add_argument("--multi_step_mode", type=str, default="direct", choices=["direct", "recursive"], help="多步预测模式: direct=直接预测, recursive=递归预测")

    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--s", type=int, default=50)
    parser.add_argument("--trainlength", type=int, default=100)
    parser.add_argument("--n_jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_delay_embedding", action="store_true", help="使用延迟嵌入而非空间集成")
    parser.add_argument("--max_delay", type=int, default=20, help="延迟嵌入的最大延迟值")

    args = parser.parse_args()

    print("=" * 80)
    print("EEG RDE-GPR 预测")
    print("=" * 80)

    set_global_seed(args.seed)

    imputed_data = np.load(args.imputed_path)
    if imputed_data.ndim == 3:
        imputed_data = imputed_data.reshape(-1, imputed_data.shape[-1])

    ground_data = np.load(args.ground_path)
    n_timesteps, n_dim = ground_data.shape

    if args.impute_meta_path and os.path.exists(args.impute_meta_path):
        with open(args.impute_meta_path, 'r') as f:
            impute_meta = json.load(f)
        total_len = impute_meta['total_len']
        print(f"从impute_meta.json加载: total_len={total_len}")
    else:
        total_len = args.total_len if args.total_len > 0 else min(n_timesteps, len(imputed_data))
        print(f"使用默认值: total_len={total_len}")

    if len(imputed_data) > total_len:
        print(f"警告: imputed数据({len(imputed_data)})比total_len({total_len})长，截断到total_len")
        imputed_data = imputed_data[:total_len]

    horizon = args.horizon_steps
    history_timesteps = min(args.history_timesteps, total_len - horizon)

    history = imputed_data[:history_timesteps]
    future_truth = ground_data[history_timesteps:history_timesteps + horizon]

    meta = {
        "total_len": total_len,
        "n_dim": n_dim,
        "history_timesteps": history_timesteps,
        "horizon": horizon,
        "missing_ratio": args.missing_ratio,
        "missing_mode": args.missing_mode,
    }
    print(json.dumps(meta, indent=4, ensure_ascii=False))

    if args.target_indices:
        target_indices = [int(x) for x in args.target_indices.split(",")]
    else:
        target_indices = list(range(n_dim))
    print(f"target_indices = {target_indices}")

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"./save/eeg_pred_{args.missing_mode}_ratio{args.missing_ratio}_horizon{horizon}_{now}/"
    ensure_dir(out_dir)

    print(f"\nRDE-GPR rolling prediction (L={args.L}, s={args.s}, trainlength={args.trainlength})...")

    seq_true = np.vstack([history, future_truth])

    preds = np.full((horizon, n_dim), np.nan)
    rng = np.random.default_rng(args.seed)

    with mp.Pool(processes=args.n_jobs) as pool:
        print(f"\nRDE-GPR {'延迟嵌入' if args.use_delay_embedding else '空间集成'} prediction (L={args.L}, s={args.s}, trainlength={args.trainlength}, max_delay={args.max_delay})...")
        print(f"预测模式: {'多步直接预测' if args.multi_step else '单步滚动预测'}")

        if args.multi_step:
            for d in target_indices:
                for h in tqdm(range(1, horizon + 1), desc=f"Multi-step dim {d}"):
                    t_pred = len(history)
                    start = t_pred - args.trainlength
                    end = t_pred
                    traindata = seq_true[start:end].copy()
                    trainlength = len(traindata)

                    if args.use_delay_embedding:
                        if trainlength < args.max_delay + args.L + h + 1:
                            preds[h-1, d] = np.nan
                            continue
                    else:
                        if trainlength < args.L + h + 1:
                            preds[h-1, d] = np.nan
                            continue

                    if args.use_delay_embedding:
                        pred, std, dbg = rdegpr_predict_with_delay(
                            traindata, d, args.L, args.s, steps_ahead=h, pool=pool, rng=rng, max_delay=args.max_delay
                        )
                    else:
                        pred, std, dbg = rdegpr_predict_next_for_target(
                            traindata, d, args.L, args.s, steps_ahead=h, pool=pool, rng=rng
                        )
                    preds[h-1, d] = pred
        else:
            for h in tqdm(range(horizon), desc="RDE-GPR one-step rolling"):
                t_pred = len(history) + h
                start = t_pred - args.trainlength
                end = t_pred
                traindata = seq_true[start:end].copy()

                trainlength = len(traindata) - 1
                if args.use_delay_embedding:
                    if trainlength < args.max_delay + args.L + 1:
                        continue
                else:
                    if trainlength < args.L + 1:
                        continue

                for d in target_indices:
                    if args.use_delay_embedding:
                        pred, std, dbg = rdegpr_predict_with_delay(
                            traindata, d, args.L, args.s, steps_ahead=1, pool=pool, rng=rng, max_delay=args.max_delay
                        )
                    else:
                        pred, std, dbg = rdegpr_predict_next_for_target(
                            traindata, d, args.L, args.s, steps_ahead=1, pool=pool, rng=rng
                        )
                    preds[h, d] = pred

    pred_df = pd.DataFrame(preds, columns=[f"dim_{i}" for i in range(n_dim)])
    pred_df.to_csv(os.path.join(out_dir, "future_pred.csv"), index=False)

    print(f"\n预测保存：\n   {os.path.join(out_dir, 'future_pred.csv')}")

    if not args.skip_metrics:
        y_true = future_truth[:, target_indices]
        y_pred = preds[:, target_indices]

        overall = compute_metrics(y_true.flatten(), y_pred.flatten())
        safe_json_dump({"overall": overall, "horizon": horizon, "target_indices": target_indices}, os.path.join(out_dir, "metrics.json"))

        per_dim = []
        for j, dim_idx in enumerate(target_indices):
            m = compute_metrics(y_true[:, j], y_pred[:, j])
            per_dim.append({"dim": dim_idx, "rmse": m["rmse"], "mae": m["mae"]})
        pd.DataFrame(per_dim).to_csv(os.path.join(out_dir, "metrics_per_dim.csv"), index=False)

        print(f"\n整体评估：")
        print(json.dumps(overall, indent=4))

        missing_positions = None
        if args.missing_positions_path and os.path.exists(args.missing_positions_path):
            missing_positions = pd.read_csv(args.missing_positions_path)

        print("\n" + "=" * 80)
        print("生成可视化...")
        print("=" * 80)

        visualize_prediction_results(
            y_true, y_pred, history, missing_positions, out_dir,
            plot_dim=args.plot_dim,
            history_timesteps=history_timesteps,
            missing_mode=args.missing_mode,
            missing_ratio=args.missing_ratio,
            hist_full=ground_data[:total_len],
            total_len=total_len
        )

        print("\n生成的可视化文件:")
        print(f"  - prediction_detail_dim{args.plot_dim}.png (预测详情)")
        print("  - prediction_multi_dim.png (多维度预测)")
        print("  - prediction_rmse_per_dim.png (RMSE柱状图)")
        print("  - prediction_error_heatmap.png (误差热力图)")

    print(f"\n输出目录： {out_dir}")


if __name__ == "__main__":
    main()