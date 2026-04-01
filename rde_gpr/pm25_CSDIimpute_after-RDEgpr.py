# -*- coding: utf-8 -*-
"""
PM2.5 后续预测脚本（不再跑CSDI）- Debug增强版
- 输入：history_imputed.csv（前半段历史已补值完整）
- 输出：future_pred.csv / future_pred_std.csv / metrics / plots / debug_report.json

示例（预测1天）：
python rde_gpr/pm25_CSDIimpute_after-RDEgpr.py \
  --imputed_history_path ./save/pm25_history_imputed_split0.5_seed42_20260324_154400/history_imputed.csv \
  --ground_path ./data/pm25/Code/STMVL/SampleData/pm25_ground.txt \
  --missing_positions_path ./save/pm25_history_imputed_split0.5_seed42_20260324_154400/history_missing_positions.csv \
  --split_ratio 0.5 \
  --horizon_days 1 \
  --L 4 --s 50 --trainlength 4000 --n_jobs 8 \
  --target_indices 0,1,2 \
  --history_timesteps 72 \
  --debug

注意：
- 预测阶段不会读取 fut_full 的数值，仅使用其 datetime 索引。
- 若你没有未来真值，不要评估：加 --skip_metrics
- --missing_positions_path 用于在图中标注补值点（橙色点）
- --history_timesteps 控制作图时显示多少历史时间步（默认72步=3天）
"""

import os
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


# =============================================================================
# Utils
# =============================================================================
def set_global_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_json_dump(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False, default=str)


def infer_steps_per_day_from_index(idx: pd.DatetimeIndex, default_steps_per_day: int = 24) -> int:
    """根据 datetime 间隔推断每天多少步。"""
    if len(idx) < 2:
        return int(default_steps_per_day)
    deltas = idx.to_series().diff().dropna()
    if len(deltas) == 0:
        return int(default_steps_per_day)
    dt = deltas.median()
    if dt <= pd.Timedelta(0):
        return int(default_steps_per_day)
    steps_per_day = int(round(pd.Timedelta(days=1) / dt))
    return max(1, steps_per_day)


def time_split_df(df_full: pd.DataFrame, split_ratio: float):
    total_len = len(df_full)
    split_point = int(total_len * float(split_ratio))
    hist = df_full.iloc[:split_point].copy()
    fut = df_full.iloc[split_point:].copy()
    meta = {
        "total_len": total_len,
        "split_ratio": float(split_ratio),
        "split_point": split_point,
        "hist_len": len(hist),
        "fut_len": len(fut),
        "hist_start": str(hist.index.min()),
        "hist_end": str(hist.index.max()),
        "fut_start": str(fut.index.min()) if len(fut) else None,
        "fut_end": str(fut.index.max()) if len(fut) else None,
    }
    return hist, fut, meta


def basic_array_stats(x: np.ndarray, name: str):
    """返回数组的基础统计，用于 debug_report。"""
    x = np.asarray(x)
    out = {"name": name, "shape": list(x.shape)}
    out["nan_count"] = int(np.isnan(x).sum())
    out["inf_count"] = int(np.isinf(x).sum())
    if np.all(np.isnan(x)):
        out["all_nan"] = True
        return out
    out["min"] = float(np.nanmin(x))
    out["max"] = float(np.nanmax(x))
    out["mean"] = float(np.nanmean(x))
    out["std"] = float(np.nanstd(x))
    return out


def assert_or_raise(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)


# =============================================================================
# GPR (增强数值稳定性 + debug)
# =============================================================================
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
        diag = np.maximum(diag, 0.0)  # 避免 sqrt NaN
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


def _parallel_predict_one_comb(comb, traindata, target_idx, steps_ahead=1, optimize_hyp=True):
    """
    返回 (pred, std, status_code)
    status_code:
      0 OK
      1 FAIL_EXCEPTION
      2 FAIL_DEGENERATE_INPUT
    """
    try:
        trainlength = len(traindata)
        if trainlength - steps_ahead <= 2:
            return np.nan, np.nan, 2

        X = traindata[:trainlength - steps_ahead, list(comb)]
        y = traindata[steps_ahead:trainlength, target_idx]
        x_test = traindata[trainlength - steps_ahead, list(comb)].reshape(1, -1)

        if np.isnan(X).any() or np.isnan(y).any() or np.isnan(x_test).any():
            return np.nan, np.nan, 2

        # 全常数会让标准化/核矩阵退化
        if np.allclose(np.std(X, axis=0), 0.0) or np.isclose(np.std(y), 0.0):
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


def rdegpr_predict_next_for_target(traindata, target_idx, L, s, steps_ahead, pool, rng, optimize_hyp=True, debug=False):
    """
    返回 pred, std, debug_info
    """
    D = traindata.shape[1]
    combs = list(itertools.combinations(range(D), int(L)))
    rng.shuffle(combs)
    selected = combs[:min(int(s), len(combs))]

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
    status = np.array([r[2] for r in results], dtype=np.int64)

    valid = ~np.isnan(pred_values) & ~np.isnan(pred_stds)
    valid_preds = pred_values[valid]

    dbg = None
    if debug:
        dbg = {
            "target_idx": int(target_idx),
            "L": int(L),
            "s_selected": int(len(selected)),
            "ok_count": int((status == 0).sum()),
            "fail_exception_count": int((status == 1).sum()),
            "fail_degenerate_count": int((status == 2).sum()),
            "valid_pred_count": int(valid.sum()),
            "pred_min": None if len(valid_preds) == 0 else float(np.min(valid_preds)),
            "pred_max": None if len(valid_preds) == 0 else float(np.max(valid_preds)),
            "pred_mean": None if len(valid_preds) == 0 else float(np.mean(valid_preds)),
        }

    if len(valid_preds) == 0:
        return np.nan, np.nan, dbg
    if len(valid_preds) == 1:
        return float(valid_preds[0]), 0.0, dbg

    try:
        kde = gaussian_kde(valid_preds)
        xi = np.linspace(valid_preds.min(), valid_preds.max(), 1000)
        density = kde(xi)
        final_pred = float(np.sum(xi * density) / np.sum(density))
        final_std = float(np.std(valid_preds))
        return final_pred, final_std, dbg
    except Exception:
        return float(np.mean(valid_preds)), float(np.std(valid_preds)), dbg


def rdegpr_forecast_multivariate(
    history,
    future_truth,   # <-- 新增：用于 teacher forcing 推进窗口
    horizon,
    trainlength,
    L,
    s,
    steps_ahead,
    n_jobs,
    seed,
    noise_strength=0.0,
    optimize_hyp=True,
    target_indices=None,
    debug=False,
    debug_steps=3,
    debug_out_dir=None,
):
    """
    一步滚动预测（Teacher Forcing / Open-loop）：
    - 每一步都用“真实数据”推进滑动窗口，不把预测点写回序列
    - step=0：用 history 的最后 trainlength 点预测 future_truth[0]
    - step>0：用 (history + future_truth[:step]) 的最后 trainlength 点预测 future_truth[step]
    - 未预测维度默认持久性（上一时刻真实值）
    """
    history = np.asarray(history, dtype=np.float64)
    future_truth = np.asarray(future_truth, dtype=np.float64)

    T_hist, D = history.shape

    horizon = int(horizon)
    trainlength = int(trainlength)
    L = int(L)
    s = int(s)
    steps_ahead = int(steps_ahead)
    n_jobs = int(n_jobs)
    noise_strength = float(noise_strength)

    # 参数合法性检查
    assert_or_raise(trainlength >= 2, "trainlength 必须 >= 2")
    assert_or_raise(steps_ahead >= 1, "steps_ahead 必须 >= 1")
    assert_or_raise(horizon >= 1, "horizon 必须 >= 1")
    assert_or_raise(1 <= L <= D, f"L 必须在 [1, {D}]，当前 L={L}")
    assert_or_raise(s >= 1, "s 必须 >= 1")
    assert_or_raise(T_hist >= trainlength, f"history长度({T_hist}) < trainlength({trainlength})")
    assert_or_raise(future_truth.shape[0] >= horizon,
                    f"future_truth长度({future_truth.shape[0]}) < horizon({horizon})："
                    "一步滚动需要未来真值来推进窗口。")

    if target_indices is None:
        target_indices = list(range(D))
    else:
        target_indices = list(target_indices)
        assert_or_raise(len(target_indices) > 0, "target_indices 解析为空列表，请检查参数。")

    # 拼成“已知的完整序列”：history(补值) + future_truth(真值，用来推进窗口)
    # 注意：这里不会把预测值写回去
    seq_true = np.vstack([history, future_truth[:horizon]])

    preds = np.zeros((horizon, D), dtype=np.float64)
    stds = np.zeros((horizon, D), dtype=np.float64)

    base_rng = np.random.default_rng(int(seed))
    pool = mp.Pool(processes=n_jobs)

    debug_records = []
    try:
        for step in tqdm(range(horizon), desc="RDE-GPR one-step rolling"):
            # 这一步要预测的是 seq_true 的位置：t = T_hist + step
            # 窗口用它前面的 trainlength 个真实点： [t-trainlength, ..., t-1]
            t_pred = T_hist + step
            start = t_pred - trainlength
            end = t_pred
            traindata = seq_true[start:end].copy()

            # 可选加噪（只作用于训练窗口）
            if noise_strength > 0:
                traindata = traindata + noise_strength * base_rng.standard_normal(size=traindata.shape)

            # baseline：上一时刻真实值（持久性用）
            prev_true = seq_true[t_pred - 1].copy()
            next_vec = prev_true.copy()
            next_std = np.zeros((D,), dtype=np.float64)

            step_dbg = None
            if debug and step < debug_steps:
                step_dbg = {
                    "step": step,
                    "traindata_stats": basic_array_stats(traindata, "traindata"),
                    "baseline_prev_true_stats": basic_array_stats(prev_true, "baseline_prev_true"),
                    "targets": []
                }

            for j in target_indices:
                tj_rng = np.random.default_rng(int(seed + 100000 * step + 1000 * int(j)))
                pred_j, std_j, dbg_j = rdegpr_predict_next_for_target(
                    traindata=traindata,
                    target_idx=int(j),
                    L=L,
                    s=s,
                    steps_ahead=steps_ahead,
                    pool=pool,
                    rng=tj_rng,
                    optimize_hyp=optimize_hyp,
                    debug=(debug and step < debug_steps),
                )

                # 失败就回退到持久性（上一时刻真实值），避免 NaN 传播
                if np.isnan(pred_j):
                    pred_j = next_vec[int(j)]
                    std_j = 0.0

                next_vec[int(j)] = float(pred_j)
                next_std[int(j)] = float(std_j)

                if step_dbg is not None and dbg_j is not None:
                    step_dbg["targets"].append(dbg_j)

            preds[step] = next_vec
            stds[step] = next_std

            if step_dbg is not None:
                step_dbg["pred_row_stats"] = basic_array_stats(next_vec, "pred_row")
                # 方便你对比：这一时刻的真值（可选）
                step_dbg["true_row_stats"] = basic_array_stats(seq_true[t_pred], "true_row")
                debug_records.append(step_dbg)

    finally:
        pool.close()
        pool.join()

    # 输出 debug step 细节
    if debug and debug_out_dir is not None:
        ensure_dir(debug_out_dir)
        for rec in debug_records:
            safe_json_dump(rec, os.path.join(debug_out_dir, f"debug_sample_step{rec['step']}.json"))

    return preds, stds, debug_records

# =============================================================================
# Metrics / Plots
# =============================================================================
def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan}
    diff = y_true[mask] - y_pred[mask]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    return {"rmse": rmse, "mae": mae}


def save_plots(out_dir: str, fut_index: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray, plot_dim: int,
               hist_index: pd.DatetimeIndex = None, hist_values: np.ndarray = None,
               missing_positions: pd.DataFrame = None, history_timesteps: int = 72,
               column_names: list = None):
    ensure_dir(out_dir)
    d = int(plot_dim)
    d = max(0, min(d, y_true.shape[1] - 1))

    col_name = None
    if column_names is not None and len(column_names) > d:
        col_name = str(column_names[d])
    elif missing_positions is not None and 'column' in missing_positions.columns:
        cols = missing_positions['column'].unique()
        if len(cols) > d:
            col_name = str(cols[d])

    fig, ax = plt.subplots(figsize=(16, 6))

    if hist_index is not None and hist_values is not None:
        hist_len = min(history_timesteps, len(hist_index))
        hist_index_plot = hist_index[-hist_len:]
        hist_values_plot = hist_values[-hist_len:, d]

        ax.plot(hist_index_plot, hist_values_plot, 'b-', alpha=0.7, linewidth=1.2, label='History (observed)')

        if missing_positions is not None and col_name is not None:
            col_missing = missing_positions[missing_positions['column'] == col_name].copy()
            if 'datetime' in col_missing.columns and len(col_missing) > 0:
                if not pd.api.types.is_datetime64_any_dtype(col_missing['datetime']):
                    col_missing['datetime'] = pd.to_datetime(col_missing['datetime'])

                hist_start = hist_index_plot.min()
                hist_end = hist_index_plot.max()
                mask_hist = (col_missing['datetime'] >= hist_start) & (col_missing['datetime'] <= hist_end)
                col_missing_hist = col_missing[mask_hist]

                if len(col_missing_hist) > 0:
                    imputed_vals = col_missing_hist['imputed_value'].values
                    imputed_times = col_missing_hist['datetime'].values
                    ax.scatter(imputed_times, imputed_vals, c='orange', s=30, marker='o',
                              label=f'Imputed points (n={len(col_missing_hist)})', zorder=5, alpha=0.8)

    ax.axvline(x=fut_index[0], color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Forecast start')

    ax.plot(fut_index, y_true[:, d], 'g-', linewidth=1.5, label=f"True (dim {d})")
    ax.plot(fut_index, y_pred[:, d], 'r--', linewidth=1.5, label=f"Pred (dim {d})")

    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("PM2.5", fontsize=11)
    ax.set_title(f"Forecast vs True with History & Imputed Points (dim {d})", fontsize=12)
    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"plot_forecast_with_history_dim{d}.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(14, 5))
    plt.plot(fut_index, y_true[:, d], label=f"True (dim {d})")
    plt.plot(fut_index, y_pred[:, d], label=f"Pred (dim {d})")
    plt.xlabel("Time")
    plt.ylabel("PM2.5")
    plt.title(f"Forecast vs True on Future Segment (dim {d})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"plot_forecast_dim{d}.png"), dpi=150)
    plt.close()

    rmse_list = []
    for j in range(y_true.shape[1]):
        rmse_list.append(compute_metrics(y_true[:, j], y_pred[:, j])["rmse"])

    plt.figure(figsize=(14, 5))
    plt.bar(np.arange(len(rmse_list)), rmse_list)
    plt.xlabel("Dimension")
    plt.ylabel("RMSE")
    plt.title("RMSE per Dimension on Future Segment")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_rmse_per_dim.png"), dpi=150)
    plt.close()


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="PM2.5 Forecasting: RDE-GPR from Imputed History (Debug Enhanced)")

    parser.add_argument("--imputed_history_path", type=str, required=True, help="history_imputed.csv（前半段补值结果）")
    parser.add_argument("--ground_path", type=str, required=True, help="pm25_ground.txt（用于时间索引 + 可选评估）")
    parser.add_argument("--split_ratio", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=42)

    # horizon 控制
    parser.add_argument("--horizon_days", type=float, default=0.0, help="预测多少天（>0优先）")
    parser.add_argument("--horizon_steps", type=int, default=0, help="预测多少步（>0生效）")

    # RDE-GPR 参数
    parser.add_argument("--trainlength", type=int, default=30)
    parser.add_argument("--L", type=int, default=4, help="随机嵌入组合维数 L（可调）")
    parser.add_argument("--s", type=int, default=50, help="每步抽样组合数")
    parser.add_argument("--steps_ahead", type=int, default=1)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--noise_strength", type=float, default=0.0, help="建议从 1e-4 开始试")
    parser.add_argument("--no_optimize_hyp", action="store_true")

    parser.add_argument("--target_indices", type=str, default="", help="只预测部分维度：如 '0,1,2'；为空=全维")
    parser.add_argument("--plot_dim", type=int, default=0)
    parser.add_argument("--skip_metrics", action="store_true", help="无未来真值时跳过 metrics 与 plot")
    parser.add_argument("--out_dir", type=str, default="", help="输出目录（可选）")

    parser.add_argument("--missing_positions_path", type=str, default="", help="history_missing_positions.csv（用于标注补值点）")
    parser.add_argument("--history_timesteps", type=int, default=72, help="作图时显示的历史时间步数")

    # Debug
    parser.add_argument("--debug", action="store_true", help="开启更详细的自检/定位输出")
    parser.add_argument("--debug_steps", type=int, default=3, help="保存前多少步的详细debug信息")

    args = parser.parse_args()

    set_global_seed(args.seed)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"./save/pm25_rdegpr_debug_split{args.split_ratio}_seed{args.seed}_{now}/"
    ensure_dir(out_dir)

    safe_json_dump(vars(args), os.path.join(out_dir, "args.json"))

    # 读 ground 拿索引并切分
    df_full = pd.read_csv(args.ground_path, index_col="datetime", parse_dates=True).sort_index()
    hist_full, fut_full, meta = time_split_df(df_full, args.split_ratio)

    # 读补值历史
    df_hist_imputed = pd.read_csv(args.imputed_history_path, index_col="datetime", parse_dates=True).sort_index()

    # 对齐校验：索引 & 列
    assert_or_raise(df_hist_imputed.index.equals(hist_full.index),
                    "imputed_history 的 datetime 索引与 ground 切分出的前半段不一致（split_ratio必须一致）。")
    assert_or_raise(list(df_hist_imputed.columns) == list(hist_full.columns),
                    "imputed_history 的列与 ground 列不一致。")

    # history 基础自检
    history = df_hist_imputed.values.astype(np.float64)
    report = {
        "meta": meta,
        "history_stats": basic_array_stats(history, "history_imputed"),
        "history_last_row_stats": basic_array_stats(history[-1], "history_last_row"),
        "checks": {}
    }

    # 强制检查：NaN/Inf
    report["checks"]["history_has_nan"] = bool(np.isnan(history).any())
    report["checks"]["history_has_inf"] = bool(np.isinf(history).any())
    assert_or_raise(not report["checks"]["history_has_nan"], "history_imputed 内仍包含 NaN：请先确保补值完全。")
    assert_or_raise(not report["checks"]["history_has_inf"], "history_imputed 内包含 Inf：数据异常。")

    # 常见“全0导致未来全0”检查
    report["checks"]["history_last_row_all_zero"] = bool(np.allclose(history[-1], 0.0))
    if report["checks"]["history_last_row_all_zero"]:
        # 不直接 raise，给更明确提示（也写入 debug_report）
        report["checks"]["warning"] = (
            "history_imputed 最后一行全0：若RDE-GPR大量失败/或target_indices为空，将导致未来预测全0。"
        )

    full_horizon = meta["fut_len"]
    assert_or_raise(full_horizon > 0, "后半段长度为0，请检查 split_ratio。")

    # horizon 计算
    horizon = full_horizon
    if args.horizon_days and args.horizon_days > 0:
        steps_per_day = infer_steps_per_day_from_index(fut_full.index, default_steps_per_day=24)
        horizon = int(round(args.horizon_days * steps_per_day))
        report["checks"]["steps_per_day_inferred"] = int(steps_per_day)
    elif args.horizon_steps and args.horizon_steps > 0:
        horizon = int(args.horizon_steps)

    horizon = max(1, min(horizon, full_horizon))
    fut_full = fut_full.iloc[:horizon].copy()
    report["checks"]["final_horizon_steps"] = int(horizon)

    # target_indices 解析（防空列表）
    if args.target_indices is not None and args.target_indices.strip() != "":
        target_indices = [int(x) for x in args.target_indices.split(",") if x.strip() != ""]
        assert_or_raise(len(target_indices) > 0,
                        "target_indices 解析为空列表（例如你传了 ',' 或 ',,,'）。请传如 '0,1,2' 或不传。")
    else:
        target_indices = None

    report["checks"]["target_indices"] = ("ALL" if target_indices is None else target_indices)

    # 参数合法性检查（写入 report，便于定位）
    D = history.shape[1]
    report["checks"]["D"] = int(D)
    report["checks"]["L"] = int(args.L)
    report["checks"]["s"] = int(args.s)
    report["checks"]["trainlength"] = int(args.trainlength)

    # 关键：L合法
    assert_or_raise(1 <= int(args.L) <= D, f"L 必须在 [1,{D}]，当前 L={args.L}")
    assert_or_raise(int(args.s) >= 1, "s 必须 >= 1")
    assert_or_raise(int(args.trainlength) <= history.shape[0], "trainlength 大于 history 长度")

    optimize_hyp = (not args.no_optimize_hyp)

    print("=" * 80)
    print("PM2.5 后续预测（RDE-GPR，读取补值历史，不跑CSDI） - Debug增强版")
    print("=" * 80)
    print(json.dumps(meta, indent=4, ensure_ascii=False))
    print(f"\nforecast horizon = {horizon} steps")
    print("target_indices =", report["checks"]["target_indices"])

    # 预测
    debug_out = os.path.join(out_dir, "debug_steps") if args.debug else None
    preds, stds, debug_records = rdegpr_forecast_multivariate(
        history=history,
        future_truth=fut_full.values.astype(np.float64), 
        horizon=horizon,
        trainlength=args.trainlength,
        L=args.L,
        s=args.s,
        steps_ahead=args.steps_ahead,
        n_jobs=args.n_jobs,
        seed=args.seed,
        noise_strength=args.noise_strength,
        optimize_hyp=optimize_hyp,
        target_indices=target_indices,
        debug=args.debug,
        debug_steps=args.debug_steps,
        debug_out_dir=debug_out,
    )

    # 预测输出自检（0/NaN）
    report["pred_stats"] = basic_array_stats(preds, "preds")
    report["pred_first_row_stats"] = basic_array_stats(preds[0], "preds_first_row")
    report["pred_last_row_stats"] = basic_array_stats(preds[-1], "preds_last_row")
    report["checks"]["pred_all_zero"] = bool(np.allclose(preds, 0.0))
    report["checks"]["pred_has_nan"] = bool(np.isnan(preds).any())

    # 保存预测
    df_pred = pd.DataFrame(preds, index=fut_full.index, columns=df_full.columns)
    df_std = pd.DataFrame(stds, index=fut_full.index, columns=df_full.columns)

    pred_csv = os.path.join(out_dir, "future_pred.csv")
    std_csv = os.path.join(out_dir, "future_pred_std.csv")
    df_pred.to_csv(pred_csv)
    df_std.to_csv(std_csv)
    np.save(os.path.join(out_dir, "future_pred.npy"), preds)
    np.save(os.path.join(out_dir, "future_pred_std.npy"), stds)

    # debug report
    safe_json_dump(report, os.path.join(out_dir, "debug_report.json"))

    print("\n预测保存：")
    print("  ", pred_csv)
    print("  ", std_csv)
    print("debug_report：", os.path.join(out_dir, "debug_report.json"))
    if args.debug:
        print("debug_steps：", debug_out)

    # 可选评估与可视化（只用真值做对比，不参与预测）
    if not args.skip_metrics:
        y_true = fut_full.values.astype(np.float64)
        y_pred = preds.astype(np.float64)

        overall = compute_metrics(y_true, y_pred)
        safe_json_dump({"overall": overall, "horizon": horizon}, os.path.join(out_dir, "metrics.json"))

        per_dim = []
        for j, col in enumerate(df_full.columns):
            m = compute_metrics(y_true[:, j], y_pred[:, j])
            per_dim.append({"dim": j, "name": str(col), "rmse": m["rmse"], "mae": m["mae"]})
        pd.DataFrame(per_dim).to_csv(os.path.join(out_dir, "metrics_per_dim.csv"), index=False)

        missing_positions = None
        if args.missing_positions_path and os.path.exists(args.missing_positions_path):
            missing_positions = pd.read_csv(args.missing_positions_path, dtype={'column': str})
            print(f"已加载补值位置文件：{args.missing_positions_path}")

        save_plots(out_dir, fut_full.index, y_true, y_pred, args.plot_dim,
                   hist_index=hist_full.index, hist_values=history,
                   missing_positions=missing_positions, history_timesteps=args.history_timesteps,
                   column_names=list(df_full.columns))

        print("\n整体评估（仅对比，不参与预测）：")
        print(json.dumps(overall, indent=4, ensure_ascii=False))
        print("\n额外输出：metrics.json / metrics_per_dim.csv / plot_*.png")
    else:
        print("\n已跳过 metrics 与 plot（--skip_metrics）")

    # 最后输出一些高价值诊断提示
    if report["checks"]["pred_all_zero"]:
        print("\n[诊断提示] preds 全为0：")
        print("  1) 看 debug_report.json 里的 target_indices 是否为 ALL 或有效列表（不是空列表）")
        print("  2) 看 history_last_row_stats 是否全0（history末行全0时，持久性回退会导致未来全0）")
        print("  3) 开启 --debug 并查看 debug_steps/debug_sample_step*.json 的 ok_count/valid_pred_count")
        print("  4) 尝试加 --noise_strength 1e-4，或减小 L/s，先预测少量维度")

    if report["checks"]["pred_has_nan"]:
        print("\n[诊断提示] preds 存在 NaN：说明大量GPR失败且仍有部分未被回退覆盖。")
        print("  建议：开启 --debug，看 fail_degenerate_count 是否很高；再检查 history 是否存在异常列/常数列。")

    print("\n输出目录：", out_dir)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
