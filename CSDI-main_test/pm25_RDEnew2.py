# -*- coding: utf-8 -*-
"""
PM2.5 后续预测主程序：CSDI补值(前半段) + RDE-GPR滚动预测(后半段)

严格匹配你 main_model.py 的维度约定：
- observed_data / cond_mask: (B,K,L)
- observed_tp: (B,L)
- impute 输出: (B,n_samples,K,L)

流程：
1) 读取 pm25_ground.txt / pm25_missing.txt
2) 按 split_ratio 做时间切分：前半段用于补值&作为历史，后半段作为未来预测区间（预测时不看未来值）
3) 用训练好的 CSDI_PM25 对前半段 missing 进行补值，得到完整历史序列
4) 用 RDE-GPR 对后半段滚动预测（可以选择只预测部分维度，其余维度用“持久性”保持上一步值）
5) 保存 future_pred.csv / future_pred_std.csv；若有真值可计算 RMSE/MAE（仅评估，不参与预测）
6) 额外集成：
   - horizon 可调：--horizon_days / --horizon_steps
   - 可视化保存到 out_dir（png）
   - 指标统计保存：metrics.json / metrics_per_dim.csv

注意：
- RDE-GPR 对 36维 + 长horizon 非常慢。建议先用 --target_indices 0,1,2 和较小 s 验证流程。
"""

import os
import json
import random
import argparse
import datetime
import itertools
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize

import matplotlib.pyplot as plt

from main_model import CSDI_PM25


# =============================================================================
# 0) 可复现
# =============================================================================
def set_global_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)  # 如不兼容可注释


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# =============================================================================
# 1) 读取与时间切分
# =============================================================================
def load_pm25_files(ground_path: str, missing_path: str):
    df_full = pd.read_csv(ground_path, index_col="datetime", parse_dates=True).sort_index()
    df_miss = pd.read_csv(missing_path, index_col="datetime", parse_dates=True).sort_index()

    if not df_full.index.equals(df_miss.index):
        raise ValueError("ground 与 missing 的 datetime 索引不一致，请检查数据文件。")
    if df_full.shape[1] != df_miss.shape[1]:
        raise ValueError("ground 与 missing 的列数不一致，请检查数据文件。")
    return df_full, df_miss


def time_split_df(df_full: pd.DataFrame, df_miss: pd.DataFrame, split_ratio: float):
    total_len = len(df_full)
    split_point = int(total_len * split_ratio)

    hist_full = df_full.iloc[:split_point].copy()
    hist_miss = df_miss.iloc[:split_point].copy()

    fut_full = df_full.iloc[split_point:].copy()
    fut_miss = df_miss.iloc[split_point:].copy()

    meta = {
        "total_len": total_len,
        "split_point": split_point,
        "hist_len": len(hist_full),
        "fut_len": len(fut_full),
        "hist_start": str(hist_full.index.min()),
        "hist_end": str(hist_full.index.max()),
        "fut_start": str(fut_full.index.min()) if len(fut_full) else None,
        "fut_end": str(fut_full.index.max()) if len(fut_full) else None,
    }
    return hist_full, hist_miss, fut_full, fut_miss, meta


def infer_steps_per_day_from_index(idx: pd.DatetimeIndex, default_steps_per_day: int = 24) -> int:
    """
    根据时间索引推断每天有多少步（比如小时数据应为24）。
    用 median(diff) 来抗异常。
    """
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


# =============================================================================
# 2) CSDI：加载与“整段补值”
# =============================================================================
def load_csdi_pm25(model_path: str, config_json_path: str, device: str):
    """
    读取训练脚本保存的 config.json:
      {"args":..., "model_config":...}
    """
    with open(config_json_path, "r") as f:
        full_cfg = json.load(f)
    config = full_cfg["model_config"]

    model = CSDI_PM25(config, device).to(device)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model, config, full_cfg


def csdi_impute_chunk_pm25(
    model,
    chunk_values_LK: np.ndarray,     # (L,K) with NaN
    mean_K: np.ndarray,              # (K,)
    std_K: np.ndarray,               # (K,)
    device: str,
    n_samples: int,
    seed: int,
):
    """
    对一个 chunk (L,K) 做补值，严格匹配你的 main_model：
    - observed_data/cond_mask 传入 (B,K,L)
    - observed_tp 传入 (B,L)
    - impute 输出 (B,n,K,L)
    """
    L, K = chunk_values_LK.shape

    # cond_mask：已知位置=1，缺失=0
    cond_mask_LK = (~np.isnan(chunk_values_LK)).astype(np.float32)

    # observed_data：缺失填0，然后标准化，再乘 cond_mask（与你训练数据准备一致）
    x0_LK = np.nan_to_num(chunk_values_LK, nan=0.0).astype(np.float32)
    x_norm_LK = ((x0_LK - mean_K) / std_K) * cond_mask_LK  # (L,K)

    # -> torch (B,K,L)
    observed_data = torch.from_numpy(x_norm_LK).unsqueeze(0).to(device)     # (1,L,K)
    observed_data = observed_data.permute(0, 2, 1).contiguous()             # (1,K,L)

    cond_mask = torch.from_numpy(cond_mask_LK).unsqueeze(0).to(device)      # (1,L,K)
    cond_mask = cond_mask.permute(0, 2, 1).contiguous()                     # (1,K,L)

    # timepoints: (B,L)
    observed_tp = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)  # (1,L)

    # 固定采样随机性（impute 内部用 torch.randn_like）
    torch.manual_seed(int(seed))

    with torch.no_grad():
        side_info = model.get_side_info(observed_tp, cond_mask)                 # (1,side_dim,K,L)
        samples = model.impute(observed_data, cond_mask, side_info, n_samples)  # (1,n,K,L)

    # -> numpy (n,L,K)
    samples_nLK = samples[0].permute(0, 2, 1).contiguous().cpu().numpy()

    # 样本均值：归一化空间
    pred_norm_LK = samples_nLK.mean(axis=0)  # (L,K)

    # 反归一化
    pred_LK = pred_norm_LK * std_K + mean_K

    # 只填缺失位置，保留已知位置
    out = chunk_values_LK.copy()
    miss = np.isnan(out)
    out[miss] = pred_LK[miss]
    return out


def csdi_impute_history_long(
    model,
    df_hist_missing: pd.DataFrame,
    mean_K: np.ndarray,
    std_K: np.ndarray,
    device: str,
    n_samples: int,
    chunk_len: int,
    stride: int,
    seed: int,
):
    """
    对“前半段历史序列”做整段补值（分块，避免显存爆）
    - chunk_len 建议等于训练 eval_length（比如 36）
    - stride=chunk_len 表示不重叠
    """
    values = df_hist_missing.values.astype(np.float32)  # (T,K)
    T, K = values.shape
    out = values.copy()

    nan_mask = np.isnan(values)

    for start in tqdm(range(0, T, stride), desc="CSDI imputing history"):
        end = min(start + chunk_len, T)
        chunk = out[start:end].copy()

        if not np.isnan(chunk).any():
            continue

        chunk_seed = int(seed + start)  # 与 start 绑定，保证可复现
        filled = csdi_impute_chunk_pm25(
            model=model,
            chunk_values_LK=chunk,
            mean_K=mean_K,
            std_K=std_K,
            device=device,
            n_samples=n_samples,
            seed=chunk_seed,
        )
        out[start:end] = filled

    # 再保险：已知值不变
    out[~nan_mask] = values[~nan_mask]
    return pd.DataFrame(out, index=df_hist_missing.index, columns=df_hist_missing.columns)


# =============================================================================
# 3) RDE-GPR
# =============================================================================
class GaussianProcessRegressor:
    def __init__(self, kernel='rbf', noise=1e-6):
        self.kernel = kernel
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.L = None
        self.alpha = None
        self.params = None
        self.mu_X, self.sigma_X = None, None
        self.mu_y, self.sigma_y = None, None

    def _rbf_kernel(self, X1, X2, sigma_f, l):
        sqdist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return sigma_f**2 * np.exp(-sqdist / (2 * l**2))

    def _kernel_matrix(self, X1, X2):
        sigma_f, l, sigma_n = self.params
        K = self._rbf_kernel(X1, X2, sigma_f, l)
        if X1 is X2:
            K += (sigma_n**2 + self.noise) * np.eye(X1.shape[0])
        return K

    def fit(self, X_train, y_train, init_params=(1.0, 1.0, 0.1), optimize=False):
        X_train, self.mu_X, self.sigma_X = self._normalize(X_train)
        y_train, self.mu_y, self.sigma_y = self._normalize(y_train)

        self.X_train = X_train
        self.y_train = y_train

        self.params = self._optimize_hyperparams(init_params) if optimize else np.array(init_params)

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

        # 数值误差可能导致对角线出现负值，clip 到 0 避免 sqrt warning
        diag = np.diag(y_cov)
        diag = np.maximum(diag, 0.0)
        y_std = np.sqrt(diag) * self.sigma_y
        return y_mean, y_std

    def _optimize_hyperparams(self, init_params):
        def nll(params):
            sigma_f, l, sigma_n = params
            K = self._rbf_kernel(self.X_train, self.X_train, sigma_f, l) + (sigma_n**2 + 1e-5) * np.eye(len(self.X_train))
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
    对一个组合 comb 拟合一次 GPR 并做一步预测
    """
    try:
        trainlength = len(traindata)
        trainX = traindata[:trainlength - steps_ahead, list(comb)]
        trainy = traindata[steps_ahead:trainlength, target_idx]
        testX = traindata[trainlength - steps_ahead, list(comb)].reshape(1, -1)

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        combined_X = np.vstack([trainX, testX])
        combined_X_scaled = scaler_X.fit_transform(combined_X)
        trainX_scaled = combined_X_scaled[:-1]
        testX_scaled = combined_X_scaled[-1:]

        trainy_scaled = scaler_y.fit_transform(trainy.reshape(-1, 1)).flatten()

        gp = GaussianProcessRegressor(noise=1e-6)
        gp.fit(trainX_scaled, trainy_scaled, init_params=(1.0, 0.1, 0.1), optimize=optimize_hyp)

        pred_scaled, std_scaled = gp.predict(testX_scaled, return_std=True)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        return float(pred), float(std_scaled[0])
    except Exception:
        return np.nan, np.nan


def rdegpr_predict_next_for_target(traindata, target_idx, L, s, steps_ahead, pool, rng, optimize_hyp=True):
    """
    对单个 target_idx 做“一步预测”，随机抽 s 个组合并行拟合，KDE 融合
    """
    D = traindata.shape[1]
    combs = list(itertools.combinations(range(D), L))
    rng.shuffle(combs)
    selected = combs[:min(s, len(combs))]

    preds = pool.map(
        partial(_parallel_predict_one_comb, traindata=traindata, target_idx=target_idx,
                steps_ahead=steps_ahead, optimize_hyp=optimize_hyp),
        selected
    )

    pred_values = np.array([p[0] for p in preds], dtype=np.float64)
    pred_stds = np.array([p[1] for p in preds], dtype=np.float64)
    valid = ~np.isnan(pred_values) & ~np.isnan(pred_stds)
    valid_preds = pred_values[valid]

    if len(valid_preds) == 0:
        return np.nan, np.nan
    if len(valid_preds) == 1:
        return float(valid_preds[0]), 0.0

    try:
        kde = gaussian_kde(valid_preds)
        xi = np.linspace(valid_preds.min(), valid_preds.max(), 1000)
        density = kde(xi)
        final_pred = float(np.sum(xi * density) / np.sum(density))
        final_std = float(np.std(valid_preds))
        return final_pred, final_std
    except Exception:
        return float(np.mean(valid_preds)), float(np.std(valid_preds))


def rdegpr_forecast_multivariate(history, horizon, trainlength, L, s, steps_ahead, n_jobs, seed,
                                 noise_strength=0.0, optimize_hyp=True, target_indices=None):
    """
    多变量滚动预测：
    - 每一步用最近 trainlength 的窗口 traindata
    - 对 target_indices 指定的维度做预测
    - 未预测的维度使用“持久性”（保持上一时刻值）
    """
    history = np.asarray(history, dtype=np.float64)
    T_hist, D = history.shape

    if T_hist < trainlength:
        raise ValueError(f"history长度({T_hist}) < trainlength({trainlength})")

    if target_indices is None:
        target_indices = list(range(D))
    else:
        target_indices = list(target_indices)

    preds = np.zeros((horizon, D), dtype=np.float64)
    stds = np.zeros((horizon, D), dtype=np.float64)

    base_rng = np.random.default_rng(int(seed))
    pool = mp.Pool(processes=int(n_jobs))

    try:
        seq = history.copy()

        for step in tqdm(range(horizon), desc="RDE-GPR forecasting"):
            traindata = seq[-trainlength:].copy()

            if noise_strength > 0:
                traindata = traindata + noise_strength * base_rng.standard_normal(size=traindata.shape)

            next_vec = seq[-1].copy()
            next_std = np.zeros((D,), dtype=np.float64)

            for j in target_indices:
                tj_rng = np.random.default_rng(int(seed + 100000 * step + 1000 * j))
                pred_j, std_j = rdegpr_predict_next_for_target(
                    traindata=traindata,
                    target_idx=j,
                    L=L,
                    s=s,
                    steps_ahead=steps_ahead,
                    pool=pool,
                    rng=tj_rng,
                    optimize_hyp=optimize_hyp,
                )
                next_vec[j] = pred_j
                next_std[j] = std_j

            preds[step] = next_vec
            stds[step] = next_std
            seq = np.vstack([seq, next_vec.reshape(1, -1)])
    finally:
        pool.close()
        pool.join()

    return preds, stds


# =============================================================================
# 4) 指标与可视化
# =============================================================================
def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan}
    diff = y_true[mask] - y_pred[mask]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    return {"rmse": rmse, "mae": mae}


def save_plots(out_dir: str, fut_index: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray,
               dim0: int = 0):
    """
    保存两张图：
    1) dim0 真值 vs 预测（折线）
    2) 每维 RMSE（条形图）
    """
    ensure_dir(out_dir)

    # 1) dim0 time series
    if y_true is not None and y_pred is not None and y_true.ndim == 2 and y_pred.ndim == 2:
        dim0 = int(dim0)
        dim0 = max(0, min(dim0, y_true.shape[1] - 1))

        plt.figure(figsize=(14, 5))
        plt.plot(fut_index, y_true[:, dim0], label=f"True (dim {dim0})")
        plt.plot(fut_index, y_pred[:, dim0], label=f"Pred (dim {dim0})")
        plt.xlabel("Time")
        plt.ylabel("PM2.5")
        plt.title(f"Forecast vs True on Future Segment (dim {dim0})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"plot_forecast_dim{dim0}.png"), dpi=150)
        plt.close()

        # 2) RMSE per dim
        rmse_list = []
        for j in range(y_true.shape[1]):
            m = compute_metrics(y_true[:, j], y_pred[:, j])
            rmse_list.append(m["rmse"])

        plt.figure(figsize=(14, 5))
        plt.bar(np.arange(len(rmse_list)), rmse_list)
        plt.xlabel("Dimension")
        plt.ylabel("RMSE")
        plt.title("RMSE per Dimension on Future Segment")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "plot_rmse_per_dim.png"), dpi=150)
        plt.close()


# =============================================================================
# 5) main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="PM2.5 Forecasting: CSDI Impute first-half + RDE-GPR forecast second-half")

    # ---- CSDI 模型 ----
    parser.add_argument("--run_folder", type=str,
                        default="/home/rhl/CSDI-main_test/save/pm25_imputation_split0.5_train0.7_valid0.15_mask0.1_seed42_20260127_151505",
                        help="训练脚本生成的文件夹（里面应有 model.pth / config.json）")
    parser.add_argument("--model_path", type=str,
                        default="/home/rhl/CSDI-main_test/save/pm25_imputation_split0.5_train0.7_valid0.15_mask0.1_seed42_20260127_151505/model.pth",
                        help="显式指定 model.pth 路径（可选）")
    parser.add_argument("--config_json", type=str,
                        default="/home/rhl/CSDI-main_test/save/pm25_imputation_split0.5_train0.7_valid0.15_mask0.1_seed42_20260127_151505/config.json",
                        help="显式指定 config.json 路径（可选）")

    # ---- 数据 ----
    parser.add_argument("--ground_path", type=str,
                        default="./data/pm25/Code/STMVL/SampleData/pm25_ground.txt")
    parser.add_argument("--missing_path", type=str,
                        default="./data/pm25/Code/STMVL/SampleData/pm25_missing.txt")
    parser.add_argument("--meanstd_path", type=str,
                        default="./data/pm25/pm25_meanstd.pk")

    # ---- 基础 ----
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_ratio", type=float, default=0.5)

    # ---- 未来预测长度控制（新增）----
    parser.add_argument("--horizon_days", type=float, default=0.0,
                        help="预测未来多少天（>0 生效），例如 10 表示预测 10 天")
    parser.add_argument("--horizon_steps", type=int, default=0,
                        help="预测未来多少步（>0 生效）。例如小时数据 10天=240步。若同时给了 horizon_days，则以 days 为准。")

    # ---- CSDI补值 ----
    parser.add_argument("--imputed_history_path", type=str, default="",
                        help="如果已有补值好的前半段 csv，可直接读它跳过CSDI补值")
    parser.add_argument("--impute_n_samples", type=int, default=50)
    parser.add_argument("--chunk_len", type=int, default=36)
    parser.add_argument("--stride", type=int, default=36)

    # ---- RDE-GPR ----
    parser.add_argument("--trainlength", type=int, default=30)
    parser.add_argument("--L", type=int, default=4, help="随机嵌入组合维数 L（你要可调的那个）")
    parser.add_argument("--s", type=int, default=50, help="组合抽样数（建议先小，验证流程）")
    parser.add_argument("--steps_ahead", type=int, default=1)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--noise_strength", type=float, default=0.0)
    parser.add_argument("--no_optimize_hyp", action="store_true")

    parser.add_argument("--target_indices", type=str, default="",
                        help="要预测的维度索引，逗号分隔；为空=预测全部维度。未预测维度用持久性。")

    # ---- 输出与可视化 ----
    parser.add_argument("--out_dir", type=str, default="", help="输出目录（可选）")
    parser.add_argument("--plot_dim", type=int, default=0, help="可视化画哪一维（默认0）")

    args = parser.parse_args()

    set_global_seed(args.seed)

    # 输出目录
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"./save/pm25_forecast_rdegpr_split{args.split_ratio}_seed{args.seed}_{now}/"
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    print("=" * 80)
    print("PM2.5 后续预测：CSDI补值(前半) + RDE-GPR预测(后半)")
    print("=" * 80)
    print(json.dumps(vars(args), indent=4, ensure_ascii=False))

    # 读数据切分
    df_full, df_miss = load_pm25_files(args.ground_path, args.missing_path)
    hist_full, hist_miss, fut_full, fut_miss, meta = time_split_df(df_full, df_miss, args.split_ratio)
    print("\n数据切分信息：")
    print(json.dumps(meta, indent=4, ensure_ascii=False))

    if meta["fut_len"] <= 0:
        raise ValueError("后半段长度为0，检查 split_ratio。")

    # mean/std
    import pickle
    with open(args.meanstd_path, "rb") as f:
        mean_K, std_K = pickle.load(f)
    mean_K = np.asarray(mean_K, dtype=np.float32)
    std_K = np.asarray(std_K, dtype=np.float32)
    std_K = np.where(std_K == 0, 1.0, std_K).astype(np.float32)

    # 历史补值
    if args.imputed_history_path:
        print("\n读取已补值历史：", args.imputed_history_path)
        df_hist_imputed = pd.read_csv(args.imputed_history_path, index_col="datetime", parse_dates=True).sort_index()
        if not df_hist_imputed.index.equals(hist_miss.index):
            raise ValueError("imputed_history 的 datetime 索引与前半段不一致。")
        if df_hist_imputed.shape[1] != hist_miss.shape[1]:
            raise ValueError("imputed_history 的维度与PM2.5维度不一致。")
    else:
        if args.run_folder:
            model_path = args.model_path or os.path.join(args.run_folder, "model.pth")
            config_json = args.config_json or os.path.join(args.run_folder, "config.json")
        else:
            model_path = args.model_path
            config_json = args.config_json

        if not model_path or not config_json:
            raise ValueError("请提供 --run_folder 或显式提供 --model_path 与 --config_json。")

        print("\n加载CSDI模型：")
        print("  model_path =", model_path)
        print("  config_json =", config_json)
        model, _, _ = load_csdi_pm25(model_path, config_json, args.device)

        print("\n开始CSDI补值（前半段）...")
        df_hist_imputed = csdi_impute_history_long(
            model=model,
            df_hist_missing=hist_miss,
            mean_K=mean_K,
            std_K=std_K,
            device=args.device,
            n_samples=args.impute_n_samples,
            chunk_len=args.chunk_len,
            stride=args.stride,
            seed=args.seed,
        )
        hist_csv = os.path.join(out_dir, "history_imputed.csv")
        df_hist_imputed.to_csv(hist_csv)
        print("历史补值已保存：", hist_csv)

    # 解析 target_indices
    if args.target_indices.strip():
        target_indices = [int(x) for x in args.target_indices.split(",") if x.strip() != ""]
    else:
        target_indices = None  # 全维

    optimize_hyp = (not args.no_optimize_hyp)

    # =========================
    # 预测长度 horizon（可控）
    # =========================
    full_horizon = meta["fut_len"]
    horizon = full_horizon

    if args.horizon_days and args.horizon_days > 0:
        steps_per_day = infer_steps_per_day_from_index(fut_full.index, default_steps_per_day=24)
        horizon = int(round(args.horizon_days * steps_per_day))
    elif args.horizon_steps and args.horizon_steps > 0:
        horizon = int(args.horizon_steps)

    horizon = max(1, min(horizon, full_horizon))

    # 截断未来段（只为索引对齐 + 事后评估；预测过程不使用未来真值）
    fut_full = fut_full.iloc[:horizon].copy()

    # 预测
    history = df_hist_imputed.values.astype(np.float64)

    if target_indices is None:
        print("\n[警告] 你选择预测全部 36 维时会很慢。建议先用 --target_indices 0,1,2 测试流程。")

    print(f"\n开始RDE-GPR滚动预测：horizon={horizon} steps (~ {args.horizon_days} days if set) ...")
    preds, stds = rdegpr_forecast_multivariate(
        history=history,
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
    )

    # 保存预测
    df_pred = pd.DataFrame(preds, index=fut_full.index, columns=df_full.columns)
    df_std = pd.DataFrame(stds, index=fut_full.index, columns=df_full.columns)

    pred_csv = os.path.join(out_dir, "future_pred.csv")
    std_csv = os.path.join(out_dir, "future_pred_std.csv")
    df_pred.to_csv(pred_csv)
    df_std.to_csv(std_csv)

    np.save(os.path.join(out_dir, "future_pred.npy"), preds)
    np.save(os.path.join(out_dir, "future_pred_std.npy"), stds)

    print("\n预测保存：")
    print("  ", pred_csv)
    print("  ", std_csv)

    # 评估（不参与预测）
    y_true = fut_full.values.astype(np.float64)
    y_pred = preds.astype(np.float64)

    overall = compute_metrics(y_true, y_pred)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"overall": overall, "horizon": horizon}, f, indent=4, ensure_ascii=False)

    # 每维指标
    per_dim = []
    for j, col in enumerate(df_full.columns):
        m = compute_metrics(y_true[:, j], y_pred[:, j])
        per_dim.append({"dim": j, "name": str(col), "rmse": m["rmse"], "mae": m["mae"]})
    df_per_dim = pd.DataFrame(per_dim)
    df_per_dim.to_csv(os.path.join(out_dir, "metrics_per_dim.csv"), index=False)

    # 可视化
    save_plots(
        out_dir=out_dir,
        fut_index=fut_full.index,
        y_true=y_true,
        y_pred=y_pred,
        dim0=args.plot_dim,
    )

    print("\n后半段整体评估（仅对比，不参与预测）：")
    print(json.dumps(overall, indent=4, ensure_ascii=False))
    print("\n输出目录：", out_dir)
    print("\n额外输出：")
    print("  metrics.json")
    print("  metrics_per_dim.csv")
    print(f"  plot_forecast_dim{int(args.plot_dim)}.png")
    print("  plot_rmse_per_dim.png")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
