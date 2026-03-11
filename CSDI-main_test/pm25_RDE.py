# -*- coding: utf-8 -*-
"""
PM2.5 后续预测主程序：CSDI补值(前半段) + RDE-GPR预测(后半段)

整体流程（与你当前“随机时间划分补值”逻辑一致）：
1) 读取 pm25_ground.txt / pm25_missing.txt（datetime 为索引）
2) 按 split_ratio 把数据分为两段：
   - 前 split_ratio：只用于“补值 + 作为历史训练序列”
   - 后 (1-split_ratio)：作为“未来预测区间”（预测时不看未来数值）
3) 用训练好的 CSDI_PM25 对前半段（missing版本）进行补值，得到完整历史序列
4) 使用 RDE-GPR（随机嵌入 + GPR 集成）对后半段逐步滚动预测
5) 保存预测结果到 csv/npy；若提供真值则计算 RMSE/MAE（仅评估用）

重要说明（可复现性）：
- 所有随机（CSDI采样 / RDE组合抽样 / 噪声扰动）都绑定到同一个 seed
- 多进程并行（multiprocessing）不引入额外随机性

依赖：
- torch, numpy, pandas, pyyaml
- scipy, scikit-learn, tqdm

使用示例：
python pm25_forecast_rdegpr.py \
  --run_folder ./save/pm25_imputation_split0.5_train0.7_valid0.15_mask0.1_seed42_20260127_123456 \
  --config_json ./save/pm25_imputation_split0.5_train0.7_valid0.15_mask0.1_seed42_20260127_123456/config.json \
  --device cuda:0 \
  --split_ratio 0.5 \
  --seed 42 \
  --impute_n_samples 50 \
  --trainlength 30 --L 4 --s 200 --n_jobs 8

如果你不想重新跑CSDI补值（你已经补好了历史序列），可用：
  --imputed_history_path your_imputed_first_half.csv
（csv 要求：datetime索引 + 36列站点，缺失已填好）

"""

import os
import json
import yaml
import time
import math
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

# 你的工程里已有
from main_model import CSDI_PM25


# =============================================================================
# 0) 全局可复现设置
# =============================================================================
def set_global_seed(seed: int):
    """固定 random / numpy / torch 的随机种子，尽量保证跨运行可复现。"""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 更强确定性（可能略慢）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 如果你环境支持且想更强约束，可打开：
    # torch.use_deterministic_algorithms(True)


# =============================================================================
# 1) 读取 PM2.5 数据并按时间切分
# =============================================================================
def load_pm25_files(ground_path: str, missing_path: str):
    """
    读取 PM2.5 文件：
    - ground_path: 完整/真值（通常无缺失或较少缺失）
    - missing_path: 带缺失的版本（用于补值 conditioning）
    """
    df_full = pd.read_csv(ground_path, index_col="datetime", parse_dates=True).sort_index()
    df_miss = pd.read_csv(missing_path, index_col="datetime", parse_dates=True).sort_index()

    if not df_full.index.equals(df_miss.index):
        raise ValueError("ground 与 missing 的 datetime 索引不一致，请检查数据文件。")

    if df_full.shape[1] != df_miss.shape[1]:
        raise ValueError("ground 与 missing 的维度不一致，请检查数据文件。")

    return df_full, df_miss


def time_split_df(df_full: pd.DataFrame, df_miss: pd.DataFrame, split_ratio: float):
    """按时间位置切分为历史（前半）和未来（后半）。"""
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
        "fut_start": str(fut_full.index.min()) if len(fut_full) > 0 else None,
        "fut_end": str(fut_full.index.max()) if len(fut_full) > 0 else None,
    }
    return hist_full, hist_miss, fut_full, fut_miss, meta


# =============================================================================
# 2) CSDI 模型加载与“整段补值”
# =============================================================================
def load_csdi_pm25_from_config_json(
    model_path: str,
    config_json_path: str,
    device: str = "cuda:0",
):
    """
    读取你训练脚本保存的 config.json（里面包含 model_config），并加载模型参数。
    config.json结构（你训练脚本保存的）：
      {
        "args": {...},
        "model_config": {...}   # 训练时用于构建模型的 config dict
      }
    """
    with open(config_json_path, "r") as f:
        full_cfg = json.load(f)

    config = full_cfg["model_config"]
    model = CSDI_PM25(config, device).to(device)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model, config, full_cfg


def csdi_impute_chunk(
    model,
    chunk_values: np.ndarray,   # shape (T, D) with NaN
    mean: np.ndarray,           # shape (D,)
    std: np.ndarray,            # shape (D,)
    device: str,
    n_samples: int,
    seed: int,
):
    """
    对一个时间块（T x D）做一次 CSDI 补值，返回填好的 chunk（T x D）。
    - conditioning：原来非NaN位置
    - 只填 NaN 的位置，并保留原始观测值
    """
    T, D = chunk_values.shape
    observed_mask = (~np.isnan(chunk_values)).astype(np.float32)  # 1=观测，0=缺失

    # 归一化（缺失先填0，再乘mask保持缺失处为0）
    x0 = np.nan_to_num(chunk_values, nan=0.0).astype(np.float32)
    x_norm = ((x0 - mean) / std) * observed_mask  # (T,D)

    # 转张量并整理成 (B, D, T)
    observed_data = torch.from_numpy(x_norm).unsqueeze(0).to(device)       # (1,T,D)
    observed_mask_t = torch.from_numpy(observed_mask).unsqueeze(0).to(device)  # (1,T,D)
    observed_data = observed_data.permute(0, 2, 1)      # (1,D,T)
    cond_mask = observed_mask_t.permute(0, 2, 1)        # (1,D,T)

    # timepoints (B,T)
    observed_tp = torch.arange(T, dtype=torch.float32).unsqueeze(0).to(device)

    # 固定采样随机性
    torch.manual_seed(int(seed))

    with torch.no_grad():
        side_info = model.get_side_info(observed_tp, cond_mask)
        samples = model.impute(observed_data, cond_mask, side_info, n_samples)  # (1, n, D, T) 通常如此

        # -> (n, T, D)
        samples = samples.squeeze(0).permute(0, 2, 1).contiguous().cpu().numpy()

    # 样本均值作为补值结果（仍是归一化空间）
    pred_norm = samples.mean(axis=0)  # (T,D)

    # 反归一化
    pred = pred_norm * std + mean  # (T,D)

    # 只填缺失处，保留观测值
    out = chunk_values.copy()
    miss_pos = np.isnan(out)
    out[miss_pos] = pred[miss_pos]
    return out


def csdi_impute_long_sequence(
    model,
    df_hist_missing: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    n_samples: int,
    chunk_len: int,
    stride: int,
    seed: int,
):
    """
    对“前半段历史序列”做整段补值（支持长序列分块，避免一次性塞满显存）：
    - 使用 chunk_len 分块（默认建议与训练 eval_length 一致，比如36）
    - stride 默认等于 chunk_len（不重叠）；你也可以设更小做重叠融合（这里先给最稳定简单版本）
    """
    values = df_hist_missing.values.astype(np.float32)  # (T,D)
    T, D = values.shape
    out = values.copy()

    # 记录原始 NaN
    nan_mask = np.isnan(values)

    # 分块补值
    for start in tqdm(range(0, T, stride), desc="CSDI imputing history"):
        end = min(start + chunk_len, T)
        chunk = out[start:end].copy()

        # 如果块里没有缺失，跳过（节省时间）
        if not np.isnan(chunk).any():
            continue

        # 固定每个chunk的采样随机性：seed + start
        chunk_seed = int(seed + start)

        filled = csdi_impute_chunk(
            model=model,
            chunk_values=chunk,
            mean=mean,
            std=std,
            device=device,
            n_samples=n_samples,
            seed=chunk_seed,
        )
        out[start:end] = filled

    # 确保观测值不被改动（再保险）
    out[~nan_mask] = values[~nan_mask]
    df_out = pd.DataFrame(out, index=df_hist_missing.index, columns=df_hist_missing.columns)
    return df_out


# =============================================================================
# 3) RDE-GPR：高斯过程回归（与你给的版本一致，整理成可复用）
# =============================================================================
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize


class GaussianProcessRegressor:
    def __init__(self, kernel='rbf', noise=1e-6):
        """
        高斯过程回归模型初始化
        :param kernel: 核函数类型，默认为RBF核
        :param noise: 噪声项（数值稳定性）
        """
        self.kernel = kernel
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.L = None          # Cholesky分解下三角矩阵
        self.alpha = None       # 后验均值参数
        self.params = None      # 超参数 [sigma_f, l, sigma_n]
        self.mu_X, self.sigma_X = None, None  # 数据标准化参数
        self.mu_y, self.sigma_y = None, None

    def _rbf_kernel(self, X1, X2, sigma_f, l):
        """RBF核函数实现"""
        sqdist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return sigma_f**2 * np.exp(-sqdist / (2 * l**2))

    def _kernel_matrix(self, X1, X2):
        """根据核类型生成协方差矩阵"""
        sigma_f, l, sigma_n = self.params
        K = self._rbf_kernel(X1, X2, sigma_f, l)
        if X1 is X2:
            K += (sigma_n**2 + self.noise) * np.eye(X1.shape[0])
        return K

    def fit(self, X_train, y_train, init_params=(1.0, 1.0, 0.1), optimize=False):
        """
        训练GPR模型
        :param X_train: 训练数据 (n_samples, n_features)
        :param y_train: 训练标签 (n_samples,)
        :param init_params: 初始超参数 [sigma_f, l, sigma_n]
        :param optimize: 是否优化超参数
        """
        # 数据标准化
        X_train, self.mu_X, self.sigma_X = self._normalize(X_train)
        y_train, self.mu_y, self.sigma_y = self._normalize(y_train)
        self.X_train = X_train
        self.y_train = y_train

        # 超参数优化
        if optimize:
            self.params = self._optimize_hyperparams(init_params)
        else:
            self.params = np.array(init_params)

        # 计算核矩阵并进行Cholesky分解
        K = self._kernel_matrix(X_train, X_train)
        try:
            self.L = cholesky(K, lower=True)
        except np.linalg.LinAlgError:
            K += self.noise * np.eye(K.shape[0])
            self.L = cholesky(K, lower=True)
        self.alpha = solve_triangular(self.L.T, solve_triangular(self.L, y_train, lower=True))

    def predict(self, X_test, return_std=False):
        """对新数据进行预测"""
        X_test = (X_test - self.mu_X) / self.sigma_X  # 标准化
        K_star = self._kernel_matrix(self.X_train, X_test)
        y_mean = K_star.T @ self.alpha
        y_mean = y_mean * self.sigma_y + self.mu_y    # 逆标准化

        if return_std:
            v = solve_triangular(self.L, K_star, lower=True)
            K_starstar = self._kernel_matrix(X_test, X_test)
            y_cov = K_starstar - v.T @ v
            y_std = np.sqrt(np.diag(y_cov)) * self.sigma_y
            return y_mean, y_std
        else:
            return y_mean

    def _optimize_hyperparams(self, init_params):
        def neg_log_likelihood(params):
            sigma_f, l, sigma_n = params
            K = self._rbf_kernel(self.X_train, self.X_train, sigma_f, l) + (sigma_n**2 + 1e-5) * np.eye(len(self.X_train))
            try:
                L = cholesky(K, lower=True)
            except np.linalg.LinAlgError:
                return np.inf
            alpha = solve_triangular(L.T, solve_triangular(L, self.y_train, lower=True))
            return 0.5 * self.y_train.T @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * len(self.y_train) * np.log(2*np.pi)

        bounds = [(1e-5, 1e2), (1e-5, 1e2), (1e-5, 1e2)]
        res = minimize(neg_log_likelihood, init_params, method='L-BFGS-B', bounds=bounds)
        return res.x

    @staticmethod
    def _normalize(X):
        """数据标准化"""
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma = np.where(sigma == 0, 1.0, sigma)
        return (X - mu) / sigma, mu, sigma


def _parallel_predict_one_comb(comb, traindata, target_idx, steps_ahead=1, optimize_hyp=True):
    """
    给定一个组合 comb（若干维特征索引），拟合一个 GPR 并做一步预测。
    返回 (pred, std)
    """
    try:
        trainlength = len(traindata)
        # X_t -> y_{t+steps_ahead}
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


def rdegpr_predict_next_for_target(
    traindata: np.ndarray,     # (trainlength, D)
    target_idx: int,
    L: int,
    s: int,
    steps_ahead: int,
    pool,
    rng: np.random.Generator,
    optimize_hyp: bool = True,
):
    """
    对单个目标维 target_idx 做“一步预测”，使用：
    - 从所有组合 C(D,L) 中随机抽 s 个组合
    - 并行拟合 s 个 GPR
    - 对预测分布用 KDE 融合（失败则退化为均值/中位数）
    """
    D = traindata.shape[1]
    combs = list(itertools.combinations(range(D), L))
    # 用 rng 打乱并选取前 s 个，保证可复现
    rng.shuffle(combs)
    selected_combs = combs[:min(s, len(combs))]

    preds = pool.map(
        partial(
            _parallel_predict_one_comb,
            traindata=traindata,
            target_idx=target_idx,
            steps_ahead=steps_ahead,
            optimize_hyp=optimize_hyp,
        ),
        selected_combs
    )

    pred_values = np.array([p[0] for p in preds], dtype=np.float64)
    pred_stds = np.array([p[1] for p in preds], dtype=np.float64)

    valid = ~np.isnan(pred_values) & ~np.isnan(pred_stds)
    valid_preds = pred_values[valid]

    if len(valid_preds) == 0:
        return np.nan, np.nan
    if len(valid_preds) == 1:
        return float(valid_preds[0]), 0.0

    # KDE 融合（与你 test2.py 的精神一致）
    try:
        kde = gaussian_kde(valid_preds)
        xi = np.linspace(valid_preds.min(), valid_preds.max(), 1000)
        density = kde(xi)
        final_pred = float(np.sum(xi * density) / np.sum(density))
        final_std = float(np.std(valid_preds))
        return final_pred, final_std
    except Exception:
        return float(np.mean(valid_preds)), float(np.std(valid_preds))


def rdegpr_forecast_multivariate(
    history: np.ndarray,        # (T_hist, D) 已补值完整历史
    horizon: int,               # 未来步数（后半段长度）
    trainlength: int,
    L: int,
    s: int,
    steps_ahead: int,
    n_jobs: int,
    seed: int,
    noise_strength: float = 0.0,
    optimize_hyp: bool = True,
    target_indices=None,        # None 表示全维；也可传 list[int]
):
    """
    多变量滚动预测：
    - 每一步：用最近 trainlength 的窗口拟合（对每个 target 各自做RDE-GPR集成）
    - 得到下一时刻的 D 维预测向量，append 到序列末尾
    - 预测过程中不使用未来真值

    注意：该方法计算量非常大（尤其 D=36 且 horizon 很长）。
    建议你先用较小 horizon / 较小 s / 只预测部分站点做验证。
    """
    history = np.asarray(history, dtype=np.float64)
    T_hist, D = history.shape

    if target_indices is None:
        target_indices = list(range(D))
    else:
        target_indices = list(target_indices)

    if T_hist < trainlength:
        raise ValueError(f"history长度({T_hist}) < trainlength({trainlength})，无法开始滚动预测。")

    # 用一个固定 RNG 控制所有“组合抽样”和“噪声”
    base_rng = np.random.default_rng(int(seed))

    # 未来预测结果
    preds = np.zeros((horizon, D), dtype=np.float64)
    stds = np.zeros((horizon, D), dtype=np.float64)

    # 多进程池
    pool = mp.Pool(processes=int(n_jobs))

    try:
        # 工作序列（包含 history + 逐步 append 的预测）
        seq = history.copy()

        for step in tqdm(range(horizon), desc="RDE-GPR forecasting"):
            # 取训练窗口
            traindata = seq[-trainlength:].copy()

            # 可选：加一点噪声（数值稳定）；为了可复现，用 base_rng
            if noise_strength > 0:
                traindata = traindata + noise_strength * base_rng.standard_normal(size=traindata.shape)

            # 每一步再派生一个 rng（保证与 step 绑定、可复现）
            step_rng = np.random.default_rng(int(seed + 100000 * step))

            next_vec = seq[-1].copy()  # 默认先复制一份，未预测的维度保持上一时刻（如果你只预测部分站点）
            next_std = np.zeros((D,), dtype=np.float64)

            # 对每个目标维做一次一步预测
            for j in target_indices:
                # 每个 target 的 rng 再细分，保证并行与顺序变化不影响抽样结果
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

            # append 到序列末尾，继续滚动
            seq = np.vstack([seq, next_vec.reshape(1, -1)])

    finally:
        pool.close()
        pool.join()

    return preds, stds


# =============================================================================
# 4) 评估与保存
# =============================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """计算 RMSE / MAE（忽略 NaN）。"""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan}
    diff = y_true[mask] - y_pred[mask]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    return {"rmse": rmse, "mae": mae}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# =============================================================================
# 5) 主程序
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="PM2.5 Forecasting: CSDI Imputation + RDE-GPR Prediction")

    # ========== CSDI 模型相关 ==========
    parser.add_argument("--run_folder", type=str, default="/home/rhl/CSDI-main_test/save/pm25_imputation_split0.5_train0.7_valid0.15_mask0.1_seed42_20260127_151505", help="训练脚本生成的文件夹（里面应有 model.pth / config.json）")
    parser.add_argument("--model_path", type=str, default="/home/rhl/CSDI-main_test/save/pm25_imputation_split0.5_train0.7_valid0.15_mask0.1_seed42_20260127_151505/model.pth", help="显式指定 model.pth 路径（可选）")
    parser.add_argument("--config_json", type=str, default="/home/rhl/CSDI-main_test/save/pm25_imputation_split0.5_train0.7_valid0.15_mask0.1_seed42_20260127_151505/config.json", help="显式指定 config.json 路径（可选）")

    # ========== 数据路径 ==========
    parser.add_argument("--ground_path", type=str,
                        default="./data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
                        help="pm25_ground.txt 路径（用于时间索引/可选评估）")
    parser.add_argument("--missing_path", type=str,
                        default="./data/pm25/Code/STMVL/SampleData/pm25_missing.txt",
                        help="pm25_missing.txt 路径（用于补值conditioning）")
    parser.add_argument("--meanstd_path", type=str,
                        default="./data/pm25/pm25_meanstd.pk",
                        help="pm25_meanstd.pk 路径（训练集均值方差）")

    # ========== 基础参数 ==========
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 / cpu")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--split_ratio", type=float, default=0.5, help="时间切分比例：前半用于补值与训练历史")

    # ========== CSDI补值参数 ==========
    parser.add_argument("--imputed_history_path", type=str, default="",
                        help="如果你已经有补值好的前半段历史序列（csv），可直接读它跳过CSDI补值")
    parser.add_argument("--impute_n_samples", type=int, default=50, help="CSDI补值采样次数")
    parser.add_argument("--chunk_len", type=int, default=36, help="CSDI长序列分块长度（建议=训练eval_length）")
    parser.add_argument("--stride", type=int, default=36, help="CSDI分块步长（默认不重叠）")

    # ========== RDE-GPR预测参数 ==========
    parser.add_argument("--trainlength", type=int, default=30, help="RDE-GPR训练窗口长度")
    parser.add_argument("--L", type=int, default=4, help="随机嵌入维度（组合大小）")
    parser.add_argument("--s", type=int, default=200, help="每步抽取的组合数量（越大越慢）")
    parser.add_argument("--steps_ahead", type=int, default=1, help="预测步长（推荐=1用于滚动预测）")
    parser.add_argument("--n_jobs", type=int, default=8, help="并行进程数")
    parser.add_argument("--noise_strength", type=float, default=0.0, help="训练窗口加噪声强度（0表示不加）")
    parser.add_argument("--no_optimize_hyp", action="store_true", help="关闭GPR超参优化（更快但可能更差）")

    # 只预测部分站点（可选）
    parser.add_argument("--target_indices", type=str, default="",
                        help="要预测的维度索引，逗号分隔，例如 '0,1,2'；为空表示预测全部36维")

    args = parser.parse_args()

    set_global_seed(args.seed)

    # ========== 输出文件夹 ==========
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"./save/pm25_forecast_rdegpr_split{args.split_ratio}_seed{args.seed}_{now}/"
    ensure_dir(out_dir)

    # 保存参数
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    print("=" * 80)
    print("PM2.5 后续预测：CSDI补值(前半) + RDE-GPR预测(后半)")
    print("=" * 80)
    print(json.dumps(vars(args), indent=4, ensure_ascii=False))

    # ========== 读数据并时间切分 ==========
    df_full, df_miss = load_pm25_files(args.ground_path, args.missing_path)
    hist_full, hist_miss, fut_full, fut_miss, meta = time_split_df(df_full, df_miss, args.split_ratio)

    print("\n数据切分信息：")
    print(json.dumps(meta, indent=4, ensure_ascii=False))
    if meta["fut_len"] <= 0:
        raise ValueError("后半段长度为0，无法进行预测，请检查 split_ratio。")

    # ========== 加载 mean/std ==========
    import pickle
    with open(args.meanstd_path, "rb") as f:
        train_mean, train_std = pickle.load(f)
    train_mean = np.asarray(train_mean, dtype=np.float32)
    train_std = np.asarray(train_std, dtype=np.float32)
    train_std = np.where(train_std == 0, 1.0, train_std).astype(np.float32)

    # ========== 准备历史序列（已补值） ==========
    if args.imputed_history_path:
        print("\n读取已补值历史序列：", args.imputed_history_path)
        df_hist_imputed = pd.read_csv(args.imputed_history_path, index_col="datetime", parse_dates=True).sort_index()
        # 对齐检查
        if not df_hist_imputed.index.equals(hist_miss.index):
            raise ValueError("imputed_history 的 datetime 索引与前半段切分结果不一致。")
        if df_hist_imputed.shape[1] != hist_miss.shape[1]:
            raise ValueError("imputed_history 的维度与PM2.5站点维度不一致。")
    else:
        # 加载CSDI模型
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

        model, model_cfg, full_cfg = load_csdi_pm25_from_config_json(
            model_path=model_path,
            config_json_path=config_json,
            device=args.device,
        )

        # 对历史段做补值：用 missing 版本做 conditioning
        print("\n开始CSDI补值（仅前半段）...")
        df_hist_imputed = csdi_impute_long_sequence(
            model=model,
            df_hist_missing=hist_miss,
            mean=train_mean,
            std=train_std,
            device=args.device,
            n_samples=args.impute_n_samples,
            chunk_len=args.chunk_len,
            stride=args.stride,
            seed=args.seed,
        )

        # 保存历史补值结果
        hist_out_csv = os.path.join(out_dir, "history_imputed.csv")
        df_hist_imputed.to_csv(hist_out_csv)
        print("历史补值结果已保存：", hist_out_csv)

    # ========== RDE-GPR预测后半段 ==========
    horizon = meta["fut_len"]
    history = df_hist_imputed.values.astype(np.float64)  # (T_hist, D)

    if args.target_indices.strip():
        target_indices = [int(x) for x in args.target_indices.split(",") if x.strip() != ""]
    else:
        target_indices = None  # 全维

    optimize_hyp = (not args.no_optimize_hyp)

    print("\n开始RDE-GPR滚动预测（仅使用历史，不看未来数值）...")
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

    # 组织成DataFrame，索引用未来半段的时间戳（只用时间，不用值）
    df_pred = pd.DataFrame(preds, index=fut_full.index, columns=df_full.columns)
    df_std = pd.DataFrame(stds, index=fut_full.index, columns=df_full.columns)

    pred_csv = os.path.join(out_dir, "future_pred.csv")
    std_csv = os.path.join(out_dir, "future_pred_std.csv")
    df_pred.to_csv(pred_csv)
    df_std.to_csv(std_csv)

    np.save(os.path.join(out_dir, "future_pred.npy"), preds)
    np.save(os.path.join(out_dir, "future_pred_std.npy"), stds)

    print("\n预测结果已保存：")
    print("  ", pred_csv)
    print("  ", std_csv)

    # ========== 可选：用真值评估（不影响预测，仅事后对比） ==========
    # 注意：预测阶段从未读取 fut_full 的数值，只在这里用于评估打印。
    y_true = fut_full.values.astype(np.float64)
    y_pred = preds.astype(np.float64)

    metrics_all = compute_metrics(y_true, y_pred)
    print("\n后半段整体评估（仅用于对比，不参与预测）：")
    print(json.dumps(metrics_all, indent=4, ensure_ascii=False))

    # 每维评估（可选，输出到文件）
    per_dim = []
    for j, col in enumerate(df_full.columns):
        m = compute_metrics(y_true[:, j], y_pred[:, j])
        per_dim.append({"dim": j, "name": col, **m})

    df_per_dim = pd.DataFrame(per_dim)
    df_per_dim.to_csv(os.path.join(out_dir, "metrics_per_dim.csv"), index=False)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"overall": metrics_all, "per_dim_csv": "metrics_per_dim.csv"}, f, indent=4)

    print("\n程序执行完成！输出目录：", out_dir)


if __name__ == "__main__":
    # Windows下多进程需要保护；Linux也建议保留
    mp.set_start_method("spawn", force=True)
    main()
