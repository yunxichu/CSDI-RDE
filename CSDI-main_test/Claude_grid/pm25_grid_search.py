# -*- coding: utf-8 -*-
"""
PM2.5 预测网格搜索脚本 - 对 L 和 trainlength 参数进行网格搜索
- 输入：history_imputed.csv（前半段历史已补值完整）
- 输出：每组参数的预测结果、性能指标、可视化图片，以及汇总报告

网格搜索参数：
- L: 5, 8, 11, 14, 17, 20 (间隔3)
- trainlength: 200, 400, 600, ..., 2000 (间隔200)

示例用法：
python pm25_grid_search.py \
  --imputed_history_path /path/to/history_imputed.csv \
  --ground_path ./data/pm25_ground.txt \
  --split_ratio 0.5 \
  --horizon_days 1 \
  --s 50 --n_jobs 8 \
  --target_indices 0,1,2
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
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

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
    """返回数组的基础统计。"""
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
# GPR
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


def _parallel_predict_one_comb(comb, traindata, target_idx, steps_ahead=1, optimize_hyp=True):
    """
    返回 (pred, std, status_code)
    status_code: 0=OK, 1=EXCEPTION, 2=DEGENERATE
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


def rdegpr_predict_next_for_target(traindata, target_idx, L, s, steps_ahead, pool, rng, optimize_hyp=True):
    """单步预测"""
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


def rdegpr_forecast_multivariate(
    history,
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
):
    """多变量滚动预测"""
    history = np.asarray(history, dtype=np.float64)
    T_hist, D = history.shape

    horizon = int(horizon)
    trainlength = int(trainlength)
    L = int(L)
    s = int(s)
    steps_ahead = int(steps_ahead)
    n_jobs = int(n_jobs)
    noise_strength = float(noise_strength)

    assert_or_raise(trainlength >= 2, "trainlength 必须 >= 2")
    assert_or_raise(steps_ahead >= 1, "steps_ahead 必须 >= 1")
    assert_or_raise(horizon >= 1, "horizon 必须 >= 1")
    assert_or_raise(1 <= L <= D, f"L 必须在 [1, {D}]，当前 L={L}")
    assert_or_raise(s >= 1, "s 必须 >= 1")
    assert_or_raise(T_hist >= trainlength, f"history长度({T_hist}) < trainlength({trainlength})")

    if target_indices is None:
        target_indices = list(range(D))
    else:
        target_indices = list(target_indices)
        assert_or_raise(len(target_indices) > 0, "target_indices 不能为空")

    preds = np.zeros((horizon, D), dtype=np.float64)
    stds = np.zeros((horizon, D), dtype=np.float64)

    base_rng = np.random.default_rng(int(seed))
    pool = mp.Pool(processes=n_jobs)

    try:
        seq = history.copy()

        for step in tqdm(range(horizon), desc=f"Forecasting (L={L}, trainlen={trainlength})", leave=False):
            traindata = seq[-trainlength:].copy()

            if noise_strength > 0:
                traindata = traindata + noise_strength * base_rng.standard_normal(size=traindata.shape)

            next_vec = seq[-1].copy()
            next_std = np.zeros((D,), dtype=np.float64)

            for j in target_indices:
                tj_rng = np.random.default_rng(int(seed + 100000 * step + 1000 * int(j)))
                pred_j, std_j = rdegpr_predict_next_for_target(
                    traindata=traindata,
                    target_idx=int(j),
                    L=L,
                    s=s,
                    steps_ahead=steps_ahead,
                    pool=pool,
                    rng=tj_rng,
                    optimize_hyp=optimize_hyp,
                )

                if np.isnan(pred_j):
                    pred_j = next_vec[int(j)]
                    std_j = 0.0

                next_vec[int(j)] = float(pred_j)
                next_std[int(j)] = float(std_j)

            preds[step] = next_vec
            stds[step] = next_std
            seq = np.vstack([seq, next_vec.reshape(1, -1)])

    finally:
        pool.close()
        pool.join()

    return preds, stds


# =============================================================================
# Metrics / Plots
# =============================================================================
def compute_metrics(y_true, y_pred):
    """计算RMSE和MAE"""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan, "valid_points": 0}
    diff = y_true[mask] - y_pred[mask]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    return {"rmse": rmse, "mae": mae, "valid_points": int(mask.sum())}


def save_plots(out_dir: str, fut_index: pd.DatetimeIndex, y_true: np.ndarray, 
               y_pred: np.ndarray, L: int, trainlength: int, plot_dims=[0]):
    """保存可视化图片"""
    ensure_dir(out_dir)
    
    # 为每个指定维度绘制时序对比图
    for d in plot_dims:
        d = int(d)
        if d >= y_true.shape[1]:
            continue
            
        plt.figure(figsize=(14, 5))
        plt.plot(fut_index, y_true[:, d], label=f"True (dim {d})", linewidth=2)
        plt.plot(fut_index, y_pred[:, d], label=f"Pred (dim {d})", linewidth=2, alpha=0.7)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("PM2.5", fontsize=12)
        plt.title(f"Forecast vs True (L={L}, trainlen={trainlength}, dim={d})", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"forecast_dim{d}.png"), dpi=150)
        plt.close()

    # 绘制每个维度的RMSE柱状图
    rmse_list = []
    for j in range(y_true.shape[1]):
        rmse_list.append(compute_metrics(y_true[:, j], y_pred[:, j])["rmse"])

    plt.figure(figsize=(12, 5))
    plt.bar(np.arange(len(rmse_list)), rmse_list, color='steelblue', alpha=0.7)
    plt.xlabel("Dimension", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.title(f"RMSE per Dimension (L={L}, trainlen={trainlength})", fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rmse_per_dim.png"), dpi=150)
    plt.close()


# =============================================================================
# 网格搜索主函数
# =============================================================================
def run_grid_search(args):
    """执行网格搜索"""
    
    # 设置随机种子
    set_global_seed(args.seed)
    
    # 创建主输出目录
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    main_out_dir = args.out_dir or f"./save/pm25_grid_search_{now}/"
    ensure_dir(main_out_dir)
    
    # 保存参数
    safe_json_dump(vars(args), os.path.join(main_out_dir, "args.json"))
    
    print("=" * 80)
    print("PM2.5 预测网格搜索 (RDE-GPR)")
    print("=" * 80)
    
    # 读取数据
    print("\n[1/4] 读取数据...")
    df_full = pd.read_csv(args.ground_path, index_col="datetime", parse_dates=True).sort_index()
    hist_full, fut_full, meta = time_split_df(df_full, args.split_ratio)
    
    df_hist_imputed = pd.read_csv(args.imputed_history_path, index_col="datetime", parse_dates=True).sort_index()
    
    # 数据验证
    assert_or_raise(df_hist_imputed.index.equals(hist_full.index),
                    "imputed_history 的索引与 ground 切分不一致")
    assert_or_raise(list(df_hist_imputed.columns) == list(hist_full.columns),
                    "imputed_history 的列与 ground 列不一致")
    
    history = df_hist_imputed.values.astype(np.float64)
    assert_or_raise(not np.isnan(history).any(), "history_imputed 包含 NaN")
    assert_or_raise(not np.isinf(history).any(), "history_imputed 包含 Inf")
    
    # 计算horizon
    full_horizon = meta["fut_len"]
    horizon = full_horizon
    if args.horizon_days and args.horizon_days > 0:
        steps_per_day = infer_steps_per_day_from_index(fut_full.index, default_steps_per_day=24)
        horizon = int(round(args.horizon_days * steps_per_day))
    elif args.horizon_steps and args.horizon_steps > 0:
        horizon = int(args.horizon_steps)
    
    horizon = max(1, min(horizon, full_horizon))
    fut_full = fut_full.iloc[:horizon].copy()
    y_true = fut_full.values.astype(np.float64)
    
    # 解析target_indices
    if args.target_indices is not None and args.target_indices.strip() != "":
        target_indices = [int(x) for x in args.target_indices.split(",") if x.strip() != ""]
        assert_or_raise(len(target_indices) > 0, "target_indices 不能为空")
    else:
        target_indices = None
    
    # 定义网格搜索参数
    L_values = list(range(5, 21, 3))  # [5, 8, 11, 14, 17, 20]
    trainlength_values = list(range(200, 2001, 200))  # [200, 400, ..., 2000]
    
    # 过滤掉超过history长度的trainlength
    trainlength_values = [t for t in trainlength_values if t <= len(history)]
    
    print(f"\n[2/4] 网格搜索参数:")
    print(f"  L 候选值: {L_values}")
    print(f"  trainlength 候选值: {trainlength_values}")
    print(f"  总组合数: {len(L_values) * len(trainlength_values)}")
    print(f"  预测horizon: {horizon} steps")
    print(f"  target_indices: {target_indices if target_indices else 'ALL'}")
    
    # 存储所有结果
    all_results = []
    
    # 网格搜索
    print(f"\n[3/4] 开始网格搜索...")
    total_combinations = len(L_values) * len(trainlength_values)
    
    with tqdm(total=total_combinations, desc="Grid Search Progress") as pbar:
        for L in L_values:
            for trainlength in trainlength_values:
                # 跳过不合法的组合
                D = history.shape[1]
                if L > D:
                    pbar.update(1)
                    continue
                
                # 创建此组合的输出目录
                param_dir = os.path.join(main_out_dir, f"L{L}_trainlen{trainlength}")
                ensure_dir(param_dir)
                
                # 记录开始时间
                start_time = time.time()
                
                try:
                    # 执行预测
                    preds, stds = rdegpr_forecast_multivariate(
                        history=history,
                        horizon=horizon,
                        trainlength=trainlength,
                        L=L,
                        s=args.s,
                        steps_ahead=args.steps_ahead,
                        n_jobs=args.n_jobs,
                        seed=args.seed,
                        noise_strength=args.noise_strength,
                        optimize_hyp=(not args.no_optimize_hyp),
                        target_indices=target_indices,
                    )
                    
                    elapsed_time = time.time() - start_time
                    
                    # 保存预测结果
                    df_pred = pd.DataFrame(preds, index=fut_full.index, columns=df_full.columns)
                    df_std = pd.DataFrame(stds, index=fut_full.index, columns=df_full.columns)
                    df_pred.to_csv(os.path.join(param_dir, "future_pred.csv"))
                    df_std.to_csv(os.path.join(param_dir, "future_pred_std.csv"))
                    
                    # 计算指标
                    overall_metrics = compute_metrics(y_true, preds)
                    
                    # 计算每个维度的指标
                    per_dim_metrics = []
                    for j, col in enumerate(df_full.columns):
                        m = compute_metrics(y_true[:, j], preds[:, j])
                        per_dim_metrics.append({
                            "dim": j,
                            "name": str(col),
                            "rmse": m["rmse"],
                            "mae": m["mae"]
                        })
                    
                    pd.DataFrame(per_dim_metrics).to_csv(
                        os.path.join(param_dir, "metrics_per_dim.csv"), index=False
                    )
                    
                    # 保存可视化
                    plot_dims = [0, 1, 2] if len(df_full.columns) >= 3 else [0]
                    save_plots(param_dir, fut_full.index, y_true, preds, L, trainlength, plot_dims)
                    
                    # 记录结果
                    result = {
                        "L": L,
                        "trainlength": trainlength,
                        "rmse": overall_metrics["rmse"],
                        "mae": overall_metrics["mae"],
                        "valid_points": overall_metrics["valid_points"],
                        "elapsed_time": elapsed_time,
                        "status": "success",
                        "output_dir": param_dir
                    }
                    
                    # 保存单组参数的详细结果
                    safe_json_dump(result, os.path.join(param_dir, "result.json"))
                    
                except Exception as e:
                    elapsed_time = time.time() - start_time
                    result = {
                        "L": L,
                        "trainlength": trainlength,
                        "rmse": np.nan,
                        "mae": np.nan,
                        "valid_points": 0,
                        "elapsed_time": elapsed_time,
                        "status": "failed",
                        "error": str(e),
                        "output_dir": param_dir
                    }
                    
                    safe_json_dump(result, os.path.join(param_dir, "result.json"))
                
                all_results.append(result)
                pbar.update(1)
    
    # 生成汇总报告
    print(f"\n[4/4] 生成汇总报告...")
    
    # 保存所有结果
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(main_out_dir, "grid_search_results.csv"), index=False)
    
    # 找出最优参数
    successful_results = results_df[results_df['status'] == 'success'].copy()
    
    if len(successful_results) > 0:
        best_rmse_idx = successful_results['rmse'].idxmin()
        best_mae_idx = successful_results['mae'].idxmin()
        
        best_rmse = successful_results.loc[best_rmse_idx]
        best_mae = successful_results.loc[best_mae_idx]
        
        summary = {
            "total_combinations": total_combinations,
            "successful_runs": len(successful_results),
            "failed_runs": len(results_df) - len(successful_results),
            "best_rmse": {
                "L": int(best_rmse['L']),
                "trainlength": int(best_rmse['trainlength']),
                "rmse": float(best_rmse['rmse']),
                "mae": float(best_rmse['mae']),
                "output_dir": str(best_rmse['output_dir'])
            },
            "best_mae": {
                "L": int(best_mae['L']),
                "trainlength": int(best_mae['trainlength']),
                "rmse": float(best_mae['rmse']),
                "mae": float(best_mae['mae']),
                "output_dir": str(best_mae['output_dir'])
            }
        }
        
        safe_json_dump(summary, os.path.join(main_out_dir, "summary.json"))
        
        # 创建可视化：参数vs性能热力图
        create_heatmaps(successful_results, main_out_dir)
        
        # 打印摘要
        print("\n" + "=" * 80)
        print("网格搜索完成！")
        print("=" * 80)
        print(f"\n成功运行: {len(successful_results)}/{total_combinations}")
        print(f"\n最优 RMSE 参数:")
        print(f"  L = {summary['best_rmse']['L']}")
        print(f"  trainlength = {summary['best_rmse']['trainlength']}")
        print(f"  RMSE = {summary['best_rmse']['rmse']:.4f}")
        print(f"  MAE = {summary['best_rmse']['mae']:.4f}")
        print(f"\n最优 MAE 参数:")
        print(f"  L = {summary['best_mae']['L']}")
        print(f"  trainlength = {summary['best_mae']['trainlength']}")
        print(f"  RMSE = {summary['best_mae']['rmse']:.4f}")
        print(f"  MAE = {summary['best_mae']['mae']:.4f}")
        print(f"\n详细结果保存在: {main_out_dir}")
        
    else:
        print("\n警告: 所有运行都失败了！")
        print(f"详细日志保存在: {main_out_dir}")


def create_heatmaps(results_df, out_dir):
    """创建参数vs性能的热力图"""
    
    # 准备数据
    L_values = sorted(results_df['L'].unique())
    trainlen_values = sorted(results_df['trainlength'].unique())
    
    # 创建RMSE热力图
    rmse_matrix = np.full((len(trainlen_values), len(L_values)), np.nan)
    for i, trainlen in enumerate(trainlen_values):
        for j, L in enumerate(L_values):
            mask = (results_df['L'] == L) & (results_df['trainlength'] == trainlen)
            if mask.any():
                rmse_matrix[i, j] = results_df.loc[mask, 'rmse'].values[0]
    
    # 创建MAE热力图
    mae_matrix = np.full((len(trainlen_values), len(L_values)), np.nan)
    for i, trainlen in enumerate(trainlen_values):
        for j, L in enumerate(L_values):
            mask = (results_df['L'] == L) & (results_df['trainlength'] == trainlen)
            if mask.any():
                mae_matrix[i, j] = results_df.loc[mask, 'mae'].values[0]
    
    # 绘制RMSE热力图
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(rmse_matrix, aspect='auto', cmap='viridis_r', interpolation='nearest')
    
    ax.set_xticks(np.arange(len(L_values)))
    ax.set_yticks(np.arange(len(trainlen_values)))
    ax.set_xticklabels(L_values)
    ax.set_yticklabels(trainlen_values)
    
    ax.set_xlabel('L (embedding dimension)', fontsize=12)
    ax.set_ylabel('trainlength', fontsize=12)
    ax.set_title('RMSE Heatmap: Grid Search Results', fontsize=14, fontweight='bold')
    
    # 添加数值标注
    for i in range(len(trainlen_values)):
        for j in range(len(L_values)):
            if not np.isnan(rmse_matrix[i, j]):
                text = ax.text(j, i, f'{rmse_matrix[i, j]:.2f}',
                             ha="center", va="center", color="white", fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('RMSE', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'heatmap_rmse.png'), dpi=200)
    plt.close()
    
    # 绘制MAE热力图
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(mae_matrix, aspect='auto', cmap='plasma_r', interpolation='nearest')
    
    ax.set_xticks(np.arange(len(L_values)))
    ax.set_yticks(np.arange(len(trainlen_values)))
    ax.set_xticklabels(L_values)
    ax.set_yticklabels(trainlen_values)
    
    ax.set_xlabel('L (embedding dimension)', fontsize=12)
    ax.set_ylabel('trainlength', fontsize=12)
    ax.set_title('MAE Heatmap: Grid Search Results', fontsize=14, fontweight='bold')
    
    # 添加数值标注
    for i in range(len(trainlen_values)):
        for j in range(len(L_values)):
            if not np.isnan(mae_matrix[i, j]):
                text = ax.text(j, i, f'{mae_matrix[i, j]:.2f}',
                             ha="center", va="center", color="white", fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MAE', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'heatmap_mae.png'), dpi=200)
    plt.close()
    
    # 绘制参数趋势图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # RMSE vs L (对每个trainlength)
    for trainlen in trainlen_values[::2]:  # 只显示部分trainlength避免过于拥挤
        mask = results_df['trainlength'] == trainlen
        subset = results_df[mask].sort_values('L')
        ax1.plot(subset['L'], subset['rmse'], marker='o', label=f'trainlen={trainlen}', linewidth=2)
    
    ax1.set_xlabel('L', fontsize=12)
    ax1.set_ylabel('RMSE', fontsize=12)
    ax1.set_title('RMSE vs L (for different trainlength)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # RMSE vs trainlength (对每个L)
    for L in L_values[::2]:  # 只显示部分L避免过于拥挤
        mask = results_df['L'] == L
        subset = results_df[mask].sort_values('trainlength')
        ax2.plot(subset['trainlength'], subset['rmse'], marker='s', label=f'L={L}', linewidth=2)
    
    ax2.set_xlabel('trainlength', fontsize=12)
    ax2.set_ylabel('RMSE', fontsize=12)
    ax2.set_title('RMSE vs trainlength (for different L)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'trend_plots.png'), dpi=200)
    plt.close()


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="PM2.5 预测网格搜索: 对 L 和 trainlength 参数进行网格搜索"
    )

    # 数据路径
    parser.add_argument("--imputed_history_path", type=str, required=True,
                       help="history_imputed.csv（前半段补值结果）")
    parser.add_argument("--ground_path", type=str, required=True,
                       help="pm25_ground.txt（用于时间索引和评估）")
    parser.add_argument("--split_ratio", type=float, default=0.5)

    # 随机种子
    parser.add_argument("--seed", type=int, default=42)

    # horizon 控制
    parser.add_argument("--horizon_days", type=float, default=0.0,
                       help="预测多少天（>0优先）")
    parser.add_argument("--horizon_steps", type=int, default=0,
                       help="预测多少步（>0生效）")

    # RDE-GPR 固定参数（L和trainlength由网格搜索控制）
    parser.add_argument("--s", type=int, default=50, help="每步抽样组合数")
    parser.add_argument("--steps_ahead", type=int, default=1)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--noise_strength", type=float, default=0.0)
    parser.add_argument("--no_optimize_hyp", action="store_true")

    # 其他参数
    parser.add_argument("--target_indices", type=str, default="",
                       help="只预测部分维度：如 '0,1,2'；为空=全维")
    parser.add_argument("--out_dir", type=str, default="",
                       help="输出目录（可选）")

    args = parser.parse_args()

    # 执行网格搜索
    run_grid_search(args)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
