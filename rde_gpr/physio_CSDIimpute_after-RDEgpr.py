# -*- coding: utf-8 -*-
"""
Physio 后续预测脚本（样本内预测）
- 对每个样本：用前36个时间步预测后12个时间步
- 输出：future_pred.npy / future_pred_std.npy / metrics.json / plots

示例：
python rde_gpr/physio_CSDIimpute_after-RDEgpr.py \
  --imputed_history_path ./save/physio_history_imputed_split0.5_seed1_XXXXXXXX/history_imputed.npy \
  --history_timesteps 36 \
  --horizon_timesteps 12 \
  --L 4 --s 50 --trainlength 20 --n_jobs 2 \
  --target_indices 0,1,2
"""

import os
import json
import time
import random
import argparse
import datetime
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


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_json_dump(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False, default=str)


def rde_feature_transform(X, L):
    n, d = X.shape
    W = np.random.randn(d, L) / np.sqrt(L)
    return X @ W


def rde_gpr_fit_predict(train_X, train_y, test_X, alpha=1.0, beta=1.0):
    n = train_X.shape[0]
    K = np.dot(train_X, train_X.T) + alpha * np.eye(n)
    try:
        L = cholesky(K, lower=True)
    except:
        return np.nan, np.nan
    
    alpha_vec = solve_triangular(L, train_y, lower=True)
    alpha_vec = solve_triangular(L.T, alpha_vec, lower=False)
    
    k_star = np.dot(test_X, train_X.T)
    pred = np.dot(k_star, alpha_vec)
    
    v = solve_triangular(L, k_star.T, lower=True)
    var = np.dot(test_X, test_X.T) - np.dot(v.T, v) + beta
    
    return pred.item() if pred.size == 1 else pred, var.item() if var.size == 1 else var


def rde_predict_ensemble(train_data, target_dim, L=4, s=50, n_jobs=1):
    n_samples, n_features = train_data.shape
    
    predictions = []
    variances = []
    
    for i in range(s):
        W = np.random.randn(n_features, L) / np.sqrt(L)
        train_X = np.dot(train_data, W)
        
        train_y = train_data[1:, target_dim]
        train_X = train_X[:-1]
        
        test_X = train_X[-1:].reshape(1, -1)
        
        pred, var = rde_gpr_fit_predict(train_X, train_y, test_X)
        
        if not np.isnan(pred):
            predictions.append(pred)
            variances.append(var)
    
    if len(predictions) == 0:
        return np.nan, np.nan
    
    predictions = np.array(predictions)
    variances = np.array(variances)
    
    final_pred = np.mean(predictions)
    final_var = np.mean(variances) + np.var(predictions)
    
    return final_pred, np.sqrt(final_var)


def main():
    parser = argparse.ArgumentParser(description="Physio Prediction (RDE-GPR, sample-wise)")

    parser.add_argument("--imputed_history_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="./data/physio/")
    parser.add_argument("--missing_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--history_timesteps", type=int, default=36, help="用于预测的历史时间步数")
    parser.add_argument("--horizon_timesteps", type=int, default=12, help="预测的未来时间步数")

    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--s", type=int, default=50)
    parser.add_argument("--trainlength", type=int, default=20)
    parser.add_argument("--n_jobs", type=int, default=1)

    parser.add_argument("--target_indices", type=str, default="0,1,2")
    parser.add_argument("--max_samples", type=int, default=100, help="最多处理多少个样本")
    parser.add_argument("--use_ground_truth_sliding", action="store_true", help="使用真实值更新滑窗")
    parser.add_argument("--missing_positions_path", type=str, default="", help="缺失位置文件路径")

    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    set_global_seed(args.seed)

    print("=" * 80)
    print("Physio 样本内预测（RDE-GPR）")
    print("=" * 80)

    hist_imputed = np.load(args.imputed_history_path)
    print(f"Loaded data shape: {hist_imputed.shape}")

    n_samples, n_timesteps, n_features = hist_imputed.shape
    
    assert args.history_timesteps + args.horizon_timesteps <= n_timesteps, \
        f"history_timesteps({args.history_timesteps}) + horizon_timesteps({args.horizon_timesteps}) > total_timesteps({n_timesteps})"

    meta = {
        "total_samples": n_samples,
        "n_timesteps": n_timesteps,
        "n_features": n_features,
        "history_timesteps": args.history_timesteps,
        "horizon_timesteps": args.horizon_timesteps,
        "max_samples": args.max_samples,
    }
    print(json.dumps(meta, indent=4, ensure_ascii=False))

    target_indices = [int(x) for x in args.target_indices.split(",")]
    print(f"target_indices = {target_indices}")

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"./save/physio_rdegpr_hist{args.history_timesteps}_hor{args.horizon_timesteps}_seed{args.seed}_{now}/"
    ensure_dir(out_dir)

    n_test_samples = min(args.max_samples, n_samples)
    
    all_preds = []
    all_stds = []
    all_truths = []
    all_histories = []  # 保存历史数据

    print(f"\nProcessing {n_test_samples} samples...")
    
    for sample_idx in tqdm(range(n_test_samples), desc="Sample-wise prediction"):
        sample_data = hist_imputed[sample_idx]
        
        history = sample_data[:args.history_timesteps].copy()
        future_truth = sample_data[args.history_timesteps:args.history_timesteps + args.horizon_timesteps]
        
        sample_pred = np.zeros((args.horizon_timesteps, n_features))
        sample_std = np.zeros((args.horizon_timesteps, n_features))
        
        for step in range(args.horizon_timesteps):
            for dim in target_indices:
                train_data = history[-args.trainlength:]
                
                pred, std = rde_predict_ensemble(
                    train_data, dim, L=args.L, s=args.s, n_jobs=args.n_jobs
                )
                
                sample_pred[step, dim] = pred if not np.isnan(pred) else history[-1, dim]
                sample_std[step, dim] = std if not np.isnan(std) else 1.0
            
            # 滑窗更新
            if args.use_ground_truth_sliding:
                # 使用真实值更新滑窗
                history = np.vstack([history, future_truth[step:step+1]])
            else:
                # 使用预测值更新滑窗
                history = np.vstack([history, sample_pred[step:step+1]])
        
        all_preds.append(sample_pred)
        all_stds.append(sample_std)
        all_truths.append(future_truth)
        all_histories.append(sample_data[:args.history_timesteps])  # 保存历史

    all_preds = np.array(all_preds)
    all_stds = np.array(all_stds)
    all_truths = np.array(all_truths)
    all_histories = np.array(all_histories)

    np.save(os.path.join(out_dir, "future_pred.npy"), all_preds)
    np.save(os.path.join(out_dir, "future_pred_std.npy"), all_stds)
    np.save(os.path.join(out_dir, "future_truth.npy"), all_truths)

    metrics = {}
    for dim in target_indices:
        y_true = all_truths[:, :, dim].flatten()
        y_pred = all_preds[:, :, dim].flatten()
        
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if mask.sum() > 0:
            rmse = np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))
            mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            metrics[f"dim_{dim}"] = {"rmse": float(rmse), "mae": float(mae)}
    
    safe_json_dump(metrics, os.path.join(out_dir, "metrics.json"))
    print(f"\nMetrics: {metrics}")

    # 加载缺失位置信息
    missing_positions_path = args.missing_positions_path
    if not missing_positions_path:
        base_dir = os.path.dirname(args.imputed_history_path)
        missing_positions_path = os.path.join(base_dir, 'history_missing_positions.csv')
    
    missing_df = None
    if os.path.exists(missing_positions_path):
        missing_df = pd.read_csv(missing_positions_path)
        print(f"\nLoaded missing positions from: {missing_positions_path}")
        print(f"Total missing positions: {len(missing_df)}")
    else:
        print(f"\nWarning: Missing positions file not found: {missing_positions_path}")

    # 绘制单个样本的补值对比图
    sample_idx_to_plot = 0  # 选择第一个样本
    sample_data = hist_imputed[sample_idx_to_plot]
    
    for dim in target_indices[:3]:
        plt.figure(figsize=(16, 6))
        
        # 获取当前维度的缺失位置
        if missing_df is not None:
            dim_missing = missing_df[
                (missing_df['feature'] == dim) & 
                (missing_df['sample_idx'] == sample_idx_to_plot)
            ]
            missing_time_steps = set(dim_missing['time_step'].values)
        else:
            missing_time_steps = set()
        
        # 绘制完整的时间序列
        time_steps = np.arange(48)
        observed_values = sample_data[:, dim]
        
        # 分离观测值和补值
        observed_mask = np.ones(48, dtype=bool)
        for t in missing_time_steps:
            if t < 48:
                observed_mask[t] = False
        
        # 绘制观测值（蓝色点）
        plt.scatter(time_steps[observed_mask], observed_values[observed_mask], 
                   s=50, c='blue', label='Observed', zorder=5, alpha=0.7)
        
        # 绘制补值点（橙色点）
        if len(missing_time_steps) > 0:
            missing_ts = sorted([t for t in missing_time_steps if t < 48])
            plt.scatter(missing_ts, observed_values[missing_ts], 
                       s=80, c='orange', marker='s', label='Imputed', zorder=6, edgecolors='red', linewidths=1.5)
        
        # 绘制连接线
        plt.plot(time_steps, observed_values, 'gray', alpha=0.3, linewidth=1)
        
        # 标注预测区域
        plt.axvline(x=args.history_timesteps, color='green', linestyle=':', linewidth=2, label='Prediction Start')
        plt.axvspan(args.history_timesteps, 48, alpha=0.1, color='green', label='Prediction Region')
        
        plt.xlabel('Time Step')
        plt.ylabel(f'Feature {dim}')
        plt.title(f'Sample {sample_idx_to_plot} - Feature {dim}: Observed vs Imputed Points')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"plot_dim{dim}_sample{sample_idx_to_plot}_imputation.png"), dpi=150)
        plt.close()
        
        print(f"Dim {dim}: {len(missing_time_steps)} imputed positions")

    # 绘制平均值的预测对比图（包含补值点和原始点）
    sample_idx_to_plot = 0  # 选择第一个样本作为代表
    
    for dim in target_indices[:3]:
        plt.figure(figsize=(16, 6))
        
        mean_truth_full = np.zeros(48)
        mean_pred_full = np.zeros(48)
        mean_std_full = np.zeros(48)
        
        mean_truth_full[:args.history_timesteps] = all_histories[:, :, dim].mean(axis=0)
        mean_pred_full[:args.history_timesteps] = all_histories[:, :, dim].mean(axis=0)
        
        mean_truth_full[args.history_timesteps:] = all_truths[:, :, dim].mean(axis=0)
        mean_pred_full[args.history_timesteps:] = all_preds[:, :, dim].mean(axis=0)
        mean_std_full[args.history_timesteps:] = all_stds[:, :, dim].mean(axis=0)
        
        # 绘制真实值曲线
        plt.plot(range(48), mean_truth_full, 'b-', label='Ground Truth (mean)', linewidth=2, alpha=0.7)
        
        # 绘制预测值曲线
        plt.plot(range(args.history_timesteps, 48), mean_pred_full[args.history_timesteps:], 
                'r--', label='Prediction', linewidth=2)
        
        # 绘制置信区间
        plt.fill_between(
            range(args.history_timesteps, 48),
            mean_pred_full[args.history_timesteps:] - 2 * mean_std_full[args.history_timesteps:],
            mean_pred_full[args.history_timesteps:] + 2 * mean_std_full[args.history_timesteps:],
            alpha=0.3, color='red', label='95% CI'
        )
        
        # 叠加显示单个样本的观测点和补值点
        if missing_df is not None:
            dim_missing = missing_df[
                (missing_df['feature'] == dim) & 
                (missing_df['sample_idx'] == sample_idx_to_plot)
            ]
            missing_time_steps = set(dim_missing['time_step'].values)
        else:
            missing_time_steps = set()
        
        sample_data = hist_imputed[sample_idx_to_plot]
        observed_values = sample_data[:, dim]
        
        # 分离观测值和补值
        observed_mask = np.ones(48, dtype=bool)
        for t in missing_time_steps:
            if t < 48:
                observed_mask[t] = False
        
        # 绘制观测点（蓝色小点）
        plt.scatter(np.arange(48)[observed_mask], observed_values[observed_mask], 
                   s=30, c='blue', alpha=0.5, label='Observed (sample 0)', zorder=4)
        
        # 绘制补值点（橙色方块）
        if len(missing_time_steps) > 0:
            missing_ts = sorted([t for t in missing_time_steps if t < 48])
            plt.scatter(missing_ts, observed_values[missing_ts], 
                       s=60, c='orange', marker='s', label='Imputed (sample 0)', 
                       zorder=5, edgecolors='red', linewidths=1.5, alpha=0.8)
        
        # 标注预测区域
        plt.axvline(x=args.history_timesteps, color='green', linestyle=':', linewidth=2, label='Prediction Start')
        plt.axvspan(args.history_timesteps, 48, alpha=0.05, color='green')
        
        plt.xlabel('Time Step')
        plt.ylabel(f'Feature {dim}')
        plt.title(f'Physio Prediction - Feature {dim} (mean curve + sample {sample_idx_to_plot} points)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"plot_dim{dim}_full.png"), dpi=150)
        plt.close()

    meta_out = {
        **meta,
        "future_pred_npy": "future_pred.npy",
        "future_pred_std_npy": "future_pred_std.npy",
        "future_truth_npy": "future_truth.npy",
        "target_indices": target_indices,
        "L": args.L,
        "s": args.s,
        "trainlength": args.trainlength,
        "metrics": metrics,
    }
    safe_json_dump(meta_out, os.path.join(out_dir, "prediction_meta.json"))

    print(f"\n完成！输出目录： {out_dir}")
    print(f"future_pred.npy shape: {all_preds.shape}")
    print(f"future_truth.npy shape: {all_truths.shape}")


if __name__ == "__main__":
    main()
