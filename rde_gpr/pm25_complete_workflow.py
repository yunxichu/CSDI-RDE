#!/usr/bin/env python3
"""
PM2.5 完整流程: CSDI补值 + RDE/RDE-Delay预测 + 可视化
分步式实现

流程:
1. CSDI补值：生成history_imputed.csv
2. RDE/RDE-Delay预测
3. 结果可视化

使用方式:
    python pm25_complete_workflow.py --device cuda:0 --run_folder <模型文件夹>
"""

import os
import sys
import json
import time
import random
import argparse
import datetime
import pickle
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

import torch

base_dir = '/home/rhl/Github'
sys.path.insert(0, os.path.join(base_dir, 'rde_gpr', 'csdi'))
sys.path.insert(0, os.path.join(base_dir, 'datasets'))

from dataset_pm25 import get_dataloader
from main_model import CSDI_PM25


def set_global_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_config(run_folder):
    """加载模型配置"""
    config_path = os.path.join(base_dir, 'rde_gpr', 'csdi', 'save', run_folder, 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)


def csdi_impute_testdata(model, test_loader, scaler, mean_scaler, nsample=100, device='cuda:0'):
    """对测试数据进行CSDI补值"""
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="CSDI Imputation"):
            observed_data = batch['observed_data'].to(device)
            observed_mask = batch['observed_mask'].to(device)
            gt_mask = batch['gt_mask'].to(device)
            cond_mask = observed_mask.clone()

            B, T, D = observed_data.shape
            observed_data = observed_data.permute(0, 2, 1)
            observed_mask = observed_mask.permute(0, 2, 1)
            cond_mask = cond_mask.permute(0, 2, 1)
            observed_tp = torch.arange(T, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(B, D, -1).to(device)

            side_info = model.get_side_info(observed_tp, cond_mask)
            samples = model.impute(observed_data, cond_mask, side_info, nsample)
            samples = samples.permute(0, 1, 3, 2)

            samples = samples.cpu().numpy()
            observed_data = observed_data.cpu().numpy()
            gt_mask = gt_mask.cpu().numpy()

            imputed = np.mean(samples, axis=1)
            imputed = imputed * (1 - gt_mask) + observed_data * gt_mask
            results.append(imputed)

    return np.concatenate(results, axis=0)


def rde_predict(traindata, target_idx, L=4, s=50, steps_ahead=1, n_jobs=4):
    """RDE预测 - 空间维度组合嵌入"""
    from gpr_module import GaussianProcessRegressor

    trainlength = len(traindata)
    if trainlength - steps_ahead <= L:
        return np.nan, np.nan

    X = traindata[:trainlength - steps_ahead, :]
    y = traindata[steps_ahead:, target_idx]
    x_test = traindata[trainlength - steps_ahead, :].reshape(1, -1)

    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        return np.nan, np.nan

    combs = list(combinations(range(X.shape[1]), L))
    np.random.shuffle(combs)
    selected = combs[:min(s, len(combs))]

    pool = mp.Pool(processes=n_jobs)
    results = pool.map(
        partial(_rde_single_comb, X=X, y=y, x_test=x_test),
        selected
    )
    pool.close()
    pool.join()

    preds = np.array([r[0] for r in results])
    stds = np.array([r[1] for r in results])
    valid = ~np.isnan(preds) & ~np.isnan(stds)
    vp, vs = preds[valid], stds[valid]

    if len(vp) == 0:
        return np.nan, np.nan

    try:
        kde = gaussian_kde(vp)
        xi = np.linspace(vp.min(), vp.max(), 1000)
        density = kde(xi)
        pred = np.sum(xi * density) / np.sum(density)
    except:
        pred = np.mean(vp)

    return pred, np.std(vp)


def _rde_single_comb(comb, X, y, x_test):
    """单个RDE组合预测"""
    from gpr_module import GaussianProcessRegressor

    try:
        X_c = X[:, list(comb)]
        x_test_c = x_test[:, list(comb)]

        sx = StandardScaler()
        sy = StandardScaler()

        X_all = np.vstack([X_c, x_test_c])
        X_all_s = sx.fit_transform(X_all)
        X_s = X_all_s[:-1]
        x_test_s = X_all_s[-1:]
        y_s = sy.fit_transform(y.reshape(-1, 1)).flatten()

        if np.std(y_s) < 1e-8:
            return np.nan, np.nan

        gp = GaussianProcessRegressor(noise=1e-6)
        gp.fit(X_s, y_s, init_params=(1.0, 0.1, 0.1), optimize=True)
        pred_s, std_s = gp.predict(x_test_s, return_std=True)

        pred = sy.inverse_transform(pred_s.reshape(-1, 1))[0, 0]
        return pred, std_s[0]
    except:
        return np.nan, np.nan


def rde_delay_predict(traindata, target_idx, max_delay=50, M=4, num_samples=100, steps_ahead=1):
    """RDE-Delay预测 - 时间延迟嵌入"""
    from gpr_module import GaussianProcessRegressor

    T, D = traindata.shape
    tau_max = min(max_delay, T // (M + 1))

    pred_list, std_list = [], []

    for _ in range(num_samples):
        try:
            delays = np.random.choice(tau_max, size=M, replace=False) + 1
            dims = np.random.choice(D, size=M, replace=False)

            X, y = [], []
            for t in range(tau_max, T - steps_ahead):
                feat = [traindata[t - d, di] for d, di in zip(delays, dims)]
                if not np.any(np.isnan(feat)):
                    X.append(feat)
                    y.append(traindata[t + steps_ahead, target_idx])

            if len(X) < M + 5:
                continue

            X = np.array(X)
            y = np.array(y)
            x_test = np.array([traindata[T - steps_ahead - d, di] for d, di in zip(delays, dims)]).reshape(1, -1)

            if np.std(y) < 1e-8:
                continue

            sx = StandardScaler()
            sy = StandardScaler()

            X_all = np.vstack([X, x_test])
            X_all_s = sx.fit_transform(X_all)
            X_s = X_all_s[:-1]
            x_test_s = X_all_s[-1:]
            y_s = sy.fit_transform(y.reshape(-1, 1)).flatten()

            gp = GaussianProcessRegressor(noise=1e-6)
            gp.fit(X_s, y_s, init_params=(1.0, 1.0, 0.1), optimize=True)
            pred_s, std_s = gp.predict(x_test_s, return_std=True)

            pred = sy.inverse_transform(pred_s.reshape(-1, 1))[0, 0]
            std = std_s[0] * sy.scale_[0]

            pred_list.append(pred)
            std_list.append(std)
        except:
            continue

    if len(pred_list) == 0:
        return np.nan, np.nan

    preds = np.array(pred_list)
    stds = np.array(std_list)

    inter_var = np.var(preds)
    intra_var = np.mean(stds ** 2)
    final_std = np.sqrt(inter_var + intra_var)

    try:
        kde = gaussian_kde(preds)
        xi = np.linspace(preds.min(), preds.max(), 500)
        density = kde(xi)
        final_pred = np.sum(xi * density) / np.sum(density)
    except:
        final_pred = np.mean(preds)

    return final_pred, final_std


def compute_metrics(y_true, y_pred):
    """计算RMSE和MAE"""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return np.nan, np.nan
    diff = y_true[mask] - y_pred[mask]
    return np.sqrt(np.mean(diff ** 2)), np.mean(np.abs(diff))


def plot_imputation_quality(observed_data, imputed_data, gt_data, output_dir, timestamp, n_show=200):
    """可视化补值质量"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    n = min(n_show, len(gt_data))
    t = np.arange(n)

    ax = axes[0]
    ax.plot(t, gt_data[:n], 'k-', lw=1.5, alpha=0.7, label='Ground Truth')
    ax.plot(t, imputed_data[:n], 'r--', lw=1.5, alpha=0.8, label='Imputed')
    mask = ~np.isnan(observed_data[:n])
    ax.scatter(t[mask], observed_data[:n][mask], s=30, c='blue', zorder=5, label='Observed')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('PM2.5')
    ax.set_title('CSDI Imputation Quality')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    err = imputed_data[:n] - gt_data[:n]
    ax2.bar(t, err, color=['crimson' if e > 0 else 'steelblue' for e in err], alpha=0.7)
    ax2.axhline(0, color='k', lw=1)
    rmse, mae = compute_metrics(gt_data[:n], imputed_data[:n])
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Error')
    ax2.set_title(f'Imputation Error | RMSE={rmse:.4f} | MAE={mae:.4f}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f'pm25_imputation_quality_{timestamp}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return path, rmse, mae


def plot_prediction_comparison(fut_index, y_true, y_pred_rde, y_pred_rde_delay, stds_rde, stds_rde_delay, output_dir, timestamp, target_dim=0):
    """可视化预测对比"""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

    n = min(100, len(y_true))
    t = fut_index[:n]

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, y_true[:n, target_dim], 'k-', lw=2, label='Ground Truth')
    ax1.plot(t, y_pred_rde[:n, target_dim], 'b--', lw=1.5, label='RDE Prediction')
    ax1.fill_between(t,
                     y_pred_rde[:n, target_dim] - 2 * stds_rde[:n, target_dim],
                     y_pred_rde[:n, target_dim] + 2 * stds_rde[:n, target_dim],
                     alpha=0.2, color='blue', label='RDE ±2σ')
    ax1.plot(t, y_pred_rde_delay[:n, target_dim], 'r--', lw=1.5, label='RDE-Delay Prediction')
    ax1.fill_between(t,
                     y_pred_rde_delay[:n, target_dim] - 2 * stds_rde_delay[:n, target_dim],
                     y_pred_rde_delay[:n, target_dim] + 2 * stds_rde_delay[:n, target_dim],
                     alpha=0.2, color='red', label='RDE-Delay ±2σ')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('PM2.5')
    ax1.set_title(f'PM2.5 Prediction Comparison (dim={target_dim})')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    err_rde = y_true[:n, target_dim] - y_pred_rde[:n, target_dim]
    err_rde_delay = y_true[:n, target_dim] - y_pred_rde_delay[:n, target_dim]
    ax2.bar(t - 0.2, err_rde, width=0.4, color='blue', alpha=0.7, label='RDE Error')
    ax2.bar(t + 0.2, err_rde_delay, width=0.4, color='red', alpha=0.7, label='RDE-Delay Error')
    ax2.axhline(0, color='k', lw=1)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Prediction Error')
    ax2.set_title('Prediction Errors')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    rmse_rde, _ = compute_metrics(y_true[:, target_dim], y_pred_rde[:, target_dim])
    rmse_rde_delay, _ = compute_metrics(y_true[:, target_dim], y_pred_rde_delay[:, target_dim])

    ax3 = fig.add_subplot(gs[1, 1])
    methods = ['RDE', 'RDE-Delay']
    rmses = [rmse_rde, rmse_rde_delay]
    colors = ['#4C72B0', '#DD8452']
    bars = ax3.bar(methods, rmses, color=colors, alpha=0.75)
    for bar, v in zip(bars, rmses):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                  f'{v:.4f}', ha='center', va='bottom', fontsize=10)
    ax3.set_ylabel('RMSE')
    ax3.set_title('RMSE Comparison')
    ax3.grid(True, alpha=0.3, axis='y')

    n_dims = y_true.shape[1]
    rmses_rde = [compute_metrics(y_true[:, j], y_pred_rde[:, j])[0] for j in range(n_dims)]
    rmses_rde_delay = [compute_metrics(y_true[:, j], y_pred_rde_delay[:, j])[0] for j in range(n_dims)]

    ax4 = fig.add_subplot(gs[2, :])
    x = np.arange(n_dims)
    width = 0.35
    ax4.bar(x - width/2, rmses_rde, width, color='blue', alpha=0.7, label='RDE')
    ax4.bar(x + width/2, rmses_rde_delay, width, color='red', alpha=0.7, label='RDE-Delay')
    ax4.set_xlabel('Dimension')
    ax4.set_ylabel('RMSE')
    ax4.set_title('RMSE per Dimension')
    ax4.set_xticks(x)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, f'pm25_prediction_comparison_{timestamp}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return path


def main():
    parser = argparse.ArgumentParser(description="PM2.5 Complete Workflow")
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--run_folder', type=str, required=True, help='Model folder name (e.g., pm25_validationindex0_20260320_123456)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--nsample', type=int, default=100, help='CSDI imputation samples')
    parser.add_argument('--trainlength', type=int, default=4000, help='RDE training length')
    parser.add_argument('--L', type=int, default=4, help='RDE embedding dimension')
    parser.add_argument('--s', type=int, default=50, help='RDE number of samples')
    parser.add_argument('--max_delay', type=int, default=50, help='RDE-Delay max delay')
    parser.add_argument('--M', type=int, default=4, help='RDE-Delay embedding dimension')
    parser.add_argument('--num_samples', type=int, default=100, help='RDE-Delay number of samples')
    parser.add_argument('--target_dim', type=int, default=0, help='Target dimension for visualization')
    parser.add_argument('--n_jobs', type=int, default=8, help='Number of parallel jobs')
    parser.add_argument('--skip_imputation', action='store_true', help='Skip CSDI imputation')
    parser.add_argument('--skip_prediction', action='store_true', help='Skip RDE prediction')

    args = parser.parse_args()
    set_global_seed(args.seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, 'rde_gpr', 'results', f'pm25_{timestamp}')
    ensure_dir(out_dir)

    print("=" * 80)
    print("PM2.5: CSDI Imputation + RDE/RDE-Delay Prediction")
    print("=" * 80)

    model_path = os.path.join(base_dir, 'rde_gpr', 'csdi', 'save', args.run_folder, 'model.pth')
    config_path = os.path.join(base_dir, 'rde_gpr', 'csdi', 'save', args.run_folder, 'config.json')

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return

    config = load_config(args.run_folder)

    # 1. Load data and model
    print("\n[1] Loading data and model...")
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        config['train']['batch_size'], device=args.device, validindex=config['model'].get('validationindex', 0)
    )

    model = CSDI_PM25(config, args.device).to(args.device)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.eval()

    print(f"  Test batches: {len(test_loader)}")

    # 2. CSDI Imputation
    imputed_data = None
    imp_rmse, imp_mae = np.nan, np.nan
    imp_fig_path = None

    if not args.skip_imputation:
        print("\n[2] CSDI Imputation...")
        imputed_data = csdi_impute_testdata(model, test_loader, scaler, mean_scaler, args.nsample, args.device)
        print(f"  Imputed data shape: {imputed_data.shape}")

        # Save imputed data
        np.savetxt(os.path.join(out_dir, 'imputed_data.csv'), imputed_data, delimiter=',')
        print(f"  Saved to: {os.path.join(out_dir, 'imputed_data.csv')}")

        # Get ground truth for evaluation
        batch = next(iter(test_loader))
        gt_data = batch['gt_data'].numpy()
        observed_data = batch['observed_data'].numpy()

        # Plot imputation quality
        imp_fig_path, imp_rmse, imp_mae = plot_imputation_quality(
            observed_data[:, :, 0],
            imputed_data[:, 0],
            gt_data[:, :, 0],
            out_dir, timestamp
        )
        print(f"  Imputation RMSE: {imp_rmse:.4f}, MAE: {imp_mae:.4f}")

    # 3. RDE/RDE-Delay Prediction
    if not args.skip_prediction:
        print("\n[3] RDE/RDE-Delay Prediction...")

        # Prepare data
        if imputed_data is None:
            batch = next(iter(test_loader))
            imputed_data = batch['observed_data'].numpy()[:, :, 0]
            gt_data = batch['gt_data'].numpy()

        horizon = min(100, len(imputed_data) - args.trainlength)
        history = imputed_data[:args.trainlength]
        future_truth = gt_data[:horizon, :, 0]

        print(f"  History length: {len(history)}, Horizon: {horizon}")

        # RDE Prediction
        print("  Running RDE prediction...")
        preds_rde = np.zeros((horizon, imputed_data.shape[1]))
        stds_rde = np.zeros((horizon, imputed_data.shape[1]))

        for step in tqdm(range(horizon), desc="RDE"):
            traindata = imputed_data[step:step + args.trainlength]
            for dim in range(imputed_data.shape[1]):
                pred, std = rde_predict(traindata, dim, L=args.L, s=args.s, n_jobs=args.n_jobs)
                preds_rde[step, dim] = pred if not np.isnan(pred) else traindata[-1, dim]
                stds_rde[step, dim] = std if not np.isnan(std) else 0.0

        # RDE-Delay Prediction
        print("  Running RDE-Delay prediction...")
        preds_rde_delay = np.zeros((horizon, imputed_data.shape[1]))
        stds_rde_delay = np.zeros((horizon, imputed_data.shape[1]))

        for step in tqdm(range(horizon), desc="RDE-Delay"):
            traindata = imputed_data[step:step + args.trainlength]
            for dim in range(imputed_data.shape[1]):
                pred, std = rde_delay_predict(traindata, dim, max_delay=args.max_delay, M=args.M, num_samples=args.num_samples)
                preds_rde_delay[step, dim] = pred if not np.isnan(pred) else traindata[-1, dim]
                stds_rde_delay[step, dim] = std if not np.isnan(std) else 0.0

        # Save predictions
        np.savetxt(os.path.join(out_dir, 'predictions_rde.csv'), preds_rde, delimiter=',')
        np.savetxt(os.path.join(out_dir, 'predictions_rde_delay.csv'), preds_rde_delay, delimiter=',')
        np.savetxt(os.path.join(out_dir, 'stds_rde.csv'), stds_rde, delimiter=',')
        np.savetxt(os.path.join(out_dir, 'stds_rde_delay.csv'), stds_rde_delay, delimiter=',')

        # Compute metrics
        overall_rde_rmse, overall_rde_mae = compute_metrics(future_truth, preds_rde)
        overall_rde_delay_rmse, overall_rde_delay_mae = compute_metrics(future_truth, preds_rde_delay)

        print(f"\n  RDE RMSE: {overall_rde_rmse:.4f}, MAE: {overall_rde_mae:.4f}")
        print(f"  RDE-Delay RMSE: {overall_rde_delay_rmse:.4f}, MAE: {overall_rde_delay_mae:.4f}")

        # Plot
        fut_index = pd.date_range(start='2020-01-01', periods=horizon, freq='h')
        pred_fig_path = plot_prediction_comparison(
            fut_index, future_truth, preds_rde, preds_rde_delay,
            stds_rde, stds_rde_delay, out_dir, timestamp, args.target_dim
        )

        # Save summary
        summary = {
            'timestamp': timestamp,
            'run_folder': args.run_folder,
            'imputation': {'rmse': float(imp_rmse), 'mae': float(imp_mae)},
            'rde': {'rmse': float(overall_rde_rmse), 'mae': float(overall_rde_mae)},
            'rde_delay': {'rmse': float(overall_rde_delay_rmse), 'mae': float(overall_rde_delay_mae)},
            'params': {
                'trainlength': args.trainlength,
                'L': args.L,
                's': args.s,
                'max_delay': args.max_delay,
                'M': args.M,
                'num_samples': args.num_samples
            }
        }

        with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

        print(f"\n  Results saved to: {out_dir}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    from itertools import combinations
    main()