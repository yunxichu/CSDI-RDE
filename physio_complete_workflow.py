#!/usr/bin/env python3
"""
Physio 完整流程: CSDI补值 + RDE/RDE-Delay预测 + 可视化

流程:
1. CSDI补值
2. RDE/RDE-Delay预测
3. 结果可视化

使用方式:
    python physio_complete_workflow.py --device cuda:0 --run_folder <模型文件夹>
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
sys.path.insert(0, os.path.join(base_dir, 'csdi'))
sys.path.insert(0, os.path.join(base_dir, 'datasets'))

from dataset_physio import get_dataloader, Physio_Dataset
from main_model import CSDI_Physio


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
    config_path = os.path.join(base_dir, 'csdi', 'save', run_folder, 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)


def csdi_impute_testdata(model, test_loader, nsample=100, device='cuda:0', scaler=None, mean_scaler=None, max_batches=1):
    model.eval()
    results = []
    batch_count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="CSDI Imputation"):
            if batch_count >= max_batches:
                break
            batch_count += 1
            
            observed_data = batch['observed_data'].to(device).float()
            observed_mask = batch['observed_mask'].to(device).float()
            gt_mask = batch['gt_mask'].to(device).float()
            cond_mask = observed_mask.clone()

            B, T, D = observed_data.shape
            observed_data = observed_data.permute(0, 2, 1)
            observed_mask = observed_mask.permute(0, 2, 1)
            cond_mask = cond_mask.permute(0, 2, 1)
            observed_tp = torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(B, -1).to(device)

            side_info = model.get_side_info(observed_tp, cond_mask)
            samples = model.impute(observed_data, cond_mask, side_info, nsample)
            samples = samples.permute(0, 1, 3, 2)

            samples = samples.cpu().numpy()
            observed_data_np = observed_data.cpu().numpy()
            gt_mask_np = gt_mask.cpu().numpy()
            
            # samples shape after permute: (B, nsample, T, D)
            # mean axis=1 -> (B, T, D)
            # gt_mask_np shape: (B, T, D)
            # observed_data_np shape: (B, D, T) -> need to transpose to (B, T, D)
            observed_data_np = observed_data_np.transpose(0, 2, 1)

            imputed = np.mean(samples, axis=1)  # (B, T, D)
            imputed = imputed * (1 - gt_mask_np) + observed_data_np * gt_mask_np
            
            # imputed shape: (B, T, D) - already correct shape
            
            if scaler is not None and mean_scaler is not None:
                scaler_np = scaler.cpu().numpy() if isinstance(scaler, torch.Tensor) else scaler
                mean_scaler_np = mean_scaler.cpu().numpy() if isinstance(mean_scaler, torch.Tensor) else mean_scaler
                imputed = imputed * scaler_np + mean_scaler_np
            
            results.append(imputed)

    return np.concatenate(results, axis=0)


def rde_predict(traindata, target_idx, L=4, s=50, steps_ahead=1, n_jobs=4):
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


def rde_delay_predict(traindata, target_idx, max_delay=10, M=4, num_samples=100, steps_ahead=1):
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
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return np.nan, np.nan
    diff = y_true[mask] - y_pred[mask]
    return np.sqrt(np.mean(diff ** 2)), np.mean(np.abs(diff))


def plot_imputation_quality(observed_data, imputed_data, gt_data, output_dir, timestamp, n_show=48):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    n = min(n_show, len(gt_data))
    t = np.arange(n)

    ax = axes[0]
    ax.plot(t, gt_data[:n], 'k-', lw=1.5, alpha=0.7, label='Ground Truth')
    ax.plot(t, imputed_data[:n], 'r--', lw=1.5, alpha=0.8, label='Imputed')
    mask = ~np.isnan(observed_data[:n])
    ax.scatter(t[mask], observed_data[:n][mask], s=30, c='blue', zorder=5, label='Observed')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Physio Value')
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
    path = os.path.join(output_dir, f'physio_imputation_quality_{timestamp}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return path, rmse, mae


def plot_prediction_comparison(fut_index, y_true, y_pred_rde, y_pred_rde_delay, stds_rde, stds_rde_delay, output_dir, timestamp, target_dim=0):
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

    n = min(48, len(y_true))
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
    ax1.set_ylabel('Physio Value')
    ax1.set_title(f'Physio Prediction Comparison (dim={target_dim})')
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
    path = os.path.join(output_dir, f'physio_prediction_comparison_{timestamp}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return path


def main():
    parser = argparse.ArgumentParser(description="Physio Complete Workflow")
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--run_folder', type=str, required=True, help='Model folder name')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--nsample', type=int, default=100, help='CSDI imputation samples')
    parser.add_argument('--trainlength', type=int, default=40, help='RDE training length')
    parser.add_argument('--L', type=int, default=4, help='RDE embedding dimension')
    parser.add_argument('--s', type=int, default=50, help='RDE number of samples')
    parser.add_argument('--max_delay', type=int, default=10, help='RDE-Delay max delay')
    parser.add_argument('--M', type=int, default=4, help='RDE-Delay embedding dimension')
    parser.add_argument('--num_samples', type=int, default=100, help='RDE-Delay number of samples')
    parser.add_argument('--target_dim', type=int, default=0, help='Target dimension')
    parser.add_argument('--n_jobs', type=int, default=8, help='Number of parallel jobs')
    parser.add_argument('--skip_imputation', action='store_true', help='Skip CSDI imputation')
    parser.add_argument('--skip_prediction', action='store_true', help='Skip RDE prediction')

    args = parser.parse_args()
    set_global_seed(args.seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, 'csdi', 'results', f'physio_{timestamp}')
    ensure_dir(out_dir)

    print("=" * 80)
    print("Physio: CSDI Imputation + RDE/RDE-Delay Prediction")
    print("=" * 80)

    model_path = os.path.join(base_dir, 'csdi', 'save', args.run_folder, 'model.pth')
    config_path = os.path.join(base_dir, 'csdi', 'save', args.run_folder, 'config.json')

    config = load_config(args.run_folder)

    print("\n[1] Loading data and model...")
    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        seed=args.seed,
        nfold=config['model'].get('nfold', 0),
        batch_size=config['train']['batch_size'],
        missing_ratio=config['model'].get('test_missing_ratio', 0.1),
        device=args.device
    )

    model = CSDI_Physio(config, args.device).to(args.device)
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.eval()

    print(f"  Test batches: {len(test_loader)}")

    imputed_data = None
    imp_rmse, imp_mae = np.nan, np.nan

    if not args.skip_imputation:
        print("\n[2] CSDI Imputation...")
        imputed_data = csdi_impute_testdata(model, test_loader, args.nsample, args.device, scaler, mean_scaler)
        print(f"  Imputed data shape: {imputed_data.shape}")

        np.savetxt(os.path.join(out_dir, 'imputed_data.csv'), imputed_data.reshape(-1), delimiter=',')
        print(f"  Saved to: {os.path.join(out_dir, 'imputed_data.csv')}")

        batch = next(iter(test_loader))
        gt_data = batch['gt_data'].numpy()
        observed_data = batch['observed_data'].numpy()
        
        # 反标准化gt_data和observed_data以便正确比较
        if scaler is not None and mean_scaler is not None:
            scaler_np = scaler.cpu().numpy() if isinstance(scaler, torch.Tensor) else scaler
            mean_scaler_np = mean_scaler.cpu().numpy() if isinstance(mean_scaler, torch.Tensor) else mean_scaler
            gt_data = gt_data * scaler_np + mean_scaler_np
            observed_data = observed_data * scaler_np + mean_scaler_np

        # 只显示第一个样本的第一个维度
        imp_fig_path, imp_rmse, imp_mae = plot_imputation_quality(
            observed_data[0, :, 0],  # (T,) 形状
            imputed_data[0, :, 0],   # (T,) 形状
            gt_data[0, :, 0],        # (T,) 形状
            out_dir, timestamp
        )
        print(f"  Imputation RMSE: {imp_rmse:.4f}, MAE: {imp_mae:.4f}")

    if not args.skip_prediction:
        print("\n[3] RDE/RDE-Delay Prediction...")

        if imputed_data is None:
            batch = next(iter(test_loader))
            imputed_data = batch['observed_data'].numpy()
            gt_data = batch['gt_data'].numpy()
            # 反标准化
            if scaler is not None and mean_scaler is not None:
                scaler_np = scaler.cpu().numpy() if isinstance(scaler, torch.Tensor) else scaler
                mean_scaler_np = mean_scaler.cpu().numpy() if isinstance(mean_scaler, torch.Tensor) else mean_scaler
                imputed_data = imputed_data * scaler_np + mean_scaler_np
                gt_data = gt_data * scaler_np + mean_scaler_np

        # 确保horizon为正数
        effective_trainlength = min(args.trainlength, len(imputed_data) - 1)
        horizon = min(48, len(imputed_data) - effective_trainlength)
        if horizon <= 0:
            print(f"  Warning: horizon={horizon} <= 0, skipping prediction (need more data or smaller trainlength)")
            print("  Tip: Use --trainlength 20 or less with --max_batches 1")
            args.skip_prediction = True
        
        if not args.skip_prediction:
            history = imputed_data[:, :effective_trainlength, :]
            future_truth = gt_data[:, :horizon, :]

            print(f"  History shape: {history.shape}, Horizon: {horizon}")

            all_preds_rde = []
            all_preds_rde_delay = []

            for sample_idx in range(imputed_data.shape[0]):
                print(f"  Sample {sample_idx + 1}/{imputed_data.shape[0]}...")

                sample_data = imputed_data[sample_idx]
                sample_future = future_truth[sample_idx]

                preds_rde = np.zeros((horizon, sample_data.shape[1]))
                preds_rde_delay = np.zeros((horizon, sample_data.shape[1]))

                for step in range(horizon):
                    traindata = sample_data[step:step + effective_trainlength, :]
                    for dim in range(sample_data.shape[1]):
                        pred, _ = rde_predict(traindata, dim, L=args.L, s=args.s, n_jobs=args.n_jobs)
                        preds_rde[step, dim] = pred if not np.isnan(pred) else traindata[-1, dim]

                        pred, _ = rde_delay_predict(traindata, dim, max_delay=args.max_delay, M=args.M, num_samples=args.num_samples)
                        preds_rde_delay[step, dim] = pred if not np.isnan(pred) else traindata[-1, dim]

                all_preds_rde.append(preds_rde)
                all_preds_rde_delay.append(preds_rde_delay)

            preds_rde = np.mean(all_preds_rde, axis=0)
            preds_rde_delay = np.mean(all_preds_rde_delay, axis=0)

            np.savetxt(os.path.join(out_dir, 'predictions_rde.csv'), preds_rde, delimiter=',')
            np.savetxt(os.path.join(out_dir, 'predictions_rde_delay.csv'), preds_rde_delay, delimiter=',')

            overall_rde_rmse, overall_rde_mae = compute_metrics(future_truth.flatten(), preds_rde.flatten())
            overall_rde_delay_rmse, overall_rde_delay_mae = compute_metrics(future_truth.flatten(), preds_rde_delay.flatten())

            print(f"\n  RDE RMSE: {overall_rde_rmse:.4f}, MAE: {overall_rde_mae:.4f}")
            print(f"  RDE-Delay RMSE: {overall_rde_delay_rmse:.4f}, MAE: {overall_rde_delay_mae:.4f}")

            fut_index = np.arange(horizon)
            pred_fig_path = plot_prediction_comparison(
                fut_index, sample_future, preds_rde, preds_rde_delay,
                np.ones_like(preds_rde) * 0.1, np.ones_like(preds_rde_delay) * 0.1,
                out_dir, timestamp, args.target_dim
            )

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