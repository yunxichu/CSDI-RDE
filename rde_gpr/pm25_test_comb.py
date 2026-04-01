#!/usr/bin/env python3
"""
PM2.5 完整预测流程: CSDI补值 + RDE/RDE-Delay预测 + 可视化

流程:
1. CSDI补值：将稀疏/缺失数据补全
2. RDE预测：空间维度组合嵌入
3. RDE-Delay预测：时间延迟嵌入
4. 结果对比与可视化
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import multiprocessing as mp
from functools import partial
from itertools import combinations

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'csdi'))
sys.path.insert(0, os.path.join(base_dir, 'datasets'))

import torch
import yaml
from dataset_pm25 import get_dataloader
from main_model import CSDI_PM25


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(model_path, config_path, device='cpu'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model = CSDI_PM25(config, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, config


def csdi_impute(model, observed_data, observed_mask, cond_mask, observed_tp, device, n_samples=100):
    """CSDI补值"""
    observed_data = observed_data.unsqueeze(0).to(device)
    observed_mask = observed_mask.unsqueeze(0).to(device)
    cond_mask = cond_mask.clone().unsqueeze(0).to(device)
    observed_tp = observed_tp.unsqueeze(0).to(device)

    observed_data = observed_data.permute(0, 2, 1)
    observed_mask = observed_mask.permute(0, 2, 1)
    cond_mask = cond_mask.permute(0, 2, 1)

    with torch.no_grad():
        side_info = model.get_side_info(observed_tp, cond_mask)
        samples = model.impute(observed_data, cond_mask, side_info, n_samples)
        samples = samples.permute(0, 1, 3, 2)
        samples = samples.squeeze(0)
        result = np.mean(samples.cpu().numpy(), axis=0)

    return result


def rde_predict(traindata, target_idx, L=4, s=50, steps_ahead=1, n_jobs=4):
    """RDE预测 - 空间维度组合嵌入"""
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import gaussian_kde
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
    try:
        from sklearn.preprocessing import StandardScaler
        from gpr_module import GaussianProcessRegressor

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
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import gaussian_kde
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
            x_test = np.array([traindata[len(traindata) - 1 - d, di] for d, di in zip(delays, dims)]).reshape(1, -1)

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


def plot_imputation_quality(observed_data, imputed_data, gt_data, output_dir, timestamp):
    """可视化补值质量"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    n_steps = min(100, len(gt_data))
    t = np.arange(n_steps)

    ax = axes[0]
    ax.plot(t, gt_data[:n_steps], 'k-', lw=1.5, alpha=0.7, label='Ground Truth')
    ax.plot(t, imputed_data[:n_steps], 'r--', lw=1.5, alpha=0.8, label='Imputed')
    ax.scatter(t[::4], observed_data[:n_steps:4], s=30, c='blue', zorder=5, label='Observed')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('PM2.5')
    ax.set_title('CSDI Imputation Quality')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    err = imputed_data[:n_steps] - gt_data[:n_steps]
    ax2.bar(t, err, color=['crimson' if e > 0 else 'steelblue' for e in err], alpha=0.7)
    ax2.axhline(0, color='k', lw=1)
    rmse, mae = compute_metrics(gt_data[:n_steps], imputed_data[:n_steps])
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

    n_steps = min(100, len(y_true))
    t = fut_index[:n_steps]

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, y_true[:n_steps, target_dim], 'k-', lw=2, label='Ground Truth')
    ax1.plot(t, y_pred_rde[:n_steps, target_dim], 'b--', lw=1.5, label='RDE Prediction')
    ax1.fill_between(t,
                     y_pred_rde[:n_steps, target_dim] - 2 * stds_rde[:n_steps, target_dim],
                     y_pred_rde[:n_steps, target_dim] + 2 * stds_rde[:n_steps, target_dim],
                     alpha=0.2, color='blue', label='RDE ±2σ')
    ax1.plot(t, y_pred_rde_delay[:n_steps, target_dim], 'r--', lw=1.5, label='RDE-Delay Prediction')
    ax1.fill_between(t,
                     y_pred_rde_delay[:n_steps, target_dim] - 2 * stds_rde_delay[:n_steps, target_dim],
                     y_pred_rde_delay[:n_steps, target_dim] + 2 * stds_rde_delay[:n_steps, target_dim],
                     alpha=0.2, color='red', label='RDE-Delay ±2σ')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('PM2.5')
    ax1.set_title(f'PM2.5 Prediction Comparison (dim={target_dim})')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    err_rde = y_true[:n_steps, target_dim] - y_pred_rde[:n_steps, target_dim]
    err_rde_delay = y_true[:n_steps, target_dim] - y_pred_rde_delay[:n_steps, target_dim]
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
    parser = argparse.ArgumentParser(description="PM2.5: CSDI Imputation + RDE/RDE-Delay Prediction")
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--model_path', type=str, default='', help='Path to CSDI model')
    parser.add_argument('--config_path', type=str, default='config/base.yaml', help='Path to config')
    parser.add_argument('--data_path', type=str, default='./data/pm25/', help='Data path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--split_ratio', type=float, default=0.5, help='Train/test split ratio')
    parser.add_argument('--n_samples', type=int, default=100, help='CSDI imputation samples')
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
    parser.add_argument('--out_dir', type=str, default='', help='Output directory')

    args = parser.parse_args()
    set_seed(args.seed)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"./results/pm25_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    os.chdir(base_dir)

    print("=" * 80)
    print("PM2.5: CSDI Imputation + RDE/RDE-Delay Prediction")
    print("=" * 80)

    # 1. Load data
    print("\n[1] Loading PM2.5 data...")

    datasets_dir = os.path.join(base_dir, 'datasets')
    os.chdir(datasets_dir)

    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        8, device=args.device, validindex=0
    )

    os.chdir(base_dir)

    # Get a sample batch for testing
    batch = next(iter(test_loader))
    observed_data = batch['observed_data'].numpy()
    observed_mask = batch['observed_mask'].numpy()
    gt_mask = batch['gt_mask'].numpy()
    gt_data = batch['gt_data'].numpy()

    print(f"  Observed data shape: {observed_data.shape}")
    print(f"  GT data shape: {gt_data.shape}")

    # For demo, use first sample
    obs = observed_data[0]  # (T, D)
    mask = observed_mask[0]
    gt = gt_data[0]

    # 2. CSDI Imputation
    imputed_data = None
    imp_rmse, imp_mae = np.nan, np.nan
    imp_fig_path = None

    if not args.skip_imputation:
        print("\n[2] CSDI Imputation...")
        if args.model_path and os.path.exists(args.model_path):
            model, config = load_model(args.model_path, args.config_path, args.device)

            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            mask_tensor = torch.tensor(mask, dtype=torch.float32)
            cond_tensor = mask_tensor.clone()
            tp_tensor = torch.arange(len(obs), dtype=torch.float32)

            imputed_data = csdi_impute(model, obs_tensor, mask_tensor, cond_tensor, tp_tensor, args.device, args.n_samples)

            # Save imputed data
            np.savetxt(os.path.join(out_dir, f'imputed_data_{timestamp}.csv'), imputed_data, delimiter=',')

            # Plot imputation quality
            known_mask = mask[:, 0] > 0.5
            imp_fig_path, imp_rmse, imp_mae = plot_imputation_quality(
                np.where(known_mask, obs[:, 0], np.nan),
                imputed_data[:, 0],
                gt[:, 0],
                out_dir, timestamp
            )
            print(f"  Imputation RMSE: {imp_rmse:.4f}, MAE: {imp_mae:.4f}")
        else:
            print("  WARNING: No model path provided, using observed data as imputed")
            imputed_data = obs.copy()

    # 3. RDE/RDE-Delay Prediction
    if not args.skip_prediction:
        print("\n[3] RDE/RDE-Delay Prediction...")
        horizon = min(100, len(gt) - args.trainlength)

        # Split into history and future
        history = imputed_data[:args.trainlength] if imputed_data is not None else obs[:args.trainlength]
        future_truth = gt[args.trainlength:args.trainlength + horizon]

        print(f"  History length: {len(history)}, Horizon: {horizon}")

        # RDE Prediction
        print("  Running RDE prediction...")
        preds_rde = np.zeros((horizon, gt.shape[1]))
        stds_rde = np.zeros((horizon, gt.shape[1]))

        for step in range(horizon):
            traindata = imputed_data[step:step + args.trainlength] if imputed_data is not None else obs[step:step + args.trainlength]
            for dim in range(gt.shape[1]):
                pred, std = rde_predict(traindata, dim, L=args.L, s=args.s, n_jobs=args.n_jobs)
                preds_rde[step, dim] = pred if not np.isnan(pred) else traindata[-1, dim]
                stds_rde[step, dim] = std if not np.isnan(std) else 0.0

        # RDE-Delay Prediction
        print("  Running RDE-Delay prediction...")
        preds_rde_delay = np.zeros((horizon, gt.shape[1]))
        stds_rde_delay = np.zeros((horizon, gt.shape[1]))

        for step in range(horizon):
            traindata = imputed_data[step:step + args.trainlength] if imputed_data is not None else obs[step:step + args.trainlength]
            for dim in range(gt.shape[1]):
                pred, std = rde_delay_predict(traindata, dim, max_delay=args.max_delay, M=args.M, num_samples=args.num_samples)
                preds_rde_delay[step, dim] = pred if not np.isnan(pred) else traindata[-1, dim]
                stds_rde_delay[step, dim] = std if not np.isnan(std) else 0.0

        # Save predictions
        np.savetxt(os.path.join(out_dir, f'predictions_rde_{timestamp}.csv'), preds_rde, delimiter=',')
        np.savetxt(os.path.join(out_dir, f'predictions_rde_delay_{timestamp}.csv'), preds_rde_delay, delimiter=',')
        np.savetxt(os.path.join(out_dir, f'stds_rde_{timestamp}.csv'), stds_rde, delimiter=',')
        np.savetxt(os.path.join(out_dir, f'stds_rde_delay_{timestamp}.csv'), stds_rde_delay, delimiter=',')

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
            'imputation': {'rmse': imp_rmse, 'mae': imp_mae},
            'rde': {'rmse': overall_rde_rmse, 'mae': overall_rde_mae},
            'rde_delay': {'rmse': overall_rde_delay_rmse, 'mae': overall_rde_delay_mae},
            'params': {
                'trainlength': args.trainlength,
                'L': args.L,
                's': args.s,
                'max_delay': args.max_delay,
                'M': args.M,
                'num_samples': args.num_samples
            }
        }

        with open(os.path.join(out_dir, f'summary_{timestamp}.json'), 'w') as f:
            json.dump(summary, f, indent=4)

        print(f"\n  Results saved to: {out_dir}")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    import random
    main()