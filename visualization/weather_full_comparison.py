# -*- coding: utf-8 -*-
"""
Weather 多方法对比可视化脚本
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr


def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan, "corr": np.nan}
    y_true_m = y_true[mask]
    y_pred_m = y_pred[mask]
    rmse = np.sqrt(np.mean((y_true_m - y_pred_m) ** 2))
    mae = np.mean(np.abs(y_true_m - y_pred_m))
    try:
        corr, _ = pearsonr(y_true_m, y_pred_m)
    except:
        corr = np.nan
    return {"rmse": rmse, "mae": mae, "corr": corr}


def find_latest_prediction(method, base_dir="./save"):
    if method == 'rdegpr':
        pattern = f"{base_dir}/weather_pred_random*/future_pred.csv"
    elif method == 'gru_sliding':
        pattern = f"{base_dir}/weather_gru_sliding_*/future_pred.csv"
    else:
        pattern = f"{base_dir}/weather_{method}_*/future_pred.csv"
    files = glob.glob(pattern)
    if files:
        return sorted(files)[-1]
    return None


def load_prediction(pred_path):
    if pred_path and os.path.exists(pred_path):
        return pd.read_csv(pred_path)
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_path", type=str, default="./data/weather/weather_ground.npy")
    parser.add_argument("--impute_meta_path", type=str, default="")
    parser.add_argument("--methods", type=str, default="rdegpr,gru,lstm")
    parser.add_argument("--output_dir", type=str, default="./save/weather_comparison")
    parser.add_argument("--plot_dims", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ground = np.load(args.ground_path)

    if args.impute_meta_path and os.path.exists(args.impute_meta_path):
        with open(args.impute_meta_path, 'r') as f:
            meta = json.load(f)
        split_point = meta['split_point']
    else:
        split_point = int(len(ground) * 0.5)

    plot_dims = [int(x) for x in args.plot_dims.split(',')]
    n_dims = len(plot_dims)
    methods = [m.strip() for m in args.methods.split(',')]

    y_true = ground[split_point:split_point + args.horizon, plot_dims]

    predictions = {}
    metrics_all = {}

    for method in methods:
        method_display = {
            'rdegpr': 'RDE-GPR',
            'gru': 'GRU',
            'lstm': 'LSTM',
            'gruodebayes': 'GRU-ODE-Bayes',
        }.get(method, method)

        pred_path = find_latest_prediction(method)
        pred_df = load_prediction(pred_path)

        if pred_df is not None:
            n_cols = len(pred_df.columns)
            if n_cols == 1:
                y_pred = np.tile(pred_df.values, (1, len(plot_dims)))
            else:
                y_pred = pred_df.iloc[:, plot_dims].values
            predictions[method] = y_pred

            metrics = []
            for d, dim in enumerate(plot_dims):
                m = compute_metrics(y_true[:, d], y_pred[:, d])
                metrics.append({'dim': dim, 'rmse': m['rmse'], 'mae': m['mae'], 'corr': m['corr']})
            metrics_all[method] = pd.DataFrame(metrics)
            print(f"Loaded {method_display}: {pred_path}")
        else:
            print(f"Warning: {method_display} prediction not found")

    if not predictions:
        print("Error: No predictions found!")
        return

    print("\n" + "=" * 60)
    print("Weather Multi-Method Comparison")
    print("=" * 60)

    for method in predictions:
        method_display = {'rdegpr': 'RDE-GPR', 'gru': 'GRU', 'lstm': 'LSTM', 'gru_sliding': 'GRU-Sliding', 'gruodebayes': 'GRU-ODE-Bayes'}.get(method, method)
        m_df = metrics_all[method]
        overall_rmse = np.sqrt(np.nanmean(m_df['rmse'] ** 2))
        overall_mae = np.nanmean(m_df['mae'])
        print(f"\n{method_display}: RMSE={overall_rmse:.4f}, MAE={overall_mae:.4f}")
        for _, row in m_df.iterrows():
            print(f"  Dim {int(row['dim'])}: RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}")

    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions) + 1))

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, n_dims, figure=fig, hspace=0.4, wspace=0.3)

    for d, dim in enumerate(plot_dims):
        ax = fig.add_subplot(gs[0, d])
        ax.plot(y_true[:, d], 'k-', linewidth=2.5, label='True', alpha=0.9)
        for mi, method in enumerate(predictions.keys()):
            y_pred = predictions[method]
            method_display = {'rdegpr': 'RDE-GPR', 'gru': 'GRU', 'lstm': 'LSTM', 'gru_sliding': 'GRU-Sliding', 'gruodebayes': 'GRU-ODE-Bayes'}.get(method, method)
            ax.plot(y_pred[:, d], '--', linewidth=2, label=method_display, color=colors[mi + 1], alpha=0.8)
        ax.set_title(f'Dim {dim}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        rmse_text = ', '.join([f'{m[:3]}: {metrics_all[m].iloc[d]["rmse"]:.3f}' for m in predictions.keys()])
        ax.text(0.02, 0.02, f'RMSE: {rmse_text}', transform=ax.transAxes, fontsize=8,
               verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    for d, dim in enumerate(plot_dims):
        ax = fig.add_subplot(gs[1, d])
        for mi, method in enumerate(predictions.keys()):
            y_pred = predictions[method]
            method_display = {'rdegpr': 'RDE-GPR', 'gru': 'GRU', 'lstm': 'LSTM', 'gru_sliding': 'GRU-Sliding', 'gruodebayes': 'GRU-ODE-Bayes'}.get(method, method)
            ax.plot(np.abs(y_true[:, d] - y_pred[:, d]), '-', linewidth=1.5, label=method_display, color=colors[mi + 1], alpha=0.8)
        ax.set_title(f'Dim {dim} - |Error|', fontsize=11)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('|Error|')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    for d, dim in enumerate(plot_dims):
        ax = fig.add_subplot(gs[2, d])
        for mi, method in enumerate(predictions.keys()):
            y_pred = predictions[method]
            method_display = {'rdegpr': 'RDE-GPR', 'gru': 'GRU', 'lstm': 'LSTM', 'gru_sliding': 'GRU-Sliding', 'gruodebayes': 'GRU-ODE-Bayes'}.get(method, method)
            ax.scatter(y_true[:, d], y_pred[:, d], alpha=0.5, s=30, color=colors[mi + 1], label=method_display)
        min_val = min(np.nanmin(y_true[:, d]), min(np.nanmin(predictions[m][:, d]) for m in predictions))
        max_val = max(np.nanmax(y_true[:, d]), max(np.nanmax(predictions[m][:, d]) for m in predictions))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, alpha=0.5)
        ax.set_title(f'Dim {dim} - True vs Pred', fontsize=11)
        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

    ax_bar = fig.add_subplot(gs[3, :])
    x = np.arange(len(plot_dims))
    width = 0.8 / len(predictions)
    offset = -0.4 + width / 2

    for mi, method in enumerate(predictions.keys()):
        method_display = {'rdegpr': 'RDE-GPR', 'gru': 'GRU', 'lstm': 'LSTM', 'gru_sliding': 'GRU-Sliding', 'gruodebayes': 'GRU-ODE-Bayes'}.get(method, method)
        rmse_vals = [metrics_all[method].iloc[d]['rmse'] for d in range(len(plot_dims))]
        bars = ax_bar.bar(x + offset + mi * width, rmse_vals, width, label=method_display, color=colors[mi + 1], alpha=0.8)
        for bar, val in zip(bars, rmse_vals):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{val:.2f}',
                       ha='center', va='bottom', fontsize=8, rotation=45)

    ax_bar.set_xlabel('Dimension')
    ax_bar.set_ylabel('RMSE')
    ax_bar.set_title('RMSE Comparison per Dimension', fontsize=12, fontweight='bold')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f'Dim {d}' for d in plot_dims])
    ax_bar.legend(loc='upper right', fontsize=10)
    ax_bar.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Weather Multi-Method Comparison', fontsize=14, fontweight='bold')

    plt.savefig(os.path.join(args.output_dir, 'full_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved: {os.path.join(args.output_dir, 'full_comparison.png')}")

    results = {}
    for method in predictions:
        method_display = {'rdegpr': 'RDE-GPR', 'gru': 'GRU', 'lstm': 'LSTM', 'gru_sliding': 'GRU-Sliding', 'gruodebayes': 'GRU-ODE-Bayes'}.get(method, method)
        m_df = metrics_all[method]
        results[method_display] = {
            'overall_rmse': float(np.sqrt(np.nanmean(m_df['rmse'] ** 2))),
            'overall_mae': float(np.nanmean(m_df['mae'])),
            'per_dimension': m_df.to_dict('records')
        }

    with open(os.path.join(args.output_dir, 'comparison_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Done! Output: {args.output_dir}")


if __name__ == "__main__":
    main()
