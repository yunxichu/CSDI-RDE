# -*- coding: utf-8 -*-
"""
全数据集基线对比可视化脚本
============================================================
生成4个数据集上所有方法的RMSE/MAE柱状对比图

用法:
  python visualization/baseline_comparison.py
"""

import os, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = "./experiments_v1/comparison_figures"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = {
    "lorenz63": {"horizon": 40, "rde_rmse": 0.23, "rde_delay_rmse": 0.12},
    "lorenz96": {"horizon": 40, "rde_rmse": 1.22, "rde_delay_rmse": 0.23},
    "pm25":     {"horizon": 24, "rde_rmse": None, "rde_delay_rmse": None},
    "eeg":      {"horizon": 24, "rde_rmse": None, "rde_delay_rmse": None},
}

BASELINES = ["neuralcde", "gruodebayes", "sssd"]
BASELINE_LABELS = {"neuralcde": "NeuralCDE", "gruodebayes": "GRU-ODE-Bayes", "sssd": "SSSD"}
BASELINE_COLORS = {"neuralcde": "#2196F3", "gruodebayes": "#FF9800", "sssd": "#9C27B0",
                   "RDE": "#4CAF50", "RDE-Delay": "#F44336"}

EXP_BASE = "./experiments_v1"


def load_baseline_metrics(dataset, method):
    mpath = os.path.join(EXP_BASE, dataset, method, "metrics.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            m = json.load(f)
        if "overall" in m:
            return m["overall"]
        return m
    return None


def load_rde_metrics():
    rde_results = {}
    pm25_path = "./best_record/pm25_rc_rde_0.5_42_20260317_122531/metrics.json"
    if os.path.exists(pm25_path):
        with open(pm25_path) as f:
            rde_results["pm25"] = json.load(f)

    eeg_dirs = ["./save/eeg_rde", "./save/eeg_rde_delay"]
    for d in eeg_dirs:
        mpath = os.path.join(d, "metrics.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                rde_results["eeg"] = json.load(f)

    l63_path = "./lorenz_rde_delay/results/25experiments.csv"
    if os.path.exists(l63_path):
        data = np.loadtxt(l63_path, delimiter=',')
        rde_results["lorenz63"] = {"rmse": float(np.mean(data[:, 0])), "mae": float(np.mean(data[:, 1]))}

    l96_path = "./lorenz96_rde_delay/results/25experiments.csv"
    if os.path.exists(l96_path):
        data = np.loadtxt(l96_path, delimiter=',')
        rde_results["lorenz96"] = {"rmse": float(np.mean(data[:, 0])), "mae": float(np.mean(data[:, 1]))}

    return rde_results


def plot_rmse_comparison(all_results):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    datasets_order = ["lorenz63", "lorenz96", "pm25", "eeg"]
    dataset_labels = {"lorenz63": "Lorenz63", "lorenz96": "Lorenz96", "pm25": "PM2.5", "eeg": "EEG"}

    for ax, ds in zip(axes, datasets_order):
        methods, rmses, colors = [], [], []

        for method in BASELINES:
            m = all_results.get(ds, {}).get(method)
            if m and "rmse" in m:
                methods.append(BASELINE_LABELS[method])
                rmses.append(m["rmse"])
                colors.append(BASELINE_COLORS[method])

        rde_m = all_results.get(ds, {}).get("RDE-Delay")
        if rde_m and "rmse" in rde_m:
            methods.append("RDE-Delay")
            rmses.append(rde_m["rmse"])
            colors.append(BASELINE_COLORS["RDE-Delay"])

        if methods:
            bars = ax.bar(range(len(methods)), rmses, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
            ax.set_ylabel('RMSE', fontsize=11)
            ax.set_title(dataset_labels[ds], fontsize=13, fontweight='bold')
            for bar, val in zip(bars, rmses):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8)
            ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Baseline Comparison across Datasets (RMSE)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "rmse_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUT_DIR}/rmse_comparison.png")


def plot_mae_comparison(all_results):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    datasets_order = ["lorenz63", "lorenz96", "pm25", "eeg"]
    dataset_labels = {"lorenz63": "Lorenz63", "lorenz96": "Lorenz96", "pm25": "PM2.5", "eeg": "EEG"}

    for ax, ds in zip(axes, datasets_order):
        methods, maes, colors = [], [], []

        for method in BASELINES:
            m = all_results.get(ds, {}).get(method)
            if m and "mae" in m:
                methods.append(BASELINE_LABELS[method])
                maes.append(m["mae"])
                colors.append(BASELINE_COLORS[method])

        rde_m = all_results.get(ds, {}).get("RDE-Delay")
        if rde_m and "mae" in rde_m:
            methods.append("RDE-Delay")
            maes.append(rde_m["mae"])
            colors.append(BASELINE_COLORS["RDE-Delay"])

        if methods:
            bars = ax.bar(range(len(methods)), maes, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
            ax.set_ylabel('MAE', fontsize=11)
            ax.set_title(dataset_labels[ds], fontsize=13, fontweight='bold')
            for bar, val in zip(bars, maes):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.2f}', ha='center', va='bottom', fontsize=8)
            ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Baseline Comparison across Datasets (MAE)', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mae_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUT_DIR}/mae_comparison.png")


def plot_trajectory_comparison(dataset, gt_path, rde_pred_dir=None):
    ds_labels = {"lorenz63": "Lorenz63", "lorenz96": "Lorenz96", "pm25": "PM2.5", "eeg": "EEG"}

    if dataset in ("lorenz63", "lorenz96"):
        gt = np.loadtxt(gt_path, delimiter=',')
    elif dataset == "pm25":
        df = pd.read_csv(gt_path)
        gt = df.iloc[:, 1:].values
    else:
        gt = np.load(gt_path)

    if gt.ndim == 1:
        gt = gt.reshape(-1, 1)

    horizon = DATASETS[dataset]["horizon"]
    hist_len = gt.shape[0] - horizon
    gt_future = gt[hist_len:hist_len+horizon, 0]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(horizon), gt_future, 'k-', linewidth=2, label='Ground Truth')

    colors = ["#2196F3", "#FF9800", "#9C27B0"]
    for method, color in zip(BASELINES, colors):
        pred_path = os.path.join(EXP_BASE, dataset, method, "future_pred.csv")
        if os.path.exists(pred_path):
            try:
                pred = pd.read_csv(pred_path, header=0).values
            except Exception:
                pred = np.loadtxt(pred_path, delimiter=',')
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            if pred.shape[0] >= horizon:
                ax.plot(range(horizon), pred[:horizon, 0], '--', color=color,
                       linewidth=1.5, label=BASELINE_LABELS[method])

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value (dim 0)', fontsize=12)
    ax.set_title(f'{ds_labels[dataset]} - Forecast Comparison (dim 0)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{dataset}_trajectory_comparison.png"), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUT_DIR}/{dataset}_trajectory_comparison.png")


def generate_summary_table(all_results):
    rows = []
    datasets_order = ["lorenz63", "lorenz96", "pm25", "eeg"]
    ds_labels = {"lorenz63": "Lorenz63", "lorenz96": "Lorenz96", "pm25": "PM2.5", "eeg": "EEG"}
    all_methods = BASELINES + ["RDE-Delay"]

    for ds in datasets_order:
        row = {"Dataset": ds_labels[ds]}
        for method in all_methods:
            m = all_results.get(ds, {}).get(method)
            label = BASELINE_LABELS.get(method, method)
            if m and "rmse" in m:
                row[f"{label}_RMSE"] = f"{m['rmse']:.4f}"
                row[f"{label}_MAE"] = f"{m['mae']:.4f}" if "mae" in m else "N/A"
            else:
                row[f"{label}_RMSE"] = "N/A"
                row[f"{label}_MAE"] = "N/A"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "summary_table.csv"), index=False)
    print(f"\nSummary Table:")
    print(df.to_string(index=False))
    print(f"\nSaved: {OUT_DIR}/summary_table.csv")
    return df


def main():
    all_results = {}

    for ds in DATASETS:
        all_results[ds] = {}
        for method in BASELINES:
            m = load_baseline_metrics(ds, method)
            if m:
                all_results[ds][method] = m

    rde_results = load_rde_metrics()
    for ds, m in rde_results.items():
        if ds in all_results:
            all_results[ds]["RDE-Delay"] = m

    print("=" * 60)
    print("  基线对比实验结果汇总")
    print("=" * 60)
    for ds, methods in all_results.items():
        print(f"\n{ds.upper()}:")
        for method, m in methods.items():
            rmse_val = m.get('rmse', 'N/A')
            mae_val = m.get('mae', 'N/A')
            rmse_str = f"{rmse_val:.4f}" if isinstance(rmse_val, (int, float)) else str(rmse_val)
            mae_str = f"{mae_val:.4f}" if isinstance(mae_val, (int, float)) else str(mae_val)
            print(f"  {method}: RMSE={rmse_str}, MAE={mae_str}")

    plot_rmse_comparison(all_results)
    plot_mae_comparison(all_results)

    gt_paths = {
        "lorenz63": "./lorenz_rde_delay/results/gt_100_20260320_110418.csv",
        "lorenz96": "./lorenz96_rde_delay/results/gt_100_20260323_192045.csv",
        "pm25": "./data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
        "eeg": "./save/eeg_csdi_imputed/eeg_full.npy",
    }
    for ds, gtp in gt_paths.items():
        if os.path.exists(gtp):
            plot_trajectory_comparison(ds, gtp)

    generate_summary_table(all_results)


if __name__ == "__main__":
    main()
