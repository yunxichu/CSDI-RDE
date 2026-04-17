# -*- coding: utf-8 -*-
"""
实验对比汇总生成器（Mode B 滚动 teacher-forcing 主赛道 + Mode A 前馈参考）
------------------------------------------------------------------------
读取 experiments_v2/ 下所有可用结果，聚合成:
  data/all_metrics.csv              四数据集 × 所有方法 × 所有可用模式的指标
  data/mode_B_aligned.csv           Mode B 对齐主对比
  data/mode_A_reference.csv         Mode A 前馈参考
  figures/rmse_bar_mode_B.png       四数据集 RMSE 柱状图
  figures/mae_bar_mode_B.png        MAE 柱状图
  figures/{dataset}_trajectory.png  dim 0 轨迹对比
  figures/mode_comparison.png       同一方法在 Mode A vs Mode B 的差异
  summary.md                        简要结论

用法:
  python experiments_v2/comparison_v3/build_comparison.py
"""

import os, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(OUT_DIR, "data")
FIG_DIR   = os.path.join(OUT_DIR, "figures")
EXP_BASE  = "/home/rhl/Github/experiments_v2"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ========== 数据集 × 方法 × 模式 matrix ==========

DATASETS = ["lorenz63", "lorenz96", "pm25", "eeg"]
DATASET_LABELS = {"lorenz63": "Lorenz63", "lorenz96": "Lorenz96",
                  "pm25": "PM2.5", "eeg": "EEG"}
HORIZONS = {"lorenz63": 40, "lorenz96": 40, "pm25": 24, "eeg": 24}

GT_PATHS = {
    "lorenz63": "/home/rhl/Github/lorenz_rde_delay/results/gt_100_20260320_110418.csv",
    "lorenz96": "/home/rhl/Github/lorenz96_rde_delay/results/gt_100_20260323_192045.csv",
    "pm25":     "/home/rhl/Github/data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
    "eeg":      "/home/rhl/Github/save/eeg_csdi_imputed/eeg_full.npy",
}

METHOD_COLORS = {
    "NeuralCDE":     "#2196F3",
    "GRU-ODE-Bayes": "#FF9800",
    "SSSD_v1":       "#CE93D8",
    "SSSD_v2":       "#9C27B0",
    "RDE":           "#4CAF50",
    "RDE-Delay":     "#F44336",
    "RDE-GPR":       "#4CAF50",
}


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def extract_overall(metrics_json):
    if metrics_json is None:
        return None
    if "overall" in metrics_json:
        return metrics_json["overall"]
    if "rmse" in metrics_json:
        return {"rmse": metrics_json["rmse"],
                "mae": metrics_json.get("mae", float('nan'))}
    return None


def collect_baselines_mode_B():
    """experiments_v2/ 下基线结果 (Mode B 默认滚动 teacher-forcing)"""
    rows = []
    for ds in DATASETS:
        for method_dir, method_label in [
            ("neuralcde",   "NeuralCDE"),
            ("gruodebayes", "GRU-ODE-Bayes"),
            ("sssd",        "SSSD_v1"),
            ("sssd_v2",     "SSSD_v2"),
        ]:
            m = load_json(f"{EXP_BASE}/{ds}/{method_dir}/metrics.json")
            o = extract_overall(m)
            if o is None:
                continue
            rows.append({
                "dataset":  DATASET_LABELS[ds],
                "method":   method_label,
                "mode":     "Mode B (rolling+TF)",
                "rmse":     o["rmse"],
                "mae":      o.get("mae", np.nan),
                "source":   f"experiments_v2/{ds}/{method_dir}/metrics.json",
            })
    return rows


def collect_rde_mode_B():
    """RDE/RDE-Delay/RDE-GPR Mode B (滚动) 对齐结果"""
    rows = []

    # Lorenz63/96 RDE, RDE-Delay (5 seeds 均值)
    for ds in ["lorenz63", "lorenz96"]:
        summary_path = f"{EXP_BASE}/{ds}/rde_delay/summary_mean.json"
        sm = load_json(summary_path)
        if sm is None:
            continue
        rows.append({
            "dataset":  DATASET_LABELS[ds],
            "method":   "RDE",
            "mode":     "Mode B (rolling+TF)",
            "rmse":     sm["rde_full40"]["rmse_mean"],
            "rmse_std": sm["rde_full40"]["rmse_std"],
            "mae":      np.nan,
            "source":   f"experiments_v2/{ds}/rde_delay/summary_mean.json (5 seeds)",
            "notes":    "dim 0, horizon 40, 5 seeds 均值",
        })
        rows.append({
            "dataset":  DATASET_LABELS[ds],
            "method":   "RDE-Delay",
            "mode":     "Mode B (rolling+TF)",
            "rmse":     sm["rde_delay_full40"]["rmse_mean"],
            "rmse_std": sm["rde_delay_full40"]["rmse_std"],
            "mae":      np.nan,
            "source":   f"experiments_v2/{ds}/rde_delay/summary_mean.json (5 seeds)",
            "notes":    "dim 0, horizon 40, 5 seeds 均值",
        })

    # EEG RDE-GPR Mode B
    m = load_json(f"{EXP_BASE}/eeg/rdegpr_modeB/metrics.json")
    o = extract_overall(m)
    if o is not None:
        rows.append({
            "dataset":  "EEG",
            "method":   "RDE-GPR",
            "mode":     "Mode B (rolling+TF)",
            "rmse":     o["rmse"],
            "mae":      o.get("mae", np.nan),
            "source":   "experiments_v2/eeg/rdegpr_modeB/metrics.json",
            "notes":    "history=976, horizon=24, target=0,1,2, trainlength=300",
        })

    # PM25 RDE-GPR Mode B (若已完成)
    m = load_json(f"{EXP_BASE}/pm25/rdegpr_modeB/metrics.json")
    o = extract_overall(m)
    if o is not None:
        rows.append({
            "dataset":  "PM2.5",
            "method":   "RDE-GPR",
            "mode":     "Mode B (rolling+TF)",
            "rmse":     o["rmse"],
            "mae":      o.get("mae", np.nan),
            "source":   "experiments_v2/pm25/rdegpr_modeB/metrics.json",
            "notes":    "split_ratio=0.5, horizon=24, 全36站, trainlength=500",
        })

    return rows


def collect_mode_A_reference():
    """Mode A (前馈 / direct multi-step) 参考数据"""
    rows = []
    m = load_json(f"{EXP_BASE}/eeg/rdegpr_aligned/metrics.json")
    o = extract_overall(m)
    if o is not None:
        rows.append({
            "dataset":  "EEG",
            "method":   "RDE-GPR",
            "mode":     "Mode A (feed-forward direct)",
            "rmse":     o["rmse"],
            "mae":      o.get("mae", np.nan),
            "source":   "experiments_v2/eeg/rdegpr_aligned/metrics.json",
            "notes":    "--multi_step --multi_step_mode direct",
        })
    return rows


def dim0_from_npy(pred_path, gt_path, horizon, is_csv=False):
    if not os.path.exists(pred_path):
        return None, None
    if pred_path.endswith(".npy"):
        pred = np.load(pred_path)
    else:
        pred = np.loadtxt(pred_path, delimiter=',', skiprows=1)
    if gt_path.endswith(".csv") or gt_path.endswith(".txt"):
        try:
            gt = np.loadtxt(gt_path, delimiter=',')
        except Exception:
            df = pd.read_csv(gt_path, index_col="datetime", parse_dates=True)
            gt = df.values
    else:
        gt = np.load(gt_path)
    if gt.ndim == 1:
        gt = gt.reshape(-1, 1)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    return pred, gt


def plot_rmse_bar(df_modeB):
    """四数据集 RMSE 柱状图 (Mode B)"""
    methods_order = ["NeuralCDE", "GRU-ODE-Bayes", "SSSD_v2", "RDE", "RDE-Delay", "RDE-GPR"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, ds in zip(axes, DATASETS):
        sub = df_modeB[df_modeB["dataset"] == DATASET_LABELS[ds]]
        methods, rmses, errs, colors = [], [], [], []
        for m in methods_order:
            row = sub[sub["method"] == m]
            if len(row) == 0:
                continue
            methods.append(m)
            rmses.append(float(row["rmse"].iloc[0]))
            errs.append(float(row["rmse_std"].iloc[0]) if "rmse_std" in row and not pd.isna(row["rmse_std"].iloc[0]) else 0)
            colors.append(METHOD_COLORS.get(m, "#999"))
        if not methods:
            ax.set_title(f"{DATASET_LABELS[ds]}\n(no data)")
            continue
        bars = ax.bar(range(len(methods)), rmses, yerr=errs,
                      color=colors, edgecolor='black', linewidth=0.5, capsize=3)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('RMSE', fontsize=11)
        ax.set_title(f"{DATASET_LABELS[ds]} (horizon={HORIZONS[ds]})", fontsize=13, fontweight='bold')
        for bar, v in zip(bars, rmses):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Mode B (rolling + teacher-forcing) RMSE Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "rmse_bar_mode_B.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_mae_bar(df_modeB):
    methods_order = ["NeuralCDE", "GRU-ODE-Bayes", "SSSD_v2", "RDE", "RDE-Delay", "RDE-GPR"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, ds in zip(axes, DATASETS):
        sub = df_modeB[df_modeB["dataset"] == DATASET_LABELS[ds]]
        methods, maes, colors = [], [], []
        for m in methods_order:
            row = sub[sub["method"] == m]
            if len(row) == 0:
                continue
            mae_v = row["mae"].iloc[0]
            if pd.isna(mae_v):
                continue
            methods.append(m)
            maes.append(float(mae_v))
            colors.append(METHOD_COLORS.get(m, "#999"))
        if not methods:
            ax.set_title(f"{DATASET_LABELS[ds]}\n(no MAE data)")
            continue
        bars = ax.bar(range(len(methods)), maes, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('MAE', fontsize=11)
        ax.set_title(f"{DATASET_LABELS[ds]}", fontsize=13, fontweight='bold')
        for bar, v in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Mode B (rolling + teacher-forcing) MAE Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "mae_bar_mode_B.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_trajectory(dataset):
    """dim 0 轨迹对比图 (Mode B)"""
    horizon = HORIZONS[dataset]
    gt_path = GT_PATHS[dataset]

    # 读 ground truth
    if gt_path.endswith(".csv"):
        gt = np.loadtxt(gt_path, delimiter=',')
    elif gt_path.endswith(".txt"):
        df = pd.read_csv(gt_path, index_col="datetime", parse_dates=True).sort_index()
        gt = df.values
    else:
        gt = np.load(gt_path)
    if gt.ndim == 1:
        gt = gt.reshape(-1, 1)

    # horizon 的 GT
    hist_len = gt.shape[0] - horizon
    gt_fut = gt[hist_len:hist_len + horizon, 0]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(range(horizon), gt_fut, 'k-', linewidth=2, label='Ground Truth')

    methods_files = [
        ("NeuralCDE",     f"{EXP_BASE}/{dataset}/neuralcde/future_pred.npy",   "#2196F3"),
        ("GRU-ODE-Bayes", f"{EXP_BASE}/{dataset}/gruodebayes/future_pred.npy", "#FF9800"),
        ("SSSD_v2",       f"{EXP_BASE}/{dataset}/sssd_v2/future_pred.npy",     "#9C27B0"),
    ]
    for name, path, color in methods_files:
        if os.path.exists(path):
            pred = np.load(path)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            if pred.shape[0] >= horizon:
                ax.plot(range(horizon), pred[:horizon, 0], '--', color=color,
                        linewidth=1.5, label=name, alpha=0.8)

    # RDE / RDE-Delay (Lorenz only, 5 seeds 取第一个)
    if dataset in ("lorenz63", "lorenz96"):
        runs = sorted(glob.glob(f"{EXP_BASE}/{dataset}/rde_delay/run_*_seed42"))
        if runs:
            run0 = runs[0]
            for name, color in [("RDE-Delay", "#F44336")]:
                pred_path = f"{run0}/rde_delay_pred_full40.npy"
                if os.path.exists(pred_path):
                    pred = np.load(pred_path)
                    ax.plot(range(len(pred)), pred, '-', color=color, linewidth=2,
                            label=f"{name} (seed 42)", alpha=0.85)

    # EEG RDE-GPR Mode B
    if dataset == "eeg":
        p = f"{EXP_BASE}/eeg/rdegpr_modeB/future_pred.csv"
        if os.path.exists(p):
            pred = pd.read_csv(p).values  # 包含 index 列可能，尝试
            # future_pred.csv 可能首列是 step index
            if pred.shape[1] > 3:
                pred = pred[:, 1:]
            ax.plot(range(min(horizon, pred.shape[0])), pred[:horizon, 0],
                    '-', color="#4CAF50", linewidth=2, label="RDE-GPR (Mode B)", alpha=0.85)

    # PM25 RDE-GPR Mode B (if available)
    if dataset == "pm25":
        p = f"{EXP_BASE}/pm25/rdegpr_modeB/future_pred.csv"
        if os.path.exists(p):
            try:
                df_p = pd.read_csv(p)
                if "datetime" in df_p.columns:
                    df_p = df_p.set_index("datetime")
                pred = df_p.values
                ax.plot(range(min(horizon, pred.shape[0])), pred[:horizon, 0],
                        '-', color="#4CAF50", linewidth=2, label="RDE-GPR (Mode B)", alpha=0.85)
            except Exception as e:
                print(f"[warn] PM25 RDE-GPR pred read failed: {e}")

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value (dim 0)', fontsize=12)
    ax.set_title(f"{DATASET_LABELS[dataset]} — Mode B Trajectory Comparison (dim 0, horizon={horizon})",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f"{dataset}_trajectory.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_mode_compare(df_all):
    """RDE-GPR 在 Mode A 前馈 vs Mode B 滚动的差异 (目前只有 EEG)"""
    sub = df_all[(df_all["method"] == "RDE-GPR") &
                 (df_all["dataset"].isin(["EEG"]))].copy()
    if len(sub) < 2:
        print("[warn] 不够数据做 mode compare")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    modes = sub["mode"].tolist()
    rmses = sub["rmse"].tolist()
    colors = ["#999", "#4CAF50"]
    bars = ax.bar(modes, rmses, color=colors, edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{v:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('EEG RDE-GPR: Mode A 前馈 vs Mode B 滚动 (h=976, horizon=24)',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "mode_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def write_summary_md(df_all, df_modeB):
    lines = []
    lines.append("# 实验对比总结（Mode B 滚动 teacher-forcing 为主赛道）")
    lines.append("")
    lines.append(f"生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
    lines.append("")
    lines.append("## 对齐基线 Mode B 的主对比表")
    lines.append("")
    lines.append("| 数据集 | NeuralCDE | GRU-ODE-Bayes | SSSD_v2 | RDE | RDE-Delay | RDE-GPR |")
    lines.append("|--------|-----------|---------------|---------|-----|-----------|---------|")
    for ds in ["Lorenz63", "Lorenz96", "PM2.5", "EEG"]:
        row = [ds]
        sub = df_modeB[df_modeB["dataset"] == ds]
        for m in ["NeuralCDE", "GRU-ODE-Bayes", "SSSD_v2", "RDE", "RDE-Delay", "RDE-GPR"]:
            r = sub[sub["method"] == m]
            if len(r) == 0:
                row.append("—")
            else:
                rmse = float(r["rmse"].iloc[0])
                std = r.get("rmse_std", [np.nan]).iloc[0] if "rmse_std" in r else np.nan
                if not pd.isna(std) and std > 0:
                    row.append(f"{rmse:.2f}±{std:.2f}")
                else:
                    row.append(f"{rmse:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("（数字是 RMSE；Lorenz63/96 是 dim 0 / 5 seeds 均值；PM2.5/EEG 是全 target 维度 overall）")
    lines.append("")
    lines.append("## 对齐设置说明")
    lines.append("")
    lines.append("- **Lorenz63/96**: trainlength=60, horizon=40, dim=0（5 seeds 均值）")
    lines.append("- **PM2.5**: split_ratio=0.5, horizon=24, target=全 36 站, trainlength=500（RDE-GPR）/ horizon 长度窗 (基线)")
    lines.append("- **EEG**: history=976, horizon=24, target=0,1,2, trainlength=300 (RDE-GPR)")
    lines.append("- **预测模式**: 所有方法 **单步滚动 + teacher-forcing** — 每步预测 1 步，下一步窗口引入真值")
    lines.append("")
    lines.append("## Mode A 前馈参考（不是主赛道）")
    lines.append("")
    lines.append("| 实验 | RMSE | 备注 |")
    lines.append("|------|------|------|")
    ref = df_all[df_all["mode"].str.contains("Mode A", na=False)]
    for _, r in ref.iterrows():
        lines.append(f"| {r['dataset']} {r['method']} | {r['rmse']:.2f} | {r.get('notes','')} |")
    lines.append("")
    lines.append("## 运行中/待更新")
    if not (df_modeB[(df_modeB["dataset"] == "PM2.5") & (df_modeB["method"] == "RDE-GPR")]).shape[0]:
        lines.append("- PM2.5 RDE-GPR Mode B 仍在跑（CPU 2 核，预计 1-2h）")
    if not (df_modeB[(df_modeB["dataset"] == "PM2.5") & (df_modeB["method"] == "SSSD_v2")]).shape[0]:
        lines.append("- PM2.5 SSSD v2 Mode B 仍在 GPU 7 跑（~30% 进度）")
    lines.append("")
    lines.append("完成后重新运行 `python experiments_v2/comparison_v3/build_comparison.py` 更新图表。")
    path = os.path.join(OUT_DIR, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {path}")


def main():
    print("=" * 60)
    print("  Building experiments_v2/comparison_v3/")
    print("=" * 60)

    rows_b = collect_baselines_mode_B()
    rows_b += collect_rde_mode_B()
    rows_a = collect_mode_A_reference()

    df_all = pd.DataFrame(rows_b + rows_a)
    df_modeB = pd.DataFrame(rows_b)

    df_all.to_csv(os.path.join(DATA_DIR, "all_metrics.csv"), index=False)
    df_modeB.to_csv(os.path.join(DATA_DIR, "mode_B_aligned.csv"), index=False)
    pd.DataFrame(rows_a).to_csv(os.path.join(DATA_DIR, "mode_A_reference.csv"), index=False)

    print("\nMode B aligned data:")
    print(df_modeB[["dataset", "method", "rmse", "mae"]].to_string(index=False))

    plot_rmse_bar(df_modeB)
    plot_mae_bar(df_modeB)
    for ds in DATASETS:
        plot_trajectory(ds)
    plot_mode_compare(df_all)

    write_summary_md(df_all, df_modeB)

    print("\n✓ Done. 输出在:", OUT_DIR)


if __name__ == "__main__":
    main()
