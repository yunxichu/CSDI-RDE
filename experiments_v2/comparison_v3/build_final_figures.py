# -*- coding: utf-8 -*-
"""
生成终版对比图 + 数据表
每个数据集一个子图，所有方法（Track-A 基线 / Track-B 基线 / 我的方法）并排对比

输出:
  figures_final/per_dataset_full_comparison.png    (2×2 子图, 4 数据集一张大图)
  figures_final/{dataset}_all_methods.png          (单数据集详细图)
  figures_final/table_full.png                     (完整对比表图片版)
  data_final/full_comparison.csv                   (完整数据)
  data_final/table_human_readable.md               (人可读 markdown 表)
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 解决中文显示
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR  = os.path.join(OUT_ROOT, "figures_final")
DATA_DIR = os.path.join(OUT_ROOT, "data_final")
EXP_BASE = "/home/rhl/Github/experiments_v2"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# =============================================================================
# 数据结构: list of dicts {dataset, method, track, rmse, mae, std, note, source}
# =============================================================================

DATASETS = ["Lorenz63", "Lorenz96", "PM2.5", "EEG"]

# 读 json overall
def _load(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        o = d.get("overall", d)
        rmse = o.get("rmse", None)
        mae = o.get("mae", None)
        if rmse is None or (isinstance(rmse, float) and np.isnan(rmse)):
            return None
        return {"rmse": float(rmse), "mae": float(mae) if mae is not None else np.nan}
    except Exception:
        return None


def collect_all():
    rows = []

    # ──────── Track-A: 所有方法都用 CSDI 补值数据 ────────
    for ds, ds_key in [("Lorenz63", "lorenz63"), ("Lorenz96", "lorenz96"),
                       ("PM2.5", "pm25"), ("EEG", "eeg")]:
        # 基线
        for sub, label in [("neuralcde", "NeuralCDE"),
                           ("gruodebayes", "GRU-ODE-Bayes"),
                           ("sssd_v2", "SSSD_v2"),
                           ("sssd", "SSSD_v1")]:
            d = _load(f"{EXP_BASE}/{ds_key}/{sub}/metrics.json")
            if d is None:
                continue
            rows.append({
                "dataset": ds, "method": label, "track": "Track-A",
                "rmse": d["rmse"], "mae": d["mae"], "std": np.nan,
                "source": f"experiments_v2/{ds_key}/{sub}",
                "note": "CSDI 补值输入",
            })

    # Lorenz63/96 RDE-GPR 5 seeds 均值 (Track-A)
    for ds, ds_key in [("Lorenz63", "lorenz63"), ("Lorenz96", "lorenz96")]:
        s = _load(f"{EXP_BASE}/{ds_key}/rde_delay/summary_mean.json")
        if not s:
            try:
                with open(f"{EXP_BASE}/{ds_key}/rde_delay/summary_mean.json") as f:
                    s = json.load(f)
            except Exception:
                continue
        if "rde_full40" in s:
            rows.append({
                "dataset": ds, "method": "RDE-GPR (ours)",
                "track": "Track-A",
                "rmse": s["rde_full40"]["rmse_mean"],
                "mae":  np.nan,
                "std":  s["rde_full40"].get("rmse_std", np.nan),
                "source": f"experiments_v2/{ds_key}/rde_delay/summary_mean.json (5 seeds)",
                "note": "CSDI 补值 → RDE-GPR (空间集成), 5 seeds 均值",
            })
            rows.append({
                "dataset": ds, "method": "RDE-Delay-GPR (ours)",
                "track": "Track-A",
                "rmse": s["rde_delay_full40"]["rmse_mean"],
                "mae":  np.nan,
                "std":  s["rde_delay_full40"].get("rmse_std", np.nan),
                "source": f"experiments_v2/{ds_key}/rde_delay/summary_mean.json (5 seeds)",
                "note": "CSDI 补值 → RDE-Delay-GPR (延迟嵌入), 5 seeds 均值",
            })

    # PM2.5 RDE-GPR (Track-A)
    d = _load(f"{EXP_BASE}/pm25/rdegpr_modeB/metrics.json")
    if d:
        rows.append({
            "dataset": "PM2.5", "method": "RDE-GPR (ours)",
            "track": "Track-A",
            "rmse": d["rmse"], "mae": d["mae"], "std": np.nan,
            "source": "experiments_v2/pm25/rdegpr_modeB",
            "note": "CSDI 补值 → RDE-GPR, 全 36 站, trainlength=200",
        })

    # EEG RDE-Delay-GPR (Track-A, 用 eeg_imputed.npy redo 版)
    d = _load(f"{EXP_BASE}/eeg/rde_delay_gpr_modeB_redo/metrics.json")
    if d:
        rows.append({
            "dataset": "EEG", "method": "RDE-Delay-GPR (ours)",
            "track": "Track-A",
            "rmse": d["rmse"], "mae": d["mae"], "std": np.nan,
            "source": "experiments_v2/eeg/rde_delay_gpr_modeB_redo",
            "note": "CSDI 补值 → RDE-Delay-GPR",
        })

    # EEG RDE-GPR 空间版 (对照, 表现差说明 EEG 需要 delay)
    d = _load(f"{EXP_BASE}/eeg/rdegpr_spatial_modeB/metrics.json")
    if d:
        rows.append({
            "dataset": "EEG", "method": "RDE-GPR (ours)",
            "track": "Track-A",
            "rmse": d["rmse"], "mae": d["mae"], "std": np.nan,
            "source": "experiments_v2/eeg/rdegpr_spatial_modeB",
            "note": "CSDI 补值 → RDE-GPR (空间版, 对照)",
        })

    # ──────── Track-B: 基线直接吃稀疏/缺失数据 ────────
    for ds, ds_key, horizon_note in [("Lorenz63", "lorenz63", "sparse_50, h=20"),
                                     ("Lorenz96", "lorenz96", "sparse_50, h=20")]:
        for sub, label in [("neuralcde_sparse", "NeuralCDE"),
                           ("gruodebayes_sparse", "GRU-ODE-Bayes")]:
            d = _load(f"{EXP_BASE}/{ds_key}/{sub}/metrics.json")
            if d is None:
                continue
            rows.append({
                "dataset": ds, "method": label, "track": "Track-B",
                "rmse": d["rmse"], "mae": d["mae"], "std": np.nan,
                "source": f"experiments_v2/{ds_key}/{sub}",
                "note": f"基线直接吃 {horizon_note}",
            })

    # EEG Track-B
    for sub, label, note in [
        ("neuralcde_naive",   "NeuralCDE",     "基线 + forward-fill 预处理"),
        ("gruodebayes_mask",  "GRU-ODE-Bayes", "基线 + NaN mask (论文机制)"),
    ]:
        d = _load(f"{EXP_BASE}/eeg/{sub}/metrics.json")
        if d is None:
            continue
        rows.append({
            "dataset": "EEG", "method": label, "track": "Track-B",
            "rmse": d["rmse"], "mae": d["mae"], "std": np.nan,
            "source": f"experiments_v2/eeg/{sub}",
            "note": note,
        })

    # PM2.5 Track-B
    for sub, label, note in [
        ("neuralcde_mask",   "NeuralCDE",     "基线 + NaN mask (论文机制)"),
        ("gruodebayes_mask", "GRU-ODE-Bayes", "基线 + NaN mask"),
    ]:
        d = _load(f"{EXP_BASE}/pm25/{sub}/metrics.json")
        if d is None:
            continue
        rows.append({
            "dataset": "PM2.5", "method": label, "track": "Track-B",
            "rmse": d["rmse"], "mae": d["mae"], "std": np.nan,
            "source": f"experiments_v2/pm25/{sub}",
            "note": note,
        })

    return pd.DataFrame(rows)


# =============================================================================
# 可视化
# =============================================================================

# 方法家族颜色
COLOR_BY_FAMILY = {
    "NeuralCDE":           ("#2196F3", "#BBDEFB"),      # Track-A 深, Track-B 浅
    "GRU-ODE-Bayes":       ("#FF9800", "#FFE0B2"),
    "SSSD_v1":             ("#CE93D8", "#CE93D8"),
    "SSSD_v2":             ("#9C27B0", "#E1BEE7"),
    "RDE-GPR (ours)":      ("#4CAF50", "#4CAF50"),
    "RDE-Delay-GPR (ours)":("#F44336", "#F44336"),
}


def method_sort_key(m):
    order = {
        "NeuralCDE": 0, "GRU-ODE-Bayes": 1,
        "SSSD_v1": 2, "SSSD_v2": 3,
        "RDE-GPR (ours)": 4, "RDE-Delay-GPR (ours)": 5,
    }
    return order.get(m, 9)


def plot_per_dataset(df, ds, ax):
    sub = df[df["dataset"] == ds].copy()
    if len(sub) == 0:
        ax.set_title(f"{ds} (no data)")
        return

    # 构造 (method, track) 对
    sub["sort_key"] = sub["method"].apply(method_sort_key)
    sub = sub.sort_values(["sort_key", "track"])

    labels, heights, colors, hatches, errs = [], [], [], [], []
    for _, r in sub.iterrows():
        base_c, light_c = COLOR_BY_FAMILY.get(r["method"], ("#999", "#ccc"))
        if r["track"] == "Track-A":
            colors.append(base_c); hatches.append("")
        else:
            colors.append(light_c); hatches.append("///")
        label = f"{r['method']}\n({r['track']})"
        labels.append(label)
        heights.append(r["rmse"])
        errs.append(r["std"] if not pd.isna(r["std"]) else 0)

    xs = np.arange(len(labels))
    bars = ax.bar(xs, heights, yerr=errs, color=colors, hatch=hatches,
                  edgecolor="black", linewidth=0.6, capsize=3, alpha=0.95)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("RMSE", fontsize=10)
    ax.set_title(f"{ds}", fontsize=13, fontweight="bold")
    for b, v in zip(bars, heights):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_full_comparison_grid(df):
    """2×2 大图, 4 数据集"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    for ax, ds in zip(axes.flatten(), DATASETS):
        plot_per_dataset(df, ds, ax)

    # Legend
    legend_items = [
        Patch(facecolor="#4CAF50", edgecolor="black", label="RDE-GPR (ours, Track-A)"),
        Patch(facecolor="#F44336", edgecolor="black", label="RDE-Delay-GPR (ours, Track-A)"),
        Patch(facecolor="#2196F3", edgecolor="black", label="Baseline (Track-A: CSDI 补值)"),
        Patch(facecolor="#BBDEFB", edgecolor="black", hatch="///",
              label="Baseline (Track-B: 原始稀疏/缺失)"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=True)
    fig.suptitle("CSDI-RDE-GPR vs 基线 · Track-A (CSDI 补值) / Track-B (原始稀疏/缺失)",
                 fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    path = os.path.join(FIG_DIR, "per_dataset_full_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_individual(df, ds):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_per_dataset(df, ds, ax)
    legend_items = [
        Patch(facecolor="gray", edgecolor="black", label="Track-A (CSDI 补值)"),
        Patch(facecolor="lightgray", edgecolor="black", hatch="///",
              label="Track-B (原始稀疏/缺失)"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=9)
    plt.tight_layout()
    fn = ds.replace(".", "").replace(" ", "_").lower()
    path = os.path.join(FIG_DIR, f"{fn}_all_methods.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def write_table_md(df):
    lines = []
    lines.append("# CSDI-RDE-GPR 对比表 (完整数据)")
    lines.append("")
    lines.append(f"生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
    lines.append("")
    lines.append("## 对比设计说明")
    lines.append("- **Track-A 预处理对齐**: 所有方法（基线 + 我的）都用 **CSDI 补值后的数据** → 比较纯预测能力")
    lines.append("- **Track-B 完整 pipeline**: 基线直接吃**原始稀疏/缺失数据** → 展示 CSDI+RDE-GPR 整套 pipeline 的价值")
    lines.append("- RDE-GPR (ours) / RDE-Delay-GPR (ours) = CSDI-RDE-GPR 完整方法, 见 `/home/rhl/Github/README.md`")
    lines.append("")

    for ds in DATASETS:
        sub = df[df["dataset"] == ds].copy()
        if len(sub) == 0:
            continue
        sub["sort_key"] = sub["method"].apply(method_sort_key)
        sub = sub.sort_values(["track", "sort_key"])
        lines.append(f"## {ds}")
        lines.append("")
        lines.append("| Track | Method | RMSE | MAE | 说明 | 来源 |")
        lines.append("|-------|--------|------|-----|------|------|")
        for _, r in sub.iterrows():
            rmse_s = f"{r['rmse']:.3f}"
            if not pd.isna(r["std"]):
                rmse_s += f" ± {r['std']:.3f}"
            mae_s = f"{r['mae']:.3f}" if not pd.isna(r["mae"]) else "—"
            highlight = " 🏆" if "ours" in r["method"] else ""
            lines.append(f"| {r['track']} | **{r['method']}**{highlight} | {rmse_s} | {mae_s} | {r['note']} | `{r['source']}` |")
        lines.append("")

    def _pivot_md(dfin, title):
        pv = dfin.pivot_table(index="method", columns="dataset", values="rmse", aggfunc="first")
        method_order = sorted(pv.index, key=method_sort_key)
        pv = pv.reindex(method_order)
        cols = [c for c in DATASETS if c in pv.columns]
        lines.append(f"## {title}")
        lines.append("")
        header = "| Method | " + " | ".join(cols) + " |"
        sep = "|--------|" + "|".join(["------" for _ in cols]) + "|"
        lines.append(header); lines.append(sep)
        for m in pv.index:
            cells = []
            for c in cols:
                v = pv.loc[m, c]
                cells.append("—" if pd.isna(v) else f"{v:.3f}")
            hl = " 🏆" if "ours" in m else ""
            lines.append(f"| **{m}**{hl} | " + " | ".join(cells) + " |")
        lines.append("")

    _pivot_md(df[df["track"] == "Track-A"], "一页速览 Track-A (CSDI 补值, 所有方法, RMSE)")
    _pivot_md(df[df["track"] == "Track-B"], "一页速览 Track-B (基线吃稀疏/缺失, RMSE)")

    path = os.path.join(DATA_DIR, "table_human_readable.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {path}")


def main():
    print("="*60)
    print("  Building final comparison figures + data")
    print("="*60)
    df = collect_all()
    print(f"\nTotal rows: {len(df)}")
    df.to_csv(os.path.join(DATA_DIR, "full_comparison.csv"), index=False)
    print(f"Saved: {DATA_DIR}/full_comparison.csv")

    # 简洁终端输出
    for ds in DATASETS:
        sub = df[df["dataset"] == ds].copy()
        print(f"\n── {ds} ──")
        sub["sort_key"] = sub["method"].apply(method_sort_key)
        sub = sub.sort_values(["track", "sort_key"])
        for _, r in sub.iterrows():
            note = r["note"]
            std = f" ± {r['std']:.2f}" if not pd.isna(r["std"]) else ""
            print(f"  [{r['track']}] {r['method']:<25} RMSE={r['rmse']:8.3f}{std} | {note}")

    plot_full_comparison_grid(df)
    for ds in DATASETS:
        plot_individual(df, ds)
    write_table_md(df)

    print("\n✓ Done. 输出目录:", OUT_ROOT)


if __name__ == "__main__":
    main()
