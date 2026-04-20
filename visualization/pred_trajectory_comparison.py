"""
v2 基线预测轨迹可视化
对比 NeuralCDE / GRU-ODE-Bayes / SSSD 在四个数据集上的预测效果
输出：experiments_v2/figures/ 下的 PNG 图片
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json, os

OUT_DIR = "experiments_v2/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 颜色方案 ──────────────────────────────────────────────────────────────────
C_GT    = "#222222"
C_HIST  = "#888888"
C_NCDE  = "#1f77b4"   # blue
C_GRUB  = "#ff7f0e"   # orange
C_SSSD  = "#d62728"   # red
ALPHA_HIST = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LORENZ 63
# ─────────────────────────────────────────────────────────────────────────────
def plot_lorenz63():
    gt  = pd.read_csv("lorenz_rde_delay/results/gt_100_20260320_110418.csv",
                      header=None).values          # (100, 15)
    ncde = np.load("experiments_v2/lorenz63/neuralcde/future_pred.npy")   # (40, 15)
    grub = np.load("experiments_v2/lorenz63/gruodebayes/future_pred.npy") # (40, 15)
    sssd = np.load("experiments_v2/lorenz63/sssd/future_pred.npy")        # (40, 15)

    T_total = 100
    T_hist  = 60
    T_fut   = 40
    t_all   = np.arange(T_total)
    t_hist  = t_all[:T_hist]
    t_fut   = t_all[T_hist:]

    # Show first trajectory (cols 0,1,2 = x,y,z) and second (cols 3,4,5)
    traj_list = [(0, "Traj-1  x"), (1, "Traj-1  y"), (2, "Traj-1  z"),
                 (3, "Traj-2  x"), (4, "Traj-2  y"), (5, "Traj-2  z")]

    fig, axes = plt.subplots(len(traj_list), 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Lorenz63 — 预测轨迹对比（history 60 → horizon 40）",
                 fontsize=13, fontweight='bold')

    for ax, (col, label) in zip(axes, traj_list):
        ax.plot(t_hist, gt[:T_hist, col], color=C_HIST, alpha=ALPHA_HIST,
                lw=1.5, label="历史 (已知)")
        ax.plot(t_fut,  gt[T_hist:, col], color=C_GT, lw=2,
                linestyle="--", label="真实未来")
        ax.plot(t_fut,  ncde[:, col],  color=C_NCDE, lw=1.5, label="NeuralCDE")
        ax.plot(t_fut,  grub[:, col],  color=C_GRUB, lw=1.5, label="GRU-ODE-Bayes")
        ax.plot(t_fut,  sssd[:, col],  color=C_SSSD, lw=1.5, alpha=0.8,
                label="SSSD")
        ax.axvline(T_hist, color="gray", linestyle=":", lw=1)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper left", fontsize=8, ncol=5)
    axes[-1].set_xlabel("时间步")
    plt.tight_layout()
    out = f"{OUT_DIR}/lorenz63_trajectory.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LORENZ 96
# ─────────────────────────────────────────────────────────────────────────────
def plot_lorenz96():
    gt   = pd.read_csv("lorenz96_rde_delay/results/gt_100_20260323_192045.csv",
                       header=None).values         # (100, 100)
    ncde = np.load("experiments_v2/lorenz96/neuralcde/future_pred.npy")   # (40, 100)
    grub = np.load("experiments_v2/lorenz96/gruodebayes/future_pred.npy")
    sssd = np.load("experiments_v2/lorenz96/sssd/future_pred.npy")

    T_hist = 60
    t_all  = np.arange(100)
    t_hist = t_all[:T_hist]
    t_fut  = t_all[T_hist:]

    # Show 4 representative dimensions
    dims   = [0, 24, 49, 74]
    labels = [f"dim {d}" for d in dims]

    fig, axes = plt.subplots(len(dims), 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Lorenz96 — 预测轨迹对比（history 60 → horizon 40，4个维度）",
                 fontsize=13, fontweight='bold')

    for ax, d, lab in zip(axes, dims, labels):
        ax.plot(t_hist, gt[:T_hist, d], color=C_HIST, alpha=ALPHA_HIST,
                lw=1.5, label="历史")
        ax.plot(t_fut,  gt[T_hist:, d], color=C_GT,   lw=2,
                linestyle="--", label="真实")
        ax.plot(t_fut,  ncde[:, d],  color=C_NCDE, lw=1.5, label="NeuralCDE")
        ax.plot(t_fut,  grub[:, d],  color=C_GRUB, lw=1.5, label="GRU-ODE-Bayes")
        ax.plot(t_fut,  sssd[:, d],  color=C_SSSD, lw=1.5, alpha=0.8, label="SSSD")
        ax.axvline(T_hist, color="gray", linestyle=":", lw=1)
        ax.set_ylabel(lab, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper left", fontsize=8, ncol=5)
    axes[-1].set_xlabel("时间步")
    plt.tight_layout()
    out = f"{OUT_DIR}/lorenz96_trajectory.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PM2.5
# ─────────────────────────────────────────────────────────────────────────────
def plot_pm25():
    # Load history (imputed, 4379 timesteps x 36 stations)
    hist_df = pd.read_csv(
        "save/pm25_history_imputed_split0.5_seed42_20260324_154400/history_imputed.csv"
    )
    stations = [c for c in hist_df.columns if c != "datetime"]  # 36 stations

    # Ground truth: parse raw pm25_ground.txt (tab-sep, embedded header in row 0)
    raw = pd.read_csv(
        "data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
        sep=",", skiprows=1, header=None
    )
    # Try to parse properly
    gt_raw = pd.read_csv(
        "data/pm25/Code/STMVL/SampleData/pm25_ground.txt"
    )
    # Merge header: some variants have tab-sep
    try:
        gt_raw2 = pd.read_csv(
            "data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
            sep=None, engine='python'
        )
    except Exception:
        gt_raw2 = gt_raw

    # The ground truth file first line is a header row embedded as data
    # Re-read properly
    with open("data/pm25/Code/STMVL/SampleData/pm25_ground.txt") as f:
        first_line = f.readline().strip()
    if ',' in first_line:
        sep = ','
    else:
        sep = '\t'

    gt_full = pd.read_csv(
        "data/pm25/Code/STMVL/SampleData/pm25_ground.txt", sep=sep
    )
    # Rename first col to datetime if needed
    gt_full.columns = [c.strip() for c in gt_full.columns]
    dt_col = gt_full.columns[0]
    gt_full = gt_full.rename(columns={dt_col: "datetime"})

    # Align with stations in history
    # Future window: rows 4379 to 4379+24 of the full GT
    gt_future = gt_full.iloc[4379:4379+24].reset_index(drop=True)

    # Predictions: (24, 36)
    ncde = np.load("experiments_v2/pm25/neuralcde/future_pred.npy")   # (24, 36)
    grub = np.load("experiments_v2/pm25/gruodebayes/future_pred.npy") # (24, 36)

    # Pick 4 stations to visualize (use first 4)
    show_stations = stations[:4]
    t_hist = np.arange(-48, 0)    # show last 48 history steps
    t_fut  = np.arange(0, 24)

    fig, axes = plt.subplots(len(show_stations), 1, figsize=(12, 9), sharex=True)
    fig.suptitle("PM2.5 — 预测对比（最后48小时历史 → 未来24小时）",
                 fontsize=13, fontweight='bold')

    for ax, st in zip(axes, show_stations):
        # History (last 48 timesteps)
        hist_vals = hist_df[st].values[-48:]
        ax.plot(t_hist, hist_vals, color=C_HIST, alpha=ALPHA_HIST,
                lw=1.5, label="历史")

        # GT future
        if st in gt_future.columns:
            gt_vals = pd.to_numeric(gt_future[st], errors='coerce').values
            ax.plot(t_fut, gt_vals, color=C_GT, lw=2,
                    linestyle="--", label="真实")
        else:
            # Try by position
            st_idx = stations.index(st)
            gt_vals = pd.to_numeric(gt_future.iloc[:, st_idx+1],
                                    errors='coerce').values
            ax.plot(t_fut, gt_vals, color=C_GT, lw=2,
                    linestyle="--", label="真实")

        # Predictions
        st_idx = stations.index(st)
        ax.plot(t_fut, ncde[:, st_idx], color=C_NCDE, lw=1.5,
                label="NeuralCDE")
        ax.plot(t_fut, grub[:, st_idx], color=C_GRUB, lw=1.5,
                label="GRU-ODE-Bayes")
        ax.axvline(0, color="gray", linestyle=":", lw=1)
        ax.set_ylabel(f"站点 {st}", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper left", fontsize=8, ncol=4)
    axes[-1].set_xlabel("相对时间（小时）")
    plt.tight_layout()
    out = f"{OUT_DIR}/pm25_trajectory.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  EEG
# ─────────────────────────────────────────────────────────────────────────────
def plot_eeg():
    eeg_full = np.load("save/eeg_csdi_imputed/eeg_full.npy")   # (1000, 64)
    ncde = np.load("experiments_v2/eeg/neuralcde/future_pred.npy")   # (24, 3)
    grub = np.load("experiments_v2/eeg/gruodebayes/future_pred.npy") # (24, 3)
    sssd = np.load("experiments_v2/eeg/sssd/future_pred.npy")        # (24, 3)

    T_hist = 976
    target_dims = [0, 1, 2]
    dim_names   = ["CH-0", "CH-1", "CH-2"]

    # History: last 48 steps of the 976
    show_hist = 48
    t_hist = np.arange(-show_hist, 0)
    t_fut  = np.arange(0, 24)

    fig, axes = plt.subplots(len(target_dims), 1, figsize=(12, 7), sharex=True)
    fig.suptitle("EEG — 预测对比（最后48步历史 → 未来24步，目标通道0/1/2）",
                 fontsize=13, fontweight='bold')

    for ax, d, name in zip(axes, target_dims, dim_names):
        hist_vals = eeg_full[T_hist-show_hist:T_hist, d]
        gt_vals   = eeg_full[T_hist:T_hist+24, d]

        ax.plot(t_hist, hist_vals, color=C_HIST, alpha=ALPHA_HIST,
                lw=1.5, label="历史")
        ax.plot(t_fut,  gt_vals,   color=C_GT, lw=2,
                linestyle="--", label="真实")
        ax.plot(t_fut,  ncde[:, d], color=C_NCDE, lw=1.5, label="NeuralCDE")
        ax.plot(t_fut,  grub[:, d], color=C_GRUB, lw=1.5, label="GRU-ODE-Bayes")
        ax.plot(t_fut,  sssd[:, d], color=C_SSSD, lw=1.5, alpha=0.8, label="SSSD")
        ax.axvline(0, color="gray", linestyle=":", lw=1)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper left", fontsize=8, ncol=5)
    axes[-1].set_xlabel("相对时间步")
    plt.tight_layout()
    out = f"{OUT_DIR}/eeg_trajectory.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  METRICS BAR CHART（RMSE）
# ─────────────────────────────────────────────────────────────────────────────
def plot_rmse_bars():
    data = {
        "Lorenz63":  {"NeuralCDE": 6.05,  "GRU-ODE-Bayes": 5.97,  "SSSD": 18.80},
        "Lorenz96":  {"NeuralCDE": 9.94,  "GRU-ODE-Bayes": 4.10,  "SSSD": 5.59},
        "PM2.5":     {"NeuralCDE": 15.06, "GRU-ODE-Bayes": 20.99, "SSSD": 90.72},
        "EEG":       {"NeuralCDE": 17.04, "GRU-ODE-Bayes": 6.24,  "SSSD": 87.57},
    }

    datasets = list(data.keys())
    methods  = ["NeuralCDE", "GRU-ODE-Bayes", "SSSD"]
    colors   = [C_NCDE, C_GRUB, C_SSSD]
    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (m, c) in enumerate(zip(methods, colors)):
        vals = [data[ds][m] for ds in datasets]
        bars = ax.bar(x + i*width, vals, width, label=m, color=c, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{v:.2f}", ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("RMSE（↓越好）", fontsize=11)
    ax.set_title("四数据集基线RMSE对比（v2修复版）", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(90.72, 87.57) * 1.15)
    plt.tight_layout()
    out = f"{OUT_DIR}/rmse_bar_v2.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  LORENZ63 — PHASE SPACE (x-y plane)
# ─────────────────────────────────────────────────────────────────────────────
def plot_lorenz63_phase():
    gt   = pd.read_csv("lorenz_rde_delay/results/gt_100_20260320_110418.csv",
                       header=None).values
    ncde = np.load("experiments_v2/lorenz63/neuralcde/future_pred.npy")
    grub = np.load("experiments_v2/lorenz63/gruodebayes/future_pred.npy")
    sssd = np.load("experiments_v2/lorenz63/sssd/future_pred.npy")

    T_hist = 60

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Lorenz63 — 相空间轨迹（x-y，trajectory 1）",
                 fontsize=13, fontweight='bold')

    gt_hist_x, gt_hist_y = gt[:T_hist, 0], gt[:T_hist, 1]
    gt_fut_x,  gt_fut_y  = gt[T_hist:, 0], gt[T_hist:, 1]

    panels = [
        ("真实", gt_fut_x, gt_fut_y, C_GT),
        ("NeuralCDE", ncde[:, 0], ncde[:, 1], C_NCDE),
        ("GRU-ODE-Bayes", grub[:, 0], grub[:, 1], C_GRUB),
        ("SSSD", sssd[:, 0], sssd[:, 1], C_SSSD),
    ]

    for ax, (title, px, py, c) in zip(axes, panels):
        ax.plot(gt_hist_x, gt_hist_y, color=C_HIST, alpha=0.4, lw=1, label="历史")
        ax.plot(gt_fut_x,  gt_fut_y,  color=C_GT,   lw=1.5,
                linestyle="--", alpha=0.7, label="真实未来")
        ax.plot(px, py, color=c, lw=1.5, label=title)
        ax.scatter([px[0]], [py[0]], color=c, s=40, zorder=5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
        if ax is axes[0]:
            ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()
    out = f"{OUT_DIR}/lorenz63_phase.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("生成可视化图表...")
    plot_lorenz63()
    plot_lorenz63_phase()
    plot_lorenz96()
    plot_pm25()
    plot_eeg()
    plot_rmse_bars()
    print(f"\n全部完成，图片保存在 {OUT_DIR}/")
