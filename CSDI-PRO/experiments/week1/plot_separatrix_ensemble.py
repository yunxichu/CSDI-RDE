"""Separatrix-focused figure: SVGP ensemble paths vs ground truth at lobe switches.

Shows that our probabilistic rollout (K sample paths from the GP posterior) naturally
produces **multi-modal trajectory distributions** at bifurcation points — e.g. the
two Lorenz63 wings — which point-forecast baselines (parrot, Panda) cannot do.

Story panels (paper Figure 3 candidate):
  (a) x(t): 20 ensemble paths overlay the ground-truth, colour-coded by terminal wing.
  (b) y(t): same
  (c) z(t): same (lobe switches show up as dips)
  (d) 3-D xz projection: ensemble cloud + truth — shows spatial spread on attractor

The σ scaling factor for the per-step sample noise is exposed as a knob; the default
multiplies the SVGP predictive std by a Lyapunov-growth factor sqrt(exp(2λ·dt))
so long-horizon spread matches the natural error growth rate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.week1.full_pipeline_rollout import full_pipeline_ensemble_forecast
from experiments.week1.lorenz63_utils import (
    LORENZ63_ATTRACTOR_STD,
    LORENZ63_LYAP,
    integrate_lorenz63,
    make_sparse_noisy,
    valid_prediction_time,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "experiments" / "week1" / "figures"


def find_switch_indices(x: np.ndarray, min_gap: int = 5) -> np.ndarray:
    """Indices where the sign of x flips; treat these as lobe-switch anchors."""
    sign = np.sign(x)
    flips = np.where(np.diff(sign) != 0)[0] + 1
    if len(flips) == 0:
        return flips
    keep = [flips[0]]
    for f in flips[1:]:
        if f - keep[-1] >= min_gap:
            keep.append(f)
    return np.asarray(keep)


def plot_ensemble(
    seed: int, scenario_sparsity: float, scenario_noise: float,
    K: int, pred_len: int, n_ctx: int, dt: float,
    ctx_show: int, fig_path: Path,
):
    traj = integrate_lorenz63(n_ctx + pred_len, dt=dt, seed=seed, spinup=2000)
    ctx_true = traj[:n_ctx]
    future_true = traj[n_ctx:]

    observed, _ = make_sparse_noisy(
        ctx_true, sparsity=scenario_sparsity, noise_std_frac=scenario_noise,
        attractor_std=LORENZ63_ATTRACTOR_STD, seed=seed,
    )
    print(f"running ensemble rollout K={K} …")
    pred_ens = full_pipeline_ensemble_forecast(
        observed, pred_len=pred_len, K=K, seed=seed,
        bayes_calls=10, n_epochs=150,
    )  # [K, pred_len, 3]

    # Per-sample VPT — useful summary
    vpts = np.array([valid_prediction_time(future_true, pred_ens[k], dt=dt, threshold=1.0)
                     for k in range(K)])
    pred_median = np.median(pred_ens, axis=0)
    vpt_median = valid_prediction_time(future_true, pred_median, dt=dt, threshold=1.0)

    # Coverage check
    lo = np.quantile(pred_ens, 0.05, axis=0)
    hi = np.quantile(pred_ens, 0.95, axis=0)
    picp_per_ch = ((future_true >= lo) & (future_true <= hi)).mean(axis=0)
    picp_all = float(((future_true >= lo) & (future_true <= hi)).all(axis=1).mean())

    # Classify each sample by terminal wing (sign of x at final step)
    terminal_wing = np.sign(pred_ens[:, -1, 0])
    truth_terminal = np.sign(future_true[-1, 0])

    # Identify switch indices in the truth for annotation
    switches = find_switch_indices(future_true[:, 0])
    print(f"truth has {len(switches)} lobe switches in the 128-step future at "
          f"steps {switches.tolist()}")

    # --- plot ---
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 3, width_ratios=[2, 2, 1.2])

    t_ctx = np.arange(n_ctx) * dt
    t_fut = n_ctx * dt + np.arange(pred_len) * dt
    t_ctx_tail = t_ctx[-ctx_show:]

    ch_names = ["x", "y", "z"]
    for ch in range(3):
        ax = fig.add_subplot(gs[ch, 0])
        ax.plot(t_ctx_tail, ctx_true[-ctx_show:, ch], color="black", lw=1.2, alpha=0.5)
        ax.plot(t_fut, future_true[:, ch], color="black", lw=2.2, label="truth")
        for k in range(K):
            col = "#1b9e77" if terminal_wing[k] > 0 else "#9467bd"
            ax.plot(t_fut, pred_ens[k, :, ch], color=col, lw=0.7, alpha=0.4)
        # 5-95% band
        ax.fill_between(t_fut, lo[:, ch], hi[:, ch], color="#1b9e77", alpha=0.12, label="ens 90% PI")
        ax.axvline(n_ctx * dt, color="red", lw=0.8, alpha=0.6)
        for sw in switches:
            ax.axvline(t_fut[sw], color="orange", lw=0.8, linestyle=":", alpha=0.5)
        if ch == 0:
            ax.set_title(f"seed={seed}  sp={scenario_sparsity:.0%}  σ={scenario_noise:.2f}\n"
                         f"ens VPT med={np.median(vpts):.2f}Λ  [{vpts.min():.2f},{vpts.max():.2f}]  "
                         f"PICP x/y/z={picp_per_ch[0]:.2f}/{picp_per_ch[1]:.2f}/{picp_per_ch[2]:.2f}  "
                         f"all-ch={picp_all:.2f}",
                         fontsize=10)
        ax.set_ylabel(f"channel {ch_names[ch]}")
        if ch == 2:
            ax.set_xlabel("time (simulation units)")
        ax.grid(True, alpha=0.2)
        if ch == 0:
            ax.legend(loc="upper left", fontsize=8)

    # middle column — x|z phase diagram
    ax_xz = fig.add_subplot(gs[:, 1])
    ax_xz.plot(ctx_true[-ctx_show:, 0], ctx_true[-ctx_show:, 2], color="grey", lw=0.8, alpha=0.4, label="ctx")
    ax_xz.plot(future_true[:, 0], future_true[:, 2], color="black", lw=2.0, label="truth")
    for k in range(K):
        col = "#1b9e77" if terminal_wing[k] > 0 else "#9467bd"
        ax_xz.plot(pred_ens[k, :, 0], pred_ens[k, :, 2], color=col, lw=0.7, alpha=0.5)
    ax_xz.scatter([future_true[0, 0]], [future_true[0, 2]], color="red", s=40, zorder=5,
                  label="start of forecast")
    ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z")
    ax_xz.set_title("Phase plot (x–z):\nensemble paths colour-coded by terminal x-sign")
    ax_xz.legend(fontsize=8)
    ax_xz.grid(True, alpha=0.2)

    # right column: histogram of terminal wing + VPT histogram
    ax_hist = fig.add_subplot(gs[0, 2])
    ax_hist.bar(["−x wing", "+x wing"],
                [(terminal_wing < 0).sum(), (terminal_wing > 0).sum()],
                color=["#9467bd", "#1b9e77"])
    ax_hist.axhline(K * 0.5, color="grey", linestyle="--", lw=0.7)
    ax_hist.set_title(f"Terminal x-wing counts\ntruth ends: {'−x' if truth_terminal < 0 else '+x'}",
                      fontsize=10)
    ax_hist.set_ylabel("# of K={} paths".format(K))

    ax_vpt = fig.add_subplot(gs[1, 2])
    ax_vpt.hist(vpts, bins=12, color="#1b9e77", alpha=0.7, edgecolor="black")
    ax_vpt.axvline(vpt_median, color="red", lw=2, label=f"median={vpt_median:.2f}")
    ax_vpt.set_xlabel("VPT@1.0 (Λ times)")
    ax_vpt.set_title("Per-sample VPT histogram")
    ax_vpt.legend(fontsize=8)

    # ensemble spread vs horizon
    ax_spread = fig.add_subplot(gs[2, 2])
    spread_x = pred_ens[:, :, 0].std(axis=0)
    spread_y = pred_ens[:, :, 1].std(axis=0)
    spread_z = pred_ens[:, :, 2].std(axis=0)
    h_axis = np.arange(pred_len) * dt * LORENZ63_LYAP
    ax_spread.plot(h_axis, spread_x, label="x", color="C0")
    ax_spread.plot(h_axis, spread_y, label="y", color="C1")
    ax_spread.plot(h_axis, spread_z, label="z", color="C2")
    ax_spread.set_xlabel("horizon (Λ times)")
    ax_spread.set_ylabel("ensemble std")
    ax_spread.set_title("Spread growth vs horizon")
    ax_spread.legend(fontsize=8)
    ax_spread.grid(True, alpha=0.3)

    fig.suptitle(
        "Probabilistic ensemble rollout — does the SVGP posterior split at lobe switches?",
        y=1.00, fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"[fig] saved {fig_path}")
    return dict(vpt_median=float(vpt_median), picp_all=picp_all, n_switches=int(len(switches)),
                terminal_wing_counts={
                    "+x": int((terminal_wing > 0).sum()),
                    "-x": int((terminal_wing < 0).sum()),
                    "truth": int(truth_terminal),
                })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--sparsity", type=float, default=0.0, help="scenario sparsity")
    ap.add_argument("--noise", type=float, default=0.0, help="scenario noise_std_frac")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--pred_len", type=int, default=128)
    ap.add_argument("--n_ctx", type=int, default=512)
    ap.add_argument("--dt", type=float, default=0.025)
    ap.add_argument("--ctx_show", type=int, default=128)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"seed{args.seed}_sp{int(args.sparsity*100):02d}_n{int(args.noise*100):02d}_K{args.K}"
    fig_path = FIG_DIR / f"separatrix_ensemble_{tag}.png"
    summary = plot_ensemble(
        args.seed, args.sparsity, args.noise, args.K, args.pred_len, args.n_ctx, args.dt,
        args.ctx_show, fig_path,
    )
    print(summary)


if __name__ == "__main__":
    main()
