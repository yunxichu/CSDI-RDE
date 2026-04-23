"""Visualize CSDI imputation vs ground truth on selected L96 dims.

Shows:
- ground truth trajectory (solid line)
- observed points (circles, kept)
- linear-interp baseline (dashed)
- CSDI imputed points (colored dots, ONLY at masked positions)

This answers "does CSDI actually learn anything on L96" isolated from SVGP rollout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from experiments.week1.lorenz96_utils import (
    integrate_lorenz96, lorenz96_attractor_std, LORENZ96_F_DEFAULT,
)
from experiments.week1.lorenz63_utils import make_sparse_noisy, linear_interp_fill, HarshnessScenario


SCENARIOS = {
    "S0": HarshnessScenario("S0", 0.00, 0.00),
    "S2": HarshnessScenario("S2", 0.40, 0.30),
    "S3": HarshnessScenario("S3", 0.60, 0.50),
    "S4": HarshnessScenario("S4", 0.75, 0.80),
    "S5": HarshnessScenario("S5", 0.90, 1.20),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="L96 CSDI ckpt (post-patch)")
    ap.add_argument("--scenario", default="S3", choices=list(SCENARIOS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--F", type=float, default=LORENZ96_F_DEFAULT)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--dims", nargs="+", type=int, default=[0, 5, 10, 15])
    ap.add_argument("--tag", default="default")
    args = ap.parse_args()

    sc = SCENARIOS[args.scenario]
    attr_std = lorenz96_attractor_std(N=args.N, F=args.F)
    print(f"[setup] scenario={sc.name} s={sc.sparsity} σ={sc.noise_std_frac} attr_std={attr_std:.3f}")

    traj = integrate_lorenz96(args.seq_len, N=args.N, F=args.F, dt=args.dt,
                               seed=args.seed, spinup=2000)
    obs, mask = make_sparse_noisy(traj, sparsity=sc.sparsity,
                                   noise_std_frac=sc.noise_std_frac,
                                   attractor_std=attr_std, seed=args.seed)

    # Linear baseline
    linear = linear_interp_fill(obs)
    rmse_linear = float(np.sqrt(((linear - traj) ** 2).mean()))

    # CSDI imputation (with fixed attractor_std!)
    from methods.csdi_impute_adapter import set_csdi_checkpoint, csdi_impute
    set_csdi_checkpoint(args.ckpt)
    csdi_out = csdi_impute(obs, n_samples=8,
                            sigma_override=sc.noise_std_frac * attr_std,
                            attractor_std=attr_std)
    rmse_csdi = float(np.sqrt(((csdi_out - traj) ** 2).mean()))

    print(f"[rmse] linear={rmse_linear:.3f}  csdi={rmse_csdi:.3f}  "
          f"{'✓ CSDI wins' if rmse_csdi < rmse_linear else '✗ linear wins'} "
          f"({(rmse_csdi/rmse_linear - 1)*100:+.1f}% vs linear)")

    # Plot selected dims
    T = traj.shape[0]
    t = np.arange(T) * args.dt * 1.68  # in Lyapunov time
    n_plot = len(args.dims)
    fig, axes = plt.subplots(n_plot, 1, figsize=(11, 2.5 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]

    for ax, d in zip(axes, args.dims):
        m = mask[:, d].astype(bool) if mask.ndim == 2 else mask.astype(bool)
        t_obs = t[m]
        y_obs = obs[m, d]
        t_miss = t[~m]
        y_true_miss = traj[~m, d]
        y_csdi_miss = csdi_out[~m, d]
        y_linear_miss = linear[~m, d]

        # Truth curve
        ax.plot(t, traj[:, d], "k-", linewidth=1.5, alpha=0.7, label="ground truth")
        # Observed points
        ax.scatter(t_obs, y_obs, s=18, c="green", marker="o", alpha=0.7,
                   label=f"observed (n={m.sum()})", zorder=3)
        # Linear interp at missing positions
        ax.scatter(t_miss, y_linear_miss, s=12, c="C0", marker="x", alpha=0.5,
                   label="linear@missing")
        # CSDI imputed points at missing positions
        ax.scatter(t_miss, y_csdi_miss, s=15, c="red", marker="^", alpha=0.8,
                   label="CSDI@missing", zorder=4)

        ax.set_ylabel(f"dim {d}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5)

    axes[0].set_title(f"L96 N={args.N} F={args.F} — {sc.name} (s={sc.sparsity}, σ={sc.noise_std_frac}), seed={args.seed}\n"
                       f"linear RMSE={rmse_linear:.3f}, CSDI RMSE={rmse_csdi:.3f} "
                       f"({(rmse_csdi/rmse_linear - 1)*100:+.1f}% vs linear)  --  ckpt: {Path(args.ckpt).name}",
                       fontsize=11)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("time (Λ, Lyapunov)")
    plt.tight_layout()

    out = Path(__file__).resolve().parents[1] / "week1" / "figures" / f"l96_csdi_imputation_{args.scenario}_seed{args.seed}_{args.tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[saved] {out}")
    plt.close()


if __name__ == "__main__":
    main()
