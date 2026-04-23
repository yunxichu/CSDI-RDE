"""Debug: does CSDI at S0 (clean, mask=all-1) return something close to truth?

If CSDI output != obs at S0, there's a bug in the imputation chain.
If SVGP forecast on TRUE ctx still collapses, the bug is M3/SVGP, not M1.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from experiments.week1.lorenz96_utils import integrate_lorenz96, lorenz96_attractor_std
from experiments.week1.lorenz63_utils import make_sparse_noisy
from methods.csdi_impute_adapter import set_csdi_checkpoint, csdi_impute, set_csdi_attractor_std

# Setup
N, F, dt, seq_len = 20, 8.0, 0.05, 128
attr_std = lorenz96_attractor_std(N=N, F=F)
set_csdi_attractor_std(attr_std)  # global override

# Pick ckpt
ckpt = "/home/rhl/Github/CSDI-PRO/experiments/week2_modules/ckpts/dyn_csdi_l96_no_noise_c192_vales_best.pt"
set_csdi_checkpoint(ckpt)
print(f"[ckpt] {Path(ckpt).name}")
print(f"[setup] attr_std={attr_std:.4f}")

# Ground truth at seed 0
traj = integrate_lorenz96(seq_len, N=N, F=F, dt=dt, seed=0, spinup=2000)
print(f"[truth] shape={traj.shape}  min={traj.min():.2f}  max={traj.max():.2f}  "
      f"std={traj.std():.3f}  mean={traj.mean():.3f}")

# S0: no sparsity, no noise → obs = truth
observed, mask = make_sparse_noisy(traj, sparsity=0.0, noise_std_frac=0.0,
                                    attractor_std=attr_std, seed=0)
print(f"[S0] observed shape={observed.shape}  NaN count={np.isnan(observed).sum()}")
print(f"[S0] observed vs truth max-abs-diff = {np.abs(observed - traj).max():.6f}  (should be 0)")

# Test 1: CSDI with sigma_override=0 (explicitly clean)
imp0 = csdi_impute(observed, n_samples=8, sigma_override=0.0)
rmse0 = float(np.sqrt(((imp0 - traj) ** 2).mean()))
print(f"[CSDI sigma=0] RMSE vs truth = {rmse0:.4f}  (ideal ~0)")

# Test 2: CSDI with default sigma (MAD estimate)
imp_def = csdi_impute(observed, n_samples=8)
rmse_def = float(np.sqrt(((imp_def - traj) ** 2).mean()))
print(f"[CSDI default sigma=MAD] RMSE vs truth = {rmse_def:.4f}")

# Ratio check: is CSDI output just a scaled version of truth?
# If so, min-square-error scale factor α = <imp, truth> / <truth, truth>
alpha_0 = float((imp0 * traj).sum() / (traj * traj).sum())
alpha_def = float((imp_def * traj).sum() / (traj * traj).sum())
print(f"[scale ratio α = <imp, truth>/<truth, truth>] sigma=0: α={alpha_0:.4f}  MAD: α={alpha_def:.4f}")
print("  (α=1 means perfect scale; α<1 means CSDI output is shrunk)")

# Plot comparison for 4 selected dims
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
rng = np.arange(seq_len) * dt * 1.68
for ax, d in zip(axes, [0, 5, 10, 15]):
    ax.plot(rng, traj[:, d], "k-", linewidth=2, label="truth", alpha=0.7)
    ax.plot(rng, imp0[:, d], "r--", linewidth=1, label=f"CSDI σ=0 (RMSE={rmse0:.3f})", alpha=0.8)
    ax.plot(rng, imp_def[:, d], "b:", linewidth=1, label=f"CSDI σ=MAD (RMSE={rmse_def:.3f})", alpha=0.8)
    ax.set_ylabel(f"dim {d}")
    ax.grid(alpha=0.3)
axes[0].set_title(f"CSDI at S0 clean (mask=all-1, no noise) vs truth — seed 0 — {Path(ckpt).name}\n"
                   f"α_σ=0={alpha_0:.3f}, α_MAD={alpha_def:.3f}  (α=1 ideal)")
axes[0].legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("time (Λ)")
plt.tight_layout()
out = Path("/home/rhl/Github/CSDI-PRO/experiments/week1/figures/l96_csdi_S0_debug.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"[saved] {out}")
