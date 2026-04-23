"""B1: Panda OOD KL measurement — quantify the distributional shift of patch
distributions between clean and linearly-interpolated sparse contexts (closes
Theorem 2(b) lemma L2's non-physical-segment KL jump claim).

Hypothesis (Theorem 2 (b)):
  Panda's PatchTST embeds non-overlapping 16-wide patches. Linear interpolation
  at s > 0 inserts straight segments into otherwise-chaotic contexts; these
  patches have near-zero local curvature and would lie far from the training
  distribution of "chaotic-dynamics patches". The KL between clean-patch and
  sparse-interp-patch distributions should:
    (a) grow monotonically with s
    (b) cross a threshold at s ≈ 0.5 (the theoretical OOD activation point)

Metrics measured per (s, σ):
  - patch curvature |∂²x/∂t²| distribution (clean vs. sparse-interp)
  - Jensen-Shannon divergence between patch-distribution histograms
  - Fraction of patches with curvature < 0.01 (≈ linear segments)
  - Wasserstein-1 distance (robust complement to JS)

Usage:
    python -m experiments.week2_modules.run_panda_ood_kl \\
        --n_trajectories 20 --s_values 0 0.1 0.2 0.35 0.5 0.6 0.7 0.85 0.95 \\
        --sigma_values 0 0.5 \\
        --out_json experiments/week2_modules/results/panda_ood_kl_v1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def make_patches(ctx: np.ndarray, patch_length: int = 16) -> np.ndarray:
    """Split [T, D] context into [n_patches, patch_length, D] non-overlapping patches.
    T is truncated to a multiple of patch_length from the right.
    """
    T, D = ctx.shape
    n_patches = T // patch_length
    ctx = ctx[:n_patches * patch_length]
    return ctx.reshape(n_patches, patch_length, D)


def patch_curvature(patches: np.ndarray) -> np.ndarray:
    """Return per-patch mean absolute second derivative (proxy for nonlinearity).

    Input: [n_patches, patch_length, D]
    Output: [n_patches, D] — per-patch, per-channel curvature
    """
    # second derivative along patch axis: x[i+1] - 2 x[i] + x[i-1]
    second_diff = patches[:, 2:, :] - 2 * patches[:, 1:-1, :] + patches[:, :-2, :]
    return np.abs(second_diff).mean(axis=1)  # [n_patches, D]


def js_divergence(p: np.ndarray, q: np.ndarray, bins: int = 50) -> float:
    """Symmetric Jensen-Shannon divergence between two 1D empirical distributions."""
    bin_edges = np.histogram_bin_edges(np.concatenate([p, q]), bins=bins)
    ph, _ = np.histogram(p, bins=bin_edges, density=False)
    qh, _ = np.histogram(q, bins=bin_edges, density=False)
    ph = ph / (ph.sum() + 1e-12)
    qh = qh / (qh.sum() + 1e-12)
    m = 0.5 * (ph + qh)

    def kl(a, b):
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return 0.5 * kl(ph, m) + 0.5 * kl(qh, m)


def wasserstein1(p: np.ndarray, q: np.ndarray) -> float:
    """1D Wasserstein-1 distance between empirical distributions (scipy backup)."""
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(p, q))


def gather_patches(n_trajectories: int, s: float, sigma: float,
                    n_ctx: int = 512, dt: float = 0.025,
                    patch_length: int = 16) -> np.ndarray:
    """Generate trajectories + apply (s, sigma) + impute → return patches."""
    from experiments.week1.lorenz63_utils import (
        integrate_lorenz63, make_sparse_noisy, LORENZ63_ATTRACTOR_STD,
    )
    from methods.dynamics_impute import impute

    all_patches = []
    for traj_idx in range(n_trajectories):
        traj = integrate_lorenz63(n_ctx, dt=dt, seed=traj_idx, spinup=2000)
        if s == 0 and sigma == 0:
            ctx_filled = traj
        else:
            obs, _ = make_sparse_noisy(
                traj, sparsity=s, noise_std_frac=sigma,
                attractor_std=LORENZ63_ATTRACTOR_STD, seed=traj_idx,
            )
            ctx_filled = impute(obs, kind="linear")
        patches = make_patches(ctx_filled, patch_length=patch_length)  # [P, 16, 3]
        all_patches.append(patches)
    return np.concatenate(all_patches, axis=0)  # [n_traj * P, 16, 3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trajectories", type=int, default=20)
    ap.add_argument("--s_values", nargs="+", type=float,
                    default=[0.0, 0.1, 0.2, 0.35, 0.5, 0.6, 0.7, 0.85, 0.95])
    ap.add_argument("--sigma_values", nargs="+", type=float, default=[0.0, 0.5])
    ap.add_argument("--patch_length", type=int, default=16)
    ap.add_argument("--n_ctx", type=int, default=512)
    ap.add_argument("--out_json", default="experiments/week2_modules/results/panda_ood_kl_v1.json")
    args = ap.parse_args()

    print(f"=== Panda OOD KL measurement (B1) ===")
    print(f"  n_trajectories={args.n_trajectories}")
    print(f"  patch_length={args.patch_length}  n_ctx={args.n_ctx}")
    print(f"  s_values={args.s_values}")
    print(f"  sigma_values={args.sigma_values}")

    # Reference: clean patches (s=0, σ=0)
    print(f"\n--- gathering reference patches (s=0, σ=0) ---")
    ref_patches = gather_patches(args.n_trajectories, 0.0, 0.0,
                                   n_ctx=args.n_ctx, patch_length=args.patch_length)
    ref_curv = patch_curvature(ref_patches)  # [n_ref_patches, D]
    ref_curv_flat = ref_curv.mean(axis=1)    # aggregate over D: [n_ref_patches]
    print(f"  ref patches: {ref_patches.shape}  ref curvature distribution: "
          f"mean={ref_curv_flat.mean():.4f}, std={ref_curv_flat.std():.4f}, "
          f"median={np.median(ref_curv_flat):.4f}")

    records = [dict(s=0.0, sigma=0.0, is_ref=True,
                    curvature_mean=float(ref_curv_flat.mean()),
                    curvature_median=float(np.median(ref_curv_flat)),
                    curvature_low_frac=float((ref_curv_flat < 0.01).mean()),
                    js_vs_ref=0.0, wasserstein_vs_ref=0.0)]

    print(f"\n{'s':>6s} {'σ':>6s}  {'curv_mean':>10s} {'curv_med':>10s} "
          f"{'low_frac':>9s} {'JS':>8s} {'W1':>8s}")

    for sigma in args.sigma_values:
        for s in args.s_values:
            if s == 0 and sigma == 0:
                continue
            test_patches = gather_patches(args.n_trajectories, s, sigma,
                                           n_ctx=args.n_ctx, patch_length=args.patch_length)
            test_curv = patch_curvature(test_patches).mean(axis=1)
            js = js_divergence(ref_curv_flat, test_curv)
            w1 = wasserstein1(ref_curv_flat, test_curv)
            low_frac = float((test_curv < 0.01).mean())

            records.append(dict(
                s=float(s), sigma=float(sigma), is_ref=False,
                curvature_mean=float(test_curv.mean()),
                curvature_median=float(np.median(test_curv)),
                curvature_low_frac=low_frac,
                js_vs_ref=float(js),
                wasserstein_vs_ref=float(w1),
                n_patches=int(len(test_curv)),
            ))
            print(f"{s:>6.2f} {sigma:>6.2f}  {test_curv.mean():>10.4f} "
                  f"{np.median(test_curv):>10.4f} {low_frac:>9.3f} "
                  f"{js:>8.4f} {w1:>8.4f}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(
        config=dict(n_trajectories=args.n_trajectories,
                    patch_length=args.patch_length,
                    n_ctx=args.n_ctx,
                    s_values=args.s_values,
                    sigma_values=args.sigma_values),
        records=records,
    ), indent=2))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
