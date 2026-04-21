"""Targeted evaluation: Does CSDI beat linear on LONG-GAP scenarios?

Linear interp is unbeatable on Lorenz63 when gaps are 1-3 steps (the signal is
smooth at dt=0.025). But with **contiguous** gaps of 10-30 steps, linear
extrapolates along a straight chord across the chaotic orbit — CSDI has a
chance to win because it has learned the attractor geometry.

This script evaluates both imputers under three gap regimes:

  - iid-sparse:       random independent missing, sparsity 0.6
  - burst-sparse:     sparsity 0.6 but in contiguous chunks of length {8, 16}
  - long-holdout:     single contiguous missing block of length {16, 24, 32}

Produces a table + figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.week1.lorenz63_utils import (
    LORENZ63_ATTRACTOR_STD,
    integrate_lorenz63,
)
from methods.dynamics_csdi import DynamicsCSDI
from methods.dynamics_impute import impute as baseline_impute

REPO_ROOT = Path(__file__).resolve().parents[2]
RES = REPO_ROOT / "experiments" / "week2_modules" / "results"
RES.mkdir(parents=True, exist_ok=True)


def iid_mask(rng, T, sparsity):
    return (rng.random(T) > sparsity).astype(np.float32)


def burst_mask(rng, T, sparsity, burst_len):
    """Sparsity roughly = sparsity with gaps of length ``burst_len``."""
    mask = np.ones(T, dtype=np.float32)
    n_burst = int(T * sparsity / burst_len)
    for _ in range(n_burst):
        start = rng.integers(0, T - burst_len)
        mask[start : start + burst_len] = 0
    return mask


def long_holdout_mask(T, hold_len, center=None):
    mask = np.ones(T, dtype=np.float32)
    if center is None:
        center = T // 2
    s = center - hold_len // 2
    e = s + hold_len
    mask[s:e] = 0
    return mask


def eval_one(model: DynamicsCSDI, traj: np.ndarray, mask: np.ndarray, sigma: float = 0.0) -> dict:
    obs = traj.copy()
    mask_2d = np.repeat(mask[:, None], 3, axis=1)
    obs = obs + np.random.default_rng(0).normal(scale=sigma, size=obs.shape) if sigma > 0 else obs
    obs_with_nan = obs.copy()
    obs_with_nan[mask == 0] = np.nan

    lin = baseline_impute(obs_with_nan, kind="linear")
    kal = baseline_impute(obs_with_nan, kind="ar_kalman")
    samp = model.impute(obs_with_nan, mask_2d, sigma=sigma, n_samples=8)
    dyn = samp.mean(0)
    dyn_std = samp.std(0)

    # eval ON MISSING positions only (fair comparison)
    miss = mask == 0
    if miss.sum() == 0:
        return {}
    r_lin = float(np.sqrt(((lin[miss] - traj[miss]) ** 2).mean()))
    r_kal = float(np.sqrt(((kal[miss] - traj[miss]) ** 2).mean()))
    r_dyn = float(np.sqrt(((dyn[miss] - traj[miss]) ** 2).mean()))
    unc_dyn = float(dyn_std[miss].mean())
    return dict(lin=r_lin, kal=r_kal, dyn=r_dyn, dyn_unc=unc_dyn)


def run(model_path: str, n_trials: int = 20) -> dict:
    import torch
    model = DynamicsCSDI.load(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
    seq_len = model.cfg.seq_len
    rng = np.random.default_rng(2028)

    regimes = {
        "iid_s0.6":       lambda r, T: iid_mask(r, T, 0.6),
        "burst_s0.6_b8":  lambda r, T: burst_mask(r, T, 0.6, 8),
        "burst_s0.6_b16": lambda r, T: burst_mask(r, T, 0.6, 16),
        "hold_16":        lambda r, T: long_holdout_mask(T, 16),
        "hold_24":        lambda r, T: long_holdout_mask(T, 24),
        "hold_32":        lambda r, T: long_holdout_mask(T, 32),
    }
    summary: dict = {}
    for name, mask_fn in regimes.items():
        lin_rmses, kal_rmses, dyn_rmses, uncs = [], [], [], []
        for _ in range(n_trials):
            seed = int(rng.integers(10_000, 100_000))
            traj = integrate_lorenz63(seq_len, dt=0.025, seed=seed)
            mask = mask_fn(np.random.default_rng(seed), seq_len)
            r = eval_one(model, traj, mask, sigma=0.0)
            if not r: continue
            lin_rmses.append(r["lin"]); kal_rmses.append(r["kal"])
            dyn_rmses.append(r["dyn"]); uncs.append(r["dyn_unc"])
        summary[name] = dict(
            n=len(dyn_rmses),
            rmse_lin=float(np.mean(lin_rmses)), rmse_lin_std=float(np.std(lin_rmses)),
            rmse_kal=float(np.mean(kal_rmses)), rmse_kal_std=float(np.std(kal_rmses)),
            rmse_dyn=float(np.mean(dyn_rmses)), rmse_dyn_std=float(np.std(dyn_rmses)),
            dyn_unc=float(np.mean(uncs)),
        )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "experiments/week2_modules/ckpts/dyn_csdi_full_ep120_big.pt"))
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--tag", default="default")
    args = ap.parse_args()

    summary = run(args.ckpt, n_trials=args.n_trials)

    print(f"\n=== CSDI long-gap eval (ckpt={Path(args.ckpt).name}) ===")
    print(f"{'regime':<20} {'n':>3}  {'linear':>12}  {'kalman':>12}  {'dyn-csdi':>12}  {'dyn-unc':>8}")
    for name, r in summary.items():
        print(f"{name:<20} {r['n']:>3}  {r['rmse_lin']:>6.2f}±{r['rmse_lin_std']:.2f}   "
              f"{r['rmse_kal']:>6.2f}±{r['rmse_kal_std']:.2f}   "
              f"{r['rmse_dyn']:>6.2f}±{r['rmse_dyn_std']:.2f}   "
              f"{r['dyn_unc']:>7.2f}")

    out = RES / f"csdi_long_gap_{args.tag}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
