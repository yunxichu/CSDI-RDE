"""Evaluate CSDI L96 ckpts on imputation quality ONLY (no downstream SVGP).

Answers: "Is CSDI itself good at imputing sparse/noisy L96 data?"
- Compares CSDI M1 rmse to linear interp and AR-Kalman baselines on n_eval random windows.
- Sweeps a list of candidate L96 ckpts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.week1.lorenz96_utils import (
    integrate_lorenz96, lorenz96_attractor_std, LORENZ96_F_DEFAULT,
)
from experiments.week1.lorenz63_utils import make_sparse_noisy
from methods.dynamics_impute import impute as baseline_impute
from methods.csdi_impute_adapter import set_csdi_checkpoint, csdi_impute


def eval_one_ckpt(ckpt: Path, N: int, F: float, dt: float,
                   attr_std: float, n_eval: int = 30, seq_len: int = 128, seed: int = 2026):
    set_csdi_checkpoint(str(ckpt))
    rng = np.random.default_rng(seed)
    rmse_csdi_all = []
    rmse_lin_all = []
    rmse_kal_all = []

    for k in range(n_eval):
        seed_k = int(rng.integers(10_000, 100_000))
        traj = integrate_lorenz96(seq_len, N=N, F=F, dt=dt, seed=seed_k)
        sparsity = float(rng.uniform(0.2, 0.90))
        noise_frac = float(rng.uniform(0.0, 1.2))
        obs, mask = make_sparse_noisy(traj, sparsity=sparsity, noise_std_frac=noise_frac,
                                       attractor_std=attr_std, seed=seed_k)

        # Baselines
        rmse_lin = float(np.sqrt(((baseline_impute(obs, kind="linear") - traj) ** 2).mean()))
        try:
            rmse_kal = float(np.sqrt(((baseline_impute(obs, kind="ar_kalman") - traj) ** 2).mean()))
        except Exception:
            rmse_kal = float("nan")

        # CSDI
        imputed = csdi_impute(obs, n_samples=8, sigma_override=noise_frac * attr_std)
        rmse_csdi = float(np.sqrt(((imputed - traj) ** 2).mean()))

        rmse_csdi_all.append(rmse_csdi)
        rmse_lin_all.append(rmse_lin)
        rmse_kal_all.append(rmse_kal)

    return dict(
        rmse_csdi_mean=float(np.mean(rmse_csdi_all)),
        rmse_csdi_std=float(np.std(rmse_csdi_all)),
        rmse_lin_mean=float(np.mean(rmse_lin_all)),
        rmse_kal_mean=float(np.nanmean(rmse_kal_all)),
        n_eval=n_eval,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="L96 CSDI ckpts to eval")
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--F", type=float, default=LORENZ96_F_DEFAULT)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--n_eval", type=int, default=30)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--out", default="experiments/week2_modules/results/csdi_l96_imputation_bench.json")
    args = ap.parse_args()

    attr_std = lorenz96_attractor_std(N=args.N, F=args.F)
    print(f"[setup] L96 N={args.N} F={args.F} attr_std={attr_std:.3f}")
    print(f"[setup] n_eval={args.n_eval}  seq_len={args.seq_len}")
    print()

    results = {"attr_std": attr_std, "N": args.N, "F": args.F, "n_eval": args.n_eval,
               "per_ckpt": {}}
    print(f"{'Ckpt':60s}  {'CSDI':>14s}  {'Linear':>8s}  {'AR-Kal':>8s}")
    print("-" * 100)
    for c in args.ckpts:
        ck = Path(c)
        if not ck.exists():
            print(f"{ck.name:60s}  [MISSING]")
            continue
        try:
            r = eval_one_ckpt(ck, args.N, args.F, args.dt, attr_std,
                              n_eval=args.n_eval, seq_len=args.seq_len)
            results["per_ckpt"][ck.name] = r
            delta_lin = (r["rmse_csdi_mean"] - r["rmse_lin_mean"]) / r["rmse_lin_mean"] * 100
            flag = "✓" if r["rmse_csdi_mean"] < r["rmse_lin_mean"] else "✗"
            print(f"{ck.name:60s}  {r['rmse_csdi_mean']:5.3f}±{r['rmse_csdi_std']:4.3f}  "
                  f"{r['rmse_lin_mean']:>8.3f}  {r['rmse_kal_mean']:>8.3f}  "
                  f"{flag} {delta_lin:+.1f}% vs linear")
        except Exception as e:
            print(f"{ck.name:60s}  ERR: {e}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
