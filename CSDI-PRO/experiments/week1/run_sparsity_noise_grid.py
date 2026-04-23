"""Sparsity × noise 2D grid ablation (paper §5.X3, REFACTOR_PLAN §6.2 extension).

Motivation (from Option C, 2026-04-23):
  The 4-point n_eff unified experiment (§5.X2) showed that n_eff alone does NOT
  fold (s, σ) into a single control parameter — ours' NRMSE varies 2.4× between
  pure_sparse (0.204) and pure_noise (0.496) at fixed n_eff/n ≈ 0.30.

  §5.X3 extends this to a 3×3 (s, σ) grid to characterize the FAILURE FRONTIER
  of each method in 2D. This is the data needed for:

    - Proposition 5 (new): n_eff is necessary but not sufficient; sparsity and
      noise each have independent failure channels for ambient predictors.
    - Figure X3 (new): contour / heatmap of NRMSE over (s, σ) plane, one panel
      per method. The orthogonal failure modes are visually apparent.

Design — 3×3 grid:
      σ/σ_attr →    0.00      0.50       1.53
  s ↓
  0.00           (clean)   (pure mod σ)   (pure high σ)
  0.35           (mild s)  (mixed mild)   (mild s + high σ)
  0.70           (high s)  (high s + σ)   (harsh)

  These span the interior of the (s, σ) square and include the corners
  (clean / pure-sparse / pure-noise / harsh) so we can see failure frontiers.

Runs: 9 configs × 2 methods (ours_csdi, panda) × 5 seeds = 90 runs.
ETA on 1 V100: ~2 min/run for ours_csdi, ~0.5 min for panda → ~3 hr total.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m experiments.week1.run_sparsity_noise_grid \\
        --ckpt experiments/week2_modules/ckpts/dyn_csdi_full_v6_center_ep20.pt \\
        --n_seeds 5 --methods ours_csdi panda --tag ssgrid_v1
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np


@dataclass
class GridConfig:
    name: str
    sparsity: float
    noise_std_frac: float

    @property
    def n_eff_ratio(self) -> float:
        return (1 - self.sparsity) / (1 + self.noise_std_frac ** 2)


S_VALUES = [0.00, 0.35, 0.70]
SIGMA_VALUES = [0.00, 0.50, 1.53]


def build_grid() -> list[GridConfig]:
    grid = []
    for i, s in enumerate(S_VALUES):
        for j, sig in enumerate(SIGMA_VALUES):
            grid.append(GridConfig(name=f"G{i}{j}_s{s:.2f}_n{sig:.2f}",
                                    sparsity=s, noise_std_frac=sig))
    return grid


def run_one(cfg: GridConfig, seed: int, method: str, ckpt: str | None,
            n_ctx: int = 1200, pred_len: int = 128, dt: float = 0.025) -> dict:
    """Single (config, seed, method) run. Returns record dict."""
    from experiments.week1.lorenz63_utils import (
        integrate_lorenz63, make_sparse_noisy, LORENZ63_ATTRACTOR_STD,
    )

    traj = integrate_lorenz63(n_ctx + pred_len, dt=dt, seed=seed, spinup=2000)
    ctx_true = traj[:n_ctx]
    future_true = traj[n_ctx:]
    obs, mask = make_sparse_noisy(
        ctx_true, sparsity=cfg.sparsity, noise_std_frac=cfg.noise_std_frac,
        attractor_std=LORENZ63_ATTRACTOR_STD, seed=seed,
    )

    t0 = time.time()
    if method == "ours_csdi":
        from methods.csdi_impute_adapter import set_csdi_checkpoint
        from methods.dynamics_impute import impute
        from methods.mi_lyap import robust_lyapunov
        from experiments.week2_modules.run_ablation import build_pipeline, evaluate_horizons
        if ckpt is not None:
            set_csdi_checkpoint(ckpt)
        ctx_filled = impute(obs, kind="csdi")
        lam_hat = robust_lyapunov(ctx_filled[:, 0], dt=dt, emb_dim=5, lag=2,
                                  trajectory_len=50, prefilter=True)
        cfg_pipe = dict(imp="csdi", tau="mi_lyap", gp="svgp", cp="lyap", growth="empirical")
        per_hgp, taus, tau_sec = build_pipeline(ctx_filled, cfg_pipe,
                                                 lam_hat=lam_hat * dt, seed=seed)
        metrics = evaluate_horizons(per_hgp, ctx_filled, future_true, taus,
                                    lam_hat=lam_hat * dt, dt=dt,
                                    cp_kind="lyap", growth="empirical")
    elif method == "panda":
        from baselines.panda_adapter import panda_forecast
        from methods.dynamics_impute import impute
        ctx_filled = impute(obs, kind="linear")
        forecast = panda_forecast(ctx_filled, pred_len=pred_len)
        horizons = [1, 4, 16, 64]
        metrics = {}
        for h in horizons:
            if h > forecast.shape[0]:
                continue
            err_sq = float(np.mean((forecast[h-1] - future_true[h-1]) ** 2))
            rmse = float(np.sqrt(err_sq))
            metrics[h] = dict(nrmse=rmse / LORENZ63_ATTRACTOR_STD, picp=None, mpiw=None)
        lam_hat = None
    else:
        raise ValueError(f"unknown method {method!r}")

    elapsed = time.time() - t0
    return dict(
        config=cfg.name, sparsity=cfg.sparsity, noise=cfg.noise_std_frac,
        n_eff_ratio=cfg.n_eff_ratio, seed=seed, method=method,
        lam_hat=float(lam_hat) if lam_hat is not None else None,
        metrics=metrics, elapsed_sec=elapsed,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="CSDI ckpt for ours_csdi")
    ap.add_argument("--n_seeds", type=int, default=5,
                    help="Run seeds 0..n_seeds-1 (ignored if --seeds is given)")
    ap.add_argument("--seeds", default=None,
                    help="Comma-sep list of seeds, e.g. '0,1' (overrides --n_seeds)")
    ap.add_argument("--methods", nargs="+", default=["ours_csdi", "panda"])
    ap.add_argument("--tag", default="ssgrid_v1")
    ap.add_argument("--only_method", default=None,
                    help="If set, only run this method (for parallel-by-GPU split)")
    args = ap.parse_args()

    methods = [args.only_method] if args.only_method else args.methods
    grid = build_grid()
    if args.seeds is not None:
        seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        seed_list = list(range(args.n_seeds))
    total = len(grid) * len(seed_list) * len(methods)

    print(f"=== Sparsity × Noise 2D grid (paper §5.X3) ===")
    print(f"  {len(grid)} configs × seeds={seed_list} × {len(methods)} methods = {total} runs")
    print(f"  s values     : {S_VALUES}")
    print(f"  σ/σ_attr vals: {SIGMA_VALUES}")
    print(f"  Grid (n_eff/n):")
    for cfg in grid:
        print(f"    {cfg.name:22s}  s={cfg.sparsity:.2f}  σ={cfg.noise_std_frac:.2f}  "
              f"n_eff/n={cfg.n_eff_ratio:.3f}")

    all_records = []
    t_global = time.time()
    run_idx = 0
    for cfg in grid:
        print(f"\n--- {cfg.name}  s={cfg.sparsity:.2f} σ={cfg.noise_std_frac:.2f} "
              f"(n_eff/n={cfg.n_eff_ratio:.3f}) ---")
        for method in methods:
            for seed in seed_list:
                run_idx += 1
                try:
                    rec = run_one(cfg, seed, method, args.ckpt)
                    all_records.append(rec)
                    h1 = rec["metrics"].get(1, {}) or {}
                    print(f"  [{run_idx:3d}/{total}][{method:10s}] seed={seed}  "
                          f"h=1 nrmse={h1.get('nrmse', 0):.3f}  "
                          f"elapsed={rec['elapsed_sec']:.1f}s")
                except Exception as e:
                    print(f"  [{run_idx:3d}/{total}][{method:10s}] seed={seed}  FAILED: {e}")
                    all_records.append(dict(config=cfg.name, seed=seed, method=method,
                                             error=str(e)))

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix_parts = []
    if args.only_method:
        suffix_parts.append(args.only_method)
    if args.seeds is not None:
        suffix_parts.append("seeds" + "-".join(str(s) for s in seed_list))
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""
    out_json = out_dir / f"{args.tag}{suffix}.json"
    out_json.write_text(json.dumps({
        "config_defs": [cfg.__dict__ for cfg in grid],
        "s_values": S_VALUES,
        "sigma_values": SIGMA_VALUES,
        "seeds": seed_list,
        "methods": methods,
        "ckpt": args.ckpt,
        "records": all_records,
    }, indent=2, default=str))
    print(f"\n[saved] {out_json}  (total {time.time()-t_global:.0f}s)")


if __name__ == "__main__":
    main()
