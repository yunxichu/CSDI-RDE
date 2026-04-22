"""n_eff unified parameter verification (paper §5.X2, REFACTOR_PLAN §6.2).

Tests whether VPT is a function of n_eff/n only, not (s, σ) separately.

Design: at fixed n_eff/n = 0.3, vary (s, σ) combinations:
  - (s=0.6,  σ=0.5 )  — the canonical S3 point (mixed sparsity+noise)
  - (s=0.5,  σ=0.77)  — less sparse, more noise
  - (s=0.7,  σ=0.00)  — pure sparsity, no noise
  - (s=0.0,  σ=1.53)  — pure noise, no sparsity
For all four, n_eff/n = (1-s)/(1+σ²/σ_attr²) = 0.32 ± 0.02 (approximately constant).

Expected outcomes (predictions from §4 theory):
  - Ours (manifold): VPT roughly constant across 4 configs (n_eff-driven smooth退化)
  - Panda (ambient): pure-sparsity config (s=0.7) worse than mixed (s=0.6); pure-noise
    config (s=0.0) may behave differently because tokenizer OOD triggers on sparsity
    not noise. So Panda's VPT should NOT collapse to one curve → ambient predictor
    depends on (s, σ) individually, not just n_eff.

Even a partial result (ours collapses, Panda doesn't) is a win — directly supports
Theorem 2's distinction between manifold (n_eff-only) and ambient (OOD-sensitive).

Usage (4 configs × 5 seeds × 2 methods = 40 runs, ~30 min on 1 GPU):
    CUDA_VISIBLE_DEVICES=0 python -m experiments.week1.run_neff_unified_ablation \\
        --ckpt experiments/week2_modules/ckpts/dyn_csdi_full_v6_center_ep20.pt \\
        --n_seeds 5 --methods ours_csdi panda --tag neff_unified_v1
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class NeffConfig:
    name: str
    sparsity: float
    noise_std_frac: float  # σ / σ_attr

    @property
    def n_eff_ratio(self) -> float:
        return (1 - self.sparsity) / (1 + self.noise_std_frac ** 2)


# Four configurations targeting n_eff/n ≈ 0.32 (S3's value).
# Verified analytically: (1-s)/(1+σ²) for each:
#   U1: (1-0.6)/(1+0.25) = 0.40/1.25 = 0.320
#   U2: (1-0.5)/(1+0.59) = 0.50/1.59 = 0.314
#   U3: (1-0.7)/(1+0.00) = 0.30/1.00 = 0.300
#   U4: (1-0.0)/(1+2.34) = 1.00/3.34 = 0.299
NEFF_CONFIGS: list[NeffConfig] = [
    NeffConfig("U1_mixed_S3",     0.60, 0.50),
    NeffConfig("U2_mixed_alt",    0.50, 0.77),
    NeffConfig("U3_pure_sparse",  0.70, 0.00),
    NeffConfig("U4_pure_noise",   0.00, 1.53),
]


def run_config(cfg: NeffConfig, seed: int, method: str, ckpt: str | None,
               n_ctx: int = 1200, pred_len: int = 128, dt: float = 0.025):
    """Run one config-seed-method combination. Returns record dict."""
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
        # Use the full manifold pipeline with CSDI M1
        from methods.csdi_impute_adapter import set_csdi_checkpoint
        from methods.dynamics_impute import impute
        from methods.mi_lyap import mi_lyap_bayes_tau, robust_lyapunov
        from experiments.week2_modules.run_ablation import (
            L_EMBED, TAU_MAX, BAYES_CALLS, build_pipeline, evaluate_horizons,
        )
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
        # Placeholder: wire in panda adapter. The adapter expects linear-interp context.
        from baselines.panda_adapter import panda_forecast
        # Linear interp NaNs for panda
        from methods.dynamics_impute import impute
        ctx_filled = impute(obs, kind="linear")
        forecast = panda_forecast(ctx_filled, horizon=pred_len)
        # Compute NRMSE/VPT directly
        from metrics.chaos_metrics import nrmse as chaos_nrmse
        horizons = [1, 4, 16, 64]
        metrics = {}
        for h in horizons:
            if h > forecast.shape[0]:
                continue
            rmse = float(np.sqrt(np.mean((forecast[h-1] - future_true[h-1]) ** 2)))
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
    ap.add_argument("--ckpt", default=None, help="CSDI ckpt for ours_csdi method")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--methods", nargs="+", default=["ours_csdi", "panda"])
    ap.add_argument("--tag", default="neff_unified_v1")
    args = ap.parse_args()

    print(f"=== n_eff unified parameter ablation (paper §5.X2) ===")
    print(f"  4 configs × {args.n_seeds} seeds × {len(args.methods)} methods = "
          f"{4 * args.n_seeds * len(args.methods)} runs")
    print(f"  n_eff/n targets:")
    for cfg in NEFF_CONFIGS:
        print(f"    {cfg.name:18s} s={cfg.sparsity:.2f}  σ/σ_attr={cfg.noise_std_frac:.2f}  "
              f"n_eff/n={cfg.n_eff_ratio:.3f}")

    all_records = []
    for cfg in NEFF_CONFIGS:
        print(f"\n--- {cfg.name}  s={cfg.sparsity} σ={cfg.noise_std_frac} ---")
        for method in args.methods:
            for seed in range(args.n_seeds):
                try:
                    rec = run_config(cfg, seed, method, args.ckpt)
                    all_records.append(rec)
                    h1 = rec["metrics"].get(1, {}) or {}
                    print(f"  [{method:10s}] seed={seed}  h=1 nrmse={h1.get('nrmse', 0):.3f}  "
                          f"elapsed={rec['elapsed_sec']:.1f}s")
                except Exception as e:
                    print(f"  [{method:10s}] seed={seed}  FAILED: {e}")
                    all_records.append(dict(config=cfg.name, seed=seed, method=method,
                                             error=str(e)))

    from pathlib import Path
    OUT_DIR = Path(__file__).resolve().parent.parent / "week1" / "results"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"{args.tag}.json"
    out_json.write_text(json.dumps({
        "config_defs": [cfg.__dict__ for cfg in NEFF_CONFIGS],
        "n_seeds": args.n_seeds,
        "methods": args.methods,
        "ckpt": args.ckpt,
        "records": all_records,
    }, indent=2, default=str))
    print(f"\n[saved] {out_json}")


if __name__ == "__main__":
    main()
