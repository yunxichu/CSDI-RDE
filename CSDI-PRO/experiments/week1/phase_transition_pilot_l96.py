"""Lorenz96 N=20 phase-transition main figure (§5.2 extension for cross-system evidence).

Mirrors phase_transition_pilot_v2.py but on Lorenz96 at N=20, F=8.
Purpose: verify that the sparsity-noise phase transition observed on Lorenz63
(§5.2 Fig 1) also appears on a higher-dimensional system — external validity
for Theorem 2 / Proposition 5.

Ours uses AR-Kalman M1 (apples-to-apples with §5.2 Lorenz63 main figure;
CSDI-on-L96 is deferred to future work since M1 needs L96 retraining).

Methods compared: ours (AR-K) / panda / parrot. Chronos omitted (univariate
tokenizer poorly suited to N=20). Persist optional.

Run:
  CUDA_VISIBLE_DEVICES=2 python -m experiments.week1.phase_transition_pilot_l96 \
      --N 20 --n_seeds 3 --methods ours panda parrot --tag l96_N20_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from experiments.week1.baselines import (
    context_parroting_forecast,
    persistence_forecast,
)
from experiments.week1.full_pipeline_rollout import full_pipeline_forecast

try:
    from baselines.panda_adapter import panda_forecast
    _HAS_PANDA = True
except ImportError:
    _HAS_PANDA = False

from experiments.week1.lorenz63_utils import (
    HarshnessScenario,
    make_sparse_noisy,
    valid_prediction_time,
)
from experiments.week1.lorenz96_utils import (
    LORENZ96_LYAP_F8,
    LORENZ96_F_DEFAULT,
    integrate_lorenz96,
    lorenz96_attractor_std,
)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "experiments" / "week1" / "results"
FIG_DIR = REPO_ROOT / "experiments" / "week1" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["ours_csdi", "ours", "panda", "parrot", "persist"]

# Reuse L63 S0-S6 (s, σ) harshness levels — the sparsity & noise levels are
# system-agnostic by design (all reported as fraction of attractor std).
L96_SCENARIOS: list[HarshnessScenario] = [
    HarshnessScenario("S0", 0.00, 0.00),
    HarshnessScenario("S1", 0.20, 0.10),
    HarshnessScenario("S2", 0.40, 0.30),
    HarshnessScenario("S3", 0.60, 0.50),
    HarshnessScenario("S4", 0.75, 0.80),
    HarshnessScenario("S5", 0.90, 1.20),
    HarshnessScenario("S6", 0.95, 1.50),
]


def run_pilot(
    N: int,
    n_seeds: int,
    n_ctx: int,
    pred_len: int,
    dt: float,
    spinup: int,
    methods: list[str],
    F: float = LORENZ96_F_DEFAULT,
    seed_offset: int = 0,
) -> tuple[list[dict], float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attr_std = lorenz96_attractor_std(N=N, F=F)
    lyap = LORENZ96_LYAP_F8
    print(
        f"[pilot-l96] device={device} N={N} F={F} attr_std={attr_std:.3f} "
        f"lyap={lyap} n_seeds={n_seeds} n_ctx={n_ctx} pred_len={pred_len} dt={dt} "
        f"methods={methods}"
    )

    records: list[dict] = []
    for i in range(n_seeds):
        seed = seed_offset + i
        traj = integrate_lorenz96(n_ctx + pred_len, N=N, F=F, dt=dt,
                                   spinup=spinup, seed=seed)
        ctx_true = traj[:n_ctx]
        future_true = traj[n_ctx:]

        for sc in L96_SCENARIOS:
            observed, mask = make_sparse_noisy(
                ctx_true,
                sparsity=sc.sparsity,
                noise_std_frac=sc.noise_std_frac,
                attractor_std=attr_std,
                seed=1000 * seed + hash(sc.name) % 10000,
            )
            # Linear-interp fill (same treatment as L63 main figure)
            from experiments.week1.lorenz63_utils import linear_interp_fill
            ctx_filled = linear_interp_fill(observed)

            for method in methods:
                t0 = time.time()
                try:
                    if method == "parrot":
                        mean = context_parroting_forecast(ctx_filled, pred_len=pred_len)
                    elif method == "persist":
                        mean = persistence_forecast(ctx_filled, pred_len)
                    elif method == "ours":
                        mean = full_pipeline_forecast(
                            observed, pred_len=pred_len, seed=seed,
                            bayes_calls=10, n_epochs=150,
                        )
                    elif method == "ours_csdi":
                        # requires set_csdi_checkpoint() called upstream (via --csdi_ckpt)
                        mean = full_pipeline_forecast(
                            observed, pred_len=pred_len, seed=seed,
                            imp_kind="csdi", bayes_calls=10, n_epochs=150,
                        )
                    elif method == "panda":
                        if not _HAS_PANDA:
                            raise RuntimeError("panda adapter not available")
                        mean = panda_forecast(ctx_filled, pred_len=pred_len)
                    else:
                        raise ValueError(method)
                    t_infer = time.time() - t0

                    vpt03 = valid_prediction_time(future_true, mean, dt=dt,
                                                   lyap=lyap, threshold=0.3,
                                                   attractor_std=attr_std)
                    vpt05 = valid_prediction_time(future_true, mean, dt=dt,
                                                   lyap=lyap, threshold=0.5,
                                                   attractor_std=attr_std)
                    vpt10 = valid_prediction_time(future_true, mean, dt=dt,
                                                   lyap=lyap, threshold=1.0,
                                                   attractor_std=attr_std)
                    rmse_norm = float(
                        np.sqrt(((future_true[: min(100, pred_len)]
                                  - mean[: min(100, pred_len)]) ** 2).mean())
                        / attr_std
                    )
                    err_str = None
                except Exception as e:
                    t_infer = time.time() - t0
                    vpt03 = vpt05 = vpt10 = rmse_norm = float("nan")
                    err_str = str(e)[:200]

                rec = dict(
                    seed=seed,
                    scenario=sc.name,
                    method=method,
                    sparsity=sc.sparsity,
                    noise_std_frac=sc.noise_std_frac,
                    keep_frac=float(mask.mean()),
                    vpt03=float(vpt03),
                    vpt05=float(vpt05),
                    vpt10=float(vpt10),
                    rmse_norm_first100=rmse_norm,
                    infer_time_s=t_infer,
                    error=err_str,
                )
                records.append(rec)
                if err_str:
                    print(f"  seed={seed} {sc.name} {method:8s}  FAILED: {err_str}")
                else:
                    print(
                        f"  seed={seed} {sc.name} {method:8s} keep={mask.mean():.2f} "
                        f"σ={sc.noise_std_frac:.2f}  VPT@0.3={vpt03:5.2f}  "
                        f"VPT@1.0={vpt10:5.2f}  rmse/std={rmse_norm:.3f}  t={t_infer:.1f}s"
                    )
    return records, attr_std


def summarize(records: list[dict]) -> dict:
    import collections

    acc: dict[tuple[str, str], dict] = collections.defaultdict(
        lambda: {"vpt03": [], "vpt05": [], "vpt10": [], "rmse": [], "keep": [],
                 "sparsity": None, "noise": None}
    )
    for r in records:
        if r.get("error"):
            continue
        k = acc[(r["method"], r["scenario"])]
        k["vpt03"].append(r["vpt03"])
        k["vpt05"].append(r["vpt05"])
        k["vpt10"].append(r["vpt10"])
        k["rmse"].append(r["rmse_norm_first100"])
        k["keep"].append(r["keep_frac"])
        k["sparsity"] = r["sparsity"]
        k["noise"] = r["noise_std_frac"]

    summary: dict = {}
    for (method, sc), d in acc.items():
        if not d["vpt10"]:
            continue
        summary.setdefault(method, {})[sc] = {
            "sparsity": d["sparsity"],
            "noise_std_frac": d["noise"],
            "keep_mean": float(np.mean(d["keep"])),
            "vpt03_mean": float(np.mean(d["vpt03"])),
            "vpt03_std": float(np.std(d["vpt03"])),
            "vpt05_mean": float(np.mean(d["vpt05"])),
            "vpt05_std": float(np.std(d["vpt05"])),
            "vpt10_mean": float(np.mean(d["vpt10"])),
            "vpt10_std": float(np.std(d["vpt10"])),
            "rmse_mean": float(np.mean(d["rmse"])),
            "rmse_std": float(np.std(d["rmse"])),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20, help="Lorenz96 ring dim")
    ap.add_argument("--F", type=float, default=LORENZ96_F_DEFAULT)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--seed_offset", type=int, default=0,
                    help="Start seeds at this offset (e.g. --seed_offset 3 --n_seeds 2 runs seeds 3,4)")
    ap.add_argument("--n_ctx", type=int, default=512)
    ap.add_argument("--pred_len", type=int, default=128)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--methods", nargs="+",
                    default=["ours", "panda", "parrot", "persist"])
    ap.add_argument("--tag", default="l96_N20_v1")
    ap.add_argument("--csdi_ckpt", default=None,
                    help="L96 CSDI checkpoint path (required if 'ours_csdi' in --methods)")
    args = ap.parse_args()

    if "ours_csdi" in args.methods:
        if not args.csdi_ckpt:
            raise SystemExit("--csdi_ckpt is required when 'ours_csdi' is in --methods")
        from methods.csdi_impute_adapter import set_csdi_checkpoint
        set_csdi_checkpoint(args.csdi_ckpt)
        print(f"[pilot-l96] CSDI ckpt loaded: {args.csdi_ckpt}")

    records, attr_std = run_pilot(
        N=args.N, F=args.F,
        n_seeds=args.n_seeds, n_ctx=args.n_ctx, pred_len=args.pred_len,
        dt=args.dt, spinup=2000, methods=args.methods,
        seed_offset=args.seed_offset,
    )
    summary = summarize(records)

    out_json = OUT_DIR / f"pt_l96_{args.tag}.json"
    out_json.write_text(
        json.dumps(
            dict(
                config=vars(args),
                records=records,
                summary=summary,
                meta=dict(attractor_std=attr_std, lyap=LORENZ96_LYAP_F8,
                          N=args.N, F=args.F),
            ),
            indent=2,
        )
    )
    print(f"[pilot-l96] records saved to {out_json}")

    scenario_names = [sc.name for sc in L96_SCENARIOS]
    print(f"\n[verdict] VPT@1.0 by (method, scenario) on Lorenz96 N={args.N}:")
    for method in METHOD_ORDER:
        if method not in summary:
            continue
        line = f"  {method:8s}"
        for s in scenario_names:
            cell = summary[method].get(s, None)
            val = cell["vpt10_mean"] if cell else float("nan")
            line += f"  {s}={val:4.2f}"
        print(line)

    # Quick phase-transition check: is panda S0→S3 > 50% drop?
    if "panda" in summary and "S0" in summary["panda"] and "S3" in summary["panda"]:
        s0 = summary["panda"]["S0"]["vpt10_mean"]
        s3 = summary["panda"]["S3"]["vpt10_mean"]
        if s0 > 0.5 and s3 < 0.5 * s0:
            print(
                f"\n  -> PANDA phase transition on L96: S0={s0:.2f} → S3={s3:.2f} "
                f"({(1 - s3 / max(s0, 1e-6)) * 100:.0f}% drop). "
                "Cross-system universal claim HOLDS."
            )
        else:
            print(
                f"\n  -> PANDA does NOT show crisp L96 phase transition "
                f"(S0={s0:.2f} S3={s3:.2f}). Title must narrow to Lorenz63 case study."
            )


if __name__ == "__main__":
    main()
