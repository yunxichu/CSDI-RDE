"""Rössler phase-transition evaluation (§5.10 — weak chaos clean-regime showcase).

λ_1 ≈ 0.07 (very weak) → long Lyapunov horizon, high VPT ceiling even under
sparsity+noise. This complements the hard-chaos L96 section by showing our
pipeline gracefully scales down to easy problems.
"""
from __future__ import annotations

import argparse
import json
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
    HarshnessScenario, make_sparse_noisy, valid_prediction_time, linear_interp_fill,
)
from systems.rossler import (
    integrate_rossler, ROSSLER_LYAP, ROSSLER_ATTRACTOR_STD, ROSSLER_DT,
)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "experiments" / "week1" / "results"
FIG_DIR = REPO_ROOT / "experiments" / "week1" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS: list[HarshnessScenario] = [
    HarshnessScenario("S0", 0.00, 0.00),
    HarshnessScenario("S1", 0.20, 0.10),
    HarshnessScenario("S2", 0.40, 0.30),
    HarshnessScenario("S3", 0.60, 0.50),
    HarshnessScenario("S4", 0.75, 0.80),
    HarshnessScenario("S5", 0.90, 1.20),
    HarshnessScenario("S6", 0.95, 1.50),
]


def run_pilot(n_seeds, n_ctx, pred_len, dt, spinup, methods, seed_offset=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attr_std = ROSSLER_ATTRACTOR_STD
    lyap = ROSSLER_LYAP
    print(
        f"[pilot-rossler] device={device} dt={dt} attr_std={attr_std:.3f} "
        f"lyap={lyap} n_seeds={n_seeds} n_ctx={n_ctx} pred_len={pred_len} "
        f"methods={methods}",
        flush=True,
    )
    records = []
    for i in range(n_seeds):
        seed = seed_offset + i
        traj = integrate_rossler(n_ctx + pred_len, dt=dt, spinup=spinup, seed=seed)
        ctx_true, future_true = traj[:n_ctx], traj[n_ctx:]
        for sc in SCENARIOS:
            observed, mask = make_sparse_noisy(
                ctx_true, sparsity=sc.sparsity, noise_std_frac=sc.noise_std_frac,
                attractor_std=attr_std, seed=1000 * seed + hash(sc.name) % 10000,
            )
            ctx_filled = linear_interp_fill(observed)
            for method in methods:
                t0 = time.time()
                try:
                    if method == "parrot":
                        mean = context_parroting_forecast(ctx_filled, pred_len=pred_len)
                    elif method == "persist":
                        mean = persistence_forecast(ctx_filled, pred_len)
                    elif method == "ours_csdi_svgp":
                        mean = full_pipeline_forecast(
                            observed, pred_len=pred_len, seed=seed,
                            imp_kind="csdi", bayes_calls=10, n_epochs=150,
                            backbone="svgp",
                        )
                    elif method == "ours_csdi_deepedm":
                        mean = full_pipeline_forecast(
                            observed, pred_len=pred_len, seed=seed,
                            imp_kind="csdi", bayes_calls=10, backbone="deepedm",
                        )
                    elif method == "panda":
                        if not _HAS_PANDA:
                            raise RuntimeError("panda adapter not available")
                        mean = panda_forecast(ctx_filled, pred_len=pred_len)
                    else:
                        raise ValueError(method)
                    t_infer = time.time() - t0
                    vpt03 = valid_prediction_time(future_true, mean, dt=dt, lyap=lyap,
                                                   threshold=0.3, attractor_std=attr_std)
                    vpt05 = valid_prediction_time(future_true, mean, dt=dt, lyap=lyap,
                                                   threshold=0.5, attractor_std=attr_std)
                    vpt10 = valid_prediction_time(future_true, mean, dt=dt, lyap=lyap,
                                                   threshold=1.0, attractor_std=attr_std)
                    rmse_norm = float(
                        np.sqrt(((future_true[:min(200, pred_len)] -
                                   mean[:min(200, pred_len)]) ** 2).mean()) / attr_std
                    )
                    err_str = None
                except Exception as e:
                    t_infer = time.time() - t0
                    vpt03 = vpt05 = vpt10 = rmse_norm = float("nan")
                    err_str = str(e)[:200]
                rec = dict(
                    seed=seed, scenario=sc.name, method=method,
                    sparsity=sc.sparsity, noise_std_frac=sc.noise_std_frac,
                    keep_frac=float(mask.mean()),
                    vpt03=float(vpt03), vpt05=float(vpt05), vpt10=float(vpt10),
                    rmse_norm_first200=rmse_norm, infer_time_s=t_infer, error=err_str,
                )
                records.append(rec)
                if err_str:
                    print(f"  seed={seed} {sc.name} {method:20s}  FAILED: {err_str}", flush=True)
                else:
                    print(f"  seed={seed} {sc.name} {method:20s} keep={mask.mean():.2f} "
                          f"σ={sc.noise_std_frac:.2f}  VPT@0.3={vpt03:5.2f}  "
                          f"VPT@1.0={vpt10:5.2f}  rmse/std={rmse_norm:.3f}  t={t_infer:.1f}s",
                          flush=True)
    return records, attr_std


def summarize(records):
    import collections
    acc = collections.defaultdict(lambda: {"vpt03": [], "vpt05": [], "vpt10": [],
                                             "rmse": [], "keep": [],
                                             "sparsity": None, "noise": None})
    for r in records:
        if r.get("error"):
            continue
        k = acc[(r["method"], r["scenario"])]
        k["vpt03"].append(r["vpt03"]); k["vpt05"].append(r["vpt05"])
        k["vpt10"].append(r["vpt10"]); k["rmse"].append(r["rmse_norm_first200"])
        k["keep"].append(r["keep_frac"])
        k["sparsity"] = r["sparsity"]; k["noise"] = r["noise_std_frac"]
    summary = {}
    for (method, sc), d in acc.items():
        if not d["vpt10"]:
            continue
        summary.setdefault(method, {})[sc] = {
            "sparsity": d["sparsity"], "noise_std_frac": d["noise"],
            "keep_mean": float(np.mean(d["keep"])),
            "vpt03_mean": float(np.mean(d["vpt03"])), "vpt03_std": float(np.std(d["vpt03"])),
            "vpt05_mean": float(np.mean(d["vpt05"])), "vpt05_std": float(np.std(d["vpt05"])),
            "vpt10_mean": float(np.mean(d["vpt10"])), "vpt10_std": float(np.std(d["vpt10"])),
            "rmse_mean": float(np.mean(d["rmse"])), "rmse_std": float(np.std(d["rmse"])),
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--n_ctx", type=int, default=512)
    ap.add_argument("--pred_len", type=int, default=1024,
                    help="~7 Λ at dt=0.1, λ=0.07. Increase for longer horizon")
    ap.add_argument("--dt", type=float, default=ROSSLER_DT)
    ap.add_argument("--methods", nargs="+",
                    default=["ours_csdi_svgp", "ours_csdi_deepedm",
                             "panda", "parrot", "persist"])
    ap.add_argument("--tag", default="rossler_v1")
    ap.add_argument("--csdi_ckpt", default=None)
    args = ap.parse_args()
    if any(m.startswith("ours_csdi") for m in args.methods):
        if not args.csdi_ckpt:
            raise SystemExit("--csdi_ckpt required")
        from methods.csdi_impute_adapter import set_csdi_checkpoint, set_csdi_attractor_std
        set_csdi_checkpoint(args.csdi_ckpt)
        set_csdi_attractor_std(ROSSLER_ATTRACTOR_STD)
        print(f"[pilot-rossler] CSDI ckpt loaded: {args.csdi_ckpt}", flush=True)

    records, attr_std = run_pilot(
        args.n_seeds, args.n_ctx, args.pred_len, args.dt,
        spinup=2000, methods=args.methods, seed_offset=args.seed_offset,
    )
    summary = summarize(records)
    out_json = OUT_DIR / f"pt_rossler_{args.tag}.json"
    out_json.write_text(
        json.dumps({"summary": summary, "records": records,
                    "attractor_std": attr_std, "lyap": ROSSLER_LYAP,
                    "dt": args.dt, "n_ctx": args.n_ctx, "pred_len": args.pred_len},
                   indent=2)
    )
    print(f"[pilot-rossler] saved {out_json}", flush=True)
    print("\n[verdict] VPT@1.0 by (method, scenario):")
    for method, cells in summary.items():
        row = [f"  {method:20s}"] + [
            f"{cells[sc.name]['vpt10_mean']:.2f}" if sc.name in cells else "  —"
            for sc in SCENARIOS
        ]
        print("  ".join(row))


if __name__ == "__main__":
    main()
