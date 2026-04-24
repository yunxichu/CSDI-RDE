"""Mackey-Glass phase-transition evaluation (§5.8 — canonical delay-embedding testbed).

Mackey-Glass is the canonical test for delay-embedding methods: the system is
*defined* by a delay ODE, so our delay-coordinate pipeline has direct
inductive-bias alignment. The 3 M3 backbones (SVGP / DeepEDM / FNO) are each
paired with CSDI M1 and compared head-to-head with Panda-72M / Parrot /
Persist on the same S0-S6 harshness grid.

Run:
  CUDA_VISIBLE_DEVICES=1 python -m experiments.week1.phase_transition_pilot_mackey_glass \
      --n_seeds 5 --csdi_ckpt experiments/week2_modules/ckpts/dyn_csdi_mg_full_vales_best.pt \
      --methods ours_csdi_svgp ours_csdi_deepedm ours_csdi_fno panda parrot persist
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
    HarshnessScenario,
    make_sparse_noisy,
    valid_prediction_time,
    linear_interp_fill,
)
from systems.mackey_glass import (
    integrate_mackey_glass,
    MACKEY_GLASS_LYAP,
    MACKEY_GLASS_ATTRACTOR_STD,
    MACKEY_GLASS_TAU,
    MACKEY_GLASS_DT,
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


def run_pilot(
    n_seeds: int, n_ctx: int, pred_len: int, dt: float, spinup: int,
    methods: list[str], seed_offset: int = 0,
) -> tuple[list[dict], float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attr_std = MACKEY_GLASS_ATTRACTOR_STD
    lyap = MACKEY_GLASS_LYAP
    print(
        f"[pilot-mg] device={device} tau={MACKEY_GLASS_TAU} dt={dt} "
        f"attr_std={attr_std:.3f} lyap={lyap} n_seeds={n_seeds} "
        f"n_ctx={n_ctx} pred_len={pred_len} methods={methods}",
        flush=True,
    )

    records: list[dict] = []
    for i in range(n_seeds):
        seed = seed_offset + i
        traj = integrate_mackey_glass(n_ctx + pred_len, dt=dt, spinup=spinup, seed=seed)
        ctx_true = traj[:n_ctx]
        future_true = traj[n_ctx:]

        for sc in SCENARIOS:
            observed, mask = make_sparse_noisy(
                ctx_true,
                sparsity=sc.sparsity,
                noise_std_frac=sc.noise_std_frac,
                attractor_std=attr_std,
                seed=1000 * seed + hash(sc.name) % 10000,
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
                    elif method == "ours_csdi_fno":
                        mean = full_pipeline_forecast(
                            observed, pred_len=pred_len, seed=seed,
                            imp_kind="csdi", bayes_calls=10, backbone="fno",
                        )
                    elif method == "ours_ark_deepedm":
                        mean = full_pipeline_forecast(
                            observed, pred_len=pred_len, seed=seed,
                            bayes_calls=10, backbone="deepedm",
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
                        np.sqrt(((future_true[: min(200, pred_len)]
                                  - mean[: min(200, pred_len)]) ** 2).mean())
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
                    rmse_norm_first200=rmse_norm,
                    infer_time_s=t_infer,
                    error=err_str,
                )
                records.append(rec)
                if err_str:
                    print(f"  seed={seed} {sc.name} {method:20s}  FAILED: {err_str}", flush=True)
                else:
                    print(
                        f"  seed={seed} {sc.name} {method:20s} keep={mask.mean():.2f} "
                        f"σ={sc.noise_std_frac:.2f}  VPT@0.3={vpt03:5.2f}  "
                        f"VPT@1.0={vpt10:5.2f}  rmse/std={rmse_norm:.3f}  t={t_infer:.1f}s",
                        flush=True,
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
        k["rmse"].append(r["rmse_norm_first200"])
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
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--n_ctx", type=int, default=512)
    ap.add_argument("--pred_len", type=int, default=1024,
                    help="larger than L63/L96 because MG λ is weaker (1024 × 1.0 × 0.006 ≈ 6 Λ)")
    ap.add_argument("--dt", type=float, default=MACKEY_GLASS_DT)
    ap.add_argument("--methods", nargs="+",
                    default=["ours_csdi_svgp", "ours_csdi_deepedm", "ours_csdi_fno",
                             "panda", "parrot", "persist"])
    ap.add_argument("--tag", default="mg_v1")
    ap.add_argument("--csdi_ckpt", default=None,
                    help="MG CSDI checkpoint path (required if any ours_csdi_* method)")
    args = ap.parse_args()

    needs_csdi = any(m.startswith("ours_csdi") for m in args.methods)
    if needs_csdi:
        if not args.csdi_ckpt:
            raise SystemExit("--csdi_ckpt is required when an ours_csdi_* method is selected")
        from methods.csdi_impute_adapter import set_csdi_checkpoint, set_csdi_attractor_std
        set_csdi_checkpoint(args.csdi_ckpt)
        set_csdi_attractor_std(MACKEY_GLASS_ATTRACTOR_STD)
        print(f"[pilot-mg] CSDI ckpt loaded: {args.csdi_ckpt}", flush=True)
        print(f"[pilot-mg] CSDI inference attr_std: {MACKEY_GLASS_ATTRACTOR_STD:.4f}", flush=True)

    records, attr_std = run_pilot(
        n_seeds=args.n_seeds, n_ctx=args.n_ctx, pred_len=args.pred_len,
        dt=args.dt, spinup=2000, methods=args.methods,
        seed_offset=args.seed_offset,
    )
    summary = summarize(records)

    out_json = OUT_DIR / f"pt_mg_{args.tag}.json"
    out_json.write_text(
        json.dumps(
            {"summary": summary, "records": records,
             "attractor_std": attr_std, "lyap": MACKEY_GLASS_LYAP,
             "tau": MACKEY_GLASS_TAU, "dt": args.dt,
             "n_ctx": args.n_ctx, "pred_len": args.pred_len},
            indent=2,
        )
    )
    print(f"[pilot-mg] records saved to {out_json}", flush=True)

    print("\n[verdict] VPT@1.0 by (method, scenario) on Mackey-Glass τ=17:")
    for method, cells in summary.items():
        row = [f"  {method:20s}"] + [
            f"{cells[sc.name]['vpt10_mean']:.2f}" if sc.name in cells else "  — "
            for sc in SCENARIOS
        ]
        print("  ".join(row))


if __name__ == "__main__":
    main()
