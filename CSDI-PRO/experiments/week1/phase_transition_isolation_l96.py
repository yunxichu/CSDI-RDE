"""Isolation ablation on L96 N=20 — the "reviewer killer" experiment for the
2026-04-26 paper pivot (see deliverable/PIVOT.md).

Question: when foundation forecasters (Panda) collapse beyond the tokenizer-OOD
threshold, is it because the *preprocessing* (linear-fill of sparse obs) creates
non-physical patches that fall out of the tokenizer's training distribution, or
because the *ambient/tokenized forecaster* itself is fundamentally unsuited for
sparse-noisy chaos?

Design: 2 forecasters × 3 imputers, 6 cells:
                 │ Panda (ambient/tokenized) │ DeepEDM (delay-manifold)
    ─────────────┼───────────────────────────┼──────────────────────────
    linear-fill  │ panda_linear (= baseline) │ deepedm_linear (NEW)
    Kalman-fill  │ panda_kalman (NEW)        │ deepedm_kalman (≈ ours_deepedm)
    CSDI-fill    │ panda_csdi   (NEW)        │ deepedm_csdi   (= ours_csdi_deepedm)

Read-out:
  * If panda_csdi ≈ panda_linear: preprocessing is NOT the failure source,
    ambient-tokenizer IS — the mechanism claim ("delay-manifold avoids the OOD
    channel") is well-supported.
  * If panda_csdi ≫ panda_linear and ≈ deepedm_csdi: corruption-aware
    preprocessing is the lever — paper pivots toward "imputation is the trick".
  * deepedm_linear vs deepedm_csdi: tests delay-manifold robustness to imputer
    quality.

Scenarios: S2/S3/S4/S5 (the phase-transition band — S0/S1 clean; S6 mostly zeros).

Run:
  CUDA_VISIBLE_DEVICES=2 python -u -m experiments.week1.phase_transition_isolation_l96 \\
      --n_seeds 5 \\
      --csdi_ckpt experiments/week2_modules/ckpts/dyn_csdi_l96_N20_full_c192_vales_best.pt \\
      --tag iso_l96N20_v1
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

from experiments.week1.full_pipeline_rollout import full_pipeline_forecast
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
from methods.dynamics_impute import impute

try:
    from baselines.panda_adapter import panda_forecast
    _HAS_PANDA = True
except ImportError:
    _HAS_PANDA = False

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "experiments" / "week1" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Phase-transition band — clean and S6-collapse scenarios skipped to keep
# isolation matrix focused on the band where the action is.
ISOLATION_SCENARIOS: list[HarshnessScenario] = [
    HarshnessScenario("S2", 0.40, 0.30),
    HarshnessScenario("S3", 0.60, 0.50),
    HarshnessScenario("S4", 0.75, 0.80),
    HarshnessScenario("S5", 0.90, 1.20),
]
SCENARIO_SEEDS = {"S2": 2002, "S3": 2003, "S4": 2004, "S5": 2005}

# (imputer_kind, forecaster_kind) — forecaster ∈ {"panda", "deepedm"}
ISOLATION_CELLS: list[tuple[str, str, str]] = [
    # (label, imputer, forecaster)
    ("panda_linear",   "linear",    "panda"),
    ("panda_kalman",   "ar_kalman", "panda"),
    ("panda_csdi",     "csdi",      "panda"),
    ("deepedm_linear", "linear",    "deepedm"),
    ("deepedm_kalman", "ar_kalman", "deepedm"),
    ("deepedm_csdi",   "csdi",      "deepedm"),
]


def run_one_cell(
    label: str,
    imputer: str,
    forecaster: str,
    observed: np.ndarray,
    pred_len: int,
    seed: int,
    cached_fills: dict[str, np.ndarray],
    sigma_override: float | None = None,
) -> tuple[np.ndarray | None, str | None, float]:
    """Returns (mean_forecast, error_str, infer_time_s)."""
    t0 = time.time()
    try:
        if forecaster == "panda":
            if not _HAS_PANDA:
                raise RuntimeError("panda adapter not available")
            # Panda is purely a function of the filled context — reuse cache.
            filled = cached_fills.get(imputer)
            if filled is None:
                kwargs = {"sigma_override": sigma_override} if imputer == "csdi" else {}
                filled = impute(observed, kind=imputer, **kwargs)
                cached_fills[imputer] = filled
            mean = panda_forecast(filled, pred_len=pred_len)
        elif forecaster == "deepedm":
            # full_pipeline_forecast does its own imputation internally — pass raw obs.
            impute_kwargs = {"sigma_override": sigma_override} if imputer == "csdi" else {}
            mean = full_pipeline_forecast(
                observed, pred_len=pred_len, seed=seed,
                imp_kind=imputer, bayes_calls=10, backbone="deepedm",
                impute_kwargs=impute_kwargs,
            )
        else:
            raise ValueError(forecaster)
        return mean, None, time.time() - t0
    except Exception as e:
        return None, str(e)[:200], time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--F", type=float, default=LORENZ96_F_DEFAULT)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--n_ctx", type=int, default=512)
    ap.add_argument("--pred_len", type=int, default=128)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--csdi_ckpt", required=True,
                    help="L96 N=20 CSDI checkpoint (e.g. dyn_csdi_l96_N20_full_c192_vales_best.pt)")
    ap.add_argument("--tag", default="iso_l96N20_v1")
    ap.add_argument("--cells", nargs="+", default=None,
                    help="Subset of cell labels to run. Default: all 6.")
    args = ap.parse_args()

    cells = ISOLATION_CELLS
    if args.cells is not None:
        wanted = set(args.cells)
        cells = [c for c in ISOLATION_CELLS if c[0] in wanted]
        if not cells:
            raise SystemExit(f"--cells {args.cells} matched none of {[c[0] for c in ISOLATION_CELLS]}")

    # CSDI ckpt setup (needed if any csdi cell is selected)
    if any(c[1] == "csdi" for c in cells):
        from methods.csdi_impute_adapter import set_csdi_checkpoint, set_csdi_attractor_std
        set_csdi_checkpoint(args.csdi_ckpt)
        attr_std = lorenz96_attractor_std(N=args.N, F=args.F)
        set_csdi_attractor_std(attr_std)
        print(f"[iso] CSDI ckpt: {args.csdi_ckpt}")
        print(f"[iso] CSDI attractor_std override: {attr_std:.4f}")
    attr_std = lorenz96_attractor_std(N=args.N, F=args.F)
    lyap = LORENZ96_LYAP_F8

    print(f"[iso] N={args.N} F={args.F} attr_std={attr_std:.3f} lyap={lyap}")
    print(f"[iso] cells: {[c[0] for c in cells]}")
    print(f"[iso] scenarios: {[s.name for s in ISOLATION_SCENARIOS]}")
    print(f"[iso] n_seeds={args.n_seeds} (offset={args.seed_offset})")

    records: list[dict] = []
    for i in range(args.n_seeds):
        seed = args.seed_offset + i
        traj = integrate_lorenz96(
            args.n_ctx + args.pred_len, N=args.N, F=args.F,
            dt=args.dt, spinup=2000, seed=seed,
        )
        ctx_true = traj[: args.n_ctx]
        future_true = traj[args.n_ctx :]

        for sc in ISOLATION_SCENARIOS:
            observed, mask = make_sparse_noisy(
                ctx_true,
                sparsity=sc.sparsity,
                noise_std_frac=sc.noise_std_frac,
                attractor_std=attr_std,
                seed=1000 * seed + SCENARIO_SEEDS[sc.name],
            )
            keep = float(mask.mean())
            cached_fills: dict[str, np.ndarray] = {}
            sigma_override = sc.noise_std_frac * attr_std

            for label, imputer, forecaster in cells:
                mean, err, t_infer = run_one_cell(
                    label, imputer, forecaster, observed,
                    pred_len=args.pred_len, seed=seed, cached_fills=cached_fills,
                    sigma_override=sigma_override,
                )
                if mean is None:
                    rec = dict(
                        seed=seed, scenario=sc.name, label=label,
                        imputer=imputer, forecaster=forecaster,
                        sparsity=sc.sparsity, noise_std_frac=sc.noise_std_frac,
                        keep_frac=keep,
                        vpt03=float("nan"), vpt05=float("nan"), vpt10=float("nan"),
                        rmse_norm_first100=float("nan"),
                        infer_time_s=t_infer, error=err,
                    )
                    print(f"  seed={seed} {sc.name} {label:16s}  FAILED: {err}")
                else:
                    vpt03 = valid_prediction_time(future_true, mean, dt=args.dt,
                                                   lyap=lyap, threshold=0.3,
                                                   attractor_std=attr_std)
                    vpt05 = valid_prediction_time(future_true, mean, dt=args.dt,
                                                   lyap=lyap, threshold=0.5,
                                                   attractor_std=attr_std)
                    vpt10 = valid_prediction_time(future_true, mean, dt=args.dt,
                                                   lyap=lyap, threshold=1.0,
                                                   attractor_std=attr_std)
                    rmse_norm = float(
                        np.sqrt(((future_true[: min(100, args.pred_len)]
                                  - mean[: min(100, args.pred_len)]) ** 2).mean())
                        / attr_std
                    )
                    rec = dict(
                        seed=seed, scenario=sc.name, label=label,
                        imputer=imputer, forecaster=forecaster,
                        sparsity=sc.sparsity, noise_std_frac=sc.noise_std_frac,
                        keep_frac=keep,
                        vpt03=float(vpt03), vpt05=float(vpt05), vpt10=float(vpt10),
                        rmse_norm_first100=rmse_norm,
                        infer_time_s=t_infer, error=None,
                    )
                    print(
                        f"  seed={seed} {sc.name} {label:16s} keep={keep:.2f} "
                        f"VPT@1.0={vpt10:5.2f}  rmse/std={rmse_norm:.3f}  t={t_infer:.1f}s"
                    )
                records.append(rec)

    # Summarize: mean ± std + Pr(VPT>0.5) per (label, scenario)
    import collections
    acc: dict[tuple[str, str], dict] = collections.defaultdict(
        lambda: {"vpt03": [], "vpt05": [], "vpt10": [], "rmse": []}
    )
    for r in records:
        if r.get("error"):
            continue
        k = acc[(r["label"], r["scenario"])]
        k["vpt03"].append(r["vpt03"])
        k["vpt05"].append(r["vpt05"])
        k["vpt10"].append(r["vpt10"])
        k["rmse"].append(r["rmse_norm_first100"])

    summary: dict = {}
    for (label, sc), d in acc.items():
        if not d["vpt10"]:
            continue
        vpt10_arr = np.array(d["vpt10"])
        summary.setdefault(label, {})[sc] = {
            "vpt03_mean": float(np.mean(d["vpt03"])),
            "vpt03_std":  float(np.std(d["vpt03"])),
            "vpt05_mean": float(np.mean(d["vpt05"])),
            "vpt05_std":  float(np.std(d["vpt05"])),
            "vpt10_mean": float(np.mean(d["vpt10"])),
            "vpt10_std":  float(np.std(d["vpt10"])),
            "vpt10_survival_05": float((vpt10_arr > 0.5).mean()),
            "vpt10_survival_10": float((vpt10_arr > 1.0).mean()),
            "n_seeds": int(len(vpt10_arr)),
            "rmse_mean": float(np.mean(d["rmse"])),
            "rmse_std":  float(np.std(d["rmse"])),
        }

    out_json = OUT_DIR / f"pt_l96_{args.tag}.json"
    out_json.write_text(json.dumps(dict(
        config=vars(args),
        records=records,
        summary=summary,
        meta=dict(attractor_std=attr_std, lyap=lyap, N=args.N, F=args.F),
    ), indent=2))
    print(f"\n[iso] saved → {out_json}")

    # Quick verdict print
    print(f"\n[verdict] VPT@1.0 mean (Pr(VPT>0.5)) on L96 N={args.N}:")
    scen_names = [s.name for s in ISOLATION_SCENARIOS]
    header = f"  {'label':16s}" + "".join(f"  {s:>16s}" for s in scen_names)
    print(header)
    for label, _, _ in cells:
        if label not in summary:
            continue
        line = f"  {label:16s}"
        for s in scen_names:
            cell = summary[label].get(s)
            if cell:
                line += f"  {cell['vpt10_mean']:5.2f}({cell['vpt10_survival_05']:.0%})  "
            else:
                line += f"  {'—':>16s}"
        print(line)


if __name__ == "__main__":
    main()
