"""L96 N=20 v2 grid/smoke.

Per LIVE_DIALOGUE.md consensus: locate L96 N=20 transition band by running
SP65 (sparsity 0.65, sigma 0) with panda_linear vs panda_csdi, 5 seeds.
If too easy / too hard, run SP55 / SP75 same protocol.

After the L63 Figure-1 lock, the same runner is also used for the cross-system
v2 replication with cells:
  panda_linear / panda_csdi / deepedm_linear / deepedm_csdi.

CSDI checkpoint pinned: dyn_csdi_l96_full_c192_vales_best.pt (D=20, 11MB).

Run:
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \\
  MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \\
  python -u -m experiments.week1.smoke_l96N20_v2 \\
      --configs SP65 \\
      --n_seeds 5 \\
      --tag l96N20_smoke_sp65_5seed
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from experiments.week1.corruption import make_corrupted_observations
from experiments.week1.full_pipeline_rollout import full_pipeline_forecast
from experiments.week1.lorenz63_utils import valid_prediction_time
from experiments.week1.lorenz96_utils import (
    LORENZ96_LYAP_F8, LORENZ96_F_DEFAULT,
    integrate_lorenz96, lorenz96_attractor_std,
)
from methods.dynamics_impute import impute

try:
    from baselines.panda_adapter import panda_forecast
    _HAS_PANDA = True
except ImportError:
    _HAS_PANDA = False

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "week1" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
CONFIG_JSON = REPO / "experiments" / "week1" / "configs" / "corruption_grid_v2.json"
# CRITICAL: L96 N=20 ckpt, NOT L63's full_v6_center_ep20.pt.
L96_N20_CKPT = REPO / "experiments/week2_modules/ckpts/dyn_csdi_l96_full_c192_vales_best.pt"


def load_named_configs(names: list[str]) -> list[dict]:
    """Load named configs from corruption_grid_v2.json (across all sections)."""
    doc = json.loads(CONFIG_JSON.read_text())
    pool = []
    for key in ("legacy_diagonal", "fine_s_line", "fine_sigma_line",
                 "summary_path_candidate", "pattern_grid"):
        for i, cfg in enumerate(doc.get(key, [])):
            c = dict(cfg)
            c["_grid_section"] = key
            c["_grid_index"] = i
            pool.append(c)
    by_name = {c["name"]: c for c in pool}
    chosen = []
    for n in names:
        if n not in by_name:
            raise SystemExit(f"config {n!r} not in corruption_grid_v2.json; "
                             f"available: {sorted(by_name)}")
        chosen.append(by_name[n])
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["SP65"],
                    help="Scenario names from corruption_grid_v2.json")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--n_ctx", type=int, default=512)
    ap.add_argument("--pred_len", type=int, default=128)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--F", type=float, default=LORENZ96_F_DEFAULT)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--tag", default="l96N20_smoke_sp65_5seed")
    ap.add_argument("--cells", nargs="+", default=["panda_linear", "panda_csdi"],
                    help="Cells: panda_linear, panda_csdi, deepedm_linear, deepedm_csdi")
    args = ap.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit("Set CUDA_VISIBLE_DEVICES explicitly per RUN_PLAN_V2.md.")

    configs = load_named_configs(args.configs)

    if "panda_csdi" in args.cells:
        from methods.csdi_impute_adapter import set_csdi_checkpoint, set_csdi_attractor_std
        set_csdi_checkpoint(str(L96_N20_CKPT))
        attr_std = lorenz96_attractor_std(N=args.N, F=args.F)
        set_csdi_attractor_std(attr_std)
        print(f"[smoke-l96] CSDI ckpt: {L96_N20_CKPT}")
        print(f"[smoke-l96] CSDI attractor_std override: {attr_std:.4f}")

    attr_std = lorenz96_attractor_std(N=args.N, F=args.F)
    lyap = LORENZ96_LYAP_F8
    print(f"[smoke-l96] N={args.N} F={args.F} attr_std={attr_std:.3f} lyap={lyap}")
    print(f"[smoke-l96] configs: {[c['name'] for c in configs]}")
    print(f"[smoke-l96] cells: {args.cells}")
    print(f"[smoke-l96] n_seeds={args.n_seeds} (offset={args.seed_offset})")

    records: list[dict] = []
    for i in range(args.n_seeds):
        seed = args.seed_offset + i
        traj = integrate_lorenz96(
            args.n_ctx + args.pred_len, N=args.N, F=args.F,
            dt=args.dt, spinup=2000, seed=seed,
        )
        ctx_true = traj[: args.n_ctx]
        future_true = traj[args.n_ctx :]

        for cfg in configs:
            sparsity = float(cfg["sparsity"])
            sigma = float(cfg["noise_std_frac"])
            obs_res = make_corrupted_observations(
                ctx_true,
                mask_regime=cfg.get("mask_regime", "iid_time"),
                sparsity=sparsity, noise_std_frac=sigma,
                attractor_std=attr_std,
                seed=1000 * seed + 5000 + int(cfg["_grid_index"]),
                block_len=cfg.get("block_len"),
                period=cfg.get("period"),
                jitter=cfg.get("jitter"),
                mnar_strength=float(cfg.get("mnar_strength", 1.0)),
                dt=args.dt, lyap=lyap, patch_length=16,
            )
            observed = obs_res.observed
            keep = float(obs_res.metadata["keep_frac"])
            cached_fills: dict[str, np.ndarray] = {}

            for cell in args.cells:
                t0 = time.time()
                imputer = "linear" if cell.endswith("_linear") else \
                          "ar_kalman" if cell.endswith("_kalman") else "csdi"
                forecaster = "panda" if cell.startswith("panda_") else "deepedm"
                try:
                    if forecaster == "panda":
                        if not _HAS_PANDA:
                            raise RuntimeError("panda adapter not available")
                        filled = cached_fills.get(imputer)
                        if filled is None:
                            kwargs = {"sigma_override": sigma * attr_std} if imputer == "csdi" else {}
                            filled = impute(observed, kind=imputer, **kwargs)
                            cached_fills[imputer] = filled
                        mean = panda_forecast(filled, pred_len=args.pred_len)
                    elif forecaster == "deepedm":
                        impute_kwargs = {"sigma_override": sigma * attr_std} if imputer == "csdi" else {}
                        mean = full_pipeline_forecast(
                            observed, pred_len=args.pred_len, seed=seed,
                            imp_kind=imputer, bayes_calls=10, backbone="deepedm",
                            impute_kwargs=impute_kwargs,
                        )
                    else:
                        raise ValueError(forecaster)
                    err = None
                except Exception as e:
                    mean = None
                    err = str(e)[:200]
                t_infer = time.time() - t0

                if mean is None:
                    rec = dict(seed=seed, scenario=cfg["name"], cell=cell,
                                imputer=imputer, forecaster=forecaster,
                                sparsity=sparsity, noise_std_frac=sigma,
                                keep_frac=keep,
                                grid_section=cfg.get("_grid_section"),
                                grid_index=cfg.get("_grid_index"),
                                vpt03=float("nan"), vpt05=float("nan"),
                                vpt10=float("nan"),
                                infer_time_s=t_infer, error=err,
                                obs_per_patch_mean=obs_res.metadata.get("expected_obs_per_patch"),
                                max_gap_lyap=obs_res.metadata.get("all_missing_gap_max_lyap"))
                    print(f"  seed={seed} {cfg['name']} {cell:14s}  FAILED: {err}")
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
                    rec = dict(seed=seed, scenario=cfg["name"], cell=cell,
                                imputer=imputer, forecaster=forecaster,
                                sparsity=sparsity, noise_std_frac=sigma,
                                keep_frac=keep,
                                grid_section=cfg.get("_grid_section"),
                                grid_index=cfg.get("_grid_index"),
                                vpt03=float(vpt03), vpt05=float(vpt05),
                                vpt10=float(vpt10),
                                infer_time_s=t_infer, error=None,
                                obs_per_patch_mean=obs_res.metadata.get("expected_obs_per_patch"),
                                max_gap_lyap=obs_res.metadata.get("all_missing_gap_max_lyap"))
                    print(f"  seed={seed} {cfg['name']} {cell:14s} keep={keep:.2f} "
                          f"VPT@1.0={vpt10:5.2f}  t={t_infer:.1f}s")
                records.append(rec)

    # Aggregate
    import collections
    acc = collections.defaultdict(lambda: {"vpt10": []})
    for r in records:
        if r.get("error"):
            continue
        acc[(r["cell"], r["scenario"])]["vpt10"].append(r["vpt10"])
    summary = {}
    for (cell, sc), d in acc.items():
        v = np.array(d["vpt10"])
        summary.setdefault(cell, {})[sc] = {
            "vpt10_mean": float(v.mean()),
            "vpt10_std": float(v.std()),
            "vpt10_median": float(np.median(v)),
            "vpt10_survival_05": float((v > 0.5).mean()),
            "vpt10_survival_10": float((v > 1.0).mean()),
            "n_seeds": int(len(v)),
        }

    out = RESULTS / f"pt_l96_smoke_{args.tag}.json"
    out.write_text(json.dumps(dict(
        config=vars(args), records=records, summary=summary,
        meta=dict(attractor_std=attr_std, lyap=lyap, N=args.N, F=args.F,
                   csdi_ckpt=str(L96_N20_CKPT)),
    ), indent=2))
    print(f"\n[smoke-l96] saved → {out}")
    print("\n[verdict] VPT@1.0 mean (median, Pr>0.5, Pr>1.0):")
    for cell in args.cells:
        if cell not in summary:
            continue
        for sc in [c["name"] for c in configs]:
            c = summary[cell].get(sc)
            if c is None:
                continue
            print(f"  {cell:14s}  {sc:6s}  μ={c['vpt10_mean']:.2f}  "
                  f"med={c['vpt10_median']:.2f}  "
                  f"Pr>0.5={c['vpt10_survival_05']:.0%}  "
                  f"Pr>1.0={c['vpt10_survival_10']:.0%}")


if __name__ == "__main__":
    main()
