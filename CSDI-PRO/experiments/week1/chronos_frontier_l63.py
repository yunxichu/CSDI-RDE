"""Chronos mini-frontier on L63 — cross-foundation-model evidence.

Reproduces the §3 sparsity transition band on a second pretrained
foundation forecaster (Chronos) so the §3 / §4 mechanism does not look
Panda-only.

Cells: `linear -> Chronos`, `CSDI -> Chronos`.
Configs: SP55 / SP65 / SP75 / SP82 from corruption_grid_v2.json.
Protocol: identical to phase_transition_grid_l63_v2.py
(`LORENZ63_ATTRACTOR_STD = 8.51`, `dt = 0.025`, `n_ctx = 512`,
corruption seed `1000 * seed + 5000 + grid_index`,
CSDI inference with `sigma_override = noise_std_frac * attractor_std`).

Run:
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  python -u -m experiments.week1.chronos_frontier_l63 \
      --configs SP55 SP65 SP75 SP82 --n_seeds 5 \
      --tag chronos_l63_sp55_sp82_5seed
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.week1.corruption import make_corrupted_observations
from experiments.week1.lorenz63_utils import (
    LORENZ63_LYAP, LORENZ63_ATTRACTOR_STD, integrate_lorenz63,
    valid_prediction_time,
)
from methods.dynamics_impute import impute


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "week1" / "results"
FIGS = REPO / "experiments" / "week1" / "figures"
RESULTS.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
CONFIG_JSON = REPO / "experiments" / "week1" / "configs" / "corruption_grid_v2.json"
CSDI_CKPT = REPO / "experiments/week2_modules/ckpts/dyn_csdi_full_v6_center_ep20.pt"

CELLS = ("linear", "csdi")


def load_named_configs(names: list[str]) -> list[dict[str, Any]]:
    doc = json.loads(CONFIG_JSON.read_text())
    pool = []
    for key in ("legacy_diagonal", "fine_s_line", "fine_sigma_line",
                 "summary_path_candidate", "pattern_grid"):
        for i, cfg in enumerate(doc.get(key, [])):
            c = dict(cfg)
            c["_grid_index"] = i
            pool.append(c)
    by_name = {c["name"]: c for c in pool}
    return [by_name[n] for n in names]


def _paired_bootstrap(a, b, n_boot=5000, seed=29):
    rng = np.random.default_rng(seed)
    diff = np.asarray(a) - np.asarray(b)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                      for _ in range(n_boot)])
    return float(diff.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["SP55", "SP65", "SP75", "SP82"])
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--n_ctx", type=int, default=512)
    ap.add_argument("--pred_len", type=int, default=128)
    ap.add_argument("--dt", type=float, default=0.025)
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--chronos_model", default="amazon/chronos-bolt-small")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="chronos_l63_sp55_sp82_5seed")
    args = ap.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit("Set CUDA_VISIBLE_DEVICES explicitly per RUN_PLAN_V2.md.")

    configs = load_named_configs(args.configs)
    attr_std = float(LORENZ63_ATTRACTOR_STD); lyap = float(LORENZ63_LYAP)

    if "csdi" in args.cells:
        from methods.csdi_impute_adapter import set_csdi_attractor_std, set_csdi_checkpoint
        set_csdi_checkpoint(str(CSDI_CKPT))
        set_csdi_attractor_std(attr_std)
        print(f"[chronos-l63] CSDI ckpt: {CSDI_CKPT}")

    from baselines.chronos_adapter import chronos_forecast
    print(f"[chronos-l63] Chronos model: {args.chronos_model}")
    print(f"[chronos-l63] cells: {args.cells}  configs: {[c['name'] for c in configs]}")
    print(f"[chronos-l63] n_seeds={args.n_seeds}")

    records: list[dict[str, Any]] = []
    for cfg in configs:
        sparsity = float(cfg["sparsity"])
        sigma = float(cfg["noise_std_frac"])
        sigma_override = sigma * attr_std
        print(f"\n=== {cfg['name']}  s={sparsity}  sigma={sigma} ===")
        for i in range(args.n_seeds):
            seed = args.seed_offset + i
            traj = integrate_lorenz63(args.n_ctx + args.pred_len, dt=args.dt,
                                       spinup=2000, seed=seed).astype(np.float32)
            ctx_true = traj[: args.n_ctx]; future_true = traj[args.n_ctx :]
            obs_res = make_corrupted_observations(
                ctx_true, mask_regime="iid_time",
                sparsity=sparsity, noise_std_frac=sigma,
                attractor_std=attr_std,
                seed=1000 * seed + 5000 + int(cfg["_grid_index"]),
                dt=args.dt, lyap=lyap, patch_length=16,
            )
            observed = obs_res.observed
            keep = float(obs_res.metadata["keep_frac"])
            cached_fills: dict[str, np.ndarray] = {}

            for cell in args.cells:
                t0 = time.time()
                try:
                    if cell == "linear":
                        filled = cached_fills.get("linear")
                        if filled is None:
                            filled = impute(observed, kind="linear").astype(np.float32)
                            cached_fills["linear"] = filled
                    elif cell == "csdi":
                        filled = cached_fills.get("csdi")
                        if filled is None:
                            filled = impute(observed, kind="csdi",
                                              sigma_override=sigma_override).astype(np.float32)
                            cached_fills["csdi"] = filled
                    else:
                        raise ValueError(cell)
                    mean = chronos_forecast(filled, pred_len=args.pred_len,
                                              model_name=args.chronos_model,
                                              device=args.device)
                    err = None
                except Exception as e:
                    mean = None; err = str(e)[:200]
                t_infer = time.time() - t0

                if mean is None:
                    rec = dict(seed=int(seed), scenario=cfg["name"], cell=cell,
                                sparsity=sparsity, noise_std_frac=sigma,
                                keep_frac=keep,
                                vpt03=float("nan"), vpt05=float("nan"), vpt10=float("nan"),
                                infer_time_s=t_infer, error=err)
                    print(f"  seed={seed} {cell:8s}  FAILED: {err}")
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
                    rec = dict(seed=int(seed), scenario=cfg["name"], cell=cell,
                                sparsity=sparsity, noise_std_frac=sigma,
                                keep_frac=keep,
                                vpt03=float(vpt03), vpt05=float(vpt05),
                                vpt10=float(vpt10),
                                infer_time_s=t_infer, error=None)
                    print(f"  seed={seed} {cell:8s}  keep={keep:.2f}  "
                          f"VPT@1.0={vpt10:5.2f}  t={t_infer:.1f}s")
                records.append(rec)

    # Summary + paired bootstrap
    acc = collections.defaultdict(list)
    for r in records:
        if r.get("error"): continue
        acc[(r["scenario"], r["cell"])].append(float(r["vpt10"]))
    summary: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    for cfg in configs:
        sc = cfg["name"]
        summary[sc] = {}
        for cell in args.cells:
            v = np.array(acc[(sc, cell)])
            if len(v) == 0: continue
            summary[sc][cell] = {
                "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "median": float(np.median(v)),
                "pr_gt_0p5": float((v > 0.5).mean()),
                "pr_gt_1p0": float((v > 1.0).mean()),
                "n": int(len(v)),
            }
        if "linear" in args.cells and "csdi" in args.cells:
            a = np.array(acc[(sc, "csdi")]); b = np.array(acc[(sc, "linear")])
            if len(a) > 0 and len(b) == len(a):
                m, lo, hi = _paired_bootstrap(a, b)
                contrasts[sc] = {"csdi_minus_linear": {"mean": m, "ci95": [lo, hi]}}

    out_json = RESULTS / f"chronos_frontier_l63_{args.tag}.json"
    out_json.write_text(json.dumps(dict(
        config=vars(args), records=records, summary=summary, contrasts=contrasts,
        meta=dict(attractor_std=attr_std, lyap=lyap),
    ), indent=2))
    print(f"\n[chronos-l63] saved -> {out_json}")
    print("\n[verdict] Chronos mean VPT@1.0 (median, Pr>0.5, Pr>1.0):")
    for sc, cells in summary.items():
        for cell, s in cells.items():
            print(f"  {sc:6s}  {cell:8s}  μ={s['mean']:5.2f}  med={s['median']:5.2f}  "
                  f"Pr>0.5={s['pr_gt_0p5']:.0%}  Pr>1.0={s['pr_gt_1p0']:.0%}")
        if sc in contrasts:
            c = contrasts[sc]["csdi_minus_linear"]
            sgn = "↑" if c["ci95"][0] > 0 else ("↓" if c["ci95"][1] < 0 else "≈")
            print(f"          paired csdi-linear  Δ={c['mean']:+.2f}  "
                  f"CI=[{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]  {sgn}")


if __name__ == "__main__":
    main()
