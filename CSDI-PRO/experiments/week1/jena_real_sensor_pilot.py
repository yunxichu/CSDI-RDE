"""Real-sensor pilot (P2.1) — Jena Climate 2016 test year.

Tests whether the §4.4 finding "corpus-pretrained structured imputation is
the lever" carries from synthetic L63 / L96 to a real multivariate sensor
stream. Forecaster = Chronos-bolt-small (NOT Panda — Jena is outside
Panda's chaotic pretraining domain). Imputers = linear, SAITS pretrained on
the Jena 2009–2014 train split (separate from the 2016 test year).

Metric: normalized valid horizon. Per-feature error normalized by train
per-feature std (= 1.0 in z-score units since `clean` is already z-scored
per-feature using the train split). Valid horizon = first lead step h
where RMSE(features) over the future window of length h exceeds threshold.
We report mean valid horizon, median, and Pr(horizon > 16, 32, 64) at
horizon thresholds 0.5 / 1.0 / 2.0.

Run:
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  PYTHONPATH=/home/rhl/Github/CSDI-PRO python -u \
      -m experiments.week1.jena_real_sensor_pilot \
      --saits_ckpt experiments/week2_modules/ckpts/saits_jena_pretrained/<run-id>/SAITS.pypots \
      --configs SP55 SP65 SP75 SP82 --n_seeds 3 \
      --tag jena_real_sensor_pilot_3seed
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
from methods.dynamics_impute import impute


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "week1" / "results"
FIGS = REPO / "experiments" / "week1" / "figures"
RESULTS.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
CONFIG_JSON = REPO / "experiments" / "week1" / "configs" / "corruption_grid_v2.json"
JENA_NPZ = REPO / "experiments/week2_modules/data/real/jena_clean_hourly_L128.npz"


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


def normalized_valid_horizon(future_true: np.ndarray, forecast: np.ndarray,
                               threshold: float = 1.0) -> int:
    """Return the largest h in [0, H] such that RMSE(features) over the
    forecast at lead-time t ≤ h does not exceed `threshold` z-units. RMSE is
    computed across features per timestep, then we ask "how far into the
    forecast does the per-step RMSE stay below the threshold". Returns an
    integer in 0..H (number of valid steps).
    """
    H = future_true.shape[0]
    err = future_true - forecast  # (H, D)
    # RMSE per timestep across features
    per_step = np.sqrt((err * err).mean(axis=1))  # (H,)
    above = per_step > threshold
    if not above.any():
        return H
    return int(np.argmax(above))


def _saits_pretrained_impute(observed_2d, n_features, n_steps,
                              ckpt_path, train_n_steps=128,
                              cache: dict | None = None):
    from pypots.imputation import SAITS
    if n_steps % train_n_steps != 0:
        raise ValueError(f"n_steps {n_steps} must be divisible by {train_n_steps}")
    if cache is None:
        cache = {}
    key = (train_n_steps, n_features, ckpt_path)
    saits = cache.get(key)
    if saits is None:
        saits = SAITS(n_steps=train_n_steps, n_features=n_features,
                       n_layers=2, d_model=64, n_heads=4, d_k=16, d_v=16, d_ffn=128,
                       batch_size=64, epochs=0, verbose=False, device="cuda")
        saits.load(ckpt_path)
        cache[key] = saits
    n_chunks = n_steps // train_n_steps
    chunks = observed_2d.reshape(n_chunks, train_n_steps, n_features).astype(np.float32)
    imp = saits.impute({"X": chunks})
    imp = np.asarray(imp).reshape(n_chunks, train_n_steps, n_features)
    return imp.reshape(n_steps, n_features).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["SP55", "SP65", "SP75", "SP82"])
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--n_ctx", type=int, default=512)
    ap.add_argument("--pred_len", type=int, default=64)
    ap.add_argument("--saits_ckpt", required=True)
    ap.add_argument("--cells", nargs="+", default=["linear", "saits_pretrained"])
    ap.add_argument("--chronos_model", default="amazon/chronos-bolt-small")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="jena_real_sensor_pilot_3seed")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="z-score RMSE threshold for valid-horizon")
    args = ap.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit("Set CUDA_VISIBLE_DEVICES explicitly.")

    print(f"[jena-pilot] loading {JENA_NPZ}")
    d = np.load(JENA_NPZ)
    test_stream = d["test_hourly_z"]  # (T_test, D)
    feat_std_unscaled = float(d["attractor_std"])
    n_test, n_features = test_stream.shape
    print(f"[jena-pilot] test hourly shape: {test_stream.shape}; "
          f"surrogate attractor_std (unscaled) = {feat_std_unscaled:.3f}")

    window_len = args.n_ctx + args.pred_len
    if window_len > n_test:
        raise SystemExit(f"need {window_len} hours, have {n_test}")

    configs = load_named_configs(args.configs)
    print(f"[jena-pilot] cells: {args.cells}  configs: {[c['name'] for c in configs]}")

    from baselines.chronos_adapter import chronos_forecast
    saits_cache: dict = {}

    records: list[dict[str, Any]] = []
    for cfg in configs:
        sparsity = float(cfg["sparsity"])
        sigma = float(cfg["noise_std_frac"])
        # On real sensor data, "attractor_std" is a surrogate; CSDI is not
        # used here, but SAITS infers in z-space so corruption noise is
        # already in z-units. We pass sigma_override = sigma * 1.0 for the
        # corruption builder (which expects z-space noise scale on z-data).
        sigma_override = sigma * 1.0
        print(f"\n=== {cfg['name']}  s={sparsity}  sigma={sigma} ===")
        for i in range(args.n_seeds):
            seed = args.seed_offset + i
            rng = np.random.default_rng(seed)
            start = int(rng.integers(0, n_test - window_len + 1))
            window = test_stream[start:start + window_len]
            ctx_true = window[: args.n_ctx]
            future_true = window[args.n_ctx :]

            obs_res = make_corrupted_observations(
                ctx_true, mask_regime="iid_time",
                sparsity=sparsity, noise_std_frac=sigma,
                attractor_std=1.0,  # z-space data
                seed=1000 * seed + 5000 + int(cfg["_grid_index"]),
                dt=1.0, lyap=1.0, patch_length=16,
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
                    elif cell == "saits_pretrained":
                        filled = cached_fills.get("saits_pretrained")
                        if filled is None:
                            filled = _saits_pretrained_impute(
                                observed, n_features, args.n_ctx,
                                ckpt_path=args.saits_ckpt, cache=saits_cache,
                            )
                            cached_fills["saits_pretrained"] = filled
                    else:
                        raise ValueError(cell)
                    forecast = chronos_forecast(filled, pred_len=args.pred_len,
                                                  model_name=args.chronos_model,
                                                  device=args.device)
                    err = None
                except Exception as e:
                    forecast = None; err = str(e)[:200]
                t_infer = time.time() - t0

                if forecast is None:
                    rec = dict(seed=int(seed), scenario=cfg["name"], cell=cell,
                                sparsity=sparsity, noise_std_frac=sigma,
                                keep_frac=keep,
                                vh_t05=int(-1), vh_t10=int(-1), vh_t20=int(-1),
                                infer_time_s=t_infer, error=err,
                                start_offset=start)
                    print(f"  seed={seed} {cell:18s}  FAILED: {err}")
                else:
                    vh03 = normalized_valid_horizon(future_true, forecast, threshold=0.3)
                    vh05 = normalized_valid_horizon(future_true, forecast, threshold=0.5)
                    vh10 = normalized_valid_horizon(future_true, forecast, threshold=1.0)
                    vh20 = normalized_valid_horizon(future_true, forecast, threshold=2.0)
                    rec = dict(seed=int(seed), scenario=cfg["name"], cell=cell,
                                sparsity=sparsity, noise_std_frac=sigma,
                                keep_frac=keep,
                                vh_t03=int(vh03), vh_t05=int(vh05),
                                vh_t10=int(vh10), vh_t20=int(vh20),
                                infer_time_s=t_infer, error=None,
                                start_offset=start)
                    print(f"  seed={seed} {cell:18s}  keep={keep:.2f}  "
                          f"vh@0.3={vh03:3d} vh@0.5={vh05:3d} vh@1.0={vh10:3d}/{args.pred_len}  "
                          f"t={t_infer:.1f}s")
                records.append(rec)

    # Aggregate per (scenario, cell)
    acc = collections.defaultdict(lambda: {"vh_t03": [], "vh_t05": [], "vh_t10": [], "vh_t20": []})
    for r in records:
        if r.get("error"): continue
        key = (r["scenario"], r["cell"])
        for t in ("vh_t03", "vh_t05", "vh_t10", "vh_t20"):
            if t in r:
                acc[key][t].append(int(r[t]))

    def _paired_bootstrap(a, b, n=5000, seed=29):
        if len(a) != len(b) or len(a) == 0:
            return None
        rng = np.random.default_rng(seed)
        d = np.asarray(a) - np.asarray(b)
        boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
        return float(d.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

    summary: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    for cfg in configs:
        sc = cfg["name"]
        summary[sc] = {}
        for cell in args.cells:
            v = acc.get((sc, cell), None)
            if v is None or not v["vh_t10"]:
                continue
            arr03 = np.array(v["vh_t03"]) if v["vh_t03"] else np.array([])
            arr05 = np.array(v["vh_t05"])
            arr10 = np.array(v["vh_t10"])
            entry = {
                "n": int(arr10.size),
                "vh10_mean": float(arr10.mean()), "vh10_median": float(np.median(arr10)),
                "vh10_pr_gt_16": float((arr10 > 16).mean()),
                "vh10_pr_gt_32": float((arr10 > 32).mean()),
                "vh10_pr_eq_64": float((arr10 == args.pred_len).mean()),
                "vh05_mean": float(arr05.mean()), "vh05_median": float(np.median(arr05)),
            }
            if arr03.size:
                entry["vh03_mean"] = float(arr03.mean())
                entry["vh03_median"] = float(np.median(arr03))
            summary[sc][cell] = entry
        # Paired contrast SAITS vs linear if both present
        if "linear" in args.cells and "saits_pretrained" in args.cells:
            a = np.array(acc[(sc, "saits_pretrained")]["vh_t10"])
            b = np.array(acc[(sc, "linear")]["vh_t10"])
            res = _paired_bootstrap(a, b)
            if res is not None:
                m, lo, hi = res
                contrasts[sc] = {"saits_pretrained_minus_linear_vh10":
                                 {"mean": m, "ci95": [lo, hi]}}
                # also at vh05
                a5 = np.array(acc[(sc, "saits_pretrained")]["vh_t05"])
                b5 = np.array(acc[(sc, "linear")]["vh_t05"])
                res5 = _paired_bootstrap(a5, b5)
                if res5:
                    m5, lo5, hi5 = res5
                    contrasts[sc]["saits_pretrained_minus_linear_vh05"] = {
                        "mean": m5, "ci95": [lo5, hi5]
                    }

    out_json = RESULTS / f"jena_real_sensor_{args.tag}.json"
    out_json.write_text(json.dumps(dict(
        config=vars(args), records=records, summary=summary,
        contrasts=contrasts,
        meta=dict(metric="normalized_valid_horizon",
                  threshold_units="z-score-RMSE",
                  feat_std_unscaled=feat_std_unscaled,
                  n_features=n_features),
    ), indent=2))
    print(f"\n[jena-pilot] saved -> {out_json}")
    print("\n[verdict] valid-horizon @ 1.0 z-RMSE (mean / median / Pr>16 / Pr=64):")
    for sc, cells in summary.items():
        for cell, s in cells.items():
            print(f"  {sc:6s}  {cell:18s}  μ={s['vh10_mean']:5.1f}  med={s['vh10_median']:5.1f}  "
                  f"Pr>16={s['vh10_pr_gt_16']:.0%}  Pr=64={s['vh10_pr_eq_64']:.0%}")


if __name__ == "__main__":
    main()
