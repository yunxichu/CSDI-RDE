"""Lorenz63 v2 corruption grid for the paper pivot.

This runner separates sparsity, noise, and missingness pattern instead of using
only the legacy diagonal S0-S6 path.  It can run in a metadata-only dry mode, so
we can inspect whether proposed stages are too coarse before spending GPU time.

Resource guard:
  - CPU threads default to 4 per process.
  - CUDA runs require CUDA_VISIBLE_DEVICES to be set explicitly.
  - The script refuses more than --max_gpus visible GPUs.

Example dry run:
  python -m experiments.week1.phase_transition_grid_l63_v2 \
      --grid fine_s_line --max_configs 3 --n_seeds 1 --dry_run_metadata

Example GPU run:
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -u -m experiments.week1.phase_transition_grid_l63_v2 \
      --grid summary_path_candidate --n_seeds 10 \
      --cells panda_linear panda_csdi deepedm_linear deepedm_csdi \
      --csdi_ckpt experiments/week2_modules/ckpts/dyn_csdi_full_v6_center_ep20.pt
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _var in THREAD_ENV_VARS:
    os.environ.setdefault(_var, "4")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np

from experiments.week1.corruption import make_corrupted_observations
from experiments.week1.lorenz63_utils import (
    LORENZ63_ATTRACTOR_STD,
    LORENZ63_LYAP,
    integrate_lorenz63,
    valid_prediction_time,
)

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "week1" / "configs" / "corruption_grid_v2.json"
OUT_DIR = REPO_ROOT / "experiments" / "week1" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELL_DEFS: dict[str, tuple[str, str]] = {
    "panda_linear": ("linear", "panda"),
    "panda_kalman": ("ar_kalman", "panda"),
    "panda_csdi": ("csdi", "panda"),
    "deepedm_linear": ("linear", "deepedm"),
    "deepedm_kalman": ("ar_kalman", "deepedm"),
    "deepedm_csdi": ("csdi", "deepedm"),
}
DEFAULT_CELLS = ["panda_linear", "panda_csdi", "deepedm_linear", "deepedm_csdi"]


def _set_thread_env(n_threads: int) -> None:
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(max(1, int(n_threads)))


def _visible_gpu_count() -> int | None:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        return None
    cvd = cvd.strip()
    if not cvd or cvd in {"-1", "none", "None", "cpu"}:
        return 0
    return len([x for x in cvd.split(",") if x.strip()])


def _check_resources(args: argparse.Namespace) -> None:
    _set_thread_env(args.cpu_threads)
    visible = _visible_gpu_count()
    if visible is not None and visible > args.max_gpus:
        raise SystemExit(
            f"CUDA_VISIBLE_DEVICES exposes {visible} GPUs, exceeding --max_gpus={args.max_gpus}"
        )
    if not args.dry_run_metadata and args.device.startswith("cuda") and visible is None:
        raise SystemExit(
            "CUDA run requires CUDA_VISIBLE_DEVICES to be set explicitly, e.g. "
            "CUDA_VISIBLE_DEVICES=0 or CUDA_VISIBLE_DEVICES=0,1,2,3."
        )
    if args.dry_run_metadata:
        return
    try:
        import torch

        torch.set_num_threads(max(1, int(args.cpu_threads)))
        try:
            torch.set_num_interop_threads(max(1, min(2, int(args.cpu_threads))))
        except RuntimeError:
            pass
    except Exception as exc:
        print(f"[resource] torch thread setup skipped: {exc}")


def _float_code(x: float) -> str:
    return f"{int(round(100 * x)):03d}"


def _expand_grid(config: dict[str, Any], grid_name: str) -> list[dict[str, Any]]:
    defaults = dict(config.get("defaults", {}))
    if grid_name == "transition_rectangle":
        grid = config[grid_name]
        items = []
        for s in grid["s_values"]:
            for sig in grid["noise_values"]:
                items.append({
                    "name": f"TR_s{_float_code(float(s))}_n{_float_code(float(sig))}",
                    "sparsity": float(s),
                    "noise_std_frac": float(sig),
                })
    else:
        raw = config.get(grid_name)
        if not isinstance(raw, list):
            raise SystemExit(f"Unknown grid {grid_name!r}; keys={sorted(config.keys())}")
        items = [dict(x) for x in raw]

    out = []
    for i, item in enumerate(items):
        merged = {**defaults, **item}
        merged.setdefault("name", f"{grid_name}_{i:03d}")
        merged["_grid_index"] = i
        out.append(merged)
    return out


def _cell_plan(names: list[str] | None) -> list[tuple[str, str, str]]:
    wanted = names or DEFAULT_CELLS
    unknown = [x for x in wanted if x not in CELL_DEFS]
    if unknown:
        raise SystemExit(f"Unknown cells {unknown}; choices={sorted(CELL_DEFS)}")
    return [(label, *CELL_DEFS[label]) for label in wanted]


def _setup_csdi_if_needed(cells: list[tuple[str, str, str]], ckpt: str | None) -> None:
    if not any(imputer == "csdi" for _, imputer, _ in cells):
        return
    if not ckpt:
        raise SystemExit("--csdi_ckpt is required when any *_csdi cell is selected")
    from methods.csdi_impute_adapter import set_csdi_attractor_std, set_csdi_checkpoint

    set_csdi_checkpoint(ckpt)
    set_csdi_attractor_std(LORENZ63_ATTRACTOR_STD)
    print(f"[grid-l63] CSDI ckpt: {ckpt}")
    print(f"[grid-l63] CSDI attractor_std override: {LORENZ63_ATTRACTOR_STD:.4f}")


def _run_one_cell(
    *,
    label: str,
    imputer: str,
    forecaster: str,
    observed: np.ndarray,
    pred_len: int,
    seed: int,
    cached_fills: dict[str, np.ndarray],
    device: str,
    pipeline_kwargs: dict[str, Any],
    sigma_override: float | None,
) -> tuple[np.ndarray | None, str | None, float]:
    t0 = time.time()
    try:
        if forecaster == "panda":
            from baselines.panda_adapter import panda_forecast
            from methods.dynamics_impute import impute

            filled = cached_fills.get(imputer)
            if filled is None:
                kwargs = {"sigma_override": sigma_override} if imputer == "csdi" else {}
                filled = impute(observed, kind=imputer, **kwargs)
                cached_fills[imputer] = filled
            mean = panda_forecast(filled, pred_len=pred_len, device=device)
        elif forecaster == "deepedm":
            from experiments.week1.full_pipeline_rollout import full_pipeline_forecast

            run_kwargs = dict(pipeline_kwargs)
            if imputer == "csdi":
                run_kwargs["impute_kwargs"] = {"sigma_override": sigma_override}
            mean = full_pipeline_forecast(
                observed,
                pred_len=pred_len,
                seed=seed,
                imp_kind=imputer,
                backbone="deepedm",
                **run_kwargs,
            )
        else:
            raise ValueError(forecaster)
        return mean, None, time.time() - t0
    except Exception as exc:
        return None, str(exc)[:300], time.time() - t0


def _base_record(
    *,
    seed: int,
    cfg: dict[str, Any],
    grid_name: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    rec = {
        "seed": int(seed),
        "scenario": cfg["name"],
        "config_name": cfg["name"],
        "grid": grid_name,
        "sparsity": float(cfg["sparsity"]),
        "noise_std_frac": float(cfg["noise_std_frac"]),
        "mask_regime": cfg.get("mask_regime", "iid_time"),
        "metadata": metadata,
    }
    for key in [
        "keep_frac",
        "keep_frac_time_any",
        "keep_frac_time_all",
        "expected_obs_per_patch",
        "expected_full_timesteps_per_patch",
        "all_missing_gap_max_lyap",
        "not_full_gap_max_lyap",
    ]:
        rec[key] = metadata.get(key)
    return rec


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    acc = collections.defaultdict(lambda: {
        "vpt03": [],
        "vpt05": [],
        "vpt10": [],
        "rmse": [],
        "keep": [],
    })
    for rec in records:
        if rec.get("dry_run") or rec.get("error"):
            continue
        if rec.get("vpt10") != rec.get("vpt10"):
            continue
        key = (rec["label"], rec["scenario"])
        acc[key]["vpt03"].append(rec["vpt03"])
        acc[key]["vpt05"].append(rec["vpt05"])
        acc[key]["vpt10"].append(rec["vpt10"])
        acc[key]["rmse"].append(rec["rmse_norm_first100"])
        acc[key]["keep"].append(rec["keep_frac"])

    summary: dict[str, Any] = {}
    for (label, scenario), vals in acc.items():
        vpt10 = np.asarray(vals["vpt10"], dtype=np.float64)
        summary.setdefault(label, {})[scenario] = {
            "n": int(len(vpt10)),
            "keep_frac_mean": float(np.mean(vals["keep"])),
            "vpt03_mean": float(np.mean(vals["vpt03"])),
            "vpt05_mean": float(np.mean(vals["vpt05"])),
            "vpt10_mean": float(np.mean(vpt10)),
            "vpt10_median": float(np.median(vpt10)),
            "vpt10_std": float(np.std(vpt10)),
            "vpt10_survival_00": float((vpt10 > 0.0).mean()),
            "vpt10_survival_05": float((vpt10 > 0.5).mean()),
            "rmse_mean": float(np.mean(vals["rmse"])),
            "rmse_std": float(np.std(vals["rmse"])),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--grid", default="summary_path_candidate",
                    choices=[
                        "legacy_diagonal",
                        "fine_s_line",
                        "fine_sigma_line",
                        "transition_rectangle",
                        "summary_path_candidate",
                        "pattern_grid",
                    ])
    ap.add_argument("--max_configs", type=int, default=None)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--n_ctx", type=int, default=512)
    ap.add_argument("--pred_len", type=int, default=128)
    ap.add_argument("--dt", type=float, default=0.025)
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--csdi_ckpt", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dry_run_metadata", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cpu_threads", type=int, default=4)
    ap.add_argument("--max_gpus", type=int, default=4)
    ap.add_argument("--bayes_calls", type=int, default=10)
    ap.add_argument("--fast_tau", action="store_true")
    ap.add_argument("--dm_epochs", type=int, default=400)
    args = ap.parse_args()

    _check_resources(args)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = json.loads(config_path.read_text())
    configs = _expand_grid(config, args.grid)
    if args.max_configs is not None:
        configs = configs[: args.max_configs]

    cells = _cell_plan(args.cells)
    if not args.dry_run_metadata:
        _setup_csdi_if_needed(cells, args.csdi_ckpt)

    visible = _visible_gpu_count()
    print(f"[grid-l63] grid={args.grid} configs={len(configs)} seeds={args.n_seeds}")
    print(f"[grid-l63] cells={[x[0] for x in cells]} dry_run={args.dry_run_metadata}")
    print(f"[resource] cpu_threads={args.cpu_threads} visible_gpus={visible} max_gpus={args.max_gpus}")

    pipeline_kwargs = {
        "bayes_calls": args.bayes_calls,
        "fast_tau": bool(args.fast_tau),
        "dm_n_epochs": args.dm_epochs,
        "m3_device": args.device,
    }

    records: list[dict[str, Any]] = []
    for i_seed in range(args.n_seeds):
        seed = args.seed_offset + i_seed
        traj = integrate_lorenz63(args.n_ctx + args.pred_len, dt=args.dt, spinup=2000, seed=seed)
        ctx_true = traj[: args.n_ctx]
        future_true = traj[args.n_ctx:]

        for cfg in configs:
            corrupt_seed = 1000 * seed + 5000 + int(cfg["_grid_index"])
            result = make_corrupted_observations(
                ctx_true,
                mask_regime=cfg.get("mask_regime", "iid_time"),
                sparsity=float(cfg["sparsity"]),
                noise_std_frac=float(cfg["noise_std_frac"]),
                attractor_std=LORENZ63_ATTRACTOR_STD,
                seed=corrupt_seed,
                per_dim_noise=bool(cfg.get("per_dim_noise", True)),
                block_len=cfg.get("block_len"),
                period=cfg.get("period"),
                jitter=cfg.get("jitter"),
                mnar_strength=float(cfg.get("mnar_strength", 1.0)),
                dt=args.dt,
                lyap=LORENZ63_LYAP,
                patch_length=int(cfg.get("patch_length", 16)),
            )
            base = _base_record(seed=seed, cfg=cfg, grid_name=args.grid, metadata=result.metadata)
            if args.dry_run_metadata:
                records.append({
                    **base,
                    "label": "metadata_only",
                    "imputer": None,
                    "forecaster": None,
                    "vpt03": float("nan"),
                    "vpt05": float("nan"),
                    "vpt10": float("nan"),
                    "rmse_norm_first100": float("nan"),
                    "infer_time_s": 0.0,
                    "error": None,
                    "dry_run": True,
                })
                print(
                    f"  seed={seed} {cfg['name']:14s} {cfg.get('mask_regime', 'iid_time'):18s} "
                    f"s={cfg['sparsity']:.2f} sig={cfg['noise_std_frac']:.2f} "
                    f"keep={result.metadata['keep_frac']:.3f} "
                    f"obs/patch={result.metadata['expected_obs_per_patch']:.2f} "
                    f"max_gap_L={result.metadata.get('all_missing_gap_max_lyap', 0.0):.2f}"
                )
                continue

            cached_fills: dict[str, np.ndarray] = {}
            sigma_override = float(cfg["noise_std_frac"]) * LORENZ63_ATTRACTOR_STD
            for label, imputer, forecaster in cells:
                mean, err, t_infer = _run_one_cell(
                    label=label,
                    imputer=imputer,
                    forecaster=forecaster,
                    observed=result.observed,
                    pred_len=args.pred_len,
                    seed=seed,
                    cached_fills=cached_fills,
                    device=args.device,
                    pipeline_kwargs=pipeline_kwargs,
                    sigma_override=sigma_override,
                )
                if mean is None:
                    rec = {
                        **base,
                        "label": label,
                        "imputer": imputer,
                        "forecaster": forecaster,
                        "vpt03": float("nan"),
                        "vpt05": float("nan"),
                        "vpt10": float("nan"),
                        "rmse_norm_first100": float("nan"),
                        "infer_time_s": float(t_infer),
                        "error": err,
                        "dry_run": False,
                    }
                    print(f"  seed={seed} {cfg['name']:14s} {label:16s} FAILED: {err}")
                else:
                    mean = np.asarray(mean)
                    if mean.ndim == 1:
                        mean = mean[:, None]
                    if mean.shape[0] < args.pred_len or mean.shape[1] != future_true.shape[1]:
                        shape_err = (
                            f"{label} returned forecast shape {mean.shape}, "
                            f"expected at least ({args.pred_len}, {future_true.shape[1]})"
                        )
                        rec = {
                            **base,
                            "label": label,
                            "imputer": imputer,
                            "forecaster": forecaster,
                            "vpt03": float("nan"),
                            "vpt05": float("nan"),
                            "vpt10": float("nan"),
                            "rmse_norm_first100": float("nan"),
                            "infer_time_s": float(t_infer),
                            "error": shape_err,
                            "dry_run": False,
                        }
                        print(f"  seed={seed} {cfg['name']:14s} {label:16s} FAILED: {shape_err}")
                        records.append(rec)
                        continue
                    mean_eval = mean[: args.pred_len]
                    vpt03 = valid_prediction_time(future_true, mean_eval, dt=args.dt, threshold=0.3)
                    vpt05 = valid_prediction_time(future_true, mean_eval, dt=args.dt, threshold=0.5)
                    vpt10 = valid_prediction_time(future_true, mean_eval, dt=args.dt, threshold=1.0)
                    n_eval = min(100, args.pred_len)
                    rmse_norm = float(
                        np.sqrt(((future_true[:n_eval] - mean_eval[:n_eval]) ** 2).mean())
                        / LORENZ63_ATTRACTOR_STD
                    )
                    rec = {
                        **base,
                        "label": label,
                        "imputer": imputer,
                        "forecaster": forecaster,
                        "vpt03": float(vpt03),
                        "vpt05": float(vpt05),
                        "vpt10": float(vpt10),
                        "rmse_norm_first100": rmse_norm,
                        "infer_time_s": float(t_infer),
                        "error": None,
                        "dry_run": False,
                    }
                    print(
                        f"  seed={seed} {cfg['name']:14s} {label:16s} "
                        f"keep={rec['keep_frac']:.2f} VPT@1.0={vpt10:5.2f} "
                        f"rmse/std={rmse_norm:.3f} t={t_infer:.1f}s"
                    )
                records.append(rec)

    tag = args.tag or f"{args.grid}_{'dry' if args.dry_run_metadata else 'run'}"
    out_json = OUT_DIR / f"pt_l63_grid_v2_{tag}.json"
    out_json.write_text(json.dumps({
        "config": vars(args),
        "config_file": str(config_path),
        "records": records,
        "summary": _summarize(records),
        "meta": {
            "system": "Lorenz63",
            "attractor_std": LORENZ63_ATTRACTOR_STD,
            "lyap": LORENZ63_LYAP,
            "resource_limits": config.get("resource_limits", {}),
        },
    }, indent=2))
    print(f"\n[grid-l63] saved -> {out_json}")


if __name__ == "__main__":
    main()
