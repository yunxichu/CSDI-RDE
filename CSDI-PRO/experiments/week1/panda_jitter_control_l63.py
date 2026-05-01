"""L63 Panda control: is CSDI rescue just stochastic jitter?

The current evidence says CSDI-filled contexts often forecast better with Panda
even though they are farther from clean in raw and Panda representation space.
This control separates generic noise regularization from dynamics-aware
reconstruction.

Contexts tested on matched corruptions:
  linear
  linear_iid_jitter      linear plus iid Gaussian noise on missing entries,
                         matched to the CSDI-linear residual scale
  linear_shuffled_resid  linear plus CSDI residuals shuffled over missing entries
  csdi

If jitter/shuffled residual rescues as much as CSDI, the mechanism is likely
regularization/conditioning. If CSDI is stronger, the temporal structure of the
corruption-aware imputation matters.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path
from typing import Any

for _var in [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(_var, "4")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import matplotlib.pyplot as plt
import numpy as np

from experiments.week1.corruption import make_corrupted_observations
from experiments.week1.lorenz63_utils import (
    LORENZ63_LYAP,
    LORENZ63_ATTRACTOR_STD,
    integrate_lorenz63,
    valid_prediction_time,
)
from methods.dynamics_impute import impute


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "week1" / "results"
FIGS = REPO / "experiments" / "week1" / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

CSDI_CKPT_L63 = REPO / "experiments" / "week2_modules" / "ckpts" / "dyn_csdi_full_v6_center_ep20.pt"

# Protocol locked to phase_transition_grid_l63_v2.py (Figure 1 protocol):
#   attractor_std = LORENZ63_ATTRACTOR_STD (constant, 8.51)
#   mask seed     = 1000 * seed + 5000 + GRID_INDEX[scenario]
# GRID_INDEX must mirror the position of each cell in
# `experiments/week1/configs/corruption_grid_v2.json["fine_s_line"]` so that
# the same `seed` produces an identical mask in jitter and v2 runs.
SCENARIOS = [
    {"name": "SP65", "sparsity": 0.65, "noise_std_frac": 0.0},
    {"name": "SP82", "sparsity": 0.82, "noise_std_frac": 0.0},
]
GRID_INDEX = {"SP65": 4, "SP82": 6}  # positions in fine_s_line
CELLS = ("linear", "linear_iid_jitter", "linear_shuffled_resid", "csdi")


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _make_controls(linear: np.ndarray, csdi: np.ndarray,
                   observed: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed * 917 + 11)
    missing = ~np.isfinite(observed)
    residual = csdi - linear

    iid = linear.copy()
    shuffled = linear.copy()
    for d in range(linear.shape[1]):
        miss_d = missing[:, d]
        if not miss_d.any():
            continue
        resid_d = residual[miss_d, d]
        sigma_d = float(np.std(resid_d))
        if sigma_d > 0:
            iid[miss_d, d] += rng.normal(loc=0.0, scale=sigma_d, size=int(miss_d.sum()))
        if len(resid_d) > 0:
            shuffled[miss_d, d] += rng.choice(resid_d, size=int(miss_d.sum()), replace=True)

    return {
        "linear": linear.astype(np.float32),
        "linear_iid_jitter": iid.astype(np.float32),
        "linear_shuffled_resid": shuffled.astype(np.float32),
        "csdi": csdi.astype(np.float32),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    acc: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for r in records:
        acc[(r["scenario"], r["cell"])].append(float(r["vpt10"]))
    out: dict[str, Any] = {}
    for scenario in sorted({r["scenario"] for r in records}):
        out[scenario] = {}
        for cell in CELLS:
            vals = np.asarray(acc[(scenario, cell)], dtype=np.float64)
            if len(vals) == 0:
                continue
            out[scenario][cell] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(np.median(vals)),
                "pr_gt_0p5": float((vals > 0.5).mean()),
                "pr_gt_1p0": float((vals > 1.0).mean()),
                "n": int(len(vals)),
            }
    return out


def _paired_bootstrap(a: np.ndarray, b: np.ndarray,
                      n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(diff)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boots[i] = diff[rng.integers(0, n, size=n)].mean()
    return float(diff.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _contrast_table(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for scenario in sorted({r["scenario"] for r in records}):
        by_cell = {
            cell: np.asarray([r["vpt10"] for r in records
                              if r["scenario"] == scenario and r["cell"] == cell], dtype=np.float64)
            for cell in CELLS
        }
        out[scenario] = {}
        base = by_cell["linear"]
        for cell in CELLS:
            if cell == "linear":
                continue
            mean, lo, hi = _paired_bootstrap(by_cell[cell], base, seed=17)
            out[scenario][f"{cell}_minus_linear"] = {
                "mean": mean,
                "ci95": [lo, hi],
            }
    return out


def _plot(summary: dict[str, Any], out_png: Path) -> None:
    fig, axes = plt.subplots(1, len(summary), figsize=(6 * len(summary), 4.2), sharey=True)
    if len(summary) == 1:
        axes = [axes]
    colors = {
        "linear": "C1",
        "linear_iid_jitter": "C4",
        "linear_shuffled_resid": "C5",
        "csdi": "C2",
    }
    for ax, (scenario, data) in zip(axes, summary.items()):
        vals = [data[cell]["mean"] for cell in CELLS]
        errs = [data[cell]["std"] for cell in CELLS]
        x = np.arange(len(CELLS))
        ax.bar(x, vals, yerr=errs, capsize=3, color=[colors[c] for c in CELLS], alpha=0.85)
        ax.set_title(f"L63 {scenario}: Panda VPT@1.0")
        ax.set_xticks(x)
        ax.set_xticklabels(CELLS, rotation=25, ha="right")
        ax.set_ylabel("Lyapunov times")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _write_md(summary: dict[str, Any], contrasts: dict[str, Any], out_md: Path) -> None:
    lines = ["# L63 Panda Jitter Control", ""]
    for scenario, data in summary.items():
        lines += [
            f"## {scenario}",
            "",
            "| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |",
            "|:--|--:|--:|--:|--:|",
        ]
        for cell in CELLS:
            s = data[cell]
            lines.append(
                f"| {cell} | {s['mean']:.2f} ± {s['std']:.2f} | {s['median']:.2f} | "
                f"{100*s['pr_gt_0p5']:.0f}% | {100*s['pr_gt_1p0']:.0f}% |"
            )
        lines += ["", "Paired differences vs linear:", ""]
        lines += ["| Contrast | Δ mean | 95% CI |", "|:--|--:|:--|"]
        for name, c in contrasts[scenario].items():
            lines.append(f"| {name} | {c['mean']:+.2f} | [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] |")
        lines.append("")
    out_md.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--n_ctx", type=int, default=512)
    ap.add_argument("--pred_len", type=int, default=128)
    ap.add_argument("--dt", type=float, default=0.025)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="l63_sp65_sp82_5seed")
    args = ap.parse_args()

    from methods.csdi_impute_adapter import set_csdi_attractor_std, set_csdi_checkpoint

    attr_std = float(LORENZ63_ATTRACTOR_STD)  # v2-protocol-aligned
    set_csdi_checkpoint(str(CSDI_CKPT_L63))
    set_csdi_attractor_std(attr_std)
    print(f"[jitter-control] CSDI ckpt={CSDI_CKPT_L63}")
    print(f"[jitter-control] attr_std override={attr_std:.4f}")
    print(f"[jitter-control] device={args.device}")

    from baselines.panda_adapter import panda_forecast

    records: list[dict[str, Any]] = []
    seeds = range(args.seed_offset, args.seed_offset + args.n_seeds)
    for sc in SCENARIOS:
        print(f"\n=== {sc['name']} s={sc['sparsity']} sigma={sc['noise_std_frac']} ===")
        for seed in seeds:
            traj = integrate_lorenz63(args.n_ctx + args.pred_len, dt=args.dt, spinup=2000, seed=seed).astype(np.float32)
            ctx_true = traj[: args.n_ctx]
            future_true = traj[args.n_ctx:]
            obs_res = make_corrupted_observations(
                ctx_true,
                mask_regime="iid_time",
                sparsity=float(sc["sparsity"]),
                noise_std_frac=float(sc["noise_std_frac"]),
                attractor_std=attr_std,
                seed=1000 * seed + 5000 + GRID_INDEX[sc["name"]],  # v2-protocol-aligned
                dt=args.dt,
                lyap=LORENZ63_LYAP,
                patch_length=16,
            )
            observed = obs_res.observed
            linear = impute(observed, kind="linear").astype(np.float32)
            csdi = impute(
                observed, kind="csdi",
                sigma_override=float(sc["noise_std_frac"]) * attr_std,
            ).astype(np.float32)
            contexts = _make_controls(linear, csdi, observed, seed)

            for cell in CELLS:
                t0 = time.time()
                mean = panda_forecast(contexts[cell], pred_len=args.pred_len, device=args.device)
                mean = mean[: args.pred_len]
                vpt03 = valid_prediction_time(future_true, mean, dt=args.dt, threshold=0.3)
                vpt05 = valid_prediction_time(future_true, mean, dt=args.dt, threshold=0.5)
                vpt10 = valid_prediction_time(future_true, mean, dt=args.dt, threshold=1.0)
                rmse100 = float(np.sqrt(((future_true[: min(100, args.pred_len)] -
                                           mean[: min(100, args.pred_len)]) ** 2).mean()))
                rec = {
                    "scenario": sc["name"],
                    "seed": int(seed),
                    "cell": cell,
                    "sparsity": float(sc["sparsity"]),
                    "noise_std_frac": float(sc["noise_std_frac"]),
                    "metadata": obs_res.metadata,
                    "vpt03": float(vpt03),
                    "vpt05": float(vpt05),
                    "vpt10": float(vpt10),
                    "rmse100": rmse100,
                    "runtime_sec": float(time.time() - t0),
                }
                records.append(rec)
                print(
                    f"  seed={seed} {cell:22s} keep={obs_res.metadata['keep_frac']:.3f} "
                    f"VPT@1.0={vpt10:5.2f} t={rec['runtime_sec']:.1f}s"
                )

    summary = _summarize(records)
    contrasts = _contrast_table(records)
    out_json = RESULTS / f"panda_jitter_control_{args.tag}.json"
    out_md = FIGS / f"panda_jitter_control_{args.tag}.md"
    out_png = FIGS / f"panda_jitter_control_{args.tag}.png"
    out_json.write_text(json.dumps({
        "config": vars(args) | {"csdi_ckpt": str(CSDI_CKPT_L63)},
        "records": _jsonable(records),
        "summary": _jsonable(summary),
        "contrasts": _jsonable(contrasts),
    }, indent=2))
    _write_md(summary, contrasts, out_md)
    _plot(summary, out_png)

    print(f"\n[saved] {out_json}")
    print(f"[saved] {out_md}")
    print(f"[saved] {out_png}")
    print("\n[verdict] mean VPT@1.0:")
    for scenario, data in summary.items():
        vals = "  ".join(f"{cell}={data[cell]['mean']:.2f}" for cell in CELLS)
        print(f"  {scenario}: {vals}")


if __name__ == "__main__":
    main()
