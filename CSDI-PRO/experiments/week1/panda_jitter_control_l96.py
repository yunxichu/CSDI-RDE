"""L96 N=20 Panda jitter/residual control at the SP65 transition anchor.

This mirrors ``panda_jitter_control_l63.py`` on the high-dimensional headline
system. It asks whether the earlier L96 CSDI->Panda rescue is reproducible by
generic jitter or by shuffled CSDI residuals.
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
from experiments.week1.lorenz63_utils import valid_prediction_time
from experiments.week1.lorenz96_utils import (
    LORENZ96_F_DEFAULT,
    LORENZ96_LYAP_F8,
    integrate_lorenz96,
    lorenz96_attractor_std,
)
from methods.dynamics_impute import impute


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "week1" / "results"
FIGS = REPO / "experiments" / "week1" / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

L96_N20_CKPT = REPO / "experiments" / "week2_modules" / "ckpts" / "dyn_csdi_l96_full_c192_vales_best.pt"

SCENARIOS = [
    {"name": "SP65", "sparsity": 0.65, "noise_std_frac": 0.0},
    {"name": "SP82", "sparsity": 0.82, "noise_std_frac": 0.0},
]
GRID_INDEX = {"SP65": 4, "SP82": 6}
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
    rng = np.random.default_rng(seed * 1231 + 23)
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
            iid[miss_d, d] += rng.normal(0.0, sigma_d, size=int(miss_d.sum()))
        if len(resid_d) > 0:
            shuffled[miss_d, d] += rng.choice(resid_d, size=int(miss_d.sum()), replace=True)
    return {
        "linear": linear.astype(np.float32),
        "linear_iid_jitter": iid.astype(np.float32),
        "linear_shuffled_resid": shuffled.astype(np.float32),
        "csdi": csdi.astype(np.float32),
    }


def _paired_bootstrap(a: np.ndarray, b: np.ndarray,
                      n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(diff)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boots[i] = diff[rng.integers(0, n, size=n)].mean()
    return float(diff.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def _summarize(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    acc: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for r in records:
        acc[(r["scenario"], r["cell"])].append(float(r["vpt10"]))
    summary: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    for scenario in sorted({r["scenario"] for r in records}):
        summary[scenario] = {}
        by_cell = {}
        for cell in CELLS:
            vals = np.asarray(acc[(scenario, cell)], dtype=np.float64)
            by_cell[cell] = vals
            summary[scenario][cell] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(np.median(vals)),
                "pr_gt_0p5": float((vals > 0.5).mean()),
                "pr_gt_1p0": float((vals > 1.0).mean()),
                "n": int(len(vals)),
            }
        contrasts[scenario] = {}
        for cell in CELLS:
            if cell == "linear":
                continue
            mean, lo, hi = _paired_bootstrap(by_cell[cell], by_cell["linear"], seed=29)
            contrasts[scenario][f"{cell}_minus_linear"] = {"mean": mean, "ci95": [lo, hi]}
    return summary, contrasts


def _plot(summary: dict[str, Any], out_png: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
    colors = {
        "linear": "C1",
        "linear_iid_jitter": "C4",
        "linear_shuffled_resid": "C5",
        "csdi": "C2",
    }
    scenario = next(iter(summary))
    data = summary[scenario]
    x = np.arange(len(CELLS))
    ax.bar(x, [data[c]["mean"] for c in CELLS],
           yerr=[data[c]["std"] for c in CELLS], capsize=3,
           color=[colors[c] for c in CELLS], alpha=0.85)
    ax.set_title(f"L96 N=20 {scenario}: Panda VPT@1.0")
    ax.set_xticks(x)
    ax.set_xticklabels(CELLS, rotation=25, ha="right")
    ax.set_ylabel("Lyapunov times")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _write_md(summary: dict[str, Any], contrasts: dict[str, Any], out_md: Path) -> None:
    lines = ["# L96 N=20 Panda Jitter Control", ""]
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
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--F", type=float, default=LORENZ96_F_DEFAULT)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="l96N20_sp65_sp82_5seed")
    args = ap.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit("Set CUDA_VISIBLE_DEVICES explicitly.")

    from methods.csdi_impute_adapter import set_csdi_attractor_std, set_csdi_checkpoint
    from baselines.panda_adapter import panda_forecast

    attr_std = float(lorenz96_attractor_std(N=args.N, F=args.F))
    set_csdi_checkpoint(str(L96_N20_CKPT))
    set_csdi_attractor_std(attr_std)
    print(f"[l96-jitter] CSDI ckpt={L96_N20_CKPT}")
    print(f"[l96-jitter] N={args.N} F={args.F} attr_std={attr_std:.4f}")

    records: list[dict[str, Any]] = []
    for sc in SCENARIOS:
        print(f"\n=== {sc['name']} s={sc['sparsity']} sigma={sc['noise_std_frac']} ===")
        for i in range(args.n_seeds):
            seed = args.seed_offset + i
            traj = integrate_lorenz96(
                args.n_ctx + args.pred_len, N=args.N, F=args.F,
                dt=args.dt, spinup=2000, seed=seed,
            ).astype(np.float32)
            ctx_true = traj[: args.n_ctx]
            future_true = traj[args.n_ctx:]
            obs_res = make_corrupted_observations(
                ctx_true,
                mask_regime="iid_time",
                sparsity=float(sc["sparsity"]),
                noise_std_frac=float(sc["noise_std_frac"]),
                attractor_std=attr_std,
                seed=1000 * seed + 5000 + GRID_INDEX[sc["name"]],
                dt=args.dt,
                lyap=LORENZ96_LYAP_F8,
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
                vpt03 = valid_prediction_time(future_true, mean, dt=args.dt,
                                               lyap=LORENZ96_LYAP_F8, threshold=0.3,
                                               attractor_std=attr_std)
                vpt05 = valid_prediction_time(future_true, mean, dt=args.dt,
                                               lyap=LORENZ96_LYAP_F8, threshold=0.5,
                                               attractor_std=attr_std)
                vpt10 = valid_prediction_time(future_true, mean, dt=args.dt,
                                               lyap=LORENZ96_LYAP_F8, threshold=1.0,
                                               attractor_std=attr_std)
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
                    "runtime_sec": float(time.time() - t0),
                }
                records.append(rec)
                print(
                    f"  seed={seed} {cell:22s} keep={obs_res.metadata['keep_frac']:.3f} "
                    f"VPT@1.0={vpt10:5.2f} t={rec['runtime_sec']:.1f}s"
                )

    summary, contrasts = _summarize(records)
    out_json = RESULTS / f"panda_jitter_control_{args.tag}.json"
    out_md = FIGS / f"panda_jitter_control_{args.tag}.md"
    out_png = FIGS / f"panda_jitter_control_{args.tag}.png"
    out_json.write_text(json.dumps({
        "config": vars(args) | {"csdi_ckpt": str(L96_N20_CKPT)},
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
