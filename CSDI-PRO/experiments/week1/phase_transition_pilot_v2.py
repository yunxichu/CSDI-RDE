"""Week 1 Day 6-7 — Phase Transition pilot (multi-baseline version).

Compared to v1, this script runs three baselines side-by-side so we can see
whether the v2 sharp-story actually reproduces:

  - chronos:  amazon/chronos-t5-{small,base,large} zero-shot univariate forecast
  - parrot:   context-parroting nearest-neighbour delay forecast (strong on clean chaos)
  - persist:  repeat-last-observation (trivial lower bound)

The central question is: does ANY of these exhibit a clear phase transition as
sparsity + noise are ramped up? If parrot works well on S0-S2 and collapses at S3+,
that's the evidence we need. If nothing ever works (like Chronos-only), we need
to escalate.

Run:
  CUDA_VISIBLE_DEVICES=2 python -m experiments.week1.phase_transition_pilot_v2 \
      --n_seeds 5 --model amazon/chronos-t5-small --tag multibase
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
    chronos_forecast,
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
    LORENZ63_ATTRACTOR_STD,
    LORENZ63_LYAP,
    PILOT_SCENARIOS,
    integrate_lorenz63,
    linear_interp_fill,
    make_sparse_noisy,
    valid_prediction_time,
)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore", message=".*prediction length.*")
logging.getLogger("chronos").setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "experiments" / "week1" / "results"
FIG_DIR = REPO_ROOT / "experiments" / "week1" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ["ours", "panda", "chronos", "parrot", "persist"]


def load_chronos(model_name: str, device: str):
    from chronos import ChronosPipeline

    return ChronosPipeline.from_pretrained(model_name, device_map=device, torch_dtype=torch.float32)


def run_pilot(
    n_seeds: int,
    n_ctx: int,
    pred_len: int,
    dt: float,
    model_name: str,
    spinup: int,
    methods: list[str],
) -> list[dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[pilot] device={device} model={model_name} n_seeds={n_seeds} "
        f"n_ctx={n_ctx} pred_len={pred_len} dt={dt} methods={methods}"
    )
    pipe = load_chronos(model_name=model_name, device=device) if "chronos" in methods else None

    records: list[dict] = []
    for seed in range(n_seeds):
        traj = integrate_lorenz63(n_ctx + pred_len, dt=dt, seed=seed, spinup=spinup)
        ctx_true = traj[:n_ctx]
        future_true = traj[n_ctx:]

        for sc in PILOT_SCENARIOS:
            observed, mask = make_sparse_noisy(
                ctx_true,
                sparsity=sc.sparsity,
                noise_std_frac=sc.noise_std_frac,
                attractor_std=LORENZ63_ATTRACTOR_STD,
                seed=1000 * seed + hash(sc.name) % 10000,
            )
            ctx_filled = linear_interp_fill(observed)

            for method in methods:
                t0 = time.time()
                if method == "chronos":
                    mean, _ = chronos_forecast(pipe, ctx_filled, pred_len=pred_len, num_samples=20)
                elif method == "parrot":
                    mean = context_parroting_forecast(ctx_filled, pred_len=pred_len)
                elif method == "persist":
                    mean = persistence_forecast(ctx_filled, pred_len)
                elif method == "ours":
                    mean = full_pipeline_forecast(
                        observed, pred_len=pred_len, seed=seed,
                        bayes_calls=10, n_epochs=150,
                    )
                elif method == "panda":
                    mean = panda_forecast(ctx_filled, pred_len=pred_len)
                else:
                    raise ValueError(method)
                t_infer = time.time() - t0

                vpt03 = valid_prediction_time(future_true, mean, dt=dt, threshold=0.3)
                vpt05 = valid_prediction_time(future_true, mean, dt=dt, threshold=0.5)
                vpt10 = valid_prediction_time(future_true, mean, dt=dt, threshold=1.0)
                rmse_norm = float(
                    np.sqrt(((future_true[: min(100, pred_len)] - mean[: min(100, pred_len)]) ** 2).mean())
                    / LORENZ63_ATTRACTOR_STD
                )

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
                )
                records.append(rec)
                print(
                    f"  seed={seed} {sc.name} {method:8s} keep={mask.mean():.2f} "
                    f"σ={sc.noise_std_frac:.2f}  VPT@0.3={vpt03:5.2f}  VPT@1.0={vpt10:5.2f}  "
                    f"rmse/std={rmse_norm:.3f}  t={t_infer:.1f}s"
                )
    return records


def summarize(records: list[dict]) -> dict:
    import collections

    acc: dict[tuple[str, str], dict] = collections.defaultdict(
        lambda: {"vpt03": [], "vpt05": [], "vpt10": [], "rmse": [], "keep": [], "sparsity": None, "noise": None}
    )
    for r in records:
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


def plot_multi(summary: dict, fig_path: Path, scenario_names: list[str]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    metrics = [("vpt03_mean", "vpt03_std", "VPT (threshold=0.3)"),
               ("vpt10_mean", "vpt10_std", "VPT (threshold=1.0)"),
               ("rmse_mean", "rmse_std", "NRMSE (first 100 steps)")]

    colors = {"ours": "#1b9e77", "panda": "#9467bd", "chronos": "C3", "parrot": "C0", "persist": "grey"}
    markers = {"ours": "D", "panda": "^", "chronos": "o", "parrot": "s", "persist": "x"}

    for ax, (mkey, skey, title) in zip(axes, metrics):
        for method in METHOD_ORDER:
            if method not in summary:
                continue
            vals = [summary[method][s][mkey] for s in scenario_names]
            errs = [summary[method][s][skey] for s in scenario_names]
            ax.errorbar(
                range(len(scenario_names)), vals, yerr=errs,
                marker=markers[method], color=colors[method],
                label=method, linewidth=2, capsize=3,
            )
        ax.set_xticks(range(len(scenario_names)))
        ax.set_xticklabels(scenario_names)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    axes[0].set_ylabel("VPT (Lyapunov times)")
    axes[2].set_ylabel("NRMSE / attractor-std")
    fig.suptitle("Lorenz63 under increasing harshness (sparsity + noise)", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"[pilot] figure saved to {fig_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--n_ctx", type=int, default=512)
    ap.add_argument("--pred_len", type=int, default=128)
    ap.add_argument("--dt", type=float, default=0.025)
    ap.add_argument("--model", default="amazon/chronos-t5-small")
    ap.add_argument("--methods", nargs="+", default=["ours", "panda", "chronos", "parrot", "persist"])
    ap.add_argument("--tag", default="with_panda")
    args = ap.parse_args()

    records = run_pilot(
        n_seeds=args.n_seeds,
        n_ctx=args.n_ctx,
        pred_len=args.pred_len,
        dt=args.dt,
        model_name=args.model,
        spinup=2000,
        methods=args.methods,
    )
    summary = summarize(records)

    out_json = OUT_DIR / f"pt_v2_{args.tag}.json"
    out_json.write_text(
        json.dumps(
            dict(
                config=vars(args),
                records=records,
                summary=summary,
                meta=dict(attractor_std=LORENZ63_ATTRACTOR_STD, lyap=LORENZ63_LYAP),
            ),
            indent=2,
        )
    )
    print(f"[pilot] records saved to {out_json}")

    scenario_names = [sc.name for sc in PILOT_SCENARIOS]
    plot_multi(summary, FIG_DIR / f"pt_v2_{args.tag}.png", scenario_names)

    print("\n[verdict] VPT@1.0 by (method, scenario):")
    for method in METHOD_ORDER:
        if method not in summary:
            continue
        line = f"  {method:8s}"
        for s in scenario_names:
            line += f"  {s}={summary[method][s]['vpt10_mean']:4.2f}"
        print(line)

    if "parrot" in summary:
        s0 = summary["parrot"]["S0"]["vpt10_mean"]
        s3 = summary["parrot"]["S3"]["vpt10_mean"]
        s5 = summary["parrot"]["S5"]["vpt10_mean"]
        if s0 > 1.0 and s3 < 0.5 * s0:
            print(
                f"\n  -> PARROT phase transition: S0={s0:.2f} → S3={s3:.2f} "
                f"({(1 - s3 / max(s0, 1e-6)) * 100:.0f}% drop). "
                "v2 sharp story PLAUSIBLE via tough-to-beat baseline."
            )
        else:
            print(
                f"\n  -> PARROT does not show crisp phase transition "
                f"(S0={s0:.2f} S3={s3:.2f} S5={s5:.2f}). v2 story needs Panda or escalation."
            )


if __name__ == "__main__":
    main()
