"""Week 1 Day 6-7 — Phase Transition pilot (decisive for v2 sharp story).

Protocol (per tech.md Module 0):
  - Lorenz63 (canonical params, dt=0.01)
  - For each harshness scenario (S0..S6):
      observe context of length N_CTX with sparsity s and noise sigma = f * attractor_std
      forward-fill / linear-interp to produce a dense context Chronos can digest
      run Chronos-T5 zero-shot forecasting for PRED_LEN steps
      record ensemble mean + std + VPT
  - Repeat with N_SEED seeds.
  - Plot VPT-vs-harshness; look for a phase transition.

If Chronos degrades gracefully (no sharp drop) in all scenarios, we need to escalate
(harsher scenarios, or Lorenz96). If it drops sharply past some Sk, v2 story holds.

Run:
  CUDA_VISIBLE_DEVICES=2 python experiments/week1/phase_transition_pilot.py
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from experiments.week1.lorenz63_utils import (
    LORENZ63_ATTRACTOR_STD,
    LORENZ63_LYAP,
    PILOT_SCENARIOS,
    HarshnessScenario,
    forward_fill,
    integrate_lorenz63,
    linear_interp_fill,
    make_sparse_noisy,
    valid_prediction_time,
)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import warnings
warnings.filterwarnings("ignore", message=".*prediction length.*")
import logging
logging.getLogger("chronos").setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "experiments" / "week1" / "results"
FIG_DIR = REPO_ROOT / "experiments" / "week1" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_chronos(model_name: str = "amazon/chronos-t5-small", device: str = "cuda") -> object:
    from chronos import ChronosPipeline

    pipe = ChronosPipeline.from_pretrained(model_name, device_map=device, torch_dtype=torch.float32)
    return pipe


def chronos_predict(
    pipe,
    ctx_filled: np.ndarray,
    pred_len: int,
    num_samples: int = 20,
) -> np.ndarray:
    """Run Chronos zero-shot per-channel, return ensemble samples of shape (num_samples, pred_len, D)."""
    T, D = ctx_filled.shape
    preds = []
    for d in range(D):
        series = torch.tensor(ctx_filled[:, d], dtype=torch.float32)
        out = pipe.predict(series.unsqueeze(0), prediction_length=pred_len, num_samples=num_samples)
        preds.append(out[0].cpu().numpy())  # (num_samples, pred_len)
    stacked = np.stack(preds, axis=-1)  # (num_samples, pred_len, D)
    return stacked


def persistence_predict(ctx_filled: np.ndarray, pred_len: int) -> np.ndarray:
    """Trivial baseline: repeat the last observation."""
    last = ctx_filled[-1]
    return np.tile(last[None, :], (pred_len, 1))


def run_pilot(
    n_seeds: int = 5,
    n_ctx: int = 512,
    pred_len: int = 128,
    dt: float = 0.025,
    model_name: str = "amazon/chronos-t5-small",
    fill_method: str = "linear",
    spinup: int = 2000,
) -> list[dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[pilot] device={device} model={model_name} fill={fill_method} n_seeds={n_seeds}")
    pipe = load_chronos(model_name=model_name, device=device)
    fill_fn = {"forward": forward_fill, "linear": linear_interp_fill}[fill_method]

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
            ctx_filled = fill_fn(observed)

            t0 = time.time()
            samples = chronos_predict(pipe, ctx_filled, pred_len=pred_len, num_samples=20)
            pred_time = time.time() - t0
            mean_pred = samples.mean(0)
            std_pred = samples.std(0)

            vpt_chronos = valid_prediction_time(future_true, mean_pred, dt=dt, threshold=0.3)
            vpt_chronos_t05 = valid_prediction_time(future_true, mean_pred, dt=dt, threshold=0.5)
            vpt_chronos_t10 = valid_prediction_time(future_true, mean_pred, dt=dt, threshold=1.0)
            vpt_persist = valid_prediction_time(future_true, persistence_predict(ctx_filled, pred_len), dt=dt, threshold=0.3)
            rmse_norm = float(
                np.sqrt(((future_true[: min(100, pred_len)] - mean_pred[: min(100, pred_len)]) ** 2).mean())
                / LORENZ63_ATTRACTOR_STD
            )

            rec = dict(
                seed=seed,
                scenario=sc.name,
                sparsity=sc.sparsity,
                noise_std_frac=sc.noise_std_frac,
                keep_frac=float(mask.mean()),
                vpt_chronos=float(vpt_chronos),
                vpt_chronos_t05=float(vpt_chronos_t05),
                vpt_chronos_t10=float(vpt_chronos_t10),
                vpt_persistence=float(vpt_persist),
                rmse_norm_first100=rmse_norm,
                pred_time_s=pred_time,
            )
            records.append(rec)
            print(
                f"  seed={seed} {sc.name} keep={mask.mean():.2f} sigma={sc.noise_std_frac:.2f} "
                f"VPT_chronos={vpt_chronos:5.2f}Λ VPT_persist={vpt_persist:5.2f}Λ "
                f"rmse100/std={rmse_norm:.3f} t={pred_time:.1f}s"
            )
    return records


def summarize(records: list[dict]) -> dict:
    import collections

    by_sc: dict[str, dict] = collections.defaultdict(
        lambda: {"vpt": [], "vpt_t05": [], "vpt_t10": [], "rmse": [], "keep": [], "sparsity": None, "noise": None}
    )
    for r in records:
        k = by_sc[r["scenario"]]
        k["vpt"].append(r["vpt_chronos"])
        k["vpt_t05"].append(r.get("vpt_chronos_t05", r["vpt_chronos"]))
        k["vpt_t10"].append(r.get("vpt_chronos_t10", r["vpt_chronos"]))
        k["rmse"].append(r["rmse_norm_first100"])
        k["keep"].append(r["keep_frac"])
        k["sparsity"] = r["sparsity"]
        k["noise"] = r["noise_std_frac"]

    summary = {}
    for name, d in by_sc.items():
        summary[name] = {
            "sparsity": d["sparsity"],
            "noise_std_frac": d["noise"],
            "vpt_mean": float(np.mean(d["vpt"])),
            "vpt_std": float(np.std(d["vpt"])),
            "vpt_t05_mean": float(np.mean(d["vpt_t05"])),
            "vpt_t10_mean": float(np.mean(d["vpt_t10"])),
            "rmse_mean": float(np.mean(d["rmse"])),
            "rmse_std": float(np.std(d["rmse"])),
            "keep_mean": float(np.mean(d["keep"])),
        }
    return summary


def plot_phase_transition(summary: dict, fig_path: Path) -> None:
    import matplotlib.pyplot as plt

    names = sorted(summary.keys())
    vpt_mean = [summary[n]["vpt_mean"] for n in names]
    vpt_std = [summary[n]["vpt_std"] for n in names]
    sparsity = [summary[n]["sparsity"] for n in names]
    noise = [summary[n]["noise_std_frac"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    ax1.errorbar(range(len(names)), vpt_mean, yerr=vpt_std, marker="o", linewidth=2, capsize=4, color="C3")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([f"{n}\n(s={sparsity[i]:.2f},\nσ={noise[i]:.2f})" for i, n in enumerate(names)], fontsize=9)
    ax1.set_ylabel("VPT (Lyapunov times, threshold=0.3)")
    ax1.set_title("Chronos-T5 zero-shot on Lorenz63\nPhase transition under increasing harshness")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0.0, color="grey", lw=0.5)

    ax2.plot(range(len(names)), [summary[n]["rmse_mean"] for n in names], marker="s", color="C0")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names)
    ax2.set_ylabel("RMSE / attractor_std (first 100 steps)")
    ax2.set_title("Prediction error")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    print(f"[pilot] figure saved to {fig_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--n_ctx", type=int, default=512)
    ap.add_argument("--pred_len", type=int, default=128)
    ap.add_argument("--dt", type=float, default=0.025)
    ap.add_argument("--model", default="amazon/chronos-t5-small")
    ap.add_argument("--fill", default="linear", choices=["forward", "linear"])
    ap.add_argument("--tag", default="pilot_small")
    args = ap.parse_args()

    records = run_pilot(
        n_seeds=args.n_seeds,
        n_ctx=args.n_ctx,
        pred_len=args.pred_len,
        dt=args.dt,
        model_name=args.model,
        fill_method=args.fill,
    )
    summary = summarize(records)

    out_json = OUT_DIR / f"phase_transition_{args.tag}.json"
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

    fig_path = FIG_DIR / f"phase_transition_{args.tag}.png"
    plot_phase_transition(summary, fig_path)

    # Decision hint
    vpt_s0 = summary["S0"]["vpt_mean"]
    vpt_s3 = summary["S3"]["vpt_mean"]
    vpt_s5 = summary["S5"]["vpt_mean"]
    drop_s3 = (vpt_s0 - vpt_s3) / max(vpt_s0, 1e-6)
    drop_s5 = (vpt_s0 - vpt_s5) / max(vpt_s0, 1e-6)
    print(
        f"\n[verdict] VPT(S0)={vpt_s0:.2f}  VPT(S3)={vpt_s3:.2f} ({drop_s3 * 100:.0f}% drop)  "
        f"VPT(S5)={vpt_s5:.2f} ({drop_s5 * 100:.0f}% drop)"
    )
    if drop_s3 > 0.5:
        print("  -> Phase transition evidence: v2 sharp story LIKELY holds")
    elif drop_s5 > 0.5:
        print("  -> Late phase transition: v2 story plausibly holds (consider harsher scenarios)")
    else:
        print("  -> Chronos robust in all scenarios: v2 story AT RISK; escalate harshness or switch to Lorenz96")


if __name__ == "__main__":
    main()
