"""Observation-space trajectory overlay for qualitative inspection.

For each harshness scenario in a picked seed, show:
  - the sparse noisy context (scatter of observations)
  - the true future continuation (solid line)
  - each method's point forecast (dashed line)
  - a red vertical bar at the VPT@1.0 breakpoint per method

3 channels (x, y, z of Lorenz63) in separate rows; scenarios in separate columns.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from baselines.panda_adapter import panda_forecast
from experiments.week1.baselines import (
    chronos_forecast,
    context_parroting_forecast,
    persistence_forecast,
)
from experiments.week1.full_pipeline_rollout import full_pipeline_forecast
from experiments.week1.lorenz63_utils import (
    LORENZ63_ATTRACTOR_STD,
    LORENZ63_LYAP,
    PILOT_SCENARIOS,
    integrate_lorenz63,
    linear_interp_fill,
    make_sparse_noisy,
    valid_prediction_time,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "experiments" / "week1" / "figures"

METHOD_COLORS = {
    "truth":   "black",
    "ours":    "#1b9e77",
    "panda":   "#9467bd",
    "parrot":  "#0868ac",
    "chronos": "#d95f02",
    "persist": "#999999",
}
METHOD_LABEL = {
    "ours": "Ours (v2)", "panda": "Panda-72M", "parrot": "Parrot",
    "chronos": "Chronos-T5-small", "persist": "Persist",
}
CH_NAMES = ["x", "y", "z"]


def run_all_methods(ctx_true, observed, pred_len, seed, dt, chronos_pipe):
    """Return a dict method -> [pred_len, D] mean forecast."""
    ctx_filled = linear_interp_fill(observed)
    out = {}
    out["ours"] = full_pipeline_forecast(observed, pred_len=pred_len, seed=seed,
                                         bayes_calls=10, n_epochs=150)
    out["panda"] = panda_forecast(ctx_filled, pred_len=pred_len)
    if chronos_pipe is not None:
        mean, _ = chronos_forecast(chronos_pipe, ctx_filled, pred_len=pred_len, num_samples=20)
        out["chronos"] = mean
    out["parrot"] = context_parroting_forecast(ctx_filled, pred_len=pred_len)
    out["persist"] = persistence_forecast(ctx_filled, pred_len)
    return out


def plot_grid(
    scenarios_picked: list[str], seed: int, n_ctx: int, pred_len: int, dt: float,
    chronos_model: str, fig_path: Path, ctx_show: int = 128,
):
    # load chronos once
    from chronos import ChronosPipeline
    pipe = ChronosPipeline.from_pretrained(chronos_model, device_map="cuda", torch_dtype=torch.float32)

    # integrate trajectory once per seed
    traj = integrate_lorenz63(n_ctx + pred_len, dt=dt, seed=seed, spinup=2000)
    ctx_true = traj[:n_ctx]
    future_true = traj[n_ctx:]

    nrows = 3  # channels x, y, z
    ncols = len(scenarios_picked)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.3 * nrows), sharex="col")
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    t_ctx_full = np.arange(n_ctx) * dt
    t_fut = n_ctx * dt + np.arange(pred_len) * dt

    for col_idx, sc_name in enumerate(scenarios_picked):
        sc = next(s for s in PILOT_SCENARIOS if s.name == sc_name)
        print(f"[{sc.name}] sparsity={sc.sparsity} noise={sc.noise_std_frac}")
        observed, mask = make_sparse_noisy(
            ctx_true, sparsity=sc.sparsity, noise_std_frac=sc.noise_std_frac,
            attractor_std=LORENZ63_ATTRACTOR_STD, seed=1000 * seed + hash(sc.name) % 10000,
        )
        preds = run_all_methods(ctx_true, observed, pred_len, seed, dt, pipe)

        # VPT per method per scenario — used to mark breakpoint
        for ch in range(3):
            ax = axes[ch, col_idx]
            # show last ctx_show of the context in faded black
            t_ctx_tail = t_ctx_full[-ctx_show:]
            ax.plot(t_ctx_tail, ctx_true[-ctx_show:, ch], color="black", lw=1.5, alpha=0.5, label="truth (ctx)")
            # observed points in the same window
            obs_mask_tail = mask[-ctx_show:].astype(bool)
            obs_tail = observed[-ctx_show:, ch]
            ax.scatter(t_ctx_tail[obs_mask_tail], obs_tail[obs_mask_tail],
                       s=12, color="black", alpha=0.5, marker="o", label="observed")
            # future ground truth
            ax.plot(t_fut, future_true[:, ch], color="black", lw=2.0, label="truth (future)")
            # each method's forecast
            for m, pred in preds.items():
                ax.plot(t_fut, pred[:, ch], color=METHOD_COLORS[m], lw=1.5,
                        linestyle="--", label=METHOD_LABEL[m], alpha=0.85)
                # mark VPT@1.0 breakpoint
                vpt = valid_prediction_time(future_true, pred, dt=dt, threshold=1.0)
                t_fail = n_ctx * dt + (vpt / LORENZ63_LYAP)
                if vpt > 0 and vpt < pred_len * dt * LORENZ63_LYAP - 1e-6:
                    ax.axvline(t_fail, color=METHOD_COLORS[m], linestyle=":", lw=0.8, alpha=0.6)

            ax.axvline(n_ctx * dt, color="red", linestyle="-", lw=1, alpha=0.6)
            if ch == 0:
                vpt_str = " | ".join(f"{m}={valid_prediction_time(future_true, preds[m], dt=dt, threshold=1.0):.2f}Λ"
                                     for m in ["ours", "panda", "parrot", "chronos"])
                ax.set_title(f"{sc.name}  sp={sc.sparsity:.0%}  σ={sc.noise_std_frac:.2f}\n{vpt_str}",
                             fontsize=10)
            ax.set_ylabel(f"channel {CH_NAMES[ch]}")
            if ch == nrows - 1:
                ax.set_xlabel("time (simulation units)")
            ax.grid(True, alpha=0.2)

    # shared legend in the first axis
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Observation-space forecasts (Lorenz63, seed={seed}); "
                 f"red line = context/future boundary; dotted verticals = VPT@1.0 breakpoint per method",
                 y=1.005, fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"[fig] saved {fig_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3, help="seed 3 had the clearest phase-transition signal")
    ap.add_argument("--scenarios", nargs="+", default=["S0", "S2", "S3", "S5"])
    ap.add_argument("--n_ctx", type=int, default=512)
    ap.add_argument("--pred_len", type=int, default=128)
    ap.add_argument("--dt", type=float, default=0.025)
    ap.add_argument("--chronos", default="amazon/chronos-t5-small")
    ap.add_argument("--tag", default="seed3_4panels")
    ap.add_argument("--ctx_show", type=int, default=128)
    args = ap.parse_args()

    fig_path = FIG_DIR / f"trajectory_overlay_{args.tag}.png"
    plot_grid(args.scenarios, args.seed, args.n_ctx, args.pred_len, args.dt,
              args.chronos, fig_path, ctx_show=args.ctx_show)


if __name__ == "__main__":
    main()
