"""Visualize forecasting behavior on Lorenz96 N=20 at a fixed scenario.

Generates a figure with:
- top row: heatmap (time × dim) of true trajectory + each method's forecast
- bottom row: line plot of 3 selected dims (dim 0, 10, 15) for all methods

Helps diagnose what each method learns: magnitude, phase, spatial structure.

Usage:
    python -m experiments.week1.plot_l96_forecast_vs_truth \\
        --scenario S3 --seed 0 --ckpt <path> --tag s1337_ep20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from experiments.week1.lorenz96_utils import (
    integrate_lorenz96, lorenz96_attractor_std, LORENZ96_LYAP_F8, LORENZ96_F_DEFAULT,
)
from experiments.week1.lorenz63_utils import (
    HarshnessScenario, make_sparse_noisy, linear_interp_fill,
)
from experiments.week1.baselines import context_parroting_forecast
from experiments.week1.full_pipeline_rollout import full_pipeline_forecast
from baselines.panda_adapter import panda_forecast

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)


SCENARIOS = {
    "S0": HarshnessScenario("S0", 0.00, 0.00),
    "S1": HarshnessScenario("S1", 0.20, 0.10),
    "S2": HarshnessScenario("S2", 0.40, 0.30),
    "S3": HarshnessScenario("S3", 0.60, 0.50),
    "S4": HarshnessScenario("S4", 0.75, 0.80),
    "S5": HarshnessScenario("S5", 0.90, 1.20),
    "S6": HarshnessScenario("S6", 0.95, 1.50),
}


def run_all(scenario: HarshnessScenario, seed: int, csdi_ckpt: Path | None,
            N: int = 20, F: float = 8.0, dt: float = 0.05,
            n_ctx: int = 512, pred_len: int = 128, spinup: int = 2000):
    attr_std = lorenz96_attractor_std(N=N, F=F)
    lyap = LORENZ96_LYAP_F8
    print(f"[setup] L96 N={N} F={F} attr_std={attr_std:.3f} lyap={lyap}")
    print(f"[setup] scenario={scenario.name} s={scenario.sparsity} σ={scenario.noise_std_frac}")
    print(f"[setup] seed={seed}  n_ctx={n_ctx} pred_len={pred_len} dt={dt}")

    traj = integrate_lorenz96(n_ctx + pred_len, N=N, F=F, dt=dt, seed=seed, spinup=spinup)
    ctx_true = traj[:n_ctx]
    future_true = traj[n_ctx:n_ctx + pred_len]

    observed, mask = make_sparse_noisy(
        ctx_true, sparsity=scenario.sparsity, noise_std_frac=scenario.noise_std_frac,
        attractor_std=attr_std, seed=1000 * seed + hash(scenario.name) % 10000,
    )
    ctx_filled = linear_interp_fill(observed)

    results = {"future_true": future_true, "attr_std": attr_std, "lyap": lyap, "dt": dt}

    # Panda
    print("[predict] panda ...", flush=True)
    try:
        results["panda"] = panda_forecast(ctx_filled, pred_len=pred_len)
    except Exception as e:
        print(f"[predict] panda failed: {e}")
        results["panda"] = None

    # Parrot
    print("[predict] parrot ...", flush=True)
    results["parrot"] = context_parroting_forecast(ctx_filled, pred_len=pred_len)

    # Ours AR-Kalman (default L=5, m=128)
    print("[predict] ours_ark (default) ...", flush=True)
    try:
        results["ours_ark"] = full_pipeline_forecast(
            observed, pred_len=pred_len, seed=seed, bayes_calls=10, n_epochs=150,
        )
    except Exception as e:
        print(f"[predict] ours_ark failed: {e}")
        results["ours_ark"] = None

    # Ours AR-Kalman with L96-appropriate hyperparams (L=17 > 2*d_KY=16; m=512 for 100-D feat)
    print("[predict] ours_ark_L17 (L_embed=17, m_inducing=512) ...", flush=True)
    try:
        results["ours_ark_L17"] = full_pipeline_forecast(
            observed, pred_len=pred_len, seed=seed,
            L_embed=17, m_inducing=512, tau_max=40,
            bayes_calls=10, n_epochs=150,
        )
    except Exception as e:
        print(f"[predict] ours_ark_L17 failed: {e}")
        results["ours_ark_L17"] = None

    # Ours CSDI (if ckpt provided, default hyperparams)
    if csdi_ckpt is not None:
        print(f"[predict] ours_csdi ({csdi_ckpt.name}) ...", flush=True)
        from methods.csdi_impute_adapter import set_csdi_checkpoint
        set_csdi_checkpoint(str(csdi_ckpt))
        try:
            results["ours_csdi"] = full_pipeline_forecast(
                observed, pred_len=pred_len, seed=seed,
                imp_kind="csdi", bayes_calls=10, n_epochs=150,
            )
        except Exception as e:
            print(f"[predict] ours_csdi failed: {e}")
            results["ours_csdi"] = None

    return results


def plot(results: dict, scenario: HarshnessScenario, seed: int, tag: str,
         out_path: Path, N: int = 20):
    future_true = results["future_true"]
    attr_std = results["attr_std"]
    lyap = results["lyap"]
    dt = results["dt"]
    T = future_true.shape[0]
    t_lyap = np.arange(T) * dt * lyap

    methods = [
        ("ours_csdi", "ours_csdi (default L=5)", "#1b9e77"),
        ("ours_ark", "ours (AR-K, L=5 m=128)", "#e7298a"),
        ("ours_ark_L17", "ours (AR-K, L=17 m=512)", "#d95f02"),
        ("panda", "Panda-72M", "#9467bd"),
        ("parrot", "Parrot", "C0"),
    ]
    methods = [(k, lab, c) for (k, lab, c) in methods if results.get(k) is not None]
    n_methods = len(methods)

    # Figure: 2 rows (heatmap + line) × (n_methods + 1 truth)
    fig = plt.figure(figsize=(4 * (n_methods + 1), 8.5))
    gs = fig.add_gridspec(2, n_methods + 1, height_ratios=[1.2, 1], hspace=0.35, wspace=0.15)

    # --- Row 1: Heatmaps (time × dim) ---
    vmin, vmax = future_true.min(), future_true.max()
    all_arrs = [future_true] + [results[k] for k, _, _ in methods]
    vmin = min(a.min() for a in all_arrs)
    vmax = max(a.max() for a in all_arrs)
    for col, (label, data) in enumerate([("Truth", future_true)] + [(lab, results[k]) for k, lab, _ in methods]):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(data.T, aspect="auto", origin="lower",
                       extent=[0, t_lyap[-1], 0, N],
                       cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("time (Λ)")
        if col == 0:
            ax.set_ylabel("state dim (i of N=20)")
        else:
            ax.set_yticklabels([])
        # VPT@1.0 threshold line
        err = np.linalg.norm(data - future_true, axis=1) / (np.sqrt(N) * attr_std)
        bad = np.where(err > 1.0)[0]
        if len(bad) and label != "Truth":
            vpt_x = bad[0] * dt * lyap
            ax.axvline(vpt_x, color="black", lw=2, ls="--", alpha=0.7)
            ax.text(vpt_x + 0.1, N * 0.9, f"VPT@1.0\n={vpt_x:.2f}Λ",
                    fontsize=8, color="black", va="top")

    # Colorbar
    cbar = fig.colorbar(im, ax=fig.axes[:n_methods + 1], shrink=0.6, aspect=15, pad=0.02)
    cbar.set_label("L96 state value")

    # --- Row 2: Line plots of 3 representative dims ---
    rep_dims = [0, 10, 15]
    for col, d in enumerate(rep_dims):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(t_lyap, future_true[:, d], "k-", linewidth=2, label="truth", alpha=0.8)
        for k, lab, c in methods:
            ax.plot(t_lyap, results[k][:, d], color=c, linewidth=1.2, label=lab, alpha=0.75)
        ax.set_title(f"dim {d}", fontsize=10)
        ax.set_xlabel("time (Λ)")
        if col == 0:
            ax.set_ylabel("state value")
            ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="gray", lw=0.5)

    # Row 2 last col: error over time ||pred - truth|| / (√N σ_attr)
    ax = fig.add_subplot(gs[1, -1])
    for k, lab, c in methods:
        err = np.linalg.norm(results[k] - future_true, axis=1) / (np.sqrt(N) * attr_std)
        ax.plot(t_lyap, err, color=c, linewidth=1.5, label=lab, alpha=0.8)
    ax.axhline(1.0, color="red", lw=0.8, ls=":", label="VPT@1.0 threshold")
    ax.axhline(0.3, color="orange", lw=0.8, ls=":", label="VPT@0.3")
    ax.set_title("Normalized error over time", fontsize=10)
    ax.set_xlabel("time (Λ)")
    ax.set_ylabel("‖pred − truth‖ / (√D · σ_attr)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_yscale("log")

    fig.suptitle(f"L96 N={N} F=8 forecast vs truth — {scenario.name} (s={scenario.sparsity}, σ={scenario.noise_std_frac}) seed={seed} — {tag}",
                 fontsize=12)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"[saved] {out_path}")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S3", choices=list(SCENARIOS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None, help="L96 CSDI ckpt (optional)")
    ap.add_argument("--tag", default="default", help="label on plot")
    args = ap.parse_args()

    sc = SCENARIOS[args.scenario]
    csdi_ckpt = Path(args.ckpt) if args.ckpt else None
    results = run_all(sc, args.seed, csdi_ckpt)

    out = FIG / f"l96_forecast_{args.scenario}_seed{args.seed}_{args.tag}.png"
    plot(results, sc, args.seed, args.tag, out)


if __name__ == "__main__":
    main()
