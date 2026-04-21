"""Focused Module-4 experiment — the per-bin PICP story that tech.md §4 is about.

Unlike ``run_ablation.py`` (which calibrates CP per horizon and thus makes
Lyap-CP and Split-CP numerically equivalent), here we:

  1. Build a delay-coord dataset with **mixed horizons** h ∈ {1, 4, 16, 64}
  2. Train a single SVGP on the pooled dataset (with h as a feature)
  3. Calibrate **once** on a pooled calibration split
  4. Evaluate per-horizon PICP and MPIW on the test fold

Prediction under this setup: Split CP has per-horizon PICP that **decays** with
h (undercoverage at long horizons), while Lyap-CP's growth-rescaled scores
keep per-horizon PICP flat near the target 1-α.

Run:
    CUDA_VISIBLE_DEVICES=2 python -m experiments.week2_modules.module4_horizon_calibration
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.week1.lorenz63_utils import (
    LORENZ63_ATTRACTOR_STD,
    LORENZ63_LYAP,
    integrate_lorenz63,
    make_sparse_noisy,
)
from methods.dynamics_impute import impute
from methods.lyap_conformal import LyapConformal, SplitConformal
from methods.mi_lyap import global_lyapunov_rosenstein, mi_lyap_bayes_tau
from metrics.uq_metrics import mpiw, picp
from models.svgp import MultiOutputSVGP, SVGPConfig

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "experiments" / "week2_modules" / "results"
FIG_DIR = REPO_ROOT / "experiments" / "week2_modules" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 2, 4, 8, 16, 24, 32, 48]  # stay within ~1.1 Lyapunov time (dt=0.025, lam=0.906)
L_EMBED = 5
TAU_MAX = 30
N_CTX = 2000


def build_mixed_horizon_dataset(ctx_filled: np.ndarray, taus: np.ndarray, horizons: list[int]):
    """Return pooled (X, Y, h) with h as a feature column.

    Each row is ``(delay_features, horizon, y_true_at_t+h)``.
    """
    D = ctx_filled.shape[1]
    t0 = int(taus.max())
    T = ctx_filled.shape[0]

    feats_list, y_list, h_list = [], [], []
    for h in horizons:
        if t0 + h >= T:
            continue
        t_idx = np.arange(t0, T - h)
        cols = []
        for d in range(D):
            cols.append(ctx_filled[t_idx, d])
            for tau in taus:
                cols.append(ctx_filled[t_idx - tau, d])
        feats = np.stack(cols, axis=1)  # (n, L*D)
        # add horizon as a feature (log-scaled so SVGP learns the growth)
        feats = np.concatenate([feats, np.full((feats.shape[0], 1), np.log1p(h))], axis=1)
        Y = ctx_filled[t_idx + h, :]
        feats_list.append(feats); y_list.append(Y); h_list.append(np.full(feats.shape[0], h))

    X = np.concatenate(feats_list, axis=0)
    Y = np.concatenate(y_list, axis=0)
    H = np.concatenate(h_list, axis=0)
    return X.astype(np.float32), Y.astype(np.float32), H


def run(n_seeds: int, scenario_name: str, sparsity: float, noise: float, dt: float = 0.025,
        growth_modes: list[str] = ["exp", "saturating", "clipped", "empirical"]) -> dict:
    print(f"=== Module-4 horizon-cal experiment:  {scenario_name} (s={sparsity}, σ={noise}) seeds=[0..{n_seeds - 1}]"
          f" growth_modes={growth_modes}")
    per_horizon_picp_split: dict[int, list[float]] = {h: [] for h in HORIZONS}
    per_horizon_mpiw_split: dict[int, list[float]] = {h: [] for h in HORIZONS}
    per_horizon_picp: dict[str, dict[int, list[float]]] = {m: {h: [] for h in HORIZONS} for m in growth_modes}
    per_horizon_mpiw: dict[str, dict[int, list[float]]] = {m: {h: [] for h in HORIZONS} for m in growth_modes}
    q_records = []

    for seed in range(n_seeds):
        traj = integrate_lorenz63(N_CTX, dt=dt, seed=seed, spinup=2000)
        obs, mask = make_sparse_noisy(traj, sparsity=sparsity, noise_std_frac=noise, seed=seed)
        ctx_filled = impute(obs, kind="ar_kalman")
        # Use theoretical Lyapunov for this calibration study; robustness to
        # noisy nolds estimates is a separate ablation (future work).
        lam_true = LORENZ63_LYAP
        lam_step = lam_true * dt

        spec = mi_lyap_bayes_tau(ctx_filled[:, 0], L=L_EMBED, tau_max=TAU_MAX, horizon=1,
                                 lam=lam_step, n_calls=15, k=4, seed=seed)
        taus = spec.taus
        lam_hat = lam_true
        print(f"  seed={seed}  lam(true)={lam_true:.3f} lam_step={lam_step:.4f}  τ={taus.tolist()}")

        X, Y, H = build_mixed_horizon_dataset(ctx_filled, taus, HORIZONS)
        # split 60/20/20
        n = X.shape[0]
        idx = np.random.default_rng(seed).permutation(n)
        n_tr = int(0.6 * n); n_cal = int(0.2 * n)
        tr = idx[:n_tr]; cal = idx[n_tr : n_tr + n_cal]; te = idx[n_tr + n_cal :]

        gp = MultiOutputSVGP(SVGPConfig(m_inducing=128, n_epochs=150, verbose=False)).fit(X[tr], Y[tr])
        mu_cal, std_cal = gp.predict(X[cal], return_std=True)
        mu_te, std_te = gp.predict(X[te], return_std=True)

        # Calibrate Split-CP + each Lyap-CP growth variant on pooled cal set
        H_cal = np.repeat(H[cal][:, None], Y.shape[1], axis=1).astype(float)
        H_te = np.repeat(H[te][:, None], Y.shape[1], axis=1).astype(float)

        split_cp = SplitConformal(alpha=0.1)
        split_cp.calibrate(Y[cal], mu_cal, std_cal)
        lo_s, hi_s = split_cp.predict_interval(mu_te, std_te)
        q_record = dict(seed=seed, q_split=float(split_cp.q), lam_hat=float(lam_hat))

        for sel in HORIZONS:
            mask = H[te] == sel
            if mask.sum():
                per_horizon_picp_split[sel].append(picp(Y[te][mask], lo_s[mask], hi_s[mask]))
                per_horizon_mpiw_split[sel].append(mpiw(lo_s[mask], hi_s[mask]))

        for mode in growth_modes:
            lyap_cp = LyapConformal(alpha=0.1, lam=lam_step, dt=1.0,
                                    growth_mode=mode, growth_cap=10.0)
            lyap_cp.calibrate(Y[cal], mu_cal, std_cal, H_cal)
            lo_l, hi_l = lyap_cp.predict_interval(mu_te, std_te, H_te)
            q_record[f"q_lyap_{mode}"] = float(lyap_cp.q)
            for sel in HORIZONS:
                mask = H[te] == sel
                if mask.sum():
                    per_horizon_picp[mode][sel].append(picp(Y[te][mask], lo_l[mask], hi_l[mask]))
                    per_horizon_mpiw[mode][sel].append(mpiw(lo_l[mask], hi_l[mask]))

        q_records.append(q_record)

    summary = dict(
        scenario=dict(name=scenario_name, sparsity=sparsity, noise=noise),
        horizons=HORIZONS,
        picp=dict(lyap={h: (float(np.mean(v)), float(np.std(v))) for h, v in per_horizon_picp_lyap.items() if v},
                  split={h: (float(np.mean(v)), float(np.std(v))) for h, v in per_horizon_picp_split.items() if v}),
        mpiw=dict(lyap={h: (float(np.mean(v)), float(np.std(v))) for h, v in per_horizon_mpiw_lyap.items() if v},
                  split={h: (float(np.mean(v)), float(np.std(v))) for h, v in per_horizon_mpiw_split.items() if v}),
        q_records=q_records,
    )
    return summary


def plot(summary: dict, fig_path: Path) -> None:
    horizons = summary["horizons"]
    picp_lyap = [summary["picp"]["lyap"].get(h, (np.nan, 0))[0] for h in horizons]
    picp_split = [summary["picp"]["split"].get(h, (np.nan, 0))[0] for h in horizons]
    picp_lyap_std = [summary["picp"]["lyap"].get(h, (np.nan, 0))[1] for h in horizons]
    picp_split_std = [summary["picp"]["split"].get(h, (np.nan, 0))[1] for h in horizons]

    mpiw_lyap = [summary["mpiw"]["lyap"].get(h, (np.nan, 0))[0] for h in horizons]
    mpiw_split = [summary["mpiw"]["split"].get(h, (np.nan, 0))[0] for h in horizons]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.errorbar(horizons, picp_lyap, yerr=picp_lyap_std, marker="o", color="#1b9e77",
                 label="Lyap-CP", linewidth=2, capsize=3)
    ax1.errorbar(horizons, picp_split, yerr=picp_split_std, marker="s", color="#d95f02",
                 label="Split CP", linewidth=2, capsize=3)
    ax1.axhline(0.9, color="red", linestyle=":", label="target 0.90")
    ax1.set_xscale("log"); ax1.set_xlabel("Forecast horizon (steps)")
    ax1.set_ylabel("Per-horizon PICP (target 0.90)")
    ax1.set_title("Calibration under mixed-horizon conformal")
    ax1.grid(True, alpha=0.3); ax1.legend()

    ax2.plot(horizons, mpiw_lyap, marker="o", color="#1b9e77", label="Lyap-CP", linewidth=2)
    ax2.plot(horizons, mpiw_split, marker="s", color="#d95f02", label="Split CP", linewidth=2)
    ax2.set_xscale("log"); ax2.set_xlabel("Forecast horizon (steps)")
    ax2.set_ylabel("MPIW")
    ax2.set_title("Interval width grows with horizon")
    ax2.grid(True, alpha=0.3); ax2.legend()

    sc = summary["scenario"]
    fig.suptitle(f"Lorenz63 ({sc['name']}, sparsity={sc['sparsity']}, σ={sc['noise']}) — pooled-horizon CP",
                 y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {fig_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--scenario", default="S3")
    args = ap.parse_args()

    sc_map = {"S2": (0.4, 0.3), "S3": (0.6, 0.5), "S4": (0.75, 0.8)}
    sparsity, noise = sc_map[args.scenario]
    summary = run(args.n_seeds, args.scenario, sparsity, noise)
    out = OUT_DIR / f"module4_horizon_cal_{args.scenario}_n{args.n_seeds}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[saved] {out}")
    plot(summary, FIG_DIR / f"module4_horizon_cal_{args.scenario}.png")

    # Verdict
    print("\n[verdict]  per-horizon PICP:")
    print(f"  {'h':>5}  {'Lyap':>12}  {'Split':>12}")
    target = 0.9
    dev_lyap = []; dev_split = []
    for h in HORIZONS:
        l = summary["picp"]["lyap"].get(h, (np.nan, 0))
        s = summary["picp"]["split"].get(h, (np.nan, 0))
        print(f"  {h:>5}  {l[0]:>6.3f}±{l[1]:.2f}  {s[0]:>6.3f}±{s[1]:.2f}")
        dev_lyap.append(abs(l[0] - target)); dev_split.append(abs(s[0] - target))
    print(f"\n  mean |PICP - 0.90|:  Lyap={np.mean(dev_lyap):.3f}   Split={np.mean(dev_split):.3f}")
    print(f"  max  |PICP - 0.90|:  Lyap={np.max(dev_lyap):.3f}   Split={np.max(dev_split):.3f}")


if __name__ == "__main__":
    main()
