"""Figure D2 — Coverage Across Harshness (paper Figure 2).

Runs full CP pipeline (AR-Kalman → MI-Lyap τ → SVGP → Lyap-empirical CP) on all
7 harshness scenarios S0-S6 and reports PICP@h=1, PICP@h=4, MPIW. Two main
methods compared: Split CP vs Lyap-empirical CP.

Claim: Lyap-empirical CP 在所有 harshness 下都贴 0.90 (均匀 coverage)，
而 Split CP 在高 harshness 下 under-cover。

Run:
    python -m experiments.week2_modules.coverage_across_harshness --n_seeds 3
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
    LORENZ63_ATTRACTOR_STD, LORENZ63_LYAP, PILOT_SCENARIOS,
    integrate_lorenz63, make_sparse_noisy,
)
from methods.dynamics_impute import impute
from methods.lyap_conformal import LyapConformal, SplitConformal
from methods.mi_lyap import (
    global_lyapunov_rosenstein, mi_lyap_bayes_tau, robust_lyapunov,
)
from metrics.uq_metrics import mpiw, picp
from models.svgp import MultiOutputSVGP, SVGPConfig

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "experiments" / "week2_modules" / "results"
FIG_DIR = REPO_ROOT / "experiments" / "week2_modules" / "figures"

HORIZONS = [1, 4, 16]
L_EMBED = 5
TAU_MAX = 30
N_CTX = 1500


def build_ds(ctx_filled, taus, horizons):
    D = ctx_filled.shape[1]
    t0 = int(taus.max())
    T = ctx_filled.shape[0]
    feats, ys, hs = [], [], []
    for h in horizons:
        if t0 + h >= T:
            continue
        t_idx = np.arange(t0, T - h)
        cols = [ctx_filled[t_idx, d] for d in range(D)]
        for d in range(D):
            for tau in taus:
                cols.append(ctx_filled[t_idx - tau, d])
        x = np.stack(cols, axis=1)
        x = np.concatenate([x, np.full((x.shape[0], 1), np.log1p(h))], axis=1)
        feats.append(x); ys.append(ctx_filled[t_idx + h, :]); hs.append(np.full(x.shape[0], h))
    return (np.concatenate(feats, 0).astype(np.float32),
            np.concatenate(ys, 0).astype(np.float32),
            np.concatenate(hs, 0))


def run_one(scenario, seed: int, alpha: float = 0.1) -> dict:
    traj = integrate_lorenz63(N_CTX, dt=0.025, seed=seed, spinup=2000)
    obs, _ = make_sparse_noisy(traj, sparsity=scenario.sparsity,
                               noise_std_frac=scenario.noise_std_frac, seed=seed)
    try:
        ctx_filled = impute(obs, kind="ar_kalman")
    except Exception:
        from experiments.week1.lorenz63_utils import linear_interp_fill
        ctx_filled = linear_interp_fill(obs)

    lam_est = robust_lyapunov(ctx_filled[:, 0], dt=0.025, emb_dim=5, lag=2,
                              trajectory_len=50, prefilter=True)
    spec = mi_lyap_bayes_tau(ctx_filled[:, 0], L=L_EMBED, tau_max=TAU_MAX, horizon=1,
                             lam=lam_est * 0.025, n_calls=12, k=4, seed=seed)
    X, Y, H = build_ds(ctx_filled, spec.taus, HORIZONS)
    n = X.shape[0]; idx = np.random.default_rng(seed).permutation(n)
    n_tr = int(0.6 * n); n_cal = int(0.2 * n)
    tr = idx[:n_tr]; cal = idx[n_tr:n_tr + n_cal]; te = idx[n_tr + n_cal:]
    gp = MultiOutputSVGP(SVGPConfig(m_inducing=128, n_epochs=150, verbose=False)).fit(X[tr], Y[tr])
    mu_cal, std_cal = gp.predict(X[cal], return_std=True)
    mu_te,  std_te  = gp.predict(X[te],  return_std=True)

    H_cal = np.repeat(H[cal][:, None], Y.shape[1], axis=1).astype(float)
    H_te  = np.repeat(H[te][:, None],  Y.shape[1], axis=1).astype(float)

    out = dict(scenario=scenario.name, sparsity=scenario.sparsity,
               noise_frac=scenario.noise_std_frac, seed=seed, lam=float(lam_est))

    # Split CP
    split = SplitConformal(alpha=alpha)
    split.calibrate(Y[cal], mu_cal, std_cal)
    lo_s, hi_s = split.predict_interval(mu_te, std_te)

    # Lyap-empirical CP
    lyap = LyapConformal(alpha=alpha, lam=lam_est * 0.025, dt=1.0, growth_mode="empirical", growth_cap=10.0)
    lyap.calibrate(Y[cal], mu_cal, std_cal, H_cal)
    lo_l, hi_l = lyap.predict_interval(mu_te, std_te, H_te)

    for h in HORIZONS:
        m = H[te] == h
        if not m.any():
            continue
        out[f"split_picp_h{h}"] = float(picp(Y[te][m], lo_s[m], hi_s[m]))
        out[f"split_mpiw_h{h}"] = float(mpiw(lo_s[m], hi_s[m]))
        out[f"lyap_picp_h{h}"]  = float(picp(Y[te][m], lo_l[m], hi_l[m]))
        out[f"lyap_mpiw_h{h}"]  = float(mpiw(lo_l[m], hi_l[m]))

    return out


def plot(all_recs: list[dict], out_path: Path) -> None:
    scenarios = sorted({r["scenario"] for r in all_recs})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, h in zip(axes, HORIZONS):
        split_m, split_s, lyap_m, lyap_s = [], [], [], []
        for sc in scenarios:
            sub = [r for r in all_recs if r["scenario"] == sc and f"split_picp_h{h}" in r]
            if not sub:
                split_m.append(np.nan); split_s.append(0); lyap_m.append(np.nan); lyap_s.append(0); continue
            sp = np.array([r[f"split_picp_h{h}"] for r in sub])
            ly = np.array([r[f"lyap_picp_h{h}"]  for r in sub])
            split_m.append(sp.mean()); split_s.append(sp.std())
            lyap_m.append(ly.mean());  lyap_s.append(ly.std())
        x = np.arange(len(scenarios))
        ax.axhline(0.90, color="k", linestyle="--", linewidth=1.0, alpha=0.55,
                   label="nominal 0.90")
        ax.errorbar(x, split_m, yerr=split_s, marker="s", linewidth=1.8,
                    color="#888888", label="Split CP", markersize=7, capsize=3)
        ax.errorbar(x, lyap_m, yerr=lyap_s, marker="o", linewidth=2.2,
                    color="#e7298a", label="Lyap-empirical (ours)", markersize=7, capsize=3)
        ax.set_xticks(x); ax.set_xticklabels(scenarios)
        ax.set_xlabel("Harshness scenario")
        ax.set_title(f"h = {h}", fontsize=11)
        ax.set_ylim(0.3, 1.05)
        ax.grid(True, alpha=0.25)
        if h == HORIZONS[0]:
            ax.set_ylabel("PICP  (target 0.90)")
            ax.legend(fontsize=9, loc="lower left")
    plt.suptitle("Figure D2 — Coverage Across Harshness  (Lorenz63, S0→S6)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--tag", default="n3_v1")
    args = ap.parse_args()

    all_recs = []
    for sc in PILOT_SCENARIOS:
        print(f"\n=== scenario {sc.name} (sparsity={sc.sparsity}, σ={sc.noise_std_frac}) ===")
        for seed in range(args.n_seeds):
            rec = run_one(sc, seed)
            all_recs.append(rec)
            h1_s = rec.get("split_picp_h1", np.nan); h1_l = rec.get("lyap_picp_h1", np.nan)
            print(f"  seed={seed}  h1 PICP split={h1_s:.2f}  lyap-emp={h1_l:.2f}  λ_hat={rec['lam']:.2f}")

    out_json = OUT_DIR / f"coverage_across_harshness_{args.tag}.json"
    out_json.write_text(json.dumps(all_recs, indent=2))
    print(f"[saved] {out_json}")
    plot(all_recs, FIG_DIR / "coverage_across_harshness_paperfig.png")


if __name__ == "__main__":
    main()
