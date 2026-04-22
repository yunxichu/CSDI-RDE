"""Figure D5 — Reliability diagram (nominal vs empirical coverage), pre/post CP.

Reuses the same pipeline as ``module4_horizon_calibration.py`` but only at h=1,
and sweeps α ∈ {0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50}. For each α, compute
empirical PICP under three scoring schemes:

  1. **Raw Gaussian**  — no CP, pretend mu ± z_{1-α/2}·σ is a (1-α) PI
  2. **Split CP**       — calibrated on a held-out split
  3. **Lyap-empirical CP** — the paper's main CP method

A well-calibrated method should lie on y=x (nominal vs observed).

Run:
    CUDA_VISIBLE_DEVICES=2 python -m experiments.week2_modules.reliability_diagram \
        --n_seeds 3 --scenarios S2 S3
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
from scipy.stats import norm

from experiments.week1.lorenz63_utils import (
    LORENZ63_ATTRACTOR_STD, LORENZ63_LYAP,
    integrate_lorenz63, make_sparse_noisy,
)
from methods.dynamics_impute import impute
from methods.lyap_conformal import LyapConformal, SplitConformal
from methods.mi_lyap import (
    global_lyapunov_rosenstein, mi_lyap_bayes_tau, robust_lyapunov,
)
from metrics.uq_metrics import picp
from models.svgp import MultiOutputSVGP, SVGPConfig

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "experiments" / "week2_modules" / "results"
FIG_DIR = REPO_ROOT / "experiments" / "week2_modules" / "figures"

SCENARIOS = {
    "S2": dict(sparsity=0.4, noise=0.3),
    "S3": dict(sparsity=0.6, noise=0.5),
}
ALPHAS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
HORIZONS = [1, 4, 16, 48]   # keep for consistency with module4
H_FOCUS = 1                 # reliability computed at h=1 only
L_EMBED = 5
TAU_MAX = 30
N_CTX = 2000


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


def run_one_seed(scenario: str, seed: int) -> dict:
    cfg = SCENARIOS[scenario]
    traj = integrate_lorenz63(N_CTX, dt=0.025, seed=seed, spinup=2000)
    obs, _ = make_sparse_noisy(traj, sparsity=cfg["sparsity"], noise_std_frac=cfg["noise"], seed=seed)
    ctx_filled = impute(obs, kind="ar_kalman")
    lam_est = robust_lyapunov(ctx_filled[:, 0], dt=0.025, emb_dim=5, lag=2,
                              trajectory_len=50, prefilter=True)
    spec = mi_lyap_bayes_tau(ctx_filled[:, 0], L=L_EMBED, tau_max=TAU_MAX, horizon=1,
                             lam=lam_est * 0.025, n_calls=15, k=4, seed=seed)
    X, Y, H = build_ds(ctx_filled, spec.taus, HORIZONS)
    n = X.shape[0]; idx = np.random.default_rng(seed).permutation(n)
    n_tr = int(0.6 * n); n_cal = int(0.2 * n)
    tr = idx[:n_tr]; cal = idx[n_tr : n_tr + n_cal]; te = idx[n_tr + n_cal :]
    gp = MultiOutputSVGP(SVGPConfig(m_inducing=128, n_epochs=150, verbose=False)).fit(X[tr], Y[tr])
    mu_cal, std_cal = gp.predict(X[cal], return_std=True)
    mu_te,  std_te  = gp.predict(X[te],  return_std=True)

    # Focus on h=1 only for reliability
    mask_cal = (H[cal] == H_FOCUS)
    mask_te  = (H[te]  == H_FOCUS)
    y_cal  = Y[cal][mask_cal]; mu_c = mu_cal[mask_cal]; std_c = std_cal[mask_cal]
    y_te   = Y[te][mask_te];   mu_t = mu_te [mask_te];  std_t = std_te [mask_te]

    # Compute standardized residuals on calibration set (for Split CP)
    res_cal = np.abs(y_cal - mu_c) / np.maximum(std_c, 1e-8)       # shape (n_cal, D)
    z_te    = (y_te - mu_t) / np.maximum(std_t, 1e-8)

    out = dict(seed=seed, scenario=scenario, lam_hat=float(lam_est))
    for alpha in ALPHAS:
        # 1) Raw Gaussian PI (no CP)
        z_q = norm.ppf(1 - alpha / 2)
        lo_raw = mu_t - z_q * std_t; hi_raw = mu_t + z_q * std_t
        picp_raw = picp(y_te, lo_raw, hi_raw)

        # 2) Split CP at α
        split_q = float(np.quantile(res_cal.flatten(), 1 - alpha))
        lo_s = mu_t - split_q * std_t; hi_s = mu_t + split_q * std_t
        picp_split = picp(y_te, lo_s, hi_s)

        # 3) Lyap-empirical CP at α (same formula restricted to h=1 → no growth scaling needed)
        #    At h=1, Lyap-empirical reduces to Split CP anyway. Paper story lives at h>1,
        #    but reliability@h=1 uses Split as proxy. For fairness we report Split only here.
        out[f"alpha_{alpha}"] = dict(
            picp_raw=float(picp_raw),
            picp_split=float(picp_split),
        )
    return out


def plot(summary: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5), sharey=True)
    for ax, sc in zip(axes, sorted(summary.keys())):
        recs = summary[sc]
        alphas = ALPHAS
        nominal = np.array([1 - a for a in alphas])
        emp_raw   = np.array([[r[f"alpha_{a}"]["picp_raw"]    for a in alphas] for r in recs])
        emp_split = np.array([[r[f"alpha_{a}"]["picp_split"]  for a in alphas] for r in recs])

        m_raw, s_raw = emp_raw.mean(0),   emp_raw.std(0)
        m_spl, s_spl = emp_split.mean(0), emp_split.std(0)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.6, label="perfectly calibrated")
        ax.errorbar(nominal, m_raw, yerr=s_raw, marker="o", color="#d95f02",
                    label="Raw Gaussian (pre-CP)", linewidth=1.8, markersize=6)
        ax.errorbar(nominal, m_spl, yerr=s_spl, marker="s", color="#1b9e77",
                    label="Split CP (post-CP)", linewidth=1.8, markersize=6)
        ax.set_title(f"Scenario {sc}  (h={H_FOCUS})", fontsize=11)
        ax.set_xlabel("nominal coverage  1−α")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0.4, 1.02); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("empirical PICP")
    axes[0].legend(fontsize=8.5, loc="upper left", framealpha=0.85)
    plt.suptitle("Figure D5 — Reliability Diagram  (pre vs post Split-CP)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--scenarios", nargs="+", default=["S2", "S3"])
    ap.add_argument("--tag", default="n3_v1")
    args = ap.parse_args()

    all_summary = {}
    for sc in args.scenarios:
        print(f"\n=== Reliability diagram @ {sc}  (α ∈ {ALPHAS})")
        recs = [run_one_seed(sc, seed) for seed in range(args.n_seeds)]
        all_summary[sc] = recs

    out_json = OUT_DIR / f"reliability_diagram_{args.tag}.json"
    out_json.write_text(json.dumps(all_summary, indent=2))
    print(f"[saved] {out_json}")
    plot(all_summary, FIG_DIR / f"reliability_diagram_paperfig.png")


if __name__ == "__main__":
    main()
