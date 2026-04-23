"""Paper figure: M3 backbone head-to-head on L96 (SVGP vs DeepEDM vs FNO).

Runs the three predictors on the SAME truth context (no M1/M2 — isolates M3)
and plots α = <pred, truth>/||truth||² vs horizon h. This is the §5.7 figure
motivating the pipeline swap: SVGP collapses, DeepEDM/FNO hold up.

Light-weight enough (~3 min on single GPU) to re-run before each paper revision.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from experiments.week1.lorenz96_utils import integrate_lorenz96
from models.svgp import MultiOutputSVGP, SVGPConfig
from models.deep_edm import DeepEDMPredictor, DeepEDMConfig
from models.fno_delay import FNOPredictor, FNOConfig


def build_features(traj, taus):
    T, D = traj.shape
    t0 = int(taus.max())
    t_idx = np.arange(t0, T - 1)
    feats = []
    for d in range(D):
        cols = [traj[t_idx, d]]
        for tau in taus:
            cols.append(traj[t_idx - tau, d])
        feats.append(np.stack(cols, axis=1))
    X = np.concatenate(feats, axis=1).astype(np.float32)
    Y = traj[t_idx + 1].astype(np.float32)
    return X, Y


def query(history, taus):
    t = history.shape[0] - 1
    D = history.shape[1]
    feats = []
    for d in range(D):
        cols = [history[t, d]]
        for tau in taus:
            cols.append(history[t - tau, d])
        feats.append(np.asarray(cols, dtype=np.float32))
    return np.concatenate(feats, axis=0).reshape(1, -1)


def rollout(net, ctx, taus, pred_len):
    history = ctx.copy()
    D = ctx.shape[1]
    preds = np.empty((pred_len, D), dtype=np.float32)
    for h in range(pred_len):
        q = query(history, taus)
        mu, _ = net.predict(q, return_std=True)
        preds[h] = mu[0]
        history = np.vstack([history, mu[0][None, :]])
    return preds


def alphas(preds, fut):
    return np.array([
        float((preds[h] * fut[h]).sum() / ((fut[h] ** 2).sum() + 1e-8))
        for h in range(len(preds))
    ])


def run_one(seed: int, n_ctx: int = 512, pred_len: int = 128):
    traj = integrate_lorenz96(n_ctx + pred_len, N=20, F=8.0, dt=0.05,
                              spinup=2000, seed=seed)
    ctx = traj[:n_ctx]; fut = traj[n_ctx:]
    taus = np.array([4, 3, 2, 1])
    X, Y = build_features(ctx, taus)

    # SVGP with m_inducing=500 (the post-fix setting)
    svgp = MultiOutputSVGP(SVGPConfig(
        m_inducing=min(X.shape[0] - 1, 500), n_epochs=300, lr=1e-2, verbose=False,
    )).fit(X, Y)
    dm = DeepEDMPredictor(DeepEDMConfig(
        d_model=128, n_heads=4, n_layers=3, n_epochs=400, batch_size=256,
        lr=1e-3, patience=50, verbose=False,
    )).fit(X, Y)
    fno = FNOPredictor(FNOConfig(
        width=64, modes=3, n_layers=3, n_epochs=400, batch_size=256,
        lr=1e-3, patience=50, verbose=False,
    )).fit(X, Y)

    return dict(
        fut=fut,
        svgp=rollout(svgp, ctx, taus, pred_len),
        deepedm=rollout(dm, ctx, taus, pred_len),
        fno=rollout(fno, ctx, taus, pred_len),
    )


if __name__ == "__main__":
    seeds = [0, 1, 2]
    alpha_curves = {"svgp": [], "deepedm": [], "fno": []}
    print(f"Running L96 N=20 M3 backbone comparison on seeds {seeds}...")
    for s in seeds:
        print(f"  seed={s}")
        r = run_one(s)
        for bk in alpha_curves:
            alpha_curves[bk].append(alphas(r[bk], r["fut"]))
    for bk in alpha_curves:
        alpha_curves[bk] = np.stack(alpha_curves[bk], axis=0)  # (n_seeds, pred_len)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    horizons = np.arange(alpha_curves["svgp"].shape[1])
    colors = {"svgp": "C0", "deepedm": "C3", "fno": "C2"}
    labels = {"svgp": "SVGP (Matérn-5/2, m=500)", "deepedm": "DeepEDM (ICML 2025)",
              "fno": "FNO (1D spectral)"}
    for bk, mat in alpha_curves.items():
        m = mat.mean(axis=0); s = mat.std(axis=0)
        ax.plot(horizons, m, "-", label=labels[bk], color=colors[bk], linewidth=2)
        ax.fill_between(horizons, m - s, m + s, color=colors[bk], alpha=0.15)
    ax.axhline(1.0, color="k", linestyle=":", linewidth=0.8, label="α=1 (perfect)")
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.6)
    ax.set_xlabel("horizon h (steps)")
    ax.set_ylabel(r"scale α = $\langle \hat y, y \rangle / \|y\|^2$")
    ax.set_title(f"L96 N=20, D=20, 100-D delay features — M3 backbone comparison "
                 f"(mean ± std over {len(seeds)} seeds)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, horizons[-1])
    ax.set_ylim(-0.3, 1.15)
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "figures" / "m3_backbone_comparison_l96.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[saved] {out}")

    # Numerical summary for paper table
    print(f"\n=== α by backbone × horizon (mean ± std, {len(seeds)} seeds) ===")
    print(f"{'h':>4s}  {'svgp':>14s}  {'deepedm':>14s}  {'fno':>14s}")
    for h in [0, 1, 5, 10, 20, 50, 100]:
        if h < horizons[-1] + 1:
            row = [f"{h:4d}"]
            for bk in ["svgp", "deepedm", "fno"]:
                m = alpha_curves[bk][:, h].mean()
                s = alpha_curves[bk][:, h].std()
                row.append(f"{m:+.3f}±{s:.3f}")
            print("  ".join(row))
