"""DeepEDM sanity test: parallel to svgp_l63_vs_l96_sanity.py (M3 comparison).

Isolates M3 behaviour with clean truth context (no M1 / M2). Measures whether
the DeepEDM backbone holds up in 1-step and autoregressive rollout on the
L63 (3-D, 15-D features) and L96 N=20 (20-D, 100-D features) regimes.

The SVGP version of this script showed L96 1-step α jumps from 0.375 (m=128)
to 0.813 (m=500) but still collapses during AR rollout (α→0.07 by h=20). If
DeepEDM holds α > 0.5 at h=10-20 on L96, the pipeline swap is justified.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from experiments.week1.lorenz63_utils import integrate_lorenz63
from experiments.week1.lorenz96_utils import integrate_lorenz96
from models.deep_edm import DeepEDMPredictor, DeepEDMConfig


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


def run_system(name: str, traj_fn, taus: np.ndarray, n_ctx: int = 512,
               pred_len: int = 128, n_epochs: int = 300, seed: int = 0,
               d_model: int = 64, n_layers: int = 2):
    traj = traj_fn(n_ctx + pred_len, seed=seed)
    ctx = traj[:n_ctx]
    future = traj[n_ctx:]
    D = ctx.shape[1]
    print(f"\n--- {name} D={D} n_ctx={n_ctx} taus={list(taus)} ---")
    print(f"  ctx std={ctx.std():.3f}  range=[{ctx.min():.2f},{ctx.max():.2f}]")

    X, Y = build_features(ctx, taus)
    print(f"  input X.shape={X.shape}  Y.shape={Y.shape}  (n_train={X.shape[0]}, feat_dim={X.shape[1]})")

    net = DeepEDMPredictor(DeepEDMConfig(
        d_model=d_model, n_heads=4, n_layers=n_layers,
        n_epochs=n_epochs, batch_size=256, lr=1e-3,
        patience=50, verbose=True,
    )).fit(X, Y)

    q1 = query(ctx, taus)
    mu1, _ = net.predict(q1, return_std=True)
    err1 = np.sqrt(((mu1[0] - future[0]) ** 2).mean())
    print(f"  [1-step] pred={mu1[0][:3]} ... truth={future[0][:3]} ... rmse={err1:.3f}")
    alpha1 = float((mu1[0] * future[0]).sum() / (future[0] ** 2).sum())
    print(f"    scale α=<pred,truth>/<truth,truth> = {alpha1:.4f}")

    history = ctx.copy()
    preds = np.empty((pred_len, D), dtype=np.float32)
    for h in range(pred_len):
        q = query(history, taus)
        mu, _ = net.predict(q, return_std=True)
        preds[h] = mu[0]
        history = np.vstack([history, mu[0][None, :]])

    for h in [0, 1, 3, 5, 10, 20, 50, 100]:
        if h < pred_len:
            err = np.sqrt(((preds[h] - future[h]) ** 2).mean())
            alpha = float((preds[h] * future[h]).sum() / ((future[h] ** 2).sum() + 1e-8))
            print(f"  [h={h:3d}] pred_norm={np.linalg.norm(preds[h]):.2f}  "
                  f"truth_norm={np.linalg.norm(future[h]):.2f}  rmse={err:.3f}  α={alpha:.3f}")

    return ctx, future, preds


def l63_fn(n, seed): return integrate_lorenz63(n, dt=0.025, seed=seed, spinup=2000)
def l96_fn(n, seed): return integrate_lorenz96(n, N=20, F=8.0, dt=0.05, seed=seed, spinup=2000)


if __name__ == "__main__":
    l63_ctx, l63_fut, l63_pred = run_system(
        "L63 (D=3, 15-D features)", l63_fn, np.array([4, 3, 2, 1]),
        d_model=64, n_layers=2, n_epochs=300,
    )
    l96_ctx, l96_fut, l96_pred = run_system(
        "L96-N=20 (D=20, 100-D features)", l96_fn, np.array([4, 3, 2, 1]),
        d_model=128, n_layers=3, n_epochs=400,
    )
    print("\n=== SUMMARY (DeepEDM) ===")
    for name, pred, fut in [("L63", l63_pred, l63_fut), ("L96", l96_pred, l96_fut)]:
        for h in [0, 1, 5, 10, 20]:
            if h < pred.shape[0]:
                alpha = float((pred[h] * fut[h]).sum() / ((fut[h] ** 2).sum() + 1e-8))
                print(f"{name} h={h:3d}: α={alpha:.3f}  pred_norm={np.linalg.norm(pred[h]):.2f}")

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    for r, (name, ctx, fut, pred) in enumerate([
        ("L63 (D=3)", l63_ctx, l63_fut, l63_pred),
        ("L96 (D=20)", l96_ctx, l96_fut, l96_pred),
    ]):
        D = fut.shape[1]
        rep_dims = [0, D // 2, D - 1] if D > 3 else [0, 1, 2]
        for c, d in enumerate(rep_dims):
            ax = axes[r, c]
            ax.plot(fut[:, d], "k-", linewidth=1.5, label="truth", alpha=0.7)
            ax.plot(pred[:, d], "r-", linewidth=1, label="DeepEDM rollout", alpha=0.8)
            ax.set_title(f"{name} dim {d}", fontsize=10)
            ax.grid(alpha=0.3)
            if c == 0:
                ax.legend(fontsize=9)

    plt.suptitle("DeepEDM-on-truth sanity: L63 vs L96 forecast", fontsize=12)
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "figures" / "deepedm_l63_vs_l96_sanity.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n[saved] {out}")
