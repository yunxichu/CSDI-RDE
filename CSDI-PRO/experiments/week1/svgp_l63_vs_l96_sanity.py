"""Minimal SVGP sanity test: L63 and L96 side-by-side, using TRUTH ctx (no M1/M2).

Isolates M3 (SVGP on delay coords). If L63 works and L96 fails, confirms M3
architecture issue. If both work, bug is elsewhere (M1/M2).

For each system, we:
  1. Generate a clean trajectory T=512 (ctx) + 128 (future).
  2. Build delay features with manually-picked taus (avoid M2 bias).
  3. Fit SVGP on ctx.
  4. Predict 1-step on LAST ctx anchor → compare to truth ctx[T-1] step+1 = future[0].
  5. Also do 10-step autoregressive rollout → compare to future[0..10].
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from experiments.week1.lorenz63_utils import integrate_lorenz63
from experiments.week1.lorenz96_utils import integrate_lorenz96, lorenz96_attractor_std
from models.svgp import MultiOutputSVGP, SVGPConfig


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
               pred_len: int = 128, m_inducing: int = 128, n_epochs: int = 100,
               seed: int = 0):
    traj = traj_fn(n_ctx + pred_len, seed=seed)
    ctx = traj[:n_ctx]
    future = traj[n_ctx:]
    D = ctx.shape[1]
    print(f"\n--- {name} D={D} n_ctx={n_ctx} taus={list(taus)} ---")
    print(f"  ctx std={ctx.std():.3f}  range=[{ctx.min():.2f},{ctx.max():.2f}]")

    X, Y = build_features(ctx, taus)
    print(f"  SVGP input X.shape={X.shape}  Y.shape={Y.shape}  (n_train={X.shape[0]}, feat_dim={X.shape[1]})")

    gp = MultiOutputSVGP(SVGPConfig(m_inducing=m_inducing, n_epochs=n_epochs,
                                      lr=1e-2, verbose=False)).fit(X, Y)

    # 1-step: predict ctx[T-1] → should be future[0]
    q1 = query(ctx, taus)
    mu1, _ = gp.predict(q1, return_std=True)
    err1 = np.sqrt(((mu1[0] - future[0]) ** 2).mean())
    print(f"  [1-step] pred={mu1[0][:3]} ... truth={future[0][:3]} ... rmse={err1:.3f}")
    print(f"    scale α=<pred,truth>/<truth,truth> = {float((mu1[0]*future[0]).sum()/(future[0]**2).sum()):.4f}")

    # Rollout 10 steps
    history = ctx.copy()
    preds = np.empty((pred_len, D), dtype=np.float32)
    for h in range(pred_len):
        q = query(history, taus)
        mu, _ = gp.predict(q, return_std=True)
        preds[h] = mu[0]
        history = np.vstack([history, mu[0][None, :]])

    for h in [0, 1, 3, 5, 10, 20, 50, 100]:
        if h < pred_len:
            err = np.sqrt(((preds[h] - future[h]) ** 2).mean())
            alpha = float((preds[h] * future[h]).sum() / ((future[h] ** 2).sum() + 1e-8))
            print(f"  [h={h:3d}] pred_norm={np.linalg.norm(preds[h]):.2f}  "
                  f"truth_norm={np.linalg.norm(future[h]):.2f}  rmse={err:.3f}  α={alpha:.3f}")

    return ctx, future, preds


# L63
def l63_fn(n, seed): return integrate_lorenz63(n, dt=0.025, seed=seed, spinup=2000)

# L96
def l96_fn(n, seed): return integrate_lorenz96(n, N=20, F=8.0, dt=0.05, seed=seed, spinup=2000)


if __name__ == "__main__":
    # Same taus for both: [1, 2, 3, 4] (minimal Takens)
    l63_ctx, l63_fut, l63_pred = run_system("L63", l63_fn, np.array([4, 3, 2, 1]))
    l96_ctx, l96_fut, l96_pred = run_system("L96-N=20 (default m=128)", l96_fn, np.array([4, 3, 2, 1]))
    # FIX attempt: m_inducing = all training points (~500)
    _, _, l96_pred500 = run_system("L96-N=20 (m=500)", l96_fn, np.array([4, 3, 2, 1]),
                                      m_inducing=500, n_epochs=300)
    print(f"\n=== SUMMARY ===")
    print(f"L96 default m=128 h=0 pred_norm = {np.linalg.norm(l96_pred[0]):.2f}")
    print(f"L96 fix    m=500 h=0 pred_norm = {np.linalg.norm(l96_pred500[0]):.2f}")
    print(f"L96 truth  h=0 norm           = {np.linalg.norm(l96_fut[0]):.2f}")

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    for r, (name, ctx, fut, pred) in enumerate([
        ("L63 (D=3, 15-D features)", l63_ctx, l63_fut, l63_pred),
        ("L96 (D=20, 100-D features)", l96_ctx, l96_fut, l96_pred),
    ]):
        D = fut.shape[1]
        rep_dims = [0, D // 2, D - 1] if D > 3 else [0, 1, 2]
        for c, d in enumerate(rep_dims):
            ax = axes[r, c]
            ax.plot(fut[:, d], "k-", linewidth=1.5, label="truth", alpha=0.7)
            ax.plot(pred[:, d], "r-", linewidth=1, label="SVGP rollout", alpha=0.8)
            ax.set_title(f"{name} dim {d}", fontsize=10)
            ax.grid(alpha=0.3)
            if c == 0:
                ax.legend(fontsize=9)

    plt.suptitle("SVGP-on-truth sanity: L63 vs L96 forecast", fontsize=12)
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "figures" / "svgp_l63_vs_l96_sanity.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n[saved] {out}")
