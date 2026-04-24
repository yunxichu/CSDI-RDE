"""Chua double-scroll trajectory figure. 3D state (x, y, z)."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from experiments.week1.lorenz63_utils import (
    HarshnessScenario, make_sparse_noisy, linear_interp_fill,
)
from experiments.week1.full_pipeline_rollout import full_pipeline_forecast
from methods.csdi_impute_adapter import set_csdi_checkpoint, set_csdi_attractor_std
from systems.chua import integrate_chua, CHUA_ATTRACTOR_STD, CHUA_LYAP, CHUA_DT

try:
    from baselines.panda_adapter import panda_forecast
    HAS_PANDA = True
except ImportError:
    HAS_PANDA = False

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "pictures" / "chua_trajectory_final.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

DT, LYAP, ATTR_STD = CHUA_DT, CHUA_LYAP, CHUA_ATTRACTOR_STD
N_CTX, PRED_LEN = 512, 1024
CKPT = REPO / "experiments/week2_modules/ckpts/dyn_csdi_chua_full_vales_best.pt"

SCENARIOS = [
    HarshnessScenario("S0", 0.00, 0.00),
    HarshnessScenario("S3", 0.60, 0.50),
    HarshnessScenario("S5", 0.90, 1.20),
    HarshnessScenario("S6", 0.95, 1.50),
]
SEED = 0
DIMS = [0, 1, 2]
DIM_NAMES = ["x", "y", "z"]


def run_all():
    set_csdi_checkpoint(str(CKPT))
    set_csdi_attractor_std(ATTR_STD)
    traj = integrate_chua(N_CTX + PRED_LEN, dt=DT, spinup=3000, seed=SEED)
    ctx_true, fut_true = traj[:N_CTX], traj[N_CTX:]
    results = {}
    for sc in SCENARIOS:
        obs, mask = make_sparse_noisy(
            ctx_true, sparsity=sc.sparsity, noise_std_frac=sc.noise_std_frac,
            attractor_std=ATTR_STD, seed=1000 * SEED + hash(sc.name) % 10000,
        )
        ctx_filled = linear_interp_fill(obs)
        print(f"[{sc.name}] s={sc.sparsity} σ={sc.noise_std_frac} keep={mask.mean():.2f}",
              flush=True)
        pred_svgp = full_pipeline_forecast(obs, pred_len=PRED_LEN, seed=SEED,
            imp_kind="csdi", bayes_calls=10, n_epochs=150, backbone="svgp")
        pred_deepedm = full_pipeline_forecast(obs, pred_len=PRED_LEN, seed=SEED,
            imp_kind="csdi", bayes_calls=10, backbone="deepedm")
        pred_panda = panda_forecast(ctx_filled, pred_len=PRED_LEN) if HAS_PANDA else None
        results[sc.name] = dict(fut=fut_true, svgp=pred_svgp, deepedm=pred_deepedm,
                                  panda=pred_panda, sc=sc, keep=mask.mean())
    return results


def plot(results):
    nrows, ncols = len(SCENARIOS), len(DIMS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.3 * nrows), sharex=True)
    t = np.arange(PRED_LEN) * DT * LYAP
    for r, sc in enumerate(SCENARIOS):
        d = results[sc.name]
        for c, dim in enumerate(DIMS):
            ax = axes[r, c]
            ax.plot(t, d["fut"][:, dim], "k-", linewidth=1.6, alpha=0.9, label="truth")
            if d["panda"] is not None:
                ax.plot(t, d["panda"][:, dim], color="C1", linewidth=1.1, alpha=0.85,
                        label="Panda-72M")
            ax.plot(t, d["svgp"][:, dim], color="C5", linewidth=1.1, linestyle=":",
                    alpha=0.85, label="Ours (CSDI + SVGP)")
            ax.plot(t, d["deepedm"][:, dim], color="C3", linewidth=1.5, alpha=0.95,
                    label="Ours (CSDI + DeepEDM)")
            if c == 0:
                ax.set_ylabel(f"{sc.name}\n(s={sc.sparsity}, σ={sc.noise_std_frac})\n"
                              f"keep={d['keep']:.2f}", fontsize=9)
            if r == 0:
                ax.set_title(f"dim {DIM_NAMES[dim]}", fontsize=10)
            if r == nrows - 1:
                ax.set_xlabel("horizon (Lyapunov times Λ)", fontsize=9)
            ax.grid(alpha=0.3)
            if r == 0 and c == ncols - 1:
                ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.suptitle(f"Chua double-scroll forecast — seed {SEED}, n_ctx={N_CTX}, "
                 f"pred_len={PRED_LEN}  (λ={LYAP:.2f})", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    results = run_all()
    plot(results)
