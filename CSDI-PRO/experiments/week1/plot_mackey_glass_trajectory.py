"""Mackey-Glass trajectory figure for §5.8.

MG is 1-D (scalar state) so we lay out: 4 rows (S0, S3, S5, S6) × 3 columns
(3 different prediction horizons / zoom levels). Shows CSDI+SVGP / CSDI+DeepEDM
/ CSDI+FNO / Panda / truth, all on seed 0.

Output: CSDI-PRO/pictures/mackey_glass_trajectory_final.png
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from experiments.week1.lorenz63_utils import (
    HarshnessScenario, make_sparse_noisy, linear_interp_fill,
)
from experiments.week1.full_pipeline_rollout import full_pipeline_forecast
from methods.csdi_impute_adapter import set_csdi_checkpoint, set_csdi_attractor_std
from systems.mackey_glass import (
    integrate_mackey_glass, MACKEY_GLASS_ATTRACTOR_STD, MACKEY_GLASS_LYAP,
    MACKEY_GLASS_DT, MACKEY_GLASS_TAU,
)

try:
    from baselines.panda_adapter import panda_forecast
    HAS_PANDA = True
except ImportError:
    HAS_PANDA = False

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "pictures" / "mackey_glass_trajectory_final.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

DT, LYAP, ATTR_STD = MACKEY_GLASS_DT, MACKEY_GLASS_LYAP, MACKEY_GLASS_ATTRACTOR_STD
N_CTX, PRED_LEN = 512, 1024
CKPT = REPO / "experiments/week2_modules/ckpts/dyn_csdi_mg_full_vales_best.pt"

SCENARIOS = [
    HarshnessScenario("S0", 0.00, 0.00),
    HarshnessScenario("S3", 0.60, 0.50),
    HarshnessScenario("S5", 0.90, 1.20),
    HarshnessScenario("S6", 0.95, 1.50),
]
SEED = 0


def run_all():
    set_csdi_checkpoint(str(CKPT))
    set_csdi_attractor_std(ATTR_STD)

    traj = integrate_mackey_glass(N_CTX + PRED_LEN, dt=DT, spinup=2000, seed=SEED)
    ctx_true, fut_true = traj[:N_CTX], traj[N_CTX:]
    results = {}
    for sc in SCENARIOS:
        obs, mask = make_sparse_noisy(
            ctx_true, sparsity=sc.sparsity, noise_std_frac=sc.noise_std_frac,
            attractor_std=ATTR_STD, seed=1000 * SEED + hash(sc.name) % 10000,
        )
        ctx_filled = linear_interp_fill(obs)
        print(f"[{sc.name}] s={sc.sparsity} σ={sc.noise_std_frac} keep={mask.mean():.2f}", flush=True)

        pred_svgp = full_pipeline_forecast(
            obs, pred_len=PRED_LEN, seed=SEED,
            imp_kind="csdi", bayes_calls=10, n_epochs=150, backbone="svgp",
        )
        pred_deepedm = full_pipeline_forecast(
            obs, pred_len=PRED_LEN, seed=SEED,
            imp_kind="csdi", bayes_calls=10, backbone="deepedm",
        )
        pred_panda = panda_forecast(ctx_filled, pred_len=PRED_LEN) if HAS_PANDA else None
        results[sc.name] = dict(
            fut=fut_true, svgp=pred_svgp, deepedm=pred_deepedm,
            panda=pred_panda, sc=sc, keep=mask.mean(),
        )
    return results


def plot(results):
    nrows = len(SCENARIOS)
    zoom_horizons = [256, 512, PRED_LEN]
    ncols = len(zoom_horizons)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.3 * nrows))
    t = np.arange(PRED_LEN) * DT * LYAP

    for r, sc in enumerate(SCENARIOS):
        d = results[sc.name]
        for c, H in enumerate(zoom_horizons):
            ax = axes[r, c]
            sl = slice(0, H)
            ax.plot(t[sl], d["fut"][sl, 0], "k-", linewidth=1.6, alpha=0.9, label="truth")
            if d["panda"] is not None:
                ax.plot(t[sl], d["panda"][sl, 0], color="C1", linewidth=1.1,
                        alpha=0.85, label="Panda-72M")
            ax.plot(t[sl], d["svgp"][sl, 0], color="C5", linewidth=1.1,
                    linestyle=":", alpha=0.85, label="Ours (CSDI + SVGP)")
            ax.plot(t[sl], d["deepedm"][sl, 0], color="C3", linewidth=1.5,
                    alpha=0.95, label="Ours (CSDI + DeepEDM)")
            if c == 0:
                ax.set_ylabel(f"{sc.name}\n(s={sc.sparsity}, σ={sc.noise_std_frac})\n"
                              f"keep={d['keep']:.2f}", fontsize=9)
            if r == 0:
                ax.set_title(f"zoom to h ≤ {H*DT*LYAP:.2f} Λ", fontsize=10)
            if r == nrows - 1:
                ax.set_xlabel("horizon (Lyapunov times Λ)", fontsize=9)
            ax.grid(alpha=0.3)
            if r == 0 and c == ncols - 1:
                ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    fig.suptitle(
        f"Mackey-Glass τ={MACKEY_GLASS_TAU} forecast — seed {SEED}, n_ctx={N_CTX}, "
        f"pred_len={PRED_LEN}  (λ={LYAP:.3f}, σ_attr={ATTR_STD:.3f})\n"
        "Canonical delay-embedding testbed: CSDI + DeepEDM should track truth over many Λ.",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"[saved] {OUT}", flush=True)


if __name__ == "__main__":
    results = run_all()
    plot(results)
