"""Final L96 trajectory figure for §5.7: CSDI+DeepEDM vs AR-K+DeepEDM vs Panda vs truth.

Runs one seed per scenario (S0, S3, S5, S6) and plots representative dims to
show (a) at S0-S3 Panda dominates; (b) at S5-S6 Panda produces flat/wandering
predictions while CSDI+DeepEDM stays on the attractor.

Output: CSDI-PRO/pictures/l96_trajectory_final.png
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from experiments.week1.lorenz63_utils import (
    HarshnessScenario, make_sparse_noisy, linear_interp_fill,
)
from experiments.week1.lorenz96_utils import (
    LORENZ96_LYAP_F8, LORENZ96_F_DEFAULT, integrate_lorenz96, lorenz96_attractor_std,
)
from experiments.week1.full_pipeline_rollout import full_pipeline_forecast
from methods.csdi_impute_adapter import set_csdi_checkpoint, set_csdi_attractor_std

try:
    from baselines.panda_adapter import panda_forecast
    HAS_PANDA = True
except ImportError:
    HAS_PANDA = False


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "pictures" / "l96_trajectory_final.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

N, F, dt = 20, LORENZ96_F_DEFAULT, 0.05
LYAP = LORENZ96_LYAP_F8
ATTR_STD = lorenz96_attractor_std(N=N, F=F)
N_CTX, PRED_LEN = 512, 128
CKPT = REPO / "experiments/week2_modules/ckpts/dyn_csdi_l96_full_c192_vales_best.pt"

SCENARIOS = [
    HarshnessScenario("S0", 0.00, 0.00),
    HarshnessScenario("S3", 0.60, 0.50),
    HarshnessScenario("S5", 0.90, 1.20),
    HarshnessScenario("S6", 0.95, 1.50),
]
SEED = 0
DIMS = [0, 10, 19]  # representative dims across ring


def run_all():
    set_csdi_checkpoint(str(CKPT))
    set_csdi_attractor_std(ATTR_STD)

    traj = integrate_lorenz96(N_CTX + PRED_LEN, N=N, F=F, dt=dt, spinup=2000, seed=SEED)
    ctx_true, fut_true = traj[:N_CTX], traj[N_CTX:]
    results = {}
    for sc in SCENARIOS:
        obs, mask = make_sparse_noisy(
            ctx_true, sparsity=sc.sparsity, noise_std_frac=sc.noise_std_frac,
            attractor_std=ATTR_STD, seed=1000 * SEED + hash(sc.name) % 10000,
        )
        ctx_filled = linear_interp_fill(obs)

        print(f"[{sc.name}] s={sc.sparsity} σ={sc.noise_std_frac} keep={mask.mean():.2f}")
        pred_csdi = full_pipeline_forecast(
            obs, pred_len=PRED_LEN, seed=SEED,
            imp_kind="csdi", bayes_calls=10, backbone="deepedm",
        )
        pred_ark = full_pipeline_forecast(
            obs, pred_len=PRED_LEN, seed=SEED,
            bayes_calls=10, backbone="deepedm",
        )
        pred_panda = panda_forecast(ctx_filled, pred_len=PRED_LEN) if HAS_PANDA else None
        results[sc.name] = dict(
            ctx=ctx_true, fut=fut_true,
            csdi=pred_csdi, ark=pred_ark, panda=pred_panda,
            sc=sc, keep=mask.mean(),
        )
    return results


def plot(results):
    nrows = len(SCENARIOS)
    ncols = len(DIMS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.3 * nrows),
                             sharex=True)
    t = np.arange(PRED_LEN) * dt * LYAP  # x-axis in Lyapunov times

    for r, sc in enumerate(SCENARIOS):
        d = results[sc.name]
        for c, dim in enumerate(DIMS):
            ax = axes[r, c]
            ax.plot(t, d["fut"][:, dim], "k-", linewidth=1.8, alpha=0.85, label="truth")
            if d["panda"] is not None:
                ax.plot(t, d["panda"][:, dim], color="C1", linewidth=1.2,
                        alpha=0.9, label="Panda-72M")
            ax.plot(t, d["ark"][:, dim], color="C0", linewidth=1.2, linestyle="--",
                    alpha=0.85, label="Ours (AR-K + DeepEDM)")
            ax.plot(t, d["csdi"][:, dim], color="C3", linewidth=1.6,
                    alpha=0.95, label="Ours (CSDI + DeepEDM)")
            if c == 0:
                ax.set_ylabel(f"{sc.name}\n(s={sc.sparsity}, σ={sc.noise_std_frac})\nkeep={d['keep']:.2f}",
                              fontsize=9)
            if r == 0:
                ax.set_title(f"dim {dim}", fontsize=10)
            if r == nrows - 1:
                ax.set_xlabel("prediction horizon (Lyapunov times Λ)", fontsize=9)
            ax.grid(alpha=0.3)
            if r == 0 and c == ncols - 1:
                ax.legend(loc="upper right", fontsize=7, framealpha=0.9)

    fig.suptitle(
        f"Lorenz96 N={N} forecast trajectories — seed {SEED}, n_ctx={N_CTX}, pred_len={PRED_LEN}\n"
        "S0-S3: Panda wins cleanly. S5-S6: Panda and Parrot collapse, CSDI + DeepEDM "
        "stays on-attractor.",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    results = run_all()
    plot(results)
