"""Figure D7 v2 — τ 低秩奇异值谱 (Lorenz96 L ∈ {3, 5, 7}).

原 L=7 版本奇异值区分度不足。这个版本在不同 L 和 noise levels 上做 CMA-ES
Stage B τ search（rank 设为 full = L-1），然后绘制 UU^T 的奇异值谱，证明
"真实最优 τ 结构是低秩的"。

Input : nothing (fresh Lorenz96 integration)
Output:
    results/tau_spectrum_L{L}_n{seeds}.json
    figures/tau_lowrank_spectrum_paperfig.png

Run:
    python -m experiments.week2_modules.tau_lowrank_spectrum_v2 --L_list 3 5 7 --n_seeds 5
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.week1.lorenz96_utils import (
    integrate_lorenz96, lorenz96_attractor_std,
)
from methods.mi_lyap import mi_lyap_cmaes_tau, robust_lyapunov

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "experiments" / "week2_modules" / "results"
FIG_DIR = REPO_ROOT / "experiments" / "week2_modules" / "figures"

N_L96 = 20              # Lorenz96 ring dimension (not too big, keep CMA-ES practical)
TAU_MAX = 30
N_CTX = 1500


def run_one(L: int, seed: int) -> np.ndarray:
    """Integrate Lorenz96 clean, CMA-ES rank=L-1 (full), return SVs of UU^T."""
    traj = integrate_lorenz96(N_L96, N_CTX, seed=seed, spinup=2000)
    # use the first dimension as the 1-d series fed to τ-search
    series = traj[:, 0]
    lam = robust_lyapunov(series, dt=0.025, emb_dim=min(L, 5), lag=2, trajectory_len=50, prefilter=True)
    spec = mi_lyap_cmaes_tau(
        series, L=L, tau_max=TAU_MAX, horizon=1, lam=lam * 0.025,
        rank=max(1, L - 1), popsize=20, n_iter=30, seed=seed,
    )
    return np.asarray(spec.__dict__["singular_values"])


def plot(results: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {3: "#e7298a", 5: "#1b9e77", 7: "#7570b3", 10: "#d95f02"}
    for L in sorted(results.keys()):
        sv_list = results[L]
        arr = np.array([np.pad(s, (0, L - 1 - len(s)), constant_values=np.nan)
                        if len(s) < L - 1 else s[: L - 1] for s in sv_list])
        # normalise each row to its max for comparison
        arr = arr / np.nanmax(arr, axis=1, keepdims=True)
        mean = np.nanmean(arr, axis=0)
        std  = np.nanstd(arr, axis=0)
        idx = np.arange(1, L)
        ax.errorbar(idx, mean, yerr=std, marker="o", linewidth=2, markersize=7,
                    color=colors.get(L, None), capsize=3, label=f"L = {L}")
    ax.axhline(0.1, color="gray", linestyle="--", linewidth=1.0, alpha=0.6,
               label="10% threshold")
    ax.set_xlabel("singular value index (descending)")
    ax.set_ylabel("normalised singular value  σ_i / σ_1")
    ax.set_title("Figure D7 — τ-Matrix Singular Value Spectrum  (Lorenz96, CMA-ES Stage B)",
                 fontsize=12)
    ax.set_yscale("log")
    ax.set_ylim(0.005, 1.5)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--L_list", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args()

    all_results = {}
    for L in args.L_list:
        sv_list = []
        for seed in range(args.n_seeds):
            print(f"[run] L={L} seed={seed}")
            svs = run_one(L, seed)
            print(f"       SVs={svs.tolist()}")
            sv_list.append(svs.tolist())
        all_results[L] = sv_list

    out_json = OUT_DIR / f"tau_spectrum_{args.tag}.json"
    out_json.write_text(json.dumps(dict(
        N_L96=N_L96, tau_max=TAU_MAX, n_ctx=N_CTX, n_seeds=args.n_seeds,
        singular_values_per_L=all_results,
    ), indent=2))
    print(f"[saved] {out_json}")

    plot(all_results, FIG_DIR / "tau_lowrank_spectrum_paperfig.png")


if __name__ == "__main__":
    main()
