"""Figure D6 — MI-Lyap τ 稳定性 vs observation noise 扫描.

Setup:
  - Lorenz63 × 1000 clean steps + random sparsity 0.3 + variable noise σ_frac
  - σ_frac ∈ {0.0, 0.1, 0.3, 0.5, 1.0, 1.5}
  - n_seeds (default 15) per σ → compute stdev of chosen τ across seeds
  - Methods compared: MI-Lyap (ours, BO), Fraser-Swinney, random baseline

Claim to be supported:
  MI-Lyap picks a stable τ even as σ grows, while Fraser-Swinney 的 τ 方差
  随 noise 显著增大，random 作为 upper bound.

Run:
  python -m experiments.week2_modules.tau_stability_vs_noise --n_seeds 15
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.week1.lorenz63_utils import (
    LORENZ63_ATTRACTOR_STD, LORENZ63_LYAP,
    integrate_lorenz63, linear_interp_fill, make_sparse_noisy,
)
from methods.mi_lyap import (
    fraser_swinney_tau, mi_lyap_bayes_tau, random_tau, robust_lyapunov,
)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "experiments" / "week2_modules" / "results"
FIG_DIR = REPO_ROOT / "experiments" / "week2_modules" / "figures"

L_EMBED = 5
TAU_MAX = 30
N_CTX = 1500
SIGMAS = [0.0, 0.1, 0.3, 0.5, 1.0, 1.5]
SPARSITY = 0.3  # fixed moderate sparsity, isolate noise effect


def tau_mean_std(tau_samples: list[np.ndarray]) -> tuple[float, float]:
    """Across seeds, compute the mean and std of the *L2 norm* of chosen τ vector.

    Using τ-norm as a single-dim summary of τ stability.
    """
    norms = np.array([float(np.linalg.norm(t)) for t in tau_samples])
    return float(norms.mean()), float(norms.std())


def run_scan(n_seeds: int) -> dict:
    results = {m: {s: [] for s in SIGMAS} for m in ["mi_lyap", "fraser", "random"]}
    print(f"=== τ stability scan: σ_frac ∈ {SIGMAS}, seeds={n_seeds}, sparsity={SPARSITY}")
    for seed in range(n_seeds):
        traj = integrate_lorenz63(N_CTX, dt=0.025, seed=seed, spinup=2000)
        for sigma in SIGMAS:
            obs, _ = make_sparse_noisy(traj, sparsity=SPARSITY, noise_std_frac=sigma,
                                       attractor_std=LORENZ63_ATTRACTOR_STD,
                                       seed=10_000 * seed + int(sigma * 100))
            ctx = linear_interp_fill(obs)

            # MI-Lyap BO (with noise-robust lambda)
            lam = robust_lyapunov(ctx[:, 0], dt=0.025, emb_dim=5, lag=2, trajectory_len=50, prefilter=True)
            try:
                spec = mi_lyap_bayes_tau(ctx[:, 0], L=L_EMBED, tau_max=TAU_MAX, horizon=1,
                                         lam=lam * 0.025, n_calls=12, k=4, seed=seed)
                tau_mi = spec.taus
            except Exception as e:
                print(f"  [warn] MI-Lyap failed seed={seed} σ={sigma}: {e}")
                tau_mi = random_tau(L_EMBED, TAU_MAX, seed=seed).taus
            results["mi_lyap"][sigma].append(tau_mi)

            # Fraser-Swinney
            try:
                spec_f = fraser_swinney_tau(ctx[:, 0], L=L_EMBED, tau_max=TAU_MAX)
                tau_f = spec_f.taus
            except Exception as e:
                print(f"  [warn] Fraser failed: {e}")
                tau_f = random_tau(L_EMBED, TAU_MAX, seed=seed).taus
            results["fraser"][sigma].append(tau_f)

            # Random
            spec_r = random_tau(L=L_EMBED, tau_max=TAU_MAX, seed=seed)
            results["random"][sigma].append(spec_r.taus)

        print(f"  seed={seed} done (6 σ × 3 methods)")
    return results


def plot(results: dict, out_path: Path) -> None:
    methods_display = {
        "mi_lyap": ("MI-Lyap (ours)", "#e7298a"),
        "fraser":  ("Fraser-Swinney", "#d95f02"),
        "random":  ("Random",         "#888888"),
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    for m in ["mi_lyap", "fraser", "random"]:
        means, stds = [], []
        for sigma in SIGMAS:
            taus = results[m][sigma]
            if not taus:
                means.append(np.nan); stds.append(0); continue
            mean, std = tau_mean_std(taus)
            means.append(mean); stds.append(std)
        label, color = methods_display[m]
        ax1.errorbar(SIGMAS, means, yerr=stds, marker="o", color=color, linewidth=2,
                     label=label, markersize=6, capsize=3)

        ax2.plot(SIGMAS, stds, marker="o", color=color, linewidth=2,
                 label=label, markersize=6)

    ax1.set_xlabel("observation noise σ / σ_attractor")
    ax1.set_ylabel("|τ|  (norm of chosen delay vector)")
    ax1.set_title("Chosen τ magnitude vs noise")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=9)

    ax2.set_xlabel("observation noise σ / σ_attractor")
    ax2.set_ylabel("std(|τ|) across seeds  (↓ = more stable)")
    ax2.set_title("τ-search stability")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=9)

    plt.suptitle("Figure D6 — MI-Lyap τ Stability vs Observation Noise (Lorenz63)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=15)
    ap.add_argument("--tag", default="n15_v1")
    args = ap.parse_args()

    results = run_scan(args.n_seeds)

    # Save numeric results
    serializable = {
        m: {str(s): [t.tolist() for t in taus] for s, taus in by_s.items()}
        for m, by_s in results.items()
    }
    out_json = OUT_DIR / f"tau_stability_{args.tag}.json"
    out_json.write_text(json.dumps(dict(
        sigmas=SIGMAS, L_embed=L_EMBED, tau_max=TAU_MAX,
        sparsity=SPARSITY, n_ctx=N_CTX, n_seeds=args.n_seeds,
        per_method_per_sigma_taus=serializable,
    ), indent=2))
    print(f"[saved] {out_json}")

    plot(results, FIG_DIR / f"tau_stability_paperfig.png")


if __name__ == "__main__":
    main()
