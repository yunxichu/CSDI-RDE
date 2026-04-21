"""Lorenz96 scaling & τ-search experiment.

Three things:

  1. **SVGP scaling**: train SVGP on Lorenz96 N ∈ {10, 20, 40} delay coords and
     measure training time + test NRMSE. Compare against exact GPR up to the
     size where it blows up. This is the empirical backing for tech.md
     Proposition 2 (rate ~ n^{-ν/(2ν + d_KY)}, where d_KY ≈ 0.4·N).

  2. **τ-search scaling**: on Lorenz96 N=40 (L=10 delay coords), compare
     BayesOpt (Stage A) vs low-rank CMA-ES (Stage B). BO is expected to
     struggle as L grows; CMA-ES should handle it via low-rank parameterisation.

  3. **τ low-rank structure**: show the singular-value spectrum of the
     CMA-ES-optimised τ matrix. This is Figure 7 in tech.md
     ("neighbouring dimensions share chaotic timescales").

Run:
    CUDA_VISIBLE_DEVICES=2 python -m experiments.week2_modules.lorenz96_scaling \
        --N 10 20 40 --n_seeds 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.week1.lorenz96_utils import (
    LORENZ96_LYAP_F8,
    integrate_lorenz96,
    lorenz96_attractor_std,
)
from methods.mi_lyap import (
    construct_delay_dataset,
    fraser_swinney_tau,
    mi_lyap_bayes_tau,
    mi_lyap_cmaes_tau,
    random_tau,
)
from metrics.chaos_metrics import nrmse as chaos_nrmse
from models.svgp import MultiOutputSVGP, SVGPConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
RES = REPO_ROOT / "experiments" / "week2_modules" / "results"
FIG = REPO_ROOT / "experiments" / "week2_modules" / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def build_dataset_from_traj(traj: np.ndarray, taus: np.ndarray, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Apply delay coords on channel 0 (common convention for Lorenz96 scan).

    Returns X (n, L), Y (n, N).
    """
    N = traj.shape[1]
    # Use channel 0 for delay; condition Y on all channels
    Y_delay, _ = construct_delay_dataset(traj[:, 0], taus, horizon=horizon)
    # Build X = delay coords, Y = full state at t + horizon
    t0 = int(taus.max())
    t_idx = np.arange(t0, traj.shape[0] - horizon)
    cols = [traj[t_idx, 0]]
    for tau in taus:
        cols.append(traj[t_idx - tau, 0])
    X = np.stack(cols, axis=1).astype(np.float32)
    Y = traj[t_idx + horizon, :].astype(np.float32)
    return X, Y


def exp_svgp_scaling(N_values: list[int], n_seeds: int = 2) -> dict:
    results: dict = {}
    for N in N_values:
        per_seed = []
        for seed in range(n_seeds):
            traj = integrate_lorenz96(2000, N=N, seed=seed)
            atd = float(traj.std())
            # delay with random τ (keep the scaling experiment τ-agnostic)
            L_embed = 5
            taus = random_tau(L=L_embed, tau_max=10, seed=seed).taus
            X, Y = build_dataset_from_traj(traj, taus, horizon=1)
            n = X.shape[0]
            n_train = int(0.7 * n); n_test = n - n_train
            X_tr, Y_tr = X[:n_train], Y[:n_train]
            X_te, Y_te = X[n_train:], Y[n_train:]

            t0 = time.time()
            gp = MultiOutputSVGP(SVGPConfig(m_inducing=128, n_epochs=100, lr=1e-2)).fit(X_tr, Y_tr)
            train_t = time.time() - t0
            mu_te, std_te = gp.predict(X_te, return_std=True)
            rmse = chaos_nrmse(Y_te, mu_te, attractor_std=atd)

            per_seed.append(dict(
                seed=seed, N=N, L=L_embed, n_train=n_train, n_test=n_test,
                train_time_s=train_t, nrmse=float(rmse), attractor_std=atd,
            ))
            print(f"  [svgp N={N:3d} seed={seed}] n_train={n_train} train_t={train_t:.1f}s nrmse={rmse:.3f}")
        results[N] = per_seed
    return results


def exp_tau_search_scaling(N: int, n_seeds: int = 2, L_embed: int = 7, tau_max: int = 12) -> dict:
    """Compare BayesOpt vs low-rank CMA-ES on Lorenz96 N with L_embed delay coords."""
    results: dict = {"L_embed": L_embed, "tau_max": tau_max, "per_seed": []}

    for seed in range(n_seeds):
        traj = integrate_lorenz96(3000, N=N, seed=seed)
        x0 = traj[:, 0]
        print(f"\n  [τ-search N={N} seed={seed}]")

        # random baseline
        t0 = time.time(); spec_r = random_tau(L=L_embed, tau_max=tau_max, seed=seed); t_r = time.time() - t0
        # Fraser-Swinney
        t0 = time.time(); spec_fs = fraser_swinney_tau(x0, tau_max=tau_max, L=L_embed, k=4); t_fs = time.time() - t0
        # BayesOpt (Stage A) — Stage A shines when L small enough, struggles at L=7+
        t0 = time.time(); spec_bo = mi_lyap_bayes_tau(x0, L=L_embed, tau_max=tau_max,
                                                      horizon=1, n_calls=20, seed=seed); t_bo = time.time() - t0
        # CMA-ES rank-2 (Stage B) — low-rank parameterisation keeps dim low
        t0 = time.time(); spec_cma = mi_lyap_cmaes_tau(x0, L=L_embed, tau_max=tau_max,
                                                        horizon=1, rank=2, popsize=12, n_iter=15,
                                                        seed=seed); t_cma = time.time() - t0

        # evaluate each by downstream SVGP test NRMSE
        specs = {"random": (spec_r, t_r), "fraser_swinney": (spec_fs, t_fs),
                 "mi_lyap_bo": (spec_bo, t_bo), "mi_lyap_cmaes": (spec_cma, t_cma)}
        seed_rec = {"seed": seed, "N": N, "specs": {}}
        for name, (spec, t_search) in specs.items():
            X, Y = build_dataset_from_traj(traj, spec.taus, horizon=1)
            n = X.shape[0]; n_train = int(0.7 * n)
            gp = MultiOutputSVGP(SVGPConfig(m_inducing=128, n_epochs=80, lr=1e-2)).fit(X[:n_train], Y[:n_train])
            mu_te, _ = gp.predict(X[n_train:], return_std=True)
            rmse = chaos_nrmse(Y[n_train:], mu_te, attractor_std=float(traj.std()))
            sv = spec.__dict__.get("singular_values", None)
            seed_rec["specs"][name] = dict(
                taus=spec.taus.tolist(), search_sec=t_search, nrmse=float(rmse),
                singular_values=sv,
            )
            print(f"    {name:<16}  t={t_search:6.1f}s  τ={spec.taus.tolist()}  nrmse={rmse:.3f}")
        results["per_seed"].append(seed_rec)
    return results


def plot_singular_values(cmaes_sv: list[list[float]], fig_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    for s in cmaes_sv:
        if s is None: continue
        s = np.asarray(s)
        ax.plot(range(1, len(s) + 1), s / s[0], marker="o", alpha=0.7)
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("σ_i / σ_1  (normalised)")
    ax.set_title("τ-matrix singular-value spectrum (Lorenz96 N=40, rank-2)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_path, dpi=150)
    print(f"[plot] saved {fig_path}")


def plot_scaling(scaling_res: dict, fig_path: Path) -> None:
    import matplotlib.pyplot as plt

    Ns = sorted(scaling_res.keys())
    train_t = [np.mean([r["train_time_s"] for r in scaling_res[N]]) for N in Ns]
    train_t_std = [np.std([r["train_time_s"] for r in scaling_res[N]]) for N in Ns]
    nrmse_m = [np.mean([r["nrmse"] for r in scaling_res[N]]) for N in Ns]
    nrmse_s = [np.std([r["nrmse"] for r in scaling_res[N]]) for N in Ns]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.errorbar(Ns, train_t, yerr=train_t_std, marker="o", linewidth=2, capsize=3, color="C0")
    ax1.set_xscale("log"); ax1.set_xlabel("Ambient dim N"); ax1.set_ylabel("SVGP train time (s)")
    ax1.set_title("SVGP scaling vs N (Lorenz96)")
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(Ns, nrmse_m, yerr=nrmse_s, marker="o", linewidth=2, capsize=3, color="C1")
    ax2.set_xscale("log"); ax2.set_xlabel("Ambient dim N"); ax2.set_ylabel("Test NRMSE")
    ax2.set_title("Prediction quality vs N")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Lorenz96 Proposition 2 empirical test", y=1.02)
    fig.tight_layout(); fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {fig_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[10, 20, 40])
    ap.add_argument("--n_seeds", type=int, default=2)
    ap.add_argument("--tau_N", type=int, default=40)
    args = ap.parse_args()

    print("=== Part 1: SVGP scaling ===")
    scaling = exp_svgp_scaling(args.N, n_seeds=args.n_seeds)

    print(f"\n=== Part 2: τ-search scaling on Lorenz96 N={args.tau_N} ===")
    tau_search = exp_tau_search_scaling(N=args.tau_N, n_seeds=args.n_seeds)

    out = dict(svgp_scaling=scaling, tau_search=tau_search)
    out_path = RES / f"lorenz96_scaling_N{'_'.join(str(n) for n in args.N)}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {out_path}")

    plot_scaling(scaling, FIG / "lorenz96_svgp_scaling.png")

    cmaes_sv = [r["specs"]["mi_lyap_cmaes"]["singular_values"] for r in tau_search["per_seed"]]
    if cmaes_sv:
        plot_singular_values(cmaes_sv, FIG / "tau_low_rank_spectrum.png")


if __name__ == "__main__":
    main()
