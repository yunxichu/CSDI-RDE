"""τ-coupling ablation (paper §5.X1) — does M1's CSDI delay mask benefit when
its τ matches what M2 selects on the current trajectory?

Design (S3 × n_seeds × 5 modes):
  - default    : no override (M1 uses learned delay_bias from training)
  - A_random   : random τ (no coupling)
  - B_current  : M2-selected τ on current trajectory (the correctly coupled config)
  - C_mismatch : M2-selected τ on a *clean S0* trajectory, applied to S3 data
  - D_equidist : fixed [1, 2, 4, 8, 16] equidistant τ

REFACTOR_PLAN_zh.md §6.1 expected: B > default ≈ (well-trained default) > C > D > A,
differences amplify at high harshness.

Downstream M2/M3/M4 always use the mi_lyap-selected τ on the current trajectory's
default-M1 imputation (τ_B). This isolates the effect of M1's delay-mask τ.

Caveat on "default" baseline: when tau_override=None, M1 uses the learned
delay_bias+delay_alpha from training. When a τ is provided, set_tau() re-initializes
delay_bias AND resets delay_alpha to 0.1 (see dynamics_csdi.py:204). So the default
mode and the four override modes differ in delay_alpha (learned vs 0.1 fixed). The
A/B/C/D comparison is apples-to-apples (all share delay_alpha=0.1 + τ-init bias);
"default" is an orthogonal reference (how far does learned bias get without
explicit τ anchor).

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m experiments.week2_modules.run_tau_coupling_ablation \\
        --ckpt experiments/week2_modules/ckpts/dyn_csdi_full_v6_center_ep20.pt \\
        --n_seeds 3 --scenario S3 --tag tau_coupling_v1
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.week1.lorenz63_utils import (
    PILOT_SCENARIOS, integrate_lorenz63, make_sparse_noisy, LORENZ63_ATTRACTOR_STD,
)
from experiments.week2_modules.run_ablation import (
    HORIZONS, L_EMBED, TAU_MAX, BAYES_CALLS, N_CTX, PRED_LEN, OUT_DIR,
    build_pipeline, evaluate_horizons,
)
from methods.csdi_impute_adapter import set_csdi_checkpoint
from methods.dynamics_impute import impute, set_tau_override
from methods.mi_lyap import mi_lyap_bayes_tau, random_tau, robust_lyapunov


def get_mode_tau(mode: str, tau_B, tau_C, seed: int):
    """Return the τ vector (length L-1, int) to feed to M1's delay mask for this mode.

    Note: CSDI's set_tau() expects L-1 offsets (since the first coord at Δt=0 is
    implicit). tau_B / tau_C from mi_lyap_bayes_tau are length L, with τ[0]=0.
    We slice [1:] to get the L-1 non-zero offsets.
    """
    L = L_EMBED  # 5
    if mode == "default":
        return None
    if mode == "A_random":
        rng = np.random.default_rng(seed + 777)
        return np.sort(rng.integers(1, TAU_MAX + 1, size=L - 1))[::-1]  # descending
    if mode == "B_current":
        return np.asarray(tau_B[1:], dtype=np.int64)
    if mode == "C_mismatch":
        return np.asarray(tau_C[1:], dtype=np.int64)
    if mode == "D_equidist":
        return np.array([16, 8, 4, 2], dtype=np.int64)  # 4 offsets for L=5
    raise ValueError(f"unknown mode {mode!r}")


def run_one(seed: int, scenario, ckpt: str, modes: list[str], dt: float = 0.025):
    """Run all modes on a single seed. Returns list of records."""
    # Integrate trajectory + build S3 observations
    traj = integrate_lorenz63(N_CTX + PRED_LEN, dt=dt, seed=seed, spinup=2000)
    ctx_true = traj[:N_CTX]
    future_true = traj[N_CTX:]
    obs, mask = make_sparse_noisy(
        ctx_true, sparsity=scenario.sparsity, noise_std_frac=scenario.noise_std_frac,
        attractor_std=LORENZ63_ATTRACTOR_STD, seed=seed,
    )

    # Pass 1: default M1 to compute τ_B (current-trajectory M2 selection) + τ_C (S0 reference)
    set_tau_override(None)
    ctx_default = impute(obs, kind="csdi")
    lam_hat = robust_lyapunov(ctx_default[:, 0], dt=dt, emb_dim=5, lag=2,
                              trajectory_len=50, prefilter=True)
    spec_B = mi_lyap_bayes_tau(ctx_default[:, 0], L=L_EMBED, tau_max=TAU_MAX, horizon=1,
                               lam=lam_hat * dt, n_calls=BAYES_CALLS, k=4, seed=seed)
    tau_B = spec_B.taus  # length L

    # τ_C: M2 selection on a clean S0 trajectory (different seed to decorrelate)
    traj_s0 = integrate_lorenz63(N_CTX, dt=dt, seed=seed + 10000, spinup=2000)
    # no sparsification (S0)
    lam_s0 = robust_lyapunov(traj_s0[:, 0], dt=dt, emb_dim=5, lag=2,
                             trajectory_len=50, prefilter=True)
    spec_C = mi_lyap_bayes_tau(traj_s0[:, 0], L=L_EMBED, tau_max=TAU_MAX, horizon=1,
                               lam=lam_s0 * dt, n_calls=BAYES_CALLS, k=4,
                               seed=seed + 10000)
    tau_C = spec_C.taus

    # Downstream uses τ_B (M2 on current trajectory) for apples-to-apples comparison.
    # Construct a synthetic cfg mimicking full-csdi-empirical (M2 mi_lyap / M3 svgp / M4 lyap-emp)
    base_cfg = dict(imp="csdi", tau="mi_lyap", gp="svgp", cp="lyap", growth="empirical")

    records = []
    for mode in modes:
        tau_delay = get_mode_tau(mode, tau_B, tau_C, seed)
        set_tau_override(tau_delay)
        t0 = time.time()
        ctx_filled = impute(obs, kind="csdi")
        # Use the same τ_B downstream (not recomputed per-mode) so M1 is the only difference
        per_hgp, taus_ds, tau_sec = build_pipeline(ctx_filled, base_cfg, lam_hat=lam_hat * dt, seed=seed)
        metrics = evaluate_horizons(per_hgp, ctx_filled, future_true, taus_ds,
                                    lam_hat=lam_hat * dt, dt=dt, cp_kind="lyap",
                                    growth="empirical")
        elapsed = time.time() - t0
        rec = dict(
            mode=mode, seed=seed, scenario=scenario.name,
            sparsity=scenario.sparsity, noise=scenario.noise_std_frac,
            lam_hat=float(lam_hat),
            tau_delay=tau_delay.tolist() if tau_delay is not None else None,
            tau_B=tau_B.tolist(), tau_C=tau_C.tolist(),
            taus_downstream=taus_ds.tolist(),
            metrics=metrics, elapsed_sec=elapsed,
        )
        records.append(rec)
        h1 = metrics.get(1, {})
        h16 = metrics.get(16, {})
        print(f"  [{mode:12s}] seed={seed} τ_delay={tau_delay}  "
              f"h=1 nrmse={h1.get('nrmse', 0):.3f} picp={h1.get('picp', 0):.2f}  "
              f"h=16 nrmse={h16.get('nrmse', 0):.3f}  elapsed={elapsed:.1f}s")

    # Clear override after run
    set_tau_override(None)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to trained DynamicsCSDI ckpt")
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--scenario", default="S3")
    ap.add_argument("--modes", nargs="+",
                    default=["default", "A_random", "B_current", "C_mismatch", "D_equidist"])
    ap.add_argument("--tag", default="tau_coupling_v1")
    args = ap.parse_args()

    print(f"=== τ-coupling ablation (paper §5.X1) ===")
    print(f"  ckpt={args.ckpt}")
    print(f"  scenario={args.scenario}  n_seeds={args.n_seeds}  modes={args.modes}")
    set_csdi_checkpoint(args.ckpt)

    scenario = next(s for s in PILOT_SCENARIOS if s.name == args.scenario)
    print(f"  scenario {args.scenario}: sparsity={scenario.sparsity} σ={scenario.noise_std_frac}")

    all_records = []
    for seed in range(args.n_seeds):
        print(f"\n--- seed {seed} ---")
        all_records.extend(run_one(seed, scenario, args.ckpt, args.modes))

    out_json = OUT_DIR / f"{args.tag}.json"
    out_json.write_text(json.dumps({
        "scenario": args.scenario,
        "n_seeds": args.n_seeds,
        "modes": args.modes,
        "ckpt": args.ckpt,
        "horizons": HORIZONS,
        "records": all_records,
    }, indent=2))
    print(f"\n[saved] {out_json}")


if __name__ == "__main__":
    main()
