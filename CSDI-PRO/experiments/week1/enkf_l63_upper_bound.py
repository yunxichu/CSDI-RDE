"""Stochastic ensemble Kalman filter (EnKF) on Lorenz63 as a known-dynamics
upper bound. Used in §3 / Appendix B as a model-aware reference, not as a
direct competitor to the model-agnostic preprocessing pipeline.

Setup:
- system: Lorenz63 with known parameters (σ=10, ρ=28, β=8/3); RK4 forward.
- observation model: y_t = H x_t + η_t with H = I and
  η_t ~ N(0, R) where R = (max(σ_obs, 0.01) · σ_attr)^2 · I; here σ_obs is
  the v2 corruption-grid noise_std_frac (σ_obs = 0 in pure-sparsity cells
  is replaced by 0.01 to keep R > 0).
- mask: same as v2 grid (1000·seed + 5000 + grid_index).
- ensemble size: 100.
- after the n_ctx context window, the ensemble mean is taken as the
  initial condition for a deterministic RK4 forecast over pred_len steps.

VPT is computed against the same threshold rule as the rest of the paper.

Run:
  python -u -m experiments.week1.enkf_l63_upper_bound \
      --configs SP65 SP82 NO020 NO050 \
      --n_seeds 5 \
      --tag enkf_l63_v2_5seed
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.week1.corruption import make_corrupted_observations
from experiments.week1.lorenz63_utils import (
    LORENZ63_LYAP, LORENZ63_ATTRACTOR_STD, integrate_lorenz63,
    valid_prediction_time,
)


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "week1" / "results"
FIGS = REPO / "experiments" / "week1" / "figures"
RESULTS.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
CONFIG_JSON = REPO / "experiments" / "week1" / "configs" / "corruption_grid_v2.json"


SIGMA_LZ, RHO_LZ, BETA_LZ = 10.0, 28.0, 8.0 / 3.0


def _lorenz63_rhs(state: np.ndarray) -> np.ndarray:
    """state shape (..., 3); returns same shape."""
    x, y, z = state[..., 0], state[..., 1], state[..., 2]
    dx = SIGMA_LZ * (y - x)
    dy = x * (RHO_LZ - z) - y
    dz = x * y - BETA_LZ * z
    return np.stack([dx, dy, dz], axis=-1)


def _rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    k1 = _lorenz63_rhs(state)
    k2 = _lorenz63_rhs(state + 0.5 * dt * k1)
    k3 = _lorenz63_rhs(state + 0.5 * dt * k2)
    k4 = _lorenz63_rhs(state + dt * k3)
    return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def _rk4_rollout(x0: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
    """Deterministic forward rollout from x0 (shape (3,) or (B,3))."""
    out = np.empty((n_steps,) + x0.shape, dtype=np.float64)
    s = x0.astype(np.float64)
    for t in range(n_steps):
        s = _rk4_step(s, dt)
        out[t] = s
    return out


def enkf_assimilate(
    obs: np.ndarray,
    obs_mask: np.ndarray,
    R_diag: float,
    dt: float,
    n_members: int = 100,
    init_spread: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Run a stochastic EnKF on a Lorenz63 context window.

    obs: (T, 3) — y_t (NaN where missing, otherwise the noisy observation)
    obs_mask: (T, 3) — boolean, True where observed
    R_diag: scalar diagonal of observation-noise covariance
    Returns: posterior ensemble mean trajectory shape (T, 3) and final
    ensemble shape (n_members, 3) for forecast initialisation.
    """
    rng = np.random.default_rng(seed)
    T, D = obs.shape

    # Initialize ensemble around first observed timestep mean (or 0 if none).
    first_obs_idx = np.where(obs_mask.any(axis=1))[0]
    if len(first_obs_idx) == 0:
        init_mean = np.zeros(3)
    else:
        first = first_obs_idx[0]
        init_mean = np.where(obs_mask[first], obs[first], 0.0)
    ens = init_mean[None, :] + init_spread * rng.standard_normal((n_members, 3))

    posterior_mean = np.empty((T, 3), dtype=np.float64)
    posterior_mean[0] = ens.mean(axis=0)

    for t in range(1, T):
        ens = _rk4_step(ens, dt)
        if obs_mask[t].any():
            d_obs = int(obs_mask[t].sum())
            H = np.eye(3)[obs_mask[t]]
            R = R_diag * np.eye(d_obs)
            y_t = obs[t][obs_mask[t]]
            ens_obs = ens @ H.T
            mean_obs = ens_obs.mean(axis=0, keepdims=True)
            anomaly = ens_obs - mean_obs
            P_yy = (anomaly.T @ anomaly) / max(n_members - 1, 1) + R
            mean_state = ens.mean(axis=0, keepdims=True)
            anomaly_state = ens - mean_state
            P_xy = (anomaly_state.T @ anomaly) / max(n_members - 1, 1)
            K = P_xy @ np.linalg.inv(P_yy)
            perturbed = y_t[None, :] + rng.multivariate_normal(np.zeros(d_obs), R, n_members)
            innov = perturbed - ens_obs
            ens = ens + innov @ K.T
        posterior_mean[t] = ens.mean(axis=0)

    return posterior_mean, ens


def load_named_configs(names: list[str]) -> list[dict[str, Any]]:
    doc = json.loads(CONFIG_JSON.read_text())
    pool = []
    for key in ("legacy_diagonal", "fine_s_line", "fine_sigma_line",
                 "summary_path_candidate", "pattern_grid"):
        for i, cfg in enumerate(doc.get(key, [])):
            c = dict(cfg)
            c["_grid_index"] = i
            pool.append(c)
    by_name = {c["name"]: c for c in pool}
    out = []
    for n in names:
        if n not in by_name:
            raise SystemExit(f"config {n!r} not in v2 grid; available: {sorted(by_name)}")
        out.append(by_name[n])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["SP65", "SP82", "NO020", "NO050"])
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--n_ctx", type=int, default=512)
    ap.add_argument("--pred_len", type=int, default=128)
    ap.add_argument("--dt", type=float, default=0.025)
    ap.add_argument("--n_members", type=int, default=100)
    ap.add_argument("--init_spread", type=float, default=1.0)
    ap.add_argument("--floor_sigma_frac", type=float, default=0.01,
                    help="lower bound on observation-noise std fraction "
                         "(EnKF needs R > 0; pure-sparsity cells use this)")
    ap.add_argument("--tag", default="enkf_l63_v2_5seed")
    args = ap.parse_args()

    attr_std = float(LORENZ63_ATTRACTOR_STD)
    lyap = float(LORENZ63_LYAP)
    configs = load_named_configs(args.configs)
    print(f"[enkf-l63] attr_std={attr_std} lyap={lyap}")
    print(f"[enkf-l63] configs={[c['name'] for c in configs]} n_members={args.n_members}")

    records: list[dict[str, Any]] = []
    for cfg in configs:
        sparsity = float(cfg["sparsity"])
        sigma = float(cfg["noise_std_frac"])
        sigma_eff = max(sigma, args.floor_sigma_frac)
        R_diag = (sigma_eff * attr_std) ** 2
        print(f"\n=== {cfg['name']}  s={sparsity}  sigma={sigma}  "
              f"sigma_eff={sigma_eff:.4f}  R_diag={R_diag:.4f} ===")
        for i in range(args.n_seeds):
            seed = args.seed_offset + i
            traj = integrate_lorenz63(args.n_ctx + args.pred_len, dt=args.dt,
                                       spinup=2000, seed=seed).astype(np.float32)
            ctx_true = traj[: args.n_ctx]
            future_true = traj[args.n_ctx :]
            obs_res = make_corrupted_observations(
                ctx_true, mask_regime="iid_time",
                sparsity=sparsity, noise_std_frac=sigma,
                attractor_std=attr_std,
                seed=1000 * seed + 5000 + int(cfg["_grid_index"]),
                dt=args.dt, lyap=lyap, patch_length=16,
            )
            observed = obs_res.observed                       # (T, D), NaN at missing
            obs_mask = ~np.isnan(observed)
            obs_filled = np.where(obs_mask, observed, 0.0)

            t0 = time.time()
            try:
                _, post_ens = enkf_assimilate(
                    obs_filled, obs_mask, R_diag, args.dt,
                    n_members=args.n_members, init_spread=args.init_spread,
                    seed=seed * 31 + 11,
                )
                x0 = post_ens.mean(axis=0)
                forecast = _rk4_rollout(x0, args.pred_len, args.dt).astype(np.float32)
                err = None
            except Exception as e:
                forecast = None; err = str(e)[:200]
            t_infer = time.time() - t0

            if forecast is None:
                rec = dict(seed=int(seed), scenario=cfg["name"],
                            sparsity=sparsity, noise_std_frac=sigma,
                            keep_frac=float(obs_res.metadata["keep_frac"]),
                            vpt03=float("nan"), vpt05=float("nan"), vpt10=float("nan"),
                            infer_time_s=t_infer, error=err)
                print(f"  seed={seed} {cfg['name']:6s}  FAILED: {err}")
            else:
                vpt03 = valid_prediction_time(future_true, forecast, dt=args.dt,
                                               lyap=lyap, threshold=0.3,
                                               attractor_std=attr_std)
                vpt05 = valid_prediction_time(future_true, forecast, dt=args.dt,
                                               lyap=lyap, threshold=0.5,
                                               attractor_std=attr_std)
                vpt10 = valid_prediction_time(future_true, forecast, dt=args.dt,
                                               lyap=lyap, threshold=1.0,
                                               attractor_std=attr_std)
                rec = dict(seed=int(seed), scenario=cfg["name"],
                            sparsity=sparsity, noise_std_frac=sigma,
                            keep_frac=float(obs_res.metadata["keep_frac"]),
                            vpt03=float(vpt03), vpt05=float(vpt05),
                            vpt10=float(vpt10),
                            infer_time_s=t_infer, error=None)
                print(f"  seed={seed} {cfg['name']:6s}  keep={rec['keep_frac']:.2f}  "
                      f"VPT@1.0={vpt10:5.2f}  t={t_infer:.1f}s")
            records.append(rec)

    # Summary
    import collections
    acc = collections.defaultdict(list)
    for r in records:
        if r.get("error"): continue
        acc[r["scenario"]].append(float(r["vpt10"]))
    summary = {}
    for sc, vs in acc.items():
        v = np.array(vs)
        summary[sc] = {
            "n": int(len(v)),
            "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "median": float(np.median(v)),
            "pr_gt_0p5": float((v > 0.5).mean()),
            "pr_gt_1p0": float((v > 1.0).mean()),
        }

    out_json = RESULTS / f"enkf_l63_{args.tag}.json"
    out_json.write_text(json.dumps(dict(
        config=vars(args), records=records, summary=summary,
        meta=dict(attractor_std=attr_std, lyap=lyap),
    ), indent=2))
    print(f"\n[enkf-l63] saved -> {out_json}")
    print("\n[verdict] mean VPT@1.0 (median, Pr>0.5, Pr>1.0):")
    for sc, s in summary.items():
        print(f"  {sc:6s}  μ={s['mean']:.2f}  med={s['median']:.2f}  "
              f"Pr>0.5={s['pr_gt_0p5']:.0%}  Pr>1.0={s['pr_gt_1p0']:.0%}")


if __name__ == "__main__":
    main()
