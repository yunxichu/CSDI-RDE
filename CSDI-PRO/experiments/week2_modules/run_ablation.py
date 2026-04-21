"""Week 2 — Module-wise ablation of the v2 pipeline on Lorenz63.

Full pipeline (tech.md §Core):
    Observations
     └─ Module 1: Dynamics-Aware imputation (AR-Kalman surrogate for Week 7 CSDI)
        └─ Module 2: MI-Lyap BayesOpt τ selection
           └─ Module 3: Matern-5/2 SVGP on delay coordinates
              └─ Module 4: Lyap-Conformal

Ablations (each flips exactly one module):

    full                linear_imp  random_tau    exact_gpr    split_cp
    |                   |  (M1-)   |  (M2-a)     |  (M3-)     |  (M4-)
    |                   linear     fraser_swinney_tau          (none)
    |                                              (M2-b)

    all_off = linear + random_tau + GPR + split_cp (closest to v1 CSDI-RDE-GPR)

Metrics (reported mean ± std over seeds):
    - NRMSE at horizons {1, 4, 16, 64}
    - PICP@90 at horizons {1, 4, 16, 64}  (Lyap-CP vs Split)
    - MPIW @ same
    - CRPS at h=16
    - τ search time (s) — Module 2 only

Run:
    CUDA_VISIBLE_DEVICES=2 python -m experiments.week2_modules.run_ablation \
        --n_seeds 3 --scenario S3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np

from experiments.week1.lorenz63_utils import (
    LORENZ63_ATTRACTOR_STD,
    LORENZ63_LYAP,
    PILOT_SCENARIOS,
    HarshnessScenario,
    integrate_lorenz63,
    make_sparse_noisy,
)
from methods.dynamics_impute import impute
from methods.lyap_conformal import LyapConformal, SplitConformal
from methods.mi_lyap import (
    construct_delay_dataset,
    fraser_swinney_tau,
    global_lyapunov_rosenstein,
    mi_lyap_bayes_tau,
    random_tau,
    robust_lyapunov,
)
from metrics.chaos_metrics import nrmse as chaos_nrmse
from metrics.uq_metrics import crps_gaussian, mpiw, picp
from models.svgp import SVGP, MultiOutputSVGP, SVGPConfig

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "experiments" / "week2_modules" / "results"
FIG_DIR = REPO_ROOT / "experiments" / "week2_modules" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 4, 16, 64]
L_EMBED = 5
TAU_MAX = 30
N_CTX = 1200           # make ctx long enough for meaningful train/cal/test split
PRED_LEN = 128         # kept for the few metrics that use last-step future
BAYES_CALLS = 20       # τ-search BO iterations; keep modest for ablation speed
TRAIN_FRAC = 0.60
CAL_FRAC = 0.20
# test = remaining 20%


# ---------------------------------------------------------------------------
# Pipeline configurations (module-swap definitions)
# ---------------------------------------------------------------------------

CONFIGS: dict[str, dict] = {
    # "cp" is a tuple (kind, growth_mode) when kind=="lyap". For "split" no growth.
    "full":              dict(imp="ar_kalman", tau="mi_lyap",  gp="svgp",  cp="lyap", growth="saturating"),
    "full-empirical":    dict(imp="ar_kalman", tau="mi_lyap",  gp="svgp",  cp="lyap", growth="empirical"),
    "m1-linear":         dict(imp="linear",    tau="mi_lyap",  gp="svgp",  cp="lyap", growth="saturating"),
    "m2a-random":        dict(imp="ar_kalman", tau="random",   gp="svgp",  cp="lyap", growth="saturating"),
    "m2b-frasersw":      dict(imp="ar_kalman", tau="fraser",   gp="svgp",  cp="lyap", growth="saturating"),
    "m3-exactgpr":       dict(imp="ar_kalman", tau="mi_lyap",  gp="gpr",   cp="lyap", growth="saturating"),
    "m4-splitcp":        dict(imp="ar_kalman", tau="mi_lyap",  gp="svgp",  cp="split", growth=None),
    "m4-lyap-exp":       dict(imp="ar_kalman", tau="mi_lyap",  gp="svgp",  cp="lyap", growth="exp"),
    "all-off":           dict(imp="linear",    tau="random",   gp="gpr",   cp="split", growth=None),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ExactGPRWrapper:
    """Minimal wrapper to match SVGP API using the project's self-implemented GPR."""

    def __init__(self) -> None:
        from gpr.gpr_module import GaussianProcessRegressor
        self._cls = GaussianProcessRegressor
        self._gp: object | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExactGPRWrapper":
        # sub-sample to a max of 1000 to stay tractable
        if X.shape[0] > 1000:
            idx = np.random.default_rng(0).choice(X.shape[0], 1000, replace=False)
            X = X[idx]; y = y[idx]
        self._gp = self._cls(noise=1e-4)
        self._gp.fit(X, y, optimize=False)
        return self

    def predict(self, X: np.ndarray, return_std: bool = True):
        mean, std = self._gp.predict(X, return_std=True)
        return (mean, std) if return_std else mean


class MultiOutputExactGPR:
    def __init__(self) -> None:
        self.gps: list[ExactGPRWrapper] = []

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MultiOutputExactGPR":
        if Y.ndim == 1: Y = Y[:, None]
        self.gps = [ExactGPRWrapper().fit(X, Y[:, d]) for d in range(Y.shape[1])]
        return self

    def predict(self, X: np.ndarray, return_std: bool = True):
        means, stds = [], []
        for g in self.gps:
            m, s = g.predict(X, return_std=True)
            means.append(m); stds.append(s)
        mean = np.stack(means, axis=-1); std = np.stack(stds, axis=-1)
        return (mean, std) if return_std else mean


def build_pipeline(ctx_filled: np.ndarray, cfg: dict, lam_hat: float, seed: int):
    """Execute Modules 1→3 up to GP fitting; return (gp, taus, lam_hat, tau_sec)."""
    # Module 2: τ selection (on channel 0, then broadcast to all channels)
    series = ctx_filled[:, 0]
    t0 = time.time()
    if cfg["tau"] == "random":
        spec = random_tau(L=L_EMBED, tau_max=TAU_MAX, seed=seed)
    elif cfg["tau"] == "fraser":
        spec = fraser_swinney_tau(series, tau_max=TAU_MAX, L=L_EMBED, k=4)
    elif cfg["tau"] == "mi_lyap":
        spec = mi_lyap_bayes_tau(
            series, L=L_EMBED, tau_max=TAU_MAX, horizon=1,
            lam=lam_hat, n_calls=BAYES_CALLS, k=4, seed=seed,
        )
    else:
        raise ValueError(cfg["tau"])
    tau_sec = time.time() - t0
    taus = spec.taus

    # Build delay coords for each channel, for each horizon.
    # Multi-output regression: predict (s_1(t+h), s_2(t+h), s_3(t+h)) from delay coord of channel 0.
    # Simpler: concat delay coords from all D channels as input (richer regressor).
    D = ctx_filled.shape[1]
    per_horizon_datasets = {}
    for h in HORIZONS:
        Xs = []; Ys = []
        t0_req = int(taus.max())
        # We'll build once and accumulate for each horizon; require sample t such that t + h < T.
        T = ctx_filled.shape[0]
        if t0_req + h >= T:
            continue
        t_idx = np.arange(t0_req, T - h)
        # delay features: per-channel concat
        feats = []
        for d in range(D):
            cols = [ctx_filled[t_idx, d]]
            for tau in taus:
                cols.append(ctx_filled[t_idx - tau, d])
            feats.append(np.stack(cols, axis=1))
        X_all = np.concatenate(feats, axis=1)  # (n, L * D)
        Y_all = ctx_filled[t_idx + h, :]       # (n, D)
        per_horizon_datasets[h] = (X_all, Y_all, t_idx)

    # Module 3: GP fitting — one multi-output GP per horizon.
    # Split train(60) / cal(20) / test(20) on the delay-coord dataset.
    gp_cfg = SVGPConfig(m_inducing=128, n_epochs=120, lr=1e-2, verbose=False)
    per_horizon_gp = {}
    for h, (X_all, Y_all, t_idx) in per_horizon_datasets.items():
        n = X_all.shape[0]
        n_train = int(TRAIN_FRAC * n)
        n_cal = int(CAL_FRAC * n)
        X_tr, Y_tr = X_all[:n_train], Y_all[:n_train]
        X_cal, Y_cal = X_all[n_train : n_train + n_cal], Y_all[n_train : n_train + n_cal]
        X_test, Y_test = X_all[n_train + n_cal :], Y_all[n_train + n_cal :]
        if cfg["gp"] == "svgp":
            gp = MultiOutputSVGP(gp_cfg).fit(X_tr, Y_tr)
        elif cfg["gp"] == "gpr":
            gp = MultiOutputExactGPR().fit(X_tr, Y_tr)
        else:
            raise ValueError(cfg["gp"])
        per_horizon_gp[h] = (gp, X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test)

    return per_horizon_gp, taus, tau_sec


def evaluate_horizons(
    per_horizon_gp: dict,
    ctx_filled: np.ndarray,
    future_true: np.ndarray,
    taus: np.ndarray,
    lam_hat: float,
    dt: float,
    cp_kind: str,
    growth: str | None = "saturating",
    alpha: float = 0.1,
) -> dict:
    """Evaluate NRMSE / PICP / MPIW / CRPS using the held-out test fold of the delay-coord dataset.

    For each horizon h, we score the GP on its test fold (~20% of (X, Y)) and
    compute per-horizon metrics over ``n_test * D`` residuals. This gives PICP
    with ~50+ effective samples per horizon, so it's meaningful.
    """
    out: dict = {}
    for h, (gp, X_tr, Y_tr, X_cal, Y_cal, X_test, Y_test) in per_horizon_gp.items():
        mu_test, std_test = gp.predict(X_test, return_std=True)
        mu_cal, std_cal = gp.predict(X_cal, return_std=True)

        nrmse_h = chaos_nrmse(Y_test, mu_test, attractor_std=LORENZ63_ATTRACTOR_STD)
        crps_h = float(np.mean(crps_gaussian(Y_test, mu_test, std_test)))

        horizons_cal = np.full(Y_cal.shape, fill_value=h, dtype=float)
        horizons_test = np.full(Y_test.shape, fill_value=h, dtype=float)
        if cp_kind == "lyap":
            cp = LyapConformal(alpha=alpha, lam=lam_hat, dt=dt,
                               growth_mode=growth or "saturating", growth_cap=10.0)
            cp.calibrate(Y_cal, mu_cal, std_cal, horizons_cal)
            lo, hi = cp.predict_interval(mu_test, std_test, horizons_test)
        elif cp_kind == "split":
            cp = SplitConformal(alpha=alpha)
            cp.calibrate(Y_cal, mu_cal, std_cal)
            lo, hi = cp.predict_interval(mu_test, std_test)
        else:
            raise ValueError(cp_kind)

        out[h] = dict(
            n_train=int(X_tr.shape[0]),
            n_cal=int(X_cal.shape[0]),
            n_test=int(X_test.shape[0]),
            nrmse=float(nrmse_h),
            picp=float(picp(Y_test, lo, hi)),
            mpiw=float(mpiw(lo, hi)),
            crps=float(crps_h),
            q_conformal=float(cp.q),
        )
    return out


# ---------------------------------------------------------------------------
# Run ablation
# ---------------------------------------------------------------------------

def run_single(cfg_name: str, cfg: dict, seed: int, scenario: HarshnessScenario, dt: float = 0.025) -> dict:
    traj = integrate_lorenz63(N_CTX + PRED_LEN, dt=dt, seed=seed, spinup=2000)
    ctx_true = traj[:N_CTX]
    future_true = traj[N_CTX:]

    obs, mask = make_sparse_noisy(
        ctx_true, sparsity=scenario.sparsity, noise_std_frac=scenario.noise_std_frac,
        attractor_std=LORENZ63_ATTRACTOR_STD, seed=seed,
    )
    ctx_filled = impute(obs, kind=cfg["imp"])
    lam_hat = robust_lyapunov(ctx_filled[:, 0], dt=dt, emb_dim=5, lag=2, trajectory_len=50, prefilter=True)

    per_hgp, taus, tau_sec = build_pipeline(ctx_filled, cfg, lam_hat=lam_hat * dt, seed=seed)
    metrics = evaluate_horizons(per_hgp, ctx_filled, future_true, taus,
                                lam_hat=lam_hat * dt, dt=dt, cp_kind=cfg["cp"],
                                growth=cfg.get("growth"))
    return dict(
        cfg_name=cfg_name, seed=seed, scenario=scenario.name,
        sparsity=scenario.sparsity, noise=scenario.noise_std_frac,
        lam_hat=float(lam_hat), tau_sec=tau_sec, taus=taus.tolist(),
        metrics=metrics, cfg=cfg,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--scenario", type=str, default="S3")
    ap.add_argument("--tag", type=str, default=None)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()))
    args = ap.parse_args()

    tag = args.tag or f"{args.scenario}_n{args.n_seeds}"
    scenario = next(s for s in PILOT_SCENARIOS if s.name == args.scenario)

    print(f"=== W2 ablation: scenario={scenario.name} sparsity={scenario.sparsity} σ={scenario.noise_std_frac} "
          f"seeds=[0..{args.n_seeds - 1}] configs={args.configs}")

    all_results: list[dict] = []
    for cfg_name in args.configs:
        cfg = CONFIGS[cfg_name]
        for seed in range(args.n_seeds):
            t0 = time.time()
            rec = run_single(cfg_name, cfg, seed=seed, scenario=scenario)
            rec["total_sec"] = time.time() - t0
            all_results.append(rec)
            print(
                f"  [{cfg_name:15s}] seed={seed} τ={rec['taus']}  τ_sec={rec['tau_sec']:5.1f}  "
                f"total={rec['total_sec']:5.1f}  h=1 nrmse={rec['metrics'].get(1,{}).get('nrmse',0):.3f} "
                f"picp={rec['metrics'].get(1,{}).get('picp',0):.2f}"
            )

    out_json = OUT_DIR / f"ablation_{tag}.json"
    out_json.write_text(json.dumps({
        "scenario": scenario.__dict__,
        "config_defs": CONFIGS,
        "records": all_results,
        "horizons": HORIZONS,
        "hyper": dict(L=L_EMBED, tau_max=TAU_MAX, n_ctx=N_CTX, pred_len=PRED_LEN, bayes_calls=BAYES_CALLS),
    }, indent=2))
    print(f"\n[ablation] records saved to {out_json}")


if __name__ == "__main__":
    main()
