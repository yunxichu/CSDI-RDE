"""Full v2 pipeline rollout: sparse+noisy context → 128-step mean forecast.

Wraps Modules 1-3 to expose the same ``(observed, pred_len) → mean`` signature
as the other baselines in [phase_transition_pilot_v2.py](phase_transition_pilot_v2.py):

    M1  Dynamics-Aware imputation (AR-Kalman surrogate, Week-2 parity with ablation)
    M2  MI-Lyap BayesOpt τ selection on channel 0
    M3  SVGP on delay coords predicting Δstate := state(t+1) - state(t)
        (residual parameterisation; stabilises autoregressive rollout near a fixed point)

M4 (conformal) is deliberately omitted here — the phase-transition figure uses the
*point* forecast (VPT / NRMSE). M4 lives separately in the ablation and
horizon-calibration experiments.

This wrapper is meant to be cheap enough to run 7 scenarios × 5 seeds in ~10 min.
Knobs picked to keep each call ~10-15 s on a single GPU.
"""
from __future__ import annotations

import numpy as np

from methods.dynamics_impute import impute
from methods.mi_lyap import mi_lyap_bayes_tau, random_tau
from models.svgp import MultiOutputSVGP, SVGPConfig

_DEFAULTS = dict(
    L_embed=5, tau_max=30, bayes_calls=10,
    m_inducing=128, n_epochs=100, lr=1e-2,
    fast_tau=False,
)


def _build_delay_features(traj: np.ndarray, taus: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Input ``traj`` [T, D]; return (X, Y, t_idx) where X = delay coords across all
    channels at anchor times ``t_idx``, and Y = traj[t+1] (direct next-state target).

    Direct-state target is stabler than Δstate when the SVGP is undertrained —
    errors in Δ compound rapidly during autoregressive rollout.
    """
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
    return X, Y, t_idx


def _delay_query(history: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """Build a 1×(L*D) query vector from the LAST step of ``history``."""
    t = history.shape[0] - 1
    D = history.shape[1]
    feats = []
    for d in range(D):
        cols = [history[t, d]]
        for tau in taus:
            cols.append(history[t - tau, d])
        feats.append(np.asarray(cols, dtype=np.float32))
    return np.concatenate(feats, axis=0).reshape(1, -1)


def full_pipeline_forecast(
    observed: np.ndarray,
    pred_len: int,
    seed: int = 0,
    imp_kind: str = "ar_kalman",
    **kwargs,
) -> np.ndarray:
    """Run Modules 1-3 end-to-end and return [pred_len, D] mean forecast.

    ``observed`` may contain NaNs at missing timesteps; shape [T, D].
    Falls back gracefully: if AR-Kalman fails (singular), linear interp.
    """
    cfg = {**_DEFAULTS, **kwargs}

    # --- Module 1: imputation ---------------------------------------------------
    filled = impute(observed, kind=imp_kind)

    # --- Module 2: τ selection on channel 0 -----------------------------------
    if cfg["fast_tau"]:
        spec = random_tau(L=cfg["L_embed"], tau_max=cfg["tau_max"], seed=seed)
    else:
        try:
            spec = mi_lyap_bayes_tau(
                filled[:, 0], L=cfg["L_embed"], tau_max=cfg["tau_max"], horizon=1,
                n_calls=cfg["bayes_calls"], k=4, seed=seed,
            )
        except Exception:
            spec = random_tau(L=cfg["L_embed"], tau_max=cfg["tau_max"], seed=seed)
    taus = spec.taus

    # --- Module 3: SVGP on delta-coords --------------------------------------
    X, Y, _ = _build_delay_features(filled, taus)
    if X.shape[0] < 50:
        return np.repeat(filled[-1:, :], pred_len, axis=0)
    gp = MultiOutputSVGP(SVGPConfig(
        m_inducing=cfg["m_inducing"], n_epochs=cfg["n_epochs"], lr=cfg["lr"], verbose=False,
    )).fit(X, Y)

    # Autoregressive rollout: each step queries the SVGP with delay coords derived
    # from the predicted history; the first ``max(taus)`` steps still reach back
    # into the *imputed* context so rollout gets a valid delay query immediately.
    history = filled.copy()
    preds = np.empty((pred_len, filled.shape[1]), dtype=np.float32)
    for h in range(pred_len):
        q = _delay_query(history, taus)
        mu, _ = gp.predict(q, return_std=True)
        next_state = mu[0].astype(np.float32)
        preds[h] = next_state
        history = np.vstack([history, next_state[None, :]])
    return preds
