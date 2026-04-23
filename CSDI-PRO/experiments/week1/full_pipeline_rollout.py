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
    # Auto-scale m_inducing with feature_dim: with too few inducing points in
    # high-D (e.g. L96 N=20 -> 100-D features), Matern kernel value
    # exp(-||x-x'||^2 / 2L^2) vanishes because ||x-x'|| ~ sqrt(D) >> L=1, causing
    # GP output to collapse to Y.mean() (constant). Heuristic: >= 5x feature_dim
    # inducing points, capped at n_train - 1. L63 (15-D) uses 128 unchanged.
    n_train, feat_dim = X.shape
    auto_m = min(n_train - 1, max(cfg["m_inducing"], 5 * feat_dim))
    gp = MultiOutputSVGP(SVGPConfig(
        m_inducing=auto_m, n_epochs=cfg["n_epochs"], lr=cfg["lr"], verbose=False,
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


def full_pipeline_ensemble_forecast(
    observed: np.ndarray,
    pred_len: int,
    K: int = 20,
    seed: int = 0,
    imp_kind: str = "ar_kalman",
    ic_perturb_scale: float = 0.15,
    process_noise_scale: float = 0.0,
    return_trained_gp: bool = False,
    **kwargs,
):
    """Probabilistic rollout: K parallel sample paths with chaos-driven divergence.

    Two sources of ensemble spread (both knobbable):

    * **IC perturbation** (``ic_perturb_scale``): add ``N(0, ic_perturb_scale^2)``
      noise to each sample's *last* state before rollout starts. For a chaotic
      system, these tiny initial perturbations are amplified exponentially at the
      Lyapunov rate — this is the classical ensemble-forecasting mechanism
      (Lorenz 1965, Leith 1974). At separatrix points, different samples can
      commit to different lobes purely from IC sensitivity, no artificial noise.

    * **Process noise** (``process_noise_scale``): optional per-step Gaussian
      injection scaled by the SVGP predictive std. Default 0 — adding it at every
      step tends to overwhelm the deterministic dynamics on smooth trajectories.

    Returns
    -------
    preds : np.ndarray, [K, pred_len, D]
        Ensemble of sample paths.
    meta (if ``return_trained_gp``) : dict with {'gp', 'taus', 'filled'}
        For downstream diagnostics (e.g. separatrix figure).
    """
    cfg = {**_DEFAULTS, **kwargs}
    rng = np.random.default_rng(seed)

    filled = impute(observed, kind=imp_kind)

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

    X, Y, _ = _build_delay_features(filled, taus)
    if X.shape[0] < 50:
        preds = np.broadcast_to(filled[-1:, :], (K, pred_len, filled.shape[1])).copy()
        if return_trained_gp:
            return preds, dict(gp=None, taus=taus, filled=filled)
        return preds

    # Auto-scale m_inducing with feature dim (see full_pipeline_forecast for rationale)
    n_train, feat_dim = X.shape
    auto_m = min(n_train - 1, max(cfg["m_inducing"], 5 * feat_dim))
    gp = MultiOutputSVGP(SVGPConfig(
        m_inducing=auto_m, n_epochs=cfg["n_epochs"], lr=cfg["lr"], verbose=False,
    )).fit(X, Y)

    D = filled.shape[1]
    histories = np.tile(filled, (K, 1, 1)).astype(np.float32)  # [K, T, D]
    # IC perturbation: jitter the *last (max(taus)+1)* states so every delay-query
    # coordinate is perturbed — otherwise the first query is nearly identical
    # across samples and ensemble can't split early.
    if ic_perturb_scale > 0:
        n_ic = int(max(taus)) + 1
        histories[:, -n_ic:, :] += (ic_perturb_scale *
                                     rng.standard_normal((K, n_ic, D)).astype(np.float32))
    preds = np.empty((K, pred_len, D), dtype=np.float32)
    T = filled.shape[0]
    n_lags = len(taus)
    feats_per_channel = n_lags + 1
    for h in range(pred_len):
        t = T + h - 1
        query = np.empty((K, feats_per_channel * D), dtype=np.float32)
        for d in range(D):
            cols = [histories[:, t, d]]
            for tau in taus:
                cols.append(histories[:, t - tau, d])
            query[:, d * feats_per_channel:(d + 1) * feats_per_channel] = np.stack(cols, axis=1)
        mu, sigma = gp.predict(query, return_std=True)
        if process_noise_scale > 0:
            eps = rng.standard_normal(mu.shape).astype(np.float32)
            next_states = (mu + process_noise_scale * sigma * eps).astype(np.float32)
        else:
            next_states = mu.astype(np.float32)
        preds[:, h, :] = next_states
        histories = np.concatenate([histories, next_states[:, None, :]], axis=1)

    if return_trained_gp:
        return preds, dict(gp=gp, taus=taus, filled=filled)
    return preds
