"""Module 1 (lightweight) — Dynamics-Aware imputation for sparse + noisy observations.

This is a **stand-in** for the full Dynamics-Aware CSDI from tech.md §1.
The full CSDI retraining with noise conditioning + dynamic delay mask is a
Week-7 task that takes hours of diffusion training; here we capture the two
load-bearing mechanisms so the ablation of Module 1 can be evaluated
already in Week 2:

  1. **Noise estimation** from observations (robust MAD on first differences)
  2. **Smoothness prior** in the infill:
        - linear:        naïve piecewise-linear (= Week 1 baseline)
        - cubic:         cubic spline (better than linear, no noise awareness)
        - dynamics:      cubic spline → Gaussian smoother with bandwidth tied
                         to the estimated noise level (tech.md §1.2 spirit:
                         σ_obs conditions the smoother)
        - ar_kalman:     AR(p) Kalman smoother on the observed subset

API: all functions take a 2-D ``observed`` array of shape (T, D) with NaN on
missing entries, return a filled (T, D) array of the same shape.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d


def estimate_noise_mad(obs_1d: np.ndarray) -> float:
    """Robust σ estimate from second differences (Rice-Rosenblatt 1983).

    For a slow-varying signal plus iid noise, second differences have variance
    6σ² regardless of the underlying signal smoothness, so MAD-of-2nd-diff is
    a consistent estimator of σ.
    """
    known = np.isfinite(obs_1d)
    y = obs_1d[known]
    if y.size < 6:
        return 0.0
    d2 = np.diff(y, n=2)
    mad = np.median(np.abs(d2 - np.median(d2)))
    return float(mad * 1.4826 / np.sqrt(6.0))


def _impute_column(col: np.ndarray, kind: str, ar_order: int = 5) -> np.ndarray:
    known = np.isfinite(col)
    if known.sum() < 4:
        return np.where(known, col, 0.0)
    idx = np.arange(col.size)

    if kind == "linear":
        return np.interp(idx, idx[known], col[known])

    if kind == "cubic":
        cs = CubicSpline(idx[known], col[known], extrapolate=True)
        return np.asarray(cs(idx))

    if kind == "dynamics":
        # (a) cubic spline infill
        cs = CubicSpline(idx[known], col[known], extrapolate=True)
        filled = np.asarray(cs(idx))
        # (b) Gaussian smoother bandwidth from noise MAD
        sigma_obs = estimate_noise_mad(col)
        # bandwidth scaling: enough smoothing to remove σ_obs but preserve signal
        # scale bandwidth with sqrt(observed gap) so sparse → more smoothing
        sparsity = 1.0 - known.mean()
        bandwidth = 1.0 + 5.0 * sparsity + 2.0 * sigma_obs / (col[known].std() + 1e-8)
        return gaussian_filter1d(filled, sigma=bandwidth)

    if kind == "ar_kalman":
        return _ar_kalman_smooth(col, known, p=ar_order)

    raise ValueError(f"unknown kind {kind!r}")


def _ar_kalman_smooth(col: np.ndarray, known: np.ndarray, p: int = 5) -> np.ndarray:
    """AR(p) estimated from observed subset → RTS smoother over full grid. Falls back to linear on ill-conditioning."""
    y = col[known]
    if y.size < 2 * p + 10:
        return np.interp(np.arange(col.size), np.where(known)[0], y)
    try:
        return _ar_kalman_smooth_impl(col, known, p)
    except np.linalg.LinAlgError:
        return np.interp(np.arange(col.size), np.where(known)[0], y)


def _ar_kalman_smooth_impl(col: np.ndarray, known: np.ndarray, p: int = 5) -> np.ndarray:
    y = col[known]

    # estimate AR(p) via least squares on observed subset (ignoring gaps)
    Y = y[p:]
    X = np.stack([y[p - k - 1 : y.size - k - 1] for k in range(p)], axis=1)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ coef
    sigma_proc = float(resid.std())
    sigma_obs = estimate_noise_mad(col)
    if sigma_obs <= 0:
        sigma_obs = 0.1 * sigma_proc + 1e-6

    T = col.size
    # state: last p observations (companion form)
    F = np.zeros((p, p))
    F[0] = coef
    F[1:, :-1] = np.eye(p - 1)
    H = np.zeros((1, p)); H[0, 0] = 1.0
    Q = np.zeros((p, p)); Q[0, 0] = sigma_proc ** 2
    R = np.array([[sigma_obs ** 2]])

    # Initialise with prior from unconditional mean
    mu0 = np.full(p, float(y.mean()))
    P0 = np.eye(p) * (y.var() + 1e-6)

    mu = [mu0]; P = [P0]
    mus = []; Ps = []

    # forward Kalman filter
    for t in range(T):
        mu_pred = F @ mu[-1]
        P_pred = F @ P[-1] @ F.T + Q
        if known[t]:
            y_t = float(col[t])
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            innov = y_t - (H @ mu_pred).item()
            mu_t = mu_pred + (K * innov).ravel()
            P_t = (np.eye(p) - K @ H) @ P_pred
        else:
            mu_t = mu_pred
            P_t = P_pred
        mu.append(mu_t); P.append(P_t)
        mus.append(mu_t); Ps.append(P_t)

    # RTS smoother
    mus_s = [mus[-1]]
    for t in range(T - 2, -1, -1):
        mu_pred = F @ mus[t]
        P_pred = F @ Ps[t] @ F.T + Q
        G = Ps[t] @ F.T @ np.linalg.inv(P_pred + 1e-6 * np.eye(p))
        mu_s = mus[t] + G @ (mus_s[0] - mu_pred)
        mus_s.insert(0, mu_s)

    # first dim of state = y(t)
    return np.array([m[0] for m in mus_s])


# Module-level τ override slot for csdi kind. Set via set_tau_override() from the
# calling runner (e.g. τ-coupling ablation in run_ablation_with_csdi.py). None =
# CSDI uses its internal delay_bias as learned during training (default behavior).
_TAU_OVERRIDE = None


def set_tau_override(tau):
    """Set the τ override for subsequent csdi-kind impute() calls. Pass None to clear.

    tau: 1-D int array or list of length L-1 (delays in steps). Used by
    §5.X1 τ-coupling ablation to compare random / mismatched / equidistant τ against
    the default M2-selected τ.
    """
    global _TAU_OVERRIDE
    _TAU_OVERRIDE = tau


def impute(observed: np.ndarray, kind: str = "dynamics", **kwargs) -> np.ndarray:
    """Impute a (T, D) array with NaNs.

    kind in {"linear", "cubic", "dynamics", "ar_kalman", "csdi"}.
    """
    observed = np.asarray(observed, dtype=np.float64)
    if kind == "csdi":
        from methods.csdi_impute_adapter import csdi_impute
        return csdi_impute(observed, tau_override=_TAU_OVERRIDE, **kwargs)
    if kwargs:
        raise TypeError(f"impute(kind={kind!r}) got CSDI-only kwargs: {sorted(kwargs)}")
    if observed.ndim == 1:
        return _impute_column(observed, kind=kind)
    out = np.empty_like(observed)
    for d in range(observed.shape[1]):
        out[:, d] = _impute_column(observed[:, d], kind=kind)
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from experiments.week1.lorenz63_utils import integrate_lorenz63, make_sparse_noisy

    traj = integrate_lorenz63(2000, dt=0.025, seed=0)
    obs, mask = make_sparse_noisy(traj, sparsity=0.6, noise_std_frac=0.3, seed=0)
    for kind in ["linear", "cubic", "dynamics", "ar_kalman"]:
        filled = impute(obs, kind=kind)
        rmse = float(np.sqrt(((filled - traj) ** 2).mean()))
        print(f"  {kind:10s} RMSE(imputed vs truth) = {rmse:.3f}")
    sigma_hat = np.mean([estimate_noise_mad(obs[:, d]) for d in range(3)])
    print(f"  estimated σ_obs = {sigma_hat:.3f}  (true = 0.3 * 8.51 ≈ 2.55)")
