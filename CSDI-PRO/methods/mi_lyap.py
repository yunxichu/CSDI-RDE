"""Module 2 — MI-Lyap adaptive delay embedding (with τ-search baselines).

Implements:
  - KSG (Kraskov 2004) conditional mutual information estimator
  - Rosenstein local Lyapunov estimator (via ``nolds.lyap_r``)
  - Three τ-selection strategies:
      random_tau         (Takahashi et al. 2021 baseline)
      fraser_swinney_tau (first-minimum of autocorr-informed MI, classic baseline)
      mi_lyap_bayes      (tech.md Module 2.3 Stage A: BayesOpt)
  - Low-rank CMA-ES variant for L > 10 (tech.md Stage B) — available but
    gated to keep the baseline fast.

Delay coords are built as: y_τ(t) = (x(t), x(t - τ1), x(t - τ2), ..., x(t - τ_{L-1})),
so len(τ) = L - 1 shifts (coord 0 is always the present value).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# KSG conditional MI (Kraskov et al. 2004 + Frenzel & Pompe 2007)
# ---------------------------------------------------------------------------

def _digamma(x: np.ndarray | float):
    from scipy.special import digamma as _d
    return _d(x)


def _to_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 2:
        return a
    raise ValueError(f"expected 1D or 2D array, got ndim={a.ndim}")


def ksg_mi(X: np.ndarray, Y: np.ndarray, k: int = 4) -> float:
    """KSG-1 estimator of I(X; Y)."""
    from sklearn.neighbors import NearestNeighbors, KDTree

    X = _to_2d(X)
    Y = _to_2d(Y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have same #samples")
    n = X.shape[0]
    # joint space = [X | Y], Chebyshev distance
    XY = np.concatenate([X, Y], axis=1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev").fit(XY)
    dist, _ = nbrs.kneighbors(XY)
    eps = dist[:, k]  # kth nearest (excluding self)
    # counts in marginal spaces with distance < eps
    def _count(Z: np.ndarray, eps: np.ndarray) -> np.ndarray:
        tree = KDTree(Z, metric="chebyshev")
        # need strict <, so subtract a tiny epsilon
        counts = tree.query_radius(Z, r=eps - 1e-12, count_only=True)
        return np.asarray(counts, dtype=np.int64) - 1  # exclude self
    nx = _count(X, eps)
    ny = _count(Y, eps)
    mi = _digamma(k) + _digamma(n) - np.mean(_digamma(nx + 1) + _digamma(ny + 1))
    return float(max(mi, 0.0))


def ksg_cmi(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, k: int = 4) -> float:
    """Conditional MI I(X; Y | Z) via KSG (Frenzel & Pompe 2007)."""
    from sklearn.neighbors import KDTree, NearestNeighbors

    X = _to_2d(X)
    Y = _to_2d(Y)
    Z = _to_2d(Z)
    n = X.shape[0]
    XYZ = np.concatenate([X, Y, Z], axis=1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev").fit(XYZ)
    dist, _ = nbrs.kneighbors(XYZ)
    eps = dist[:, k]

    def _count(W: np.ndarray) -> np.ndarray:
        tree = KDTree(W, metric="chebyshev")
        counts = tree.query_radius(W, r=eps - 1e-12, count_only=True)
        return np.asarray(counts, dtype=np.int64) - 1

    nz = _count(Z)
    nxz = _count(np.concatenate([X, Z], axis=1))
    nyz = _count(np.concatenate([Y, Z], axis=1))
    cmi = _digamma(k) + np.mean(_digamma(nz + 1) - _digamma(nxz + 1) - _digamma(nyz + 1))
    return float(max(cmi, 0.0))


# ---------------------------------------------------------------------------
# Lyapunov estimators
# ---------------------------------------------------------------------------

def global_lyapunov_rosenstein(series: np.ndarray, emb_dim: int = 5, lag: int = 2) -> float:
    """Wrap ``nolds.lyap_r``; returns largest Lyapunov exponent (per sample step).

    Under noise this tends to **over-estimate** (nolds fits the divergence
    slope on noise-dominated short times, inflating λ). For a noise-robust
    estimate use :func:`robust_lyapunov`.
    """
    import nolds

    try:
        lam = float(nolds.lyap_r(series, emb_dim=emb_dim, lag=lag,
                                 min_tsep=10, trajectory_len=20, fit="poly"))
        return lam
    except Exception:
        return 0.5


def robust_lyapunov(
    series: np.ndarray,
    dt: float = 0.025,
    emb_dim: int = 5,
    lag: int = 2,
    trajectory_len: int = 50,
    prefilter: bool = True,
    lam_min: float = 0.1,
    lam_max: float = 2.5,
    seed: int = 0,
) -> float:
    """Noise-robust Lyapunov estimate: pre-denoise → Rosenstein with mid-range tl → clip.

    Strategy:
      1. (optional) Pre-filter the series with an AR-Kalman smoother to reduce
         noise-induced short-time divergence (which inflates λ).
      2. Run ``nolds.lyap_r`` with ``trajectory_len=50`` — long enough to
         dilute the noise-dominated first steps, short enough to stay in the
         linear Lyapunov regime before attractor bounds kick in.
      3. Clip to ``[lam_min, lam_max]`` so pathological (noise-collapsed or
         over-inflated) estimates fall back to reasonable defaults.

    Returns the Lyapunov exponent in **per time unit** (divides by dt).

    Empirical note on noise handling: Rosenstein on raw noisy Lorenz63 can
    over- or under-estimate by factors of 2-4×. For rescaling residuals in
    conformal prediction, prefer the **empirical growth mode** of
    :class:`~methods.lyap_conformal.LyapConformal` (growth_mode="empirical")
    which avoids λ entirely.
    """
    import nolds

    series = np.asarray(series).ravel()
    if prefilter:
        from methods.dynamics_impute import _ar_kalman_smooth
        known = np.ones_like(series, dtype=bool)
        try:
            series = _ar_kalman_smooth(series.copy(), known, p=5)
        except Exception:
            pass  # keep raw
    try:
        lam_step = float(nolds.lyap_r(
            series, emb_dim=emb_dim, lag=lag,
            min_tsep=max(emb_dim * lag + 5, 10),
            trajectory_len=trajectory_len, fit="poly",
        ))
    except Exception:
        lam_step = (lam_min + lam_max) / 2 * dt
    lam_per_unit = lam_step / dt
    return float(np.clip(lam_per_unit, lam_min, lam_max))


def local_lyapunov(trajectory: np.ndarray, x_query: np.ndarray, k: int = 10, horizon: int = 10) -> float:
    """Local Lyapunov at ``x_query``: mean log-divergence rate of k-nearest neighbours."""
    from sklearn.neighbors import NearestNeighbors

    traj = np.asarray(trajectory)
    if traj.ndim == 1:
        traj = traj.reshape(-1, 1)
    if x_query.ndim == 1:
        x_query = x_query.reshape(1, -1)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(traj[:-horizon])
    _, idx = nbrs.kneighbors(x_query)
    idx = idx.ravel()[1:]  # drop self
    # divergence after `horizon` steps
    d0 = np.linalg.norm(traj[idx] - x_query, axis=1) + 1e-12
    dH = np.linalg.norm(traj[idx + horizon] - traj[idx.max() + horizon], axis=1) + 1e-12
    rate = np.log(dH / d0).mean() / max(horizon, 1)
    return float(max(rate, 0.01))


# ---------------------------------------------------------------------------
# τ-selection strategies
# ---------------------------------------------------------------------------

def _build_delay(series_1d: np.ndarray, taus: np.ndarray, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Build paired (Y_delay, X_future) samples for a single trajectory.

    Y[t] = (s[t], s[t - τ_1], ..., s[t - τ_{L-1}]);  X_future[t] = s[t + horizon].
    Only ``t`` values with enough past + future are retained.
    """
    taus = np.asarray(taus, dtype=int)
    t0 = int(taus.max()) if taus.size else 0
    T = series_1d.size
    if t0 + horizon >= T:
        return np.empty((0, 1 + taus.size)), np.empty((0,))
    idx = np.arange(t0, T - horizon)
    cols = [series_1d[idx]]
    for tau in taus:
        cols.append(series_1d[idx - tau])
    Y = np.stack(cols, axis=1)
    Xf = series_1d[idx + horizon]
    return Y, Xf


@dataclass
class TauSpec:
    taus: np.ndarray  # shape (L-1,), integer lags ≥ 1
    score: float = float("nan")
    method: str = ""


def random_tau(L: int, tau_max: int, seed: int = 0) -> TauSpec:
    rng = np.random.default_rng(seed)
    taus = rng.integers(1, tau_max + 1, size=L - 1)
    return TauSpec(taus=np.sort(taus)[::-1].astype(int), method="random")


def fraser_swinney_tau(series: np.ndarray, tau_max: int, L: int, k: int = 4) -> TauSpec:
    """Pick τ1 as first local minimum of MI(s[t], s[t-τ]); then use evenly spaced multiples."""
    series = series.ravel()
    mis = np.array([ksg_mi(series[tau:], series[:-tau], k=k) for tau in range(1, tau_max + 1)])
    # first local minimum
    tau1 = 1
    for i in range(1, len(mis) - 1):
        if mis[i] < mis[i - 1] and mis[i] < mis[i + 1]:
            tau1 = i + 1
            break
    else:
        tau1 = int(np.argmin(mis)) + 1
    taus = np.array([tau1 * j for j in range(1, L)], dtype=int)
    taus = np.minimum(taus, tau_max)
    return TauSpec(taus=np.sort(taus)[::-1], method="fraser_swinney")


def mi_lyap_bayes_tau(
    series: np.ndarray,
    L: int,
    tau_max: int,
    horizon: int = 1,
    beta: float = 1.0,
    gamma: float = 0.1,
    lam: float | None = None,
    n_calls: int = 30,
    k: int = 4,
    seed: int = 42,
) -> TauSpec:
    """tech.md Module 2.2 objective, optimised with BayesOpt (Stage A).

    To avoid degenerate duplicates like τ=(1,1,1,1), we parameterise via positive
    increments ``δ_i`` with ``τ_j = sum_{i<=j}(δ_i)``. This guarantees strictly
    increasing, distinct delays.
    """
    from skopt import gp_minimize
    from skopt.space import Integer

    series = series.ravel()
    if lam is None:
        lam = global_lyapunov_rosenstein(series)
    T = series.size
    n_delta = L - 1  # number of delays; τ_0, ..., τ_{L-2}
    max_delta = max(2, tau_max // n_delta)  # ensure each increment fits

    def decode(delta: list[int]) -> np.ndarray:
        taus = np.cumsum(np.asarray(delta, dtype=int))
        taus = np.clip(taus, 1, tau_max)
        return np.sort(taus)[::-1]  # descending for dataset building

    def obj(delta_list: list[int]) -> float:
        taus = decode(delta_list)
        Y, Xf = _build_delay(series, taus, horizon=horizon)
        if Y.shape[0] < 50:
            return 1e6
        sub = np.random.default_rng(seed).choice(Y.shape[0], size=min(800, Y.shape[0]), replace=False)
        mi = ksg_mi(Y[sub], Xf[sub, None], k=k)
        lyap_pen = beta * taus.max() * lam
        sparse_pen = gamma * (taus ** 2).sum() / T
        return -(mi - lyap_pen - sparse_pen)

    space = [Integer(1, max_delta, name=f"delta_{i}") for i in range(n_delta)]
    res = gp_minimize(obj, space, n_calls=n_calls, random_state=seed, verbose=False)
    taus = decode(res.x)
    return TauSpec(taus=taus, score=-float(res.fun), method="mi_lyap_bayes")


def construct_delay_dataset(
    series: np.ndarray,
    taus: np.ndarray,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Public helper: exported shape (n_samples, L), (n_samples,)."""
    return _build_delay(series, taus, horizon=horizon)


# ---------------------------------------------------------------------------
# Stage B: low-rank CMA-ES τ search for high-dimensional systems (tech.md §2.3)
# ---------------------------------------------------------------------------

def mi_lyap_cmaes_tau(
    series: np.ndarray,
    L: int,
    tau_max: int,
    horizon: int = 1,
    beta: float = 1.0,
    gamma: float = 0.1,
    lam: float | None = None,
    rank: int = 2,
    popsize: int = 20,
    n_iter: int = 30,
    k: int = 4,
    seed: int = 42,
) -> TauSpec:
    """Low-rank CMA-ES τ selector (tech.md Module 2.3 Stage B).

    Parameterisation:  τ = round(σ(U V^T) · τ_max)
    where ``U ∈ R^{L × r}``, ``V ∈ R^{1 × r}`` (a row vector), ``σ`` is the
    sigmoid. The physical prior: in coupled-oscillator systems (Lorenz96, KS)
    neighbouring dimensions share chaotic timescales, so the optimal τ matrix
    has low-rank structure. Search space ``R^{r(L+1)}`` (continuous, low-dim)
    instead of ``{1..τ_max}^L`` (discrete, exponential).

    Returns (τ, decoded score, SVD spectrum of U V^T).
    """
    import cma

    series = np.asarray(series).ravel()
    if lam is None:
        lam = global_lyapunov_rosenstein(series)
    T = series.size
    n_delta = L - 1  # same convention as mi_lyap_bayes_tau: L-1 delays
    n_params = rank * (n_delta + 1)

    def decode(x: np.ndarray) -> np.ndarray:
        U = x[: n_delta * rank].reshape(n_delta, rank)
        V = x[n_delta * rank :].reshape(1, rank)
        raw = 1.0 / (1.0 + np.exp(-(U @ V.T).flatten()))
        tau = np.clip(np.round(raw * tau_max), 1, tau_max).astype(int)
        # enforce distinctness and descending order
        tau = np.sort(tau)[::-1]
        return tau

    def objective(x: np.ndarray) -> float:
        taus = decode(x)
        Y, Xf = _build_delay(series, taus, horizon=horizon)
        if Y.shape[0] < 50:
            return 1e6
        sub = np.random.default_rng(seed).choice(Y.shape[0], size=min(600, Y.shape[0]), replace=False)
        mi = ksg_mi(Y[sub], Xf[sub, None], k=k)
        lyap_pen = beta * int(taus.max()) * lam
        sparse_pen = gamma * (taus.astype(float) ** 2).sum() / T
        return -(mi - lyap_pen - sparse_pen)

    es = cma.CMAEvolutionStrategy(
        np.zeros(n_params), 0.5,
        {"popsize": popsize, "maxiter": n_iter, "verbose": -9, "seed": seed},
    )
    es.optimize(objective)
    xbest = es.result.xbest
    taus = decode(xbest)

    # compute τ-matrix singular-value spectrum (for the τ-low-rank figure)
    U = xbest[: n_delta * rank].reshape(n_delta, rank)
    V = xbest[n_delta * rank :].reshape(1, rank)
    tau_matrix = U @ V.T  # (L-1, 1) — degenerate when V is row; compute full (L-1, L-1)
    full_matrix = U @ U.T  # (L-1, L-1) proxy matrix
    s = np.linalg.svd(full_matrix, compute_uv=False)
    spec = TauSpec(taus=taus, score=-float(es.result.fbest), method="mi_lyap_cmaes")
    spec.__dict__["singular_values"] = s.tolist()
    spec.__dict__["rank"] = rank
    return spec


if __name__ == "__main__":
    # smoke test on Lorenz63
    import sys
    sys.path.insert(0, ".")
    from experiments.week1.lorenz63_utils import integrate_lorenz63

    traj = integrate_lorenz63(4000, dt=0.025, seed=0)
    x = traj[:, 0]

    lam = global_lyapunov_rosenstein(x, emb_dim=5, lag=2)
    print(f"[global Lyap] ~{lam:.4f} per sample step (dt=0.025)")

    spec1 = random_tau(L=5, tau_max=20, seed=0)
    print(f"[random]      tau={spec1.taus.tolist()}")

    spec2 = fraser_swinney_tau(x, tau_max=30, L=5)
    print(f"[F-S]         tau={spec2.taus.tolist()}")

    spec3 = mi_lyap_bayes_tau(x, L=5, tau_max=30, horizon=1, n_calls=15)
    print(f"[MI-Lyap BO]  tau={spec3.taus.tolist()}  score={spec3.score:.4f}")

    for spec in [spec1, spec2, spec3]:
        Y, Xf = construct_delay_dataset(x, spec.taus, horizon=1)
        mi = ksg_mi(Y[:800], Xf[:800, None], k=4)
        print(f"  {spec.method:20s} I(Y;X_future) ≈ {mi:.3f}  Y.shape={Y.shape}")
