"""Uncertainty-quantification metrics used across the v2 pipeline.

Functions return per-sample arrays where sensible; caller decides reduction.
All functions accept numpy arrays.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """CRPS for a Gaussian predictive, Gneiting & Raftery 2007 closed form."""
    sigma = np.maximum(sigma, 1e-12)
    z = (y_true - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_ensemble(y_true: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """CRPS from an ensemble (num_samples, ...) via kernel score identity.

    CRPS = E|X - y| - 0.5 E|X - X'|,  unbiased estimator (Zamo & Naveau 2018).
    samples: (M, *dims); y_true: (*dims,).
    """
    M = samples.shape[0]
    term1 = np.abs(samples - y_true[None]).mean(0)
    # E|X - X'| via sorting
    srt = np.sort(samples, axis=0)
    ranks = np.arange(1, M + 1)
    weights = (2 * ranks - M - 1).reshape([M] + [1] * (samples.ndim - 1))
    term2 = (weights * srt).sum(0) / (M * (M - 1))
    return term1 - term2


def picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability."""
    inside = (y_true >= lower) & (y_true <= upper)
    return float(inside.mean())


def mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean Prediction Interval Width."""
    return float((upper - lower).mean())


def winkler_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    """Winkler 1972 interval score (lower is better)."""
    width = upper - lower
    below = 2 / alpha * (lower - y_true) * (y_true < lower)
    above = 2 / alpha * (y_true - upper) * (y_true > upper)
    return width + below + above


def reliability_curve(
    y_true: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Probabilistic calibration curve: expected vs observed cumulative coverage.

    Returns ``(expected, observed)`` arrays of length ``n_bins``, each in ``[0, 1]``.
    The curve should lie on y=x for a perfectly calibrated Gaussian predictive.
    """
    sigma = np.maximum(sigma, 1e-12)
    z = (y_true - mu) / sigma
    expected = np.linspace(0.05, 0.95, n_bins)
    observed = np.array([float(np.mean(norm.cdf(z) <= q)) for q in expected])
    return expected, observed


def ece(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (lower is better)."""
    exp, obs = reliability_curve(y_true, mu, sigma, n_bins)
    return float(np.mean(np.abs(exp - obs)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 1000
    mu = rng.normal(size=n)
    sigma = np.full(n, 1.0)
    y = mu + rng.normal(size=n)  # well-calibrated
    print(f"CRPS (well-cal) = {crps_gaussian(y, mu, sigma).mean():.4f}")
    print(f"ECE  (well-cal) = {ece(y, mu, sigma):.4f}")
    lo, hi = mu - 1.96 * sigma, mu + 1.96 * sigma
    print(f"PICP (95% target) = {picp(y, lo, hi):.3f}")
    print(f"MPIW = {mpiw(lo, hi):.3f}")
