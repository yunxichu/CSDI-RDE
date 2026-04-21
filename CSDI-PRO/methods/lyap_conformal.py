"""Module 4 — Conformal prediction for chaotic forecasts.

Three calibrators are provided, stacked so the pipeline can switch at will:

  - SplitConformal      baseline (Vovk 2005, CQR-style nonconformity)
  - LyapConformal       tech.md Module 4.4 — rescale nonconformity by exp(lam * h * dt)
  - AdaptiveLyapConformal   tech.md Module 4.6 — online q update (Gibbs & Candès 2021)

All three return ``(lower, upper)`` intervals; inputs are numpy arrays.

Interface convention: ``y_pred`` and ``sigma`` are GP (or any) posterior mean/std;
``horizons`` is an integer array of forecast step indices for each point (0 for
single-step, larger for multi-step). This lets Lyap-CP grow its bands
proportionally to ``exp(lam * h * dt)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _quantile_level(n: int, alpha: float) -> float:
    """Finite-sample conformal level."""
    k = np.ceil((n + 1) * (1 - alpha))
    return float(min(k / n, 1.0))


@dataclass
class SplitConformal:
    alpha: float = 0.1
    q: float = float("nan")

    def calibrate(self, y_cal: np.ndarray, y_pred_cal: np.ndarray, sigma_cal: np.ndarray) -> None:
        """CQR-style nonconformity: |y - yhat| / sigma."""
        sigma_cal = np.maximum(sigma_cal, 1e-8)
        scores = np.abs(y_cal - y_pred_cal) / sigma_cal
        level = _quantile_level(scores.size, self.alpha)
        self.q = float(np.quantile(scores.ravel(), level))

    def predict_interval(
        self,
        y_pred: np.ndarray,
        sigma: np.ndarray,
        horizons: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        sigma = np.maximum(sigma, 1e-8)
        width = self.q * sigma
        return y_pred - width, y_pred + width


@dataclass
class LyapConformal:
    """Lyapunov-aware conformal: nonconformity divided by exp(lam * h * dt)."""

    alpha: float = 0.1
    lam: float = 1.0  # max Lyapunov
    dt: float = 0.01
    q: float = float("nan")

    def calibrate(
        self,
        y_cal: np.ndarray,
        y_pred_cal: np.ndarray,
        sigma_cal: np.ndarray,
        horizons: np.ndarray,
    ) -> None:
        sigma_cal = np.maximum(sigma_cal, 1e-8)
        growth = np.exp(self.lam * horizons * self.dt)
        scores = np.abs(y_cal - y_pred_cal) / (sigma_cal * growth)
        level = _quantile_level(scores.size, self.alpha)
        self.q = float(np.quantile(scores.ravel(), level))

    def predict_interval(
        self,
        y_pred: np.ndarray,
        sigma: np.ndarray,
        horizons: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        sigma = np.maximum(sigma, 1e-8)
        growth = np.exp(self.lam * horizons * self.dt)
        width = self.q * sigma * growth
        return y_pred - width, y_pred + width


@dataclass
class AdaptiveLyapConformal:
    """Gibbs & Candès 2021 adaptive CP, Lyapunov-rescaled.

    Online update ``q_t = q_{t-1} + eta * (miss_t - alpha)``, where
    ``miss_t`` is 1 if the last interval missed, 0 otherwise.
    """

    alpha: float = 0.1
    lam: float = 1.0
    dt: float = 0.01
    eta: float = 0.05
    q: float = 1.0

    def initialise(self, y_cal: np.ndarray, y_pred_cal: np.ndarray, sigma_cal: np.ndarray, horizons: np.ndarray) -> None:
        base = LyapConformal(alpha=self.alpha, lam=self.lam, dt=self.dt)
        base.calibrate(y_cal, y_pred_cal, sigma_cal, horizons)
        self.q = base.q

    def predict_interval(
        self,
        y_pred: np.ndarray,
        sigma: np.ndarray,
        horizons: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        sigma = np.maximum(sigma, 1e-8)
        growth = np.exp(self.lam * horizons * self.dt)
        width = self.q * sigma * growth
        return y_pred - width, y_pred + width

    def update(self, y_true_next: float, lower: float, upper: float) -> None:
        miss = float(y_true_next < lower or y_true_next > upper)
        self.q = max(self.q + self.eta * (miss - self.alpha), 1e-6)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_cal, n_test = 500, 500
    H = 50  # max horizon
    lam, dt = 1.0, 0.01

    # synthetic: predictive sigma=1, but real errors grow exp(lam*h*dt)
    h_cal = rng.integers(1, H + 1, size=n_cal)
    err_cal = rng.normal(size=n_cal) * np.exp(lam * h_cal * dt)
    y_pred_cal = np.zeros(n_cal); sigma_cal = np.ones(n_cal); y_cal = y_pred_cal + err_cal

    h_test = rng.integers(1, H + 1, size=n_test)
    err_test = rng.normal(size=n_test) * np.exp(lam * h_test * dt)
    y_pred_test = np.zeros(n_test); sigma_test = np.ones(n_test); y_test = y_pred_test + err_test

    print("=== target 90% coverage, errors grow exp(lam*h*dt) ===")
    for name, cp in [
        ("Split", SplitConformal(alpha=0.1)),
        ("Lyap ", LyapConformal(alpha=0.1, lam=lam, dt=dt)),
    ]:
        if isinstance(cp, LyapConformal):
            cp.calibrate(y_cal, y_pred_cal, sigma_cal, h_cal)
            lo, hi = cp.predict_interval(y_pred_test, sigma_test, h_test)
        else:
            cp.calibrate(y_cal, y_pred_cal, sigma_cal)
            lo, hi = cp.predict_interval(y_pred_test, sigma_test)
        cov = ((y_test >= lo) & (y_test <= hi)).mean()
        mw = (hi - lo).mean()
        # coverage by horizon bin
        bins = [1, 10, 20, 30, 40, 50]
        per_bin = [((y_test[(h_test >= a) & (h_test < b)] >= lo[(h_test >= a) & (h_test < b)]) &
                    (y_test[(h_test >= a) & (h_test < b)] <= hi[(h_test >= a) & (h_test < b)])).mean()
                   for a, b in zip(bins[:-1], bins[1:])]
        print(f"  {name}: PICP={cov:.3f}  MPIW={mw:.2f}  per-bin PICP=[{', '.join(f'{x:.2f}' for x in per_bin)}]  q={cp.q:.3f}")
