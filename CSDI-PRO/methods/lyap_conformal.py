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


def lyap_growth(
    horizons: np.ndarray,
    lam: float,
    dt: float,
    mode: str = "saturating",
    cap: float | None = None,
) -> np.ndarray:
    """Growth factor g(h) used to rescale conformal nonconformity scores.

    Modes:
      - ``"exp"``:        pure ``exp(λ h dt)`` — original tech.md §4.4
      - ``"saturating"``: ``1 + (e^{λ h dt} − 1) / (1 + (e^{λ h dt} − 1) / cap)``
                          → equals ``exp(λ h dt)`` for small h, saturates at ``cap`` for large h
      - ``"clipped"``:    ``min(exp(λ h dt), cap)`` — hard cap
      - ``"sqrt_exp"``:   ``sqrt(exp(2·λ h dt))`` = ``exp(λ h dt)`` (same as exp, kept as stub)

    ``cap`` defaults to 10×: residual at 1/λ time unit is e ≈ 2.7, so ``cap=10``
    corresponds to ~2.3 Lyapunov times of growth before saturation.
    """
    h = np.asarray(horizons, dtype=float)
    raw = np.exp(lam * h * dt)
    if cap is None:
        cap = 10.0
    if mode == "exp":
        return raw
    if mode == "saturating":
        # smooth rational saturation: small-arg ≈ raw; large-arg → cap+1
        return 1.0 + (raw - 1.0) / (1.0 + (raw - 1.0) / cap)
    if mode == "clipped":
        return np.minimum(raw, cap)
    if mode == "sqrt_exp":
        return np.sqrt(np.exp(2 * lam * h * dt))
    raise ValueError(f"unknown growth mode {mode!r}")


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
    """Lyapunov-aware conformal: nonconformity rescaled by a growth model g(h).

    Growth modes (see :func:`lyap_growth`):

      - ``"exp"``         — original tech.md §4.4 formula; over-estimates past ~1 Λ time
      - ``"saturating"``  — default; ``cap`` gate prevents long-h overshoot (empirically
                             the fix for the dip-and-spike PICP curve we saw at h ≥ 16)
      - ``"clipped"``     — hard ``min(exp(·), cap)``
      - ``"empirical"``   — no λ; growth inferred from calibration residual scale per h bin.
                             Useful as a λ-free ablation; loses the physics narrative.
    """

    alpha: float = 0.1
    lam: float = 1.0  # max Lyapunov per time unit
    dt: float = 0.01
    growth_mode: str = "saturating"
    growth_cap: float = 10.0
    q: float = float("nan")
    _empirical_scale: dict[int, float] | None = None

    def _growth(self, horizons: np.ndarray) -> np.ndarray:
        if self.growth_mode == "empirical":
            assert self._empirical_scale is not None, "call calibrate() first for empirical mode"
            arr = np.asarray(horizons, dtype=float)
            out = np.ones_like(arr)
            # nearest-bin lookup (robust to unseen h's)
            keys = np.array(sorted(self._empirical_scale.keys()), dtype=float)
            vals = np.array([self._empirical_scale[int(k)] for k in keys])
            for i, h in enumerate(arr.ravel()):
                idx = int(np.argmin(np.abs(keys - h)))
                out.ravel()[i] = vals[idx]
            return out
        return lyap_growth(horizons, self.lam, self.dt, mode=self.growth_mode, cap=self.growth_cap)

    def calibrate(
        self,
        y_cal: np.ndarray,
        y_pred_cal: np.ndarray,
        sigma_cal: np.ndarray,
        horizons: np.ndarray,
    ) -> None:
        sigma_cal = np.maximum(sigma_cal, 1e-8)
        if self.growth_mode == "empirical":
            # per-horizon empirical scale = median(|residual| / sigma)
            h_flat = np.asarray(horizons).ravel()
            r_flat = (np.abs(y_cal - y_pred_cal) / sigma_cal).ravel()
            self._empirical_scale = {}
            for h in np.unique(h_flat.astype(int)):
                sel = h_flat.astype(int) == h
                self._empirical_scale[int(h)] = float(np.median(r_flat[sel])) + 1e-8
        growth = self._growth(horizons)
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
        growth = self._growth(horizons)
        width = self.q * sigma * growth
        return y_pred - width, y_pred + width


@dataclass
class AdaptiveLyapConformal:
    """Gibbs & Candès 2021 adaptive CP, Lyapunov-rescaled with saturating growth."""

    alpha: float = 0.1
    lam: float = 1.0
    dt: float = 0.01
    eta: float = 0.05
    growth_mode: str = "saturating"
    growth_cap: float = 10.0
    q: float = 1.0

    def initialise(self, y_cal: np.ndarray, y_pred_cal: np.ndarray, sigma_cal: np.ndarray, horizons: np.ndarray) -> None:
        base = LyapConformal(alpha=self.alpha, lam=self.lam, dt=self.dt,
                             growth_mode=self.growth_mode, growth_cap=self.growth_cap)
        base.calibrate(y_cal, y_pred_cal, sigma_cal, horizons)
        self.q = base.q

    def predict_interval(
        self,
        y_pred: np.ndarray,
        sigma: np.ndarray,
        horizons: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        sigma = np.maximum(sigma, 1e-8)
        growth = lyap_growth(horizons, self.lam, self.dt, mode=self.growth_mode, cap=self.growth_cap)
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
