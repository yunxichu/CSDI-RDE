"""Canonical Lorenz63 utilities (not using dysts-normalised time so VPT is in standard Lyapunov units)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import odeint


LORENZ63_LYAP = 0.906  # Sprott et al., standard Lorenz63 (sigma=10, rho=28, beta=8/3)
LORENZ63_ATTRACTOR_STD = 8.51  # mean per-axis std over 10^5 step trajectory (x:7.92, y:9.00, z:8.61)


def lorenz63_rhs(state: np.ndarray, t: float, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> np.ndarray:
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])


def integrate_lorenz63(
    n_steps: int,
    dt: float = 0.01,
    x0: np.ndarray | None = None,
    spinup: int = 2000,
    seed: int | None = None,
) -> np.ndarray:
    """Integrate Lorenz63 and discard spin-up so we start on the attractor."""
    if x0 is None:
        rng = np.random.default_rng(seed)
        x0 = rng.normal(size=3) * 0.5 + np.array([1.0, 1.0, 20.0])
    t_total = np.arange(n_steps + spinup) * dt
    traj = odeint(lorenz63_rhs, x0, t_total, rtol=1e-8, atol=1e-10)
    return traj[spinup:]


def make_sparse_noisy(
    traj: np.ndarray,
    sparsity: float,
    noise_std_frac: float,
    attractor_std: float = LORENZ63_ATTRACTOR_STD,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop fraction `sparsity` of observations + add Gaussian noise sigma = noise_std_frac * attractor_std.

    Returns (observed, mask) of same shape as traj. observed[~mask] is np.nan.
    """
    rng = np.random.default_rng(seed)
    mask = rng.random(traj.shape[0]) > sparsity  # True = kept
    sigma = noise_std_frac * attractor_std
    noise = rng.normal(scale=sigma, size=traj.shape)
    observed = traj + noise
    observed[~mask] = np.nan
    return observed, mask


def forward_fill(ts: np.ndarray) -> np.ndarray:
    """Classic forward-fill for 2-D (T, D) with NaNs. First NaNs fall back to mean of known values."""
    out = ts.copy()
    T, D = out.shape
    for d in range(D):
        col = out[:, d]
        known = ~np.isnan(col)
        if not known.any():
            out[:, d] = 0.0
            continue
        first_idx = np.argmax(known)
        fill = col[first_idx]
        for t in range(T):
            if np.isnan(col[t]):
                col[t] = fill
            else:
                fill = col[t]
        out[:, d] = col
    return out


def linear_interp_fill(ts: np.ndarray) -> np.ndarray:
    """Linear interpolation for each column; extrapolates using nearest value."""
    out = ts.copy()
    T, D = out.shape
    for d in range(D):
        col = out[:, d]
        known = ~np.isnan(col)
        if not known.any():
            out[:, d] = 0.0
            continue
        idx = np.arange(T)
        out[:, d] = np.interp(idx, idx[known], col[known])
    return out


@dataclass
class HarshnessScenario:
    name: str
    sparsity: float  # fraction of time steps dropped
    noise_std_frac: float  # noise sigma / attractor_std


PILOT_SCENARIOS: list[HarshnessScenario] = [
    HarshnessScenario("S0", 0.00, 0.00),
    HarshnessScenario("S1", 0.20, 0.10),
    HarshnessScenario("S2", 0.40, 0.30),
    HarshnessScenario("S3", 0.60, 0.50),
    HarshnessScenario("S4", 0.75, 0.80),
    HarshnessScenario("S5", 0.90, 1.20),
    HarshnessScenario("S6", 0.95, 1.50),
]


def valid_prediction_time(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dt: float,
    lyap: float = LORENZ63_LYAP,
    threshold: float = 0.3,
    attractor_std: float = LORENZ63_ATTRACTOR_STD,
) -> float:
    """VPT in Lyapunov times.

    We use relative error normalised by attractor std (so all-zeros baseline gives finite VPT=0):
        eps_t = ||y_true[t] - y_pred[t]|| / (sqrt(D) * attractor_std)
    VPT = (first t where eps_t > threshold) * dt * lyap. Capped at trajectory length.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {y_pred.shape}")
    T, D = y_true.shape
    err = np.linalg.norm(y_true - y_pred, axis=1) / (np.sqrt(D) * attractor_std)
    bad = np.where(err > threshold)[0]
    t_fail = bad[0] if len(bad) else T
    return t_fail * dt * lyap


def compute_attractor_std() -> float:
    """Utility to re-compute attractor std from a long trajectory."""
    traj = integrate_lorenz63(100_000, dt=0.01, seed=0)
    return float(traj.std())


if __name__ == "__main__":
    # Verify attractor std constant
    s = compute_attractor_std()
    print(f"attractor std (10^5 steps) = {s:.4f}")
    # Verify VPT calibration
    traj = integrate_lorenz63(2000, dt=0.01, seed=0)
    print(f"traj range: x [{traj[:,0].min():.2f}, {traj[:,0].max():.2f}]")
    # perfect predictor
    print(f"VPT perfect={valid_prediction_time(traj, traj, dt=0.01):.3f}")
    # zero predictor
    zero_pred = np.zeros_like(traj)
    print(f"VPT zero   ={valid_prediction_time(traj, zero_pred, dt=0.01):.3f}")
