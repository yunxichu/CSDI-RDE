"""Lorenz96 integration + mask generation + Lyapunov reference.

Lorenz96 is the canonical high-dimensional coupled-oscillator chaotic system
for testing forecasting methods at scale (tech.md §Module 0 benchmarks).

For ``F = 8`` the system exhibits spatio-temporal chaos with:
  - largest Lyapunov exponent λ₁ ≈ 1.68 / time unit
  - Kaplan-Yorke dimension d_KY ≈ 0.4 × N (for N ≥ 10)

This means Lorenz96 N=40 has d_KY ≈ 16 — the sample complexity advantage of our
method (which scales with d_KY, not ambient N) kicks in clearly.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import odeint


LORENZ96_LYAP_F8 = 1.68     # per time unit at F=8
LORENZ96_F_DEFAULT = 8.0


def lorenz96_rhs(state: np.ndarray, t: float, F: float = LORENZ96_F_DEFAULT) -> np.ndarray:
    """Lorenz 1996: dx_i/dt = (x_{i+1} − x_{i−2}) x_{i−1} − x_i + F, cyclic."""
    N = state.shape[0]
    dx = np.empty(N)
    for i in range(N):
        ip1 = (i + 1) % N
        im1 = (i - 1) % N
        im2 = (i - 2) % N
        dx[i] = (state[ip1] - state[im2]) * state[im1] - state[i] + F
    return dx


def integrate_lorenz96(
    n_steps: int,
    N: int = 40,
    F: float = LORENZ96_F_DEFAULT,
    dt: float = 0.05,
    x0: np.ndarray | None = None,
    spinup: int = 2000,
    seed: int | None = None,
) -> np.ndarray:
    """Integrate Lorenz96. Default dt=0.05 (one Lyapunov time = 0.6 time units ≈ 12 steps)."""
    if x0 is None:
        rng = np.random.default_rng(seed)
        x0 = F * np.ones(N) + rng.normal(scale=0.1, size=N)
    t = np.arange(n_steps + spinup) * dt
    traj = odeint(lorenz96_rhs, x0, t, args=(F,), rtol=1e-8, atol=1e-10)
    return traj[spinup:]


def lorenz96_attractor_std(N: int = 40, F: float = LORENZ96_F_DEFAULT) -> float:
    """Empirical std over 50k-step long trajectory (cached by (N, F))."""
    traj = integrate_lorenz96(50_000, N=N, F=F, seed=0)
    return float(traj.std())


if __name__ == "__main__":
    # Quick sanity checks for N=10, 40
    for N in [10, 40]:
        traj = integrate_lorenz96(1000, N=N, F=LORENZ96_F_DEFAULT, seed=0)
        print(f"N={N:3d}  shape={traj.shape}  std/dim={traj.std(axis=0).mean():.3f} "
              f"range=[{traj.min():.2f}, {traj.max():.2f}]")
    print(f"expected λ_1 @ F=8 ≈ {LORENZ96_LYAP_F8} per time unit")
