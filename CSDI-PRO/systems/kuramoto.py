"""Kuramoto coupled-oscillator system integrator.

    θ̇_i = ω_i + (K/N) Σ_j sin(θ_j - θ_i)

We use a fixed natural-frequency vector ω (drawn once from N(0, 1), seed=0)
so all trajectories share the same ω and differ only in initial conditions —
matches the L63/L96 convention where the system is fully deterministic modulo
initial state.

State representation: to avoid the 2π-wraparound discontinuity during
delay-embedding + rollout, we store state as **(cos θ_i, sin θ_i)** rather
than raw θ_i. So for N=10 oscillators, state dim D = 2N = 20 (matching the
L96 N=20 feature scale).

Parameters (N=10):
  - K = 1.5 (above the critical K_c ≈ 2σ_ω ≈ 2 for unit-Gaussian ω;
    K=1.5 gives partial synchronization with residual chaos)
  - d_KY ≈ 4-6 (depends on K; slightly below N-1 rotating modes)
  - λ_1 ≈ 0.2-0.4 (per time unit, weak-to-moderate chaos)
"""
from __future__ import annotations

import numpy as np


KURAMOTO_N = 10
KURAMOTO_K = 1.5
KURAMOTO_DT = 0.1
KURAMOTO_LYAP = 0.3     # empirical estimate; refine with Rosenstein later
KURAMOTO_ATTRACTOR_STD = 0.7  # empirical std of (cos,sin) components


def _default_omega(N: int, seed: int = 0) -> np.ndarray:
    """Deterministic natural-frequency draw (same across all integrations)."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(N)


def integrate_kuramoto(
    n_steps: int,
    N: int = KURAMOTO_N, K: float = KURAMOTO_K, dt: float = KURAMOTO_DT,
    omega: np.ndarray | None = None,
    spinup: int = 2000, seed: int | None = 0,
) -> np.ndarray:
    """Returns trajectory of shape (n_steps, 2N) — stacked (cos θ, sin θ) pairs:
    [cos θ_0, sin θ_0, cos θ_1, sin θ_1, ..., cos θ_{N-1}, sin θ_{N-1}]."""
    if omega is None:
        omega = _default_omega(N)
    rng = np.random.default_rng(seed)
    theta = 2 * np.pi * rng.random(N)  # random initial phases in [0, 2π)

    def rhs(th):
        # pairwise sine of differences: dθ_i/dt = ω_i + (K/N) Σ sin(θ_j - θ_i)
        diff = np.sin(th[None, :] - th[:, None])   # (N, N)
        return omega + (K / N) * diff.sum(axis=1)

    # spin-up
    for _ in range(spinup):
        k1 = rhs(theta)
        k2 = rhs(theta + 0.5 * dt * k1)
        k3 = rhs(theta + 0.5 * dt * k2)
        k4 = rhs(theta + dt * k3)
        theta = (theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)) % (2 * np.pi)

    out = np.empty((n_steps, 2 * N), dtype=np.float32)
    for t in range(n_steps):
        out[t, 0::2] = np.cos(theta)
        out[t, 1::2] = np.sin(theta)
        k1 = rhs(theta)
        k2 = rhs(theta + 0.5 * dt * k1)
        k3 = rhs(theta + 0.5 * dt * k2)
        k4 = rhs(theta + dt * k3)
        theta = (theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)) % (2 * np.pi)
    return out


def kuramoto_attractor_std(n_steps: int = 50000, seed: int = 0) -> float:
    traj = integrate_kuramoto(n_steps, spinup=5000, seed=seed)
    return float(traj.std())


if __name__ == "__main__":
    import time
    t0 = time.time()
    traj = integrate_kuramoto(50000, spinup=5000, seed=0)
    t1 = time.time()
    print(f"Integrated 50k Kuramoto steps in {t1 - t0:.2f}s")
    print(f"Shape: {traj.shape}")
    print(f"Per-dim std: mean={traj.std(axis=0).mean():.3f}  "
          f"min={traj.std(axis=0).min():.3f}  max={traj.std(axis=0).max():.3f}")
    print(f"Overall std = {traj.std():.3f}  (constant set at {KURAMOTO_ATTRACTOR_STD})")
