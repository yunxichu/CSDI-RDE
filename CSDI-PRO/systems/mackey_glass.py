"""Mackey-Glass delay-differential equation integrator.

    dx/dt = β * x(t-τ) / (1 + x(t-τ)^n) - γ * x(t)

Standard chaotic parameters: β=0.2, γ=0.1, n=10, τ=17 (Mackey & Glass 1977,
Farmer 1982 for chaotic regime). dt=1.0 is the conventional time step.

Key constants for this repository:
  - d_KY ≈ 2.1 at τ=17 (low-dim attractor despite infinite-dim state space)
  - λ_1 ≈ 0.006 per time unit (weak chaos → long Lyapunov horizon)
  - attractor_std ≈ 0.34 (50k-step empirical, τ=17)

Mackey-Glass is the canonical testbed for delay-embedding methods: the
dynamics are *defined* by a delay ODE, so any delay-embedding predictor has
inductive-bias alignment. Our M2 MI-Lyap τ-search should recover the true
τ=17 from the data, which serves as a diagnostic check of the τ module.
"""
from __future__ import annotations

import numpy as np


# Standard chaotic regime (Farmer 1982).
MACKEY_GLASS_BETA = 0.2
MACKEY_GLASS_GAMMA = 0.1
MACKEY_GLASS_N_EXP = 10
MACKEY_GLASS_TAU = 17
MACKEY_GLASS_DT = 1.0

# Empirical constants (see mackey_glass_attractor_std() for regeneration).
MACKEY_GLASS_LYAP = 0.006       # per time unit, τ=17, n=10
MACKEY_GLASS_ATTRACTOR_STD = 0.228  # empirical over 50k-step trajectory, τ=17


def integrate_mackey_glass(
    n_steps: int,
    beta: float = MACKEY_GLASS_BETA,
    gamma: float = MACKEY_GLASS_GAMMA,
    n_exp: int = MACKEY_GLASS_N_EXP,
    tau: int = MACKEY_GLASS_TAU,
    dt: float = MACKEY_GLASS_DT,
    spinup: int = 2000,
    seed: int | None = 0,
    initial: float | None = None,
) -> np.ndarray:
    """Integrate Mackey-Glass with RK4 + delay lookup from a history buffer.

    ``tau`` is specified in *discrete steps* at stride ``dt`` (so real delay =
    ``tau * dt`` time units); τ=17 at dt=1.0 matches the Farmer 1982 standard.

    Returns
    -------
    traj : np.ndarray, shape (n_steps, 1)
        1-D trajectory (MG is scalar). Shaped as (T, 1) so it plugs into the
        existing pipeline which expects (T, D) with D=1 here.
    """
    rng = np.random.default_rng(seed)
    total = n_steps + spinup
    buf = np.empty(total + tau, dtype=np.float64)

    if initial is None:
        initial = 1.0 + 0.1 * rng.standard_normal()
    buf[:tau] = initial
    buf[tau] = initial

    def rhs(x: float, x_delayed: float) -> float:
        return beta * x_delayed / (1.0 + x_delayed ** n_exp) - gamma * x

    for t in range(tau, total + tau - 1):
        x = buf[t]
        x_del = buf[t - tau]
        # RK4: the delay term is frozen within the step (standard DDE trick
        # for fixed-dt integrators — small error since dt ≪ τ).
        k1 = rhs(x, x_del)
        k2 = rhs(x + 0.5 * dt * k1, x_del)
        k3 = rhs(x + 0.5 * dt * k2, x_del)
        k4 = rhs(x + dt * k3, x_del)
        buf[t + 1] = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    traj = buf[tau + spinup:tau + total].reshape(-1, 1).astype(np.float32)
    return traj


def mackey_glass_attractor_std(
    n_steps: int = 50000, tau: int = MACKEY_GLASS_TAU, seed: int = 0,
) -> float:
    """Empirical attractor std after burn-in. Use this (not the constant) when
    you change (β, γ, n, τ) away from the defaults."""
    traj = integrate_mackey_glass(n_steps, tau=tau, spinup=5000, seed=seed)
    return float(traj[:, 0].std())


if __name__ == "__main__":
    import time
    t0 = time.time()
    traj = integrate_mackey_glass(50000, spinup=5000, seed=0)
    t1 = time.time()
    print(f"Integrated 50k steps in {t1 - t0:.2f}s")
    print(f"Shape: {traj.shape}")
    print(f"Range: [{traj.min():.3f}, {traj.max():.3f}]  mean={traj.mean():.3f}  std={traj.std():.3f}")
    print(f"Expected std ≈ 0.34; computed: {traj.std():.4f}")
