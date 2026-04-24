"""Rössler system integrator.

    dx/dt = -y - z
    dy/dt = x + a*y
    dz/dt = b + z*(x - c)

Standard chaotic parameters: a=0.2, b=0.2, c=5.7 (Rössler 1976).

Constants:
  - d_KY ≈ 2.01 (near-2D attractor despite D=3 ambient)
  - λ_1 ≈ 0.07 per time unit (very weak chaos; long Lyapunov horizon)
  - attractor_std ≈ 5.5 (empirical over 50k steps at dt=0.1)

Weak chaos → high VPT ceiling (15-20 Λ clean). DeepEDM paper reports direct
Rössler numbers so this serves as a calibration + "clean regime" showcase.
"""
from __future__ import annotations

import numpy as np


ROSSLER_A = 0.2
ROSSLER_B = 0.2
ROSSLER_C = 5.7
ROSSLER_DT = 0.1
ROSSLER_LYAP = 0.071      # Sprott / Rössler 1976 standard chaos
ROSSLER_ATTRACTOR_STD = 4.45  # empirical over 50k steps (per-dim mean std)


def integrate_rossler(
    n_steps: int,
    a: float = ROSSLER_A, b: float = ROSSLER_B, c: float = ROSSLER_C,
    dt: float = ROSSLER_DT, spinup: int = 2000,
    seed: int | None = 0, initial: np.ndarray | None = None,
) -> np.ndarray:
    """RK4 integration. Returns traj of shape (n_steps, 3)."""
    rng = np.random.default_rng(seed)
    if initial is None:
        initial = np.array([0.1, 0.1, 0.1], dtype=np.float64) \
                   + 0.05 * rng.standard_normal(3)
    state = np.asarray(initial, dtype=np.float64).copy()

    def rhs(s):
        x, y, z = s
        return np.array([-y - z, x + a * y, b + z * (x - c)])

    # spin-up
    for _ in range(spinup):
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    out = np.empty((n_steps, 3), dtype=np.float32)
    for t in range(n_steps):
        out[t] = state
        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return out


def rossler_attractor_std(n_steps: int = 50000, seed: int = 0) -> float:
    traj = integrate_rossler(n_steps, spinup=5000, seed=seed)
    return float(traj.std())


if __name__ == "__main__":
    import time
    t0 = time.time()
    traj = integrate_rossler(50000, spinup=5000, seed=0)
    t1 = time.time()
    print(f"Integrated 50k Rössler steps in {t1 - t0:.2f}s")
    print(f"Shape: {traj.shape}")
    print(f"Ranges: x∈[{traj[:,0].min():.2f},{traj[:,0].max():.2f}]  "
          f"y∈[{traj[:,1].min():.2f},{traj[:,1].max():.2f}]  "
          f"z∈[{traj[:,2].min():.2f},{traj[:,2].max():.2f}]")
    print(f"Per-dim std: x={traj[:,0].std():.2f}  y={traj[:,1].std():.2f}  z={traj[:,2].std():.2f}")
    print(f"Mean (all-dim) std = {traj.std():.3f}  (constant set at {ROSSLER_ATTRACTOR_STD})")
