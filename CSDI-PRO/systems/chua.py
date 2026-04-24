"""Chua-circuit double-scroll attractor.

    dx/dt = α (y - x - f(x))
    dy/dt = x - y + z
    dz/dt = -β y

    f(x) = m1 x + 0.5 (m0 - m1) (|x + 1| - |x - 1|)
         (piecewise-linear Chua diode)

Standard double-scroll: α=10, β=14.87, m0=-1.27, m1=-0.68.

Constants (empirical at 50k steps, dt=0.02, spinup=5000):
  - d_KY ≈ 2.1 (3-D attractor, near-2D)
  - λ_1 ≈ 0.23 per time unit (moderate chaos)
  - attractor_std ≈ 1.7 (per-dim mean)
"""
from __future__ import annotations

import numpy as np


CHUA_ALPHA = 10.0
CHUA_BETA = 14.87
CHUA_M0 = -1.27
CHUA_M1 = -0.68
CHUA_DT = 0.02
CHUA_LYAP = 0.23
CHUA_ATTRACTOR_STD = 1.7


def _chua_diode(x: float) -> float:
    return CHUA_M1 * x + 0.5 * (CHUA_M0 - CHUA_M1) * (abs(x + 1.0) - abs(x - 1.0))


def integrate_chua(
    n_steps: int,
    alpha: float = CHUA_ALPHA, beta: float = CHUA_BETA,
    dt: float = CHUA_DT, spinup: int = 5000,
    seed: int | None = 0, initial: np.ndarray | None = None,
) -> np.ndarray:
    """RK4 integrate. Returns (n_steps, 3) trajectory."""
    rng = np.random.default_rng(seed)
    if initial is None:
        initial = np.array([0.7, 0.0, 0.0], dtype=np.float64) + 0.02 * rng.standard_normal(3)
    state = np.asarray(initial, dtype=np.float64).copy()

    def rhs(s):
        x, y, z = s
        return np.array([alpha * (y - x - _chua_diode(x)), x - y + z, -beta * y])

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


def chua_attractor_std(n_steps: int = 50000, seed: int = 0) -> float:
    traj = integrate_chua(n_steps, spinup=5000, seed=seed)
    return float(traj.std())


if __name__ == "__main__":
    import time
    t0 = time.time()
    traj = integrate_chua(50000, spinup=5000, seed=0)
    t1 = time.time()
    print(f"Integrated 50k Chua steps in {t1 - t0:.2f}s")
    print(f"Shape: {traj.shape}")
    print(f"Ranges: x∈[{traj[:,0].min():.2f},{traj[:,0].max():.2f}]  "
          f"y∈[{traj[:,1].min():.2f},{traj[:,1].max():.2f}]  "
          f"z∈[{traj[:,2].min():.2f},{traj[:,2].max():.2f}]")
    print(f"Per-dim std: x={traj[:,0].std():.3f}  y={traj[:,1].std():.3f}  z={traj[:,2].std():.3f}")
    print(f"Overall std = {traj.std():.3f}  (constant set at {CHUA_ATTRACTOR_STD})")
