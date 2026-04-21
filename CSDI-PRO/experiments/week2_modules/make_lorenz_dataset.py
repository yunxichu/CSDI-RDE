"""Pre-generate Lorenz63 imputation training data with *independent* initial conditions.

Previous training failed because all 2048 windows were slices of a single 17k-step
trajectory — so initial-condition diversity was tiny and CSDI couldn't learn a
general imputer. Here we generate 64K windows each from an independent IC, spun
up so every window lives on the attractor.

Output: one big npz at ``experiments/week2_modules/data/lorenz63_clean_{tag}.npz``
with array ``clean`` of shape [n_samples, seq_len, 3] in raw (un-normalised)
coordinates. Training code applies sparsity/noise/normalisation on the fly.

Run:
    python -m experiments.week2_modules.make_lorenz_dataset \
        --n_samples 64000 --seq_len 128 --tag 64k_L128
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from multiprocessing import Pool

from experiments.week1.lorenz63_utils import integrate_lorenz63, LORENZ63_ATTRACTOR_STD

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "experiments" / "week2_modules" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _gen_one(args):
    idx, seq_len, dt, spinup = args
    traj = integrate_lorenz63(seq_len, dt=dt, seed=idx, spinup=spinup)
    return traj.astype(np.float32)


def generate(n_samples: int, seq_len: int, dt: float = 0.025, spinup: int = 1500,
             n_workers: int = 16) -> np.ndarray:
    """Independent integrator runs; each seed gives a distinct IC + spin-up."""
    tasks = [(i, seq_len, dt, spinup) for i in range(n_samples)]
    t0 = time.time()
    with Pool(n_workers) as pool:
        trajs = pool.map(_gen_one, tasks, chunksize=max(1, n_samples // (n_workers * 8)))
    elapsed = time.time() - t0
    out = np.stack(trajs, axis=0).astype(np.float32)
    print(f"[gen] {n_samples} windows × {seq_len} steps, {n_workers} workers, "
          f"{elapsed:.1f} s ({elapsed / n_samples * 1000:.1f} ms/sample)")
    print(f"[stat] mean={out.mean():.3f} std={out.std():.3f} shape={out.shape}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=64000)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--dt", type=float, default=0.025)
    ap.add_argument("--spinup", type=int, default=1500)
    ap.add_argument("--n_workers", type=int, default=16)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or f"{args.n_samples // 1000}k_L{args.seq_len}"
    out_path = DATA_DIR / f"lorenz63_clean_{tag}.npz"
    if out_path.exists():
        print(f"[skip] already have {out_path}, size {out_path.stat().st_size // (1024*1024)} MB")
        return

    data = generate(args.n_samples, args.seq_len, args.dt, args.spinup, args.n_workers)
    np.savez(out_path, clean=data, dt=args.dt, seq_len=args.seq_len,
             attractor_std=LORENZ63_ATTRACTOR_STD)
    print(f"[saved] {out_path}, size: {out_path.stat().st_size // (1024*1024)} MB")


if __name__ == "__main__":
    main()
