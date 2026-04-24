"""Generate Rössler clean-trajectory dataset for CSDI M1 training."""
from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from systems.rossler import (
    integrate_rossler, ROSSLER_ATTRACTOR_STD, ROSSLER_DT,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "experiments" / "week2_modules" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _gen_one(args):
    idx, seq_len, dt, spinup = args
    return integrate_rossler(seq_len, dt=dt, seed=idx, spinup=spinup).astype(np.float32)


def generate(n_samples: int, seq_len: int, dt: float = ROSSLER_DT,
             spinup: int = 2000, n_workers: int = 16,
             progress_every: int = 2000) -> np.ndarray:
    tasks = [(i, seq_len, dt, spinup) for i in range(n_samples)]
    out = np.empty((n_samples, seq_len, 3), dtype=np.float32)
    t0 = time.time(); done = 0
    chunksize = max(1, n_samples // (n_workers * 32))
    print(f"[gen] {n_workers} workers on {n_samples} (chunksize={chunksize})", flush=True)
    with Pool(n_workers) as pool:
        for i, traj in enumerate(pool.imap(_gen_one, tasks, chunksize=chunksize)):
            out[i] = traj; done += 1
            if done % progress_every == 0 or done == n_samples:
                el = time.time() - t0
                rate = done / el
                eta = (n_samples - done) / rate if rate > 0 else 0
                print(f"[gen] {done}/{n_samples} ({100*done/n_samples:.1f}%) "
                      f"rate={rate:.0f}/s elapsed={el:.1f}s ETA={eta:.1f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=512_000)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--dt", type=float, default=ROSSLER_DT)
    ap.add_argument("--n_workers", type=int, default=16)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    out_path = Path(args.out) if args.out else DATA_DIR / (
        f"rossler_{args.n_samples//1000}k_L{args.seq_len}.npz"
    )
    print(f"[gen] target: {out_path}")
    arr = generate(args.n_samples, args.seq_len, dt=args.dt, n_workers=args.n_workers)
    print(f"[gen] shape={arr.shape}  mean={arr.mean():.2f}  std={arr.std():.3f}  "
          f"(target attr_std ≈ {ROSSLER_ATTRACTOR_STD})")
    np.savez_compressed(out_path, clean=arr, dt=args.dt, seq_len=args.seq_len,
                        n_samples=args.n_samples)
    print(f"[gen] saved {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
