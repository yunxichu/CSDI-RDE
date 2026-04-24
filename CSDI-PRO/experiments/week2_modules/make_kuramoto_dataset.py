"""Generate Kuramoto N=10 K=1.5 clean-trajectory dataset."""
from __future__ import annotations

import argparse, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from systems.kuramoto import (
    integrate_kuramoto, KURAMOTO_N, KURAMOTO_K, KURAMOTO_DT,
    KURAMOTO_ATTRACTOR_STD,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "experiments" / "week2_modules" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _gen_one(args):
    idx, seq_len, N, K, dt, spinup = args
    return integrate_kuramoto(seq_len, N=N, K=K, dt=dt, seed=idx, spinup=spinup).astype(np.float32)


def generate(n_samples: int, seq_len: int, N: int = KURAMOTO_N, K: float = KURAMOTO_K,
             dt: float = KURAMOTO_DT, spinup: int = 2000,
             n_workers: int = 16, progress_every: int = 2000) -> np.ndarray:
    tasks = [(i, seq_len, N, K, dt, spinup) for i in range(n_samples)]
    out = np.empty((n_samples, seq_len, 2 * N), dtype=np.float32)
    t0 = time.time(); done = 0
    chunksize = max(1, n_samples // (n_workers * 32))
    print(f"[gen] {n_workers} workers on {n_samples} (chunksize={chunksize})", flush=True)
    with Pool(n_workers) as pool:
        for i, traj in enumerate(pool.imap(_gen_one, tasks, chunksize=chunksize)):
            out[i] = traj; done += 1
            if done % progress_every == 0 or done == n_samples:
                el = time.time() - t0; rate = done / el
                eta = (n_samples - done) / rate if rate > 0 else 0
                print(f"[gen] {done}/{n_samples} ({100*done/n_samples:.1f}%) "
                      f"rate={rate:.0f}/s ETA={eta:.0f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=512_000)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--N", type=int, default=KURAMOTO_N)
    ap.add_argument("--K", type=float, default=KURAMOTO_K)
    ap.add_argument("--n_workers", type=int, default=16)
    args = ap.parse_args()
    out_path = DATA_DIR / f"kuramoto_N{args.N}_K{args.K:.1f}_{args.n_samples//1000}k_L{args.seq_len}.npz"
    print(f"[gen] target: {out_path}")
    arr = generate(args.n_samples, args.seq_len, N=args.N, K=args.K, n_workers=args.n_workers)
    print(f"[gen] shape={arr.shape}  std={arr.std():.3f}  "
          f"(target {KURAMOTO_ATTRACTOR_STD})")
    np.savez_compressed(out_path, clean=arr, N=args.N, K=args.K, seq_len=args.seq_len)
    print(f"[gen] saved {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
