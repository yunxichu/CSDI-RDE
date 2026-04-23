"""Pre-generate Lorenz96 N=20 F=8 imputation training data with independent ICs.

Output: one big npz at ``experiments/week2_modules/data/lorenz96_clean_{tag}.npz``
with array ``clean`` of shape [n_samples, seq_len, N] in raw (un-normalised)
coordinates. Training code applies sparsity/noise/normalisation on the fly.

Run:
    python -m experiments.week2_modules.make_lorenz96_dataset \\
        --n_samples 256000 --seq_len 128 --N 20 --tag 256k_L128_N20 --n_workers 16
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from multiprocessing import Pool

from experiments.week1.lorenz96_utils import (
    integrate_lorenz96,
    lorenz96_attractor_std,
    LORENZ96_F_DEFAULT,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "experiments" / "week2_modules" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _gen_one(args):
    idx, seq_len, N, F, dt, spinup = args
    traj = integrate_lorenz96(seq_len, N=N, F=F, dt=dt, seed=idx, spinup=spinup)
    return traj.astype(np.float32)


def generate(n_samples: int, seq_len: int, N: int, F: float,
             dt: float = 0.05, spinup: int = 1500,
             n_workers: int = 16, progress_every: int = 5000) -> np.ndarray:
    """Independent integrator runs with streaming progress output.

    Uses imap_unordered to stream results as workers finish; writes directly into
    a pre-allocated array to avoid pickle-stacking overhead at the end.
    """
    import sys
    tasks = [(i, seq_len, N, F, dt, spinup) for i in range(n_samples)]
    out = np.empty((n_samples, seq_len, N), dtype=np.float32)
    t0 = time.time()
    done = 0
    chunksize = max(1, n_samples // (n_workers * 32))
    print(f"[gen] launching {n_workers} workers on {n_samples} tasks (chunksize={chunksize})",
          flush=True)
    with Pool(n_workers) as pool:
        # imap with enumerate to pair task indices with results
        for i, traj in enumerate(pool.imap(_gen_one, tasks, chunksize=chunksize)):
            out[i] = traj
            done += 1
            if done % progress_every == 0 or done == n_samples:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (n_samples - done) / max(rate, 1e-6)
                pct = 100.0 * done / n_samples
                print(f"[gen] {done:>7d}/{n_samples} ({pct:5.1f}%) "
                      f"elapsed={elapsed:6.0f}s  rate={rate:6.1f}/s  ETA={eta:6.0f}s",
                      flush=True)
                sys.stdout.flush()
    elapsed = time.time() - t0
    print(f"[gen] DONE. {n_samples} windows × {seq_len} steps × {N} dims, "
          f"{n_workers} workers, {elapsed:.1f} s ({elapsed / n_samples * 1000:.1f} ms/sample)",
          flush=True)
    print(f"[stat] mean={out.mean():.3f} std={out.std():.3f} shape={out.shape}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=256000)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--N", type=int, default=20, help="Lorenz96 ring dim")
    ap.add_argument("--F", type=float, default=LORENZ96_F_DEFAULT)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--spinup", type=int, default=1500)
    ap.add_argument("--n_workers", type=int, default=16)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    tag = args.tag or f"{args.n_samples // 1000}k_L{args.seq_len}_N{args.N}"
    out_path = DATA_DIR / f"lorenz96_clean_{tag}.npz"
    if out_path.exists():
        print(f"[skip] already have {out_path}, size {out_path.stat().st_size // (1024*1024)} MB")
        return

    attr_std = lorenz96_attractor_std(N=args.N, F=args.F)
    print(f"[attractor_std] N={args.N} F={args.F}: {attr_std:.4f}")

    data = generate(args.n_samples, args.seq_len, args.N, args.F,
                    args.dt, args.spinup, args.n_workers)
    np.savez(out_path, clean=data, dt=args.dt, seq_len=args.seq_len,
             N=args.N, F=args.F, attractor_std=attr_std)
    print(f"[saved] {out_path}, size: {out_path.stat().st_size // (1024*1024)} MB")


if __name__ == "__main__":
    main()
