"""Train Dynamics-Aware CSDI on Lorenz63 sparse+noisy masking task.

Also trains a **vanilla** CSDI (noise_cond=False, delay_mask=False) for the
ablation: which of (A) noise conditioning, (B) delay-aware mask contribute how much?

Usage:
    CUDA_VISIBLE_DEVICES=2 python -m experiments.week2_modules.train_dynamics_csdi \
        --epochs 30 --n_samples 2048 --variant full
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.week1.lorenz63_utils import (
    LORENZ63_ATTRACTOR_STD,
    integrate_lorenz63,
    make_sparse_noisy,
)
from methods.dynamics_csdi import (
    DynamicsCSDI,
    DynamicsCSDIConfig,
    Lorenz63ImputationDataset,
)
from methods.dynamics_impute import impute as baseline_impute

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = REPO_ROOT / "experiments" / "week2_modules" / "ckpts"
RES_DIR = REPO_ROOT / "experiments" / "week2_modules" / "results"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "full":      dict(use_noise_cond=True,  use_delay_mask=True),   # A + B
    "no_noise":  dict(use_noise_cond=False, use_delay_mask=True),   # B only
    "no_mask":   dict(use_noise_cond=True,  use_delay_mask=False),  # A only
    "vanilla":   dict(use_noise_cond=False, use_delay_mask=False),  # baseline CSDI architecture
}


def evaluate(model: DynamicsCSDI, n_eval: int = 20, seq_len: int = 64, seed: int = 2026,
              system: str = "lorenz63", N_l96: int = 20, F_l96: float = 8.0,
              attractor_std: float = LORENZ63_ATTRACTOR_STD,
              dt: float = 0.025) -> dict:
    """Impute on held-out random (sparsity, noise) samples and compare to linear / AR-Kalman."""
    rng = np.random.default_rng(seed)
    results = []
    for k in range(n_eval):
        seed_k = int(rng.integers(10_000, 100_000))
        if system == "lorenz96":
            from experiments.week1.lorenz96_utils import integrate_lorenz96
            traj = integrate_lorenz96(seq_len, N=N_l96, F=F_l96, dt=dt, seed=seed_k)
        elif system == "mackey_glass":
            from systems.mackey_glass import integrate_mackey_glass
            traj = integrate_mackey_glass(seq_len, dt=dt, seed=seed_k)
        elif system == "rossler":
            from systems.rossler import integrate_rossler
            traj = integrate_rossler(seq_len, dt=dt, seed=seed_k)
        elif system == "kuramoto":
            from systems.kuramoto import integrate_kuramoto
            traj = integrate_kuramoto(seq_len, dt=dt, seed=seed_k)
        elif system == "chua":
            from systems.chua import integrate_chua
            traj = integrate_chua(seq_len, dt=dt, seed=seed_k)
        else:
            traj = integrate_lorenz63(seq_len, dt=dt, seed=seed_k)
        sparsity = float(rng.uniform(0.2, 0.90))
        noise_frac = float(rng.uniform(0.0, 1.2))
        obs, mask = make_sparse_noisy(traj, sparsity=sparsity, noise_std_frac=noise_frac,
                                       attractor_std=attractor_std, seed=seed_k)

        # Dynamics-CSDI
        samples = model.impute(obs, mask, sigma=noise_frac * attractor_std, n_samples=8)
        mu = samples.mean(0)
        rmse_dyn = float(np.sqrt(((mu - traj) ** 2).mean()))

        # baselines
        rmse_lin = float(np.sqrt(((baseline_impute(obs, kind="linear") - traj) ** 2).mean()))
        rmse_kal = float(np.sqrt(((baseline_impute(obs, kind="ar_kalman") - traj) ** 2).mean()))

        results.append(dict(sparsity=sparsity, noise_frac=noise_frac,
                            rmse_dyn=rmse_dyn, rmse_linear=rmse_lin, rmse_kalman=rmse_kal))

    agg = {
        "rmse_dyn_mean":    float(np.mean([r["rmse_dyn"] for r in results])),
        "rmse_linear_mean": float(np.mean([r["rmse_linear"] for r in results])),
        "rmse_kalman_mean": float(np.mean([r["rmse_kalman"] for r in results])),
        "n_eval": n_eval,
        "rmse_dyn_std":     float(np.std([r["rmse_dyn"] for r in results])),
        "per_sample":       results,
    }
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n_samples", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--num_diff_steps", type=int, default=50)
    ap.add_argument("--variant", choices=list(VARIANTS), default="full")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--cache_path", default=None,
                    help="load pre-generated trajectories from this .npz")
    ap.add_argument("--save_every", type=int, default=0,
                    help="save intermediate checkpoint every N epochs (0=disable)")
    ap.add_argument("--seed", type=int, default=0,
                    help="random seed for reproducible initialisation")
    # L96 / generic system support
    ap.add_argument("--data_dim", type=int, default=3,
                    help="dimension of state (L63: 3, L96 N=20: 20)")
    ap.add_argument("--attractor_std", type=float, default=LORENZ63_ATTRACTOR_STD,
                    help="attractor std for (observed, clean) normalization")
    ap.add_argument("--system", choices=["lorenz63", "lorenz96", "mackey_glass",
                                           "rossler", "kuramoto", "chua"],
                    default="lorenz63",
                    help="which system to use for eval-time trajectory generation")
    ap.add_argument("--eval_N", type=int, default=20, help="Lorenz96 N for eval (if system=lorenz96)")
    ap.add_argument("--eval_F", type=float, default=8.0)
    ap.add_argument("--eval_dt", type=float, default=None,
                    help="dt for eval trajectories (default 0.025 L63 / 0.05 L96)")
    ap.add_argument("--early_stop_patience", type=int, default=0,
                    help="stop if training loss doesn't improve for this many epochs (0=disable)")
    args = ap.parse_args()

    if args.eval_dt is None:
        if args.system == "lorenz96":
            args.eval_dt = 0.05
        elif args.system == "mackey_glass":
            args.eval_dt = 1.0
        elif args.system == "rossler":
            args.eval_dt = 0.1
        elif args.system == "kuramoto":
            args.eval_dt = 0.1
        elif args.system == "chua":
            args.eval_dt = 0.02
        else:
            args.eval_dt = 0.025

    import torch as _torch
    _torch.manual_seed(args.seed)

    tag = args.tag or f"{args.variant}_ep{args.epochs}"
    cfg_kwargs = VARIANTS[args.variant]
    cfg = DynamicsCSDIConfig(
        data_dim=args.data_dim,
        seq_len=args.seq_len,
        channels=args.channels,
        step_dim=128,
        n_heads=4,
        n_layers=args.n_layers,
        num_diff_steps=args.num_diff_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **cfg_kwargs,
    )
    print(f"=== Train Dynamics-CSDI variant={args.variant}  {cfg_kwargs}")
    print(f"    seq_len={args.seq_len} channels={args.channels} layers={args.n_layers} "
          f"n_samples={args.n_samples} epochs={args.epochs} lr={args.lr} seed={args.seed}")

    model = DynamicsCSDI(cfg)
    n_params = sum(p.numel() for p in model.net.parameters())
    print(f"[params] {n_params:,}")

    ds = Lorenz63ImputationDataset(
        n_samples=args.n_samples, seq_len=args.seq_len, seed=0,
        cache_path=args.cache_path,
        attractor_std=args.attractor_std,
    )
    print(f"[data] dataset size={len(ds)}  cache={'yes' if args.cache_path else 'no (on-the-fly pool)'}")
    t0 = time.time()
    model.fit(ds, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, verbose=True,
              save_every=args.save_every, ckpt_dir=CKPT_DIR, tag=tag,
              early_stop_patience=args.early_stop_patience)
    train_time = time.time() - t0
    print(f"[train] total {train_time:.1f}s = {train_time/60:.1f} min")

    ckpt_path = CKPT_DIR / f"dyn_csdi_{tag}.pt"
    model.save(ckpt_path)
    print(f"[ckpt] saved to {ckpt_path}")

    print(f"\n[eval] imputation quality on 20 random {args.system} windows…")
    res = evaluate(model, n_eval=20, seq_len=args.seq_len,
                    system=args.system, N_l96=args.eval_N, F_l96=args.eval_F,
                    attractor_std=args.attractor_std, dt=args.eval_dt)
    res["variant"] = args.variant
    res["config"] = cfg.__dict__
    res["train_time_s"] = train_time
    res["n_params"] = n_params
    print(f"  rmse_dyn    = {res['rmse_dyn_mean']:.3f} ± {res['rmse_dyn_std']:.3f}")
    print(f"  rmse_linear = {res['rmse_linear_mean']:.3f}")
    print(f"  rmse_kalman = {res['rmse_kalman_mean']:.3f}")

    out = RES_DIR / f"train_dyn_csdi_{tag}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
