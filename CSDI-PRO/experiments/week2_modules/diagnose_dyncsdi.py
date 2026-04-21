"""Diagnose why DynamicsCSDI isn't converging.

Train with constrained noise range, probe intermediate imputations.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from experiments.week1.lorenz63_utils import LORENZ63_ATTRACTOR_STD, integrate_lorenz63, make_sparse_noisy
from methods.dynamics_csdi import DynamicsCSDI, DynamicsCSDIConfig
from methods.dynamics_impute import impute as baseline_impute


class CleanLorenzDS(Dataset):
    """Sparsity only, no observation noise — isolate the imputation ability."""
    def __init__(self, n_samples=2048, seq_len=64, seed=0):
        from experiments.week1.lorenz63_utils import integrate_lorenz63
        self.n_samples = n_samples; self.seq_len = seq_len
        rng = np.random.default_rng(seed)
        self.pool = integrate_lorenz63(seq_len * (n_samples // 8 + 10) + 1000, dt=0.025,
                                        seed=int(rng.integers(0, 10000))).astype(np.float32)
        self.attractor_std = LORENZ63_ATTRACTOR_STD

    def __len__(self): return self.n_samples
    def __getitem__(self, i):
        rng = np.random.default_rng(i + 7)
        start = rng.integers(0, self.pool.shape[0] - self.seq_len - 1)
        clean = self.pool[start:start + self.seq_len].copy() / self.attractor_std
        sparsity = float(rng.uniform(0.2, 0.8))
        mask = (rng.random(self.seq_len) > sparsity).astype(np.float32)
        mask_2d = np.repeat(mask[:, None], clean.shape[1], axis=1)
        return {
            "clean": torch.from_numpy(clean).float(),
            "observed": torch.from_numpy(clean * mask_2d).float(),   # noise-free obs
            "mask": torch.from_numpy(mask_2d).float(),
            "sigma": torch.tensor(0.0, dtype=torch.float32),
        }


def main():
    cfg = DynamicsCSDIConfig(data_dim=3, seq_len=64, channels=64, step_dim=128, n_layers=4,
                             num_diff_steps=50,
                             use_noise_cond=False, use_delay_mask=False,
                             device="cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicsCSDI(cfg)
    ds = CleanLorenzDS(n_samples=2048, seq_len=64, seed=0)
    model.fit(ds, epochs=50, batch_size=32, lr=1e-3, verbose=True)

    # Now test on clean Lorenz63 with sparsity only
    rmses = []; lin_rmses = []
    rng = np.random.default_rng(2027)
    for _ in range(10):
        seed = int(rng.integers(10_000, 100_000))
        traj = integrate_lorenz63(64, dt=0.025, seed=seed)
        sparsity = float(rng.uniform(0.3, 0.7))
        mask = (rng.random(64) > sparsity)
        obs = traj.copy()
        obs[~mask] = np.nan
        samples = model.impute(obs, mask.astype(np.float32), sigma=0.0, n_samples=4)
        mu = samples.mean(0)
        rmses.append(float(np.sqrt(((mu - traj) ** 2).mean())))
        lin = baseline_impute(obs, kind="linear")
        lin_rmses.append(float(np.sqrt(((lin - traj) ** 2).mean())))
    print(f"\n[diag clean]  dyn={np.mean(rmses):.3f} vs linear={np.mean(lin_rmses):.3f}")


if __name__ == "__main__":
    main()
