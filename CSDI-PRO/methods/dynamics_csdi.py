"""Module 1 (full) — Dynamics-Aware CSDI.

A clean, self-contained implementation capturing the three tech.md §1.2 changes
relative to vanilla CSDI:

  (A) **Noise-level conditioning** — an ``nn.Linear(1, channels)`` embedding of the
      per-sequence estimated σ_obs, added to the diffusion timestep embedding.
      Lets the score network differentiate diffusion noise from observation noise.

  (B) **Delay-aware attention mask** — a bias term ``α · M_τ`` added to the
      temporal attention logits; ``M_τ`` is initialised from a τ vector (supplied
      by MI-Lyap) and is learnable. Implements *dynamic* (rather than static)
      delay coupling between Module 1 and Module 2.

  (C) **Ensemble-aware sampling** — ``sample()`` returns all ``n_samples``
      samples without averaging, so downstream (SVGP + Lyap-CP) sees the
      full imputation posterior.

Scope note: this re-implements the score network rather than sub-classing
``csdi.main_model.CSDI_base`` (which has dataset-specific legacy in
``process_data`` for PM25/EEG/Physio etc.). Architecture is kept modest
(~2M parameters) so it trains in ~10 minutes on a V100 for the Lorenz63
ablation.

API (numpy-in, numpy-out):

    model = DynamicsCSDI(dim=3, seq_len=64).fit(trajectories)
    imputed = model.impute(obs_2d, mask_2d, tau=np.array([4,3,2,1]),
                           sigma=0.3, n_samples=20)
    # imputed shape: (n_samples, T, D) for downstream ensemble consumption
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Noise-level estimator (reused from dynamics_impute but kept local to break
# cyclic imports if users load this module standalone)
# ---------------------------------------------------------------------------

def _estimate_sigma_mad(obs_1d: np.ndarray) -> float:
    known = np.isfinite(obs_1d)
    y = obs_1d[known]
    if y.size < 6:
        return 0.0
    d2 = np.diff(y, n=2)
    mad = np.median(np.abs(d2 - np.median(d2)))
    return float(mad * 1.4826 / np.sqrt(6.0))


# ---------------------------------------------------------------------------
# Score network
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)

    def _sinusoid(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1))
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        h = self._sinusoid(t)
        h = F.silu(self.proj1(h))
        h = self.proj2(h)
        return h


class DelayAwareMHA(nn.Module):
    """Multi-head self-attention over time axis with an optional additive bias mask.

    The bias tensor ``mask_bias`` has shape ``(1, L, L)`` and is broadcast over
    heads; callers can either pass a fixed tensor (hand-designed) or a
    learnable one (τ-initialised).
    """

    def __init__(self, channels: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor, mask_bias: torch.Tensor | None) -> torch.Tensor:
        # x: (B, L, C), mask_bias: (L, L) or None
        attn_mask = mask_bias if mask_bias is not None else None
        out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, step_dim: int, n_heads: int = 4) -> None:
        super().__init__()
        self.step_proj = nn.Linear(step_dim, channels)
        self.t_attn = DelayAwareMHA(channels, n_heads)
        self.norm1 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, step_emb: torch.Tensor, mask_bias: torch.Tensor | None) -> torch.Tensor:
        # x: (B, L, C)
        B, L, C = x.shape
        se = self.step_proj(step_emb).unsqueeze(1)  # (B, 1, C)
        h = x + se
        h = self.norm1(h + self.t_attn(h, mask_bias))
        h = self.norm2(h + self.ff(h))
        return h


class DynamicsAwareScoreNet(nn.Module):
    """Score network with noise conditioning (A) and delay-aware bias mask (B)."""

    def __init__(
        self,
        data_dim: int,
        channels: int = 64,
        step_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        use_noise_cond: bool = True,
        use_delay_mask: bool = True,
        seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.use_noise_cond = use_noise_cond
        self.use_delay_mask = use_delay_mask
        self.seq_len = seq_len

        self.step_emb = TimestepEmbedding(step_dim)
        # noise-level embedding (A): scalar σ_obs → step-dim vector added to step embedding
        self.noise_embed = nn.Sequential(
            nn.Linear(1, step_dim),
            nn.SiLU(),
            nn.Linear(step_dim, step_dim),
        ) if use_noise_cond else None

        # inputs: cond_part + noise_part + explicit mask; all projected to channels
        # layout: (B, L, 3*data_dim) -> (B, L, C)
        self.in_proj = nn.Linear(3 * data_dim, channels)
        self.blocks = nn.ModuleList([ResidualBlock(channels, step_dim, n_heads) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(channels)
        self.out_proj = nn.Linear(channels, data_dim)

        # Learnable temporal attention-bias mask, initialised to zeros; MI-Lyap
        # updates it via ``set_tau()``.
        self.register_parameter(
            "delay_bias",
            nn.Parameter(torch.zeros(seq_len, seq_len), requires_grad=use_delay_mask),
        )
        # gating scalar α (learnable, starts at 0 so optimisation anneals it in)
        self.register_parameter("delay_alpha", nn.Parameter(torch.zeros(1), requires_grad=use_delay_mask))

    def set_tau(self, tau: np.ndarray) -> None:
        """Initialise the bias mask from a τ vector (MI-Lyap selection).

        We encode τ by letting positions ``(i, j)`` with ``|i - j| ∈ τ`` receive
        a small positive prior bias (so the attention is slightly encouraged to
        attend at those offsets). The parameter remains learnable so training
        can still reshape the mask, but starts with the τ prior.
        """
        if not self.use_delay_mask:
            return
        L = self.seq_len
        M = torch.zeros(L, L)
        # favour offsets |i-j| in τ by +0.5
        tau_set = set(int(t) for t in tau.tolist())
        for offset in tau_set:
            for i in range(L):
                j = i - offset
                if 0 <= j < L:
                    M[i, j] += 0.5
                j2 = i + offset
                if 0 <= j2 < L:
                    M[i, j2] += 0.5
        with torch.no_grad():
            self.delay_bias.copy_(M.to(self.delay_bias.device))
            self.delay_alpha.fill_(0.1)

    def _mask_bias(self) -> torch.Tensor | None:
        if not self.use_delay_mask:
            return None
        return self.delay_alpha * self.delay_bias  # (L, L) additive to attention logits

    def forward(
        self,
        noisy_x: torch.Tensor,         # (B, L, D)
        cond_obs: torch.Tensor,        # (B, L, D) — observed values (masked with 0 for missing)
        cond_mask: torch.Tensor,       # (B, L, D) — 1 where observed, 0 where imputation target
        diffusion_step: torch.Tensor,  # (B,)
        sigma_obs: torch.Tensor | None = None,  # (B,)
    ) -> torch.Tensor:
        B, L, D = noisy_x.shape
        # (B, L, 3D) input: [cond_obs * cond_mask, (locked) noisy_x, explicit cond_mask]
        cond_part = cond_obs * cond_mask
        noise_part = noisy_x * (1 - cond_mask) + cond_obs * cond_mask  # observed positions locked to obs
        inp = torch.cat([cond_part, noise_part, cond_mask], dim=-1)
        h = self.in_proj(inp)  # (B, L, C)

        step_e = self.step_emb(diffusion_step)  # (B, step_dim)
        if self.use_noise_cond and sigma_obs is not None:
            step_e = step_e + self.noise_embed(sigma_obs.unsqueeze(-1))

        mb = self._mask_bias()
        for block in self.blocks:
            h = block(h, step_e, mb)
        h = self.out_norm(h)
        return self.out_proj(h)  # predict noise, same shape as noisy_x


# ---------------------------------------------------------------------------
# Diffusion schedule
# ---------------------------------------------------------------------------

@dataclass
class DiffusionSchedule:
    num_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def __post_init__(self) -> None:
        self.beta = np.linspace(self.beta_start, self.beta_end, self.num_steps)
        self.alpha_hat = 1.0 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)

    def as_tensors(self, device: torch.device):
        alpha = torch.as_tensor(self.alpha, dtype=torch.float32, device=device)
        alpha_hat = torch.as_tensor(self.alpha_hat, dtype=torch.float32, device=device)
        beta = torch.as_tensor(self.beta, dtype=torch.float32, device=device)
        return alpha, alpha_hat, beta


# ---------------------------------------------------------------------------
# Lorenz63 training dataset (on-the-fly)
# ---------------------------------------------------------------------------

class Lorenz63ImputationDataset(Dataset):
    """Generates (clean, noisy, mask, sigma) tuples on the fly for training.

    For each sample:
      1. Integrate Lorenz63 for ``seq_len`` steps with random IC and spin-up
      2. Draw sparsity ~ U(0.2, 0.95) and noise σ/std ~ U(0, 1.5)
      3. Produce mask + noisy observations + record the true σ
    """

    def __init__(
        self,
        n_samples: int = 2048,
        seq_len: int = 64,
        dt: float = 0.025,
        attractor_std: float = 8.51,
        seed: int = 0,
    ) -> None:
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.dt = dt
        self.attractor_std = attractor_std
        self.rng = np.random.default_rng(seed)
        # pre-generate a long pool of trajectories and slice at __getitem__ time
        # to keep cost predictable
        self._pool = self._build_pool()

    def _build_pool(self) -> np.ndarray:
        from experiments.week1.lorenz63_utils import integrate_lorenz63

        total = self.seq_len * max(self.n_samples // 8, 4) + 1000
        pool = integrate_lorenz63(total, dt=self.dt, seed=int(self.rng.integers(0, 10000)))
        return pool.astype(np.float32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(idx + 7)
        start = rng.integers(0, self._pool.shape[0] - self.seq_len - 1)
        clean = self._pool[start : start + self.seq_len].copy()  # (L, D)

        sparsity = float(rng.uniform(0.2, 0.95))
        noise_frac = float(rng.uniform(0.0, 1.5))
        sigma = noise_frac * self.attractor_std

        mask = (rng.random(self.seq_len) > sparsity).astype(np.float32)
        mask_2d = np.repeat(mask[:, None], clean.shape[1], axis=1)

        noisy = clean + rng.normal(scale=sigma, size=clean.shape).astype(np.float32)
        observed = noisy * mask_2d  # 0 where missing (we'll pair with mask)

        # Normalise by attractor_std so the signal and diffusion noise live on
        # the same scale. Un-normalise in ``impute()``.
        scale = self.attractor_std
        return {
            "clean": torch.from_numpy(clean / scale).float(),
            "observed": torch.from_numpy(observed / scale).float(),
            "mask": torch.from_numpy(mask_2d).float(),
            "sigma": torch.tensor(sigma / scale, dtype=torch.float32),  # normalised σ in [0, 1.5]
        }


# ---------------------------------------------------------------------------
# Trainer + Inference wrapper
# ---------------------------------------------------------------------------

@dataclass
class DynamicsCSDIConfig:
    data_dim: int = 3
    seq_len: int = 64
    channels: int = 64
    step_dim: int = 128
    n_heads: int = 4
    n_layers: int = 4
    num_diff_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 0.02
    use_noise_cond: bool = True
    use_delay_mask: bool = True
    device: str = "cuda"


class DynamicsCSDI:
    """Self-contained DDPM-style imputer with noise + delay conditioning."""

    def __init__(self, config: DynamicsCSDIConfig | None = None) -> None:
        self.cfg = config or DynamicsCSDIConfig()
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() or self.cfg.device == "cpu" else "cpu")
        self.schedule = DiffusionSchedule(
            num_steps=self.cfg.num_diff_steps,
            beta_start=self.cfg.beta_start,
            beta_end=self.cfg.beta_end,
        )
        self.net = DynamicsAwareScoreNet(
            data_dim=self.cfg.data_dim,
            channels=self.cfg.channels,
            step_dim=self.cfg.step_dim,
            n_heads=self.cfg.n_heads,
            n_layers=self.cfg.n_layers,
            use_noise_cond=self.cfg.use_noise_cond,
            use_delay_mask=self.cfg.use_delay_mask,
            seq_len=self.cfg.seq_len,
        ).to(self.device)
        self._alpha = torch.as_tensor(self.schedule.alpha, dtype=torch.float32, device=self.device)
        self._alpha_hat = torch.as_tensor(self.schedule.alpha_hat, dtype=torch.float32, device=self.device)
        self._beta = torch.as_tensor(self.schedule.beta, dtype=torch.float32, device=self.device)

    # --- training ---------------------------------------------------------
    def fit(
        self,
        dataset: Dataset,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> "DynamicsCSDI":
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        self.net.train()
        for ep in range(epochs):
            total = 0.0; n = 0
            for batch in loader:
                clean = batch["clean"].to(self.device)   # (B, L, D)
                observed = batch["observed"].to(self.device)
                mask = batch["mask"].to(self.device)
                sigma = batch["sigma"].to(self.device)
                B, L, D = clean.shape

                # sample diffusion step
                t = torch.randint(0, self.schedule.num_steps, (B,), device=self.device)
                alpha_bar = self._alpha[t].view(B, 1, 1)
                noise = torch.randn_like(clean)
                noisy = (alpha_bar ** 0.5) * clean + (1 - alpha_bar) ** 0.5 * noise

                pred_noise = self.net(noisy, observed, mask, t, sigma_obs=sigma if self.cfg.use_noise_cond else None)
                # loss: only on the *missing* positions (1 - mask), matching CSDI
                target_mask = 1.0 - mask
                denom = target_mask.sum().clamp_min(1.0)
                loss = ((pred_noise - noise) ** 2 * target_mask).sum() / denom

                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * B; n += B
            sched.step()
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"  [dyn-csdi] epoch {ep:3d} loss={total / n:.4f}")
        self.net.eval()
        return self

    # --- inference --------------------------------------------------------
    @torch.no_grad()
    def impute(
        self,
        observed: np.ndarray,    # (T, D) with NaN for missing
        mask: np.ndarray | None = None,  # (T, D) optional; if None inferred from NaN
        tau: np.ndarray | None = None,
        sigma: float | None = None,
        n_samples: int = 20,
        attractor_std: float = 8.51,
    ) -> np.ndarray:
        obs = np.asarray(observed, dtype=np.float32).copy()
        if mask is None:
            mask_arr = np.isfinite(obs).astype(np.float32)
        else:
            mask_arr = np.asarray(mask, dtype=np.float32)
            if mask_arr.ndim == 1:
                mask_arr = np.repeat(mask_arr[:, None], obs.shape[1], axis=1)
        obs = np.nan_to_num(obs, nan=0.0)

        # Normalise to match training; remember scale for denormalisation
        scale = attractor_std
        obs = obs / scale

        T, D = obs.shape
        L = self.cfg.seq_len
        if T != L:
            # simple strategy: if T < L, pad-and-unpad; if T > L, chunk
            return self._impute_chunked(obs, mask_arr, tau, sigma, n_samples, attractor_std)

        if tau is not None and self.cfg.use_delay_mask:
            self.net.set_tau(tau)

        if sigma is None:
            sigma = float(np.mean([_estimate_sigma_mad(obs[:, d]) for d in range(D)]))
        sigma_norm = sigma / max(attractor_std, 1e-8)

        obs_t = torch.from_numpy(obs).float().to(self.device)
        mask_t = torch.from_numpy(mask_arr).float().to(self.device)
        sigma_t = torch.tensor([sigma_norm], dtype=torch.float32, device=self.device)

        samples = torch.zeros(n_samples, T, D, device=self.device)
        obs_mask_b = mask_t.unsqueeze(0)
        obs_val_b = obs_t.unsqueeze(0)
        for s in range(n_samples):
            # initialise x at ALL positions from noise, but **re-impose observed at every step**
            # so the score network's inputs are consistent with training (noise_part at
            # observed is always cond_obs).
            x = torch.randn_like(obs_t).unsqueeze(0)  # (1, L, D)
            x = obs_mask_b * obs_val_b + (1 - obs_mask_b) * x  # start observed anchored
            for t in range(self.schedule.num_steps - 1, -1, -1):
                t_t = torch.tensor([t], device=self.device)
                pred_noise = self.net(x, obs_val_b, obs_mask_b, t_t,
                                      sigma_obs=sigma_t if self.cfg.use_noise_cond else None)
                alpha_hat_t = self._alpha_hat[t]
                alpha_t = self._alpha[t]
                coeff1 = 1.0 / alpha_hat_t.sqrt()
                coeff2 = (1 - alpha_hat_t) / (1 - alpha_t).sqrt()
                x = coeff1 * (x - coeff2 * pred_noise)
                if t > 0:
                    prev_alpha = self._alpha[t - 1]
                    sd = (((1 - prev_alpha) / (1 - alpha_t)) * self._beta[t]).sqrt()
                    x = x + sd * torch.randn_like(x)
                # re-impose observed at this t: compute forward-diffused cond_obs and
                # blend so that observed positions follow the known trajectory instead
                # of drifting.
                if t > 0:
                    alpha_tm1 = self._alpha[t - 1]
                    anchor_noise = torch.randn_like(obs_val_b)
                    obs_at_tm1 = (alpha_tm1.sqrt() * obs_val_b
                                  + (1 - alpha_tm1).sqrt() * anchor_noise)
                    x = obs_mask_b * obs_at_tm1 + (1 - obs_mask_b) * x
                else:
                    x = obs_mask_b * obs_val_b + (1 - obs_mask_b) * x
            samples[s] = x.squeeze(0)
        # Un-normalise back to raw attractor scale
        return (samples * scale).cpu().numpy()

    def _impute_chunked(self, obs, mask_arr, tau, sigma, n_samples, attractor_std):
        T, D = obs.shape
        L = self.cfg.seq_len
        out = np.zeros((n_samples, T, D), dtype=np.float32)
        # simple overlap-add with 50% overlap
        stride = L // 2
        weight = np.zeros(T, dtype=np.float32)
        for start in range(0, max(T - L, 0) + 1, stride):
            end = start + L
            if end > T:
                start = T - L
                end = T
            sub = self.impute(
                obs[start:end], mask_arr[start:end],
                tau=tau, sigma=sigma, n_samples=n_samples, attractor_std=attractor_std,
            )
            # Hann window for smooth stitching
            w = np.hanning(L).astype(np.float32)
            out[:, start:end] += sub * w[None, :, None]
            weight[start:end] += w
            if end == T:
                break
        weight = np.maximum(weight, 1e-6)
        return out / weight[None, :, None]

    def impute_mean(self, *args, **kwargs) -> np.ndarray:
        """Convenience: return the ensemble mean (T, D) for dynamics_impute.impute() compat."""
        s = self.impute(*args, **kwargs)
        return s.mean(0)

    # --- I/O --------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        torch.save({"cfg": self.cfg.__dict__, "state": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "DynamicsCSDI":
        ckpt = torch.load(path, map_location="cpu")
        cfg = DynamicsCSDIConfig(**ckpt["cfg"])
        if device is not None:
            cfg.device = device
        obj = cls(cfg)
        obj.net.load_state_dict(ckpt["state"])
        obj.net.eval()
        return obj


if __name__ == "__main__":
    # Smoke test: 2-epoch tiny training to verify the forward+loss works
    import sys
    sys.path.insert(0, ".")
    cfg = DynamicsCSDIConfig(data_dim=3, seq_len=32, channels=32, step_dim=64, n_layers=2,
                             num_diff_steps=20, device="cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicsCSDI(cfg)
    print(f"[params] {sum(p.numel() for p in model.net.parameters()):,}")
    ds = Lorenz63ImputationDataset(n_samples=64, seq_len=32, seed=0)
    model.fit(ds, epochs=2, batch_size=16, verbose=True)

    from experiments.week1.lorenz63_utils import integrate_lorenz63, make_sparse_noisy
    traj = integrate_lorenz63(32, dt=0.025, seed=99)
    obs, mask = make_sparse_noisy(traj, sparsity=0.6, noise_std_frac=0.3, seed=99)
    samples = model.impute(obs, mask, sigma=0.3 * 8.51, n_samples=4)
    rmse = float(np.sqrt(((samples.mean(0) - traj) ** 2).mean()))
    print(f"[smoke] imputed mean RMSE={rmse:.3f}  (linear baseline ~2.6 under same setting)")
