"""Module 3 alternative — 1-D Fourier Neural Operator on delay coordinates.

Treats each anchor's delay embedding as a 1-D "function of lag":
    f(ℓ) = state(t − τ_ℓ) ∈ R^D   for  ℓ = 0, 1, …, L

Spectral convolutions along the lag dimension → multi-lag frequency-space
mixing, which is a natural parameterisation for delay-dynamics: the Fourier
coefficients along lag capture how evidence at different τ-scales blends.

API matches :class:`models.svgp.MultiOutputSVGP` so the pipeline can swap
backbones via a flag.

References
----------
- Li et al., "Fourier Neural Operator for Parametric PDEs", ICLR 2021
- Liu et al., "Modeling high-dimensional time-delay chaotic system based on
  FNO", Chaos Solitons & Fractals 2024 (motivates delay-domain FNO for chaos)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FNOConfig:
    width: int = 32          # channel width inside FNO
    modes: int = 4           # Fourier modes kept per spectral conv (must ≤ L_total/2+1)
    n_layers: int = 3
    mlp_hidden: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 300
    batch_size: int = 512
    patience: int = 30
    device: str | None = None
    verbose: bool = False


class _SpectralConv1d(nn.Module):
    """Fourier-domain multiplication with learnable complex weights (Li 2021)."""

    def __init__(self, c_in: int, c_out: int, modes: int) -> None:
        super().__init__()
        self.c_in, self.c_out, self.modes = c_in, c_out, modes
        scale = 1.0 / (c_in * c_out)
        self.w_real = nn.Parameter(scale * torch.randn(c_in, c_out, modes))
        self.w_imag = nn.Parameter(scale * torch.randn(c_in, c_out, modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, c_in, L)
        B, _, L = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, c_in, L//2+1)
        m = min(self.modes, x_ft.size(-1))
        w = torch.complex(self.w_real[:, :, :m], self.w_imag[:, :, :m])
        x_slice = x_ft[:, :, :m]  # (B, c_in, m)
        # einsum: b i m, i o m -> b o m
        out_ft_slice = torch.einsum("bim,iom->bom", x_slice, w)
        out_ft = torch.zeros(B, self.c_out, x_ft.size(-1), dtype=x_ft.dtype, device=x.device)
        out_ft[:, :, :m] = out_ft_slice
        return torch.fft.irfft(out_ft, n=L, dim=-1)


class _FNOBlock(nn.Module):
    """SpectralConv1d + pointwise (1x1) + residual + GELU."""

    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spec = _SpectralConv1d(width, width, modes)
        self.ptw = nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spec(x) + self.ptw(x))


class _FNO1dNet(nn.Module):
    def __init__(self, D: int, L_total: int, cfg: FNOConfig) -> None:
        super().__init__()
        self.D = D
        self.L_total = L_total
        # lift: D → width (pointwise on each lag position)
        self.lift = nn.Linear(D, cfg.width)
        modes = min(cfg.modes, L_total // 2 + 1)
        self.blocks = nn.ModuleList([_FNOBlock(cfg.width, modes) for _ in range(cfg.n_layers)])
        self.head = nn.Sequential(
            nn.Linear(cfg.width, cfg.mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.mlp_hidden, D),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L_total, D)
        h = self.lift(x)                # (B, L_total, width)
        h = h.permute(0, 2, 1)          # (B, width, L_total)
        for blk in self.blocks:
            h = blk(h)
        h = h.permute(0, 2, 1)          # (B, L_total, width)
        # pool: current-time position (index 0) → predict next state
        return self.head(h[:, 0, :])


class FNOPredictor:
    """Drop-in replacement for MultiOutputSVGP (same .fit/.predict contract)."""

    def __init__(self, config: FNOConfig | None = None) -> None:
        self.cfg = config or FNOConfig()
        self.device = torch.device(self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net: _FNO1dNet | None = None
        self._D: int | None = None
        self._L_total: int | None = None
        self._x_mu: np.ndarray | None = None
        self._x_sd: np.ndarray | None = None
        self._y_mu: np.ndarray | None = None
        self._y_sd: np.ndarray | None = None
        self._resid_sd: np.ndarray | None = None

    def _reshape_to_tokens(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        X_cd = X.reshape(N, self._D, self._L_total)           # (N, D, L_total)
        return np.ascontiguousarray(np.transpose(X_cd, (0, 2, 1)))  # (N, L_total, D)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "FNOPredictor":
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim == 1:
            Y = Y[:, None]
        D = Y.shape[1]
        assert X.shape[1] % D == 0
        self._D = D
        self._L_total = X.shape[1] // D

        X_tok = self._reshape_to_tokens(X)
        self._x_mu = X_tok.reshape(-1, self._D).mean(axis=0).astype(np.float32)
        self._x_sd = X_tok.reshape(-1, self._D).std(axis=0).astype(np.float32) + 1e-6
        self._y_mu = Y.mean(axis=0).astype(np.float32)
        self._y_sd = Y.std(axis=0).astype(np.float32) + 1e-6
        X_n = (X_tok - self._x_mu) / self._x_sd
        Y_n = (Y - self._y_mu) / self._y_sd

        self.net = _FNO1dNet(self._D, self._L_total, self.cfg).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        X_t = torch.from_numpy(X_n).to(self.device)
        Y_t = torch.from_numpy(Y_n).to(self.device)
        N = X_t.size(0)
        bs = min(self.cfg.batch_size, N)

        best_loss = float("inf")
        bad = 0
        best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        self.net.train()
        for ep in range(self.cfg.n_epochs):
            perm = torch.randperm(N, device=self.device)
            ep_loss, nb = 0.0, 0
            for i in range(0, N, bs):
                sub = perm[i:i + bs]
                opt.zero_grad()
                loss = F.mse_loss(self.net(X_t[sub]), Y_t[sub])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                ep_loss += float(loss.item()); nb += 1
            ep_loss /= max(nb, 1)
            if ep_loss < best_loss - 1e-5:
                best_loss = ep_loss; bad = 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.cfg.patience:
                    if self.cfg.verbose:
                        print(f"  [fno] early stop ep={ep}  best_loss={best_loss:.5f}")
                    break
            if self.cfg.verbose and (ep % 25 == 0 or ep == self.cfg.n_epochs - 1):
                print(f"  [fno] ep={ep:3d} train_mse={ep_loss:.5f}  best={best_loss:.5f}")
        self.net.load_state_dict(best_state)
        self.net.eval()

        with torch.no_grad():
            pred = self.net(X_t).cpu().numpy()
        self._resid_sd = ((Y_n - pred).std(axis=0) * self._y_sd).astype(np.float32)
        return self

    def predict(self, X: np.ndarray, return_std: bool = True):
        assert self.net is not None, "fit() before predict()"
        X = np.asarray(X, dtype=np.float32)
        X_tok = self._reshape_to_tokens(X)
        X_n = (X_tok - self._x_mu) / self._x_sd
        with torch.no_grad():
            pred_n = self.net(torch.from_numpy(X_n).to(self.device)).cpu().numpy()
        mean = pred_n * self._y_sd + self._y_mu
        if not return_std:
            return mean
        std = np.broadcast_to(self._resid_sd, mean.shape).copy()
        return mean, std


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N, D, L_total = 800, 3, 5
    X = rng.normal(size=(N, D * L_total)).astype(np.float32)
    Y = np.stack([
        np.sin(X[:, 0]) + 0.3 * X[:, 1],
        np.cos(X[:, 5]) + 0.2 * X[:, 10],
        0.5 * X[:, 0] * X[:, 10],
    ], axis=1).astype(np.float32)
    net = FNOPredictor(FNOConfig(n_epochs=100, verbose=True)).fit(X, Y)
    mu, _ = net.predict(X)
    rmse = float(np.sqrt(((mu - Y) ** 2).mean()))
    print(f"FNO train RMSE = {rmse:.4f}")
