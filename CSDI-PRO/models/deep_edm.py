"""Module 3 alternative — DeepEDM (LETS Forecast, ICML 2025).

Replaces the SVGP-on-delay-coords predictor with a small transformer that
interprets softmax-attention as *learned kernel regression on the delay
manifold*. This preserves the Takens/EDM narrative (M2 τ-search → delay
embedding → kernel-regression next-step predictor) while scaling to high-D
where SVGP's Matérn kernel collapses.

API matches :class:`models.svgp.MultiOutputSVGP` (`.fit(X,Y)/.predict(X)`)
so ``full_pipeline_rollout.py`` can swap backbones by config flag.

Input X layout (from ``full_pipeline_rollout._build_delay_features``):
    channel-major flat vector, per anchor t:
        [x_0(t), x_0(t-τ_1), ..., x_0(t-τ_L),
         x_1(t), x_1(t-τ_1), ..., x_1(t-τ_L),
         ...
         x_{D-1}(t), x_{D-1}(t-τ_1), ..., x_{D-1}(t-τ_L)]
    shape (N, D*(L+1))

We reshape to (N, D, L+1) then permute to (N, L+1, D) so attention runs
over *lag positions* with D-dim tokens at each position — that is, each
lag position is a token and attention mixes past-state evidence across
lags. The final regression head pools the current-time token (lag=0) to
predict next state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class DeepEDMConfig:
    d_model: int = 64          # token embedding dim
    n_heads: int = 4
    n_layers: int = 2
    ff_mult: int = 2           # MLP hidden = d_model * ff_mult
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 200
    batch_size: int = 512
    patience: int = 30         # early-stop on train-loss plateau (no val split here)
    device: str | None = None
    verbose: bool = False


class _AttnBlock(nn.Module):
    """Pre-norm multi-head self-attention + MLP (transformer encoder block).

    Attention(Q, K, V) = softmax(QK^T/√d)V is exactly kernel regression with
    a learned (Gaussian-like) kernel over token embeddings — that's the
    DeepEDM interpretation: tokens at different delay lags attend to each
    other, and the softmax defines the effective similarity kernel.
    """

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class _DeepEDMNet(nn.Module):
    def __init__(self, D: int, L_total: int, cfg: DeepEDMConfig) -> None:
        super().__init__()
        self.D = D
        self.L_total = L_total  # = L+1 (current + L lags)
        self.embed = nn.Linear(D, cfg.d_model)
        # learnable positional encoding over lag positions
        self.pos = nn.Parameter(torch.randn(L_total, cfg.d_model) * 0.02)
        self.blocks = nn.ModuleList([
            _AttnBlock(cfg.d_model, cfg.n_heads, cfg.ff_mult, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_out = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L_total, D)  tokens are lag positions, feature = state vec at that lag
        h = self.embed(x) + self.pos  # (B, L_total, d_model)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_out(h)
        # pool: current-time token (index 0) is the query for next-step prediction
        return self.head(h[:, 0, :])


class DeepEDMPredictor:
    """Drop-in replacement for :class:`models.svgp.MultiOutputSVGP`.

    Usage mirrors SVGP::

        net = DeepEDMPredictor(DeepEDMConfig()).fit(X, Y)  # X:(N, D*(L+1)), Y:(N, D)
        mu, sigma = net.predict(X_new, return_std=True)    # sigma via MC dropout if enabled

    By default this is deterministic and ``sigma`` is returned as a scale-free
    constant (ones × residual std on train). M4 conformal calibration is the
    proper UQ layer in the pipeline, so we don't claim calibrated predictive
    variance here.
    """

    def __init__(self, config: DeepEDMConfig | None = None) -> None:
        self.cfg = config or DeepEDMConfig()
        self.device = torch.device(self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net: _DeepEDMNet | None = None
        self._D: int | None = None
        self._L_total: int | None = None
        self._x_mu: np.ndarray | None = None
        self._x_sd: np.ndarray | None = None
        self._y_mu: np.ndarray | None = None
        self._y_sd: np.ndarray | None = None
        self._resid_sd: np.ndarray | None = None

    def _infer_shape(self, X: np.ndarray, Y: np.ndarray) -> tuple[int, int]:
        D = Y.shape[1]
        feat_dim = X.shape[1]
        assert feat_dim % D == 0, f"X feat_dim={feat_dim} not divisible by D={D}"
        L_total = feat_dim // D
        return D, L_total

    def _reshape_to_tokens(self, X: np.ndarray) -> np.ndarray:
        # X: (N, D*(L+1))  channel-major  →  (N, L+1, D)
        N = X.shape[0]
        # reshape to (N, D, L+1): channel-major → per-channel row of lags
        X_cd = X.reshape(N, self._D, self._L_total)
        # swap to (N, L+1, D): tokens = lag positions, feature dim = D
        return np.ascontiguousarray(np.transpose(X_cd, (0, 2, 1)))

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "DeepEDMPredictor":
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim == 1:
            Y = Y[:, None]
        self._D, self._L_total = self._infer_shape(X, Y)

        # per-channel standardisation on reshaped tokens (broadcast over tokens)
        X_tok = self._reshape_to_tokens(X)  # (N, L_total, D)
        self._x_mu = X_tok.reshape(-1, self._D).mean(axis=0).astype(np.float32)
        self._x_sd = X_tok.reshape(-1, self._D).std(axis=0).astype(np.float32) + 1e-6
        self._y_mu = Y.mean(axis=0).astype(np.float32)
        self._y_sd = Y.std(axis=0).astype(np.float32) + 1e-6

        X_n = (X_tok - self._x_mu) / self._x_sd
        Y_n = (Y - self._y_mu) / self._y_sd

        self.net = _DeepEDMNet(self._D, self._L_total, self.cfg).to(self.device)
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
            ep_loss = 0.0
            nb = 0
            for i in range(0, N, bs):
                sub = perm[i:i + bs]
                opt.zero_grad()
                pred = self.net(X_t[sub])
                loss = F.mse_loss(pred, Y_t[sub])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                ep_loss += float(loss.item()); nb += 1
            ep_loss /= max(nb, 1)
            if ep_loss < best_loss - 1e-5:
                best_loss = ep_loss
                bad = 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.cfg.patience:
                    if self.cfg.verbose:
                        print(f"  [deepedm] early stop ep={ep}  best_loss={best_loss:.5f}")
                    break
            if self.cfg.verbose and (ep % 25 == 0 or ep == self.cfg.n_epochs - 1):
                print(f"  [deepedm] ep={ep:3d} train_mse={ep_loss:.5f}  best={best_loss:.5f}")
        self.net.load_state_dict(best_state)
        self.net.eval()

        # residual std on train (per-output) — used as a scale-free sigma surrogate
        with torch.no_grad():
            pred = self.net(X_t).cpu().numpy()
        resid = (Y_n - pred)
        self._resid_sd = (resid.std(axis=0) * self._y_sd).astype(np.float32)
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
    # quick smoke test: 3-D chaotic-ish surrogate
    rng = np.random.default_rng(0)
    N, D, L_total = 800, 3, 5
    X = rng.normal(size=(N, D * L_total)).astype(np.float32)
    Y = np.stack([
        np.sin(X[:, 0]) + 0.3 * X[:, 1] - 0.1 * X[:, 5],
        np.cos(X[:, 5]) + 0.2 * X[:, 10],
        0.5 * X[:, 0] * X[:, 10],
    ], axis=1).astype(np.float32)
    net = DeepEDMPredictor(DeepEDMConfig(n_epochs=100, verbose=True)).fit(X, Y)
    mu, sd = net.predict(X)
    rmse = float(np.sqrt(((mu - Y) ** 2).mean()))
    print(f"DeepEDM train RMSE = {rmse:.4f}  mean_sigma = {sd.mean():.4f}")
