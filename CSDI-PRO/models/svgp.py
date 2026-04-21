"""Module 3 — Sparse Variational GP with Matern-5/2 kernel on delay coordinates.

API is numpy-in / numpy-out, intended as a drop-in replacement for
:class:`csdi_pro.gpr.gpr_module.GaussianProcessRegressor` in the
v1 pipeline. Scales to n >> few-thousand via inducing points (default M=128).
"""
from __future__ import annotations

from dataclasses import dataclass

import gpytorch
import numpy as np
import torch


class _SVGPCore(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points: torch.Tensor, nu: float = 2.5) -> None:
        var_dist = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        var_strat = gpytorch.variational.VariationalStrategy(
            self, inducing_points, var_dist, learn_inducing_locations=True
        )
        super().__init__(var_strat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=nu))

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


@dataclass
class SVGPConfig:
    m_inducing: int = 128
    nu: float = 2.5  # Matern smoothness; 2.5 is the default per tech.md Module 3.3
    lr: float = 1e-2
    n_epochs: int = 200
    batch_size: int | None = None  # full-batch if None
    device: str | None = None
    verbose: bool = False


class SVGP:
    """Matern-5/2 SVGP with GPyTorch. Single-output; wrap in a loop for multi-dim y."""

    def __init__(self, config: SVGPConfig | None = None) -> None:
        self.cfg = config or SVGPConfig()
        self.device = torch.device(self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: _SVGPCore | None = None
        self.likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self._x_mu: np.ndarray | None = None
        self._x_sd: np.ndarray | None = None
        self._y_mu: float | None = None
        self._y_sd: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVGP":
        """X: (n, L), y: (n,) — standardises X and y."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim != 1:
            y = y.reshape(-1)
        self._x_mu = X.mean(axis=0)
        self._x_sd = X.std(axis=0) + 1e-8
        self._y_mu = float(y.mean())
        self._y_sd = float(y.std() + 1e-8)
        Xn = (X - self._x_mu) / self._x_sd
        yn = (y - self._y_mu) / self._y_sd

        Xt = torch.from_numpy(Xn).to(self.device)
        yt = torch.from_numpy(yn).to(self.device)

        # inducing: random subsample (could be k-means later)
        m = min(self.cfg.m_inducing, Xt.size(0))
        idx = torch.randperm(Xt.size(0), device=self.device)[:m]
        inducing = Xt[idx].clone()

        self.model = _SVGPCore(inducing, nu=self.cfg.nu).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

        self.model.train(); self.likelihood.train()
        opt = torch.optim.Adam(
            list(self.model.parameters()) + list(self.likelihood.parameters()), lr=self.cfg.lr
        )
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=Xt.size(0))

        bs = self.cfg.batch_size or Xt.size(0)
        for ep in range(self.cfg.n_epochs):
            perm = torch.randperm(Xt.size(0), device=self.device)
            for i in range(0, Xt.size(0), bs):
                sub = perm[i : i + bs]
                opt.zero_grad()
                loss = -mll(self.model(Xt[sub]), yt[sub])
                loss.backward(); opt.step()
            if self.cfg.verbose and (ep % 50 == 0 or ep == self.cfg.n_epochs - 1):
                print(f"  [svgp] ep={ep:3d} elbo_loss={loss.item():.4f}")

        self.model.eval(); self.likelihood.eval()
        return self

    def predict(self, X: np.ndarray, return_std: bool = True) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("SVGP.fit must be called before predict")
        Xn = (np.asarray(X, dtype=np.float32) - self._x_mu) / self._x_sd
        Xt = torch.from_numpy(Xn).to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post = self.likelihood(self.model(Xt))
            mean = post.mean.cpu().numpy()
            std = post.stddev.cpu().numpy()
        mean = mean * self._y_sd + self._y_mu
        std = std * self._y_sd
        return (mean, std) if return_std else mean

    def sample(self, X: np.ndarray, n_samples: int = 20) -> np.ndarray:
        Xn = (np.asarray(X, dtype=np.float32) - self._x_mu) / self._x_sd
        Xt = torch.from_numpy(Xn).to(self.device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            post = self.likelihood(self.model(Xt))
            samples = post.sample(torch.Size([n_samples])).cpu().numpy()
        return samples * self._y_sd + self._y_mu


class MultiOutputSVGP:
    """Independent SVGPs per output dim — per tech.md Module 3.4 choice A."""

    def __init__(self, config: SVGPConfig | None = None) -> None:
        self.cfg = config or SVGPConfig()
        self.gps: list[SVGP] = []

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MultiOutputSVGP":
        """X: (n, L), Y: (n, D)."""
        Y = np.asarray(Y, dtype=np.float32)
        if Y.ndim == 1:
            Y = Y[:, None]
        self.gps = []
        for d in range(Y.shape[1]):
            gp = SVGP(self.cfg).fit(X, Y[:, d])
            self.gps.append(gp)
        return self

    def predict(self, X: np.ndarray, return_std: bool = True):
        means, stds = [], []
        for gp in self.gps:
            m, s = gp.predict(X, return_std=True)
            means.append(m); stds.append(s)
        mean = np.stack(means, axis=-1)
        std = np.stack(stds, axis=-1)
        return (mean, std) if return_std else mean

    def sample(self, X: np.ndarray, n_samples: int = 20) -> np.ndarray:
        per = [gp.sample(X, n_samples) for gp in self.gps]  # list of (S, N)
        return np.stack(per, axis=-1)  # (S, N, D)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, L = 2000, 5
    X = rng.normal(size=(n, L))
    y = np.sin(X[:, 0]) + 0.3 * X[:, 1] + 0.1 * rng.normal(size=n)
    gp = SVGP(SVGPConfig(m_inducing=128, n_epochs=200, verbose=True)).fit(X, y)
    Xtest = rng.normal(size=(200, L))
    ytest = np.sin(Xtest[:, 0]) + 0.3 * Xtest[:, 1]
    mu, sd = gp.predict(Xtest)
    rmse = float(np.sqrt(((mu - ytest) ** 2).mean()))
    print(f"SVGP RMSE = {rmse:.4f}  mean_sigma = {sd.mean():.4f}")
