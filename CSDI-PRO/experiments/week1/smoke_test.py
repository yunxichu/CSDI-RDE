"""Week 1 Day 1-2 smoke test.

Checks:
  1. dysts Lorenz63 trajectory generation
  2. GPyTorch SVGP toy regression
  3. nolds Lyapunov estimator on Lorenz63
  4. torch CUDA access

Run from repo root:
  CUDA_VISIBLE_DEVICES=2 python experiments/week1/smoke_test.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch


def smoke_dysts() -> np.ndarray:
    from dysts.flows import Lorenz

    t0 = time.time()
    model = Lorenz()
    traj = model.make_trajectory(2000, resample=True)
    dt = getattr(model, "dt", None)
    print(f"[dysts] Lorenz63 traj shape={traj.shape} dt={dt} took {time.time() - t0:.2f}s")
    assert traj.shape == (2000, 3)
    return np.asarray(traj)


def smoke_svgp() -> None:
    import gpytorch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    X = torch.linspace(-3, 3, 400).unsqueeze(-1)
    y = torch.sin(X.squeeze(-1)) + 0.1 * torch.randn(400)
    X, y = X.to(device), y.to(device)

    inducing = X[::20].clone()

    class SVGP(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points: torch.Tensor) -> None:
            var_dist = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
            var_strat = gpytorch.variational.VariationalStrategy(
                self, inducing_points, var_dist, learn_inducing_locations=True
            )
            super().__init__(var_strat)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

        def forward(self, x: torch.Tensor):
            return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

    model = SVGP(inducing).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.05)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X.size(0))

    model.train(); likelihood.train()
    t0 = time.time()
    for _ in range(200):
        optimizer.zero_grad()
        loss = -mll(model(X), y)
        loss.backward()
        optimizer.step()
    train_t = time.time() - t0

    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        Xtest = torch.linspace(-3, 3, 200, device=device).unsqueeze(-1)
        pred = likelihood(model(Xtest))
        rmse = (pred.mean - torch.sin(Xtest.squeeze(-1))).pow(2).mean().sqrt().item()
    print(f"[SVGP] device={device} train_time={train_t:.2f}s rmse_vs_truth={rmse:.4f}")
    assert rmse < 0.2


def smoke_nolds(traj: np.ndarray) -> None:
    import nolds

    t0 = time.time()
    lam = nolds.lyap_r(traj[:, 0], emb_dim=5, lag=2, min_tsep=10, trajectory_len=20, fit="poly")
    print(f"[nolds] Lorenz63 x-component lyap_r={lam:.3f} (expect ~0.9 in dysts-default time units) took {time.time() - t0:.2f}s")


def smoke_cuda() -> None:
    n = torch.cuda.device_count()
    print(f"[cuda] available={torch.cuda.is_available()} count={n} visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    if n:
        for i in range(n):
            print(f"        gpu[{i}] = {torch.cuda.get_device_name(i)}")


def main() -> None:
    smoke_cuda()
    traj = smoke_dysts()
    smoke_nolds(traj)
    smoke_svgp()
    print("\n[PASS] Week 1 Day 1-2 smoke test OK.")


if __name__ == "__main__":
    main()
