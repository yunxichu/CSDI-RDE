"""Forecasting baselines for the Week 1 pilot.

  - persistence: repeat last observation
  - context_parroting: for chaotic series, simply continue the context's endogenous
      temporal pattern by finding the closest past state and copying forward from it
      (Zhang & Gilpin 2025 call this "tough-to-beat" for clean chaos).
  - chronos: Amazon Chronos-T5 zero-shot forecasting
"""
from __future__ import annotations

import numpy as np


def persistence_forecast(ctx: np.ndarray, pred_len: int) -> np.ndarray:
    return np.tile(ctx[-1:, :], (pred_len, 1))


def context_parroting_forecast(
    ctx: np.ndarray,
    pred_len: int,
    search_window: int | None = None,
) -> np.ndarray:
    """Context parroting (aka nearest-neighbour delay forecast).

    Given dense context ``ctx`` of shape ``(T, D)``, find the past time index ``t*``
    whose state ``ctx[t*]`` is most similar to the present state ``ctx[-1]`` (in L2
    over all D dims), then parrot the forward trajectory ``ctx[t*+1:t*+1+pred_len]``.
    Falls back to linear extrapolation if the context is too short or a suitable
    match cannot advance ``pred_len`` steps.

    This is a strong baseline for clean chaotic systems: it leverages the
    recurrence structure of the attractor without learning.
    """
    T, D = ctx.shape
    present = ctx[-1]
    if search_window is None:
        search_window = T - pred_len - 1

    # candidate match indices = [0, search_window]; must leave pred_len room
    max_t = min(T - pred_len - 1, search_window)
    if max_t < 1:
        return persistence_forecast(ctx, pred_len)

    # compute distances to present
    candidates = ctx[:max_t]  # (max_t, D)
    dists = np.linalg.norm(candidates - present, axis=1)
    # exclude an immediate neighbourhood of "now" to avoid trivial copy
    guard = min(max_t - 1, 20)
    dists[-guard:] = np.inf
    t_star = int(np.argmin(dists))

    seg = ctx[t_star + 1 : t_star + 1 + pred_len]  # (pred_len, D)
    if seg.shape[0] < pred_len:
        pad = np.tile(seg[-1:], (pred_len - seg.shape[0], 1))
        seg = np.concatenate([seg, pad], axis=0)
    # anchor: shift segment so its first step matches the present
    return seg - seg[0] + present


def chronos_forecast(
    pipe,
    ctx: np.ndarray,
    pred_len: int,
    num_samples: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) predictions shaped (pred_len, D).

    Splits the D channels into independent univariate forecasts, consistent with
    how Chronos is typically applied to multivariate chaotic trajectories in the
    literature.
    """
    import torch

    T, D = ctx.shape
    means = []
    stds = []
    for d in range(D):
        series = torch.tensor(ctx[:, d], dtype=torch.float32)
        out = pipe.predict(series.unsqueeze(0), prediction_length=pred_len, num_samples=num_samples)
        samples = out[0].cpu().numpy()  # (num_samples, pred_len)
        means.append(samples.mean(0))
        stds.append(samples.std(0))
    mean = np.stack(means, axis=-1)
    std = np.stack(stds, axis=-1)
    return mean, std
