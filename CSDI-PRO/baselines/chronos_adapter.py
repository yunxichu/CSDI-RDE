"""Chronos forecasting adapter — per-channel univariate prediction.

Chronos is a univariate tokenized forecaster. For multivariate L63 (D=3) we
predict each channel independently, then stack. This is the conventional
deployment for Chronos on multivariate time series.

Usage:
    from baselines.chronos_adapter import chronos_forecast
    mean = chronos_forecast(ctx_filled, pred_len=128)   # (pred_len, D)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


_PIPELINE_CACHE: dict[str, "object"] = {}


def _get_pipeline(model_name: str = "amazon/chronos-bolt-small",
                   device: str = "cuda"):
    key = f"{model_name}|{device}"
    pipe = _PIPELINE_CACHE.get(key)
    if pipe is None:
        from chronos import BaseChronosPipeline
        pipe = BaseChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        _PIPELINE_CACHE[key] = pipe
    return pipe


def chronos_forecast(
    ctx_filled: np.ndarray,
    pred_len: int,
    model_name: str = "amazon/chronos-bolt-small",
    device: str = "cuda",
    num_samples: int | None = 20,
) -> np.ndarray:
    """Median multivariate forecast from Chronos by per-channel prediction.

    ``ctx_filled`` shape (T, D), no NaNs. Returns (pred_len, D).
    """
    arr = np.asarray(ctx_filled, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    T, D = arr.shape
    pipe = _get_pipeline(model_name=model_name, device=device)

    # Build a list of D univariate (T,) tensors, predict simultaneously.
    inputs = [torch.tensor(arr[:, d], dtype=torch.float32) for d in range(D)]

    with torch.no_grad():
        # Bolt models: predict returns (n_series, num_quantiles, pred_len) or
        # (n_series, num_samples, pred_len) depending on variant.
        # We take the median over the sample / quantile axis.
        try:
            out = pipe.predict(inputs, prediction_length=pred_len)
        except TypeError:
            out = pipe.predict(inputs, prediction_length=pred_len,
                                num_samples=num_samples)
    out = out.detach().to(torch.float32).cpu().numpy()  # (D, K, pred_len)
    if out.ndim == 3:
        median = np.median(out, axis=1)  # (D, pred_len)
    elif out.ndim == 2:
        median = out  # (D, pred_len)
    else:
        raise RuntimeError(f"unexpected Chronos output shape {out.shape}")
    return median.T.astype(np.float32)  # (pred_len, D)
