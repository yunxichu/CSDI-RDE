"""Thin Panda-72M adapter matching the ``(observed, pred_len) -> mean`` baseline API.

Uses the official upstream code from ``/home/rhl/Github/panda-src`` via ``sys.path`` —
this keeps the Panda package out of our install tree (transformers pin conflicts
with Chronos) while letting us call the faithfully-trained attention layers.

**Input contract**: ``observed`` is [T, D] with NaNs at missing timesteps. We
linearly interpolate before feeding (same treatment as chronos / parrot baselines),
so the comparison is apples-to-apples at S2+.

The Panda model is trained on dense clean chaotic trajectories; giving it the
imputed series is the generous-to-Panda setting.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_PANDA_SRC = Path("/home/rhl/Github/panda-src")
if _PANDA_SRC.exists() and str(_PANDA_SRC) not in sys.path:
    sys.path.insert(0, str(_PANDA_SRC))

_PIPE = None


def _get_pipeline(model_dir: str | Path = "/home/rhl/Github/CSDI-PRO/baselines/panda-72M",
                   device: str = "cuda"):
    """Lazy singleton — model is ~300MB, avoid reloading for every seed."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    from panda.patchtst.pipeline import PatchTSTPipeline
    pipe = PatchTSTPipeline.from_pretrained(mode="predict", pretrain_path=str(model_dir))
    pipe.model = pipe.model.to(device).eval()
    _PIPE = pipe
    return _PIPE


def panda_forecast(ctx_filled: np.ndarray, pred_len: int, device: str = "cuda") -> np.ndarray:
    """Return [pred_len, D] median forecast from Panda-72M.

    ``ctx_filled`` is [T, D], no NaNs. T is truncated/padded internally to the
    Panda context_length (512); shorter contexts get the first value replicated
    to the left (Panda's own convention via ``left_pad_and_stack_multivariate``
    is NaN-based, but we already imputed so we just pass a dense tensor).
    """
    pipe = _get_pipeline(device=device)
    ctx = torch.tensor(np.asarray(ctx_filled, dtype=np.float32))
    if ctx.ndim == 1:
        ctx = ctx.unsqueeze(-1)
    # PatchTSTPipeline expects [context_length, num_channels] or batched variants.
    with torch.no_grad():
        pred = pipe.predict(ctx, prediction_length=pred_len,
                            limit_prediction_length=False, verbose=False)
    # pred: [1, num_samples=1, pred_len, num_channels]
    return pred.median(dim=1).values.squeeze(0).cpu().numpy()
