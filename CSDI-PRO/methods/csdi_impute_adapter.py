"""Adapter so a trained Dynamics-Aware CSDI checkpoint can be used in the same
``impute(observed, kind)``-style slot as the AR-Kalman / linear / cubic surrogates.

Once the CSDI model is trained and saved, this lets ``run_ablation.py`` replace
its M1 stage with the real CSDI by passing ``kind="csdi"`` + a global checkpoint.

Usage::

    from methods.csdi_impute_adapter import set_csdi_checkpoint, csdi_impute
    set_csdi_checkpoint("/path/to/dyn_csdi_full_v3_big.pt")
    filled = csdi_impute(observed)          # observed: (T, D) with NaNs

The checkpoint is cached in module state so a single ``run_ablation`` sweep only
loads it once.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from methods.dynamics_csdi import DynamicsCSDI, DynamicsCSDIConfig
from methods.dynamics_impute import estimate_noise_mad

_GLOBAL_CSDI: Optional[DynamicsCSDI] = None
_GLOBAL_CKPT: Optional[str] = None


def set_csdi_checkpoint(ckpt_path: str | Path, device: str = "cuda") -> None:
    """Load a trained Dynamics-Aware CSDI checkpoint and cache it globally."""
    global _GLOBAL_CSDI, _GLOBAL_CKPT
    ckpt_path = str(ckpt_path)
    if ckpt_path == _GLOBAL_CKPT and _GLOBAL_CSDI is not None:
        return
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # The DynamicsCSDI.save() format stores {"cfg": dict, "state": state_dict}
    cfg_dict = ck.get("cfg") or ck.get("config")
    state = ck.get("state") or ck.get("net")
    cfg = DynamicsCSDIConfig(**cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict
    cfg.device = device
    model = DynamicsCSDI(cfg)
    model.net.load_state_dict(state)
    model.net.to(device).eval()
    _GLOBAL_CSDI = model
    _GLOBAL_CKPT = ckpt_path
    print(f"[csdi-adapter] loaded {ckpt_path}  params={sum(p.numel() for p in model.net.parameters()):,}")


def csdi_impute(observed: np.ndarray, n_samples: int = 8, sigma_override: Optional[float] = None,
                tau_override: Optional[np.ndarray] = None,
                attractor_std: Optional[float] = None) -> np.ndarray:
    """Run trained CSDI on a (T, D) observed window; return posterior mean.

    Input: observed with NaNs at missing steps.
    Output: (T, D) filled array.

    Keeps a fixed seq_len window (the model's config.seq_len) by sliding in
    non-overlapping chunks if observed is longer.

    tau_override (optional, 1-D int array of length L-1): if provided, overrides
    the delay-mask τ anchor at inference time. Used for §5.X1 τ-coupling ablation.
    See paper §3.2 for how τ parameterizes delay attention; the model learns a
    learnable delay_bias/delay_alpha whose *initialization* depends on τ, but at
    inference the bias is re-initialized via set_tau(tau_override). No retraining.
    """
    assert _GLOBAL_CSDI is not None, "call set_csdi_checkpoint() first"
    model = _GLOBAL_CSDI
    seq_len = model.cfg.seq_len
    obs = np.asarray(observed, dtype=np.float32)
    T, D = obs.shape

    # Build mask from NaN pattern before passing to CSDI impute()
    mask = (~np.isnan(obs)).astype(np.float32)
    obs_filled_zero = np.nan_to_num(obs, nan=0.0)

    # σ estimate: MAD on observed second-diff, averaged across channels
    if sigma_override is not None:
        sigma = float(sigma_override)
    else:
        sigmas = [estimate_noise_mad(obs_filled_zero[mask[:, d].astype(bool), d])
                  if mask[:, d].sum() > 4 else 0.0 for d in range(D)]
        sigma = float(np.mean(sigmas))

    # τ override: convert to torch tensor once; DynamicsCSDI.impute() forwards to set_tau()
    # .copy() ensures a C-contiguous array so torch.as_tensor doesn't trip on reversed /
    # strided views (e.g. np.sort(...)[::-1] or slice-views of mi_lyap_bayes_tau output).
    tau_arg = None
    if tau_override is not None:
        tau_arr = np.ascontiguousarray(np.asarray(tau_override, dtype=np.int64))
        tau_arg = torch.as_tensor(tau_arr, dtype=torch.long, device=model.cfg.device)

    # If T > seq_len, process in non-overlapping chunks (last chunk may overlap to fit)
    if T <= seq_len:
        pad = seq_len - T
        pad_obs = np.concatenate([obs_filled_zero, np.zeros((pad, D), dtype=np.float32)], axis=0)
        pad_mask = np.concatenate([mask, np.zeros((pad, D), dtype=np.float32)], axis=0)
        samples = model.impute(pad_obs, pad_mask, sigma=sigma, n_samples=n_samples, tau=tau_arg,
                                attractor_std=attractor_std)
        mu = samples.mean(axis=0)[:T]
        return mu

    # stitch non-overlapping chunks
    out = np.empty_like(obs_filled_zero)
    start = 0
    while start < T:
        end = min(start + seq_len, T)
        chunk_obs = obs_filled_zero[start:end]
        chunk_mask = mask[start:end]
        if chunk_obs.shape[0] < seq_len:
            pad = seq_len - chunk_obs.shape[0]
            chunk_obs = np.concatenate([chunk_obs, np.zeros((pad, D), dtype=np.float32)], axis=0)
            chunk_mask = np.concatenate([chunk_mask, np.zeros((pad, D), dtype=np.float32)], axis=0)
        samples = model.impute(chunk_obs, chunk_mask, sigma=sigma, n_samples=n_samples, tau=tau_arg,
                                attractor_std=attractor_std)
        mu = samples.mean(axis=0)[:end - start]
        out[start:end] = mu
        start = end
    return out


# Convenience: integrate with dynamics_impute.impute() signature style
def impute(observed: np.ndarray, kind: str = "csdi", **kwargs) -> np.ndarray:
    if kind != "csdi":
        from methods.dynamics_impute import impute as base_impute
        return base_impute(observed, kind=kind, **kwargs)
    return csdi_impute(observed, **kwargs)
