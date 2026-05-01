"""Reusable sparse/noisy observation corruptions for phase-transition experiments.

The legacy helper ``make_sparse_noisy`` only supports iid time-level dropout with
scalar Gaussian noise.  This module keeps that behavior available, but also adds
pattern-aware masks needed for the v2 mechanism grid:

  - iid_time: synchronized full-vector dropout at random timesteps
  - iid_channel: independent dropout per variable/sensor
  - block_time: synchronized contiguous outages
  - periodic_subsample: regular sparse sampling, optionally jittered
  - mnar_curvature: more missingness near high-curvature/high-speed regions

All functions are deterministic given ``seed`` and return metadata that can be
reported in paper tables: keep fraction, expected observations per patch, and
gap lengths in both steps and Lyapunov units.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CorruptionResult:
    observed: np.ndarray
    mask: np.ndarray
    noisy_full: np.ndarray
    metadata: dict[str, Any]


def _as_noise_scale(
    traj: np.ndarray,
    noise_std_frac: float,
    attractor_std: float | np.ndarray | None,
    per_dim_noise: bool,
) -> np.ndarray | float:
    if noise_std_frac == 0:
        return 0.0
    if per_dim_noise:
        if attractor_std is None or np.ndim(attractor_std) == 0:
            scale = np.nanstd(traj, axis=0)
        else:
            scale = np.asarray(attractor_std, dtype=np.float64)
        scale = np.where(scale > 0, scale, 1.0)
        return noise_std_frac * scale
    scalar = float(np.nanmean(np.nanstd(traj, axis=0))) if attractor_std is None else float(np.asarray(attractor_std).mean())
    return noise_std_frac * scalar


def _run_lengths(false_mask_1d: np.ndarray) -> list[int]:
    """Lengths of consecutive True runs in ``false_mask_1d``."""
    runs: list[int] = []
    cur = 0
    for v in np.asarray(false_mask_1d, dtype=bool):
        if v:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def _gap_metadata(mask: np.ndarray, dt: float | None, lyap: float | None,
                  patch_length: int) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim == 1:
        mask = mask[:, None]
    any_obs = mask.any(axis=1)
    all_obs = mask.all(axis=1)
    all_missing_runs = _run_lengths(~any_obs)
    not_full_runs = _run_lengths(~all_obs)

    def step_stats(runs: list[int], prefix: str) -> dict[str, Any]:
        if runs:
            out = {
                f"{prefix}_mean_steps": float(np.mean(runs)),
                f"{prefix}_max_steps": int(np.max(runs)),
                f"{prefix}_n": int(len(runs)),
            }
        else:
            out = {
                f"{prefix}_mean_steps": 0.0,
                f"{prefix}_max_steps": 0,
                f"{prefix}_n": 0,
            }
        if dt is not None and lyap is not None:
            out[f"{prefix}_mean_lyap"] = float(out[f"{prefix}_mean_steps"] * dt * lyap)
            out[f"{prefix}_max_lyap"] = float(out[f"{prefix}_max_steps"] * dt * lyap)
        return out

    meta: dict[str, Any] = {
        "keep_frac": float(mask.mean()),
        "keep_frac_time_any": float(any_obs.mean()),
        "keep_frac_time_all": float(all_obs.mean()),
        "expected_obs_per_patch": float(patch_length * mask.mean()),
        "expected_full_timesteps_per_patch": float(patch_length * all_obs.mean()),
        "patch_length": int(patch_length),
    }
    meta.update(step_stats(all_missing_runs, "all_missing_gap"))
    meta.update(step_stats(not_full_runs, "not_full_gap"))
    return meta


def _iid_time_mask(rng: np.random.Generator, T: int, D: int, sparsity: float) -> np.ndarray:
    keep_t = rng.random(T) > sparsity
    return np.repeat(keep_t[:, None], D, axis=1)


def _iid_channel_mask(rng: np.random.Generator, T: int, D: int, sparsity: float) -> np.ndarray:
    return rng.random((T, D)) > sparsity


def _block_time_mask(rng: np.random.Generator, T: int, D: int,
                     sparsity: float, block_len: int | None) -> np.ndarray:
    block = int(block_len or max(1, round(T * sparsity / 8)))
    block = max(1, min(block, T))
    keep = np.ones(T, dtype=bool)
    target_missing = int(round(T * sparsity))
    tries = 0
    while int((~keep).sum()) < target_missing and tries < 10 * max(1, T):
        start = int(rng.integers(0, max(1, T - block + 1)))
        keep[start:start + block] = False
        tries += 1
    return np.repeat(keep[:, None], D, axis=1)


def _periodic_subsample_mask(rng: np.random.Generator, T: int, D: int,
                             sparsity: float, period: int | None,
                             jitter: int | None) -> np.ndarray:
    if period is None:
        keep_prob = max(1e-6, 1.0 - sparsity)
        period = max(1, int(round(1.0 / keep_prob)))
    period = max(1, int(period))
    offset = int(rng.integers(0, period))
    kept = np.arange(offset, T, period)
    if jitter:
        offsets = rng.integers(-int(jitter), int(jitter) + 1, size=len(kept))
        kept = np.clip(kept + offsets, 0, T - 1)
    keep = np.zeros(T, dtype=bool)
    keep[np.unique(kept)] = True
    return np.repeat(keep[:, None], D, axis=1)


def _curvature_score(traj: np.ndarray) -> np.ndarray:
    if len(traj) < 3:
        return np.zeros(len(traj), dtype=np.float64)
    d1 = np.gradient(traj, axis=0)
    d2 = np.gradient(d1, axis=0)
    speed = np.linalg.norm(d1, axis=1)
    curv = np.linalg.norm(d2, axis=1)
    score = speed + curv
    lo, hi = np.quantile(score, [0.05, 0.95])
    if hi <= lo:
        return np.zeros_like(score)
    return np.clip((score - lo) / (hi - lo), 0.0, 1.0)


def _mnar_curvature_mask(rng: np.random.Generator, traj: np.ndarray,
                         sparsity: float, strength: float) -> np.ndarray:
    T, D = traj.shape
    score = _curvature_score(traj)
    raw = 0.05 + np.power(score, max(0.1, strength))
    p_miss = raw * (sparsity / max(raw.mean(), 1e-8))
    p_miss = np.clip(p_miss, 0.0, 0.98)
    keep = rng.random(T) > p_miss
    return np.repeat(keep[:, None], D, axis=1)


def make_corrupted_observations(
    traj: np.ndarray,
    *,
    mask_regime: str = "iid_time",
    sparsity: float = 0.0,
    noise_std_frac: float = 0.0,
    attractor_std: float | np.ndarray | None = None,
    seed: int = 0,
    per_dim_noise: bool = True,
    block_len: int | None = None,
    period: int | None = None,
    jitter: int | None = None,
    mnar_strength: float = 1.0,
    dt: float | None = None,
    lyap: float | None = None,
    patch_length: int = 16,
) -> CorruptionResult:
    """Create a sparse/noisy observation from a clean trajectory.

    Parameters are intentionally explicit so the JSON config can be mirrored in
    experiment records.  ``mask`` is always returned as a 2-D boolean array with
    shape ``[T, D]``.
    """
    clean = np.asarray(traj, dtype=np.float64)
    if clean.ndim == 1:
        clean = clean[:, None]
    T, D = clean.shape
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if noise_std_frac < 0:
        raise ValueError(f"noise_std_frac must be nonnegative, got {noise_std_frac}")

    rng = np.random.default_rng(seed)
    if mask_regime == "iid_time":
        mask = _iid_time_mask(rng, T, D, sparsity)
    elif mask_regime == "iid_channel":
        mask = _iid_channel_mask(rng, T, D, sparsity)
    elif mask_regime == "block_time":
        mask = _block_time_mask(rng, T, D, sparsity, block_len)
    elif mask_regime == "periodic_subsample":
        mask = _periodic_subsample_mask(rng, T, D, sparsity, period, jitter)
    elif mask_regime == "mnar_curvature":
        mask = _mnar_curvature_mask(rng, clean, sparsity, mnar_strength)
    else:
        raise ValueError(f"unknown mask_regime {mask_regime!r}")

    noise_scale = _as_noise_scale(clean, noise_std_frac, attractor_std, per_dim_noise)
    if np.isscalar(noise_scale):
        noise = rng.normal(scale=float(noise_scale), size=clean.shape)
    else:
        noise = rng.normal(size=clean.shape) * np.asarray(noise_scale)[None, :]
    noisy_full = clean + noise
    observed = noisy_full.copy()
    observed[~mask] = np.nan

    meta = _gap_metadata(mask, dt=dt, lyap=lyap, patch_length=patch_length)
    meta.update({
        "mask_regime": mask_regime,
        "sparsity_requested": float(sparsity),
        "noise_std_frac": float(noise_std_frac),
        "per_dim_noise": bool(per_dim_noise),
        "block_len": None if block_len is None else int(block_len),
        "period": None if period is None else int(period),
        "jitter": None if jitter is None else int(jitter),
        "mnar_strength": float(mnar_strength),
        "seed": int(seed),
    })
    if np.isscalar(noise_scale):
        meta["noise_scale"] = float(noise_scale)
    else:
        meta["noise_scale"] = [float(x) for x in np.asarray(noise_scale).ravel()]

    return CorruptionResult(
        observed=observed.astype(np.float32),
        mask=mask,
        noisy_full=noisy_full.astype(np.float32),
        metadata=meta,
    )


def result_to_jsonable(result: CorruptionResult) -> dict[str, Any]:
    """Return only JSON-safe metadata, not the arrays."""
    return {"metadata": dict(result.metadata)}
