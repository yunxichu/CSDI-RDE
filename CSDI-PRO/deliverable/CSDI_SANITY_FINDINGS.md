# CSDI Sanity Check — 2026-04-30

Question: are the CSDI results technically valid?

Short answer: CSDI is not leaking future information and the rescue signal
survives a stricter check, but we found two implementation issues that require
rerunning CSDI-related headline figures before final paper claims.

## What Is Correct

- CSDI receives only the context window `observed` and its missingness mask.
  Future values are passed only to VPT evaluation after Panda forecasts.
- Checkpoints have the expected dimensions:
  - L63: `data_dim=3`, `seq_len=128`, centered at `(0,0,16.384)`.
  - L96 N=20: `data_dim=20`, `seq_len=128`, centered near `2.192` per channel.
- The adapter uses the correct L63/L96 `attractor_std` override when the
  runner calls `set_csdi_attractor_std`.

## Issues Found

1. `DynamicsCSDI.impute()` normalized long windows before entering chunked
   inference, then normalized each chunk again in the recursive call. This
   affected direct model calls with `T > seq_len`.

2. The CSDI adapter estimated observation noise from irregular sparse
   observations even when the experimental scenario had `noise_std_frac=0`.
   This made pure-sparsity runs use a nonzero inferred `sigma`, so observed
   points were softly denoised rather than exactly anchored.

Both issues are now patched:
- `methods/dynamics_csdi.py`: chunking now happens in raw coordinates; uniform
  overlap averaging preserves observed-point anchoring.
- `methods/dynamics_impute.py`: `impute(..., kind="csdi", sigma_override=...)`
  is supported.
- Main runners now pass known experimental noise scale
  `sigma_override = noise_std_frac * attractor_std` into CSDI.

## Numerical Sanity Check

L63 SP65, seed 0, v2 protocol:

| Context | observed max error | missing RMSE vs clean | Panda VPT@1.0 |
|---|---:|---:|---:|
| linear | 0.0 | 3.207 | 1.087 |
| CSDI, old estimated sigma | 0.330 | 0.610 | 2.899 |
| CSDI, `sigma_override=0` | 7.6e-6 | 0.359 | 2.899 |

Thus the CSDI rescue is not an artifact of observed-point corruption; with the
correct sigma it becomes more faithful and remains at the horizon ceiling.

L63 NO050, seed 0, dense noisy context:

| Context | context RMSE vs clean | Panda VPT@1.0 |
|---|---:|---:|
| linear/noisy observed | 4.324 | 0.906 |
| CSDI, old estimated sigma | 3.945 | 0.408 |
| CSDI, true sigma | 3.940 | 0.408 |

This still supports the current story that CSDI is not a generic dense-noise
denoiser for Panda.

## Quick Recheck After Patch

`panda_jitter_control_l63.py` was rerun for SP65/SP82 with 5 seeds and
`sigma_override`.

| Scenario | linear mean | CSDI mean | Δ CSDI-linear | 95% CI |
|---|---:|---:|---:|---:|
| SP65 | 1.29 | 2.90 | +1.61 | [+1.15,+2.08] |
| SP82 | 0.34 | 0.95 | +0.61 | [+0.13,+1.30] |

The core L63 CSDI rescue strengthens under the corrected sigma protocol.

Files:
- `experiments/week1/results/panda_jitter_control_l63_sp65_sp82_sigmaoverride_5seed.json`
- `experiments/week1/figures/panda_jitter_control_l63_sp65_sp82_sigmaoverride_5seed.md`

## Required Before Final Paper

Rerun CSDI-dependent headline artifacts with the patched sigma protocol:

1. L63 Figure 1 v2 10-seed grid.
2. L63 representation/raw-patch diagnostics if they are used in §4.
3. L96 v2 cross-system replication.

Expected direction: CSDI pure-sparsity results likely improve or remain
unchanged; pure-noise Panda results likely remain non-rescuing.
