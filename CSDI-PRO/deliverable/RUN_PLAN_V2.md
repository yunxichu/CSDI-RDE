# V2 Experiment Run Plan

Date: 2026-04-26

Goal: turn the sparse/noisy story from a coarse S0-S6 stress test into a
reviewer-safe mechanism study.

## Resource Budget

Hard limits requested by the user:

- GPUs: use at most 4 visible GPUs total.
- CPU: use at most 20% of the system CPU.

Current machine reports 152 CPU cores, so 20% is about 30 cores. The plan below
uses at most 4 experiment processes at once, with 4 CPU threads per process
(about 16 threads total).

Every official run should set:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4
```

For CUDA runs, set `CUDA_VISIBLE_DEVICES` explicitly. The new runner refuses
CUDA runs when this is unset, and refuses more than 4 visible GPUs.

## What Changed Today

Implemented:

- `experiments/week1/corruption.py`
  - iid time dropout
  - iid channel dropout
  - block outages
  - periodic / jittered subsampling
  - curvature-dependent MNAR missingness
  - metadata: keep fraction, observations per patch, max gap in Lyapunov units

- `experiments/week1/configs/corruption_grid_v2.json`
  - legacy S0-S6
  - pure sparsity line
  - pure noise line
  - transition rectangle
  - H0-H6 summary path
  - missingness pattern grid

- `experiments/week1/phase_transition_grid_l63_v2.py`
  - Lorenz63 v2 grid runner
  - dry-run metadata mode
  - resource guards for CPU threads and visible GPUs
  - imputer x forecaster cells: `panda_linear`, `panda_csdi`, `deepedm_linear`,
    `deepedm_csdi`, with optional Kalman cells

- `experiments/week1/aggregate_corruption_grid.py`
  - Markdown table with keep fraction, obs/patch, max gap, median VPT, survival

- `experiments/week1/full_pipeline_rollout.py`
  - added `m3_device` so DeepEDM/SVGP/FNO obey the runner device budget.

Verified:

- Python compile passes.
- L63 dry-run metadata works.
- A tiny CPU non-dry smoke run works.
- A tiny GPU Panda smoke works outside the sandbox with GPU0.
- L63 3-seed v2 pilot completed on GPU0-GPU3.

Pilot readout:

- See `L63_V2_PILOT_READOUT.md`.
- The main sparsity transition band is `s=0.40-0.75`.
- The floor begins near `s=0.82`, where patches contain about 2-3 observed
  points.
- Pure noise begins degrading Panda around `sigma=0.10-0.20`, even with no
  missingness.
- Pattern effects are real: same missing rate can produce very different gap
  geometry and VPT.

## Metadata Readout So Far

The proposed H0-H6 summary path, averaged over 20 masks on L63:

| Stage | s | sigma | keep | obs/patch | max gap L |
|---|---:|---:|---:|---:|---:|
| H0 | 0.00 | 0.00 | 1.000 | 16.00 | 0.00 |
| H1 | 0.20 | 0.05 | 0.797 | 12.76 | 0.08 |
| H2 | 0.40 | 0.10 | 0.590 | 9.43 | 0.14 |
| H3 | 0.65 | 0.20 | 0.345 | 5.52 | 0.31 |
| H4 | 0.82 | 0.35 | 0.179 | 2.86 | 0.51 |
| H5 | 0.90 | 0.60 | 0.105 | 1.68 | 0.87 |
| H6 | 0.95 | 1.00 | 0.048 | 0.76 | 1.65 |

Interpretation: H0-H6 is more defensible than the old S0-S6 because H4-H6
cross the patch-information threshold gradually instead of jumping straight to
`sigma=1.2/1.5`.

The missingness pattern grid shows why missing rate alone is insufficient:

| Regime | keep | obs/patch | max gap L |
|---|---:|---:|---:|
| iid channel, s=0.60 | 0.400 | 6.40 | 0.09 |
| iid time, s=0.60 | 0.397 | 6.36 | 0.23 |
| block len 8, s=0.60 | 0.397 | 6.35 | 0.80 |
| block len 16, s=0.60 | 0.392 | 6.27 | 1.46 |
| block len 32, s=0.60 | 0.381 | 6.09 | 2.32 |
| periodic s=0.80 | 0.200 | 3.20 | 0.09 |
| jittered periodic s=0.80 | 0.200 | 3.20 | 0.14 |
| MNAR curvature s=0.60 | 0.482 | 7.71 | 0.39 |

Interpretation: the paper must report gap geometry, not just missing rate.
Block missingness is a separate failure axis from iid sparsity.

## Work Order

1. ✅ L63 v2 pilot, because it is cheap and validates the redesigned grid.
2. Extend the v2 runner template to L96 N=20 and Rössler.
3. Run pure sparsity and pure noise lines first; do not run the full rectangle
   until the transition band is located.
4. Run the imputer x forecaster matrix only around the transition band.
5. Add tokenizer/patch OOD geometry after we know which transition cells matter.
6. Rewrite paper sections around the confirmed failure law.

Immediate expansion should focus on:

- L63 pure sparsity: `SP40, SP55, SP65, SP75, SP82`
- L63 summary: `H2, H3, H4`
- L63 pure noise: `NO005, NO010, NO020, NO035, NO050`
- pattern robustness: iid time, block16, periodic, MNAR curvature

## Formal Commands

L63 summary path, one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
python -u -m experiments.week1.phase_transition_grid_l63_v2 \
  --grid summary_path_candidate \
  --n_seeds 10 \
  --cells panda_linear panda_csdi deepedm_linear deepedm_csdi \
  --csdi_ckpt experiments/week2_modules/ckpts/dyn_csdi_full_v6_center_ep20.pt \
  --tag l63_summary_v2_10seed
```

L63 pure sparsity line:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
python -u -m experiments.week1.phase_transition_grid_l63_v2 \
  --grid fine_s_line \
  --n_seeds 10 \
  --cells panda_linear panda_csdi deepedm_linear deepedm_csdi \
  --csdi_ckpt experiments/week2_modules/ckpts/dyn_csdi_full_v6_center_ep20.pt \
  --tag l63_fine_s_v2_10seed
```

L63 pure noise line:

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
python -u -m experiments.week1.phase_transition_grid_l63_v2 \
  --grid fine_sigma_line \
  --n_seeds 10 \
  --cells panda_linear panda_csdi deepedm_linear deepedm_csdi \
  --csdi_ckpt experiments/week2_modules/ckpts/dyn_csdi_full_v6_center_ep20.pt \
  --tag l63_fine_sigma_v2_10seed
```

L63 pattern grid:

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
python -u -m experiments.week1.phase_transition_grid_l63_v2 \
  --grid pattern_grid \
  --n_seeds 10 \
  --cells panda_linear panda_csdi deepedm_linear deepedm_csdi \
  --csdi_ckpt experiments/week2_modules/ckpts/dyn_csdi_full_v6_center_ep20.pt \
  --tag l63_pattern_v2_10seed
```

Aggregate any result:

```bash
python -m experiments.week1.aggregate_corruption_grid \
  --json experiments/week1/results/pt_l63_grid_v2_l63_summary_v2_10seed.json \
  --out experiments/week1/figures/l63_summary_v2_10seed.md
```

## Paper Decision Rule

If `CSDI -> Panda` rescues most of the transition band, the main claim is:

> Sparse observation preprocessing creates a patch/tokenizer OOD channel;
> corruption-aware imputation is the first mitigation lever, and
> delay-manifold forecasting is a dynamics-structured companion.

If `CSDI -> Panda` still collapses while `CSDI -> DeepEDM` survives, the stronger
claim becomes:

> Tokenized ambient forecasters remain OOD beyond the corruption threshold;
> delay-manifold estimators avoid that channel.

Either outcome is publishable, but the first is more conservative and is
currently better supported by the isolation runs.
