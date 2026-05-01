# L63 V2 Pilot Readout

Date: 2026-04-26

Run budget used: 4 concurrent GPU processes on GPU0-GPU3, 4 CPU threads per
process. All runs completed successfully.

## Outputs

JSON:

- `experiments/week1/results/pt_l63_grid_v2_l63_summary_v2_3seed.json`
- `experiments/week1/results/pt_l63_grid_v2_l63_fine_s_v2_3seed.json`
- `experiments/week1/results/pt_l63_grid_v2_l63_fine_sigma_v2_3seed.json`
- `experiments/week1/results/pt_l63_grid_v2_l63_pattern_v2_3seed.json`

Markdown:

- `experiments/week1/figures/l63_summary_v2_3seed.md`
- `experiments/week1/figures/l63_fine_s_v2_3seed.md`
- `experiments/week1/figures/l63_fine_sigma_v2_3seed.md`
- `experiments/week1/figures/l63_pattern_v2_3seed.md`

## Main Read

The v2 corruption design is useful. It separates three effects that the old
S0-S6 diagonal path confounded:

1. Pure sparsity has a patch-information threshold around `s=0.75-0.82`.
2. Pure noise already degrades Panda around `sigma=0.10-0.20`.
3. Missingness pattern changes difficulty even at the same missing rate.

This supports replacing the old headline S0-S6 ladder with:

- pure sparsity line,
- pure noise line,
- a smaller transition-band rectangle,
- and a pattern/grid robustness panel.

## Key Contrasts

Values below are VPT@1.0 means over 3 seeds. Treat them as pilot signals, not
headline statistics.

### Summary Path

| Stage | Panda linear | Panda CSDI | Δ Panda | DeepEDM linear | DeepEDM CSDI | Δ DeepEDM |
|---|---:|---:|---:|---:|---:|---:|
| H2 | 1.01 | 2.30 | +1.29 | 0.52 | 1.40 | +0.88 |
| H3 | 0.95 | 1.20 | +0.25 | 0.44 | 0.70 | +0.26 |
| H4 | 0.04 | 0.25 | +0.21 | 0.03 | 0.26 | +0.23 |
| H5 | 0.08 | 0.23 | +0.16 | 0.07 | 0.15 | +0.08 |

Interpretation: H2 is the cleanest rescue point. H4-H6 are near floor and should
be interpreted as stress/no-info regimes, not as the main evidence.

### Pure Sparsity

| Stage | obs/patch | Panda linear | Panda CSDI | Δ Panda | DeepEDM linear | DeepEDM CSDI | Δ DeepEDM |
|---|---:|---:|---:|---:|---:|---:|---:|
| SP40 | 9.31 | 1.65 | 2.90 | +1.25 | 1.53 | 2.23 | +0.70 |
| SP55 | 7.28 | 1.64 | 2.77 | +1.13 | 1.46 | 1.42 | -0.04 |
| SP65 | 5.74 | 1.03 | 2.33 | +1.29 | 0.45 | 1.81 | +1.36 |
| SP75 | 3.95 | 0.48 | 1.21 | +0.73 | 0.71 | 0.94 | +0.23 |
| SP82 | 2.46 | 0.16 | 0.25 | +0.09 | 0.33 | 0.40 | +0.07 |

Interpretation: the main sparsity transition band is `s=0.40-0.75`, with the
sharp floor beginning near `s=0.82` when patches contain about 2-3 observations.

### Pure Noise

| Stage | sigma | Panda linear | Panda CSDI | DeepEDM linear | DeepEDM CSDI |
|---|---:|---:|---:|---:|---:|
| NO005 | 0.05 | 2.80 | 2.80 | 1.82 | 1.46 |
| NO010 | 0.10 | 2.13 | 2.12 | 1.21 | 0.74 |
| NO020 | 0.20 | 1.18 | 0.97 | 0.65 | 0.39 |
| NO050 | 0.50 | 0.48 | 0.31 | 0.20 | 0.34 |

Interpretation: with no missingness, CSDI is not expected to help much because
there is no imputation gap. This line isolates observation-noise sensitivity,
not sparse-preprocessing OOD.

### Missingness Pattern

At roughly `s=0.60, sigma=0.20`, same missing rate does not mean same task:

| Regime | obs/patch | max gap L | Panda linear | Panda CSDI | DeepEDM linear | DeepEDM CSDI |
|---|---:|---:|---:|---:|---:|---:|
| iid time | 6.34 | 0.26 | 0.94 | 1.19 | 0.32 | 0.25 |
| iid channel | 6.39 | 0.11 | 1.12 | 0.95 | 0.67 | 0.48 |
| block 8 | 6.38 | 0.67 | 0.24 | 0.32 | 0.59 | 0.66 |
| block 16 | 6.30 | 2.03 | 0.22 | 0.72 | 0.35 | 0.24 |
| block 32 | 6.28 | 1.88 | 0.45 | 0.48 | 0.32 | 0.48 |
| MNAR curvature | 7.62 | 0.36 | 0.23 | 0.27 | 0.50 | 0.20 |

Interpretation: report gap geometry next to missing rate. Block outages and
MNAR curvature are not interchangeable with iid dropout.

## Recommended Next Runs

Do not immediately expand every cell to 20 seeds. The pilot says to prioritize:

1. L63 pure sparsity at `SP40, SP55, SP65, SP75, SP82`, 10-20 seeds.
2. L63 summary `H2, H3, H4`, 10-20 seeds.
3. L63 pure noise `NO005, NO010, NO020, NO035, NO050`, 10 seeds.
4. Pattern grid only for the strongest contrasts, especially iid time, block16,
   periodic, and MNAR curvature.

For paper narrative:

- CSDI rescue is strongest when missingness creates interpolation/preprocessing
  artifacts, not when corruption is pure observation noise.
- The threshold is better described by `obs/patch` and gap length than by
  missing rate alone.
- H4-H6 should be described as floor/stress conditions.
