# L63 Panda Jitter Control

## SP65

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 1.29 ± 0.57 | 1.29 | 80% | 80% |
| linear_iid_jitter | 1.36 ± 0.98 | 1.25 | 60% | 60% |
| linear_shuffled_resid | 1.11 ± 0.40 | 1.25 | 80% | 60% |
| csdi | 2.90 ± 0.00 | 2.90 | 100% | 100% |

Paired differences vs linear:

| Contrast | Δ mean | 95% CI |
|:--|--:|:--|
| linear_iid_jitter_minus_linear | +0.07 | [-0.32, +0.45] |
| linear_shuffled_resid_minus_linear | -0.19 | [-0.43, -0.05] |
| csdi_minus_linear | +1.61 | [+1.15, +2.08] |

## SP82

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 0.34 ± 0.28 | 0.29 | 40% | 0% |
| linear_iid_jitter | 0.73 ± 0.80 | 0.59 | 60% | 20% |
| linear_shuffled_resid | 0.47 ± 0.21 | 0.39 | 40% | 0% |
| csdi | 0.95 ± 0.75 | 0.79 | 60% | 40% |

Paired differences vs linear:

| Contrast | Δ mean | 95% CI |
|:--|--:|:--|
| linear_iid_jitter_minus_linear | +0.39 | [-0.00, +1.16] |
| linear_shuffled_resid_minus_linear | +0.12 | [-0.19, +0.41] |
| csdi_minus_linear | +0.61 | [+0.13, +1.30] |
