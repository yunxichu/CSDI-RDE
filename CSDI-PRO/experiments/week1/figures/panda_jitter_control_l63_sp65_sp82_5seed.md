# L63 Panda Jitter Control

## SP65

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 1.42 ± 0.44 | 1.49 | 100% | 80% |
| linear_iid_jitter | 0.82 ± 0.47 | 0.70 | 60% | 40% |
| linear_shuffled_resid | 0.94 ± 0.54 | 1.29 | 60% | 60% |
| csdi | 1.23 ± 0.59 | 1.20 | 80% | 80% |

Paired differences vs linear:

| Contrast | Δ mean | 95% CI |
|:--|--:|:--|
| linear_iid_jitter_minus_linear | -0.60 | [-1.28, +0.09] |
| linear_shuffled_resid_minus_linear | -0.48 | [-1.17, +0.20] |
| csdi_minus_linear | -0.19 | [-0.90, +0.62] |

## SP82

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 0.43 ± 0.58 | 0.09 | 40% | 20% |
| linear_iid_jitter | 0.43 ± 0.59 | 0.09 | 40% | 20% |
| linear_shuffled_resid | 0.57 ± 0.73 | 0.09 | 40% | 40% |
| csdi | 0.86 ± 0.43 | 0.68 | 60% | 40% |

Paired differences vs linear:

| Contrast | Δ mean | 95% CI |
|:--|--:|:--|
| linear_iid_jitter_minus_linear | +0.00 | [+0.00, +0.01] |
| linear_shuffled_resid_minus_linear | +0.14 | [+0.00, +0.39] |
| csdi_minus_linear | +0.43 | [+0.04, +0.86] |
