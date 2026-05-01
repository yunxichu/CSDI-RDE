# L63 Panda Jitter Control

## SP65

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 1.15 ± 0.25 | 1.22 | 100% | 60% |
| linear_iid_jitter | 1.28 ± 0.29 | 1.36 | 100% | 80% |
| linear_shuffled_resid | 1.14 ± 0.24 | 1.22 | 100% | 60% |
| csdi | 2.84 ± 0.12 | 2.90 | 100% | 100% |

Paired differences vs linear:

| Contrast | Δ mean | 95% CI |
|:--|--:|:--|
| linear_iid_jitter_minus_linear | +0.13 | [-0.03, +0.43] |
| linear_shuffled_resid_minus_linear | -0.01 | [-0.04, +0.03] |
| csdi_minus_linear | +1.69 | [+1.47, +1.92] |

## SP82

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 0.32 ± 0.28 | 0.29 | 20% | 0% |
| linear_iid_jitter | 0.48 ± 0.53 | 0.14 | 40% | 20% |
| linear_shuffled_resid | 0.38 ± 0.37 | 0.23 | 40% | 0% |
| csdi | 0.88 ± 0.21 | 0.82 | 100% | 20% |

Paired differences vs linear:

| Contrast | Δ mean | 95% CI |
|:--|--:|:--|
| linear_iid_jitter_minus_linear | +0.15 | [-0.25, +0.56] |
| linear_shuffled_resid_minus_linear | +0.06 | [-0.28, +0.38] |
| csdi_minus_linear | +0.56 | [+0.26, +0.75] |
