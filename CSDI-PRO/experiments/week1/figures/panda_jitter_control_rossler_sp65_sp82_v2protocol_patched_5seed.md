# Rössler Panda Jitter Control

## SP65

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 0.85 ± 0.13 | 0.91 | 100% | 0% |
| linear_iid_jitter | 0.84 ± 0.14 | 0.91 | 100% | 0% |
| linear_shuffled_resid | 0.79 ± 0.16 | 0.91 | 100% | 0% |
| csdi | 0.91 ± 0.00 | 0.91 | 100% | 0% |

Paired differences vs linear:

| Contrast | Δ mean | 95% CI |
|:--|--:|:--|
| linear_iid_jitter_minus_linear | -0.01 | [-0.02, +0.00] |
| linear_shuffled_resid_minus_linear | -0.06 | [-0.17, +0.00] |
| csdi_minus_linear | +0.06 | [+0.00, +0.17] |

## SP82

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 0.53 ± 0.26 | 0.59 | 60% | 0% |
| linear_iid_jitter | 0.56 ± 0.20 | 0.63 | 80% | 0% |
| linear_shuffled_resid | 0.60 ± 0.25 | 0.62 | 80% | 0% |
| csdi | 0.85 ± 0.14 | 0.91 | 100% | 0% |

Paired differences vs linear:

| Contrast | Δ mean | 95% CI |
|:--|--:|:--|
| linear_iid_jitter_minus_linear | +0.03 | [-0.11, +0.22] |
| linear_shuffled_resid_minus_linear | +0.07 | [-0.02, +0.22] |
| csdi_minus_linear | +0.31 | [+0.07, +0.56] |
