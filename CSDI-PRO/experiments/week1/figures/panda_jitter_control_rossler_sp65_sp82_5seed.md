# Rössler Panda Jitter Control

## SP65

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 0.69 ± 0.38 | 0.91 | 80% | 0% |
| linear_iid_jitter | 0.65 ± 0.29 | 0.64 | 80% | 0% |
| linear_shuffled_resid | 0.65 ± 0.29 | 0.63 | 80% | 0% |
| csdi | 0.91 ± 0.00 | 0.91 | 100% | 0% |

Paired differences vs linear:

| Contrast | Δ mean | 95% CI |
|:--|--:|:--|
| linear_iid_jitter_minus_linear | -0.04 | [-0.19, +0.10] |
| linear_shuffled_resid_minus_linear | -0.04 | [-0.19, +0.10] |
| csdi_minus_linear | +0.22 | [+0.00, +0.57] |

## SP82

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 0.49 ± 0.46 | 0.64 | 60% | 0% |
| linear_iid_jitter | 0.38 ± 0.38 | 0.23 | 40% | 0% |
| linear_shuffled_resid | 0.35 ± 0.40 | 0.23 | 40% | 0% |
| csdi | 0.72 ± 0.30 | 0.91 | 80% | 0% |

Paired differences vs linear:

| Contrast | Δ mean | 95% CI |
|:--|--:|:--|
| linear_iid_jitter_minus_linear | -0.11 | [-0.41, +0.07] |
| linear_shuffled_resid_minus_linear | -0.14 | [-0.41, +0.00] |
| csdi_minus_linear | +0.23 | [-0.27, +0.73] |
