# L63 Panda Jitter Control

## SP65

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 1.22 ± 0.44 | 1.26 | 90% | 70% |
| linear_iid_jitter | 1.39 ± 0.61 | 1.30 | 90% | 80% |
| linear_shuffled_resid | 1.06 ± 0.35 | 1.13 | 90% | 60% |
| csdi | 2.87 ± 0.08 | 2.90 | 100% | 100% |

Paired differences vs linear:

| Contrast | Δ mean | 95% CI |
|:--|--:|:--|
| linear_iid_jitter_minus_linear | +0.17 | [-0.01, +0.36] |
| linear_shuffled_resid_minus_linear | -0.16 | [-0.34, -0.02] |
| csdi_minus_linear | +1.65 | [+1.41, +1.87] |

## SP82

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 0.33 ± 0.27 | 0.28 | 30% | 0% |
| linear_iid_jitter | 0.54 ± 0.65 | 0.29 | 40% | 20% |
| linear_shuffled_resid | 0.67 ± 0.43 | 0.66 | 60% | 20% |
| csdi | 1.42 ± 0.87 | 1.28 | 80% | 70% |

Paired differences vs linear:

| Contrast | Δ mean | 95% CI |
|:--|--:|:--|
| linear_iid_jitter_minus_linear | +0.21 | [-0.12, +0.65] |
| linear_shuffled_resid_minus_linear | +0.34 | [+0.05, +0.65] |
| csdi_minus_linear | +1.09 | [+0.65, +1.61] |
