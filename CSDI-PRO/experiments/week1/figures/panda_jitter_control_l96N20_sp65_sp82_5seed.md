# L96 N=20 Panda Jitter Control

## SP65

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 1.33 ± 2.28 | 0.50 | 60% | 20% |
| linear_iid_jitter | 2.40 ± 2.56 | 0.67 | 100% | 40% |
| linear_shuffled_resid | 2.40 ± 2.56 | 0.67 | 100% | 40% |
| csdi | 2.52 ± 1.99 | 1.76 | 100% | 60% |

Paired differences vs linear:

| Contrast | Δ mean | 95% CI |
|:--|--:|:--|
| linear_iid_jitter_minus_linear | +1.08 | [+0.10, +2.55] |
| linear_shuffled_resid_minus_linear | +1.08 | [+0.13, +2.52] |
| csdi_minus_linear | +1.19 | [+0.08, +2.67] |

## SP82

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 0.91 ± 1.80 | 0.00 | 20% | 20% |
| linear_iid_jitter | 0.91 ± 1.71 | 0.08 | 40% | 20% |
| linear_shuffled_resid | 1.09 ± 1.67 | 0.59 | 60% | 20% |
| csdi | 3.31 ± 4.44 | 1.09 | 80% | 60% |

Paired differences vs linear:

| Contrast | Δ mean | 95% CI |
|:--|--:|:--|
| linear_iid_jitter_minus_linear | -0.00 | [-0.08, +0.07] |
| linear_shuffled_resid_minus_linear | +0.18 | [-0.02, +0.47] |
| csdi_minus_linear | +2.40 | [+0.08, +6.59] |
