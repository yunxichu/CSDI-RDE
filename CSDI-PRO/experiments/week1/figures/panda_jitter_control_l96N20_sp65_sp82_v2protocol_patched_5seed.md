# L96 N=20 Panda Jitter Control

## SP65

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 2.18 ± 2.26 | 0.76 | 80% | 40% |
| linear_iid_jitter | 2.45 ± 2.56 | 0.76 | 100% | 40% |
| linear_shuffled_resid | 2.47 ± 2.62 | 0.76 | 100% | 40% |
| csdi | 2.59 ± 1.93 | 1.76 | 100% | 80% |

Paired differences vs linear:

| Contrast | Δ mean | 95% CI |
|:--|--:|:--|
| linear_iid_jitter_minus_linear | +0.27 | [-0.07, +0.72] |
| linear_shuffled_resid_minus_linear | +0.29 | [-0.03, +0.76] |
| csdi_minus_linear | +0.40 | [-0.03, +1.04] |

## SP82

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 3.04 ± 4.55 | 0.50 | 60% | 40% |
| linear_iid_jitter | 1.63 ± 1.73 | 0.59 | 80% | 40% |
| linear_shuffled_resid | 1.63 ± 1.71 | 0.67 | 80% | 40% |
| csdi | 2.50 ± 1.97 | 1.76 | 100% | 60% |

Paired differences vs linear:

| Contrast | Δ mean | 95% CI |
|:--|--:|:--|
| linear_iid_jitter_minus_linear | -1.41 | [-4.54, +0.22] |
| linear_shuffled_resid_minus_linear | -1.41 | [-4.62, +0.27] |
| csdi_minus_linear | -0.54 | [-3.21, +1.21] |
