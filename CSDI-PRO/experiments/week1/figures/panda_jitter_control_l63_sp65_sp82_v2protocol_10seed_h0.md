# L63 Panda Jitter Control

## SP65

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 1.29 ± 0.57 | 1.29 | 80% | 80% |
| linear_iid_jitter | 1.39 ± 0.80 | 1.25 | 80% | 80% |
| linear_shuffled_resid | 1.11 ± 0.40 | 1.25 | 80% | 60% |
| csdi | 2.90 ± 0.00 | 2.90 | 100% | 100% |

Paired differences vs linear:

| Contrast | Δ mean | 95% CI |
|:--|--:|:--|
| linear_iid_jitter_minus_linear | +0.10 | [-0.04, +0.34] |
| linear_shuffled_resid_minus_linear | -0.19 | [-0.43, -0.05] |
| csdi_minus_linear | +1.61 | [+1.15, +2.08] |

## SP82

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 0.34 ± 0.29 | 0.27 | 40% | 0% |
| linear_iid_jitter | 0.50 ± 0.56 | 0.32 | 40% | 20% |
| linear_shuffled_resid | 0.80 ± 0.39 | 0.68 | 80% | 20% |
| csdi | 0.52 ± 0.48 | 0.41 | 40% | 20% |

Paired differences vs linear:

| Contrast | Δ mean | 95% CI |
|:--|--:|:--|
| linear_iid_jitter_minus_linear | +0.16 | [-0.29, +0.76] |
| linear_shuffled_resid_minus_linear | +0.46 | [+0.08, +0.89] |
| csdi_minus_linear | +0.18 | [-0.29, +0.70] |
