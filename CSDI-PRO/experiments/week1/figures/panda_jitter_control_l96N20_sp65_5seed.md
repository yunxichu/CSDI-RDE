# L96 N=20 Panda Jitter Control

## SP65

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 1.33 ± 2.28 | 0.50 | 60% | 20% |
| linear_iid_jitter | 2.40 ± 2.56 | 0.67 | 100% | 40% |
| linear_shuffled_resid | 2.42 ± 2.54 | 0.67 | 100% | 40% |
| csdi | 2.52 ± 1.98 | 1.68 | 100% | 80% |

Paired differences vs linear:

| Contrast | Δ mean | 95% CI |
|:--|--:|:--|
| linear_iid_jitter_minus_linear | +1.08 | [+0.10, +2.55] |
| linear_shuffled_resid_minus_linear | +1.09 | [+0.17, +2.54] |
| csdi_minus_linear | +1.19 | [+0.08, +2.65] |
