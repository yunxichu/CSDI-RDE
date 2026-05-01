# Cross-System Jitter Control — milestone summary

Each cell reports VPT@1.0 mean ± std and Pr(VPT > 1.0 Λ) from 5-seed Panda forecasts on a corruption-aware filled context.
Controls: linear (no intervention), iid jitter (Gaussian noise scaled to per-channel CSDI residual std, applied only at missing entries), shuffled residual (CSDI residual values shuffled across missing positions), CSDI (the imputation under test).

## SP65

| System | Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|:--|--:|--:|--:|--:|
| L63 | linear | 1.42 ± 0.44 | 1.49 | 100% | 80% |
| L63 | linear_iid_jitter | 0.82 ± 0.47 | 0.70 | 60% | 40% |
| L63 | linear_shuffled_resid | 0.94 ± 0.54 | 1.29 | 60% | 60% |
| L63 | csdi | 1.23 ± 0.59 | 1.20 | 80% | 80% |
|  |  |  |  |  |  |
| L96 N=20 | linear | 1.33 ± 2.28 | 0.50 | 60% | 20% |
| L96 N=20 | linear_iid_jitter | 2.40 ± 2.56 | 0.67 | 100% | 40% |
| L96 N=20 | linear_shuffled_resid | 2.40 ± 2.56 | 0.67 | 100% | 40% |
| L96 N=20 | csdi | 2.52 ± 1.99 | 1.76 | 100% | 60% |
|  |  |  |  |  |  |
| Rössler | linear | 0.69 ± 0.38 | 0.91 | 80% | 0% |
| Rössler | linear_iid_jitter | 0.65 ± 0.29 | 0.64 | 80% | 0% |
| Rössler | linear_shuffled_resid | 0.65 ± 0.29 | 0.63 | 80% | 0% |
| Rössler | csdi | 0.91 ± 0.00 | 0.91 | 100% | 0% |
|  |  |  |  |  |  |

Paired contrasts vs linear (Δ mean VPT@1.0, 95% CI):

| System | Cell | Δ | CI | sign |
|:--|:--|--:|:--|:-:|
| L63 | linear_iid_jitter | -0.60 | [-1.28, +0.09] | ≈ |
| L63 | linear_shuffled_resid | -0.48 | [-1.17, +0.20] | ≈ |
| L63 | csdi | -0.19 | [-0.90, +0.62] | ≈ |
| L96 N=20 | linear_iid_jitter | +1.08 | [+0.13, +2.54] | ↑ |
| L96 N=20 | linear_shuffled_resid | +1.08 | [+0.13, +2.52] | ↑ |
| L96 N=20 | csdi | +1.19 | [+0.08, +2.62] | ↑ |
| Rössler | linear_iid_jitter | -0.04 | [-0.19, +0.10] | ≈ |
| Rössler | linear_shuffled_resid | -0.04 | [-0.19, +0.10] | ≈ |
| Rössler | csdi | +0.22 | [+0.00, +0.57] | ≈ |

## SP82

| System | Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|:--|--:|--:|--:|--:|
| L63 | linear | 0.43 ± 0.58 | 0.09 | 40% | 20% |
| L63 | linear_iid_jitter | 0.43 ± 0.59 | 0.09 | 40% | 20% |
| L63 | linear_shuffled_resid | 0.57 ± 0.73 | 0.09 | 40% | 40% |
| L63 | csdi | 0.86 ± 0.43 | 0.68 | 60% | 40% |
|  |  |  |  |  |  |
| L96 N=20 | linear | 0.91 ± 1.80 | 0.00 | 20% | 20% |
| L96 N=20 | linear_iid_jitter | 0.91 ± 1.71 | 0.08 | 40% | 20% |
| L96 N=20 | linear_shuffled_resid | 1.09 ± 1.67 | 0.59 | 60% | 20% |
| L96 N=20 | csdi | 3.31 ± 4.44 | 1.09 | 80% | 60% |
|  |  |  |  |  |  |
| Rössler | linear | 0.49 ± 0.46 | 0.64 | 60% | 0% |
| Rössler | linear_iid_jitter | 0.38 ± 0.38 | 0.23 | 40% | 0% |
| Rössler | linear_shuffled_resid | 0.35 ± 0.40 | 0.23 | 40% | 0% |
| Rössler | csdi | 0.72 ± 0.30 | 0.91 | 80% | 0% |
|  |  |  |  |  |  |

Paired contrasts vs linear (Δ mean VPT@1.0, 95% CI):

| System | Cell | Δ | CI | sign |
|:--|:--|--:|:--|:-:|
| L63 | linear_iid_jitter | +0.00 | [+0.00, +0.01] | ≈ |
| L63 | linear_shuffled_resid | +0.14 | [+0.00, +0.39] | ≈ |
| L63 | csdi | +0.43 | [+0.04, +0.86] | ↑ |
| L96 N=20 | linear_iid_jitter | -0.00 | [-0.08, +0.07] | ≈ |
| L96 N=20 | linear_shuffled_resid | +0.18 | [-0.02, +0.47] | ≈ |
| L96 N=20 | csdi | +2.40 | [+0.10, +6.59] | ↑ |
| Rössler | linear_iid_jitter | -0.11 | [-0.41, +0.07] | ≈ |
| Rössler | linear_shuffled_resid | -0.14 | [-0.41, +0.00] | ≈ |
| Rössler | csdi | +0.23 | [-0.27, +0.73] | ≈ |
