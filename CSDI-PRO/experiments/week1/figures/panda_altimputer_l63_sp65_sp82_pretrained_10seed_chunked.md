# Panda Alt-Imputer Reviewer Defense

Question: is structured imputation per se the lever, or is CSDI specifically required?
Per-instance SAITS / BRITS train on the single test trajectory's missing pattern.
Linear and CSDI follow the same protocol as Figure 1 (LORENZ63_ATTRACTOR_STD / lorenz96_attractor_std + grid-index seeds).

## L63_SP65

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 1.22 ± 0.42 | 1.26 | 90% | 70% |
| csdi | 2.89 ± 0.01 | 2.90 | 100% | 100% |
| saits_pretrained | 2.49 ± 0.70 | 2.87 | 100% | 90% |

Paired Δ vs linear (95% bootstrap CI):

| Cell | Δ mean | CI | sign |
|:--|--:|:--|:-:|
| csdi_minus_linear | +1.67 | [+1.41, +1.92] | ↑ |
| saits_pretrained_minus_linear | +1.26 | [+0.83, +1.64] | ↑ |

## L63_SP82

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 0.29 ± 0.24 | 0.23 | 20% | 0% |
| csdi | 1.57 ± 1.01 | 1.31 | 80% | 70% |
| saits_pretrained | 1.51 ± 0.77 | 1.38 | 90% | 70% |

Paired Δ vs linear (95% bootstrap CI):

| Cell | Δ mean | CI | sign |
|:--|--:|:--|:-:|
| csdi_minus_linear | +1.28 | [+0.73, +1.85] | ↑ |
| saits_pretrained_minus_linear | +1.23 | [+0.86, +1.62] | ↑ |
