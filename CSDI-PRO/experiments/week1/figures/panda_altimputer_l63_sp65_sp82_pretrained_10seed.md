# Panda Alt-Imputer Reviewer Defense

Question: is structured imputation per se the lever, or is CSDI specifically required?
Per-instance SAITS / BRITS train on the single test trajectory's missing pattern.
Linear and CSDI follow the same protocol as Figure 1 (LORENZ63_ATTRACTOR_STD / lorenz96_attractor_std + grid-index seeds).

## L63_SP65

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 1.22 ± 0.43 | 1.26 | 90% | 70% |
| csdi | 2.76 ± 0.41 | 2.90 | 100% | 100% |

Paired Δ vs linear (95% bootstrap CI):

| Cell | Δ mean | CI | sign |
|:--|--:|:--|:-:|
| csdi_minus_linear | +1.54 | [+1.14, +1.88] | ↑ |

## L63_SP82

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 0.33 ± 0.27 | 0.28 | 30% | 0% |
| csdi | 1.62 ± 0.94 | 1.43 | 80% | 80% |

Paired Δ vs linear (95% bootstrap CI):

| Cell | Δ mean | CI | sign |
|:--|--:|:--|:-:|
| csdi_minus_linear | +1.29 | [+0.77, +1.83] | ↑ |
