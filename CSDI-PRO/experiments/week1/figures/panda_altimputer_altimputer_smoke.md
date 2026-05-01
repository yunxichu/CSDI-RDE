# Panda Alt-Imputer Reviewer Defense

Question: is structured imputation per se the lever, or is CSDI specifically required?
Per-instance SAITS / BRITS train on the single test trajectory's missing pattern.
Linear and CSDI follow the same protocol as Figure 1 (LORENZ63_ATTRACTOR_STD / lorenz96_attractor_std + grid-index seeds).

## L63_SP65

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 1.09 ± 0.00 | 1.09 | 100% | 100% |
| saits | 0.29 ± 0.00 | 0.29 | 0% | 0% |
| brits | 0.29 ± 0.00 | 0.29 | 0% | 0% |
| csdi | 2.90 ± 0.00 | 2.90 | 100% | 100% |

Paired Δ vs linear (95% bootstrap CI):

| Cell | Δ mean | CI | sign |
|:--|--:|:--|:-:|
| saits_minus_linear | -0.79 | [-0.79, -0.79] | ↓ |
| brits_minus_linear | -0.79 | [-0.79, -0.79] | ↓ |
| csdi_minus_linear | +1.81 | [+1.81, +1.81] | ↑ |
