# Panda Alt-Imputer Reviewer Defense

Question: is structured imputation per se the lever, or is CSDI specifically required?
Per-instance SAITS / BRITS train on the single test trajectory's missing pattern.
Linear and CSDI follow the same protocol as Figure 1 (LORENZ63_ATTRACTOR_STD / lorenz96_attractor_std + grid-index seeds).

## L96_SP82

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 1.81 ± 3.33 | 0.50 | 60% | 30% |
| saits_pretrained | 1.56 ± 1.60 | 0.84 | 100% | 40% |
| csdi | 1.77 ± 1.57 | 1.13 | 100% | 60% |

Paired Δ vs linear (95% bootstrap CI):

| Cell | Δ mean | CI | sign |
|:--|--:|:--|:-:|
| saits_pretrained_minus_linear | -0.24 | [-1.55, +0.51] | ≈ |
| csdi_minus_linear | -0.03 | [-1.40, +0.88] | ≈ |
