# Panda Alt-Imputer Reviewer Defense

Question: is structured imputation per se the lever, or is CSDI specifically required?
Per-instance SAITS / BRITS train on the single test trajectory's missing pattern.
Linear and CSDI follow the same protocol as Figure 1 (LORENZ63_ATTRACTOR_STD / lorenz96_attractor_std + grid-index seeds).

## L96_SP82

| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |
|:--|--:|--:|--:|--:|
| linear | 0.86 ± 2.04 | 0.25 | 37% | 20% |
| saits_pretrained | 1.57 ± 1.60 | 1.01 | 80% | 50% |
| csdi | 1.87 ± 1.51 | 1.26 | 97% | 73% |

Paired Δ vs linear (95% bootstrap CI):

| Cell | Δ mean | CI | sign |
|:--|--:|:--|:-:|
| saits_pretrained_minus_linear | +0.71 | [+0.02, +1.38] | ↑ |
| csdi_minus_linear | +1.01 | [+0.36, +1.64] | ↑ |
