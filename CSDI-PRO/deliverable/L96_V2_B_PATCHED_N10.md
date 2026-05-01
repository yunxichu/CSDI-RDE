# L96 N=20 v2 B patched n=10 summary

Merged files:
- `experiments/week1/results/pt_l96_smoke_l96N20_v2_B_patched_5seed.json`
- `experiments/week1/results/pt_l96_smoke_l96N20_v2_B_patched_seed5_9.json`

## Summary

| Scenario | Cell | mean | median | Pr>0.5 | Pr>1.0 | n |
|---|---|---:|---:|---:|---:|---:|
| SP55 | panda_linear | 1.39 | 0.63 | 80% | 30% | 10 |
| SP55 | panda_csdi | 1.99 | 1.26 | 100% | 60% | 10 |
| SP55 | deepedm_linear | 0.45 | 0.34 | 40% | 0% | 10 |
| SP55 | deepedm_csdi | 0.70 | 0.63 | 90% | 20% | 10 |
| SP65 | panda_linear | 1.39 | 0.71 | 70% | 30% | 10 |
| SP65 | panda_csdi | 1.99 | 1.26 | 100% | 70% | 10 |
| SP65 | deepedm_linear | 0.38 | 0.42 | 40% | 0% | 10 |
| SP65 | deepedm_csdi | 0.84 | 0.88 | 100% | 50% | 10 |
| SP75 | panda_linear | 1.20 | 0.42 | 50% | 20% | 10 |
| SP75 | panda_csdi | 1.97 | 1.26 | 100% | 60% | 10 |
| SP75 | deepedm_linear | 0.24 | 0.17 | 20% | 0% | 10 |
| SP75 | deepedm_csdi | 0.76 | 0.71 | 90% | 30% | 10 |
| SP82 | panda_linear | 1.79 | 0.50 | 60% | 30% | 10 |
| SP82 | panda_csdi | 1.71 | 1.05 | 100% | 60% | 10 |
| SP82 | deepedm_linear | 0.41 | 0.34 | 30% | 10% | 10 |
| SP82 | deepedm_csdi | 0.84 | 0.84 | 90% | 40% | 10 |
| NO010 | panda_linear | 1.77 | 0.88 | 100% | 40% | 10 |
| NO010 | panda_csdi | 1.77 | 0.88 | 100% | 40% | 10 |
| NO010 | deepedm_linear | 0.50 | 0.42 | 40% | 0% | 10 |
| NO010 | deepedm_csdi | 0.52 | 0.42 | 40% | 10% | 10 |
| NO020 | panda_linear | 1.63 | 0.97 | 100% | 50% | 10 |
| NO020 | panda_csdi | 1.63 | 0.97 | 100% | 50% | 10 |
| NO020 | deepedm_linear | 0.42 | 0.34 | 20% | 0% | 10 |
| NO020 | deepedm_csdi | 0.60 | 0.46 | 50% | 10% | 10 |
| NO050 | panda_linear | 1.68 | 0.92 | 100% | 40% | 10 |
| NO050 | panda_csdi | 1.58 | 0.84 | 100% | 40% | 10 |
| NO050 | deepedm_linear | 0.35 | 0.25 | 20% | 0% | 10 |
| NO050 | deepedm_csdi | 0.45 | 0.34 | 10% | 10% | 10 |

## Paired CSDI minus linear

| Scenario | Panda Δ | Panda CI | DeepEDM Δ | DeepEDM CI |
|---|---:|---:|---:|---:|
| SP55 | +0.60 | [+0.12,+1.18] | +0.24 | [+0.15,+0.33] |
| SP65 | +0.60 | [+0.18,+1.08] | +0.46 | [+0.25,+0.67] |
| SP75 | +0.77 | [+0.22,+1.43] | +0.51 | [+0.27,+0.74] |
| SP82 | -0.08 | [-1.43,+0.74] | +0.43 | [+0.29,+0.57] |
| NO010 | +0.00 | [+0.00,+0.00] | +0.02 | [-0.18,+0.29] |
| NO020 | +0.00 | [+0.00,+0.00] | +0.18 | [+0.03,+0.34] |
| NO050 | -0.10 | [-0.30,+0.04] | +0.09 | [-0.13,+0.34] |

## Interpretation

- Panda mean remains high-variance because seeds 2 and 4 produce long linear forecasts.
- CSDI improves Panda median and survival on the sparsity axis, especially SP75/SP82.
- DeepEDM+CSDI has cleaner paired gains on SP65/SP75/SP82 than Panda mean.
- Pure-noise Panda remains tied between linear and CSDI, supporting the gap-imputation framing.