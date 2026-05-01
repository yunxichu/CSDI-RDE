# L96 N=20 v2 Cross-System Replication — 2026-04-30

Purpose: check whether the L63 Figure-1 sparsity frontier story transfers to
high-dimensional Lorenz96 N=20 under the same v2 corruption grid.

Protocol:
- system: Lorenz96 N=20, F=8, dt=0.05
- cells: `panda_linear`, `panda_csdi`, `deepedm_linear`, `deepedm_csdi`
- configs: `SP55`, `SP65`, `SP75`, `SP82`, `NO010`, `NO020`, `NO050`
- seeds: 5
- corruption seed: deterministic v2 grid index in `smoke_l96N20_v2.py`
- CSDI checkpoint: `dyn_csdi_l96_full_c192_vales_best.pt`

Files:
- `experiments/week1/results/pt_l96_smoke_l96N20_v2_B_s55_s65_5seed.json`
- `experiments/week1/results/pt_l96_smoke_l96N20_v2_B_s75_s82_5seed.json`
- `experiments/week1/results/pt_l96_smoke_l96N20_v2_B_no010_no020_5seed.json`
- `experiments/week1/results/pt_l96_smoke_l96N20_v2_B_no050_5seed.json`

## Sparsity Axis

| Scenario | Cell | Mean VPT@1.0 | Median | Pr>0.5 | Pr>1.0 |
|---|---|---:|---:|---:|---:|
| SP55 | Panda linear | 2.30 | 0.76 | 100% | 40% |
| SP55 | Panda CSDI | 2.49 | 1.76 | 100% | 60% |
| SP55 | DeepEDM linear | 0.47 | 0.34 | 40% | 0% |
| SP55 | DeepEDM CSDI | 0.97 | 1.01 | 100% | 60% |
| SP65 | Panda linear | 2.18 | 0.76 | 80% | 40% |
| SP65 | Panda CSDI | 2.49 | 1.76 | 100% | 60% |
| SP65 | DeepEDM linear | 0.49 | 0.50 | 60% | 0% |
| SP65 | DeepEDM CSDI | 1.04 | 0.84 | 100% | 40% |
| SP75 | Panda linear | 2.18 | 0.76 | 80% | 40% |
| SP75 | Panda CSDI | 2.49 | 1.68 | 100% | 60% |
| SP75 | DeepEDM linear | 0.45 | 0.42 | 40% | 0% |
| SP75 | DeepEDM CSDI | 0.79 | 0.84 | 100% | 20% |
| SP82 | Panda linear | 3.04 | 0.50 | 60% | 40% |
| SP82 | Panda CSDI | 3.43 | 0.92 | 100% | 40% |
| SP82 | DeepEDM linear | 0.42 | 0.50 | 60% | 0% |
| SP82 | DeepEDM CSDI | 1.24 | 0.92 | 80% | 40% |

Paired-bootstrap CSDI minus linear:

| Scenario | Panda Δ | Panda CI | DeepEDM Δ | DeepEDM CI |
|---|---:|---:|---:|---:|
| SP55 | +0.18 | [-0.15,+0.71] | +0.50 | [+0.25,+0.71] |
| SP65 | +0.30 | [-0.07,+0.96] | +0.55 | [+0.13,+1.09] |
| SP75 | +0.30 | [-0.10,+0.96] | +0.34 | [+0.07,+0.72] |
| SP82 | +0.39 | [+0.12,+0.62] | +0.82 | [+0.05,+1.65] |

Interpretation:
- L96 Panda shows a weaker, noisier version of the L63 sparsity rescue.
  The mean paired CI is strictly positive only at SP82, but median and
  survival improve across SP55-SP82.
- DeepEDM shows a cleaner CSDI benefit across the sparsity line. This supports
  keeping delay-manifold forecasting as a real companion rather than an
  appendix-only method.
- Because Panda has occasional long lucky forecasts at high sparsity, L96
  should be presented as a 5-seed cross-system replication/inset, not as the
  primary Figure-1 evidence.

## Pure-Noise Axis

| Scenario | Panda linear | Panda CSDI | DeepEDM linear | DeepEDM CSDI |
|---|---:|---:|---:|---:|
| NO010 | 2.30 | 2.30 | 0.86 | 0.47 |
| NO020 | 2.27 | 2.27 | 0.82 | 0.52 |
| NO050 | 2.44 | 2.23 | 0.29 | 0.49 |

Interpretation:
- Panda CSDI is exactly tied with Panda linear at NO010/NO020 and slightly
  lower at NO050. This independently confirms the L63 conclusion: CSDI is a
  sparsity/gap-imputation lever, not a dense-noise denoiser.
- DeepEDM pure-noise behavior is mixed and should not be used as a headline.

## Paper Use

Use L96 N=20 v2 as:
1. A cross-system confirmation that CSDI helps on the sparsity axis.
2. Stronger support for DeepEDM as a complementary route.
3. A second-system confirmation that pure noise is not where CSDI rescues Panda.

Do not claim that L96 reproduces the L63 Figure-1 curve as cleanly as L63.
