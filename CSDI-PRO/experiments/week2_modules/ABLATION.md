# Week 2 Ablation — 4 modules on Lorenz63

**Pipeline** (tech.md §Core): *observations → M1 imputation → M2 τ-select → M3 GP regress → M4 conformal.*

Each row flips **one** module relative to the full pipeline. 3 seeds each; delay embedding L=5, τ_max=30.


## Module surrogates used in this ablation

| Module | Full | −Mk variant |
|---|---|---|
| **M1** Dynamics-Aware imputation | AR-Kalman smoother (AR(5) + RTS on observed subset) | linear interpolation (Week 1 baseline) |
| **M2** Delay-embedding τ selection | MI-Lyap via BayesOpt (20 calls) with cumulative-δ param | random τ (Takahashi 2021), Fraser-Swinney first-minimum |
| **M3** Regression | GPyTorch SVGP Matern-5/2, m=128 inducing, 120 epochs | self-implemented exact GPR (n≤1000), no hyperparam opt |
| **M4** Prediction interval | Lyap-Conformal with λ∈{est, true} | vanilla Split-Conformal |

**Note on M1**: the "full" M1 is Dynamics-Aware CSDI (Transformer + noise conditioning + dynamic delay mask, tech.md §1.2) whose training takes hours of diffusion model work. Week 2 uses an AR-Kalman stand-in that captures the load-bearing ideas (model-based + noise-aware smoother). The full CSDI re-train is Week-7 work.

**Note on M4**: per-horizon calibration (below) makes Lyap-CP ≈ Split-CP numerically. The real Lyap-CP advantage appears under **mixed-horizon calibration** — see `module4_horizon_calibration.py` for that focused experiment.


## Scenario S3  (sparsity=0.60, σ=0.50)

| Config | NRMSE h=1 | NRMSE h=4 | NRMSE h=16 | NRMSE h=64 | PICP@90 h=1 | PICP@90 h=4 | PICP@90 h=16 | PICP@90 h=64 | MPIW h=1 | MPIW h=4 | MPIW h=16 | MPIW h=64 | CRPS h=1 | CRPS h=4 | CRPS h=16 | CRPS h=64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full (4 modules, Lyap-sat) | 0.373±0.028 | 0.492±0.048 | 0.788±0.071 | 0.947±0.052 | 0.882±0.025 | 0.886±0.036 | 0.909±0.022 | 0.876±0.019 | 8.926±0.533 | 13.065±0.255 | 21.921±0.324 | 24.995±1.153 | 2.174±0.102 | 2.559±0.182 | 3.733±0.325 | 4.546±0.265 |
| Full + Lyap-empirical | 0.372±0.026 | 0.493±0.048 | 0.788±0.069 | 0.946±0.051 | 0.882±0.028 | 0.884±0.037 | 0.914±0.016 | 0.874±0.019 | 8.858±0.629 | 13.067±0.236 | 22.079±0.511 | 24.839±1.168 | 2.164±0.092 | 2.561±0.183 | 3.733±0.315 | 4.538±0.262 |
| −M1 (linear imp) | 0.480±0.019 | 0.623±0.034 | 0.925±0.044 | 1.093±0.045 | 0.896±0.028 | 0.890±0.030 | 0.910±0.014 | 0.894±0.015 | 13.076±1.414 | 17.327±0.837 | 26.223±0.697 | 30.010±1.682 | 2.729±0.050 | 3.214±0.124 | 4.434±0.186 | 5.264±0.217 |
| −M2 (random τ) | 0.476±0.032 | 0.564±0.041 | 0.744±0.073 | 0.936±0.047 | 0.865±0.042 | 0.886±0.036 | 0.920±0.010 | 0.884±0.024 | 11.652±1.294 | 14.711±1.084 | 21.295±2.468 | 24.650±2.594 | 2.578±0.114 | 2.878±0.166 | 3.562±0.335 | 4.475±0.228 |
| −M2 (Fraser-Swinney τ) | 0.487±0.049 | 0.569±0.053 | 0.751±0.065 | 0.932±0.049 | 0.882±0.047 | 0.894±0.026 | 0.921±0.004 | 0.880±0.031 | 12.370±1.037 | 15.109±1.240 | 21.731±2.398 | 24.809±3.317 | 2.618±0.181 | 2.894±0.210 | 3.585±0.301 | 4.450±0.245 |
| −M3 (exact GPR) | 0.463±0.044 | 0.600±0.065 | 0.919±0.060 | 1.189±0.080 | 0.872±0.025 | 0.903±0.017 | 0.919±0.015 | 0.869±0.013 | 10.848±0.349 | 17.540±1.191 | 32.827±3.677 | 32.965±1.447 | 2.112±0.152 | 2.854±0.303 | 4.482±0.346 | 6.170±0.543 |
| −M4 (Split CP) | 0.373±0.027 | 0.492±0.045 | 0.786±0.069 | 0.947±0.050 | 0.879±0.027 | 0.882±0.035 | 0.913±0.018 | 0.873±0.015 | 8.802±0.513 | 13.018±0.165 | 22.118±0.383 | 24.817±1.295 | 2.169±0.093 | 2.559±0.174 | 3.727±0.314 | 4.542±0.256 |
| −M4 (Lyap-exp, no sat) | 0.374±0.028 | 0.492±0.047 | 0.786±0.069 | 0.944±0.053 | 0.880±0.031 | 0.884±0.038 | 0.913±0.016 | 0.879±0.014 | 8.901±0.583 | 13.050±0.224 | 22.217±0.505 | 24.922±1.390 | 2.172±0.100 | 2.558±0.182 | 3.728±0.317 | 4.530±0.267 |
| All off (≈ v1 CSDI-RDE-GPR) | 0.760±0.052 | 0.818±0.055 | 0.900±0.047 | 1.087±0.043 | 0.886±0.049 | 0.890±0.026 | 0.925±0.004 | 0.887±0.017 | 20.404±1.666 | 22.227±1.240 | 26.992±1.809 | 29.714±2.350 | 3.652±0.199 | 3.929±0.252 | 4.302±0.231 | 5.256±0.207 |
---

## Module-4 — 4 growth modes + robust λ

Iteration of the mixed-horizon CP study. Three upgrades vs the initial pass:

1. **Saturating growth**: ``1 + (e^{λh dt}−1)/(1 + (e^{λh dt}−1)/cap)``
   (smooth rational cap at ``cap``; default cap=10)
2. **Clipped growth**: hard ``min(exp(λh dt), cap)``
3. **Empirical growth**: no λ; per-horizon-bin scale inferred from calibration residuals
4. **Robust λ estimator** (pre-filter with AR-Kalman + Rosenstein ``tl=50`` + clip to ``[0.1, 2.5]``):
   - σ=0.3: nolds err +174% → robust err **+18%**
   - σ=0.5: nolds err +152% → robust err **−1%**

### S3 (sparsity=0.60, σ=0.50), 3 seeds, λ_est ≈ 1.11 (+22% vs true 0.906)

Per-horizon PICP (target 0.90):

| h  | Split | Lyap-exp | Lyap-sat | Lyap-clipped | **Lyap-empirical** |
|:-:|:-:|:-:|:-:|:-:|:-:|
|  1 | 0.988 | 0.937 | 0.942 | 0.937 | **0.876** |
|  2 | 0.991 | 0.936 | 0.941 | 0.936 | **0.890** |
|  4 | 0.977 | 0.885 | 0.897 | 0.885 | **0.882** |
|  8 | 0.925 | 0.822 | 0.833 | 0.822 | **0.895** |
| 16 | 0.847 | 0.801 | 0.805 | 0.801 | **0.913** |
| 24 | 0.822 | 0.866 | 0.861 | 0.866 | **0.891** |
| 32 | 0.807 | 0.936 | 0.919 | 0.936 | **0.920** |
| 48 | 0.827 | 0.996 | 0.987 | 0.996 | **0.904** |

Aggregate \|PICP − 0.90\|:

| scenario | Split | Lyap-exp | Lyap-sat | **Lyap-empirical** |
|---|:-:|:-:|:-:|:-:|
| S3 mean | 0.072 | 0.054 | 0.049 | **0.013 (5.5× < Split)** |
| S3 max  | 0.093 | 0.099 | 0.095 | **0.024** |
| S2 mean | 0.084 | 0.061 | 0.056 | **0.018 (4.7× < Split)** |
| S2 max  | 0.127 | 0.127 | 0.120 | **0.039** |

### Take-aways

1. **Split CP monotonically drifts** 0.99 → 0.80 (textbook exchangeability violation under growing chaotic residuals).
2. **Saturating/clipped** only marginally improves over exp (~5-10% reduction).
3. **Lyap-empirical** — λ-free, data-driven growth — **4.7-5.5× lower aggregate miscalibration** than Split CP, worst-horizon deviation ≤ 0.04. Recommended default.
4. Robust λ estimator brings nolds's 170% over-estimate down to ~20%, making exp/sat usable.

### Files

- code: [`module4_horizon_calibration.py`](module4_horizon_calibration.py)
- λ estimator: [`methods/mi_lyap.py::robust_lyapunov`](../../methods/mi_lyap.py)
- growth fn: [`methods/lyap_conformal.py::lyap_growth`](../../methods/lyap_conformal.py)
- figures: [`figures/module4_horizon_cal_S{2,3}.png`](figures/)
- data: [`results/module4_horizon_cal_S{2,3}_n3.json`](results/)

---

## Lorenz96 SVGP scaling + τ-search comparison

High-dim validation on Lorenz96 (F=8, λ₁ ≈ 1.68 / unit, d_KY ≈ 0.4·N).

### Part 1: SVGP scaling vs ambient dim N

| N  | n_train | SVGP train time | NRMSE |
|:-:|:-:|:-:|:-:|
| 10 | 1393 | **25.6 ± 0.9 s** | 0.847 ± 0.045 |
| 20 | 1393 | 42.4 ± 3.9 s | 0.916 ± 0.014 |
| 40 | 1393 | 92.1 ± 2.1 s | 1.001 ± 0.013 |

**Finding**: SVGP training time scales **linearly in N** (MultiOutputSVGP trains one
GP per output dim). NRMSE degrades smoothly from 0.85 (N=10) to 1.00 (N=40), consistent
with tech.md Proposition 2 (rate ∝ n^{−ν/(2ν + d_KY)}, d_KY ∝ N).

Exact GPR by contrast would cost O(N³ × N) per fit and OOM at N=20+; our 92 s at N=40
is directly enabled by SVGP.

### Part 2: τ-search scaling on N=40, L_embed=7 (six lags)

2 seeds, each method evaluated by downstream SVGP NRMSE on test split:

| Method | τ-search time | NRMSE | note |
|---|:-:|:-:|---|
| random (Takahashi 2021) | **0.00 s** | 0.995 | reference baseline |
| Fraser-Swinney | 0.51 s | 0.997 | first-minimum of MI |
| **MI-Lyap BO** (Stage A) | 2.45 s | **0.990** | best |
| **MI-Lyap CMA-ES** (Stage B, rank=2) | 1.34 s | 0.991 | 1.8× faster than BO |

**Findings**:
1. At L=7 (6-dim search space), MI-Lyap methods improve NRMSE by ~0.4-0.7% over random / F-S.
   Modest at this dimensionality — **d_KY ≈ 16 is already larger than L**, so τ selection
   is not the bottleneck here (regressor capacity is).
2. **Stage B (low-rank CMA-ES) is 1.8× faster than Stage A (BO)** with essentially tied
   quality, validating the tech.md §2.3 design. At larger L (e.g., L=15 for Lorenz96 N=100),
   this gap should grow exponentially since BO scales O(L!) worst-case while CMA-ES scales
   O(r(L+1)) = O(L).
3. τ matrix singular-value spectrum (rank-2 CMA-ES solution) saved to
   [`figures/tau_low_rank_spectrum.png`](figures/tau_low_rank_spectrum.png) — the
   "coupled-oscillator timescales" Figure 7 candidate.

### Files

- code: [`lorenz96_scaling.py`](lorenz96_scaling.py) + [`../../experiments/week1/lorenz96_utils.py`](../../experiments/week1/lorenz96_utils.py)
- Lorenz96 integrator: [`../week1/lorenz96_utils.py::integrate_lorenz96`](../week1/lorenz96_utils.py)
- CMA-ES τ selector: [`../../methods/mi_lyap.py::mi_lyap_cmaes_tau`](../../methods/mi_lyap.py)
- figures: [`figures/lorenz96_svgp_scaling.png`](figures/lorenz96_svgp_scaling.png), [`figures/tau_low_rank_spectrum.png`](figures/tau_low_rank_spectrum.png)
- data: [`results/lorenz96_scaling_N10_20_40.json`](results/lorenz96_scaling_N10_20_40.json)
