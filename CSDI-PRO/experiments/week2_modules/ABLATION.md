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
| Full (all 4 modules) | 0.373±0.026 | 0.493±0.047 | 0.787±0.068 | 0.946±0.049 | 0.880±0.028 | 0.888±0.035 | 0.911±0.020 | 0.876±0.015 | 8.858±0.539 | 13.161±0.236 | 21.933±0.384 | 24.776±1.150 | 2.169±0.092 | 2.563±0.179 | 3.727±0.311 | 4.538±0.250 |
| −M1 (linear imp) | 0.481±0.019 | 0.621±0.035 | 0.925±0.045 | 1.092±0.045 | 0.898±0.025 | 0.890±0.034 | 0.910±0.015 | 0.895±0.018 | 13.130±1.316 | 17.174±0.894 | 26.239±0.575 | 30.080±1.723 | 2.732±0.050 | 3.209±0.129 | 4.431±0.188 | 5.261±0.216 |
| −M2 (random τ) | 0.477±0.030 | 0.566±0.044 | 0.742±0.075 | 0.931±0.046 | 0.866±0.039 | 0.882±0.040 | 0.919±0.005 | 0.885±0.025 | 11.749±1.286 | 14.576±1.243 | 21.230±2.613 | 24.677±2.591 | 2.578±0.112 | 2.886±0.178 | 3.554±0.340 | 4.454±0.224 |
| −M2 (Fraser-Swinney τ) | 0.491±0.047 | 0.567±0.054 | 0.751±0.063 | 0.932±0.048 | 0.882±0.045 | 0.896±0.027 | 0.923±0.003 | 0.880±0.031 | 12.410±0.874 | 15.236±1.258 | 21.710±2.364 | 24.773±3.298 | 2.629±0.176 | 2.890±0.210 | 3.583±0.291 | 4.449±0.238 |
| −M3 (exact GPR) | 0.463±0.044 | 0.600±0.065 | 0.919±0.060 | 1.189±0.080 | 0.872±0.025 | 0.903±0.017 | 0.919±0.015 | 0.869±0.013 | 10.848±0.349 | 17.540±1.191 | 32.827±3.677 | 32.965±1.447 | 2.112±0.152 | 2.854±0.303 | 4.482±0.346 | 6.170±0.543 |
| −M4 (Split CP) | 0.372±0.027 | 0.494±0.048 | 0.786±0.071 | 0.946±0.051 | 0.884±0.028 | 0.887±0.038 | 0.913±0.020 | 0.878±0.013 | 8.924±0.545 | 13.058±0.264 | 22.037±0.385 | 24.984±1.286 | 2.165±0.094 | 2.565±0.180 | 3.726±0.324 | 4.538±0.262 |
| All off (v1 CSDI-RDE-GPR-ish) | 0.760±0.052 | 0.818±0.055 | 0.900±0.047 | 1.087±0.043 | 0.886±0.049 | 0.890±0.026 | 0.925±0.004 | 0.887±0.017 | 20.404±1.666 | 22.227±1.240 | 26.992±1.809 | 29.714±2.350 | 3.652±0.199 | 3.929±0.252 | 4.302±0.231 | 5.256±0.207 |

## Scenario S2  (sparsity=0.40, σ=0.30)

| Config | NRMSE h=1 | NRMSE h=4 | NRMSE h=16 | NRMSE h=64 | PICP@90 h=1 | PICP@90 h=4 | PICP@90 h=16 | PICP@90 h=64 | MPIW h=1 | MPIW h=4 | MPIW h=16 | MPIW h=64 | CRPS h=1 | CRPS h=4 | CRPS h=16 | CRPS h=64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full (all 4 modules) | 0.289±0.052 | 0.356±0.056 | 0.701±0.099 | 0.939±0.078 | 0.906±0.024 | 0.895±0.048 | 0.929±0.026 | 0.918±0.026 | 6.934±0.047 | 8.943±0.201 | 20.088±0.815 | 27.685±3.145 | 1.963±0.099 | 2.116±0.132 | 3.257±0.375 | 4.468±0.347 |
| −M1 (linear imp) | 0.361±0.039 | 0.417±0.050 | 0.752±0.088 | 0.966±0.065 | 0.906±0.008 | 0.897±0.042 | 0.935±0.018 | 0.922±0.017 | 9.908±0.383 | 11.291±0.312 | 22.264±1.173 | 28.291±2.596 | 2.290±0.079 | 2.433±0.125 | 3.568±0.343 | 4.625±0.275 |
| −M2 (random τ) | 0.399±0.041 | 0.454±0.050 | 0.687±0.094 | 0.967±0.052 | 0.892±0.029 | 0.906±0.026 | 0.931±0.008 | 0.905±0.036 | 9.366±0.935 | 11.306±0.897 | 19.705±3.155 | 27.561±3.108 | 2.346±0.098 | 2.502±0.126 | 3.279±0.371 | 4.607±0.232 |
| −M2 (Fraser-Swinney τ) | 0.412±0.080 | 0.466±0.078 | 0.691±0.097 | 0.978±0.060 | 0.907±0.014 | 0.916±0.010 | 0.931±0.013 | 0.899±0.041 | 10.092±1.423 | 12.280±1.821 | 19.879±2.530 | 27.576±3.262 | 2.393±0.213 | 2.546±0.212 | 3.302±0.374 | 4.662±0.277 |
| −M3 (exact GPR) | 0.323±0.050 | 0.437±0.037 | 0.832±0.130 | 1.273±0.032 | 0.893±0.028 | 0.905±0.025 | 0.935±0.023 | 0.906±0.022 | 8.411±0.440 | 13.091±0.839 | 30.986±1.085 | 40.355±3.377 | 1.524±0.149 | 2.019±0.145 | 3.676±0.548 | 6.824±0.226 |
| −M4 (Split CP) | 0.291±0.052 | 0.354±0.057 | 0.701±0.100 | 0.939±0.078 | 0.901±0.021 | 0.894±0.044 | 0.931±0.025 | 0.919±0.026 | 6.886±0.092 | 8.885±0.133 | 20.142±0.996 | 27.751±3.198 | 1.961±0.100 | 2.111±0.133 | 3.257±0.379 | 4.468±0.347 |
| All off (v1 CSDI-RDE-GPR-ish) | 0.557±0.070 | 0.589±0.077 | 0.767±0.093 | 1.016±0.034 | 0.897±0.033 | 0.896±0.017 | 0.917±0.011 | 0.916±0.027 | 13.594±0.661 | 15.118±1.435 | 23.485±3.095 | 30.749±1.582 | 2.635±0.234 | 2.783±0.283 | 3.534±0.462 | 4.868±0.172 |
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

1. **Split CP monotonically drifts** 0.99 → 0.80 (textbook exchangeability violation under
   growing chaotic residuals).
2. **Saturating/clipped** shapes the pure-exp Lyap-CP at long horizons but only
   marginally improves aggregate metrics (~5-10% reduction in mean deviation).
3. **Lyap-empirical** — the **λ-free, data-driven growth** — achieves **4.7-5.5×
   lower aggregate miscalibration** than Split CP, with all per-horizon PICP
   within 0.04 of the 0.90 target. This is the recommended default.
4. Robust λ estimator brings nolds's 170% over-estimate under noise down to ~20%,
   making exp/saturating/clipped modes usable in real pipelines.

### Files

- code: [`module4_horizon_calibration.py`](module4_horizon_calibration.py) (4-mode sweep)
- λ estimator: [`methods/mi_lyap.py::robust_lyapunov`](../../methods/mi_lyap.py)
- growth functions: [`methods/lyap_conformal.py::lyap_growth`](../../methods/lyap_conformal.py)
- figures: [`figures/module4_horizon_cal_S{2,3}.png`](figures/) (5-curve overlay)
- data: [`results/module4_horizon_cal_S{2,3}_n3.json`](results/)
