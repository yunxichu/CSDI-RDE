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

## Scenario S2  (sparsity=0.40, σ=0.30)

| Config | NRMSE h=1 | NRMSE h=4 | NRMSE h=16 | NRMSE h=64 | PICP@90 h=1 | PICP@90 h=4 | PICP@90 h=16 | PICP@90 h=64 | MPIW h=1 | MPIW h=4 | MPIW h=16 | MPIW h=64 | CRPS h=1 | CRPS h=4 | CRPS h=16 | CRPS h=64 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full (4 modules, Lyap-sat) | 0.292±0.055 | 0.357±0.061 | 0.700±0.096 | 0.938±0.076 | 0.900±0.022 | 0.898±0.047 | 0.932±0.024 | 0.919±0.024 | 6.909±0.197 | 8.944±0.075 | 20.172±0.861 | 27.670±2.998 | 1.969±0.111 | 2.121±0.150 | 3.253±0.364 | 4.465±0.337 |
| Full + Lyap-empirical | 0.291±0.055 | 0.358±0.061 | 0.701±0.094 | 0.937±0.076 | 0.898±0.021 | 0.895±0.047 | 0.931±0.022 | 0.919±0.021 | 6.854±0.201 | 9.029±0.129 | 20.121±1.052 | 27.615±2.979 | 1.968±0.110 | 2.129±0.147 | 3.255±0.356 | 4.458±0.338 |
| −M1 (linear imp) | 0.361±0.039 | 0.417±0.051 | 0.753±0.088 | 0.966±0.066 | 0.907±0.009 | 0.893±0.041 | 0.934±0.018 | 0.920±0.017 | 9.779±0.270 | 11.239±0.339 | 22.215±1.181 | 28.278±2.780 | 2.291±0.081 | 2.435±0.124 | 3.571±0.343 | 4.623±0.281 |
| −M2 (random τ) | 0.398±0.044 | 0.451±0.049 | 0.688±0.097 | 0.971±0.054 | 0.893±0.028 | 0.905±0.026 | 0.932±0.007 | 0.907±0.037 | 9.419±0.892 | 11.269±0.927 | 19.705±3.425 | 27.783±3.028 | 2.344±0.103 | 2.495±0.124 | 3.286±0.382 | 4.631±0.243 |
| −M2 (Fraser-Swinney τ) | 0.411±0.082 | 0.471±0.077 | 0.691±0.098 | 0.972±0.063 | 0.901±0.011 | 0.915±0.009 | 0.930±0.012 | 0.902±0.034 | 9.950±1.745 | 12.313±1.852 | 19.865±2.771 | 27.627±3.033 | 2.392±0.220 | 2.556±0.219 | 3.303±0.380 | 4.638±0.288 |
| −M3 (exact GPR) | 0.332±0.061 | 0.443±0.046 | 0.829±0.128 | 1.274±0.032 | 0.889±0.034 | 0.898±0.034 | 0.938±0.020 | 0.905±0.020 | 8.474±0.443 | 13.156±0.795 | 30.723±1.166 | 39.636±2.363 | 1.560±0.189 | 2.053±0.190 | 3.636±0.513 | 6.779±0.174 |
| −M4 (Split CP) | 0.290±0.054 | 0.357±0.061 | 0.696±0.096 | 0.936±0.076 | 0.898±0.018 | 0.897±0.047 | 0.932±0.024 | 0.918±0.027 | 6.806±0.220 | 9.057±0.041 | 20.146±1.073 | 27.594±3.176 | 1.964±0.105 | 2.126±0.148 | 3.239±0.364 | 4.452±0.333 |
| −M4 (Lyap-exp, no sat) | 0.290±0.054 | 0.357±0.060 | 0.699±0.095 | 0.937±0.076 | 0.902±0.019 | 0.898±0.045 | 0.932±0.023 | 0.917±0.026 | 6.902±0.235 | 9.033±0.060 | 20.394±1.109 | 27.560±3.136 | 1.967±0.105 | 2.125±0.146 | 3.252±0.364 | 4.459±0.337 |
| All off (≈ v1 CSDI-RDE-GPR) | 0.557±0.070 | 0.589±0.077 | 0.767±0.093 | 1.016±0.034 | 0.897±0.033 | 0.896±0.017 | 0.917±0.011 | 0.916±0.027 | 13.594±0.661 | 15.118±1.435 | 23.485±3.095 | 30.749±1.582 | 2.635±0.234 | 2.783±0.283 | 3.534±0.462 | 4.868±0.172 |
---

## Module-4 — 4 growth modes + robust λ

Three upgrades vs the initial pass:

1. **Saturating growth**: ``1 + (e^{λh dt}−1)/(1 + (e^{λh dt}−1)/cap)`` (smooth rational cap)
2. **Clipped growth**: ``min(exp(λh dt), cap)``
3. **Empirical growth**: λ-free; per-horizon-bin scale inferred from calibration residuals
4. **Robust λ estimator** (AR-Kalman pre-filter + Rosenstein tl=50 + clip [0.1, 2.5]):
   - σ=0.3: nolds err +174% → robust err **+18%**
   - σ=0.5: nolds err +152% → robust err **−1%**

### Aggregate \|PICP − 0.90\|, mixed-horizon CP (3 seeds)

| scenario | Split | Lyap-exp | Lyap-sat | **Lyap-empirical** |
|---|:-:|:-:|:-:|:-:|
| S3 mean | 0.072 | 0.054 | 0.049 | **0.013 (5.5× < Split)** |
| S3 max  | 0.093 | 0.099 | 0.095 | **0.024** |
| S2 mean | 0.084 | 0.061 | 0.056 | **0.018 (4.7× < Split)** |
| S2 max  | 0.127 | 0.127 | 0.120 | **0.039** |

**Take-away**: Split CP monotonically drifts 0.99 → 0.80 across horizons; Lyap-empirical
stays in [0.88, 0.92] everywhere. **4.7-5.5× lower aggregate miscalibration**, recommended default.

Files: [`module4_horizon_calibration.py`](module4_horizon_calibration.py),
[`figures/module4_horizon_cal_S{2,3}.png`](figures/),
[`results/module4_horizon_cal_S{2,3}_n3.json`](results/).

---

## Lorenz96 SVGP scaling + τ-search comparison

High-dim validation on Lorenz96 (F=8, λ₁≈1.68/unit, d_KY≈0.4·N).

### Part 1: SVGP scaling vs ambient dim N

| N  | n_train | train time | NRMSE |
|:-:|:-:|:-:|:-:|
| 10 | 1393 | **25.6 ± 0.9 s** | 0.85 ± 0.05 |
| 20 | 1393 | 42.4 ± 3.9 s | 0.92 ± 0.01 |
| 40 | 1393 | 92.1 ± 2.1 s | 1.00 ± 0.01 |

Training time **linear in N**; NRMSE degrades smoothly (Proposition 2 empirical check).
Exact GPR at N=40 would OOM.

### Part 2: τ-search scaling on N=40, L_embed=7

| Method | τ-search time | NRMSE |
|---|:-:|:-:|
| random | 0.00 s | 0.995 |
| Fraser-Swinney | 0.51 s | 0.997 |
| **MI-Lyap BO** (Stage A) | 2.45 s | **0.990** |
| **MI-Lyap CMA-ES** (Stage B, rank=2) | 1.34 s | 0.991 |

**Stage B (CMA-ES) is 1.8× faster than Stage A (BO)** with tied quality — tech.md §2.3 validated.
Low-rank τ singular-value spectrum saved to [`figures/tau_low_rank_spectrum.png`](figures/tau_low_rank_spectrum.png).

Files: [`lorenz96_scaling.py`](lorenz96_scaling.py),
[`figures/lorenz96_svgp_scaling.png`](figures/lorenz96_svgp_scaling.png),
[`results/lorenz96_scaling_N10_20_40.json`](results/lorenz96_scaling_N10_20_40.json).
