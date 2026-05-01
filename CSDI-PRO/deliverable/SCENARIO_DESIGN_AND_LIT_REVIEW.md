# Scenario Design + Literature Review

Date: 2026-04-26

This note answers one question: **how should we design sparse/noisy corruption so the paper is reviewer-safe rather than an arbitrary S0-S6 stress test?**

## 1. Current Implementation

The current corruption path is defined in `experiments/week1/lorenz63_utils.py`:

```python
mask = rng.random(traj.shape[0]) > sparsity
sigma = noise_std_frac * attractor_std
observed = traj + rng.normal(scale=sigma, size=traj.shape)
observed[~mask] = np.nan
```

Current S stages:

| Scenario | sparsity s | keep p | noise sigma / attr_std |
|---|---:|---:|---:|
| S0 | 0.00 | 1.00 | 0.00 |
| S1 | 0.20 | 0.80 | 0.10 |
| S2 | 0.40 | 0.60 | 0.30 |
| S3 | 0.60 | 0.40 | 0.50 |
| S4 | 0.75 | 0.25 | 0.80 |
| S5 | 0.90 | 0.10 | 1.20 |
| S6 | 0.95 | 0.05 | 1.50 |

Important details:

- The mask is **time-level**, not variable-level: when a timestep is missing, all dimensions are missing. This simulates synchronized full-vector sensor dropout, not asynchronous sensor dropout.
- The missing pattern is **iid Bernoulli**. It produces mostly short gaps at moderate sparsity, then very long gaps only near S5/S6.
- Noise is additive Gaussian observation noise. It is scaled by a scalar attractor std, not per-coordinate std.
- S0-S6 jointly increase sparsity and noise. That gives a good headline harshness path, but it confounds two mechanisms.

## 2. Main Concern

The current S stages are too coarse and too diagonal.

They are fine for a first figure saying "something phase-transition-like happens," but not enough for a mechanism paper. A skeptical reviewer can say:

1. You changed `s` and `sigma` simultaneously, so the transition may be arbitrary.
2. iid point missingness is not representative of sensor dropout or irregular sampling.
3. S5/S6 use huge observation noise; any method can fail there, so survival may be a floor effect.
4. The same S stages are reused across systems, but systems have different Lyapunov time, dt, dimension, and patch geometry.
5. A time-synchronous mask is easier/harder than realistic asynchronous missingness depending on the forecaster.

The fix is not to delete S0-S6. The fix is to demote them to a **summary path** and add a mechanistic design underneath.

## 3. What The Literature Suggests

### 3.1 Missing time-series imputation

The imputation literature is mature. BRITS frames missing values as learnable variables in a bidirectional recurrent graph and emphasizes nonlinear dynamics support [BRITS, NeurIPS 2018](https://papers.nips.cc/paper/7911-brits-bidirectional-recurrent-imputation-for-time-series). SAITS uses diagonally masked self-attention to combine temporal and feature dependencies [SAITS, arXiv:2202.08516](https://arxiv.org/abs/2202.08516). CSDI explicitly trains a conditional diffusion model for time-series imputation and reports strong deterministic/probabilistic imputation gains [CSDI, arXiv:2107.03502](https://arxiv.org/abs/2107.03502).

Implication for us: **do not sell CSDI as a new imputation method.** Sell the observation that corruption-aware imputation changes the downstream OOD channel for pretrained chaotic forecasters.

### 3.2 Benchmarks care about missing rate and missing pattern

TSI-Bench argues that time-series imputation needs standardized evaluation across missing rates and patterns, and explicitly studies diverse missingness scenarios across many algorithms/datasets [TSI-Bench, arXiv:2406.12747](https://arxiv.org/abs/2406.12747). This directly supports adding pattern-aware corruption beyond iid Bernoulli masks.

Implication: our paper should report **missingness pattern** as a first-class experimental variable, not only `s`.

### 3.3 Irregular sampling and EDM already exist

Variable-step-size EDM was designed exactly for missing/irregular samples. It incorporates temporal spacing into delay-coordinate vectors and compares against exclusion and linear interpolation on chaotic ecological systems [Johnson & Munch 2022](https://repository.library.noaa.gov/view/noaa/62647). RAINDROP also emphasizes irregularly sampled multivariate series where different subsets of sensors are observed at different times [RAINDROP, ICLR 2022](https://openreview.net/forum?id=Kwm8I7dU-l5).

Implication: if we only use iid full-vector dropout, reviewers can say this is not realistic irregular sampling. We need at least one asynchronous or block-missing regime.

### 3.4 Foundation models and delay-manifold methods

Chronos tokenizes scaled/quantized values and trains language-model architectures on time-series tokens [Chronos, arXiv:2403.07815](https://arxiv.org/abs/2403.07815). Panda trains patched attention on chaotic systems and explicitly claims zero-shot chaotic forecasting ability [Panda, arXiv:2505.13755](https://arxiv.org/abs/2505.13755). DeepEDM/LETS Forecast is rooted in Takens/EDM and attention-as-kernel regression [DeepEDM, ICML 2025](https://proceedings.mlr.press/v267/majeedi25a.html). A separate delay-embedding theory paper connects sequence models to partially observed noisy dynamics [Delay Embedding Theory, OpenReview](https://openreview.net/forum?id=wew3SpwIqr).

Implication: the paper is timely, but the contribution must be **failure law and mechanism**, not "we combined Panda/CSDI/DeepEDM."

### 3.5 Data assimilation is the classical sparse/noisy-chaos baseline

For Lorenz96 and spatiotemporal chaos, ensemble Kalman methods are the natural classical baseline. LETKF is a standard scalable approach for large chaotic systems [Hunt et al. 2007](https://www.sciencedirect.com/science/article/abs/pii/S0167278906004647), and hybrid DA+ML work has explicitly targeted Lorenz96 from sparse noisy observations [Bocquet et al. 2020](https://www.sciencedirect.com/science/article/abs/pii/S1877750320304725).

Implication: for a stronger paper, include an EnKF/LETKF-style state-estimation baseline on L96, at least in appendix. Otherwise a dynamics reviewer may ask for it.

## 4. Better Corruption Design

We should separate the corruption design into three layers.

### Layer A: Fine Mechanism Grid

Use this to discover the transition, not just display it.

Recommended grid:

```text
s values:     0.00, 0.20, 0.40, 0.55, 0.65, 0.75, 0.82, 0.88, 0.93, 0.97
sigma values: 0.00, 0.05, 0.10, 0.20, 0.35, 0.50, 0.80, 1.20
```

Do not run the full 10x8 grid everywhere. Run:

- pure sparsity line: all `s`, `sigma=0`
- pure noise line: `s=0`, all `sigma`
- transition rectangle: `s={0.55,0.65,0.75,0.82,0.88}` × `sigma={0,0.10,0.20,0.35,0.50}`

Primary metrics:

- `Pr(VPT > 0)`
- `Pr(VPT > 0.5)`
- median VPT, not only mean
- paired bootstrap Δ for imputer/forecaster swaps
- patch OOD geometry: curvature distribution, low-curvature patch fraction, JS/Wasserstein vs clean

### Layer B: Missingness Pattern Grid

Run only on transition-band cells from Layer A.

| Regime | Meaning | Why it matters |
|---|---|---|
| `iid_time` | current setup; same mask for all dims | synchronized full-vector dropout; keeps comparison with existing results |
| `iid_channel` | independent mask for each dimension/sensor | realistic asynchronous sensors; tests whether cross-variable information helps |
| `block_time` | contiguous missing bursts of length 4/8/16/32 | real sensor outages; stress-tests interpolation chord artifacts |
| `periodic_subsample` | keep every k-th sample, optionally jittered | sparse but regular sensors; separates sampling interval from randomness |
| `mnar_curvature` | missing probability rises with speed/curvature/amplitude | hard but realistic failure: sensors fail during dynamic extremes |

Main paper can include `iid_time` and `block_time`; appendix can include the rest.

### Layer C: Paper Summary Path

After Layer A/B, choose 6-7 scenarios for compact figures. These should be **data-driven representative points**, not arbitrary diagonal points.

A safer candidate path:

| Stage | Purpose | s | sigma |
|---|---|---:|---:|
| H0 | clean | 0.00 | 0.00 |
| H1 | mild in-distribution corruption | 0.20 | 0.05 |
| H2 | moderate sparse, low noise | 0.40 | 0.10 |
| H3 | pre-threshold | 0.65 | 0.20 |
| H4 | patch threshold | 0.82 | 0.35 |
| H5 | harsh but not pure noise floor | 0.90 | 0.60 |
| H6 | no-info floor / stress test | 0.95 | 1.00 |

This is intentionally less jumpy in noise than current S0-S6. Current `sigma=1.5` is a useful stress test, but it is too easy for a reviewer to dismiss as "of course everything fails."

## 5. Normalize Stages By Geometry, Not Just Percent Missing

For tokenized/patch forecasters, the relevant unit is not just `s`; it is observations per patch.

With Panda-style patch length 16, iid missing gives:

| s | keep p | expected observed points per patch |
|---:|---:|---:|
| 0.40 | 0.60 | 9.6 |
| 0.60 | 0.40 | 6.4 |
| 0.75 | 0.25 | 4.0 |
| 0.85 | 0.15 | 2.4 |
| 0.90 | 0.10 | 1.6 |
| 0.95 | 0.05 | 0.8 |

This explains why the existing OOD curvature jump around `s≈0.85` is plausible: patches begin to contain only 2-3 true observations, so linear interpolation creates long straight segments.

For dynamical systems, also normalize by Lyapunov time:

```text
mean observation interval in Lyapunov units ≈ lambda * dt / (1 - s)
gap length in Lyapunov units ≈ lambda * dt * gap_steps
```

This matters because the same `s=0.9` is not equally hard on L63, L96, and Rössler. A paper-quality design should report `lambda * dt / keep_prob` or expected gap length in Lyapunov units next to each scenario.

## 6. Noise Injection Recommendations

Keep the basic additive Gaussian observation model, but make it more precise:

1. Add noise only to observed values or add before masking; both are equivalent for observed context, but code should state the convention.
2. Prefer per-dimension `sigma_d = noise_frac * std_d` for multivariate systems. Scalar attractor std is okay for headline simplicity, but per-dim scaling is fairer when dimensions have different amplitudes.
3. Report observation SNR or `sigma / attr_std` in every table.
4. Keep `sigma <= 1.0` in main text. Put `sigma=1.2/1.5` in appendix as no-info stress tests.
5. Store the clean trajectory, noisy observed trajectory, mask, and filled trajectory for mechanism plots.

## 7. What To Run Next

### Must-run before submission

1. **Fine pure-sparsity line** on L63, L96 N=20, Rössler:
   - methods: `linear→Panda`, `CSDI→Panda`, `linear→DeepEDM`, `CSDI→DeepEDM`
   - `sigma=0`
   - `s={0,0.2,0.4,0.55,0.65,0.75,0.82,0.88,0.93,0.97}`
   - 10 seeds for L96 headline, 5 seeds for the others initially

2. **Fine pure-noise line**:
   - `s=0`
   - `sigma={0,0.05,0.1,0.2,0.35,0.5,0.8,1.2}`
   - same methods

3. **Block missingness stress test**:
   - `s≈0.6` and `s≈0.8`
   - block lengths `{4,8,16,32}` or fixed Lyapunov-length gaps
   - compare linear, Kalman/EnKF, CSDI

4. **One modern imputer baseline**:
   - SAITS or BRITS via PyPOTS is enough
   - use it only in the imputer × Panda matrix, not everywhere

5. **One data-assimilation baseline for L96**:
   - EnKF/LETKF-style smoother if feasible
   - appendix is acceptable

### Nice-to-have

- Variable-step-size EDM baseline on L63/Rössler, because it is the most directly relevant missing/irregular EDM prior work.
- Asynchronous per-channel missingness on L96 N=20.
- MNAR curvature/speed-dependent dropout as a strong real-world stress case.

## 8. How This Becomes A Publishable Paper

The publishable gap is:

> Existing imputation benchmarks study reconstruction error and generic downstream tasks. Existing chaotic foundation models study clean or nearly clean contexts. Existing EDM/DA methods handle sparse observations, but not tokenizer/preprocessing OOD in pretrained forecasters. We identify and isolate this failure channel, then show how corruption-aware reconstruction mitigates it.

Main-paper claims should be:

1. **Phenomenon:** pretrained chaotic/time-series forecasters show non-smooth survival collapse under sparse/noisy observation protocols.
2. **Mechanism:** the collapse is strongly mediated by preprocessing-induced patch geometry, especially interpolation under high sparsity.
3. **Intervention:** corruption-aware imputation rescues both Panda and delay-manifold forecasters in the transition band.
4. **Scope:** delay-manifold forecasting is useful, but not the only survival path; CSDI is the first lever.

What not to claim:

- "Any ambient predictor fails."
- "Foundation models are intrinsically bad at chaos."
- "Delay coordinates are the only survivor."
- "S0-S6 alone proves a phase transition."

Recommended final figure set:

1. **Fine failure frontier:** `s`/`sigma` curves with survival probabilities.
2. **Mechanism geometry:** patch curvature/OOD distribution for clean vs linear-fill vs CSDI-fill.
3. **Isolation matrix:** imputer × forecaster, with paired bootstrap.
4. **Pattern robustness:** iid vs block missingness, probably appendix if space is tight.

## 9. Immediate Implementation Plan

1. ✅ Add a reusable corruption module:
   - `make_corrupted_observations(traj, mask_regime, s, sigma, seed, per_dim_noise=True, block_len=None, jitter=None)`
   - return `{observed, mask, noisy_full, metadata}`

2. ✅ Add a scenario design JSON:
   - `experiments/week1/configs/corruption_grid_v2.json`
   - keep S0-S6 as `legacy_diagonal`
   - add `fine_s_line`, `fine_sigma_line`, `transition_rectangle`, `pattern_grid`

3. ✅ Add an aggregator that reports:
   - survival probabilities
   - median VPT
   - expected observations per patch
   - mean/max gap in Lyapunov units

4. ✅ Add L63 first-run template:
   - `experiments/week1/phase_transition_grid_l63_v2.py`
   - dry-run metadata mode
   - resource guards for at most 4 visible GPUs and 4 CPU threads per process

5. Then run L63 first. L63 is cheap and will tell us whether the redesigned grid exposes a clean threshold before spending GPU time on L96.

## 10. Dry-Run Metadata Results

Generated files:

- `experiments/week1/results/pt_l63_grid_v2_summary_path_dry20.json`
- `experiments/week1/results/pt_l63_grid_v2_pattern_grid_dry20.json`
- `experiments/week1/figures/summary_path_dry20.md`
- `experiments/week1/figures/pattern_grid_dry20.md`

20-mask average for the proposed H0-H6 path:

| Stage | s | sigma | keep | obs/patch | max gap Lyap |
|---|---:|---:|---:|---:|---:|
| H0 | 0.00 | 0.00 | 1.000 | 16.00 | 0.00 |
| H1 | 0.20 | 0.05 | 0.797 | 12.76 | 0.08 |
| H2 | 0.40 | 0.10 | 0.590 | 9.43 | 0.14 |
| H3 | 0.65 | 0.20 | 0.345 | 5.52 | 0.31 |
| H4 | 0.82 | 0.35 | 0.179 | 2.86 | 0.51 |
| H5 | 0.90 | 0.60 | 0.105 | 1.68 | 0.87 |
| H6 | 0.95 | 1.00 | 0.048 | 0.76 | 1.65 |

20-mask average for the missingness pattern grid:

| Regime | keep | obs/patch | max gap Lyap |
|---|---:|---:|---:|
| iid channel, s=0.60 | 0.400 | 6.40 | 0.09 |
| iid time, s=0.60 | 0.397 | 6.36 | 0.23 |
| block len 8, s=0.60 | 0.397 | 6.35 | 0.80 |
| block len 16, s=0.60 | 0.392 | 6.27 | 1.46 |
| block len 32, s=0.60 | 0.381 | 6.09 | 2.32 |
| periodic s=0.80 | 0.200 | 3.20 | 0.09 |
| jittered periodic s=0.80 | 0.200 | 3.20 | 0.14 |
| MNAR curvature s=0.60 | 0.482 | 7.71 | 0.39 |

Immediate conclusion: the redesigned H path is usable, and missingness pattern
must be a first-class axis. Same missing rate can mean max gaps from 0.09 to
2.32 Lyapunov times, which is far too different to collapse into one `s` value.
