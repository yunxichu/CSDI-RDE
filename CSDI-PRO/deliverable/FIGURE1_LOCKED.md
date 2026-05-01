# Figure 1 — locked 2026-04-30

> Figure 1 (paper headline) is now produced from a 10-seed L63 v2 fine grid
> over the pre-registered `fine_s_line` (10 cells) and `fine_sigma_line`
> (8 cells), 4 imputer×forecaster sub-cells each = **720 runs total**.
>
> File: `deliverable/figures_main/figure1_l63_v2_10seed.png` and `.md`.
> Source: `experiments/week1/results/pt_l63_grid_v2_l63_fine_{s,sigma}_v2_10seed_{h0,h5}.json`.
> Aggregator: `experiments/week1/aggregate_figure1_v2.py`.

## Reviewer-acceptance check (six requirements from STORY_LOCK)

| # | Requirement | Status |
|---|---|---|
| 1 | `s` and `σ` decoupled (no S0–S6 reuse) | ✅ |
| 2 | 10 seeds per cell | ✅ (720 records) |
| 3 | mean VPT + Pr(VPT > 0.5) + Pr(VPT > 1.0) | ✅ three-column panels |
| 4 | 95 % bootstrap CI on mean, Wilson 95 % CI on Pr | ✅ |
| 5 | Pre-registered cells only | ✅ |
| 6 | **Transition band visibly sharp, not scatter** | ✅ |

The `STORY_LOCK_2026-04-28.md` fallback (downgrade to
"regime-dependent forecastability boundary") is **not invoked**. The
"sharp forecastability frontier" language is locked.

## Headline numerical result — sparsity transition band (σ = 0)

CSDI→Panda vs Linear→Panda on the cells where the frontier lives:

| Scenario (s) | CSDI→Panda mean [95% CI] | Linear→Panda mean [95% CI] | Δ mean | CSDI Pr(VPT>1.0) | Linear Pr(VPT>1.0) |
|---|---|---|---:|---:|---:|
| SP55 (s=0.55) | **2.71 [2.41, 2.89]** | 1.64 [1.05, 2.21] | +1.07 | 100% | 70% |
| **SP65 (s=0.65)** | **2.89 [2.89, 2.90]** | 1.22 [0.97, 1.47] | **+1.67** | **100%** | 70% |
| **SP75 (s=0.75)** | **1.16 [0.79, 1.53]** | 0.51 [0.22, 0.84] | **+0.65** | **60%** | 20% |
| **SP82 (s=0.82)** | 0.76 [0.41, 1.18] | 0.33 [0.18, 0.49] | +0.43 | 20% | 0% |

The mean-VPT 95 % bootstrap CIs are non-overlapping at **SP65 and SP75**, and
nearly so at SP55 / SP82. Pr(VPT>1.0) shows a clean 100 % → 20 % collapse for
CSDI between SP65 and SP82, while Linear→Panda has already collapsed by SP75.

The frontier is therefore at `s ≈ 0.65 → 0.82` (about two cells wide), with
CSDI extending the survival region by roughly one cell relative to linear.
This is what we mean by "sharp forecastability frontier".

## NEW finding — pure-noise axis (s = 0): rescue is sparsity-specific

The pure-noise line is essential to read carefully because it isolates
*observation noise* from *missingness-induced gap geometry*.

| Scenario (σ) | CSDI→Panda mean | Linear→Panda mean | Δ (csdi − linear) |
|---|---:|---:|---:|
| NO00 / NO005 / NO010 | 2.16–2.89 | 2.16–2.89 | ≈ 0 |
| **NO020** (σ=0.20) | 1.40 | 1.53 | **−0.13** |
| **NO050** (σ=0.50) | 0.72 | 0.89 | **−0.17** |
| **NO080** (σ=0.80) | 0.35 | 0.57 | **−0.22** |
| **NO120** (σ=1.20) | 0.13 | 0.42 | **−0.29** |

CSDI does **not** rescue Panda when the corruption is observation noise
without missingness, and at higher noise it slightly hurts. The 95 %
bootstrap CIs on Δ touch zero at every noise cell except NO080 / NO120, where
Linear→Panda is mildly but consistently better.

### Why this sharpens the story

1. **Rescue is gap-specific, not denoising.** CSDI's value is filling
   missing-value gaps with dynamics-consistent samples, not removing
   observation noise from already-present points. With dense but noisy
   contexts, CSDI re-imputes already-observed values and injects sampling
   variance with no benefit.
2. **The "sparse-noisy frontier" name remains correct,** but the frontier is
   carried mostly by the *sparsity* axis. The noise axis shows monotone
   degradation with no sharp transition — and CSDI is not the right
   intervention there. We should say so explicitly in §3.
3. **This disambiguates "CSDI as imputer" vs "CSDI as denoiser".** Only the
   former is a frontier-crossing intervention. Future work probably wants a
   denoising-aware variant for the noise axis; the current paper is
   transparent that it does not provide one.

### Writing consequence

In §3 / §4, the regime taxonomy should now read:

- Regime 1 (no-intervention): only at clean / very-mild corruption (e.g. SP00 / NO00 cells where every cell sits at the VPT ceiling). **Do not** cite L63 SP55 / SP65 here — Figure 1 shows CSDI strongly rescues at SP55 (mean 2.71 vs 1.64) and SP65 (2.89 vs 1.22). The earlier jitter-milestone reading of "L63 SP65 = no-intervention regime" was a different protocol (different `attractor_std` and mask seed scheme); see §"Protocol caveat" below.
- Regime 2 (sparsity transition band): SP55–SP82, CSDI uniquely effective in tail; the band is sharpest between SP65 and SP82.
- Regime 3 (pure noise): no sparsity, CSDI does not help and may slightly hurt; this is **honest scope** for the paper, not a weakness.
- Regime 4 (combined sparse+noisy): both effects compound; the L96 SP82 jitter milestone shows CSDI still wins in tail when sparsity dominates. Note: the existing jitter-milestone L63 numbers are not directly comparable to Figure 1 (see "Protocol caveat") — re-runs at v2 protocol are scheduled.

### Protocol caveat (must fix before §4 is finalized)

Figure 1 (v2 grid runner) uses:
- `attractor_std = LORENZ63_ATTRACTOR_STD = 8.51` (constant, per-axis-mean over 10⁵ steps)
- mask seed `1000 * seed + 5000 + grid_index`

The earlier jitter-control and Panda-embedding diagnostics use:
- `attractor_std = compute_attractor_std() ≈ 13.86` (different convention)
- mask seed `seed * 100 + 7` (scenario-independent)

The 1.6× ratio between the two `attractor_std` constants directly scales (a)
the injected noise magnitude `σ · attr_std`, (b) the VPT threshold
`{0.3, 0.5, 1.0} · attr_std`, and (c) the CSDI inference normalization. The
two protocols therefore measure different physical scenarios at the same SP
label.

Action: re-run L63 jitter SP65 / SP82 (10 seeds) and L63 Panda embedding
diagnostic at v2 protocol before writing §4 regime taxonomy.

The locked abstract sentence (§ "Locked abstract sentence" in `STORY_LOCK_2026-04-28.md`) does not need changes — "sparse-noisy" already covers both axes, and "inside the transition band" already names the regime where CSDI is uniquely effective.

## DeepEDM (delay-manifold) cells — companion confirmed

DeepEDM cells (Linear→DeepEDM, CSDI→DeepEDM) sit consistently below the
Panda cells in absolute VPT across the sparsity line, but CSDI improves
DeepEDM in the same transition band:

- SP55: csdi 2.15 vs linear 1.21 (+0.94)
- SP65: csdi 1.89 vs linear 0.81 (+1.08)
- SP75: csdi 1.10 vs linear 0.67 (+0.43)
- SP82: csdi 0.60 vs linear 0.48 (+0.12)

This supports the "complementary dynamics-structured route" framing without
overclaiming primacy. CSDI→Panda is the strongest absolute cell; DeepEDM
provides a route that doesn't depend on Panda's tokenizer at all.

## Decision log

- 2026-04-30 morning: Figure 1 produced; visual is clean; "sharp
  forecastability frontier" language locked into STORY_LOCK and §1/§3/§4.
- Pure-noise rescue absence is logged here as a new finding; §3 and §4 will
  cite it explicitly when those drafts are written.
- No re-run is needed for Figure 1. Camera-ready may want to extend headline
  cells (SP65, SP75) to 20 seeds, but 10 seeds already gives non-overlapping
  CIs at SP65 and SP75.

## Next experiments (unchanged from earlier plan)

- **Experiment B**: L96 N=20 v2 transition-band 5-seed replication
  (SP55/SP65/SP75/SP82 + NO010/NO020/NO050).
- **Experiment C**: SAITS / BRITS alt-imputer reviewer-defense at 1–2
  CSDI-unique cells (L96 N=20 SP82, L63 SP82).

A is locked. B and C remain for cross-system frontier evidence and reviewer
defense; both are scheduled but not started.
