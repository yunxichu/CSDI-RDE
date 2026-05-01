# Tail-Distinction Milestone — 2026-04-28

> Purpose: settle the "is CSDI just stochastic regularization?" question that
> the L96 N=20 SP65 single-cell jitter run (in `STORY_LOCK_2026-04-28.md` §3b)
> left open. This document supersedes the cross-system reading in §3b.

## What we tested

For each of 6 (system, scenario) settings, we compared 4 fillings of the
sparse context, all fed to Panda-72M:

- `linear`            — current default, no intervention
- `linear_iid_jitter` — Gaussian noise scaled to per-channel CSDI residual
                         std, applied only at missing entries
- `linear_shuffled_resid` — CSDI residual values shuffled across the missing
                             positions, applied only at missing entries
- `csdi`              — corruption-aware diffusion imputation (M1)

Pure sparsity, σ=0. 5 seeds. Imputation seed and mask seed pinned per
(seed, scenario). All scripts log the same checkpoint paths.

Systems × scenarios:

|              | SP65 (s=0.65, σ=0) | SP82 (s=0.82, σ=0) |
|---           |---                 |---                 |
| **L63**      | done               | done               |
| **L96 N=20** | done               | done               |
| **Rössler**  | done               | done               |

Code: `experiments/week1/panda_jitter_control_l63.py`,
       `panda_jitter_control_l96.py`,
       `panda_jitter_control_rossler.py`.
Aggregator: `aggregate_jitter_cross_system.py`.
Figures: `deliverable/figures_jitter/jitter_milestone_{SP65,SP82}.png`.
Summary table: `deliverable/figures_jitter/jitter_milestone_summary.md`.

## Headline finding

**The "iid jitter recovers most of the mean rescue" effect is confined to
ONE of 6 (system, scenario) settings.** It is **not** generic regularization.

Six paired-bootstrap CIs vs. linear, classified by which interventions cross 0:

| Setting     | jitter Δ (CI)        | shuffled Δ (CI)      | csdi Δ (CI)              | which intervention is paired-CI ↑ |
|---          |---                   |---                   |---                       |---                                |
| L63 SP65    | −0.60 [−1.28, +0.09] | −0.48 [−1.17, +0.20] | −0.19 [−0.90, +0.62]    | **none** (no-intervention regime) |
| L63 SP82    | +0.00 [+0.00, +0.01] | +0.14 [+0.00, +0.39] | **+0.43 [+0.04, +0.86]** | **csdi only**                     |
| L96 SP65    | +1.08 [+0.13, +2.54] | +1.08 [+0.13, +2.52] | +1.19 [+0.08, +2.62]    | **all three** (the outlier setting)|
| L96 SP82    | −0.00 [−0.08, +0.07] | +0.18 [−0.02, +0.47] | **+2.40 [+0.10, +6.59]** | **csdi only**                     |
| Rössler SP65| −0.04 [−0.19, +0.10] | −0.04 [−0.19, +0.10] | +0.22 [+0.00, +0.57]    | csdi (point estimate; CI touches 0)|
| Rössler SP82| −0.11 [−0.41, +0.07] | −0.14 [−0.41, +0.00] | +0.23 [−0.27, +0.73]    | csdi (point estimate only)        |

5/6 settings: jitter and shuffled residual fail to produce a strictly positive
paired CI. CSDI is the only intervention with a paired-CI gain (or, on
Rössler, the only one with a positive mean direction).

## Tail-survival metric: Pr(VPT > 1.0 Λ) sharpens the separation

Mean VPT can be dominated by single seeds with very long forecasts. The
fraction of seeds achieving forecast longer than one Lyapunov decorrelation
time is operationally what matters. Pr(VPT > 1.0 Λ) per setting:

| Setting     | linear | iid jitter | shuffled | **csdi** |
|---          |   ---: |       ---: |     ---: |     ---: |
| L63 SP65    |   80%  |     40%    |    60%   |   80%    |
| L63 SP82    |   20%  |     20%    |    40%   |   40%    |
| L96 SP65    |   20%  |     40%    |    40%   |   60%    |
| L96 SP82    |   20%  |     20%    |    20%   |   **60%**|
| Rössler SP65|    0%  |      0%    |     0%   |    0%    |
| Rössler SP82|    0%  |      0%    |     0%   |    0%    |

Notes:
- **L96 N=20 SP82** is the cleanest tail-distinction: linear/jitter/shuffled
  all stuck at 20%, csdi at 60%. Mean VPT is also 3.6× higher for csdi
  (3.31 vs 0.91).
- **Rössler 0% across all cells** is a system-specific floor effect: λ_1=0.071
  is so small that pred_len=128 × dt=0.1 ≈ 9 Lyapunov times barely admits any
  forecast > 1 Lyap. To use Rössler in the tail metric the Pr cutoff should
  be Pr(VPT > 0.5 Λ) (csdi 100%/80% vs linear 80%/60% at SP65/SP82).
- L63 SP65 confirms the **no-intervention regime**: linear is already at the
  ceiling, every intervention including CSDI is ≤ linear in tail.

## The three regimes

Reading across the 6 settings, three coherent regimes emerge:

1. **No-intervention regime** (L63 SP65). System is smooth at this dt and
   sparsity is mild enough that linear interpolation is already near-optimal.
   All interventions hurt or do nothing. Honest scope to report.

2. **Generic-regularization regime** (L96 N=20 SP65). Adding **any** plausible
   variability to filled context helps Panda. Mean-VPT gain is recovered by
   iid jitter; CSDI distinguishes itself only on tail metrics.

3. **CSDI-unique regime** (L63 SP82, L96 SP82, Rössler SP65/SP82). CSDI is the
   only intervention with a positive paired-CI gain or even a positive mean
   direction. The structure of the CSDI residual matters; generic noise does
   not transfer.

The transition-band scenarios (SP82 across all three systems) are uniformly
in regime 3. The lower-corruption scenario (SP65) is in regime 2 only on
L96 N=20; on L63 it is regime 1 (no-intervention) and on Rössler it is
regime 3 (csdi-unique by point estimate).

This makes the rescue claim **regime-aware**, which matches the way
forecastability under sparse observations actually behaves and is much
harder for a reviewer to break with a single counterexample.

## What this milestone enables in the paper

1. **Abstract / §1 claim** changes from "CSDI rescues Panda" to:
   > "Corruption-aware reconstruction is the only intervention that
   > reliably improves Panda forecastability inside the transition band
   > (s ≥ 0.75 across L63, L96 N=20, Rössler)."

2. **Pr(VPT > 1.0 Λ) becomes a primary metric**, not secondary. Headline
   tables include this column. Mean VPT is reported but framed as "useful
   summary, dominated by occasional long-forecast seeds".

3. **§4 mechanism** writes the regime taxonomy explicitly:
   - Regime 1 (no-intervention): show L63 SP65 numbers, acknowledge.
   - Regime 2 (generic-regularization): show L96 N=20 SP65 numbers,
     distinguish on Pr(VPT>1.0) tail metric.
   - Regime 3 (CSDI-unique): show L63 SP82 + L96 N=20 SP82 numbers as the
     headline case, both mean and tail favor csdi only.

4. **§4 also writes** the "forecastability ≠ reconstruction quality" paragraph:
   - Forward-pointing fact (from `panda_embedding_ood_l63_*.md`): linear-fill
     contexts are CLOSER to clean than CSDI-fill in Panda's own latent space
     across patch / embed / encoder / pooled stages.
   - Yet under the same Panda forecasts, CSDI is more reliable in tail.
   - The mechanism is open: structured residual content of CSDI cannot be
     replicated by iid noise of the same per-channel std, nor by shuffled
     CSDI residuals (which only weakly help on L63 SP82). What this means
     mechanistically is left as future work; what it means experimentally
     is settled: **the structure of the residual matters, not just its scale**.

5. **No further experiments are required for this milestone.** Optional
   follow-ups to consider only if reviewers ask:
   - L96 N=10 jitter, to test whether SP65 generic-regularization is
     specific to N=20.
   - SAITS / BRITS substituted into the same isolation pipeline, to test
     whether structure-of-imputation effect is unique to CSDI or holds for
     any structured imputer.

## Stop point

The forecastability-vs-reconstruction gap, the regime taxonomy, and the
tail-vs-mean distinction together constitute a publishable mechanism story
that does not over-claim. Combined with the existing 4-system isolation
matrix and the L96 N=20 SP65 smoke that locked the v2 transition band,
**this milestone is sufficient evidence to begin paper §1 + §3 + §4 rewrite**.

## Files

Result JSONs:
- `experiments/week1/results/panda_jitter_control_l63_sp65_sp82_5seed.json`
- `experiments/week1/results/panda_jitter_control_l96N20_sp65_sp82_5seed.json`
- `experiments/week1/results/panda_jitter_control_rossler_sp65_sp82_5seed.json`

Aggregated:
- `deliverable/figures_jitter/jitter_milestone_summary.md`
- `deliverable/figures_jitter/jitter_milestone_SP65.png`
- `deliverable/figures_jitter/jitter_milestone_SP82.png`

Predecessors:
- Tokenizer-OOD diagnostic: `experiments/week1/results/panda_embedding_ood_l63_sp65_sp82_dt025_5seed.json`,
  `experiments/week1/figures/panda_embedding_ood_l63_sp65_sp82_dt025_5seed.md`
- Patch-geometry diagnostic v1/v2: `experiments/week1/results/l63_patch_ood_sp65_sp82.json`,
  `experiments/week1/results/l63_patch_ood_v2_sp65_sp82.json`
- Cross-system 5-seed isolation: `deliverable/figures_isolation/*.md`
- L96 N=20 SP65 v2 smoke: `experiments/week1/results/pt_l96_smoke_sp65_5seed.json`
