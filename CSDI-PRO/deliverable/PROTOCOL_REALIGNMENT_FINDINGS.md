# Protocol Realignment Findings — 2026-04-30

> Critical document. Reverses or qualifies several pre-2026-04-30 findings.
> Read before writing §4 of the paper.

## Summary

The L63 diagnostics that supported the "reconstruction-forecastability
mismatch" mechanism puzzle (jitter milestone + Panda embedding diagnostic
+ raw-patch v1/v2) used a **different `attractor_std` constant** than the
v2 grid runner that produces Figure 1.

When we re-run the diagnostics under the v2 protocol so they measure the
same physical scenarios as Figure 1, **three of the diagnostics flip or
qualify substantially**:

1. **L63 jitter SP65/SP82** flips from "no-intervention regime" to
   "CSDI-unique regime" with very tight CIs.
2. **L63 Panda embedding SP65/SP82** flips from "linear closer to clean"
   to "CSDI closer to clean by 6–9× at SP65; ≈ equal at SP82".
3. **L63 raw-patch v2 metrics** flip at SP65 from "linear closer" to
   "CSDI closer" across local stdev, lag-1 autocorrelation, and mid-frequency
   power; at SP82 they become mixed rather than a clean mismatch.

The Pillar-3 *mechanism puzzle* — phrased as "rescuing context is farther
from clean in raw and Panda-token geometry" — **was mostly a protocol
artifact for the L63 entrance band**. At SP65, v2-aligned data supports the
older tokenizer/raw-OOD mitigation mechanism the project had temporarily
discarded. At SP82, the result is mixed: CSDI still improves forecastability,
but raw/token distance-to-clean no longer cleanly separates CSDI from linear.

## Source of the discrepancy

| Component | v2 grid runner (Figure 1) | Old jitter / embedding |
|---|---|---|
| `attractor_std` | `LORENZ63_ATTRACTOR_STD = 8.51` (constant, per-axis-mean) | `compute_attractor_std() ≈ 13.86` |
| Mask seed | `1000 × seed + 5000 + grid_index` | `seed × 100 + 7` (scenario-independent) |

Effect of the 1.6× `attractor_std` mismatch:
- **CSDI inference normalization** is over-scaled in the old protocol →
  CSDI's diffusion samples become noisier than necessary → its imputations
  drift further from clean.
- **VPT thresholds** `{0.3, 0.5, 1.0} × attractor_std` are stricter in the
  old protocol → mean VPT numbers compress.

Both effects penalize CSDI specifically. Once the protocol is corrected,
CSDI imputations behave properly and both jitter and embedding diagnostics
move in CSDI's favor.

## v2-aligned L63 jitter (10 seeds)

| Scenario | linear | iid jitter | shuffled resid | **CSDI** |
|---|---:|---:|---:|---:|
| SP65 mean | 1.22 | 1.33 | 1.12 | **2.87** |
| SP65 Pr(VPT > 1.0) | 70 % | 80 % | 60 % | **100 %** |
| SP82 mean | 0.33 | 0.49 | 0.59 | **0.70** |
| SP82 Pr(VPT > 1.0) | 0 % | 20 % | 10 % | **20 %** |

Paired-bootstrap Δ vs linear:

| Scenario | jitter | shuffled | csdi |
|---|---:|---:|---:|
| SP65 | +0.11 [-0.02, +0.30] ≈ | -0.10 [-0.24, -0.01] ↓ | **+1.65 [+1.39, +1.91] ↑** |
| SP82 | +0.16 [-0.16, +0.51] ≈ | +0.26 [-0.01, +0.56] ≈ | **+0.37 [+0.05, +0.67] ↑** |

L63 SP65/SP82 are now **CSDI-unique regime**, with very tight CIs at SP65.
The old jitter milestone's "L63 SP65 = no-intervention regime" reading is
**superseded**.

Files: `experiments/week1/results/panda_jitter_control_l63_sp65_sp82_v2protocol_10seed_{h0,h5}.json`.

## v2-aligned L63 Panda embedding (5 seeds)

| Stage | SP65 linear/csdi ratio | SP82 linear/csdi ratio |
|---|---:|---:|
| patch (post-encoder of Panda) | **8.03** | 1.02 |
| embed (DynamicsEmbedder) | **6.12** | 1.06 |
| encoder (final encoder token) | **6.70** | 0.87 |
| pooled (latent) | **8.85** | 1.04 |

`> 1` means linear is *farther* from clean than CSDI. At SP65 the ratios are
6–9 across all four representation stages → **CSDI tokens are dramatically
closer to the clean trajectory in Panda's own latent space**. At SP82 the
ratios are essentially 1 → no clean separation in either direction.

Files: `experiments/week1/results/panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_5seed.json`,
`experiments/week1/figures/panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_5seed.md`.

## v2-aligned L63 raw-patch diagnostic (10 seeds)

Raw-patch v2 was re-run with the Figure-1 protocol:
`LORENZ63_ATTRACTOR_STD = 8.51`, `dt = 0.025`, `n_ctx = 512`, and
`1000 × seed + 5000 + grid_index` corruption seeds.

Wasserstein distance to the clean patch distribution:

| Scenario | Metric | linear | CSDI | linear / CSDI |
|---|---:|---:|---:|---:|
| SP65 | local stdev | 0.1474 | **0.0166** | 8.87× |
| SP65 | lag-1 ρ | 0.0037 | **0.0016** | 2.28× |
| SP65 | mid-freq power | 0.0091 | **0.0006** | 16.16× |
| SP82 | local stdev | 0.4217 | **0.3069** | 1.37× |
| SP82 | lag-1 ρ | **0.0093** | 0.0934 | 0.10× |
| SP82 | mid-freq power | 0.0139 | **0.0112** | 1.24× |

At SP65, CSDI is closer to clean in all three raw-patch metrics, matching
the Panda-token diagnostic. At SP82, local stdev and mid-frequency power
slightly favor CSDI, while lag-1 autocorrelation strongly favors linear.
Thus the valid mechanism statement is not "CSDI is farther from clean"; it
is: **CSDI reduces raw/token OOD in the entrance band, while distance-to-clean
metrics become insufficient near the frontier floor.**

Files: `experiments/week1/results/l63_patch_ood_v2_v2protocol_sp65_sp82_10seed.json`,
`experiments/week1/figures/l63_patch_ood_v2_v2protocol_metrics_SP{65,82}.png`.

## What this changes in the paper

### Mechanism (§4) needs to be rewritten regime-aware

The locked abstract sentence currently says:

> "Inside the transition band, corruption-aware reconstruction is the only
> tested intervention that reliably moves models back across the frontier,
> **even though the rescuing context is farther from the clean trajectory
> than linear interpolation in both raw-patch and Panda-token geometry.**
> This exposes a reconstruction-forecastability mismatch..."

The bolded claim is **false at SP65 under v2 protocol** and mixed at SP82.
It cannot stand as-is. The replacement is:

> In the entrance band of the sparse-observation frontier, corruption-aware
> imputation moves Panda contexts closer to the clean trajectory in both raw
> patch statistics and Panda-token space, and this is where the largest
> forecastability rescue occurs. Near the frontier floor, distance-to-clean
> metrics lose explanatory power: CSDI retains a smaller survival advantage
> even though raw/token distances are mixed or nearly tied.

This keeps the useful part of the mechanism story without overclaiming a
global reconstruction-forecastability mismatch.

### What still holds

- **Sharp forecastability frontier (Pillar 1)**: locked. Figure 1 is correct.
- **Intervention law (Pillar 2)**: actually *strengthened*. v2-aligned
  jitter shows tighter, larger CSDI gains at SP65 and clean SP82 gains.
- **Entrance-band OOD mitigation (Pillar 3 revised)**: supported by both
  raw-patch and Panda-token diagnostics at SP65.
- **Pure-noise insensitivity**: still holds. CSDI does not rescue under
  s=0, σ>0; this is sparsity-specific, not noise-denoising.
- **DeepEDM as companion (Pillar 4)**: unchanged.

### What does not hold any more

- "L63 SP65 is the no-intervention regime example" — wrong; CSDI rescue is
  +1.65 Λ.
- "Linear is closer to clean than CSDI in Panda token space" — wrong at
  SP65 (ratios 6–9 the other way), neutral at SP82.
- "Linear is closer to clean than CSDI in raw-patch statistics" — wrong at
  SP65, mixed at SP82.

### What remains uncertain

- Why CSDI retains a smaller SP82 survival advantage when lag-1 raw-patch
  distance and encoder-token distance no longer favor it. This is a floor-band
  residual mechanism question, not the main Figure-1 mechanism.

## Decision

Mechanism phrasing is now locked to the regime-aware option: entrance-band
OOD mitigation plus floor-band residual effects. The old "farther from clean"
abstract sentence is superseded and has been replaced in `STORY_LOCK_2026-04-28.md`
and `deliverable/paper/story_locked_sections_1_3_4_en.md`.

## Action queue

1. ✅ Re-run L63 jitter at v2 protocol (10 seeds) — done.
2. ✅ Re-run L63 Panda embedding at v2 protocol (5 seeds) — done.
3. ✅ Re-run L63 raw-patch diagnostic (`patch_ood_diagnostic_v2_l63.py`) at
   v2 protocol — done.
4. ✅ Update `STORY_LOCK_2026-04-28.md` and
   `story_locked_sections_1_3_4_en.md` accordingly.

The L96 N=20 experiments (B in the original plan) and SAITS / BRITS /
Glocal-IB (C) can proceed after this point.
