# Story Lock — 2026-04-28

## 2026-04-30 Protocol-Realignment Override

This document contains pre-realignment mechanism text below. The following
override is authoritative for §1/§4 until the paper draft is rewritten.

**Updated story.** Pretrained chaotic forecasters exhibit a sharp
forecastability frontier under sparse observations. Inside the sparsity
transition band, CSDI-style corruption-aware imputation is the only tested
intervention that reliably moves Panda back across the frontier; DeepEDM in
Takens coordinates is a complementary, dynamics-structured route. The
mechanism is regime-dependent: at the entrance band (L63 SP65), CSDI moves
contexts closer to clean in both raw-patch statistics and Panda-token space;
near the floor (L63 SP82), distance-to-clean metrics become mixed and no
longer fully explain the smaller survival advantage.

**Do not use the old abstract sentence** saying that the rescuing context is
"farther from clean in raw-patch and Panda-token geometry." Under the
Figure-1 protocol this is false at SP65 and mixed at SP82. See
`deliverable/PROTOCOL_REALIGNMENT_FINDINGS.md`.

## Locked One-Sentence Story

Pretrained chaotic forecasters exhibit a sharp forecastability frontier under sparse observations; **inside the sparsity transition band**, corruption-aware imputation is the only intervention we tested that reliably moves Panda back across that frontier, while delay-manifold forecasting provides a complementary dynamics-structured route. In the entrance band, the rescue is consistent with reduced raw-patch and Panda-token OOD; near the frontier floor, distance-to-clean metrics become mixed, so forecastability is not explained by a single fidelity metric.

## Reviewer-Acceptance Boundary (locked 2026-04-28 evening)

We will only make claims at or below this acceptance bar. Anything stronger is
not supported by the data and will be cut from drafts.

### Claims we make (acceptable to a careful reviewer)

1. Sparse/noisy observations create **sharp forecastability frontiers** in
   chaotic forecasting.
2. **Inside the transition band**, CSDI-style corruption-aware reconstruction
   **reliably improves downstream forecast survival** across the systems we
   tested.
3. At the L63 entrance band (SP65), CSDI is closer to clean than linear in
   both raw-patch statistics and Panda-token space; at the floor band (SP82),
   those distances are mixed, so fidelity metrics alone do not fully explain
   residual survival.
4. DeepEDM in Takens coordinates is a **complementary dynamics-structured
   route**, with **explicit scope boundaries** on Mackey-Glass and Chua.

### Claims we explicitly do NOT make

- "We proved the universal law of phase transitions in chaotic forecasting."
  — We mapped the frontier in 4 systems on a chosen `(s, σ)` grid; no
  universality is demonstrated.
- "CSDI is the only imputer that can rescue Panda."
  — We tested only `linear / Kalman / CSDI / iid jitter / shuffled CSDI
  residual`. SAITS / BRITS are pending. The strongest claim allowed is
  "CSDI is the only **tested** intervention …".
- "The rescuing context is farther from clean than linear in raw/Panda space."
  — Protocol-realigned diagnostics reject this at SP65 and show mixed evidence
  at SP82. The old mismatch sentence is superseded.
- "DeepEDM is the primary survival channel."
  — Isolation matrix shows `CSDI → Panda` is often stronger than
  `CSDI → DeepEDM` in absolute VPT. DeepEDM is companion, not primary.
- "All foundation chaotic forecasters fail this way."
  — We tested Panda-72M as the headline; Chronos / Parrot / persist appear
  as side baselines. No general claim about the foundation-model family.

### Fallback if Figure 1 is not visually clean

If A's 10-seed L63 grid does not show a clean transition band (e.g. CIs
overlap heavily across SP cells or the curve is monotone-smooth), we
downgrade the language from "sharp forecastability frontier" to
"regime-dependent forecastability boundary" and reduce the Pillar-1 claim
correspondingly. Paper is still publishable, with reduced impact.

## What Figure 1 Must Show (locked)

To pass reviewer acceptance, Figure 1 must satisfy ALL of:

1. `s` and `σ` decoupled (separate panels: pure-sparsity line + pure-noise
   line). No reuse of S0–S6 confounded grid.
2. **10 seeds per cell**.
3. Three metrics overlaid or in side panels: **mean VPT**, **Pr(VPT > 0.5)**,
   **Pr(VPT > 1.0)**. Mean alone is not enough — tail is required.
4. **95 % bootstrap CI** drawn on every cell.
5. Pre-registered cells only — full **`fine_s_line`** =
   `SP00 / SP20 / SP40 / SP55 / SP65 / SP75 / SP82 / SP88 / SP93 / SP97`
   (10 cells), full **`fine_sigma_line`** =
   `NO00 / NO005 / NO010 / NO020 / NO035 / NO050 / NO080 / NO120`
   (8 cells). Total cell count is therefore (10 + 8) × 4 sub-cells × 10
   seeds = **720 runs** for Figure 1, not the 400-run minimal version that
   appeared in earlier RUN_PLAN drafts. Both endpoints (clean SP00 / NO00
   and stress-floor SP97 / NO120) are kept so the figure shows where the
   frontier *starts* and where the floor lies. No after-the-fact cell
   addition.
6. Visible transition band, not scatter noise. If the band is not visible at
   10 seeds, treat as fallback (see above).
7. **Survival panels also carry CI.** Pr(VPT > 0.5) and Pr(VPT > 1.0) are
   binomial at n = 10; report **Wilson 95 % CI** on each cell so the
   reviewer cannot dismiss step jumps (e.g. 1/10 → 4/10) as visual
   artifacts.

## What We Are No Longer Claiming

Do **not** write the main mechanism as a universal statement:

> linear interpolation creates raw non-physical patches that are farther from Panda's tokenizer distribution, and CSDI rescues by making tokens closer to clean.

The protocol-realigned diagnostics support this at L63 SP65, but not as a
global law across the entire frontier. At SP82, raw/token distances are mixed
or nearly tied while CSDI still has a smaller survival advantage. The safe
mechanism is entrance-band OOD mitigation plus floor-band residual effects.

## Evidence Status

### Pillar 1: Sharp Failure Frontier

Status: strong.

Existing 5-seed phase-transition/isolation results show threshold-like VPT collapse across L63, L96 N=10/N=20, Rössler, and Kuramoto, with MG/Chua serving as boundary cases.

### Pillar 2: Corruption-Aware Intervention Law

Status: strong.

Four-system isolation ablation shows `CSDI -> Panda` and `CSDI -> DeepEDM` improve survival in transition-band scenarios.

Key headline examples:

| System | Contrast | Gain |
|:--|:--|:--|
| L96 N=20 S4 | `CSDI -> Panda` vs `Linear -> Panda` | +3.07 Lyapunov times, 95% paired CI [+0.57, +6.45] |
| L96 N=20 SP65 smoke | `CSDI -> Panda` vs `Linear -> Panda` | mean 2.49 vs 1.14, Pr(VPT>0.5) 100% vs 60% |
| L63 S2 | `CSDI -> Panda` vs `Linear -> Panda` | +0.82 Lyapunov times, 95% paired CI [+0.32, +1.37] |
| Rössler S3/S4 | `CSDI -> DeepEDM` vs `Linear -> DeepEDM` | consistent positive paired gains |

Figures/tables:

- `deliverable/figures_isolation/*_5seed.md`
- `deliverable/figures_isolation/*_heatmap.png`
- `deliverable/figures_isolation/*_bars.png`
- `experiments/week1/results/pt_l96_smoke_sp65_5seed.json`

### Pillar 3: Regime-Aware Mechanism

Status: rewritten after protocol realignment.

The old mechanism puzzle was mostly a protocol artifact: the raw-patch,
jitter, and Panda-token diagnostics used a different `attractor_std` and mask
seed scheme than Figure 1. Under the v2 protocol, L63 SP65 supports
entrance-band OOD mitigation, while L63 SP82 shows a mixed floor-band pattern.

Panda representation diagnostics (`dt=0.025`, `attractor_std=8.51`,
v2 grid-index seeds):

| Scenario | Stage | Linear/CSDI paired distance to clean |
|:--|:--|--:|
| SP65 | patch | 8.03 |
| SP65 | embed | 6.12 |
| SP65 | encoder | 6.70 |
| SP65 | pooled | 8.85 |
| SP82 | patch | 1.02 |
| SP82 | embed | 1.06 |
| SP82 | encoder | 0.87 |
| SP82 | pooled | 1.04 |

Ratios above 1 mean linear is farther from clean than CSDI. At SP65, CSDI is
6-9x closer to clean across all Panda stages; at SP82, the ratios are near 1.

Raw-patch v2 diagnostics show the same entrance-band pattern:

| Scenario | Metric | Linear/CSDI W1-to-clean ratio |
|:--|:--|--:|
| SP65 | local stdev | 8.87 |
| SP65 | lag-1 rho | 2.28 |
| SP65 | mid-frequency power | 16.16 |
| SP82 | local stdev | 1.37 |
| SP82 | lag-1 rho | 0.10 |
| SP82 | mid-frequency power | 1.24 |

Therefore §4 should say: CSDI reduces raw/token OOD in the entrance band where
the largest rescue occurs; near the frontier floor, distance-to-clean metrics
lose explanatory power and CSDI's smaller survival advantage likely depends on
additional structured-imputation effects.

Files:

- `experiments/week1/results/panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_5seed.json`
- `experiments/week1/figures/panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_5seed.md`
- `experiments/week1/results/l63_patch_ood_v2_v2protocol_sp65_sp82_10seed.json`
- `experiments/week1/figures/l63_patch_ood_v2_v2protocol_metrics_SP65.png`
- `experiments/week1/figures/l63_patch_ood_v2_v2protocol_metrics_SP82.png`

### Pillar 3b: Jitter Controls Show a Mixed but Useful Mechanism

> **Superseded — read `TAIL_DISTINCTION_MILESTONE.md` first.** That milestone
> extends jitter controls to 6 (system, scenario) settings (L63/L96/Rössler ×
> SP65/SP82) and shows the "iid jitter recovers most of the mean rescue"
> effect is confined to L96 N=20 SP65 only. The cross-system reading below
> is still numerically correct on its 1-2 settings but is no longer the
> headline mechanism statement.

Status: first controls completed on L63 and L96 N=20.

We tested whether CSDI helps Panda merely by adding stochastic perturbations to
linear-fill. Controls used the same missing mask and added noise only at missing
entries.

L63:

| Scenario | linear | linear + iid jitter | linear + shuffled CSDI residual | CSDI |
|:--|--:|--:|--:|--:|
| SP65 | 1.42 | 0.82 | 0.94 | 1.23 |
| SP82 | 0.43 | 0.43 | 0.57 | 0.86 |

L96 N=20:

| Scenario | linear | linear + iid jitter | linear + shuffled CSDI residual | CSDI |
|:--|--:|--:|--:|--:|
| SP65 | 1.33 | 2.40 | 2.42 | 2.52 |

Interpretation:

- At SP65, linear is already strong; CSDI is not a universal improvement.
- At SP82, CSDI improves over linear by +0.43 Lyapunov times with 95% paired CI [+0.04, +0.86].
- On L63, iid jitter does not rescue; shuffled residual gives only a weaker partial improvement.
- On L96 N=20 SP65, generic jitter and shuffled residual recover much of the **mean** VPT gain, but CSDI has the best median and much stronger `Pr(VPT>1.0)` (80% vs 40%).

This supports a narrower, safer mechanism statement: part of the Panda rescue is
conditioning/regularization rather than closeness to clean tokens; CSDI is the
most reliable version of that intervention, especially in survival/tail metrics,
but we should not claim all gains require uniquely dynamics-aware residuals.

Files:

- `experiments/week1/results/panda_jitter_control_l63_sp65_sp82_5seed.json`
- `experiments/week1/figures/panda_jitter_control_l63_sp65_sp82_5seed.md`
- `experiments/week1/figures/panda_jitter_control_l63_sp65_sp82_5seed.png`
- `experiments/week1/results/panda_jitter_control_l96N20_sp65_5seed.json`
- `experiments/week1/figures/panda_jitter_control_l96N20_sp65_5seed.md`
- `experiments/week1/figures/panda_jitter_control_l96N20_sp65_5seed.png`

### Pillar 4: Scope Boundary

Status: usable.

MG and Chua remain appendix boundary cases, not hidden failures. They support the narrower claim that the current smooth-attractor/delay-manifold assumptions do not cover 1D delay equations and piecewise-linear/non-smooth circuits without additional modeling.

## New Paper Framing

### Title Shape

Forecastability Frontiers Under Corrupted Chaotic Contexts

### Abstract-Level Claim (locked — do not paraphrase without re-checking)

Pretrained chaotic forecasters fail across sharp sparse-observation forecastability frontiers. Inside the sparsity transition band, corruption-aware imputation is the only tested intervention that reliably moves Panda back across the frontier. In the entrance band, this rescue is accompanied by a large reduction in both raw-patch and Panda-token distance to the clean context; near the frontier floor, Panda-token distances still favor CSDI but one raw temporal statistic becomes mixed, so distance-to-clean is informative but not a complete account of tail survival. CSDI is therefore a gap-imputation lever, not a generic dense-noise denoiser, and structured residuals are not fully interchangeable with iid noise of matched magnitude. Delay-manifold forecasting provides a complementary dynamics-aware route, with explicit scope boundaries on non-smooth systems such as Chua and scalar delay-differential systems such as Mackey-Glass.

Patched-refresh note (2026-05-01): the authoritative numeric values are now in
`deliverable/FIGURE1_PATCHED_REFRESH.md` and `paper_draft_en.md`; older
pre-patch values in archived notes must not be used for Abstract / §3 / §4.

Three reviewer-defeating qualifiers are now baked in:
- "**sparse-observation** frontier" (not all corruption — pure-noise axis does not show CSDI rescue).
- "**inside the transition band**" (rescue is regime-conditional and strongest on the sparsity axis).
- "**not fully interchangeable**" (patched L63 jitter controls show iid/shuffled residuals do not reproduce CSDI at SP65 and do not match CSDI at SP82; L96 remains a high-variance caveat where median/survival are more reliable than mean).

Do not drop any of the three from the §1 opening or the abstract.

### Main Contributions

1. **Failure law:** map sparse/noisy phase diagrams with survival probabilities, not just mean VPT.
2. **Intervention law:** isolate imputer and forecaster effects; show CSDI rescues Panda and DeepEDM across multiple systems.
3. **Regime-aware mechanism:** show entrance-band rescue coincides with reduced raw/Panda-token OOD, while floor-band survival cannot be explained by distance-to-clean alone.
4. **Dynamics-structured companion:** keep DeepEDM/Takens coordinates as a complementary route and clarify where it works or fails.

## Immediate Next Experiment

Next, update the paper skeleton and figures around the locked story:

1. Replace global "forecastability/reconstruction mismatch" language with the regime-aware mechanism.
2. Promote survival probability `Pr(VPT>0.5)` and `Pr(VPT>1.0)` over mean-only VPT.
3. Keep raw/Panda-space diagnostics as a mechanism figure: SP65 supports OOD mitigation; SP82 is mixed/floor-band.
4. Use jitter controls to show CSDI is not interchangeable with iid or shuffled residuals on L63 v2 protocol.

## Writing Rule Going Forward

Use "sharp failure frontier" as the headline term. Use "phase-transition-like" only after defining the order parameter as survival probability, e.g. `Pr(VPT > 0.5)` or `Pr(VPT > 1.0)`.

Do not use "tokenizer OOD" as a settled global mechanism. Write:

> In the entrance band, corruption-aware imputation reduces both raw-patch and
> Panda-token distance from the clean context, consistent with an OOD-mitigation
> mechanism. Near the frontier floor, the same distance measures are mixed,
> indicating that residual forecastability is not controlled by a single
> reconstruction-fidelity metric.
