# Headline Refresh Results — 2026-05-01

This file summarizes the patched `sigma_override` refresh. It supersedes the
pre-patch CSDI-dependent numbers in older milestone notes.

## Completed Artifacts

- L63 Figure 1 patched:
  - `deliverable/figures_main/figure1_l63_v2_10seed_patched.png`
  - `deliverable/figures_main/figure1_l63_v2_10seed_patched.md`
  - `deliverable/FIGURE1_PATCHED_REFRESH.md`
- L63 jitter patched:
  - `experiments/week1/results/panda_jitter_control_l63_sp65_sp82_v2protocol_patched_10seed.json`
  - `experiments/week1/figures/panda_jitter_control_l63_sp65_sp82_v2protocol_patched_10seed.md`
- L63 Panda embedding patched:
  - `experiments/week1/results/panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_patched_5seed.json`
  - `experiments/week1/figures/panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_patched_5seed.md`
- L63 raw-patch diagnostic patched:
  - `experiments/week1/results/l63_patch_ood_v2_v2protocol_sp65_sp82_10seed.json`
- L96 N=20 v2 B patched:
  - `experiments/week1/results/pt_l96_smoke_l96N20_v2_B_patched_5seed.json`
  - `experiments/week1/results/pt_l96_smoke_l96N20_v2_B_patched_seed5_9.json`
  - `deliverable/L96_V2_B_PATCHED_N10.md`
- L96 and Rössler jitter patched:
  - `experiments/week1/results/panda_jitter_control_l96N20_sp65_sp82_v2protocol_patched_5seed.json`
  - `experiments/week1/results/panda_jitter_control_rossler_sp65_sp82_v2protocol_patched_5seed.json`

## Decision

No qualitative sign flip against the locked story. The patched protocol makes
the sparsity-axis CSDI effect stronger, especially at SP75/SP82. It also makes
the mechanism cleaner: CSDI is closer to clean in Panda-token space at both
SP65 and SP82, while raw metrics are strongly favorable at SP65 and partly
mixed only near SP82.

The L96 N=20 Panda mean remains too high-variance for an abstract headline.
Use L96 as a cross-system survival/median replication and use DeepEDM's paired
gain as the cleaner high-dimensional companion evidence.

## Main Numbers For Paper

L63 Figure 1:

| Scenario | Linear -> Panda | CSDI -> Panda | Paired CSDI-linear |
|---|---:|---:|---:|
| SP65 | 1.22 / 70% | 2.86 / 100% | +1.64 [+1.40,+1.87] |
| SP75 | 0.52 / 20% | 2.29 / 100% | +1.77 [+1.39,+2.17] |
| SP82 | 0.34 / 0% | 1.34 / 60% | +1.00 [+0.54,+1.51] |

Format: mean VPT@1.0 / `Pr(VPT>1.0)`.

L63 jitter controls:

| Scenario | linear | iid jitter | shuffled residual | CSDI |
|---|---:|---:|---:|---:|
| SP65 | 1.22 | 1.39 | 1.06 | 2.87 |
| SP82 | 0.33 | 0.54 | 0.67 | 1.42 |

CSDI paired gains: SP65 +1.65 [+1.41,+1.87], SP82 +1.09
[+0.65,+1.61].

L96 N=20 v2 B:

| Scenario | Panda median linear->CSDI | Panda Pr>0.5 linear->CSDI | DeepEDM paired CSDI-linear |
|---|---:|---:|---:|
| SP65 | 0.71 -> 1.26 | 70% -> 100% | +0.46 [+0.25,+0.67] |
| SP82 | 0.50 -> 1.05 | 60% -> 100% | +0.43 [+0.29,+0.57] |

Pure-noise axis:

- Panda CSDI is tied with linear at NO010/NO020 and slightly worse at higher
  noise (e.g. L63 NO050 -0.17 [-0.33,-0.02], NO080 -0.29
  [-0.57,-0.07]).
- This supports the gap-imputation framing and rules out "CSDI is just a
  generic denoiser."

## Paper Updates Already Applied

- `deliverable/paper/paper_draft_en.md` now uses patched L63/L96 numbers.
- `deliverable/STORY_LOCK_2026-04-28.md` now has a patched-refresh note.
- `story_locked_sections_*.md` now carry warnings that the main draft is
  authoritative for patched values.
