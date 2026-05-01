# Headline-number traceability — 2026-05-01

Every quantitative claim in **Abstract / §1 / §3 / §4 / §6.4 / §6.6** of
`paper_draft_en.md` traced to its source JSON or figure markdown. Built
in the submission-prep QA pass at `290e38b`; extended at the P2 freeze
(2026-05-02) for the L96 30-seed and Jena Climate real-sensor additions.

**Use.** Before any future numerical edit to the paper, look up the
number here and update the source as well — never edit the paper number
without re-deriving it from the JSON. If a source filename is wrong, fix
the table; do not silently update the paper.

> One stale-number bug fixed during this audit:
> §6.4 first bullet said "~500K independent-IC L63 windows" for the SAITS
> training corpus — the actual P1.1 training used the 64K subset
> (`lorenz63_clean_64k_L128.npz`). Both en and zh corrected.

## Conventions

- All `mean VPT` numbers are mean of seed-level VPT@1.0 unless explicitly
  marked `median` (in which case the L96 high-variance limitation applies).
- All paired-bootstrap CIs are 5000 resamples on per-seed differences.
- All survival-probability CIs are Wilson 95 % binomial.
- "Source JSON" is the result file under `experiments/week1/results/`;
  the corresponding markdown summary lives in `experiments/week1/figures/`
  with the same stem.

## Table

### Abstract (lines 7–46)

| Number | Description | en line(s) | Source |
|:--|:--|:--|:--|
| +0.41 Λ, [+0.05, +0.87] | L63 SP65 CSDI − SAITS-pretrained paired Δ + CI | 14–15 | `panda_altimputer_l63_sp65_sp82_pretrained_10seed_chunked.json` (P1.1) |
| +0.06 Λ, [−0.31, +0.59] | L63 SP82 CSDI − SAITS-pretrained paired Δ + CI | 16 | same as above |
| 12 to 34 | linear/CSDI distance ratios across raw + Panda stages on L63 SP65 | 19–21 | `panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_patched_5seed.json` + `l63_patch_ood_v2_v2protocol_sp65_sp82_10seed.json` |
| 2.86 vs 1.22 Λ | L63 SP65 CSDI vs linear → Panda mean VPT | 31–32 | `pt_l63_grid_v2_l63_fine_s_v2_10seed_patched_h0.json` (Figure 1 patched) |
| [+1.40, +1.87] | L63 SP65 paired-bootstrap CI (CSDI − linear) | 33 | same |
| 70 % [39 %, 90 %] → 100 % [72 %, 100 %] | L63 SP65 Pr(VPT>1.0) Wilson CI | 33–34 | same |
| +1.00 Λ, [+0.54, +1.51] | L63 SP82 CSDI − linear paired Δ + CI | 35 | same |
| 0 % → 60 % | L63 SP82 Pr(VPT>1.0) | 36 | same |
| 0.50 → 1.05 | L96 N=20 SP82 Panda median VPT linear → CSDI | 39 | `pt_l96_smoke_l96N20_v2_B_patched_seed5_9.json` |
| 60 % → 100 % | L96 N=20 SP82 Pr(VPT>0.5) | 40 | same |
| +0.43 Λ, [+0.29, +0.57] | L96 N=20 SP82 DeepEDM CSDI − linear paired Δ | 40–41 | same |

### §1 Introduction (lines 48–125)

| Number | Description | en line(s) | Source |
|:--|:--|:--|:--|
| 1.22 / 2.86 / +1.64 [+1.40, +1.87] | L63 SP65 cell repeat (linear / CSDI / paired Δ) | 91 | `pt_l63_grid_v2_l63_fine_s_v2_10seed_patched_h0.json` |
| 70 % [40 %, 89 %] → 100 % [72 %, 100 %] | L63 SP65 Pr(VPT>1.0) Wilson CI repeat | 92–93 | same |
| +1.00 [+0.54, +1.51], 0 % [0 %, 28 %] → 60 % [31 %, 83 %] | L63 SP82 cell repeat | 96–97 | same |
| 80 % [49 %, 94 %] vs 40 % [17 %, 69 %] | L96 N=20 SP65 CSDI vs jitter/shuffled/linear Pr(VPT>1.0) | 104–105 | `pt_l96_smoke_l96N20_v2_B_patched_5seed.json` |
| 0.50 → 1.05, 60 % [31 %, 83 %] → 100 % [72 %, 100 %] | L96 N=20 SP82 Panda median + Pr(VPT>0.5) | 109–110 | `pt_l96_smoke_l96N20_v2_B_patched_seed5_9.json` |
| +0.43 Λ, [+0.29, +0.57] | L96 N=20 SP82 DeepEDM repeat | 111–112 | same |

### §3 Empirical Forecastability Frontiers

| Number | Description | en line(s) | Source |
|:--|:--|:--|:--|
| 1.22 → 0.33 | L63 linear → Panda mean VPT SP65 → SP82 | 233–234 | Figure 1 patched JSON |
| 70 % → 0 % | L63 linear → Panda Pr(VPT>1.0) | 234 | same |
| 2.86 / 100 % at SP65 | L63 CSDI → Panda mean VPT / Pr(>1.0) | 235 | same |
| 1.34 / 60 % at SP82 | L63 CSDI → Panda mean VPT / Pr(>1.0) | 236 | same |
| 0.50 → 1.05 | L96 N=20 SP82 Panda median linear → CSDI | 241 | `pt_l96_smoke_l96N20_v2_B_patched_seed5_9.json` |
| 60 % → 100 % | L96 N=20 SP82 Pr(VPT>0.5) | 242 | same |
| +0.43, [+0.29, +0.57] | L96 N=20 SP82 DeepEDM | 244 | same |

### §4 Mechanism and Intervention Isolation

#### §4.1 Isolation matrix (legacy S0–S6)

| Number | Description | en line(s) | Source |
|:--|:--|:--|:--|
| 0.52 → 3.60, 60 % → 100 %, +3.07 [+0.57, +6.45] | L96 N=20 S4 cell, legacy S0–S6 | 310–311 | `pt_l96_iso_l96N20_*_5seed.json` (legacy) |
| +1.11, [+0.08, +2.22] | L96 N=10 S4 paired CSDI − linear | 312 | `pt_l96_iso_l96N10_*_5seed.json` (legacy) |
| +0.82, [+0.32, +1.37] | L63 S2 paired CSDI − linear | 312 | `pt_l63_iso_l63_*_5seed.json` (legacy) |

#### §4.2 Raw-patch + Panda-embedding diagnostics (L63 SP65 + SP82)

| Number | Description | en line(s) | Source |
|:--|:--|:--|:--|
| 21.02× / 15.02× / 33.71× | L63 SP65 raw-patch W₁ ratios (local stdev / lag-1 / mid-freq) | 334–336 | `l63_patch_ood_v2_v2protocol_sp65_sp82_10seed.json` |
| 16.77× / 12.84× / 14.02× / 21.85× | L63 SP65 Panda patch / embedder / encoder / pooled distance ratios | 339–340 | `panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_patched_5seed.md` |
| 1.63–2.43× | L63 SP82 Panda-space distance ratios | 344 | same |
| 3.54× / 5.19× / 0.62× | L63 SP82 raw W₁ ratios (lag-1 favors linear) | 346 | `l63_patch_ood_v2_v2protocol_sp65_sp82_10seed.json` |

#### §4.3 Jitter / shuffled-residual controls (patched 10-seed)

| Number | Description | en line(s) | Source |
|:--|:--|:--|:--|
| 1.22 / 2.87, +1.65 [+1.41, +1.87] | L63 SP65 linear / CSDI / paired Δ | 366–367 | `panda_jitter_control_l63_sp65_sp82_v2protocol_patched_10seed.json` |
| +0.17 [−0.01, +0.36], −0.16 [−0.34, −0.02] | L63 SP65 jitter Δ, shuffled Δ | 367–368 | same |
| 80 % vs 40 % | L96 N=20 SP65 CSDI vs jitter/shuffled Pr(VPT>1.0) | 374–375 | `pt_l96_smoke_l96N20_v2_B_patched_5seed.json` |
| +1.09 [+0.65, +1.61], 70 % | L63 SP82 CSDI Δ + Pr(>1.0) | 379–380 | `panda_jitter_control_l63_sp65_sp82_v2protocol_patched_10seed.json` |
| +0.43 [+0.29, +0.57] | L96 N=20 SP82 DeepEDM CSDI − linear | 382 | `pt_l96_smoke_l96N20_v2_B_patched_seed5_9.json` |

#### §4.4 Alt-imputer comparison (P1.1 + P1.5)

| Number | Description | en line(s) | Source |
|:--|:--|:--|:--|
| L63 SP65 / SP82 mean VPT linear / SAITS / CSDI | full table | 415–417 | `panda_altimputer_l63_sp65_sp82_pretrained_10seed_chunked.json` (P1.1) |
| All paired contrasts at SP65 / SP82 | SAITS−linear, CSDI−linear, CSDI−SAITS | 421–423 | same |
| L63 SP65 / SP82 Pr(VPT>1.0) Wilson | 70/90/100 %, 0/70/70 % with CIs | 429–431 | same |
| L96 N=20 SP82 mean / median / Wilson + paired bootstrap (n=30) | full block | ~445–467 | `panda_altimputer_l96_sp82_pretrained_30seed.json` (P2.2; supersedes the P1.5 10-seed JSON for §4.4) |
| 0.86 / 1.57 / 1.87 | L96 SP82 mean VPT linear / SAITS / CSDI (n=30) | ~448 | same |
| 0.25 / 1.01 / 1.26 | L96 SP82 median VPT linear / SAITS / CSDI (n=30) | ~448 | same |
| 20 / 50 / 73 % | L96 SP82 Pr(VPT>1.0) Wilson 95 % | ~448 | same |
| +0.71 [+0.02, +1.38] | L96 SP82 paired SAITS − linear (n=30) | ~456 | same |
| +1.01 [+0.36, +1.64] | L96 SP82 paired CSDI − linear (n=30) | ~457 | same |
| +0.31 [+0.07, +0.56] | L96 SP82 paired CSDI − SAITS (n=30) | ~458 | same |
| ~64K L96 / 64K L63 corpus sizes | training scale | 408, 434 | corpus npz file metadata |

### §6.4 Limitations

| Number | Description | en line(s) | Source |
|:--|:--|:--|:--|
| ~64K | L63 SAITS pretraining corpus | 613 | `lorenz63_clean_64k_L128.npz` (post-fix; previously stale "500K") |
| 0.34–0.50, ≤ 20 % | Chronos mean VPT@1.0 + Pr(VPT>1.0) on L63 SP55–SP82 | 623–624 | `chronos_frontier_l63_chronos_l63_sp55_sp82_5seed.json` (P1.2) |
| 0.34–0.50 | Chronos pred_len=64 native-horizon repeat | 624–625 | `chronos_frontier_l63_chronos_l63_sp55_sp82_5seed_pl64.json` (P1.4) |
| 2.84–2.85, 100 % | EnKF mean VPT + Pr(VPT>1.0) across SP55–SP82 | 636–637 | `enkf_l63_enkf_l63_v2_5seed.json` (P1.3) |

### §6.6 Real-sensor Jena Climate case study (P2.1)

| Number | Description | en line(s) | Source |
|:--|:--|:--|:--|
| 51.1 / 50.9 / 48.5 / 50.9 | Jena `linear → Chronos` vh@1.0 mean SP55/65/75/82 (n=10) | ~750 | `jena_real_sensor_jena_real_sensor_10seed.json` |
| 34.4 / 32.1 / 27.5 / 27.3 | Jena `SAITS-pretrained → Chronos` vh@1.0 mean | ~751 | same |
| −16.7 [−28.2, −5.8] | Jena SP55 paired SAITS − linear vh@1.0 (strict-negative) | ~755 | same |
| −18.8 [−29.7, −8.2] | Jena SP65 paired SAITS − linear vh@1.0 | ~755 | same |
| −21.0 [−34.3, −8.6] | Jena SP75 paired SAITS − linear vh@1.0 | ~756 | same |
| −23.6 [−39.2, −8.6] | Jena SP82 paired SAITS − linear vh@1.0 | ~756 | same |
| 14 features | Jena variable count (atmospheric, hourly) | Appendix C.2 | `experiments/week2_modules/data/real/jena_climate_2009_2016.csv` header |
| 0.62 z-units | SAITS-Jena val MAE on missing | Appendix C.2 | `experiments/week2_modules/ckpts/saits_jena_pretrained_meta.json` |

## Audit findings

1. **`paper_draft_en.md` line 613 — stale "500K" → "64K"**. Fixed.
2. **Chinese §6.4 mirror "约 50 万" → "约 64K"**. Fixed.
3. **All other numbers cross-check against listed sources without drift**
   under spot-checks.

This document is itself part of the submission round; commit alongside
`paper_draft_en.md` updates so future maintainers can trace.
