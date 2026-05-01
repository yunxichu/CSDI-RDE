# Paper Status — 2026-05-01

This is the working map for the current paper. It answers four questions:

1. What story are we writing?
2. Which experiments are already usable?
3. Where are the numbers, figures, and draft text?
4. What is still missing before submission?

## 1. Current Paper Story

Do **not** write this as a new `delaymask` / new embedding trick paper.

Write it as an empirical + mechanistic forecastability paper:

> Pretrained chaotic forecasters fail across sharp sparse-observation
> forecastability frontiers. Inside the sparsity transition band,
> corruption-aware imputation moves Panda back across the frontier by restoring
> raw-patch and Panda-token geometry toward clean contexts. This effect is a
> sparse-gap imputation effect, not a generic dense-noise denoising effect.
> DeepEDM in delay coordinates is a complementary dynamics-aware route, not the
> only survivor.

The three qualifiers that must stay in the abstract and §1:

- **Sparse-observation frontier**, not all corruption.
- **Inside the transition band**, not universal improvement everywhere.
- **Structured residuals are not interchangeable with iid noise**, especially
  in survival/tail metrics.

## 2. Where The Paper Is Written

Main English draft:

- `deliverable/paper/paper_draft_en.md`

Old English draft archive:

- `deliverable/paper/paper_draft_en_archive_2026-04-30.md`

Chinese draft:

- `deliverable/paper/paper_draft_zh.md`
- Status: still old narrative; not updated yet.

Story lock and patched-number notes:

- `deliverable/STORY_LOCK_2026-04-28.md`
- `deliverable/HEADLINE_REFRESH_RESULTS.md`
- `deliverable/FIGURE1_PATCHED_REFRESH.md`
- `deliverable/CSDI_SANITY_FINDINGS.md`

Important: `paper_draft_en.md` is the authoritative current text. The
`story_locked_sections_*.md` files are older drop-in skeletons and now carry
warnings that the main draft supersedes them for patched numbers.

## 3. Current English Draft Structure

`deliverable/paper/paper_draft_en.md` currently has:

- Abstract
- §1 Introduction
- §2 Related Work
- §3 Sharp Forecastability Frontiers
- §4 Mechanism and Intervention Isolation
- §5 Method
- §6 Discussion and Limitations
- §7 Conclusion
- Appendix A-G

Status:

- Main text §1-§7 is written in the new story.
- Appendices A/D/E/F/G are inherited from the old pipeline draft and still need
  cleanup/re-keying.
- Chinese mirror is not updated.

## 4. Completed Experiments We Can Use

### A. Main Figure 1: L63 v2 10-seed sparse/noise grid

Purpose:

- Establish the main sparse-observation forecastability frontier.
- Decouple sparsity `s` from dense observation noise `sigma`.

Patched outputs:

- `deliverable/figures_main/figure1_l63_v2_10seed_patched.png`
- `deliverable/figures_main/figure1_l63_v2_10seed_patched.md`
- `experiments/week1/results/pt_l63_grid_v2_l63_fine_s_v2_10seed_patched_h0.json`
- `experiments/week1/results/pt_l63_grid_v2_l63_fine_s_v2_10seed_patched_h5.json`
- `experiments/week1/results/pt_l63_grid_v2_l63_fine_sigma_v2_10seed_patched_h0.json`
- `experiments/week1/results/pt_l63_grid_v2_l63_fine_sigma_v2_10seed_patched_h5.json`

Headline patched numbers:

| Scenario | Linear -> Panda | CSDI -> Panda | Paired CSDI-linear |
|---|---:|---:|---:|
| SP65 | 1.22 / 70% | 2.86 / 100% | +1.64 [+1.40,+1.87] |
| SP75 | 0.52 / 20% | 2.29 / 100% | +1.77 [+1.39,+2.17] |
| SP82 | 0.34 / 0% | 1.34 / 60% | +1.00 [+0.54,+1.51] |

Format: mean VPT@1.0 / `Pr(VPT>1.0)`.

Pure-noise result:

- Panda CSDI is tied with linear at low dense noise and slightly worse at
  higher dense noise.
- This supports the claim that CSDI is a sparse-gap imputer, not a generic
  denoiser.

### B. L63 Jitter / Residual Controls

Purpose:

- Test whether CSDI is merely stochastic regularization.

Patched outputs:

- `experiments/week1/results/panda_jitter_control_l63_sp65_sp82_v2protocol_patched_10seed.json`
- `experiments/week1/figures/panda_jitter_control_l63_sp65_sp82_v2protocol_patched_10seed.md`

Headline:

| Scenario | linear | iid jitter | shuffled residual | CSDI |
|---|---:|---:|---:|---:|
| SP65 | 1.22 | 1.39 | 1.06 | 2.87 |
| SP82 | 0.33 | 0.54 | 0.67 | 1.42 |

CSDI paired gains:

- SP65: +1.65 `[+1.41,+1.87]`
- SP82: +1.09 `[+0.65,+1.61]`

Interpretation:

- Iid jitter does not reproduce CSDI.
- Shuffled CSDI residual helps at SP82 but remains much weaker than CSDI.
- Structured imputation path matters.

### C. L63 Raw-Patch and Panda-Embedding Diagnostics

Purpose:

- Mechanism evidence: does CSDI move contexts closer to clean in raw and token
  geometry?

Patched outputs:

- `experiments/week1/results/l63_patch_ood_v2_v2protocol_sp65_sp82_10seed.json`
- `experiments/week1/results/panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_patched_5seed.json`
- `experiments/week1/figures/panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_patched_5seed.md`

Headline:

SP65 raw-patch linear/CSDI W1-to-clean ratios:

- local stdev: 21.02x
- lag-1 autocorrelation: 15.02x
- mid-frequency power: 33.71x

SP65 Panda-space linear/CSDI distance ratios:

- patch: 16.77x
- embedder: 12.84x
- encoder: 14.02x
- pooled: 21.85x

SP82:

- Panda-space ratios still favor CSDI: 1.63-2.43x.
- Raw metrics are partly mixed: local stdev and mid-frequency favor CSDI, but
  lag-1 autocorrelation favors linear.

Interpretation:

- Entrance-band mechanism is strong OOD reduction.
- Floor-band mechanism is still favorable to CSDI, but not fully explained by
  one scalar raw statistic.

### D. L96 N=20 v2 Cross-System Replication

Purpose:

- Check whether L63 sparse-observation story transfers to high-dimensional
  chaos.

Patched output:

- `experiments/week1/results/pt_l96_smoke_l96N20_v2_B_patched_5seed.json`
- `experiments/week1/results/pt_l96_smoke_l96N20_v2_B_patched_seed5_9.json`
- `deliverable/L96_V2_B_PATCHED_N10.md`

Current interpretation:

- L96 Panda mean is high-variance because rare lucky linear seeds dominate.
- Use median and survival for Panda.
- Use DeepEDM paired gain as cleaner high-dimensional companion evidence.

Headline:

| Scenario | Panda median linear->CSDI | Panda Pr>0.5 linear->CSDI | DeepEDM paired CSDI-linear |
|---|---:|---:|---:|
| SP65 | 0.71 -> 1.26 | 70% -> 100% | +0.46 [+0.25,+0.67] |
| SP82 | 0.50 -> 1.05 | 60% -> 100% | +0.43 [+0.29,+0.57] |

Do not use the old L96 mean headline `0.91 -> 3.31`; it was pre-patch and
mean-sensitive.

Decision after patched n=10 readout:

- Paper writing uses L96 as a median/survival and DeepEDM companion
  replication, not as a Panda-mean headline.
- Seeds 5-9 confirmed the writing choice: L96 supports survival/median and
  DeepEDM companion claims, but Panda mean remains high-variance and should not
  be used as the headline.

### E. L96 and Rössler Jitter Controls

Purpose:

- Check cross-system tail/regularization behavior.

Outputs:

- `experiments/week1/results/panda_jitter_control_l96N20_sp65_sp82_v2protocol_patched_5seed.json`
- `experiments/week1/results/panda_jitter_control_rossler_sp65_sp82_v2protocol_patched_5seed.json`

Interpretation:

- L96: high-variance caveat. CSDI improves median/survival, not stable mean.
- Rössler: CSDI direction positive, especially SP82; VPT>1.0 is too strict due
  to small Lyapunov exponent / finite horizon, so use with caution.

### F. Alt-Imputer C0 Sanity

Purpose:

- Sanity check generic per-instance SAITS/BRITS.

Output:

- `experiments/week1/results/panda_altimputer_l63sp65_partial_5seed.json`
- `deliverable/EXPERIMENT_C_PLAN.md`

Status:

- Appendix-only sanity.
- Do not use as main reviewer-defense because per-instance SAITS/BRITS is
  unfairly weak compared to pretrained CSDI.

## 5. Experiments Not Yet Finished / Not Submission-Ready

### C1 pretrained alt-imputer comparison

Needed for stronger reviewer defense:

- Pretrain SAITS / Glocal-IB on the same chaos corpus used by CSDI.
- Evaluate at L63 SP82 and L96 SP82.

Status:

- Designed in `deliverable/EXPERIMENT_C_PLAN.md`.
- Not run.
- This is pre-submission defense, not needed to keep current main story alive.

### Appendices cleanup

Status:

- Appendix A/D/E/F/G still carry old 4-module / τ-search / theory-pipeline
  material.
- Need cleanup before sharing as a polished paper.

### Chinese mirror

Status:

- `deliverable/paper/paper_draft_zh.md` still old.
- Update after English draft stabilizes.

## 6. How To Write The Paper Now

Write it as:

1. **Abstract / §1:** sparse-observation forecastability frontier, not a new
   module trick.
2. **§3:** Figure 1 L63 patched grid is the central evidence.
3. **§4:** isolation and mechanism:
   - CSDI rescues in transition band.
   - Jitter/residual controls do not match it.
   - Raw/token diagnostics show OOD reduction in entrance band.
   - L96 confirms direction but has high mean variance.
4. **§5:** methods kept short: corruption model, CSDI, DeepEDM, metrics.
5. **§6:** limitations:
   - pure-noise non-rescue,
   - L96 high variance,
   - MG/Chua scope boundaries,
   - C1 alt-imputer still pending.

Do **not** write:

- "delaymask is the main contribution";
- "CSDI is a universal denoiser";
- "DeepEDM is the only survivor";
- "L96 Panda mean proves rescue";
- "we fully explain Panda's internal failure channel."

## 7. Immediate Next Step

The next useful action is not more prose in the abstract. It is:

1. Re-key Appendix D figure index to the new patched figures.
2. Compress or quarantine old Appendix E/F/G τ-search material.
3. Add a small "Experiment table" in §3 or Appendix B listing every dataset,
   seeds, cells, and output JSON path.

After that, the English draft will be readable by an external collaborator.
