# Headline Refresh Plan — 2026-04-30

> Pre-submission refresh of all CSDI-dependent headline numbers under the
> patched `sigma_override` imputation protocol (see
> `deliverable/CSDI_SANITY_FINDINGS.md` for protocol rationale). Required
> before the Abstract / §3 / §4 / §6 cite final numbers.

## Why this matters

All CSDI-dependent JSONs in the repo as of 2026-04-30 morning were produced
*before* the `sigma_override` patch landed in the imputation drivers. Under
the patched protocol, pure-sparsity cells use `sigma_override = 0` (CSDI
treats observed points as exact), and noisy-cell `sigma_override = σ × σ_attr`.
The patch reduces observed-anchor error to ~10⁻⁶ in pure sparsity and
removes a small over-noising bias in noisy cells. Headline VPT numbers can
shift by O(0.05–0.3 Λ) per cell.

Per the user's call (2026-04-30 PM): refresh all headline numbers ONCE
before polishing appendices, Chinese mirror, or running Experiment C1, so
nothing downstream needs rework.

## What is being refreshed

### Wave 1 — Figure 1 (running, GPU 0–3)

L63 v2 fine_s_line + fine_sigma_line, 10 seeds split halves, 4 cells each.
720 runs total.

PIDs in current bash session: 434317 / 434319 / 434321 / 434323.

Output JSONs (new tags so old pre-patch JSONs are preserved for diff):
- `pt_l63_grid_v2_l63_fine_s_v2_10seed_patched_{h0,h5}.json`
- `pt_l63_grid_v2_l63_fine_sigma_v2_10seed_patched_{h0,h5}.json`

ETA: ~30–50 min wall-clock.

### Wave 2 — diagnostics + cross-system (after Wave 1)

After Wave 1, fan out across GPU 0–3:

1. **L63 jitter v2 protocol**, 10 seeds, SP65 + SP82, split halves
   (GPU 0+1 or single GPU).
   Tag: `panda_jitter_control_l63_sp65_sp82_v2protocol_patched_10seed`.
2. **L63 Panda embedding diagnostic v2**, 5 seeds, SP65 + SP82.
   Tag: `panda_embedding_ood_l63_sp65_sp82_dt025_v2protocol_patched_5seed`.
3. **L63 raw-patch diagnostic v2**, 10 seeds, SP65 + SP82.
   Tag: `l63_patch_ood_v2_v2protocol_patched_sp65_sp82_10seed`.
4. **L96 N=20 v2 transition band B**, 5 seeds, SP55–SP82 + NO010–NO050,
   4 cells. Tag: `l96N20_v2_B_patched_5seed_<group>`.
5. **L96 jitter v2** (SP65 + SP82), 5 seeds, 4 cells.
   Tag: `panda_jitter_control_l96N20_sp65_sp82_v2protocol_patched_5seed`.
6. **Rössler jitter v2** (SP65 + SP82), 5 seeds, 4 cells.
   Tag: `panda_jitter_control_rossler_sp65_sp82_v2protocol_patched_5seed`.

ETA: ~30–60 min wall-clock with 4 GPUs.

### Not refreshed (deliberately)

- The **legacy 4-system isolation S0–S6 5-seed** dataset uses the older
  `make_sparse_noisy` corruption pipeline and is cited in §4.1 as
  cross-system replication, *not* as the headline. We keep it as-is and
  add a footnote that the v2-protocol L63/L96 numbers in §3 / §4 are the
  authoritative ones.
- The **L96 N=20 SP65 5-seed smoke** is superseded by Wave 2 item 4.
- The **alt-imputer C0 sanity** uses CSDI but the comparison axis is
  SAITS / BRITS (per-instance); pre-patch CSDI=2.90 ceiling result is
  unchanged under patched protocol (verified by user's CSDI sanity check).

## Aggregator + paper update

After Wave 2 lands:

1. Re-run `aggregate_figure1_v2.py` with patched-tag inputs, output to
   `figure1_l63_v2_10seed_patched.{png,md}`.
2. Re-run `aggregate_jitter_cross_system.py` after pointing it at the
   patched JSONs.
3. Re-run `aggregate_isolation.py` and per-system aggregators where they
   consume isolation data.
4. Diff `figure1_l63_v2_10seed_patched.md` against current
   `figure1_l63_v2_10seed.md`. If diff is small (cells move <0.1 Λ in
   mean, paired CIs unchanged in sign), update paper_draft_en.md
   in-place. If diff is large (sign flips, bands move), revisit story.
5. Update Abstract / §1 / §3 / §4 / §6 numbers in
   `paper_draft_en.md` to match patched data. Same for
   `story_locked_sections_*`.
6. Mark `paper_draft_zh.md` mirror update as next.

## Decision rule

If the patched refresh produces **no qualitative change** (every paired
CI keeps its sign; transition band still visible at the same SP cells;
tail-survival ordering preserved), this is a numerical refresh only and
the locked story stands.

If any cell **flips sign or visibly changes the transition band**, we
pause, document in this file as an addendum, and revisit STORY_LOCK
before continuing.

The latter is unlikely — patched protocol mostly tightens CSDI behavior
in CSDI's favor (CSDI in pure-sparsity is now better, not worse) — but
the discipline is to verify, not assume.
