# Dialogue State Snapshot

Last updated: 2026-04-27, after CLAUDE_TURN_2.

This is the distilled view. The full record lives in
`LIVE_DIALOGUE.md`. Decisions reached are appended to `DECISIONS_LOG.md`.

## Topic

Decide the next v2 experiment slice and the paper-mechanism priorities for
CSDI-PRO, given the post-pivot framing (sparse-noisy phase transition +
delay-manifold OOD survival).

## Progress

- Turns so far: GPT_TURN_1, CLAUDE_TURN_1, GPT_TURN_2, CLAUDE_TURN_2.
- Status: **Claude has approved (`CLAUDE_APPROVAL: yes`); awaiting
  `GPT_FINAL_APPROVAL: yes` to formally close.**
- Final approval markers present:
  - `GPT_FINAL_APPROVAL: yes` — **no** (not yet)
  - `CLAUDE_APPROVAL: yes` — **yes** (in CLAUDE_TURN_2)

## Agreed (both sides, contingent on GPT final-approval)

### Implementation work

1. Add `--only_configs SP40 SP55 ...` flag to the v2 runner.
2. Patch / OOD diagnostic on L63 SP65 vs SP82, four context types (clean
   dense / sparse+linear / sparse+CSDI / optionally delay-coord patches),
   cheap no-training metrics first (PCA, curvature, chord-length, low-curv
   fraction, W/JS distance). CPU-only or ≤ 1 GPU. Outputs:
   - `experiments/week1/results/l63_patch_ood_sp65_sp82.json`
   - `experiments/week1/figures/l63_patch_ood_sp65_sp82.png`
   - short `deliverable/` readout.
3. Add or adapt a minimal L96 N=20 v2 wrapper supporting `--only_configs`.
4. L96 N=20 sanity smoke: SP65 first (`panda_linear`, `panda_csdi`,
   5 seeds, sigma=0); fallback SP55 / SP75 if SP65 out of band. No DeepEDM.
5. L63 targeted 10-seed expansion (gated on L96 smoke):
   - cells: `panda_linear`, `panda_csdi`, `deepedm_linear`, `deepedm_csdi`
   - sparsity: SP40 / SP55 / SP65 / SP75 / SP82
   - horizon:  H2 / H3 / H4
   - noise:    NO005 / NO010 / NO020 / NO035 / NO050
   - 13 configs × 4 cells × 10 seeds = **520 cell evals**
6. Aggregation: median VPT, `Pr(VPT > tau · Lyapunov_time)` for tau ∈
   {0.5, 1.0}, paired-bootstrap deltas.
7. Phase-transition terminology memo before finalizing main-paper wording.
   Lean: lead with "sharp failure frontier" / "threshold-like collapse";
   reserve "phase transition" for passages with explicit analog + citation.

### Compute budget

- ≤ 4 GPUs, GPU0–GPU3 only, no GPU7.
- 4 CPU threads / process, ≤ 16 CPU threads total.
- Wall-clock estimate: ~1–2 h for 520 evals on 4 GPUs (treated as estimate,
  not promise).
- Fallback ladder if runtime worse than expected: drop H2 / H3 / H4 first,
  keep sparsity + noise rows complete, expand only high-variance headline
  cells to 20 seeds.

### Mechanism / scope

- Headline mechanism = preprocessing / tokenizer OOD mitigation (CSDI rescue
  of Panda under sparse-noisy contexts).
- Delay-manifold pillar carried by DeepEDM cells (`deepedm_linear`,
  `deepedm_csdi`) inside the same v2 grid. Legacy `rde_delay` deferred to
  ablation/appendix.
- All CSDI cells load
  `CSDI-PRO/experiments/week2_modules/ckpts/full_v6_center_ep20.pt` and
  log that path; no retraining inside this expansion.

## Open items

None substantive. Waiting on:

- GPT to write `GPT_FINAL_APPROVAL: yes` (or a `GPT_CONSENSUS_CANDIDATE`
  block that consolidates the agreed plan, then the marker).

After GPT_FINAL_APPROVAL, the loop will append to `DECISIONS_LOG.md` and
terminate.

## Claude's standing position

Approved. Two implementation notes attached but not blocking:

1. Report two survival thresholds (tau ∈ {0.5, 1.0}), not one.
2. Pin and log CSDI checkpoint path in every run.

## Next action

Wait for `GPT_FINAL_APPROVAL: yes` in `LIVE_DIALOGUE.md`. Polling loop is
on a 60s quick-check (because a CLAUDE_TURN was just written), then 600s
default if GPT is slower.

## Context anchors (for future turns)

- M1 best ckpt: `CSDI-PRO/experiments/week2_modules/ckpts/full_v6_center_ep20.pt`
- L63 v2 3-seed pilot results: `CSDI-PRO/experiments/week1/figures/l63_*_v2_3seed.md`
- Isolation panels (L96 N=20, L96 N=10, Rössler, L63):
  `CSDI-PRO/experiments/week1/figures/iso_*_5seed*.{md,png}`
- Aggregation scripts:
  `CSDI-PRO/experiments/week1/aggregate_isolation.py`
  `CSDI-PRO/experiments/week1/aggregate_corruption_grid.py`
  `CSDI-PRO/experiments/week1/aggregate_survival_summary.py`
- Pivot memo: see auto-memory `csdi_pro_paper_pivot.md` (2026-04-26).
- Constraints: GPU0–GPU3 only (no GPU7), 4 threads/proc, ≤16 CPU threads.
