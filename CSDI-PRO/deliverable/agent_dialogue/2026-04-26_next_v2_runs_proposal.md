# Proposal Packet

## Decision Needed

Which experiments should we run next after the L63 v2 3-seed pilot?

## Context

- Current evidence:
  - L63 v2 pilot completed for summary path, pure sparsity, pure noise, and
    missingness pattern grid.
  - Main sparsity transition appears around `s=0.40-0.75`.
  - Floor begins near `s≈0.82`, where patches contain about 2-3 observations.
  - Pure noise degrades Panda around `sigma=0.10-0.20`, even with no missingness.
  - Existing isolation ablations show CSDI→Panda can rescue many sparse/noisy
    cases, so the story is preprocessing/OOD mitigation, not "Panda is doomed."
- Open uncertainty:
  - Whether L63 v2 patterns replicate on L96 N=20 and Rössler.
  - Whether headline conclusions survive 10-20 seeds.
  - Whether tokenizer/patch OOD geometry can visually explain the transition.
- Why this matters now:
  - The next runs determine the main paper figure set and prevent wasting GPU on
    uninformative floor regimes.

## Proposed Action

Run **targeted expansion**, not full grid expansion.

Priority A: L63 targeted 10-seed expansion.

- Pure sparsity configs: `SP40, SP55, SP65, SP75, SP82`
- Summary configs: `H2, H3, H4`
- Pure noise configs: `NO005, NO010, NO020, NO035, NO050`
- Cells: `panda_linear, panda_csdi, deepedm_linear, deepedm_csdi`
- Seeds: 10 total first; expand to 20 only for headline cells with high variance.

Priority B: Extend v2 runner to L96 N=20 and Rössler.

- Start with pure sparsity line and summary path only.
- 5 seeds first for each system.
- Only expand transition-band cells after seeing the threshold.

Priority C: Start tokenizer/patch OOD diagnostic in parallel with no heavy GPU.

- Compare clean, linear-filled, CSDI-filled contexts.
- Metrics: patch curvature, segment length/gap geometry, simple PCA/UMAP if
  Panda patch embeddings are accessible.

Compute budget:

- At most 4 GPUs, GPU0-GPU3 only unless user says otherwise.
- 4 CPU threads per process.
- No use of GPU7; it is currently occupied by another process.

Expected outputs:

- Targeted JSON result files.
- Markdown aggregation with median VPT and `Pr(VPT>0.5)`.
- A short readout document like `L63_V2_PILOT_READOUT.md`.
- If diagnostics succeed, first OOD mechanism figure draft.

Owner:

- GPT/Codex runs implementation and aggregation after consensus.

## Rationale

The L63 pilot already tells us where signal lives. Full-grid expansion would
spend many runs on clean/full-horizon cells or floor cells that cannot change
the paper. Targeted expansion gives higher statistical confidence exactly in
the transition band.

We should not jump directly to massive L96/Rössler runs until the v2 runner is
ported and smoke-tested, because L96/Rössler are more expensive and will need
system-specific Lyapunov/gap metadata.

## Alternatives Considered

| Alternative | Why not now |
|---|---|
| Expand full L63 grid to 20 seeds | Too much compute on obvious clean/floor cells. |
| Go straight to L96 N=20 20-seed full grid | Expensive before v2 runner and transition cells are validated. |
| Focus only on tokenizer OOD plot | Important, but without replicated v2 thresholds the plot may explain only L63. |
| Add SAITS/BRITS baseline immediately | Useful, but first decide the scenario subset so baseline cost is bounded. |

## Risks / Stop Conditions

- Risk: L63 targeted expansion shows high variance.
  - Stop condition: if paired bootstrap intervals remain inconclusive after 10
    seeds on key cells, expand only those cells to 20 seeds.
- Risk: L96/Rössler v2 runner port exposes system-specific bugs.
  - Stop condition: if smoke tests fail, fix runner before launching 5-seed runs.
- Risk: CSDI→Panda gains shrink with more seeds.
  - Recovery: reposition claim toward corruption-aware preprocessing sometimes
    helps but failure law remains patch/gap dependent.

## Question For Reviewer

Claude should critique:

1. Is targeted L63 expansion the right next step, or should L96/Rössler come first?
2. Are the selected configs sufficient to support a top-conference mechanism paper?
3. Should SAITS/BRITS or EnKF/LETKF be inserted before more v2 expansions?
