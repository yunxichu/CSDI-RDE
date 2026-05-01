    You are Claude acting as an independent research collaborator on the CSDI-PRO paper project.

    Your task: review the proposal below and respond strictly as a Review Packet.
    Be skeptical but constructive. Focus on whether the proposed next action is
    sufficient for a top-conference mechanism paper, whether it wastes compute,
    and whether it misses a necessary baseline or diagnostic.

    Use this review template:

    # Review Packet

## Verdict

Choose one:

- accept,
- accept with changes,
- reject / needs another round.

## Strong Points

- 

## Concerns

- 

## Required Changes Before Action

- 

## Suggested Better Plan

Only fill this if the proposal needs revision.

## Minimal Consensus Candidate

State the exact plan you would approve.


    Project collaboration protocol:

    # Dual-Model Collaboration Protocol

Date: 2026-04-26

Purpose: use GPT/Codex and Claude as two independent research collaborators.
From this point on, nontrivial project decisions should be made only after a
short written dialogue reaches an explicit consensus.

## Scope

This protocol applies to:

- paper narrative changes,
- new experiment design,
- deciding which experiments to run next,
- interpreting surprising results,
- adding/removing major baselines,
- rewriting key paper sections,
- spending substantial GPU time.

It does not need to apply to:

- typo fixes,
- small formatting changes,
- read-only status checks,
- syntax checks,
- aggregating already-finished results,
- emergency bug fixes required to make a previously agreed run work.

## Shared Medium

Claude and GPT do not need direct API-to-API communication. They communicate
through short Markdown packets that the user can paste between them.

Working folder:

```text
deliverable/agent_dialogue/
```

Key files:

- `README.md`: how to run a dialogue round.
- `proposal_template.md`: GPT/Codex or Claude proposes a plan.
- `review_template.md`: the other model critiques and amends it.
- `consensus_template.md`: final agreed action plan.
- `decision_log.md`: append-only record of accepted decisions.

## Dialogue Rule

For every nontrivial next step:

1. One model writes a **Proposal Packet**.
2. The other model writes a **Review Packet**.
3. The first model writes a **Consensus Packet**.
4. The user approves or asks for another round.
5. Only then do we act.

The desired number of rounds is one. Add a second round only if there is a real
disagreement or the decision affects major compute/paper direction.

## Consensus Standard

A consensus is valid only if it states:

- decision,
- rationale,
- rejected alternatives,
- exact commands or files to change,
- compute budget,
- expected outputs,
- stop conditions,
- owner for the next action.

If Claude and GPT disagree, do not average their opinions. The consensus packet
must identify the disagreement and either:

- resolve it with a concrete test,
- defer the decision,
- or ask the user to choose.

## Decision Levels

### Level 0: No Dialogue Needed

Examples:

- inspect files,
- check GPU status,
- aggregate finished JSON,
- fix a typo,
- run `py_compile`.

Action: proceed directly.

### Level 1: Lightweight Dialogue

Examples:

- choose which already-defined L63 cells to extend,
- select a figure/table layout,
- minor section rewrite.

Action: one proposal + one review + one consensus.

### Level 2: Full Dialogue

Examples:

- start L96/Rössler v2 10-20 seed runs,
- add SAITS/BRITS or EnKF/LETKF baseline,
- change main paper claim,
- move systems between main text and appendix.

Action: proposal + review + consensus + explicit user approval.

## Required Fields For Experiment Decisions

Every experiment consensus must include:

- system: e.g. L63, L96 N=20, Rössler,
- configs: exact scenario names,
- cells/methods,
- seeds,
- GPU count and CPU thread count,
- estimated runtime,
- output filenames,
- success criteria,
- early-stop criteria,
- how the result will change the paper.

## Required Fields For Writing Decisions

Every writing consensus must include:

- target files/sections,
- claim being changed,
- evidence supporting the claim,
- evidence that weakens or bounds the claim,
- text-level action,
- what must not be claimed.

## Current Standing Consensus

As of 2026-04-26, the working consensus is:

> The paper should not sell delaymask/random delay embedding as the main
> contribution. It should sell the failure law: sparse/noisy observation
> corruptions create preprocessing/tokenizer OOD channels in pretrained
> forecasters; corruption-aware imputation is the first mitigation lever, and
> delay-manifold forecasting is a dynamics-structured companion.

Current next likely decision:

> Which v2 experiments should be expanded next: targeted L63 10-20 seeds,
> L96 N=20 v2 runner, Rössler v2 runner, tokenizer OOD plot, or extra baselines?

This decision should go through the dialogue protocol before running.


    Proposal to review:

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

