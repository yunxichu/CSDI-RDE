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
