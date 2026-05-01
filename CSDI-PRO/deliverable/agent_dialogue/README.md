# Agent Dialogue Workspace

This folder is the shared scratchpad for GPT/Codex and Claude.

## How To Use

Manual flow:

1. Ask GPT/Codex to create a proposal using `proposal_template.md`.
2. Paste that proposal to Claude.
3. Ask Claude to reply using `review_template.md`.
4. Paste Claude's review back to GPT/Codex.
5. GPT/Codex writes a final consensus using `consensus_template.md`.
6. Append the accepted decision to `decision_log.md`.

Automated / semi-automated flow:

```bash
python deliverable/agent_dialogue/auto_dialogue.py status
python deliverable/agent_dialogue/auto_dialogue.py claude-review \
  --proposal deliverable/agent_dialogue/2026-04-26_next_v2_runs_proposal.md
python deliverable/agent_dialogue/auto_dialogue.py codex-consensus \
  --proposal deliverable/agent_dialogue/2026-04-26_next_v2_runs_proposal.md \
  --review deliverable/agent_dialogue/2026-04-26_next_v2_runs_claude_review.md
```

If Claude is not configured locally, the script writes a Claude prompt file and
tells you where to save Claude's reply.

To enable automatic Claude review, use one of:

```bash
export CLAUDE_REVIEW_CMD='claude -p "$(cat {prompt_file})"'
```

or:

```bash
export ANTHROPIC_API_KEY=...
export ANTHROPIC_MODEL=...
```

Keep each packet short enough to paste comfortably. Prefer crisp evidence and
concrete commands over broad discussion.

## Naming Convention

Use:

```text
YYYY-MM-DD_topic_proposal.md
YYYY-MM-DD_topic_claude_review.md
YYYY-MM-DD_topic_consensus.md
```

Example:

```text
2026-04-26_next_v2_runs_proposal.md
2026-04-26_next_v2_runs_claude_review.md
2026-04-26_next_v2_runs_consensus.md
```

## Current Rule

No substantial new experiments or paper-direction changes should happen until
there is a consensus packet and user approval.

## Live Sidebar Dialogue

If Claude is available as an IDE/sidebar agent rather than a CLI/API, use:

```text
deliverable/agent_dialogue/LIVE_DIALOGUE.md
```

Open that file in both GPT/Codex and Claude. GPT writes `GPT_TURN_n`; Claude
appends `CLAUDE_TURN_n`. When both models agree, the file must contain:

```text
GPT_FINAL_APPROVAL: yes
CLAUDE_APPROVAL: yes
```

Check state with:

```bash
python deliverable/agent_dialogue/live_dialogue.py status
```

Keep a polling watcher running with:

```bash
python deliverable/agent_dialogue/live_dialogue.py watch --interval 20
```

The watcher does not write model content. It only reports when Claude has
changed the file and when GPT/Codex should append the next turn.

Default polling uses backoff: fast polling for 1 minute after a change, then one
check every 10 minutes if no reply arrives.

To let the watcher automatically append GPT/Codex replies when Claude has
written a turn:

```bash
python deliverable/agent_dialogue/live_dialogue.py watch --auto-gpt
```

This still does not run experiments or edit project code. It only appends GPT
dialogue blocks to `LIVE_DIALOGUE.md` using `codex exec`.
