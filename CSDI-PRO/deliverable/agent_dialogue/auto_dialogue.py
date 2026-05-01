#!/usr/bin/env python3
"""Lightweight GPT/Codex <-> Claude dialogue orchestrator.

This script is intentionally provider-tolerant:

- Codex side: uses the local `codex exec` command when available.
- Claude side:
  1. uses `CLAUDE_REVIEW_CMD` if configured,
  2. else uses Anthropic Messages API if `ANTHROPIC_API_KEY` and
     `ANTHROPIC_MODEL` are set,
  3. else writes a prompt file for manual paste into Claude.

No API key values are printed or written.

Example:

  python deliverable/agent_dialogue/auto_dialogue.py status

  python deliverable/agent_dialogue/auto_dialogue.py claude-review \
      --proposal deliverable/agent_dialogue/2026-04-26_next_v2_runs_proposal.md

  python deliverable/agent_dialogue/auto_dialogue.py codex-consensus \
      --proposal deliverable/agent_dialogue/2026-04-26_next_v2_runs_proposal.md \
      --review deliverable/agent_dialogue/2026-04-26_next_v2_runs_claude_review.md

To use a Claude CLI, set a command template. The template may reference
`{prompt_file}` and `{output_file}`. Example:

  export CLAUDE_REVIEW_CMD='claude -p "$(cat {prompt_file})"'

If the command writes to stdout, this script captures stdout into the review
file. If it writes directly to `{output_file}`, that is also accepted.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import urllib.request
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DIALOGUE_DIR = Path(__file__).resolve().parent
PROTOCOL = REPO / "deliverable" / "DUAL_MODEL_COLLAB_PROTOCOL.md"
REVIEW_TEMPLATE = DIALOGUE_DIR / "review_template.md"
CONSENSUS_TEMPLATE = DIALOGUE_DIR / "consensus_template.md"
DECISION_LOG = DIALOGUE_DIR / "decision_log.md"


def _topic_from_proposal(path: Path) -> str:
    name = path.stem
    if name.endswith("_proposal"):
        name = name[: -len("_proposal")]
    return name


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run(cmd: list[str], *, stdin: str | None = None, cwd: Path = REPO) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )


def provider_status() -> str:
    rows = []
    rows.append(f"repo: {REPO}")
    rows.append(f"codex_cli: {shutil.which('codex') or '-'}")
    rows.append(f"claude_cli: {shutil.which('claude') or '-'}")
    rows.append(f"CLAUDE_REVIEW_CMD: {'set' if os.environ.get('CLAUDE_REVIEW_CMD') else '-'}")
    rows.append(f"ANTHROPIC_API_KEY: {'set' if os.environ.get('ANTHROPIC_API_KEY') else '-'}")
    rows.append(f"ANTHROPIC_MODEL: {os.environ.get('ANTHROPIC_MODEL') or '-'}")
    return "\n".join(rows) + "\n"


def build_claude_prompt(proposal_path: Path) -> str:
    proposal = _read(proposal_path)
    protocol = _read(PROTOCOL) if PROTOCOL.exists() else ""
    template = _read(REVIEW_TEMPLATE)
    return textwrap.dedent(f"""\
    You are Claude acting as an independent research collaborator on the CSDI-PRO paper project.

    Your task: review the proposal below and respond strictly as a Review Packet.
    Be skeptical but constructive. Focus on whether the proposed next action is
    sufficient for a top-conference mechanism paper, whether it wastes compute,
    and whether it misses a necessary baseline or diagnostic.

    Use this review template:

    {template}

    Project collaboration protocol:

    {protocol}

    Proposal to review:

    {proposal}
    """)


def build_codex_consensus_prompt(proposal_path: Path, review_path: Path) -> str:
    proposal = _read(proposal_path)
    review = _read(review_path)
    protocol = _read(PROTOCOL) if PROTOCOL.exists() else ""
    template = _read(CONSENSUS_TEMPLATE)
    return textwrap.dedent(f"""\
    You are GPT/Codex acting as one side of a dual-model research collaboration.

    Your task: write the final Consensus Packet from the proposal and Claude review below.
    Do not edit files. Do not run experiments. Produce a concise but actionable consensus.
    If the review rejects the proposal or requires another round, say that clearly and
    do not pretend consensus exists.

    Use this consensus template:

    {template}

    Project collaboration protocol:

    {protocol}

    Proposal:

    {proposal}

    Claude review:

    {review}
    """)


def call_claude_api(prompt: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    model = os.environ.get("ANTHROPIC_MODEL")
    if not api_key or not model:
        raise RuntimeError("ANTHROPIC_API_KEY and ANTHROPIC_MODEL are required for API mode")
    body = {
        "model": model,
        "max_tokens": int(os.environ.get("ANTHROPIC_MAX_TOKENS", "4096")),
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    parts = []
    for item in data.get("content", []):
        if item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "\n".join(parts).strip()


def call_claude(prompt: str, prompt_path: Path, output_path: Path) -> str:
    cmd_template = os.environ.get("CLAUDE_REVIEW_CMD")
    if cmd_template:
        rendered = cmd_template.format(
            prompt_file=str(prompt_path),
            output_file=str(output_path),
        )
        proc = subprocess.run(
            rendered,
            shell=True,
            cwd=str(REPO),
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"CLAUDE_REVIEW_CMD failed with exit {proc.returncode}\n"
                f"stderr:\n{proc.stderr}"
            )
        if output_path.exists() and output_path.read_text(encoding="utf-8").strip():
            return output_path.read_text(encoding="utf-8")
        return proc.stdout.strip()

    if os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("ANTHROPIC_MODEL"):
        return call_claude_api(prompt)

    raise RuntimeError("No automatic Claude provider configured")


def call_codex_consensus(prompt: str, output_path: Path) -> str:
    if not shutil.which("codex"):
        raise RuntimeError("codex CLI not found")
    cmd = [
        "codex",
        "exec",
        "-C",
        str(REPO),
        "-s",
        "read-only",
        "-a",
        "never",
        "-o",
        str(output_path),
        "-",
    ]
    proc = _run(cmd, stdin=prompt)
    if proc.returncode != 0:
        raise RuntimeError(
            f"codex exec failed with exit {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    if output_path.exists() and output_path.read_text(encoding="utf-8").strip():
        return output_path.read_text(encoding="utf-8")
    return proc.stdout.strip()


def command_status(_: argparse.Namespace) -> int:
    print(provider_status(), end="")
    return 0


def command_claude_review(args: argparse.Namespace) -> int:
    proposal = Path(args.proposal).resolve()
    topic = _topic_from_proposal(proposal)
    prompt_path = DIALOGUE_DIR / f"{topic}_claude_prompt.md"
    output_path = Path(args.out).resolve() if args.out else DIALOGUE_DIR / f"{topic}_claude_review.md"
    prompt = build_claude_prompt(proposal)
    _write(prompt_path, prompt)

    try:
        review = call_claude(prompt, prompt_path, output_path)
    except RuntimeError as exc:
        print(f"[manual fallback] {exc}")
        print(f"[manual fallback] Claude prompt written to: {prompt_path}")
        print("Paste that file into Claude, then save the reply as:")
        print(output_path)
        return 2

    _write(output_path, review.rstrip() + "\n")
    print(f"[saved] {output_path}")
    return 0


def command_codex_consensus(args: argparse.Namespace) -> int:
    proposal = Path(args.proposal).resolve()
    review = Path(args.review).resolve()
    topic = _topic_from_proposal(proposal)
    prompt_path = DIALOGUE_DIR / f"{topic}_consensus_prompt.md"
    output_path = Path(args.out).resolve() if args.out else DIALOGUE_DIR / f"{topic}_consensus.md"
    prompt = build_codex_consensus_prompt(proposal, review)
    _write(prompt_path, prompt)
    consensus = call_codex_consensus(prompt, output_path)
    _write(output_path, consensus.rstrip() + "\n")
    print(f"[saved] {output_path}")
    return 0


def command_run_round(args: argparse.Namespace) -> int:
    rc = command_claude_review(args)
    if rc != 0:
        return rc
    proposal = Path(args.proposal).resolve()
    topic = _topic_from_proposal(proposal)
    review = Path(args.out).resolve() if args.out else DIALOGUE_DIR / f"{topic}_claude_review.md"
    consensus_args = argparse.Namespace(
        proposal=str(proposal),
        review=str(review),
        out=args.consensus_out,
    )
    return command_codex_consensus(consensus_args)


def command_append_decision(args: argparse.Namespace) -> int:
    consensus = Path(args.consensus).resolve()
    text = _read(consensus).rstrip()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with DECISION_LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n## {stamp} — Accepted Consensus\n\n")
        f.write(text)
        f.write("\n")
    print(f"[appended] {DECISION_LOG}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("status", help="show available dialogue providers")
    p.set_defaults(func=command_status)

    p = sub.add_parser("claude-review", help="ask Claude to review a proposal, or create manual prompt")
    p.add_argument("--proposal", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=command_claude_review)

    p = sub.add_parser("codex-consensus", help="ask Codex to write consensus from proposal + Claude review")
    p.add_argument("--proposal", required=True)
    p.add_argument("--review", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=command_codex_consensus)

    p = sub.add_parser("run-round", help="run Claude review then Codex consensus if providers are configured")
    p.add_argument("--proposal", required=True)
    p.add_argument("--out", default=None, help="Claude review output path")
    p.add_argument("--consensus-out", default=None)
    p.set_defaults(func=command_run_round)

    p = sub.add_parser("append-decision", help="append an accepted consensus to decision_log.md")
    p.add_argument("--consensus", required=True)
    p.set_defaults(func=command_append_decision)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
