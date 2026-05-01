#!/usr/bin/env python3
"""Append-only file dialogue helper for GPT/Codex and Claude sidebar.

The user can keep `LIVE_DIALOGUE.md` open in both model sidebars. Claude appends
its turn directly to that file; GPT/Codex can then inspect the file, append a
reply, and stop once both approval markers exist.

This script does not call any external model. It only checks the shared file and
prints the next expected action.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import re
import time
from pathlib import Path

DIALOGUE = Path(__file__).resolve().parent / "LIVE_DIALOGUE.md"
REPO = Path(__file__).resolve().parents[2]


def read_dialogue() -> str:
    return DIALOGUE.read_text(encoding="utf-8")


def write_dialogue(text: str) -> None:
    DIALOGUE.write_text(text, encoding="utf-8")


def set_status(text: str, status: str) -> str:
    return re.sub(r"^Status: .*$", f"Status: {status}", text, count=1, flags=re.MULTILINE)


def next_gpt_index(text: str) -> int:
    nums = [int(x) for x in re.findall(r"^## GPT_TURN_(\d+)", text, flags=re.MULTILINE)]
    return max(nums, default=0) + 1


def build_auto_gpt_prompt(text: str) -> str:
    idx = next_gpt_index(text)
    return f"""You are GPT/Codex participating in a live file-based dialogue with Claude.

Read the full dialogue below and output ONLY the next append block. Do not run
commands. Do not edit files. Do not include explanations outside the block.

Rules:
- If Claude approved the revised plan, output a concise `## GPT_CONSENSUS_CANDIDATE`
  that summarizes the agreed plan and include `GPT_FINAL_APPROVAL: yes`.
- If Claude requested changes, output `## GPT_TURN_{idx}` addressing the changes.
- Do not claim user approval. Execution is still blocked until user approval.
- Keep the block concrete: decision, action order, compute budget, stop conditions.

Full dialogue:

{text}
"""


def auto_append_gpt_turn() -> None:
    text = read_dialogue()
    prompt = build_auto_gpt_prompt(text)
    out = DIALOGUE.parent / ".auto_gpt_next.md"
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
        str(out),
        "-",
    ]
    proc = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
        cwd=str(REPO),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"codex exec failed with exit {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    block = out.read_text(encoding="utf-8").strip() if out.exists() else proc.stdout.strip()
    if not block:
        raise RuntimeError("codex exec produced an empty GPT dialogue block")
    with DIALOGUE.open("a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(block)
        f.write("\n")


def command_status(_: argparse.Namespace) -> int:
    text = read_dialogue()
    gpt_ok = bool(re.search(r"^GPT_FINAL_APPROVAL: yes$", text, flags=re.MULTILINE))
    claude_ok = bool(re.search(r"^CLAUDE_APPROVAL: yes$", text, flags=re.MULTILINE))
    claude_turns = len(re.findall(r"^## CLAUDE_TURN_", text, flags=re.MULTILINE))
    gpt_turns = len(re.findall(r"^## GPT_TURN_", text, flags=re.MULTILINE))
    consensus = "## GPT_CONSENSUS_CANDIDATE" in text
    print(f"file: {DIALOGUE}")
    print(f"gpt_turns: {gpt_turns}")
    print(f"claude_turns: {claude_turns}")
    print(f"consensus_candidate: {consensus}")
    print(f"GPT_FINAL_APPROVAL: {gpt_ok}")
    print(f"CLAUDE_APPROVAL: {claude_ok}")
    if gpt_ok and claude_ok:
        print("next_action: dialogue complete; user may approve execution")
    elif consensus and not claude_ok:
        print("next_action: waiting for Claude to approve or request changes")
    elif claude_turns >= gpt_turns:
        print("next_action: GPT/Codex should append the next turn")
    else:
        print("next_action: waiting for Claude to append the next turn")
    return 0


def dialogue_state(text: str) -> str:
    gpt_ok = bool(re.search(r"^GPT_FINAL_APPROVAL: yes$", text, flags=re.MULTILINE))
    claude_ok = bool(re.search(r"^CLAUDE_APPROVAL: yes$", text, flags=re.MULTILINE))
    claude_turns = len(re.findall(r"^## CLAUDE_TURN_", text, flags=re.MULTILINE))
    gpt_turns = len(re.findall(r"^## GPT_TURN_", text, flags=re.MULTILINE))
    consensus = "## GPT_CONSENSUS_CANDIDATE" in text
    if gpt_ok and claude_ok:
        return "complete"
    if consensus and not claude_ok:
        return "awaiting_claude_approval"
    if claude_turns >= gpt_turns:
        return "gpt_should_append"
    return "awaiting_claude"


def command_watch(args: argparse.Namespace) -> int:
    last_hash = None
    last_state = None
    last_change_time = time.time()
    fast_interval = args.interval if args.interval is not None else args.fast_interval
    print(f"watching: {DIALOGUE}")
    print(f"fast_interval_s: {fast_interval}")
    print(f"fast_window_s: {args.fast_window}")
    print(f"slow_interval_s: {args.slow_interval}")
    print("press Ctrl-C to stop")
    while True:
        text = read_dialogue()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        state = dialogue_state(text)
        if digest != last_hash or state != last_state:
            last_change_time = time.time()
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            claude_turns = len(re.findall(r"^## CLAUDE_TURN_", text, flags=re.MULTILINE))
            gpt_turns = len(re.findall(r"^## GPT_TURN_", text, flags=re.MULTILINE))
            print(f"[{now}] changed state={state} gpt_turns={gpt_turns} claude_turns={claude_turns}", flush=True)
            if state == "gpt_should_append":
                print("[action] GPT/Codex should read LIVE_DIALOGUE.md and append the next turn.", flush=True)
                if args.auto_gpt:
                    try:
                        auto_append_gpt_turn()
                        print("[auto-gpt] appended GPT block", flush=True)
                        text = read_dialogue()
                        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                        state = dialogue_state(text)
                    except Exception as exc:
                        print(f"[auto-gpt] failed: {exc}", flush=True)
            elif state == "complete":
                print("[action] consensus complete; stop writing and wait for user approval.", flush=True)
                if args.exit_on_complete:
                    return 0
            last_hash = digest
            last_state = state
        idle_s = time.time() - last_change_time
        sleep_s = fast_interval if idle_s < args.fast_window else args.slow_interval
        time.sleep(sleep_s)


def command_mark_complete(_: argparse.Namespace) -> int:
    text = read_dialogue()
    if "GPT_FINAL_APPROVAL: yes" not in text or "CLAUDE_APPROVAL: yes" not in text:
        raise SystemExit("Both approval markers are required before marking complete.")
    text = set_status(text, "consensus_reached")
    write_dialogue(text)
    print("marked complete")
    return 0


def command_show(_: argparse.Namespace) -> int:
    print(read_dialogue())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("status")
    p.set_defaults(func=command_status)
    p = sub.add_parser("watch")
    p.add_argument("--interval", type=float, default=None,
                   help="legacy alias for --fast-interval")
    p.add_argument("--fast-interval", type=float, default=20.0,
                   help="poll interval while inside --fast-window")
    p.add_argument("--fast-window", type=float, default=60.0,
                   help="seconds after a change to keep fast polling")
    p.add_argument("--slow-interval", type=float, default=600.0,
                   help="poll interval after the fast window expires")
    p.add_argument("--exit-on-complete", action="store_true")
    p.add_argument("--auto-gpt", action="store_true",
                   help="automatically append GPT blocks with codex exec when Claude has replied")
    p.set_defaults(func=command_watch)
    p = sub.add_parser("mark-complete")
    p.set_defaults(func=command_mark_complete)
    p = sub.add_parser("show")
    p.set_defaults(func=command_show)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
