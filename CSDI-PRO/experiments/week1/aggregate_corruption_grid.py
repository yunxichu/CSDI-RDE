"""Aggregate v2 corruption-grid JSON into a compact Markdown table."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _num(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _fmt(x: Any, nd: int = 2) -> str:
    val = _num(x)
    if val != val:
        return "-"
    return f"{val:.{nd}f}"


def _collect(path: Path) -> list[dict[str, Any]]:
    doc = json.loads(path.read_text())
    return list(doc.get("records", []))


def _group(records: list[dict[str, Any]], metric: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for rec in records:
        label = rec.get("label") or "metadata_only"
        grouped[(rec["scenario"], label)].append(rec)
    return grouped


def _summarize_group(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    first = rows[0]
    vals = np.asarray([
        _num(r.get(metric))
        for r in rows
        if not r.get("error") and not r.get("dry_run") and _num(r.get(metric)) == _num(r.get(metric))
    ], dtype=np.float64)
    keep = np.asarray([_num(r.get("keep_frac")) for r in rows], dtype=np.float64)
    obs_patch = np.asarray([_num(r.get("expected_obs_per_patch")) for r in rows], dtype=np.float64)
    gap = np.asarray([_num(r.get("all_missing_gap_max_lyap")) for r in rows], dtype=np.float64)
    out = {
        "config": first["scenario"],
        "label": first.get("label") or "metadata_only",
        "mask": first.get("mask_regime", "-"),
        "s": first.get("sparsity"),
        "sigma": first.get("noise_std_frac"),
        "keep": float(np.nanmean(keep)) if keep.size else float("nan"),
        "obs_patch": float(np.nanmean(obs_patch)) if obs_patch.size else float("nan"),
        "max_gap_L": float(np.nanmean(gap)) if gap.size else float("nan"),
        "n": int(vals.size),
        "mean": float(np.nanmean(vals)) if vals.size else float("nan"),
        "median": float(np.nanmedian(vals)) if vals.size else float("nan"),
        "sd": float(np.nanstd(vals)) if vals.size else float("nan"),
        "pr_gt0": float(np.mean(vals > 0.0)) if vals.size else float("nan"),
        "pr_gt05": float(np.mean(vals > 0.5)) if vals.size else float("nan"),
    }
    return out


def make_markdown(records: list[dict[str, Any]], metric: str) -> str:
    grouped = _group(records, metric)
    rows = [_summarize_group(v, metric) for _, v in sorted(grouped.items())]

    lines = [f"# Corruption Grid Summary ({metric})", ""]
    lines.append(
        "| config | mask | s | sigma | keep | obs/patch | max gap L | cell | n | mean | median | sd | Pr>0 | Pr>0.5 |"
    )
    lines.append("|:---|:---|---:|---:|---:|---:|---:|:---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                str(row["config"]),
                str(row["mask"]),
                _fmt(row["s"], 2),
                _fmt(row["sigma"], 2),
                _fmt(row["keep"], 3),
                _fmt(row["obs_patch"], 2),
                _fmt(row["max_gap_L"], 2),
                str(row["label"]),
                str(row["n"]),
                _fmt(row["mean"], 2),
                _fmt(row["median"], 2),
                _fmt(row["sd"], 2),
                _fmt(row["pr_gt0"], 2),
                _fmt(row["pr_gt05"], 2),
            ])
            + " |"
        )

    lines.append("")
    lines.append(
        "Design read: obs/patch and max gap L are the first sanity checks. "
        "If adjacent stages barely change either quantity, the stage ladder is too fine; "
        "if they jump by several Lyapunov tenths at once, it is too coarse."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="vpt10", choices=["vpt03", "vpt05", "vpt10"])
    args = ap.parse_args()

    in_path = Path(args.json)
    if not in_path.is_absolute():
        in_path = REPO_ROOT / in_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    text = make_markdown(_collect(in_path), args.metric)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(text)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
