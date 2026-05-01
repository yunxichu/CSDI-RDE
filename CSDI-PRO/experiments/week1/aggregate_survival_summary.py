"""Build survival-probability summaries for phase-transition result JSONs.

This is intentionally lightweight and paper-facing: it reads per-seed `records`
from the existing phase-transition runs, then reports mean +/- sd, bootstrap CI,
Pr(VPT > 0), and Pr(VPT > 0.5). The survival columns are the variance antidote
for VPT-heavy headline tables.

Run:
  python -m experiments.week1.aggregate_survival_summary
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

REPO = Path(__file__).resolve().parents[2]

DEFAULT_SOURCES = [
    ("L63", "experiments/week1/results/pt_v2_with_panda_n5_small.json", None),
    ("L63", "experiments/week1/results/pt_v2_csdi_upgrade_n5.json", {"ours_csdi"}),
    ("L96_N20", "experiments/week1/results/pt_l96_l96_N20_m3alt_5seed.json", None),
    ("L96_N20", "experiments/week1/results/pt_l96_l96_N20_panda_5seed.json", None),
    ("L96_N20", "experiments/week1/results/pt_l96_l96_N20_ours_csdi_deepedm_5seed.json", None),
    ("Rossler", "deliverable/results/Rossler_5seed.json", None),
]

METHOD_LABELS = {
    "ours": "Ours",
    "ours_csdi": "Ours (CSDI)",
    "ours_deepedm": "Ours (AR-K + DeepEDM)",
    "ours_fno": "Ours (AR-K + FNO)",
    "ours_csdi_deepedm": "Ours (CSDI + DeepEDM)",
    "ours_csdi_svgp": "Ours (CSDI + SVGP)",
    "panda": "Panda-72M",
    "parrot": "Parrot",
    "chronos": "Chronos",
    "persist": "Persist",
}

SCENARIO_ORDER = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
METHOD_ORDER = [
    "ours_csdi",
    "ours_csdi_deepedm",
    "ours",
    "ours_deepedm",
    "ours_csdi_svgp",
    "panda",
    "parrot",
    "chronos",
    "persist",
]


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, len(values), size=len(values))
        boot[i] = values[idx].mean()
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def iter_records() -> Iterable[tuple[str, dict]]:
    for system, rel_path, methods in DEFAULT_SOURCES:
        path = REPO / rel_path
        if not path.exists():
            print(f"[warn] missing {path}")
            continue
        doc = json.loads(path.read_text())
        for rec in doc.get("records", []):
            if rec.get("error"):
                continue
            method = rec.get("method")
            if methods is not None and method not in methods:
                continue
            rec = dict(rec)
            if system == "L96_N20" and method == "ours_csdi":
                rec["method"] = "ours_csdi_deepedm"
            yield system, rec


def summarize(metric: str) -> dict:
    groups: dict[tuple[str, str, str], list[float]] = {}
    for system, rec in iter_records():
        key = (system, rec["method"], rec["scenario"])
        val = rec.get(metric)
        if val == val:
            groups.setdefault(key, []).append(float(val))

    out = {}
    for key, vals in groups.items():
        arr = np.array(vals, dtype=np.float64)
        lo, hi = bootstrap_ci(arr)
        out[key] = {
            "n": int(len(arr)),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "ci_lo": lo,
            "ci_hi": hi,
            "surv_gt0": float((arr > 0).mean()),
            "surv_gt05": float((arr > 0.5).mean()),
        }
    return out


def make_markdown(summary: dict, metric: str) -> str:
    systems = sorted({k[0] for k in summary})
    chunks = [
        f"# Phase-Transition Survival Summary ({metric.upper()})",
        "",
        "Cell format: `mean +/- sd [95% bootstrap CI]; Pr(>0), Pr(>0.5)`.",
    ]
    for system in systems:
        methods = [m for m in METHOD_ORDER if any(k[:2] == (system, m) for k in summary)]
        scenarios = [s for s in SCENARIO_ORDER if any(k[0] == system and k[2] == s for k in summary)]
        chunks.extend(["", f"## {system}", ""])
        chunks.append("| Method | " + " | ".join(scenarios) + " |")
        chunks.append("|:---|" + ":-:|" * len(scenarios))
        for method in methods:
            row = [METHOD_LABELS.get(method, method)]
            for sc in scenarios:
                cell = summary.get((system, method, sc))
                if cell is None:
                    row.append("-")
                    continue
                row.append(
                    f"{cell['mean']:.2f} +/- {cell['std']:.2f} "
                    f"[{cell['ci_lo']:.2f}, {cell['ci_hi']:.2f}]; "
                    f"{cell['surv_gt0']:.0%}, {cell['surv_gt05']:.0%}"
                )
            chunks.append("| " + " | ".join(row) + " |")
    return "\n".join(chunks) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="vpt10", choices=["vpt03", "vpt05", "vpt10"])
    ap.add_argument("--out", default="experiments/week1/results/phase_transition_survival_summary.md")
    args = ap.parse_args()

    summary = summarize(args.metric)
    md = make_markdown(summary, args.metric)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
