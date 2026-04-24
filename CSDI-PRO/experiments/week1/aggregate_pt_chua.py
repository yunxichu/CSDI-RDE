"""Aggregate Chua PT eval results + markdown table + transition plot."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "week1" / "results"
FIGS = REPO / "experiments" / "week1" / "figures"

SCENARIOS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
METHODS_DISPLAY = {
    "ours_csdi_deepedm": "Ours (CSDI + DeepEDM)",
    "ours_csdi_svgp":    "Ours (CSDI + SVGP)",
    "panda":             "**Panda-72M**",
    "parrot":            "Parrot",
    "persist":           "Persist",
}
LINES = [
    ("panda",             "Panda-72M",             "C1", "-",  "o"),
    ("ours_csdi_deepedm", "Ours (CSDI + DeepEDM)", "C3", "-",  "s"),
    ("ours_csdi_svgp",    "Ours (CSDI + SVGP)",    "C5", ":",  "P"),
    ("parrot",            "Parrot",                "C4", ":",  "v"),
    ("persist",           "Persist",               "gray","-.", "x"),
]


def main(tag: str = "chua_5seed"):
    path = RESULTS / f"pt_chua_{tag}.json"
    if not path.exists():
        print(f"[warn] {path} missing"); return
    doc = json.loads(path.read_text())
    merged = doc.get("summary", {})
    print("\n### Table — Chua VPT@1.0 (mean ± std)\n")
    print("| Method | " + " | ".join(SCENARIOS) + " |")
    print("|:---|" + ":-:|" * len(SCENARIOS))
    for m in ["ours_csdi_deepedm", "ours_csdi_svgp", "panda", "parrot", "persist"]:
        if m not in merged: continue
        cells = merged[m]
        row = [METHODS_DISPLAY[m]]
        for sc in SCENARIOS:
            row.append(f"{cells[sc]['vpt10_mean']:.2f} ± {cells[sc]['vpt10_std']:.2f}"
                       if sc in cells else "—")
        print("| " + " | ".join(row) + " |")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    xs = np.arange(len(SCENARIOS))
    for key, label, color, ls, marker in LINES:
        if key not in merged: continue
        cells = merged[key]
        m_arr = [cells[s]["vpt10_mean"] if s in cells else np.nan for s in SCENARIOS]
        sd_arr = [cells[s]["vpt10_std"] if s in cells else np.nan for s in SCENARIOS]
        ax.errorbar(xs, m_arr, yerr=sd_arr, label=label, color=color, linestyle=ls,
                    marker=marker, capsize=3, linewidth=2, markersize=7, alpha=0.9)
    ax.set_xticks(xs); ax.set_xticklabels(SCENARIOS)
    ax.set_xlabel("harshness scenario")
    ax.set_ylabel("VPT@1.0 (Lyapunov times Λ)")
    ax.set_title(f"Chua double-scroll phase-transition ({tag})")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3); ax.set_ylim(0, None)
    plt.tight_layout()
    out = FIGS / f"pt_chua_{tag}_phase_transition.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else "chua_5seed"
    main(tag)
