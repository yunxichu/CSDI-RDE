"""Plot L96 N=20 phase-transition curves across M3 backbones + baselines.

Reads aggregated results from aggregate_pt_l96_m3alt.py output JSON and
produces a single-panel figure: VPT@1.0 (mean ± std) vs harshness scenario,
one line per method, highlighting:
  - SVGP (legacy) — collapses early
  - DeepEDM (new) — maintains VPT at S0-S2, transitions at S3-S4
  - FNO (new alt) — similar shape, less sharp
  - Panda-72M — strongest absolute numbers, sharpest transition
  - Parrot — degrades smoothly

This is the §5.7 main figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "week1" / "results"
FIGS = REPO / "experiments" / "week1" / "figures"

SCENARIOS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
SC_IDX = {s: i for i, s in enumerate(SCENARIOS)}

LINES = [
    ("panda",         "Panda-72M",                 "C1", "-",  "o"),
    ("ours_csdi",     "Ours (CSDI + DeepEDM)",     "C3", "-",  "s"),
    ("ours_deepedm",  "Ours (AR-K + DeepEDM)",     "C0", "-",  "D"),
    ("ours_fno",      "Ours (AR-K + FNO)",         "C2", "--", "^"),
    ("ours_svgp",     "Ours (AR-K + SVGP, legacy)","C5", ":",  "P"),
    ("parrot",        "Parrot (1-NN delay)",       "C4", ":",  "v"),
    ("persist",       "Persist",                   "gray","-.", "x"),
]


def main(merged_path: Path = RESULTS / "pt_l96_m3alt_merged.json",
         out: Path = FIGS / "pt_l96_m3alt_phase_transition.png") -> None:
    merged = json.loads(merged_path.read_text())
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    xs = np.arange(len(SCENARIOS))
    for key, label, color, ls, marker in LINES:
        if key not in merged:
            continue
        cells = merged[key]
        m = [cells[s]["vpt10_mean"] if s in cells else np.nan for s in SCENARIOS]
        sd = [cells[s]["vpt10_std"] if s in cells else np.nan for s in SCENARIOS]
        m = np.array(m); sd = np.array(sd)
        ax.errorbar(xs, m, yerr=sd, label=label, color=color, linestyle=ls,
                    marker=marker, capsize=3, linewidth=2, markersize=7, alpha=0.9)
    ax.set_xticks(xs); ax.set_xticklabels(SCENARIOS)
    ax.set_xlabel("harshness scenario (S0=clean → S6=extreme sparse+noisy)")
    ax.set_ylabel("VPT@1.0 (Lyapunov times, mean ± std, n=5 seeds)")
    ax.set_title("Lorenz96 N=20 phase transition — M3 backbone swap (§5.7)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, None)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
