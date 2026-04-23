"""Generate Figure X4 for §5.X4 — Panda OOD KL hard-threshold visualization.

Two panels:
  (a) JS divergence vs s (σ=0 and σ=0.5 lines overlaid) — shows the
      3.1× jump between s=0.70 and s=0.85 on the σ=0 line
  (b) Linear-segment fraction (curvature < 0.01) vs s — shows the 21×
      jump from 0.6% to 12.9% in the same window

Reads: experiments/week2_modules/results/panda_ood_kl_v1.json
Saves: experiments/week2_modules/figures/panda_ood_kl_threshold.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


SRC = Path(__file__).resolve().parent / "results" / "panda_ood_kl_v1.json"
OUT = Path(__file__).resolve().parent / "figures" / "panda_ood_kl_threshold.png"


def main():
    data = json.loads(SRC.read_text())
    records = data["records"]

    # Split by sigma
    by_sigma = {}
    for r in records:
        sig = r["sigma"]
        by_sigma.setdefault(sig, []).append(r)
    for sig in by_sigma:
        by_sigma[sig].sort(key=lambda r: r["s"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    # Panel (a): JS vs s
    ax = axes[0]
    colors = {0.0: "C0", 0.5: "C3"}
    markers = {0.0: "o", 0.5: "s"}
    for sig, recs in sorted(by_sigma.items()):
        s_vals = [r["s"] for r in recs if not r.get("is_ref")]
        js_vals = [r["js_vs_ref"] for r in recs if not r.get("is_ref")]
        ax.plot(s_vals, js_vals, marker=markers[sig], color=colors[sig],
                label=f"σ/σ_attr = {sig}", linewidth=2, markersize=8)
    # Annotate the jump
    ax.axvspan(0.70, 0.85, color="red", alpha=0.1, label="hard threshold zone")
    ax.annotate("3.1× JS jump\n(0.042→0.131)",
                xy=(0.85, 0.131), xytext=(0.55, 0.25),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                fontsize=9, color="red")
    ax.set_xlabel("sparsity s")
    ax.set_ylabel("JS divergence (patch curvature) vs clean ref")
    ax.set_title("(a) Patch-distribution JS vs s\nsupports Theorem 2(b) lemma L2 (§5.X4)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, 0.8)

    # Panel (b): Linear-segment fraction
    ax = axes[1]
    for sig, recs in sorted(by_sigma.items()):
        s_vals = [r["s"] for r in recs]
        frac_vals = [r["curvature_low_frac"] for r in recs]
        ax.plot(s_vals, frac_vals, marker=markers[sig], color=colors[sig],
                label=f"σ/σ_attr = {sig}", linewidth=2, markersize=8)
    ax.axvspan(0.70, 0.85, color="red", alpha=0.1)
    ax.annotate("21× jump\n(0.6%→12.9%)",
                xy=(0.85, 0.129), xytext=(0.50, 0.30),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
                fontsize=9, color="red")
    # S3 and U3/G20 reference
    for sc, x in [("S3 (s=0.60)", 0.60), ("U3/G20 (s=0.70)", 0.70)]:
        ax.axvline(x, color="gray", lw=0.5, ls="--", alpha=0.7)
        ax.text(x, 0.52, sc, rotation=90, ha="right", va="top", fontsize=8, color="gray")
    ax.set_xlabel("sparsity s")
    ax.set_ylabel("fraction of patches with curvature < 0.01\n(linear-segment indicator)")
    ax.set_title("(b) Linear-segment fraction vs s\nthreshold at s ≈ 0.85 (patch_length=16 geometric condition)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, 0.6)

    fig.suptitle("Figure X4: Panda OOD KL hard threshold — §5.X4 closes Theorem 2(b) lemma L2", fontsize=12)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"[saved] {OUT}")
    plt.close()


if __name__ == "__main__":
    main()
