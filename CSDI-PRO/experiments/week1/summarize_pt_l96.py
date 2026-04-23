"""Summarize Lorenz96 N=20 phase-transition results (§5.7 cross-system verification).

Reads:  pt_l96_N20_v1.json + pt_l96_N20_v1_seeds34.json (merges all available seeds)
Writes: pt_l96_N20_merged_summary.md + pt_l96_N20_merged.json
Plots:  figures/pt_l96_N20_phase_transition.png

Usage: python -m experiments.week1.summarize_pt_l96
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main():
    candidates = [
        RES / "pt_l96_l96_N20_v1.json",
        RES / "pt_l96_l96_N20_v1_seeds34.json",
    ]
    all_records = []
    meta = None
    for p in candidates:
        if not p.exists():
            print(f"[skip] {p} not found")
            continue
        d = json.loads(p.read_text())
        all_records.extend(d.get("records", []))
        if meta is None:
            meta = d.get("meta", {})
        print(f"[loaded] {p.name}  records={len(d.get('records', []))}")

    print(f"\nTotal merged records: {len(all_records)}")

    # Aggregate by (method, scenario)
    by = defaultdict(list)
    for r in all_records:
        if r.get("error"):
            continue
        k = (r["method"], r["scenario"])
        by[k].append(r)

    scenarios = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
    methods = ["ours", "panda", "parrot"]

    summary = {}
    for m in methods:
        summary[m] = {}
        for s in scenarios:
            rs = by.get((m, s), [])
            if not rs:
                continue
            vpt10 = np.array([r["vpt10"] for r in rs])
            rmse = np.array([r["rmse_norm_first100"] for r in rs])
            summary[m][s] = dict(
                n=len(rs),
                vpt10_mean=float(vpt10.mean()),
                vpt10_std=float(vpt10.std()),
                vpt10_values=[float(v) for v in vpt10],
                rmse_mean=float(rmse.mean()),
                rmse_std=float(rmse.std()),
            )

    # Print table
    print("\n=== Lorenz96 N=20 Phase Transition (VPT@1.0 mean ± std) ===")
    header = "Method   " + " ".join(f"{s:>12s}" for s in scenarios)
    print(header)
    for m in methods:
        row = f"{m:8s} "
        for s in scenarios:
            cell = summary[m].get(s)
            if cell is None:
                row += f"{'-':>12s} "
            else:
                row += f"{cell['vpt10_mean']:6.2f}±{cell['vpt10_std']:4.2f} (n={cell['n']:2d}) "
        print(row)

    # Phase transition signal (S0 → S3)
    print("\n=== S0 → S3 drop (Phase Transition signal) ===")
    for m in methods:
        if "S0" in summary[m] and "S3" in summary[m]:
            s0 = summary[m]["S0"]["vpt10_mean"]
            s3 = summary[m]["S3"]["vpt10_mean"]
            drop_pct = (1 - s3 / max(s0, 1e-6)) * 100
            print(f"  {m:8s}: S0={s0:.2f} → S3={s3:.2f}  drop={drop_pct:+.0f}%")

    # Save merged JSON
    out_json = RES / "pt_l96_N20_merged.json"
    out_json.write_text(json.dumps(dict(
        meta=meta,
        methods=methods,
        scenarios=scenarios,
        summary=summary,
        total_records=len(all_records),
    ), indent=2))
    print(f"\n[saved] {out_json}")

    # Figure: VPT@1.0 with error bars vs scenario
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {"ours": "#1b9e77", "panda": "#9467bd", "parrot": "C0"}
    markers = {"ours": "D", "panda": "^", "parrot": "s"}
    labels = {"ours": "Ours (AR-K M1)", "panda": "Panda-72M", "parrot": "Context-Parroting"}
    x = np.arange(len(scenarios))
    for m in methods:
        ys, es = [], []
        for s in scenarios:
            cell = summary[m].get(s)
            if cell is None:
                ys.append(np.nan); es.append(0)
            else:
                ys.append(cell["vpt10_mean"]); es.append(cell["vpt10_std"])
        ax.errorbar(x, ys, yerr=es, marker=markers[m], color=colors[m],
                    label=labels[m], linewidth=2, capsize=3, markersize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("VPT@1.0 (Lyapunov times)")
    ax.set_xlabel("Harshness scenario")
    ax.set_title(f"Lorenz96 N=20, F=8 — Phase Transition (n_seeds={max(summary[m]['S0']['n'] for m in methods if 'S0' in summary[m])})\n"
                  f"Cross-system verification of §5.2 Lorenz63 phase transition")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axvspan(2.5, 3.5, alpha=0.1, color="red")  # highlight S3
    ax.text(3.0, ax.get_ylim()[1] * 0.9, "S3 = main\ntransition", ha="center", fontsize=8, color="red")
    plt.tight_layout()
    out_fig = FIG / "pt_l96_N20_phase_transition.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_fig}")
    plt.close()

    # Markdown summary (for paper)
    md = [f"# Lorenz96 N=20 Phase Transition Summary\n"]
    md.append(f"Generated from {len(all_records)} records across candidate JSON files.\n")
    md.append(f"Methods: {methods}; Scenarios: {scenarios}.\n")
    md.append(f"\n## VPT@1.0 table (mean ± std, n seeds per cell)\n")
    md.append("| Method | " + " | ".join(scenarios) + " |")
    md.append("|" + "---|" * (1 + len(scenarios)))
    for m in methods:
        row = f"| {labels[m]} "
        for s in scenarios:
            cell = summary[m].get(s)
            if cell is None:
                row += "| — "
            else:
                row += f"| {cell['vpt10_mean']:.2f} ± {cell['vpt10_std']:.2f} (n={cell['n']}) "
        row += "|"
        md.append(row)
    md.append(f"\n## S0 → S3 drop (Phase Transition signal)\n")
    md.append("| Method | S0 VPT | S3 VPT | % drop |")
    md.append("|---|---|---|---|")
    for m in methods:
        if "S0" in summary[m] and "S3" in summary[m]:
            s0 = summary[m]["S0"]["vpt10_mean"]
            s3 = summary[m]["S3"]["vpt10_mean"]
            drop_pct = (1 - s3 / max(s0, 1e-6)) * 100
            md.append(f"| {labels[m]} | {s0:.2f} | {s3:.2f} | {drop_pct:+.0f}% |")

    out_md = RES / "pt_l96_N20_merged_summary.md"
    out_md.write_text("\n".join(md))
    print(f"[saved] {out_md}")


if __name__ == "__main__":
    main()
