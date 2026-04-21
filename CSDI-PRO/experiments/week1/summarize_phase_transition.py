"""Turn ``pt_v2_*.json`` into a paper-grade phase-transition figure + markdown table.

The auto-generated figure in [phase_transition_pilot_v2.py](phase_transition_pilot_v2.py)
is a 1×3 panel of raw VPT / RMSE lines. For the main-paper figure we want:

  - Headline panel (VPT@1.0 across harshness) foregrounding the graceful-vs-sharp
    contrast between ours and parrot.
  - Secondary panels showing VPT@0.3 (strict) and NRMSE for completeness.
  - Marker at the harshness where parrot's VPT drops > 50% vs S0 — that's the
    "phase transition" point we claim.

Run:
    python -m experiments.week1.summarize_phase_transition \
        --json experiments/week1/results/pt_v2_with_ours_n5_small.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "experiments" / "week1" / "figures"


METHOD_LABEL = {
    "ours":    "Ours (full v2 pipeline)",
    "panda":   "Panda-72M (zero-shot)",
    "parrot":  "Context parroting (2025)",
    "chronos": "Chronos-T5-small (zero-shot)",
    "persist": "Persist-last",
}
COLORS = {"ours": "#1b9e77", "panda": "#9467bd", "parrot": "#0868ac", "chronos": "#d95f02", "persist": "#999999"}
LW = {"ours": 2.6, "panda": 2.0, "parrot": 1.8, "chronos": 1.5, "persist": 1.2}
MARKERS = {"ours": "D", "panda": "^", "parrot": "s", "chronos": "o", "persist": "x"}


def find_phase_transition_scenario(summary: dict, method: str = "parrot") -> str | None:
    """Scenario where ``method``'s VPT@1.0 first drops below half of S0."""
    if method not in summary:
        return None
    scenarios = sorted(summary[method].keys())
    base = summary[method][scenarios[0]]["vpt10_mean"]
    for s in scenarios[1:]:
        if summary[method][s]["vpt10_mean"] < 0.5 * base:
            return s
    return None


def fmt(vals):
    m = float(np.mean(vals)); s = float(np.std(vals))
    return f"{m:.2f}±{s:.2f}"


def make_table(records: list[dict], scenario_order: list[str]) -> str:
    per_cell: dict[tuple[str, str], dict] = {}
    for r in records:
        k = (r["method"], r["scenario"])
        per_cell.setdefault(k, {"vpt10": [], "vpt03": [], "rmse": []})
        per_cell[k]["vpt10"].append(r["vpt10"])
        per_cell[k]["vpt03"].append(r["vpt03"])
        per_cell[k]["rmse"].append(r["rmse_norm_first100"])

    methods = ["ours", "panda", "parrot", "chronos", "persist"]
    lines = ["| Method | " + " | ".join(scenario_order) + " |"]
    lines.append("| --- | " + " | ".join("---" for _ in scenario_order) + " |")
    for m in methods:
        row = [METHOD_LABEL.get(m, m)]
        for s in scenario_order:
            cell = per_cell.get((m, s), {})
            if not cell:
                row.append("—")
            else:
                row.append(fmt(cell["vpt10"]))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def plot_phase_transition(summary: dict, scenario_order: list[str], fig_path: Path) -> None:
    methods = [m for m in ["ours", "panda", "parrot", "chronos", "persist"] if m in summary]
    keep_fracs = [summary[methods[0]][s]["keep_mean"] for s in scenario_order]
    noise_fracs = [summary[methods[0]][s]["noise_std_frac"] for s in scenario_order]
    pt_scenario = find_phase_transition_scenario(summary, method="parrot")

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    xs = np.arange(len(scenario_order))
    for m in methods:
        means = [summary[m][s]["vpt10_mean"] for s in scenario_order]
        stds = [summary[m][s]["vpt10_std"] for s in scenario_order]
        ax0.errorbar(xs, means, yerr=stds, color=COLORS[m], marker=MARKERS[m],
                     linewidth=LW[m], capsize=3, label=METHOD_LABEL[m])
    if pt_scenario is not None:
        idx = scenario_order.index(pt_scenario)
        ax0.axvline(idx - 0.5, color="#e41a1c", linestyle=":", linewidth=1.5)
        ax0.text(idx - 0.5, ax0.get_ylim()[1] * 0.9, "  parrot phase transition",
                 color="#e41a1c", fontsize=9, rotation=90, va="top")
    ax0.set_xticks(xs)
    xlabels = [f"{s}\n{k:.0%} keep\nσ={n:.2f}" for s, k, n in zip(scenario_order, keep_fracs, noise_fracs)]
    ax0.set_xticklabels(xlabels, fontsize=9)
    ax0.set_ylabel("VPT (Lyapunov times, threshold=1.0)")
    ax0.set_title("Graceful vs phase-transition degradation\n(Lorenz63, 5 seeds)")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right", fontsize=9)

    # VPT@0.3 (strict)
    for m in methods:
        means = [summary[m][s]["vpt03_mean"] for s in scenario_order]
        stds = [summary[m][s]["vpt03_std"] for s in scenario_order]
        ax1.errorbar(xs, means, yerr=stds, color=COLORS[m], marker=MARKERS[m],
                     linewidth=LW[m], capsize=3, label=METHOD_LABEL[m])
    ax1.set_xticks(xs); ax1.set_xticklabels(scenario_order, fontsize=9)
    ax1.set_title("VPT (threshold=0.3)")
    ax1.set_ylabel("VPT")
    ax1.grid(True, alpha=0.3)

    # NRMSE / attractor-std
    for m in methods:
        means = [summary[m][s]["rmse_mean"] for s in scenario_order]
        stds = [summary[m][s]["rmse_std"] for s in scenario_order]
        ax2.errorbar(xs, means, yerr=stds, color=COLORS[m], marker=MARKERS[m],
                     linewidth=LW[m], capsize=3, label=METHOD_LABEL[m])
    ax2.set_xticks(xs); ax2.set_xticklabels(scenario_order, fontsize=9)
    ax2.set_title("NRMSE (first 100 steps)")
    ax2.set_ylabel("NRMSE / attractor-std")
    ax2.axhline(1.0, color="grey", linestyle=":", linewidth=1)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Phase-transition pilot — Lorenz63 under progressive harshness", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"[fig] saved {fig_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--fig_out", default=None)
    ap.add_argument("--md_out", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.json).read_text())
    records = data["records"]
    summary = data["summary"]
    scenario_order = sorted({r["scenario"] for r in records})

    fig_out = Path(args.fig_out) if args.fig_out else FIG_DIR / (Path(args.json).stem + "_paperfig.png")
    plot_phase_transition(summary, scenario_order, fig_out)

    md_out_path = Path(args.md_out) if args.md_out else Path(args.json).with_suffix(".md")
    md = [f"# Phase-transition results — {Path(args.json).stem}\n",
          f"Seeds: {data['config']['n_seeds']}.  n_ctx={data['config']['n_ctx']}.  dt={data['config']['dt']}.\n",
          f"Scenarios: {', '.join(scenario_order)}\n\n## VPT@1.0 table\n",
          make_table(records, scenario_order)]
    md_out_path.write_text("\n".join(md))
    print(f"[md] saved {md_out_path}")

    pt = find_phase_transition_scenario(summary, method="parrot")
    if pt:
        print(f"[verdict] parrot phase-transition point: {pt}")
    if "ours" in summary:
        ours_min = min(summary["ours"][s]["vpt10_mean"] for s in scenario_order)
        others_max_crash = max(
            summary[m][scenario_order[-1]]["vpt10_mean"]
            for m in ["panda", "parrot", "chronos", "persist"] if m in summary
        )
        print(f"[verdict] ours min VPT@1.0 across all scenarios: {ours_min:.2f}")
        print(f"[verdict] others at hardest scenario: max={others_max_crash:.2f}")


if __name__ == "__main__":
    main()
