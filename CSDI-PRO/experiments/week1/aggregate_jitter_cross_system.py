"""Cross-system aggregation of Panda jitter control results.

Loads the per-system jitter-control JSONs and produces a 2-row figure that is
the visual core of the new paper story:
  Row 1: mean VPT@1.0 — to show iid_jitter often closes the mean gap.
  Row 2: Pr(VPT > 1.0 Lyapunov times) — to show CSDI keeps the survival tail.

Also prints a consolidated markdown table.

Reads:
  panda_jitter_control_l63_sp65_sp82_5seed.json
  panda_jitter_control_l96N20_sp65_sp82_5seed.json
  panda_jitter_control_rossler_sp65_sp82_5seed.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "week1" / "results"
FIGS = REPO / "experiments" / "week1" / "figures"
DELIV = REPO / "deliverable"

CELLS = ["linear", "linear_iid_jitter", "linear_shuffled_resid", "csdi"]
CELL_LABELS = {
    "linear": "Linear",
    "linear_iid_jitter": "+ iid jitter",
    "linear_shuffled_resid": "+ shuffled\nresidual",
    "csdi": "CSDI (ours)",
}
CELL_COLORS = {
    "linear": "C1",
    "linear_iid_jitter": "C4",
    "linear_shuffled_resid": "C5",
    "csdi": "C2",
}

SYSTEMS = [
    ("L63",        RESULTS / "panda_jitter_control_l63_sp65_sp82_5seed.json"),
    ("L96 N=20",   RESULTS / "panda_jitter_control_l96N20_sp65_sp82_5seed.json"),
    ("Rössler",    RESULTS / "panda_jitter_control_rossler_sp65_sp82_5seed.json"),
]
SCENARIOS = ["SP65", "SP82"]


def _records_for(doc, scenario, cell):
    return np.array([float(r["vpt10"]) for r in doc["records"]
                     if r["scenario"] == scenario and r["cell"] == cell],
                    dtype=np.float64)


def _paired_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 5000,
                     seed: int = 17) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diff = a - b
    n = len(diff)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boots[i] = diff[rng.integers(0, n, size=n)].mean()
    return float(diff.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def make_figure(out_png: Path, scenario: str):
    fig, axes = plt.subplots(2, len(SYSTEMS), figsize=(4.4 * len(SYSTEMS), 7),
                              sharey="row")
    bar_x = np.arange(len(CELLS))

    for col, (sys_name, sys_json) in enumerate(SYSTEMS):
        if not sys_json.exists():
            for row in (0, 1):
                axes[row, col].text(0.5, 0.5, f"{sys_name} {scenario}\nNOT YET RUN",
                                     ha="center", va="center",
                                     transform=axes[row, col].transAxes,
                                     fontsize=11, color="grey")
                axes[row, col].axis("off")
            continue
        doc = json.loads(sys_json.read_text())
        if scenario not in doc.get("summary", {}):
            for row in (0, 1):
                axes[row, col].text(0.5, 0.5, f"{sys_name}\n{scenario} missing",
                                     ha="center", va="center",
                                     transform=axes[row, col].transAxes,
                                     color="grey")
                axes[row, col].axis("off")
            continue
        s = doc["summary"][scenario]
        means = [s[c]["mean"] for c in CELLS]
        stds = [s[c]["std"] for c in CELLS]
        prs = [s[c]["pr_gt_1p0"] for c in CELLS]
        colors = [CELL_COLORS[c] for c in CELLS]

        ax = axes[0, col]
        ax.bar(bar_x, means, yerr=stds, capsize=3, color=colors, alpha=0.85,
                edgecolor="black", linewidth=0.4)
        ax.set_title(f"{sys_name} {scenario} — mean VPT@1.0")
        ax.set_xticks(bar_x)
        ax.set_xticklabels([CELL_LABELS[c] for c in CELLS], fontsize=8)
        if col == 0:
            ax.set_ylabel("mean VPT@1.0 (Λ)")
        ax.grid(axis="y", alpha=0.25)

        ax = axes[1, col]
        ax.bar(bar_x, [100*p for p in prs], color=colors, alpha=0.85,
                edgecolor="black", linewidth=0.4)
        ax.set_title(f"{sys_name} {scenario} — Pr(VPT > 1.0 Λ)")
        ax.set_xticks(bar_x)
        ax.set_xticklabels([CELL_LABELS[c] for c in CELLS], fontsize=8)
        if col == 0:
            ax.set_ylabel("Pr(VPT > 1.0 Λ)  (%)")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(f"Jitter-control milestone — {scenario}\n"
                 f"(top = mean; bottom = tail-survival probability)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}")


def write_md(out_md: Path):
    lines = ["# Cross-System Jitter Control — milestone summary",
             "",
             "Each cell reports VPT@1.0 mean ± std and Pr(VPT > 1.0 Λ) "
             "from 5-seed Panda forecasts on a corruption-aware filled context.",
             "Controls: linear (no intervention), iid jitter (Gaussian noise scaled "
             "to per-channel CSDI residual std, applied only at missing entries), "
             "shuffled residual (CSDI residual values shuffled across missing "
             "positions), CSDI (the imputation under test).", ""]
    for scenario in SCENARIOS:
        lines += [f"## {scenario}", ""]
        lines += ["| System | Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |",
                  "|:--|:--|--:|--:|--:|--:|"]
        for sys_name, sys_json in SYSTEMS:
            if not sys_json.exists():
                continue
            doc = json.loads(sys_json.read_text())
            if scenario not in doc.get("summary", {}):
                continue
            for c in CELLS:
                s = doc["summary"][scenario][c]
                lines.append(
                    f"| {sys_name} | {c} | {s['mean']:.2f} ± {s['std']:.2f} | "
                    f"{s['median']:.2f} | {100*s['pr_gt_0p5']:.0f}% | "
                    f"{100*s['pr_gt_1p0']:.0f}% |"
                )
            lines.append("|  |  |  |  |  |  |")
        lines += ["", "Paired contrasts vs linear (Δ mean VPT@1.0, 95% CI):", ""]
        lines += ["| System | Cell | Δ | CI | sign |",
                  "|:--|:--|--:|:--|:-:|"]
        for sys_name, sys_json in SYSTEMS:
            if not sys_json.exists():
                continue
            doc = json.loads(sys_json.read_text())
            if scenario not in doc.get("summary", {}):
                continue
            lin = _records_for(doc, scenario, "linear")
            for c in CELLS:
                if c == "linear":
                    continue
                arr = _records_for(doc, scenario, c)
                if len(arr) != len(lin):
                    continue
                m, lo, hi = _paired_diff_ci(arr, lin)
                sgn = "↑" if lo > 0 else ("↓" if hi < 0 else "≈")
                lines.append(f"| {sys_name} | {c} | {m:+.2f} | "
                             f"[{lo:+.2f}, {hi:+.2f}] | {sgn} |")
        lines.append("")
    out_md.write_text("\n".join(lines))
    print(f"[saved] {out_md}")


def main():
    DELIV.mkdir(parents=True, exist_ok=True)
    figs_out = DELIV / "figures_jitter"
    figs_out.mkdir(parents=True, exist_ok=True)
    for sc in SCENARIOS:
        make_figure(figs_out / f"jitter_milestone_{sc}.png", scenario=sc)
    write_md(figs_out / "jitter_milestone_summary.md")


if __name__ == "__main__":
    main()
