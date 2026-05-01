"""Aggregate isolation-ablation results: 2×3 imputer×forecaster heatmap +
paired-bootstrap 95% CI + survival probability Pr(VPT > θ).

Reads `pt_<system>_iso_<tag>.json` (or any pt_*_iso*.json) produced by
phase_transition_isolation_<system>.py.

Output:
  1. Markdown table:    label × scenario,  cell = "mean ± sd  (Pr>0.5)"
  2. PNG heatmap:       imputer × forecaster, panel per scenario, color = mean VPT
  3. PNG bar chart:     scenario × cell,  bar height = mean VPT, errorbar = bootstrap CI

Run:
  python -m experiments.week1.aggregate_isolation \\
      --json experiments/week1/results/pt_l96_iso_l96N20_5seed.json \\
      --out_prefix experiments/week1/figures/iso_l96N20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]

# Ordered for consistent display: imputer rows, forecaster cols
IMPUTERS = ["linear", "ar_kalman", "csdi"]
IMPUTER_LABELS = {"linear": "Linear", "ar_kalman": "Kalman", "csdi": "CSDI (ours)"}
FORECASTERS = ["panda", "deepedm"]
FORECASTER_LABELS = {"panda": "Panda-72M", "deepedm": "DeepEDM (delay-manifold)"}


def paired_bootstrap_ci(values: np.ndarray, n_boot: int = 5000,
                         alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Bootstrap CI on the mean of `values`. Resamples seed dimension."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = values[idx].mean()
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return lo, hi


def paired_bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 5000,
                              alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap CI on the *paired* difference (a - b). Returns (mean_diff, lo, hi)."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, f"paired bootstrap requires same length: {a.shape} vs {b.shape}"
    rng = np.random.default_rng(seed)
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    diffs = a - b
    boot = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[k] = diffs[idx].mean()
    return float(diffs.mean()), float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def load_records(path: Path) -> tuple[list[dict], list[str], list[tuple[str, str, str]]]:
    doc = json.loads(path.read_text())
    records = doc["records"]
    scenarios = sorted({r["scenario"] for r in records if not r.get("error")})
    cells = []
    seen = set()
    for r in records:
        if r.get("error"):
            continue
        key = (r["label"], r["imputer"], r["forecaster"])
        if key not in seen:
            seen.add(key)
            cells.append(key)
    return records, scenarios, cells


def collect(records: list[dict], label: str, scenario: str,
            metric: str = "vpt10") -> np.ndarray:
    vals = [r[metric] for r in records
            if r["label"] == label and r["scenario"] == scenario
            and not r.get("error") and r[metric] == r[metric]]  # NaN filter
    return np.array(vals)


def make_table(records: list[dict], cells: list[tuple], scenarios: list[str],
                metric: str = "vpt10", surv_thresh: float = 0.5) -> str:
    """Markdown table with mean±sd and Pr(VPT>θ)."""
    lines = []
    header = f"\n### Isolation table — {metric.upper()} mean ± sd  [Pr(>{surv_thresh})]\n"
    lines.append(header)
    lines.append("| Cell (imputer → forecaster) | " + " | ".join(scenarios) + " |")
    lines.append("|:---|" + ":-:|" * len(scenarios))
    for label, imp, fc in cells:
        row = [f"{IMPUTER_LABELS.get(imp, imp)} → {FORECASTER_LABELS.get(fc, fc)}"]
        for sc in scenarios:
            v = collect(records, label, sc, metric)
            if len(v) == 0:
                row.append("—")
            else:
                surv = float((v > surv_thresh).mean())
                row.append(f"{v.mean():.2f} ± {v.std():.2f}  [{surv:.0%}]")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def make_heatmap(records: list[dict], scenarios: list[str], out_png: Path,
                  metric: str = "vpt10", title_prefix: str = "Isolation"):
    """One subplot per scenario; rows=imputer, cols=forecaster, cell color=mean VPT."""
    n_sc = len(scenarios)
    fig, axes = plt.subplots(1, n_sc, figsize=(3.6 * n_sc, 3.5), squeeze=False)
    # Determine global vmin/vmax for consistent color scale
    all_means = []
    grid_data = {}
    for sc in scenarios:
        m = np.full((len(IMPUTERS), len(FORECASTERS)), np.nan)
        for i, imp in enumerate(IMPUTERS):
            for j, fc in enumerate(FORECASTERS):
                label = f"{fc}_{'kalman' if imp == 'ar_kalman' else imp}"
                v = collect(records, label, sc, metric)
                if len(v) > 0:
                    m[i, j] = v.mean()
                    all_means.append(v.mean())
        grid_data[sc] = m
    vmax = max(all_means) if all_means else 1.0
    vmin = 0.0
    for k, sc in enumerate(scenarios):
        ax = axes[0, k]
        m = grid_data[sc]
        im = ax.imshow(m, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(FORECASTERS)))
        ax.set_xticklabels([FORECASTER_LABELS[fc] for fc in FORECASTERS],
                            rotation=20, ha="right", fontsize=8)
        ax.set_yticks(range(len(IMPUTERS)))
        ax.set_yticklabels([IMPUTER_LABELS[imp] for imp in IMPUTERS], fontsize=9)
        ax.set_title(sc, fontsize=10)
        for i in range(len(IMPUTERS)):
            for j in range(len(FORECASTERS)):
                txt = "—" if np.isnan(m[i, j]) else f"{m[i, j]:.2f}"
                color = "white" if (np.isnan(m[i, j]) or m[i, j] < (vmax * 0.5)) else "black"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=10)
    fig.suptitle(f"{title_prefix} — VPT@1.0 mean (rows=imputer, cols=forecaster)",
                 fontsize=11)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="VPT@1.0 (Λ)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"[saved] {out_png}")


def make_bar_with_ci(records: list[dict], cells: list[tuple], scenarios: list[str],
                      out_png: Path, metric: str = "vpt10",
                      title_prefix: str = "Isolation"):
    """Grouped bar chart: scenarios on x, one bar per cell, errorbar = 95% bootstrap CI."""
    n_cells = len(cells)
    n_sc = len(scenarios)
    width = 0.8 / n_cells
    xs = np.arange(n_sc)
    fig, ax = plt.subplots(1, 1, figsize=(1.5 + 1.6 * n_sc, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, n_cells))
    for i, (label, imp, fc) in enumerate(cells):
        means = []
        los = []
        his = []
        for sc in scenarios:
            v = collect(records, label, sc, metric)
            if len(v) == 0:
                means.append(np.nan); los.append(np.nan); his.append(np.nan); continue
            mu = v.mean()
            lo, hi = paired_bootstrap_ci(v, n_boot=5000, seed=42)
            means.append(mu); los.append(mu - lo); his.append(hi - mu)
        offset = (i - (n_cells - 1) / 2) * width
        pretty = f"{IMPUTER_LABELS.get(imp, imp)} → {FORECASTER_LABELS.get(fc, fc)}"
        ax.bar(xs + offset, means, width=width * 0.95,
                yerr=[los, his], capsize=2.5,
                label=pretty, color=colors[i], alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs); ax.set_xticklabels(scenarios)
    ax.set_xlabel("harshness scenario")
    ax.set_ylabel("VPT@1.0  (mean ± 95% bootstrap CI)")
    ax.set_title(f"{title_prefix} — isolation ablation with paired-bootstrap CI")
    ax.legend(loc="best", fontsize=8, framealpha=0.85, ncol=1)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"[saved] {out_png}")


def make_paired_diff_table(records: list[dict], scenarios: list[str],
                            metric: str = "vpt10") -> str:
    """Headline contrasts: paired bootstrap CI on (csdi - linear) per forecaster."""
    lines = ["\n### Paired-bootstrap headline contrasts (Δ = mean(csdi) - mean(linear))\n"]
    lines.append("| Forecaster | Scenario | Δ VPT@1.0 | 95% paired CI | sign |")
    lines.append("|:---|:---|:-:|:-:|:-:|")
    for fc in FORECASTERS:
        for sc in scenarios:
            label_csdi = f"{fc}_csdi"
            label_lin = f"{fc}_linear"
            a = collect(records, label_csdi, sc, metric)
            b = collect(records, label_lin, sc, metric)
            if len(a) != len(b) or len(a) == 0:
                lines.append(f"| {FORECASTER_LABELS[fc]} | {sc} | — | — | — |")
                continue
            mu, lo, hi = paired_bootstrap_diff_ci(a, b, seed=42)
            sign = "↑" if lo > 0 else ("↓" if hi < 0 else "≈")
            lines.append(f"| {FORECASTER_LABELS[fc]} | {sc} | {mu:+.2f} | "
                         f"[{lo:+.2f}, {hi:+.2f}] | {sign} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to pt_<system>_iso_<tag>.json")
    ap.add_argument("--out_prefix", required=True,
                    help="Path prefix for outputs (will append _heatmap.png / _bars.png)")
    ap.add_argument("--metric", default="vpt10",
                    choices=["vpt03", "vpt05", "vpt10"])
    ap.add_argument("--title", default="L96 N=20 isolation")
    args = ap.parse_args()

    path = Path(args.json)
    if not path.is_absolute():
        path = REPO / path
    records, scenarios, cells = load_records(path)
    out_prefix = Path(args.out_prefix)
    if not out_prefix.is_absolute():
        out_prefix = REPO / out_prefix

    table = make_table(records, cells, scenarios, metric=args.metric)
    diffs = make_paired_diff_table(records, scenarios, metric=args.metric)
    md_text = table + "\n\n" + diffs + "\n"
    print(md_text)

    out_md = out_prefix.parent / f"{out_prefix.name}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md_text)
    print(f"[saved] {out_md}")

    make_heatmap(records, scenarios,
                 out_prefix.parent / f"{out_prefix.name}_heatmap.png",
                 metric=args.metric, title_prefix=args.title)
    make_bar_with_ci(records, cells, scenarios,
                     out_prefix.parent / f"{out_prefix.name}_bars.png",
                     metric=args.metric, title_prefix=args.title)


if __name__ == "__main__":
    main()
