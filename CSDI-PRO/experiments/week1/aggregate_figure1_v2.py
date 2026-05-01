"""Paper Figure 1 aggregator — built to the reviewer-acceptance spec locked in
STORY_LOCK_2026-04-28.md.

Six requirements (from `STORY_LOCK_2026-04-28.md` §"What Figure 1 Must Show"):
  1. s and σ decoupled (separate panels, not S0–S6).
  2. 10 seeds per cell.
  3. mean VPT, Pr(VPT > 0.5), Pr(VPT > 1.0) — three metrics.
  4. 95 % bootstrap CI on every cell.
  5. Pre-registered cells only (the v2 fine_s_line and fine_sigma_line names).
  6. Transition band visible — not scatter noise.

Inputs (10-seed JSONs after merging the _h0 / _h5 halves):
  experiments/week1/results/pt_l63_grid_v2_l63_fine_s_v2_10seed.json
  experiments/week1/results/pt_l63_grid_v2_l63_fine_sigma_v2_10seed.json

Or, alternatively, two halves (this script will merge if --halves is passed):
  pt_l63_grid_v2_l63_fine_s_v2_10seed_h0.json  +  ..._h5.json
  pt_l63_grid_v2_l63_fine_sigma_v2_10seed_h0.json  +  ..._h5.json

Outputs:
  deliverable/figures_main/figure1_l63_v2_10seed.png  (six-panel main figure)
  deliverable/figures_main/figure1_l63_v2_10seed.md   (table)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "week1" / "results"
DELIV = REPO / "deliverable" / "figures_main"
DELIV.mkdir(parents=True, exist_ok=True)

# Pre-registered cells (locked, do not edit after the fact)
SPARSITY_CELLS = ["SP00", "SP20", "SP40", "SP55", "SP65", "SP75",
                   "SP82", "SP88", "SP93", "SP97"]
NOISE_CELLS = ["NO00", "NO005", "NO010", "NO020", "NO035", "NO050",
                "NO080", "NO120"]

# Cell labels (imputer × forecaster) — same naming as the v2 runner
CELL_LABELS = ["panda_linear", "panda_csdi", "deepedm_linear", "deepedm_csdi"]
CELL_DISPLAY = {
    "panda_linear":   "Linear → Panda",
    "panda_csdi":     "CSDI → Panda",
    "deepedm_linear": "Linear → DeepEDM",
    "deepedm_csdi":   "CSDI → DeepEDM",
}
CELL_COLOR = {
    "panda_linear":   "C1",
    "panda_csdi":     "C2",
    "deepedm_linear": "C0",
    "deepedm_csdi":   "C3",
}
CELL_LINESTYLE = {
    "panda_linear":   "-",
    "panda_csdi":     "-",
    "deepedm_linear": "--",
    "deepedm_csdi":   "--",
}
CELL_MARKER = {
    "panda_linear":   "o",
    "panda_csdi":     "s",
    "deepedm_linear": "v",
    "deepedm_csdi":   "D",
}


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    doc = json.loads(path.read_text())
    return doc.get("records", [])


def merge_halves(h0: Path, h5: Path) -> list[dict]:
    return load_records(h0) + load_records(h5)


def collect(records: list[dict], cell: str, scenario: str,
            metric: str = "vpt10") -> np.ndarray:
    vals = []
    for r in records:
        if r.get("error"):
            continue
        # v2 runner uses keys: cell or label
        cell_key = r.get("cell") or r.get("label")
        sc_key = r.get("scenario") or r.get("config_name")
        if cell_key == cell and sc_key == scenario:
            v = r.get(metric)
            if v is not None and v == v:
                vals.append(float(v))
    return np.array(vals)


def bootstrap_ci(vals: np.ndarray, n_boot: int = 5000,
                  alpha: float = 0.05, seed: int = 17) -> tuple[float, float]:
    if len(vals) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(vals)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boot[i] = vals[rng.integers(0, n, size=n)].mean()
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95 % score interval for binomial proportion k/n. Defined at
    p=0 and p=1 (unlike normal approximation). Returns (lo, hi) on [0, 1]."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(max(p * (1.0 - p) / n + z * z / (4.0 * n * n), 0.0)) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def cell_stats(records: list[dict], cell: str, scenario: str) -> dict:
    v = collect(records, cell, scenario, metric="vpt10")
    if len(v) == 0:
        return {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "median": float("nan"),
                "pr05": float("nan"), "pr05_lo": float("nan"), "pr05_hi": float("nan"),
                "pr10": float("nan"), "pr10_lo": float("nan"), "pr10_hi": float("nan")}
    lo, hi = bootstrap_ci(v)
    n = int(len(v))
    k05 = int((v > 0.5).sum())
    k10 = int((v > 1.0).sum())
    p05_lo, p05_hi = wilson_ci(k05, n)
    p10_lo, p10_hi = wilson_ci(k10, n)
    return {
        "n": n,
        "mean": float(v.mean()),
        "lo": lo, "hi": hi,
        "median": float(np.median(v)),
        "pr05": k05 / n, "pr05_lo": p05_lo, "pr05_hi": p05_hi,
        "pr10": k10 / n, "pr10_lo": p10_lo, "pr10_hi": p10_hi,
    }


def get_axis_values(scenario_list: list[str], records: list[dict]) -> list[float]:
    """Read the actual sparsity / noise per scenario from the records (any cell)."""
    out = []
    for sc in scenario_list:
        for r in records:
            sc_key = r.get("scenario") or r.get("config_name")
            if sc_key == sc:
                # Pure-sparsity line uses noise=0; pure-noise line uses sparsity=0.
                # We pick the varying axis.
                if sc.startswith("SP"):
                    out.append(float(r["sparsity"]))
                else:
                    out.append(float(r["noise_std_frac"]))
                break
        else:
            out.append(float("nan"))
    return out


def make_panel(ax, x, scenario_list, records, metric_kind: str,
                xlabel: str, ylabel: str, title: str,
                show_band: bool = True):
    """Plot one panel: x-axis = corruption level, y-axis = chosen metric.
       metric_kind ∈ {"mean_ci", "pr05", "pr10"}.
       mean_ci uses paired bootstrap CI on VPT mean.
       pr05/pr10 use Wilson 95 % CI on binomial proportion (n = seeds)."""
    for cell in CELL_LABELS:
        ys, lo_err, hi_err = [], [], []
        for sc in scenario_list:
            s = cell_stats(records, cell, sc)
            if metric_kind == "mean_ci":
                ys.append(s["mean"])
                lo_err.append(max(0.0, s["mean"] - s["lo"]) if s["mean"] == s["mean"] else 0)
                hi_err.append(max(0.0, s["hi"] - s["mean"]) if s["mean"] == s["mean"] else 0)
            elif metric_kind == "pr05":
                p = s["pr05"]
                ys.append(100 * p)
                lo_err.append(max(0.0, 100 * (p - s["pr05_lo"])) if p == p else 0)
                hi_err.append(max(0.0, 100 * (s["pr05_hi"] - p)) if p == p else 0)
            elif metric_kind == "pr10":
                p = s["pr10"]
                ys.append(100 * p)
                lo_err.append(max(0.0, 100 * (p - s["pr10_lo"])) if p == p else 0)
                hi_err.append(max(0.0, 100 * (s["pr10_hi"] - p)) if p == p else 0)
        ys = np.array(ys); lo_err = np.array(lo_err); hi_err = np.array(hi_err)
        ax.errorbar(x, ys, yerr=[lo_err, hi_err],
                     color=CELL_COLOR[cell], linestyle=CELL_LINESTYLE[cell],
                     marker=CELL_MARKER[cell], markersize=5,
                     capsize=2, linewidth=1.5, alpha=0.92,
                     label=CELL_DISPLAY[cell])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=0.3)
    if metric_kind != "mean_ci":
        ax.set_ylim(-5, 110)


def make_figure(records_s: list[dict], records_n: list[dict], out_png: Path):
    s_axis = get_axis_values(SPARSITY_CELLS, records_s)
    n_axis = get_axis_values(NOISE_CELLS, records_n)

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.5), sharex="col")

    # Row 1: pure sparsity line (σ=0)
    make_panel(axes[0, 0], s_axis, SPARSITY_CELLS, records_s, "mean_ci",
                "sparsity s (σ = 0)", "mean VPT@1.0 (Λ)",
                "(a)  Mean VPT vs. sparsity (95 % bootstrap CI)")
    make_panel(axes[0, 1], s_axis, SPARSITY_CELLS, records_s, "pr05",
                "sparsity s (σ = 0)", "Pr(VPT > 0.5 Λ)  (%)",
                "(b)  Survival prob. > 0.5 Λ")
    make_panel(axes[0, 2], s_axis, SPARSITY_CELLS, records_s, "pr10",
                "sparsity s (σ = 0)", "Pr(VPT > 1.0 Λ)  (%)",
                "(c)  Tail survival > 1.0 Λ")

    # Row 2: pure noise line (s=0)
    make_panel(axes[1, 0], n_axis, NOISE_CELLS, records_n, "mean_ci",
                "noise σ / σ_attr (s = 0)", "mean VPT@1.0 (Λ)",
                "(d)  Mean VPT vs. noise (95 % bootstrap CI)")
    make_panel(axes[1, 1], n_axis, NOISE_CELLS, records_n, "pr05",
                "noise σ / σ_attr (s = 0)", "Pr(VPT > 0.5 Λ)  (%)",
                "(e)  Survival prob. > 0.5 Λ")
    make_panel(axes[1, 2], n_axis, NOISE_CELLS, records_n, "pr10",
                "noise σ / σ_attr (s = 0)", "Pr(VPT > 1.0 Λ)  (%)",
                "(f)  Tail survival > 1.0 Λ")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
                bbox_to_anchor=(0.5, 1.005), frameon=False, fontsize=10)
    fig.suptitle("Figure 1 — Sparse-noisy forecastability frontier on Lorenz63 (10 seeds, v2 grid)",
                 y=1.04, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_png}")


def write_md(records_s: list[dict], records_n: list[dict], out_md: Path):
    lines = ["# Figure 1 — L63 v2 10-seed fine grid (paper headline figure)",
             "",
             "Reviewer-acceptance spec: pre-registered SP/NO cells only, "
             "10 seeds per cell, three metrics (mean / Pr>0.5 / Pr>1.0), "
             "95 % bootstrap CI on mean, decoupled (s, σ).",
             ""]

    for label, scenario_list, records in [
        ("Pure sparsity (σ = 0)", SPARSITY_CELLS, records_s),
        ("Pure noise (s = 0)", NOISE_CELLS, records_n),
    ]:
        lines += [f"## {label}", ""]
        header = "| Cell | " + " | ".join(scenario_list) + " |"
        lines += [header, "|:---|" + ":-:|" * len(scenario_list)]
        for cell in CELL_LABELS:
            row = [CELL_DISPLAY[cell]]
            for sc in scenario_list:
                s = cell_stats(records, cell, sc)
                if s["n"] == 0:
                    row.append("—")
                else:
                    row.append(f"{s['mean']:.2f} [{s['lo']:.2f},{s['hi']:.2f}] "
                               f"({100*s['pr10']:.0f}%)")
            lines.append("| " + " | ".join(row) + " |")
        lines += ["",
                  "Cell format: `mean VPT@1.0 [95 % CI lo, hi] (Pr>1.0)`. "
                  f"n_seeds = {cell_stats(records, CELL_LABELS[0], scenario_list[0])['n']}.",
                  ""]
    out_md.write_text("\n".join(lines))
    print(f"[saved] {out_md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--halves", action="store_true",
                    help="Read _h0 + _h5 halves and merge into 10-seed records.")
    ap.add_argument("--s_tag", default="l63_fine_s_v2_10seed",
                    help="Either single-tag or base for halves (will append _h0/_h5).")
    ap.add_argument("--n_tag", default="l63_fine_sigma_v2_10seed")
    ap.add_argument("--out_prefix", default=str(DELIV / "figure1_l63_v2_10seed"))
    args = ap.parse_args()

    if args.halves:
        s_records = merge_halves(
            RESULTS / f"pt_l63_grid_v2_{args.s_tag}_h0.json",
            RESULTS / f"pt_l63_grid_v2_{args.s_tag}_h5.json",
        )
        n_records = merge_halves(
            RESULTS / f"pt_l63_grid_v2_{args.n_tag}_h0.json",
            RESULTS / f"pt_l63_grid_v2_{args.n_tag}_h5.json",
        )
    else:
        s_records = load_records(RESULTS / f"pt_l63_grid_v2_{args.s_tag}.json")
        n_records = load_records(RESULTS / f"pt_l63_grid_v2_{args.n_tag}.json")

    print(f"[fig1] loaded sparsity records: {len(s_records)}")
    print(f"[fig1] loaded noise records: {len(n_records)}")

    out_png = Path(args.out_prefix + ".png")
    out_md = Path(args.out_prefix + ".md")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    make_figure(s_records, n_records, out_png)
    write_md(s_records, n_records, out_md)


if __name__ == "__main__":
    main()
