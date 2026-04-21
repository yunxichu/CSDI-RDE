"""Aggregate ablation JSONs into a markdown table + figures.

Run:
    python -m experiments.week2_modules.summarize_ablation \
        --inputs results/ablation_S3_n3.json results/ablation_S2_n3.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RES_DIR = REPO_ROOT / "experiments" / "week2_modules" / "results"
FIG_DIR = REPO_ROOT / "experiments" / "week2_modules" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


CFG_ORDER = [
    "full", "full-empirical", "m1-linear", "m2a-random", "m2b-frasersw",
    "m3-exactgpr", "m4-splitcp", "m4-lyap-exp", "all-off",
]
CFG_LABEL = {
    "full":             "Full (4 modules, Lyap-sat)",
    "full-empirical":   "Full + Lyap-empirical",
    "m1-linear":        "−M1 (linear imp)",
    "m2a-random":       "−M2 (random τ)",
    "m2b-frasersw":     "−M2 (Fraser-Swinney τ)",
    "m3-exactgpr":      "−M3 (exact GPR)",
    "m4-splitcp":       "−M4 (Split CP)",
    "m4-lyap-exp":      "−M4 (Lyap-exp, no sat)",
    "all-off":          "All off (≈ v1 CSDI-RDE-GPR)",
}


def aggregate(records: list[dict], horizons: list[int]) -> dict:
    """group -> metric -> horizon -> array. JSON keys are strings; normalise."""
    acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in records:
        c = r["cfg_name"]
        metrics = {int(k): v for k, v in r["metrics"].items()}
        for h in horizons:
            m = metrics.get(h)
            if not m:
                continue
            for key in ("nrmse", "picp", "mpiw", "crps", "q_conformal"):
                acc[c][key][h].append(m[key])
    return acc


def fmt(vals: list[float]) -> str:
    if not vals:
        return "—"
    return f"{np.mean(vals):.3f}±{np.std(vals):.3f}"


def make_table(agg: dict, horizons: list[int]) -> str:
    lines = []
    headers = ["Config"] + [f"NRMSE h={h}" for h in horizons] + [f"PICP@90 h={h}" for h in horizons] + [f"MPIW h={h}" for h in horizons] + [f"CRPS h={h}" for h in horizons]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for c in CFG_ORDER:
        if c not in agg:
            continue
        row = [CFG_LABEL[c]]
        for h in horizons:
            row.append(fmt(agg[c]["nrmse"][h]))
        for h in horizons:
            row.append(fmt(agg[c]["picp"][h]))
        for h in horizons:
            row.append(fmt(agg[c]["mpiw"][h]))
        for h in horizons:
            row.append(fmt(agg[c]["crps"][h]))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def plot_scenario(agg: dict, horizons: list[int], scenario_name: str, fig_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    metric_names = ["nrmse", "picp", "mpiw", "crps"]
    titles = ["NRMSE (lower better)", "PICP@90 (target 0.90)", "MPIW", "CRPS"]
    colors = {
        "full":             "#1b9e77",
        "full-empirical":   "#2ca25f",
        "m1-linear":        "#d95f02",
        "m2a-random":       "#7570b3",
        "m2b-frasersw":     "#a65628",
        "m3-exactgpr":      "#e7298a",
        "m4-splitcp":       "#e6ab02",
        "m4-lyap-exp":      "#bbbb00",
        "all-off":          "#666666",
    }
    markers = {c: ("o" if c.startswith("full") else "s") for c in CFG_ORDER}

    for ax, mkey, title in zip(axes, metric_names, titles):
        for c in CFG_ORDER:
            if c not in agg:
                continue
            means = [np.mean(agg[c][mkey][h]) if agg[c][mkey][h] else np.nan for h in horizons]
            stds = [np.std(agg[c][mkey][h]) if agg[c][mkey][h] else 0.0 for h in horizons]
            ax.errorbar(
                horizons, means, yerr=stds, marker=markers[c], linewidth=2 if c == "full" else 1.2,
                color=colors[c], capsize=3, label=CFG_LABEL[c],
                linestyle="-" if c == "full" else "--",
            )
        ax.set_xscale("log")
        ax.set_xlabel("Forecast horizon h")
        ax.set_title(title)
        ax.set_xticks(horizons); ax.set_xticklabels([str(h) for h in horizons])
        ax.grid(True, alpha=0.3)
        if mkey == "picp":
            ax.axhline(0.90, color="red", linestyle=":", linewidth=1)

    axes[0].set_ylabel("value")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Module-wise ablation on Lorenz63 — scenario {scenario_name}", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {fig_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="ablation JSON files")
    ap.add_argument("--md_out", default=str(REPO_ROOT / "experiments" / "week2_modules" / "ABLATION.md"))
    args = ap.parse_args()

    md_chunks = ["# Week 2 Ablation — 4 modules on Lorenz63\n"]
    md_chunks.append("**Pipeline** (tech.md §Core): *observations → M1 imputation → M2 τ-select → M3 GP regress → M4 conformal.*\n")
    md_chunks.append("Each row flips **one** module relative to the full pipeline. 3 seeds each; delay embedding L=5, τ_max=30.\n")
    md_chunks.append("""
## Module surrogates used in this ablation

| Module | Full | −Mk variant |
|---|---|---|
| **M1** Dynamics-Aware imputation | AR-Kalman smoother (AR(5) + RTS on observed subset) | linear interpolation (Week 1 baseline) |
| **M2** Delay-embedding τ selection | MI-Lyap via BayesOpt (20 calls) with cumulative-δ param | random τ (Takahashi 2021), Fraser-Swinney first-minimum |
| **M3** Regression | GPyTorch SVGP Matern-5/2, m=128 inducing, 120 epochs | self-implemented exact GPR (n≤1000), no hyperparam opt |
| **M4** Prediction interval | Lyap-Conformal with λ∈{est, true} | vanilla Split-Conformal |

**Note on M1**: the "full" M1 is Dynamics-Aware CSDI (Transformer + noise conditioning + dynamic delay mask, tech.md §1.2) whose training takes hours of diffusion model work. Week 2 uses an AR-Kalman stand-in that captures the load-bearing ideas (model-based + noise-aware smoother). The full CSDI re-train is Week-7 work.

**Note on M4**: per-horizon calibration (below) makes Lyap-CP ≈ Split-CP numerically. The real Lyap-CP advantage appears under **mixed-horizon calibration** — see `module4_horizon_calibration.py` for that focused experiment.
""")

    for fp_str in args.inputs:
        fp = Path(fp_str) if Path(fp_str).exists() else RES_DIR / fp_str
        data = json.loads(fp.read_text())
        scenario = data["scenario"]
        horizons = data["horizons"]
        agg = aggregate(data["records"], horizons)

        sc_label = f"{scenario['name']}  (sparsity={scenario['sparsity']:.2f}, σ={scenario['noise_std_frac']:.2f})"
        md_chunks.append(f"\n## Scenario {sc_label}\n")
        md_chunks.append(make_table(agg, horizons))

        plot_scenario(agg, horizons, scenario["name"],
                      FIG_DIR / f"ablation_{scenario['name']}.png")

    Path(args.md_out).write_text("\n".join(md_chunks))
    print(f"[summary] wrote {args.md_out}")


if __name__ == "__main__":
    main()
