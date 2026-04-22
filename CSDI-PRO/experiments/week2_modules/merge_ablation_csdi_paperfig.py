"""Merge original AR-Kalman 9-config ablation + CSDI 9-config ablation into the
paper-facing Table 2 and a side-by-side bar figure.

Inputs:
    results/ablation_S3_n3_v2.json        — original (AR-Kalman M1), 9 configs × 3 seeds × 4 horizons
    results/ablation_with_csdi_v6_ep20.json         — first CSDI batch (S2+S3, 5 configs overlap)
    results/ablation_with_csdi_v6_ep20_9cfg_S3.json — second CSDI batch (S3, 5 new configs)

Output:
    results/ablation_final_s3_merged.json    — merged table
    results/ablation_final_s3_merged.md      — markdown Table 2
    figures/ablation_final_s3_paperfig.png   — NRMSE bar chart per horizon, dual-M1
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RES_DIR = REPO_ROOT / "experiments" / "week2_modules" / "results"
FIG_DIR = REPO_ROOT / "experiments" / "week2_modules" / "figures"

# Ablation pairing: each M1-style column lists the 9 configs in order
CFG_ROWS = [
    # display_label,  AR-Kalman cfg_name,  CSDI cfg_name
    ("Full (Lyap-sat)",        "full",             "full-csdi"),
    ("Full + Lyap-empirical",  "full-empirical",   "full-csdi-empirical"),
    ("−M1 (linear)",           "m1-linear",        "m1-linear"),      # linear M1 = same under both
    ("−M2a (random τ)",        "m2a-random",       "csdi-m2a-random"),
    ("−M2b (Fraser-Swinney)",  "m2b-frasersw",     "csdi-m2b-frasersw"),
    ("−M3 (exact GPR)",        "m3-exactgpr",      "csdi-m3-exactgpr"),
    ("−M4 (Split CP)",         "m4-splitcp",       "csdi-m4-splitcp"),
    ("−M4 (Lyap-exp)",         "m4-lyap-exp",      "csdi-m4-lyap-exp"),
    ("all-off (≈ v1)",         "all-off",          "all-off"),        # linear M1 = no CSDI either
]

HORIZONS = [1, 4, 16, 64]


def load_runs(path: Path) -> list[dict]:
    return json.loads(path.read_text())["records"]


def aggregate(records: list[dict], cfg_name: str, scenario: str = "S3") -> dict:
    """Return {h: (mean, std, picp_mean, mpiw_mean)} for given cfg."""
    per_h = defaultdict(list)
    picp_h = defaultdict(list)
    mpiw_h = defaultdict(list)
    for r in records:
        if r.get("cfg_name") != cfg_name or r.get("scenario") != scenario:
            continue
        for h, m in r["metrics"].items():
            h = int(h)
            per_h[h].append(m["nrmse"])
            picp_h[h].append(m["picp"])
            mpiw_h[h].append(m["mpiw"])
    return {
        h: dict(
            mean=float(np.mean(per_h[h])), std=float(np.std(per_h[h])),
            picp=float(np.mean(picp_h[h])), mpiw=float(np.mean(mpiw_h[h])),
            n=len(per_h[h]),
        )
        for h in sorted(per_h)
    }


def main() -> None:
    # Load all scenarios
    ar_records = load_runs(RES_DIR / "ablation_S3_n3_v2.json") + load_runs(RES_DIR / "ablation_S2_n3_v2.json")
    csdi_batch1 = load_runs(RES_DIR / "ablation_with_csdi_v6_ep20.json")
    csdi_batch2 = load_runs(RES_DIR / "ablation_with_csdi_v6_ep20_9cfg_S3.json")
    csdi_batch3_path = RES_DIR / "ablation_with_csdi_v6_ep20_9cfg_S2.json"
    csdi_batch3 = load_runs(csdi_batch3_path) if csdi_batch3_path.exists() else []
    csdi_records = csdi_batch1 + csdi_batch2 + csdi_batch3
    print(f"[loaded] AR-Kalman records: {len(ar_records)},  CSDI records: {len(csdi_records)}")

    # Build merged table for both S2 and S3
    tables = {}
    for sc in ["S2", "S3"]:
        table = []
        for label, ar_name, csdi_name in CFG_ROWS:
            ar_stats = aggregate(ar_records, ar_name, scenario=sc)
            csdi_stats = aggregate(csdi_records, csdi_name, scenario=sc)
            table.append(dict(
                label=label, ar_cfg=ar_name, csdi_cfg=csdi_name,
                ar=ar_stats, csdi=csdi_stats,
            ))
        tables[sc] = table
    table = tables["S3"]  # kept for existing downstream code

    # Save merged JSON (both scenarios)
    out_json = RES_DIR / "ablation_final_dualM1_merged.json"
    out_json.write_text(json.dumps(tables, indent=2))
    print(f"[saved] {out_json}")
    # also keep the legacy S3-only dump for backward compat
    (RES_DIR / "ablation_final_s3_merged.json").write_text(json.dumps(tables["S3"], indent=2))

    # Build markdown Table for both scenarios
    lines = [
        "# Ablation Table 2 — AR-Kalman M1 vs CSDI M1 (S2 + S3)",
        "",
        "3 seeds per cell.  Format:  AR-Kalman  /  **CSDI**",
        "",
    ]
    for sc in ["S2", "S3"]:
        sparsity_noise = "s=0.4, σ=0.3" if sc == "S2" else "s=0.6, σ=0.5"
        lines += [f"## Scenario {sc}  ({sparsity_noise})", "",
                  "| Config | NRMSE @ h=1 | h=4 | h=16 | CSDI Δ @ h=4 |",
                  "|---|:-:|:-:|:-:|:-:|"]
        for row in tables[sc]:
            ar = row["ar"]; c = row["csdi"]
            def fmt(h, s):
                if h not in s: return "—"
                return f"{s[h]['mean']:.3f}±{s[h]['std']:.3f}"
            gain = "—"
            if 4 in ar and 4 in c and ar[4]['mean'] > 0:
                d = (c[4]['mean'] - ar[4]['mean']) / ar[4]['mean'] * 100
                gain = f"{d:+.0f}%"
            lines.append(
                f"| {row['label']} | {fmt(1, ar)} / **{fmt(1, c)}** | "
                f"{fmt(4, ar)} / **{fmt(4, c)}** | {fmt(16, ar)} / **{fmt(16, c)}** | {gain} |"
            )
        lines.append("")
    out_md = RES_DIR / "ablation_final_dualM1_merged.md"
    out_md.write_text("\n".join(lines))
    print(f"[saved] {out_md}")
    # legacy
    (RES_DIR / "ablation_final_s3_merged.md").write_text("\n".join(
        lines[: 4 + 2 + len(CFG_ROWS) * 2]
    ))

    # Side-by-side bar figure: 2 scenarios × 3 horizons grid
    horizons_plot = [1, 4, 16]
    fig, axes = plt.subplots(2, len(horizons_plot), figsize=(14, 9), sharey=False)
    x = np.arange(len(CFG_ROWS))
    width = 0.38
    for row_i, sc in enumerate(["S2", "S3"]):
        tab = tables[sc]
        for col_i, h in enumerate(horizons_plot):
            ax = axes[row_i, col_i]
            ar_means = [r["ar"].get(h, {}).get("mean", np.nan) for r in tab]
            ar_stds  = [r["ar"].get(h, {}).get("std", 0)    for r in tab]
            c_means  = [r["csdi"].get(h, {}).get("mean", np.nan) for r in tab]
            c_stds   = [r["csdi"].get(h, {}).get("std", 0)     for r in tab]
            ax.bar(x - width/2, ar_means, width, yerr=ar_stds, capsize=3,
                   color="#888888", label="M1 = AR-Kalman", alpha=0.85)
            ax.bar(x + width/2, c_means,  width, yerr=c_stds,  capsize=3,
                   color="#e7298a", label="M1 = CSDI (ours, v6)", alpha=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels([r["label"] for r in tab], rotation=40, ha="right", fontsize=8)
            ax.set_title(f"{sc}  ·  horizon h = {h}", fontsize=11)
            if col_i == 0:
                ax.set_ylabel(f"{sc}   NRMSE", fontsize=10)
            ax.grid(True, alpha=0.25, axis="y")
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=9, loc="upper left")
    plt.suptitle("Figure 4b — 9-config Ablation with dual-M1  (AR-Kalman vs CSDI)  on S2 and S3",
                 fontsize=12)
    plt.tight_layout()
    out_fig = FIG_DIR / "ablation_final_dualM1_paperfig.png"
    plt.savefig(out_fig, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_fig}")
    # legacy S3-only figure for backward compat
    fig2, axes2 = plt.subplots(1, len(horizons_plot), figsize=(14, 5), sharey=False)
    for ax, h in zip(axes2, horizons_plot):
        ar_means = [r["ar"].get(h, {}).get("mean", np.nan) for r in tables["S3"]]
        ar_stds  = [r["ar"].get(h, {}).get("std", 0)    for r in tables["S3"]]
        c_means  = [r["csdi"].get(h, {}).get("mean", np.nan) for r in tables["S3"]]
        c_stds   = [r["csdi"].get(h, {}).get("std", 0)     for r in tables["S3"]]
        ax.bar(x - width/2, ar_means, width, yerr=ar_stds, capsize=3, color="#888888", label="AR-Kalman", alpha=0.85)
        ax.bar(x + width/2, c_means, width, yerr=c_stds, capsize=3, color="#e7298a", label="CSDI", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([r["label"] for r in tables["S3"]], rotation=40, ha="right", fontsize=8)
        ax.set_title(f"h = {h}", fontsize=11); ax.set_ylabel("NRMSE")
        ax.grid(True, alpha=0.25, axis="y")
        if h == horizons_plot[0]: ax.legend(fontsize=9, loc="upper left")
    plt.suptitle("Figure 4b — 9-config Ablation on S3: M1 = AR-Kalman vs CSDI", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ablation_final_s3_paperfig.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Print summary to stdout
    for sc in ["S2", "S3"]:
        print(f"\n=== merged Table 2 ({sc}, h=1 / h=4 / h=16) ===")
        for row in tables[sc]:
            ar = row["ar"]; c = row["csdi"]
            def val(h, s):
                return f"{s[h]['mean']:.3f}" if h in s else "—"
            print(f"  {row['label']:30s}  h=1  {val(1, ar):>6s} / {val(1, c):>6s}   "
                  f"h=4  {val(4, ar):>6s} / {val(4, c):>6s}   "
                  f"h=16 {val(16, ar):>6s} / {val(16, c):>6s}")


if __name__ == "__main__":
    main()
