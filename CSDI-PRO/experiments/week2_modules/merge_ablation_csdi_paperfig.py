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
    ar_records = load_runs(RES_DIR / "ablation_S3_n3_v2.json")
    csdi_batch1 = load_runs(RES_DIR / "ablation_with_csdi_v6_ep20.json")
    csdi_batch2 = load_runs(RES_DIR / "ablation_with_csdi_v6_ep20_9cfg_S3.json")
    csdi_records = csdi_batch1 + csdi_batch2
    print(f"[loaded] AR-Kalman records: {len(ar_records)},  CSDI records: {len(csdi_records)}")

    # Build merged table
    table = []
    for label, ar_name, csdi_name in CFG_ROWS:
        ar_stats = aggregate(ar_records, ar_name, scenario="S3")
        csdi_stats = aggregate(csdi_records, csdi_name, scenario="S3")
        table.append(dict(
            label=label, ar_cfg=ar_name, csdi_cfg=csdi_name,
            ar=ar_stats, csdi=csdi_stats,
        ))

    # Save merged JSON
    out_json = RES_DIR / "ablation_final_s3_merged.json"
    out_json.write_text(json.dumps(table, indent=2))
    print(f"[saved] {out_json}")

    # Build markdown Table
    lines = [
        "# Ablation Table 2 (S3) — AR-Kalman M1 vs CSDI M1 (merged)",
        "",
        "**scenario**: S3 (sparsity=0.6, σ_frac=0.5)  ·  3 seeds per cell",
        "",
        "| Config | NRMSE@h=1 (AR-K / CSDI) | h=4 | h=16 | CSDI gain vs AR-K @ h=4 |",
        "|---|:-:|:-:|:-:|:-:|",
    ]
    for row in table:
        ar = row["ar"]; c = row["csdi"]
        def fmt(h, s):
            if h not in s: return "—"
            return f"{s[h]['mean']:.3f}±{s[h]['std']:.3f}"
        gain = ""
        if 4 in ar and 4 in c:
            d = (c[4]['mean'] - ar[4]['mean']) / ar[4]['mean'] * 100
            gain = f"{d:+.0f}%"
        lines.append(
            f"| {row['label']} | {fmt(1, ar)} / **{fmt(1, c)}** | "
            f"{fmt(4, ar)} / **{fmt(4, c)}** | {fmt(16, ar)} / **{fmt(16, c)}** | {gain} |"
        )
    out_md = RES_DIR / "ablation_final_s3_merged.md"
    out_md.write_text("\n".join(lines))
    print(f"[saved] {out_md}")

    # Side-by-side bar figure: 9 rows × horizons
    horizons_plot = [1, 4, 16]
    fig, axes = plt.subplots(1, len(horizons_plot), figsize=(14, 5), sharey=False)
    x = np.arange(len(CFG_ROWS))
    width = 0.38
    for ax, h in zip(axes, horizons_plot):
        ar_means = [row["ar"].get(h, {}).get("mean", np.nan) for row in table]
        ar_stds  = [row["ar"].get(h, {}).get("std", 0) for row in table]
        c_means  = [row["csdi"].get(h, {}).get("mean", np.nan) for row in table]
        c_stds   = [row["csdi"].get(h, {}).get("std", 0) for row in table]
        ax.bar(x - width/2, ar_means, width, yerr=ar_stds, capsize=3,
               color="#888888", label="M1 = AR-Kalman", alpha=0.85)
        ax.bar(x + width/2, c_means,  width, yerr=c_stds,  capsize=3,
               color="#e7298a", label="M1 = CSDI (ours, v6)", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([row["label"] for row in table], rotation=40, ha="right", fontsize=8)
        ax.set_title(f"horizon h = {h}", fontsize=11)
        ax.set_ylabel("NRMSE")
        ax.grid(True, alpha=0.25, axis="y")
        if h == horizons_plot[0]:
            ax.legend(fontsize=9, loc="upper left")
    plt.suptitle("Figure 4 (updated) — 9-config Ablation on S3:  M1 = AR-Kalman vs M1 = CSDI",
                 fontsize=12)
    plt.tight_layout()
    out_fig = FIG_DIR / "ablation_final_s3_paperfig.png"
    plt.savefig(out_fig, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_fig}")

    # Print summary to stdout
    print()
    print("=== merged Table 2 (S3, h=1 / h=4 / h=16) ===")
    for row in table:
        ar = row["ar"]; c = row["csdi"]
        def val(h, s):
            return f"{s[h]['mean']:.3f}" if h in s else "—"
        print(f"  {row['label']:30s}  h=1  {val(1, ar):>6s} / {val(1, c):>6s}   "
              f"h=4  {val(4, ar):>6s} / {val(4, c):>6s}   "
              f"h=16 {val(16, ar):>6s} / {val(16, c):>6s}")


if __name__ == "__main__":
    main()
