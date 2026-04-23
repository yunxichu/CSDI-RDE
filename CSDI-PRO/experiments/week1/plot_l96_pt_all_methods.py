"""Plot L96 N=20 Phase Transition with all method variants on one figure.

Combines:
- ours_csdi (no_noise ep25) from pt_l96_l96_N20_csdi_nonoise.json
- ours_csdi (full ep25) from pt_l96_l96_N20_csdi_full.json
- panda (from nonoise run, same trajectories)
- parrot (from nonoise run)
- ours (AR-K M1) from pt_l96_N20_merged.json (5 seeds, earlier result)

Produces one multi-line figure with VPT@1.0 mean ± std across S0-S6.
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

SCENARIOS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]


def agg_vpt10(records: list[dict]) -> dict[tuple[str, str], dict]:
    """Aggregate VPT@1.0 by (method, scenario)."""
    by = defaultdict(list)
    for r in records:
        if r.get("error"):
            continue
        by[(r["method"], r["scenario"])].append(r["vpt10"])
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} for k, v in by.items()}


def load(path: Path):
    if not path.exists():
        print(f"[skip] {path} not found")
        return {}
    d = json.loads(path.read_text())
    return agg_vpt10(d.get("records", []))


def main():
    nonoise = load(RES / "pt_l96_l96_N20_csdi_nonoise.json")
    full = load(RES / "pt_l96_l96_N20_csdi_full.json")
    merged_ark = load(RES / "pt_l96_l96_N20_v1.json")
    merged_ark_seeds34 = load(RES / "pt_l96_l96_N20_v1_seeds34.json")

    # Merge AR-K runs (3 + 2 seeds → 5 seeds)
    ark = {}
    for k in set(list(merged_ark.keys()) + list(merged_ark_seeds34.keys())):
        vs = []
        if k in merged_ark:
            vs.extend([merged_ark[k]["mean"]] * merged_ark[k]["n"])
        if k in merged_ark_seeds34:
            vs.extend([merged_ark_seeds34[k]["mean"]] * merged_ark_seeds34[k]["n"])
        # Actually we lost per-sample info, just take simple mean
        if vs:
            ark[k] = {"mean": float(np.mean(vs)), "std": 0.0, "n": sum(
                merged_ark.get(k, {}).get("n", 0) + merged_ark_seeds34.get(k, {}).get("n", 0)
                for _ in [0])}

    # Actually re-aggregate properly from raw records
    def load_raw(path):
        if not path.exists(): return []
        return json.loads(path.read_text()).get("records", [])

    all_ark = load_raw(RES / "pt_l96_l96_N20_v1.json") + load_raw(RES / "pt_l96_l96_N20_v1_seeds34.json")
    ark_agg = agg_vpt10(all_ark)

    # Extract series
    def series(src, method):
        means = []; stds = []
        for s in SCENARIOS:
            k = (method, s)
            if k in src:
                means.append(src[k]["mean"])
                stds.append(src[k]["std"])
            else:
                means.append(np.nan); stds.append(0)
        return np.array(means), np.array(stds)

    method_plots = [
        ("ours_csdi (full ep25)", series(full, "ours_csdi"), "#1b9e77", "D"),
        ("ours_csdi (no_noise ep25)", series(nonoise, "ours_csdi"), "#2ca02c", "s"),
        ("ours (AR-K M1, 5 seeds)", series(ark_agg, "ours"), "#e7298a", "v"),
        ("Panda-72M", series(ark_agg, "panda"), "#9467bd", "^"),
        ("Context-Parroting", series(ark_agg, "parrot"), "C0", "o"),
        ("Persistence", series(ark_agg, "persist"), "grey", "x"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(SCENARIOS))
    for label, (means, stds), color, marker in method_plots:
        # Skip if all NaN
        if np.all(np.isnan(means)):
            continue
        n_samples = max([(ark_agg.get(("ours", s), {}) or full.get(("ours_csdi", s), {})).get("n", 3) for s in SCENARIOS])
        ax.errorbar(x, means, yerr=stds, marker=marker, color=color,
                    label=f"{label}", linewidth=2, capsize=3, markersize=8, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS, fontsize=11)
    ax.set_ylabel("VPT@1.0 (Lyapunov times)", fontsize=12)
    ax.set_xlabel("Harshness scenario (s, σ)", fontsize=12)
    ax.set_title("Lorenz96 N=20 F=8 — Phase Transition comparison\n"
                  "Current CSDI (25 ep) underperforms AR-K due to training plateau "
                  "(loss 0.075 vs L63's 0.014 @ same epochs)",
                  fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.axvspan(2.5, 3.5, alpha=0.08, color="red")
    ylim_top = max(3.5, ax.get_ylim()[1])
    ax.set_ylim(-0.1, ylim_top)
    ax.text(3.0, ylim_top * 0.95, "S3 (target transition)", ha="center", fontsize=9,
            color="red", alpha=0.7)

    plt.tight_layout()
    out = FIG / "pt_l96_N20_all_methods.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[saved] {out}")
    plt.close()

    # Also print the numeric table
    print("\n=== VPT@1.0 table (mean ± std, n seeds per cell) ===")
    header = f"{'Method':28s} " + " ".join(f"{s:>14s}" for s in SCENARIOS)
    print(header)
    for label, (means, stds), _, _ in method_plots:
        row = f"{label:28s} "
        for m, sd in zip(means, stds):
            if np.isnan(m):
                row += f"{'-':>14s} "
            else:
                row += f"{m:5.2f}±{sd:4.2f}   "
        print(row)


if __name__ == "__main__":
    main()
