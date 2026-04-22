"""Figure D3 / D4 — Horizon × Coverage 和 Horizon × PI Width（paper 独立版）.

Input : results/module4_horizon_cal_{S2,S3}_n3.json  (已存在)
Output:
    figures/horizon_coverage_paperfig.png  (D3)
    figures/horizon_piwidth_paperfig.png   (D4)

每张图 2 面板（S2 / S3）× 5 种 CP 方法（split / lyap_exp / lyap_saturating / lyap_clipped /
lyap_empirical）。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RES_DIR = REPO_ROOT / "experiments" / "week2_modules" / "results"
FIG_DIR = REPO_ROOT / "experiments" / "week2_modules" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["split", "lyap_exp", "lyap_saturating", "lyap_clipped", "lyap_empirical"]
METHOD_LABELS = {
    "split":           "Split CP",
    "lyap_exp":        "Lyap-exp",
    "lyap_saturating": "Lyap-sat",
    "lyap_clipped":    "Lyap-clipped",
    "lyap_empirical":  "Lyap-empirical (ours)",
}
METHOD_COLORS = {
    "split":           "#888888",
    "lyap_exp":        "#d95f02",
    "lyap_saturating": "#1b9e77",
    "lyap_clipped":    "#7570b3",
    "lyap_empirical":  "#e7298a",
}


def _load(sc: str) -> dict:
    return json.loads((RES_DIR / f"module4_horizon_cal_{sc}_n3.json").read_text())


def _series(metric_dict: dict, method: str, horizons: list[int]) -> tuple[np.ndarray, np.ndarray]:
    rows = metric_dict[method]
    mean = np.array([rows[str(h)][0] for h in horizons])
    std  = np.array([rows[str(h)][1] for h in horizons])
    return mean, std


def plot_d3_coverage() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, sc in zip(axes, ["S2", "S3"]):
        d = _load(sc)
        horizons = sorted(int(h) for h in d["picp"]["split"].keys())
        for m in METHODS:
            mean, std = _series(d["picp"], m, horizons)
            ax.plot(horizons, mean, "o-", color=METHOD_COLORS[m],
                    label=METHOD_LABELS[m], linewidth=2 if m == "lyap_empirical" else 1.3,
                    markersize=6 if m == "lyap_empirical" else 4)
            ax.fill_between(horizons, mean - std, mean + std,
                            color=METHOD_COLORS[m], alpha=0.12)
        ax.axhline(0.90, color="k", linestyle="--", linewidth=1.0, alpha=0.6,
                   label="nominal 0.90")
        ax.set_title(f"Scenario {sc}  (sparsity={d.get('sparsity', '?')}, σ_frac={d.get('noise', '?')})",
                     fontsize=11)
        ax.set_xlabel("horizon h")
        ax.grid(True, alpha=0.25)
        ax.set_xscale("log")
        ax.set_xticks(horizons, labels=[str(h) for h in horizons])
    axes[0].set_ylabel("PICP @ α=0.10  (target 0.90)")
    axes[0].legend(fontsize=8.5, loc="lower left", framealpha=0.85)
    plt.suptitle("Figure D3 — Conditional Coverage across Horizon  (mixed-horizon calibration)",
                 fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "horizon_coverage_paperfig.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


def plot_d4_piwidth() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for ax, sc in zip(axes, ["S2", "S3"]):
        d = _load(sc)
        horizons = sorted(int(h) for h in d["mpiw"]["split"].keys())
        for m in METHODS:
            mean, std = _series(d["mpiw"], m, horizons)
            ax.plot(horizons, mean, "o-", color=METHOD_COLORS[m],
                    label=METHOD_LABELS[m], linewidth=2 if m == "lyap_empirical" else 1.3,
                    markersize=6 if m == "lyap_empirical" else 4)
            ax.fill_between(horizons, mean - std, mean + std,
                            color=METHOD_COLORS[m], alpha=0.12)
        ax.set_title(f"Scenario {sc}", fontsize=11)
        ax.set_xlabel("horizon h")
        ax.set_ylabel("MPIW  (mean PI width)")
        ax.grid(True, alpha=0.25)
        ax.set_xscale("log")
        ax.set_xticks(horizons, labels=[str(h) for h in horizons])
    axes[0].legend(fontsize=8.5, loc="upper left", framealpha=0.85)
    plt.suptitle("Figure D4 — PI Width across Horizon  (mixed-horizon calibration)",
                 fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "horizon_piwidth_paperfig.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


if __name__ == "__main__":
    plot_d3_coverage()
    plot_d4_piwidth()
