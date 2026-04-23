"""Quick summariser for tau-coupling ablation JSON → mean ± std table.

Usage:
    python experiments/week2_modules/analyze_tau_coupling.py \
        experiments/week2_modules/results/tau_coupling_S3_n3_v1.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(json_path: Path):
    data = json.loads(Path(json_path).read_text())
    records = data["records"]
    horizons = data.get("horizons", [1, 4, 16, 64])
    modes = data["modes"]

    # group by (mode, horizon)
    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        if "error" in r:
            continue
        for h, m in r["metrics"].items():
            h_int = int(h) if isinstance(h, str) else h
            by[r["mode"]][h_int].append(m)

    # print summary
    print(f"\n=== τ-coupling summary: {json_path.name} ===")
    print(f"scenario={data.get('scenario','?')}  n_seeds={data.get('n_seeds','?')}  modes={modes}\n")

    # header
    print(f"{'Mode':<14s}", end="")
    for h in horizons:
        print(f"  NRMSE@h={h:<3d} (μ±σ)", end="")
    print(f"  PICP@h=1 (μ)")

    for mode in modes:
        print(f"{mode:<14s}", end="")
        for h in horizons:
            metrics_list = by[mode].get(h, [])
            if not metrics_list:
                print(f"  {'--':<18s}", end="")
                continue
            nrmses = [m.get("nrmse", np.nan) for m in metrics_list]
            nr_mean = np.nanmean(nrmses)
            nr_std = np.nanstd(nrmses)
            print(f"  {nr_mean:.3f}±{nr_std:.3f}     ", end="")
        # picp at h=1
        h1 = by[mode].get(1, [])
        picp_mean = np.nanmean([m.get("picp", np.nan) for m in h1]) if h1 else np.nan
        print(f"  {picp_mean:.3f}")

    # deltas vs B_current at each horizon
    if "B_current" in by:
        print(f"\nΔ NRMSE vs B_current (%):")
        for h in horizons:
            b_nrmse = np.nanmean([m.get("nrmse", np.nan) for m in by["B_current"].get(h, [])])
            if np.isnan(b_nrmse):
                continue
            print(f"  h={h}: ", end="")
            for mode in modes:
                if mode == "B_current":
                    continue
                m_nrmse = np.nanmean([m.get("nrmse", np.nan) for m in by[mode].get(h, [])])
                if np.isnan(m_nrmse):
                    continue
                delta = 100 * (m_nrmse - b_nrmse) / b_nrmse
                print(f"{mode}={delta:+.1f}%  ", end="")
            print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    summarize(Path(sys.argv[1]))
