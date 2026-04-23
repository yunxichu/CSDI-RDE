"""Aggregate L96 PT eval results across the m3-alt runs (GPU 1 + GPU 4 + GPU 7).

Reads the three JSONs produced by phase_transition_pilot_l96.py:
  - pt_l96_l96_N20_m3alt_5seed.json            (deepedm + fno + parrot + persist)
  - pt_l96_l96_N20_panda_5seed.json            (panda only)
  - pt_l96_l96_N20_ours_csdi_deepedm_5seed.json (CSDI M1 + DeepEDM M3)

Emits a single markdown table suitable for paper §5.7 and a JSON with all
mean ± std cells. Used by the paper draft to pull in latest numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "week1" / "results"

SOURCES = {
    "m3alt": RESULTS / "pt_l96_l96_N20_m3alt_5seed.json",
    "panda": RESULTS / "pt_l96_l96_N20_panda_5seed.json",
    "ours_csdi": RESULTS / "pt_l96_l96_N20_ours_csdi_deepedm_5seed.json",
    "ours_svgp": RESULTS / "pt_l96_l96_N20_ours_svgp_3seed.json",
}

SCENARIOS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
METHODS_DISPLAY = {
    "ours_csdi": "Ours (CSDI M1 + DeepEDM M3)",
    "ours_deepedm": "Ours (AR-K + DeepEDM)",
    "ours_fno": "Ours (AR-K + FNO)",
    "ours_svgp": "Ours (AR-K + SVGP, legacy)",
    "panda": "**Panda-72M**",
    "parrot": "Parrot",
    "persist": "Persist",
}


def load_summary(p: Path) -> dict | None:
    if not p.exists():
        return None
    doc = json.loads(p.read_text())
    return doc.get("summary", {})


def merge_summaries(srcs: dict[str, Path]) -> dict:
    merged: dict = {}
    for tag, p in srcs.items():
        s = load_summary(p)
        if s is None:
            print(f"[warn] missing {p}")
            continue
        for method, cells in s.items():
            merged.setdefault(method, {}).update(cells)
    return merged


def fmt_cell(d: dict, key_mean: str = "vpt10_mean", key_std: str = "vpt10_std") -> str:
    return f"{d[key_mean]:.2f} ± {d[key_std]:.2f}"


def fmt_drop(d_s0: dict, d_sk: dict, key: str = "vpt10_mean") -> str:
    a, b = d_s0[key], d_sk[key]
    if a <= 1e-6:
        return "—"
    drop = (b - a) / a * 100
    return f"{drop:+.0f}%"


if __name__ == "__main__":
    merged = merge_summaries(SOURCES)
    method_order = ["ours_csdi", "ours_deepedm", "ours_fno", "ours_svgp", "panda", "parrot", "persist"]
    method_order = [m for m in method_order if m in merged]
    print("\n### Table — L96 N=20 VPT@1.0 (mean ± std over 5 seeds)\n")
    hdr = "| Method | " + " | ".join(SCENARIOS) + " |"
    sep = "|:---|" + ":-:|" * len(SCENARIOS)
    print(hdr); print(sep)
    for m in method_order:
        cells = merged[m]
        row = [METHODS_DISPLAY.get(m, m)]
        for sc in SCENARIOS:
            row.append(fmt_cell(cells[sc]) if sc in cells else "—")
        print("| " + " | ".join(row) + " |")

    print("\n### Phase-transition drops (VPT@1.0, S0 → Sk)\n")
    print("| Method | S0 | S0→S2 | S0→S3 | S0→S4 | S0→S5 |")
    print("|:---|:-:|:-:|:-:|:-:|:-:|")
    for m in method_order:
        cells = merged[m]
        if "S0" not in cells:
            continue
        d0 = cells["S0"]
        row = [METHODS_DISPLAY.get(m, m), f"{d0['vpt10_mean']:.2f}"]
        for sc in ["S2", "S3", "S4", "S5"]:
            row.append(fmt_drop(d0, cells[sc]) if sc in cells else "—")
        print("| " + " | ".join(row) + " |")

    # Save merged
    out = RESULTS / "pt_l96_m3alt_merged.json"
    out.write_text(json.dumps(merged, indent=2))
    print(f"\n[saved] {out}")
