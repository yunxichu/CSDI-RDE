"""Re-run the main v2 ablation with **real Dynamics-Aware CSDI** plugged into M1.

Previously M1 was the AR-Kalman surrogate. This script:
  1. Loads a trained CSDI checkpoint via ``set_csdi_checkpoint()``
  2. Monkey-patches ``methods.dynamics_impute.impute`` so ``kind="csdi"`` works
  3. Adds two new configs:
       - ``full-csdi``: CSDI-M1 + MI-Lyap + SVGP + Lyap-sat
       - ``m1-csdi-only``: CSDI-M1 + (other modules default) — isolates M1 impact
  4. Runs the ablation on {S2, S3} with n_seeds=3, saves JSON + updates summarize

Run:
    CUDA_VISIBLE_DEVICES=0 python -m experiments.week2_modules.run_ablation_with_csdi \
        --ckpt experiments/week2_modules/ckpts/dyn_csdi_full_v3_big.pt \
        --n_seeds 3 --scenarios S2 S3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.week1.lorenz63_utils import PILOT_SCENARIOS
from experiments.week2_modules.run_ablation import (
    CONFIGS, HORIZONS, OUT_DIR, run_single,
)
from methods.csdi_impute_adapter import set_csdi_checkpoint, csdi_impute

# Monkey-patch impute dispatcher so kind="csdi" works in build_pipeline
import methods.dynamics_impute as _dyn_imp

_ORIGINAL_IMPUTE = _dyn_imp.impute


def impute_with_csdi(observed, kind: str = "dynamics"):
    if kind == "csdi":
        return csdi_impute(observed)
    return _ORIGINAL_IMPUTE(observed, kind=kind)


_dyn_imp.impute = impute_with_csdi

# New configs using trained CSDI. The CSDI-paired ablations keep M1=CSDI and flip
# one of M2/M3/M4, matching the original 9-config Table 2 but with the upgraded M1.
EXTRA_CONFIGS = {
    "full-csdi":            dict(imp="csdi",   tau="mi_lyap", gp="svgp", cp="lyap",  growth="saturating"),
    "full-csdi-empirical":  dict(imp="csdi",   tau="mi_lyap", gp="svgp", cp="lyap",  growth="empirical"),
    "csdi-m2a-random":      dict(imp="csdi",   tau="random",  gp="svgp", cp="lyap",  growth="saturating"),
    "csdi-m2b-frasersw":    dict(imp="csdi",   tau="fraser",  gp="svgp", cp="lyap",  growth="saturating"),
    "csdi-m3-exactgpr":     dict(imp="csdi",   tau="mi_lyap", gp="gpr",  cp="lyap",  growth="saturating"),
    "csdi-m4-splitcp":      dict(imp="csdi",   tau="mi_lyap", gp="svgp", cp="split", growth=None),
    "csdi-m4-lyap-exp":     dict(imp="csdi",   tau="mi_lyap", gp="svgp", cp="lyap",  growth="exp"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to trained DynamicsCSDI ckpt")
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--scenarios", nargs="+", default=["S2", "S3"])
    ap.add_argument("--tag", default="csdi")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="which configs to run; default = all 12 (original 9 + csdi 5 + "
                         "overlapping full-csdi/full-csdi-empirical/m1-linear)")
    args = ap.parse_args()

    print(f"=== Ablation with real CSDI M1 ===")
    print(f"  ckpt={args.ckpt}")
    set_csdi_checkpoint(args.ckpt)

    # Merge extra configs
    all_configs = {**CONFIGS, **EXTRA_CONFIGS}

    # default: run CSDI upgrade set (the 5 new configs; full/baselines re-used from disk)
    default_configs = [
        "full-csdi", "full-csdi-empirical",
        "csdi-m2a-random", "csdi-m2b-frasersw", "csdi-m3-exactgpr",
        "csdi-m4-splitcp", "csdi-m4-lyap-exp",
    ]
    cfg_names = args.configs or default_configs

    all_results = []
    for sc_name in args.scenarios:
        scenario = next(s for s in PILOT_SCENARIOS if s.name == sc_name)
        print(f"\n--- scenario {sc_name}  sparsity={scenario.sparsity} σ={scenario.noise_std_frac} ---")
        for cfg_name in cfg_names:
            if cfg_name not in all_configs:
                print(f"  [skip] unknown config {cfg_name!r}")
                continue
            cfg = all_configs[cfg_name]
            for seed in range(args.n_seeds):
                t0 = time.time()
                rec = run_single(cfg_name, cfg, seed=seed, scenario=scenario)
                rec["total_sec"] = time.time() - t0
                all_results.append(rec)
                h1 = rec["metrics"].get(1, {})
                print(f"  [{cfg_name:22s}] seed={seed} τ_sec={rec['tau_sec']:5.1f} "
                      f"total={rec['total_sec']:5.1f}s  h=1 nrmse={h1.get('nrmse', 0):.3f} "
                      f"picp={h1.get('picp', 0):.2f} mpiw={h1.get('mpiw', 0):.2f}")

    tag = args.tag
    out_json = OUT_DIR / f"ablation_with_csdi_{tag}.json"
    out_json.write_text(json.dumps({
        "scenarios": args.scenarios,
        "config_defs": {k: all_configs[k] for k in cfg_names if k in all_configs},
        "ckpt": args.ckpt,
        "records": all_results,
        "horizons": HORIZONS,
    }, indent=2))
    print(f"\n[saved] {out_json}")


if __name__ == "__main__":
    main()
