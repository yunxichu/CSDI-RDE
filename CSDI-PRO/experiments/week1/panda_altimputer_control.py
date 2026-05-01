"""Alt-imputer reviewer-defense control: linear / SAITS / BRITS / CSDI -> Panda
on the cells where CSDI is most decisively positive under v2 protocol.

Question: is corruption-aware imputation specifically important, or would any
modern structured imputer rescue Panda equally inside the transition band?

Setting (chosen because they are CSDI-decisive under v2 protocol):
  - L63 SP65 (s=0.65 σ=0)  — strongest CSDI gain on entrance band
  - L63 SP82 (s=0.82 σ=0)  — floor band, mixed-distance regime
  - L96 N=20 SP82 (s=0.82 σ=0) — only L96 cell with strict positive Panda CI

Cells:
  - linear (existing baseline)
  - SAITS  (PyPOTS, per-instance fit)
  - BRITS  (PyPOTS, per-instance fit)
  - CSDI   (M1, pre-trained)

Forecaster: Panda only. DeepEDM is not the question here.

Protocol locked to v2:
  - L63 attractor_std = LORENZ63_ATTRACTOR_STD = 8.51
  - L96 attractor_std = lorenz96_attractor_std(N=20, F=8) = 3.6387
  - mask seed = 1000 * seed + 5000 + grid_index (matching v2 grid runner)

5 seeds × 4 cells × 3 settings = 60 runs. Per-instance SAITS/BRITS train
on each context independently (no pretraining corpus). This biases against
SAITS/BRITS — if they still match CSDI, it's evidence that "structured
imputation in general is the lever". If they fail to match CSDI even with
their training advantage on the test trajectory, CSDI's pretrained dynamics
prior is doing real work.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path
from typing import Any

for _var in [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(_var, "4")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np

from experiments.week1.corruption import make_corrupted_observations
from experiments.week1.lorenz63_utils import (
    LORENZ63_LYAP, LORENZ63_ATTRACTOR_STD, integrate_lorenz63,
    valid_prediction_time,
)
from experiments.week1.lorenz96_utils import (
    LORENZ96_F_DEFAULT, LORENZ96_LYAP_F8,
    integrate_lorenz96, lorenz96_attractor_std,
)
from methods.dynamics_impute import impute  # gives us linear / csdi


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "week1" / "results"
FIGS = REPO / "experiments" / "week1" / "figures"
RESULTS.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)

L63_CKPT = REPO / "experiments/week2_modules/ckpts/dyn_csdi_full_v6_center_ep20.pt"
L96_CKPT = REPO / "experiments/week2_modules/ckpts/dyn_csdi_l96_full_c192_vales_best.pt"

# v2 protocol grid_index per scenario name (mirroring fine_s_line / fine_sigma_line
# in corruption_grid_v2.json). Only the cells we use are listed.
GRID_INDEX_FINE_S = {"SP55": 3, "SP65": 4, "SP75": 5, "SP82": 6}

# 3 (system, scenario) settings
SETTINGS = [
    {"system": "L63", "scenario": "SP65", "sparsity": 0.65, "noise_std_frac": 0.0,
     "n_features": 3, "dt": 0.025, "lyap": LORENZ63_LYAP,
     "attr_std": LORENZ63_ATTRACTOR_STD, "ckpt": L63_CKPT},
    {"system": "L63", "scenario": "SP82", "sparsity": 0.82, "noise_std_frac": 0.0,
     "n_features": 3, "dt": 0.025, "lyap": LORENZ63_LYAP,
     "attr_std": LORENZ63_ATTRACTOR_STD, "ckpt": L63_CKPT},
    {"system": "L96", "scenario": "SP82", "sparsity": 0.82, "noise_std_frac": 0.0,
     "n_features": 20, "dt": 0.05, "lyap": LORENZ96_LYAP_F8,
     "attr_std": float(lorenz96_attractor_std(N=20, F=LORENZ96_F_DEFAULT)),
     "ckpt": L96_CKPT},
]
CELLS = ("linear", "saits", "brits", "csdi")


def _integrate(system: str, n_ctx: int, pred_len: int, dt: float, seed: int) -> np.ndarray:
    if system == "L63":
        return integrate_lorenz63(n_ctx + pred_len, dt=dt, spinup=2000, seed=seed).astype(np.float32)
    if system == "L96":
        return integrate_lorenz96(n_ctx + pred_len, N=20, F=LORENZ96_F_DEFAULT,
                                    dt=dt, spinup=2000, seed=seed).astype(np.float32)
    raise ValueError(system)


def _saits_impute(observed_2d: np.ndarray, n_features: int, n_steps: int) -> np.ndarray:
    """Per-instance SAITS fit + impute on a single (n_steps, n_features) trajectory."""
    from pypots.imputation import SAITS
    X = observed_2d[None, :, :].astype(np.float32)  # (1, T, D)
    saits = SAITS(n_steps=n_steps, n_features=n_features,
                   n_layers=2, d_model=64, n_heads=4, d_k=16, d_v=16, d_ffn=128,
                   batch_size=1, epochs=50, verbose=False, device="cuda")
    saits.fit({"X": X})
    imp = saits.impute({"X": X})
    return np.asarray(imp).reshape(n_steps, n_features).astype(np.float32)


def _brits_impute(observed_2d: np.ndarray, n_features: int, n_steps: int) -> np.ndarray:
    from pypots.imputation import BRITS
    X = observed_2d[None, :, :].astype(np.float32)
    brits = BRITS(n_steps=n_steps, n_features=n_features,
                   rnn_hidden_size=64, batch_size=1, epochs=50,
                   verbose=False, device="cuda")
    brits.fit({"X": X})
    imp = brits.impute({"X": X})
    return np.asarray(imp).reshape(n_steps, n_features).astype(np.float32)


def _make_filled(
    observed: np.ndarray,
    cell: str,
    n_features: int,
    n_steps: int,
    sigma_override: float,
) -> np.ndarray:
    if cell == "linear":
        return impute(observed, kind="linear").astype(np.float32)
    if cell == "csdi":
        return impute(observed, kind="csdi", sigma_override=sigma_override).astype(np.float32)
    if cell == "saits":
        return _saits_impute(observed, n_features=n_features, n_steps=n_steps)
    if cell == "brits":
        return _brits_impute(observed, n_features=n_features, n_steps=n_steps)
    raise ValueError(cell)


def _paired_bootstrap(a: np.ndarray, b: np.ndarray,
                      n_boot: int = 5000, seed: int = 29) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(diff)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        boots[i] = diff[rng.integers(0, n, size=n)].mean()
    return float(diff.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--n_ctx", type=int, default=512)
    ap.add_argument("--pred_len", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    ap.add_argument("--settings", nargs="+", default=None,
                    help="Subset of setting names like 'L63_SP65 L63_SP82 L96_SP82'")
    ap.add_argument("--tag", default="altimputer_5seed")
    args = ap.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit("Set CUDA_VISIBLE_DEVICES explicitly.")

    settings = SETTINGS
    if args.settings:
        wanted = set(args.settings)
        settings = [s for s in SETTINGS if f"{s['system']}_{s['scenario']}" in wanted]

    from baselines.panda_adapter import panda_forecast
    from methods.csdi_impute_adapter import set_csdi_attractor_std, set_csdi_checkpoint

    records: list[dict[str, Any]] = []
    last_ckpt = None

    for setting in settings:
        sys_name = setting["system"]; sc_name = setting["scenario"]
        attr_std = float(setting["attr_std"]); lyap = float(setting["lyap"])
        dt = float(setting["dt"]); n_features = int(setting["n_features"])
        ckpt = setting["ckpt"]
        if any(c == "csdi" for c in args.cells) and str(ckpt) != last_ckpt:
            set_csdi_checkpoint(str(ckpt))
            set_csdi_attractor_std(attr_std)
            last_ckpt = str(ckpt)
            print(f"[alt] csdi ckpt -> {ckpt} attr_std={attr_std:.4f}")

        print(f"\n=== {sys_name} {sc_name}  s={setting['sparsity']} σ={setting['noise_std_frac']} ===")
        for i in range(args.n_seeds):
            seed = args.seed_offset + i
            traj = _integrate(sys_name, args.n_ctx, args.pred_len, dt, seed)
            ctx_true = traj[: args.n_ctx]; future_true = traj[args.n_ctx:]
            corrupt_seed = 1000 * seed + 5000 + GRID_INDEX_FINE_S[sc_name]
            obs_res = make_corrupted_observations(
                ctx_true, mask_regime="iid_time",
                sparsity=float(setting["sparsity"]),
                noise_std_frac=float(setting["noise_std_frac"]),
                attractor_std=attr_std, seed=corrupt_seed,
                dt=dt, lyap=lyap, patch_length=16,
            )
            observed = obs_res.observed
            keep = float(obs_res.metadata["keep_frac"])

            for cell in args.cells:
                t0 = time.time()
                try:
                    filled = _make_filled(
                        observed, cell, n_features, args.n_ctx,
                        sigma_override=float(setting["noise_std_frac"]) * attr_std,
                    )
                    mean = panda_forecast(filled, pred_len=args.pred_len, device=args.device)
                    mean = mean[: args.pred_len]
                    err = None
                except Exception as e:
                    mean = None; err = str(e)[:200]
                t_infer = time.time() - t0

                if mean is None:
                    rec = dict(system=sys_name, scenario=sc_name, seed=int(seed),
                                cell=cell, sparsity=setting["sparsity"],
                                noise_std_frac=setting["noise_std_frac"],
                                keep_frac=keep,
                                vpt03=float("nan"), vpt05=float("nan"), vpt10=float("nan"),
                                infer_time_s=t_infer, error=err)
                    print(f"  seed={seed} {cell:8s}  FAILED: {err}")
                else:
                    vpt03 = valid_prediction_time(future_true, mean, dt=dt, lyap=lyap,
                                                   threshold=0.3, attractor_std=attr_std)
                    vpt05 = valid_prediction_time(future_true, mean, dt=dt, lyap=lyap,
                                                   threshold=0.5, attractor_std=attr_std)
                    vpt10 = valid_prediction_time(future_true, mean, dt=dt, lyap=lyap,
                                                   threshold=1.0, attractor_std=attr_std)
                    rec = dict(system=sys_name, scenario=sc_name, seed=int(seed),
                                cell=cell, sparsity=setting["sparsity"],
                                noise_std_frac=setting["noise_std_frac"],
                                keep_frac=keep,
                                vpt03=float(vpt03), vpt05=float(vpt05), vpt10=float(vpt10),
                                infer_time_s=t_infer, error=None)
                    print(f"  seed={seed} {cell:8s} keep={keep:.2f} VPT@1.0={vpt10:5.2f} t={t_infer:.1f}s")
                records.append(rec)

    # Aggregate per (system, scenario, cell)
    acc: dict[tuple[str, str, str], list[float]] = collections.defaultdict(list)
    for r in records:
        if r.get("error"): continue
        acc[(r["system"], r["scenario"], r["cell"])].append(float(r["vpt10"]))

    summary: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    for setting in settings:
        sys_name = setting["system"]; sc_name = setting["scenario"]
        key = f"{sys_name}_{sc_name}"
        summary[key] = {}
        for cell in args.cells:
            v = np.array(acc[(sys_name, sc_name, cell)])
            if len(v) == 0: continue
            summary[key][cell] = {
                "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "median": float(np.median(v)),
                "pr_gt_0p5": float((v > 0.5).mean()),
                "pr_gt_1p0": float((v > 1.0).mean()),
                "n": int(len(v)),
            }
        contrasts[key] = {}
        for cell in args.cells:
            if cell == "linear": continue
            a = np.array(acc[(sys_name, sc_name, cell)])
            b = np.array(acc[(sys_name, sc_name, "linear")])
            if len(a) != len(b) or len(a) == 0: continue
            m, lo, hi = _paired_bootstrap(a, b)
            contrasts[key][f"{cell}_minus_linear"] = {"mean": m, "ci95": [lo, hi]}

    out_json = RESULTS / f"panda_altimputer_{args.tag}.json"
    out_md = FIGS / f"panda_altimputer_{args.tag}.md"
    out_json.write_text(json.dumps({
        "config": vars(args),
        "settings": [{k: (str(v) if isinstance(v, Path) else v) for k, v in s.items()} for s in settings],
        "records": records, "summary": summary, "contrasts": contrasts,
    }, indent=2, default=str))

    # Markdown
    lines = ["# Panda Alt-Imputer Reviewer Defense", "",
             "Question: is structured imputation per se the lever, or is CSDI specifically required?",
             "Per-instance SAITS / BRITS train on the single test trajectory's missing pattern.",
             "Linear and CSDI follow the same protocol as Figure 1 (LORENZ63_ATTRACTOR_STD / lorenz96_attractor_std + grid-index seeds).", ""]
    for setting in settings:
        key = f"{setting['system']}_{setting['scenario']}"
        if key not in summary: continue
        lines += [f"## {key}", "",
                  "| Cell | mean ± std | median | Pr>0.5 | Pr>1.0 |",
                  "|:--|--:|--:|--:|--:|"]
        for cell in args.cells:
            s = summary[key].get(cell)
            if s is None: continue
            lines.append(f"| {cell} | {s['mean']:.2f} ± {s['std']:.2f} | {s['median']:.2f} | {100*s['pr_gt_0p5']:.0f}% | {100*s['pr_gt_1p0']:.0f}% |")
        lines += ["", "Paired Δ vs linear (95% bootstrap CI):", "",
                  "| Cell | Δ mean | CI | sign |", "|:--|--:|:--|:-:|"]
        for cname, c in contrasts.get(key, {}).items():
            sgn = "↑" if c["ci95"][0] > 0 else ("↓" if c["ci95"][1] < 0 else "≈")
            lines.append(f"| {cname} | {c['mean']:+.2f} | [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] | {sgn} |")
        lines.append("")
    out_md.write_text("\n".join(lines))

    print(f"\n[alt] saved -> {out_json}")
    print(f"[alt] saved -> {out_md}")
    print("\n[verdict] mean VPT@1.0:")
    for key, s in summary.items():
        vals = "  ".join(f"{c}={s[c]['mean']:.2f}" for c in args.cells if c in s)
        print(f"  {key}: {vals}")


if __name__ == "__main__":
    main()
