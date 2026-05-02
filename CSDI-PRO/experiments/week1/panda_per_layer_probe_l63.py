"""Per-layer Panda encoder probe (P3.B2 follow-up to B1).

Background (B1 finding, see §4.2): on L63 SP65, CSDI's token-distance to
clean is ~6× smaller than SAITS-pretrained's, matching CSDI's small VPT
advantage. On SP82, SAITS is *closer* to clean than CSDI in every
checked stage (patch / embedder / encoder-final / pooled), yet they tie
on VPT (CSDI − SAITS = +0.06, [−0.31, +0.59] from §4.4). Encoder
geometry therefore does NOT order SAITS-vs-CSDI at the floor band.

The original §6.4 hypothesis blamed "decoder-side latent dynamics". But
Panda-72M has no decoder — its head is a linear projection from pooled
encoder. So the residual difference must live in **per-layer encoder
dynamics**: even if the final pooled state has SAITS < CSDI, an
intermediate layer might flip the order in a way that's relevant for
the autoregressive rollout (which iteratively re-encodes Panda's own
forecast as new context).

This script registers forward hooks on each of the 12 PandaLayer outputs,
runs the same {clean, linear, SAITS-pretrained, CSDI} contexts as B1
(matched seeds and corruption draws), and reports paired L2 distance to
clean at every layer.

Pre-registered reading:
  - If at any intermediate layer SAITS distance > CSDI distance at SP82
    (matching VPT order), we have a measured discriminator and §4.2 /
    §6.4 mechanism story closes.
  - If every layer keeps SAITS < CSDI at SP82 (matching final pooled),
    encoder geometry is *not* the residual discriminator. We update
    §6.4 to "encoder-side distances at every layer do not order
    SAITS-vs-CSDI at the floor; the residual difference must lie in
    something we have not measured (e.g. autoregressive-rollout
    dynamics or the head's linear projection coefficients)" — open
    but bounded.

Run:
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  PYTHONPATH=/home/rhl/Github/CSDI-PRO python -u \
      -m experiments.week1.panda_per_layer_probe_l63 \
      --saits_ckpt experiments/week2_modules/ckpts/saits_l63_pretrained/20260501_T153756/SAITS.pypots \
      --n_seeds 5 --tag l63_sp65_sp82_per_layer_5seed
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

for _var in [
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(_var, "4")

import numpy as np
import torch

from baselines.panda_model import PandaModel
from experiments.week1.corruption import make_corrupted_observations
from experiments.week1.lorenz63_utils import (
    LORENZ63_ATTRACTOR_STD,
    integrate_lorenz63,
)
from experiments.week1.panda_embedding_ood_l63 import (
    _prepare_panda_patches,
    _saits_pretrained_impute,
)
from methods.dynamics_impute import impute


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "week1" / "results"
FIGS = REPO / "experiments" / "week1" / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
PANDA_DIR = REPO / "baselines" / "panda-72M"
CSDI_CKPT_L63 = REPO / "experiments" / "week2_modules" / "ckpts" / "dyn_csdi_full_v6_center_ep20.pt"

SCENARIOS = [
    {"name": "SP65", "sparsity": 0.65, "noise_std_frac": 0.0},
    {"name": "SP82", "sparsity": 0.82, "noise_std_frac": 0.0},
]
GRID_INDEX = {"SP65": 4, "SP82": 6}


def _make_contexts(seed, sc, attr_std, n_ctx, dt, saits_ckpt):
    traj = integrate_lorenz63(n_ctx, dt=dt, spinup=2000, seed=seed).astype(np.float32)
    obs_res = make_corrupted_observations(
        traj, mask_regime="iid_time",
        sparsity=float(sc["sparsity"]),
        noise_std_frac=float(sc["noise_std_frac"]),
        attractor_std=attr_std,
        seed=1000 * seed + 5000 + GRID_INDEX[sc["name"]],
        patch_length=16,
    )
    observed = obs_res.observed
    return {
        "clean": traj,
        "linear": impute(observed, kind="linear").astype(np.float32),
        "csdi": impute(observed, kind="csdi",
                        sigma_override=float(sc["noise_std_frac"]) * attr_std).astype(np.float32),
        "saits_pretrained": _saits_pretrained_impute(
            observed, n_features=traj.shape[1], n_steps=n_ctx, ckpt_path=saits_ckpt),
    }, obs_res.metadata


@torch.no_grad()
def _per_layer_reps(model: PandaModel, contexts, device):
    """Return {ctx: {layer_i: [B,C,P,D]}} for i in 0..n_layers (0 = embedder output, 1..L = post-layer-i)."""
    n_layers = len(model.model["encoder"].layers)
    out: dict[str, dict[int, np.ndarray]] = {}
    handles = []
    captured: dict[int, torch.Tensor] = {}

    def _make_hook(idx):
        def hook(_module, _inputs, output):
            captured[idx] = output.detach()
        return hook

    for i, layer in enumerate(model.model["encoder"].layers):
        h = layer.register_forward_hook(_make_hook(i + 1))  # post-layer index 1..n_layers
        handles.append(h)

    try:
        for name, ctx in contexts.items():
            captured.clear()
            patches = _prepare_panda_patches(model, ctx, device)
            embed = model.model["encoder"].embedder(patches)  # post-embedder
            captured[0] = embed
            _ = model.model["encoder"](patches)  # triggers all layer hooks
            out[name] = {idx: t.detach().cpu().numpy() for idx, t in captured.items()}
    finally:
        for h in handles:
            h.remove()
    return out


def _flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, x.shape[-1])


def _paired_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa, bb = _flatten(a), _flatten(b)
    return np.linalg.norm(aa - bb, axis=1)


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa, bb = _flatten(a), _flatten(b)
    num = (aa * bb).sum(axis=1)
    den = np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1) + 1e-12
    return 1.0 - num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--n_ctx", type=int, default=512)
    ap.add_argument("--dt", type=float, default=0.025)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--saits_ckpt", required=True)
    ap.add_argument("--tag", default="l63_sp65_sp82_per_layer_5seed")
    args = ap.parse_args()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit("Set CUDA_VISIBLE_DEVICES explicitly")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    attr_std = float(LORENZ63_ATTRACTOR_STD)

    from methods.csdi_impute_adapter import set_csdi_attractor_std, set_csdi_checkpoint
    set_csdi_checkpoint(str(CSDI_CKPT_L63))
    set_csdi_attractor_std(attr_std)

    print(f"[per-layer] device={device}")
    print(f"[per-layer] L63 attractor_std={attr_std:.4f}")
    print(f"[per-layer] CSDI ckpt: {CSDI_CKPT_L63}")
    print(f"[per-layer] SAITS ckpt: {args.saits_ckpt}")

    model = PandaModel.from_pretrained(PANDA_DIR).to(device).eval()
    n_layers = len(model.model["encoder"].layers)
    print(f"[per-layer] Panda loaded; {n_layers} encoder layers")

    seeds = list(range(args.seed_offset, args.seed_offset + args.n_seeds))
    records: list[dict[str, Any]] = []

    for sc in SCENARIOS:
        print(f"\n=== {sc['name']}  s={sc['sparsity']} sigma={sc['noise_std_frac']} ===")
        for seed in seeds:
            contexts, meta = _make_contexts(seed, sc, attr_std, args.n_ctx, args.dt,
                                              args.saits_ckpt)
            reps = _per_layer_reps(model, contexts, device)
            distances: dict[str, dict[int, dict[str, list[float]]]] = {}
            for ctx in ("linear", "saits_pretrained", "csdi"):
                distances[ctx] = {}
                for layer_idx, rep_arr in reps[ctx].items():
                    clean_arr = reps["clean"][layer_idx]
                    distances[ctx][layer_idx] = {
                        "paired_l2": _paired_l2(clean_arr, rep_arr).tolist(),
                        "cosine": _cosine(clean_arr, rep_arr).tolist(),
                    }
            records.append({
                "scenario": sc["name"], "seed": seed,
                "sparsity": float(sc["sparsity"]),
                "noise_std_frac": float(sc["noise_std_frac"]),
                "metadata": meta,
                "distances": distances,
            })
            # quick log: layer-12 paired L2
            l_lin = float(np.mean(distances["linear"][n_layers]["paired_l2"]))
            l_sai = float(np.mean(distances["saits_pretrained"][n_layers]["paired_l2"]))
            l_csd = float(np.mean(distances["csdi"][n_layers]["paired_l2"]))
            print(f"  seed={seed} layer{n_layers}  linear={l_lin:6.2f}  "
                  f"saits={l_sai:6.2f}  csdi={l_csd:6.2f}  saits/csdi={l_sai/(l_csd+1e-12):.2f}")

    # Aggregate per (scenario, ctx, layer)
    summary: dict[str, Any] = {}
    for sc in SCENARIOS:
        sc_name = sc["name"]
        sub = [r for r in records if r["scenario"] == sc_name]
        out_sc: dict[str, Any] = {}
        for ctx in ("linear", "saits_pretrained", "csdi"):
            out_ctx: dict[int, dict[str, float]] = {}
            for layer_idx in sorted(sub[0]["distances"][ctx].keys()):
                vals = np.concatenate([np.asarray(r["distances"][ctx][layer_idx]["paired_l2"])
                                         for r in sub])
                cos = np.concatenate([np.asarray(r["distances"][ctx][layer_idx]["cosine"])
                                        for r in sub])
                out_ctx[layer_idx] = {
                    "paired_l2_mean": float(vals.mean()),
                    "paired_l2_median": float(np.median(vals)),
                    "cosine_mean": float(cos.mean()),
                    "n": int(len(vals)),
                }
            out_sc[ctx] = out_ctx
        # Ratios at every layer
        out_sc["ratios"] = {}
        for layer_idx in sorted(out_sc["linear"].keys()):
            lin = out_sc["linear"][layer_idx]["paired_l2_mean"]
            sai = out_sc["saits_pretrained"][layer_idx]["paired_l2_mean"]
            csd = out_sc["csdi"][layer_idx]["paired_l2_mean"]
            out_sc["ratios"][layer_idx] = {
                "linear_over_csdi": float(lin / (csd + 1e-12)),
                "linear_over_saits": float(lin / (sai + 1e-12)),
                "saits_over_csdi": float(sai / (csd + 1e-12)),
            }
        summary[sc_name] = out_sc

    out_json = RESULTS / f"panda_per_layer_probe_{args.tag}.json"
    out_json.write_text(json.dumps({
        "config": vars(args), "summary": summary, "n_layers": n_layers,
    }, indent=2))
    print(f"\n[saved] {out_json}")

    # Pretty verdict
    print("\n[verdict] paired L2 saits/csdi ratio per layer (< 1 ⇒ SAITS closer to clean than CSDI):")
    print(f"{'layer':>6}  " + "  ".join(f"{sc['name']:>6}" for sc in SCENARIOS))
    for layer_idx in sorted(summary[SCENARIOS[0]['name']]['ratios'].keys()):
        row = [f"{summary[sc['name']]['ratios'][layer_idx]['saits_over_csdi']:6.2f}"
               for sc in SCENARIOS]
        print(f"{layer_idx:>6}  " + "  ".join(row))


if __name__ == "__main__":
    main()
