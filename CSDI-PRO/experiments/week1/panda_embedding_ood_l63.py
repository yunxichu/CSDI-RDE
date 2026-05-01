"""Panda representation-space OOD diagnostic for sparse L63 contexts.

Raw-patch diagnostics gave a useful negative result: linear interpolation is
closer to the clean trajectory than CSDI under curvature, local variance,
lag-1 autocorrelation, and spectral metrics, even though CSDI often rescues
Panda forecasts. This script asks the narrower mechanistic question:

    Does the rescue become visible in Panda's own token space?

For each scenario and seed we construct matched contexts:
  clean, linear-fill, CSDI-fill

Then we compare each corrupted context against the matched clean context at:
  1. Panda-normalized patch inputs
  2. DynamicsEmbedder output (RFF/token projection)
  3. Final encoder tokens
  4. Mean-pooled encoder latent

If linear-fill is farther from clean than CSDI in Panda representation space,
the "tokenizer/preprocessing OOD" mechanism is directly supported. If not, the
paper should phrase the mechanism as a puzzle and keep the main claim on the
well-supported sharp frontier + CSDI intervention law.

Run:
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python -u -m experiments.week1.panda_embedding_ood_l63
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

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA

from baselines.panda_model import PandaModel
from experiments.week1.corruption import make_corrupted_observations
from experiments.week1.lorenz63_utils import (
    LORENZ63_ATTRACTOR_STD,
    integrate_lorenz63,
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
CONTEXTS = ("clean", "linear", "csdi")
STAGES = ("patch", "embed", "encoder", "pooled")


def _as_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _as_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_as_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _prepare_panda_patches(model: PandaModel, ctx_td: np.ndarray, device: torch.device) -> torch.Tensor:
    """Replicate PandaModel.forecast preprocessing through patch extraction.

    Input ctx_td: [T, D]. Output: [1, D, P, patch_length] normalized patches.
    """
    cfg = model.cfg
    arr = np.asarray(ctx_td, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    context = torch.tensor(arr.T[None, :, :], device=device, dtype=torch.float32)
    B, C, L = context.shape
    if L >= cfg.context_length:
        x = context[:, :, -cfg.context_length:]
    else:
        pad_len = cfg.context_length - L
        pad = context[:, :, :1].expand(B, C, pad_len)
        x = torch.cat([pad, context], dim=-1)
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp_min(1e-5)
    x_n = (x - mean) / std
    patches = x_n.unfold(-1, cfg.patch_length, cfg.patch_stride)
    expected = cfg.context_length // cfg.patch_stride
    if patches.shape[2] != expected:
        raise RuntimeError(f"Panda patch count {patches.shape[2]} != {expected}")
    return patches


@torch.no_grad()
def _panda_representations(
    model: PandaModel,
    contexts: dict[str, np.ndarray],
    device: torch.device,
) -> dict[str, dict[str, np.ndarray]]:
    """Return representation arrays keyed by context then stage."""
    model.eval()
    out: dict[str, dict[str, np.ndarray]] = {}
    encoder = model.model["encoder"]
    for name, ctx in contexts.items():
        patches = _prepare_panda_patches(model, ctx, device)
        embed = encoder.embedder(patches)
        enc = encoder(patches)
        pooled = enc.mean(dim=2)
        out[name] = {
            "patch": patches.detach().cpu().numpy(),
            "embed": embed.detach().cpu().numpy(),
            "encoder": enc.detach().cpu().numpy(),
            "pooled": pooled.detach().cpu().numpy(),
        }
    return out


def _flatten_tokens(x: np.ndarray) -> np.ndarray:
    """Flatten [B,C,P,D] or [B,C,D] into token rows."""
    if x.ndim == 4:
        return x.reshape(-1, x.shape[-1])
    if x.ndim == 3:
        return x.reshape(-1, x.shape[-1])
    raise ValueError(f"unexpected representation shape {x.shape}")


def _paired_l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-token L2 distance for matched representation tensors."""
    aa = _flatten_tokens(a)
    bb = _flatten_tokens(b)
    if aa.shape != bb.shape:
        raise ValueError(f"paired shape mismatch {aa.shape} vs {bb.shape}")
    return np.linalg.norm(aa - bb, axis=1)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = _flatten_tokens(a)
    bb = _flatten_tokens(b)
    num = np.sum(aa * bb, axis=1)
    den = np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1) + 1e-12
    return 1.0 - num / den


def _nearest_clean_l2(clean: np.ndarray, test: np.ndarray, max_clean: int = 2048) -> np.ndarray:
    """Nearest-neighbor distance from each test token to the clean token cloud."""
    c = _flatten_tokens(clean).astype(np.float32)
    t = _flatten_tokens(test).astype(np.float32)
    if len(c) > max_clean:
        idx = np.linspace(0, len(c) - 1, max_clean).astype(int)
        c = c[idx]
    # Small here: L63 uses 3 channels x 32 patches = 96 tokens per seed.
    d2 = ((t[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)
    return np.sqrt(d2.min(axis=1))


def _make_contexts(seed: int, sc: dict[str, Any], attr_std: float,
                   n_ctx: int, dt: float) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    # v2-protocol-aligned: same mask seed scheme as phase_transition_grid_l63_v2
    # so this diagnostic measures the same scenarios as Figure 1.
    GRID_INDEX = {"SP65": 4, "SP82": 6}
    traj = integrate_lorenz63(n_ctx, dt=dt, spinup=2000, seed=seed).astype(np.float32)
    obs_res = make_corrupted_observations(
        traj,
        mask_regime="iid_time",
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
        "csdi": impute(
            observed, kind="csdi",
            sigma_override=float(sc["noise_std_frac"]) * attr_std,
        ).astype(np.float32),
    }, obs_res.metadata


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_scenario: dict[str, Any] = {}
    for sc in sorted({r["scenario"] for r in records}):
        sub = [r for r in records if r["scenario"] == sc]
        out_sc: dict[str, Any] = {}
        for ctx in ("linear", "csdi"):
            out_ctx: dict[str, Any] = {}
            for stage in STAGES:
                vals = np.concatenate([np.asarray(r["distances"][ctx][stage]["paired_l2"]) for r in sub])
                cos = np.concatenate([np.asarray(r["distances"][ctx][stage]["cosine"]) for r in sub])
                nn = np.concatenate([np.asarray(r["distances"][ctx][stage]["nn_l2"]) for r in sub])
                out_ctx[stage] = {
                    "paired_l2_mean": float(vals.mean()),
                    "paired_l2_median": float(np.median(vals)),
                    "paired_l2_q90": float(np.quantile(vals, 0.90)),
                    "cosine_mean": float(cos.mean()),
                    "nn_l2_mean": float(nn.mean()),
                    "nn_l2_median": float(np.median(nn)),
                    "n_tokens": int(len(vals)),
                }
            out_sc[ctx] = out_ctx

        ratios: dict[str, Any] = {}
        for stage in STAGES:
            lin = out_sc["linear"][stage]["paired_l2_mean"]
            csdi = out_sc["csdi"][stage]["paired_l2_mean"]
            ratios[stage] = {
                "paired_l2_linear_over_csdi": float(lin / (csdi + 1e-12)),
                "mechanism_support": bool(lin > csdi),
            }
        out_sc["ratios"] = ratios
        by_scenario[sc] = out_sc
    return by_scenario


def _plot_distance_bars(summary: dict[str, Any], out_png: Path) -> None:
    fig, axes = plt.subplots(1, len(summary), figsize=(6 * len(summary), 4.5), sharey=False)
    if len(summary) == 1:
        axes = [axes]
    x = np.arange(len(STAGES))
    width = 0.36
    for ax, (sc, data) in zip(axes, summary.items()):
        lin = [data["linear"][stage]["paired_l2_mean"] for stage in STAGES]
        csdi = [data["csdi"][stage]["paired_l2_mean"] for stage in STAGES]
        ax.bar(x - width / 2, lin, width, label="linear vs clean", color="C1")
        ax.bar(x + width / 2, csdi, width, label="CSDI vs clean", color="C2")
        ax.set_title(f"{sc}: Panda-space paired distance")
        ax.set_xticks(x)
        ax.set_xticklabels(STAGES, rotation=25, ha="right")
        ax.set_ylabel("mean paired L2")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _plot_pca(stage_reps: dict[str, list[np.ndarray]], scenario: str, stage: str, out_png: Path) -> None:
    clouds = {ctx: np.concatenate([_flatten_tokens(x) for x in xs], axis=0)
              for ctx, xs in stage_reps.items()}
    clean = clouds["clean"]
    pca = PCA(n_components=2, random_state=0).fit(clean)
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 5.2))
    for ctx, color, marker, alpha in [
        ("clean", "C0", "o", 0.35),
        ("linear", "C1", "x", 0.55),
        ("csdi", "C2", "+", 0.55),
    ]:
        z = pca.transform(clouds[ctx])
        ax.scatter(z[:, 0], z[:, 1], s=12, alpha=alpha, label=ctx, c=color, marker=marker)
    ax.set_title(f"{scenario}: Panda {stage} PCA (fit on clean)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _write_markdown(summary: dict[str, Any], out_md: Path) -> None:
    lines = [
        "# Panda Representation-Space OOD Diagnostic (L63)",
        "",
        "Distance is matched-to-clean paired L2 unless noted.",
        "",
    ]
    for sc, data in summary.items():
        lines += [
            f"## {sc}",
            "",
            "| Stage | Linear mean | CSDI mean | Linear/CSDI | Supports tokenizer-OOD? |",
            "|:--|--:|--:|--:|:--:|",
        ]
        for stage in STAGES:
            lin = data["linear"][stage]["paired_l2_mean"]
            csdi = data["csdi"][stage]["paired_l2_mean"]
            ratio = data["ratios"][stage]["paired_l2_linear_over_csdi"]
            support = "yes" if data["ratios"][stage]["mechanism_support"] else "no"
            lines.append(f"| {stage} | {lin:.4f} | {csdi:.4f} | {ratio:.3f} | {support} |")
        lines.append("")
    support_count = sum(
        int(data["ratios"][stage]["mechanism_support"])
        for data in summary.values()
        for stage in STAGES
    )
    total = len(summary) * len(STAGES)
    if support_count >= total // 2 + 1:
        verdict = (
            "Verdict: Panda representation distances support the tokenizer/OOD "
            "mechanism in a majority of tested scenario-stage cells."
        )
    else:
        verdict = (
            "Verdict: Panda representation distances do not support the simple "
            "claim that linear-fill is farther from clean than CSDI-fill. The "
            "mechanism should be written as a non-obvious puzzle unless a more "
            "targeted internal signal is found."
        )
    lines += ["## Verdict", "", verdict, ""]
    out_md.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed_offset", type=int, default=0)
    ap.add_argument("--n_ctx", type=int, default=512)
    ap.add_argument("--dt", type=float, default=0.025)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_tag", default="l63_sp65_sp82_5seed")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    from methods.csdi_impute_adapter import set_csdi_attractor_std, set_csdi_checkpoint

    attr_std = float(LORENZ63_ATTRACTOR_STD)  # v2-protocol-aligned
    set_csdi_checkpoint(str(CSDI_CKPT_L63))
    set_csdi_attractor_std(attr_std)
    print(f"[panda-embed] device={device}")
    print(f"[panda-embed] CSDI ckpt={CSDI_CKPT_L63}")
    print(f"[panda-embed] L63 attractor_std={attr_std:.4f}")

    model = PandaModel.from_pretrained(PANDA_DIR).to(device).eval()
    print(f"[panda-embed] Panda loaded from {PANDA_DIR}")

    records: list[dict[str, Any]] = []
    pca_reps: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        sc["name"]: {stage: {ctx: [] for ctx in CONTEXTS} for stage in ("embed", "encoder")}
        for sc in SCENARIOS
    }

    seeds = list(range(args.seed_offset, args.seed_offset + args.n_seeds))
    for sc in SCENARIOS:
        print(f"\n=== {sc['name']}  s={sc['sparsity']} sigma={sc['noise_std_frac']} ===")
        for seed in seeds:
            contexts, meta = _make_contexts(seed, sc, attr_std, args.n_ctx, args.dt)
            reps = _panda_representations(model, contexts, device)
            distances: dict[str, dict[str, Any]] = {"linear": {}, "csdi": {}}
            for ctx in ("linear", "csdi"):
                for stage in STAGES:
                    distances[ctx][stage] = {
                        "paired_l2": _paired_l2(reps["clean"][stage], reps[ctx][stage]),
                        "cosine": _cosine_distance(reps["clean"][stage], reps[ctx][stage]),
                        "nn_l2": _nearest_clean_l2(reps["clean"][stage], reps[ctx][stage]),
                    }
            for stage in ("embed", "encoder"):
                for ctx in CONTEXTS:
                    pca_reps[sc["name"]][stage][ctx].append(reps[ctx][stage])

            rec = {
                "scenario": sc["name"],
                "seed": seed,
                "sparsity": float(sc["sparsity"]),
                "noise_std_frac": float(sc["noise_std_frac"]),
                "metadata": meta,
                "distances": distances,
            }
            records.append(rec)
            lin_enc = np.mean(distances["linear"]["encoder"]["paired_l2"])
            csdi_enc = np.mean(distances["csdi"]["encoder"]["paired_l2"])
            print(
                f"  seed={seed} keep={meta.get('keep_frac', meta.get('keep_fraction', np.nan)):.3f} "
                f"encoder L2: linear={lin_enc:.4f} csdi={csdi_enc:.4f} "
                f"ratio={lin_enc / (csdi_enc + 1e-12):.3f}"
            )

    summary = _aggregate(records)

    out_json = RESULTS / f"panda_embedding_ood_{args.out_tag}.json"
    out_md = FIGS / f"panda_embedding_ood_{args.out_tag}.md"
    out_bar = FIGS / f"panda_embedding_ood_{args.out_tag}_bars.png"
    out_json.write_text(json.dumps({
        "config": {
            "n_seeds": args.n_seeds,
            "seed_offset": args.seed_offset,
            "n_ctx": args.n_ctx,
            "dt": args.dt,
            "panda_dir": str(PANDA_DIR),
            "csdi_ckpt": str(CSDI_CKPT_L63),
        },
        "records": _as_jsonable(records),
        "summary": _as_jsonable(summary),
    }, indent=2))
    _write_markdown(summary, out_md)
    _plot_distance_bars(summary, out_bar)

    for scenario, by_stage in pca_reps.items():
        for stage, reps in by_stage.items():
            _plot_pca(stage_reps=reps, scenario=scenario, stage=stage,
                      out_png=FIGS / f"panda_embedding_ood_{args.out_tag}_{scenario}_{stage}_pca.png")

    print(f"\n[saved] {out_json}")
    print(f"[saved] {out_md}")
    print(f"[saved] {out_bar}")
    print("\n[verdict] paired L2 linear/CSDI ratios (<1 means linear closer to clean):")
    for sc, data in summary.items():
        print(f"  {sc}")
        for stage in STAGES:
            ratio = data["ratios"][stage]["paired_l2_linear_over_csdi"]
            print(f"    {stage:7s}: {ratio:.3f}")


if __name__ == "__main__":
    main()
