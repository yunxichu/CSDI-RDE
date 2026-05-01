"""Patch-OOD geometry diagnostic on L63 — the mechanism figure for the 2026-04-26
paper pivot.

Question: when linear-fill of sparse contexts collapses Panda but CSDI-fill
rescues it, what changes in the patch geometry?  We compare four context types
on L63 at SP65/SP82 (pure sparsity, sigma=0):

  1. Clean dense:                  raw L63 trajectory
  2. Sparse + linear-fill:         linear interpolation of NaN gaps
  3. Sparse + CSDI-fill:           dynamics-aware diffusion imputation
  4. Delay-coord (Takens) patches: from the clean trajectory, in delay space

Per-patch metrics (no training, no GPU besides CSDI imputation):
  * patch curvature: mean ‖x[t+1] − 2 x[t] + x[t−1]‖₂ over interior steps
  * chord-length efficiency: arc length / endpoint distance
  * low-curvature fraction: Pr(curvature < tau_lc); quantifies "straight-line"
    interpolation artifacts
  * Wasserstein-1 distance to clean distribution (per-metric)
  * PCA scatter (separate fits for ambient vs delay; cannot be combined because
    of dimension mismatch — flatten ambient patch [16, 3] vs delay vector
    [(L+1)*3])

Output:
  results/l63_patch_ood_sp65_sp82.json
  figures/l63_patch_ood_sp65_sp82.png

Run:
  CUDA_VISIBLE_DEVICES=0 python -u -m experiments.week1.patch_ood_diagnostic_l63

Mandatory: include delay-coord context (do not skip even if running cheap-mode);
otherwise the mechanism figure shows only the CSDI half of the story.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance

from experiments.week1.corruption import make_corrupted_observations
from experiments.week1.lorenz63_utils import (
    integrate_lorenz63, compute_attractor_std as l63_attractor_std,
)
from methods.dynamics_impute import impute

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "week1" / "results"
FIGS = REPO / "experiments" / "week1" / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

CSDI_CKPT_L63 = REPO / "experiments/week2_modules/ckpts/dyn_csdi_full_v6_center_ep20.pt"

PATCH_LEN = 16  # match Panda patch length
DELAY_LAGS = (1, 5, 10, 15)  # fixed reasonable Takens window for L63 (L=4)
LOW_CURVATURE_TAU_FRAC = 0.10  # fraction of clean median curvature

SCENARIOS = [
    {"name": "SP65", "sparsity": 0.65, "noise_std_frac": 0.0},
    {"name": "SP82", "sparsity": 0.82, "noise_std_frac": 0.0},
]
SEEDS = (0, 1, 2)

# Each context is (label, color) for plotting
CONTEXTS = [
    ("clean",  "C0"),
    ("linear", "C1"),
    ("csdi",   "C2"),
    ("delay",  "C3"),
]


# ----------------------------- patch builders --------------------------------

def make_patches(traj: np.ndarray, patch_len: int = PATCH_LEN,
                  stride: int = 1) -> np.ndarray:
    """Sliding window of shape [N_patches, patch_len, D]."""
    T, D = traj.shape
    N = max(0, (T - patch_len) // stride + 1)
    if N <= 0:
        return np.empty((0, patch_len, D), dtype=np.float32)
    out = np.stack([traj[i * stride : i * stride + patch_len]
                    for i in range(N)], axis=0)
    return out.astype(np.float32)


def build_delay_vectors(traj: np.ndarray, lags: tuple[int, ...] = DELAY_LAGS,
                         stride: int = 1) -> np.ndarray:
    """Per timestep, build delay vector [x(t), x(t-l1), ..., x(t-lL)] flattened.
    Returns [N, (L+1)*D] where N = T - max(lags), one delay vector per anchor t.
    """
    T, D = traj.shape
    Lmax = max(lags)
    anchors = np.arange(Lmax, T, stride)
    rows = []
    for t in anchors:
        cols = [traj[t]]
        for tau in lags:
            cols.append(traj[t - tau])
        rows.append(np.concatenate(cols, axis=0))
    return np.stack(rows, axis=0).astype(np.float32)


# ----------------------------- metrics ---------------------------------------

def patch_curvature(patches: np.ndarray) -> np.ndarray:
    """Mean ‖x[t+1] − 2 x[t] + x[t−1]‖₂ over interior points of each patch.
    Input [N, L, D] → output [N]."""
    if patches.ndim == 2:  # delay vectors — treat each as a 1×D "patch"
        # Curvature is undefined for single points; reshape (N, (L+1)*D) into
        # (N, L+1, D_inner) so we can take 2nd-differences along the lag axis.
        N, F = patches.shape
        # F = (L+1) * D_inner; we don't know D_inner a priori, but for L63 it's 3.
        # Fall back: treat as a 1-D series of length F and take 1-D 2nd diff.
        d2 = patches[:, 2:] - 2 * patches[:, 1:-1] + patches[:, :-2]
        return np.linalg.norm(d2, axis=1) / max(d2.shape[1], 1)
    d2 = patches[:, 2:, :] - 2 * patches[:, 1:-1, :] + patches[:, :-2, :]
    return np.linalg.norm(d2, axis=(1, 2)) / max(d2.shape[1], 1)


def patch_chord_efficiency(patches: np.ndarray) -> np.ndarray:
    """Arc length / endpoint distance.  >= 1; 1 means perfectly straight."""
    if patches.ndim == 2:  # delay vectors — same fallback as curvature
        seg = np.diff(patches, axis=1)  # treat features as 1-D series
        arc = np.abs(seg).sum(axis=1)
        endp = np.abs(patches[:, -1] - patches[:, 0]) + 1e-9
        return arc / endp
    seg = np.diff(patches, axis=1)  # [N, L-1, D]
    arc = np.linalg.norm(seg, axis=2).sum(axis=1)  # path length
    endp = np.linalg.norm(patches[:, -1, :] - patches[:, 0, :], axis=1) + 1e-9
    return arc / endp


# ----------------------------- main pipeline ---------------------------------

def run_one_seed(seed: int, sc: dict, attr_std: float):
    """Compute all 4 contexts and their metrics for one (scenario, seed)."""
    n_ctx = 768
    traj = integrate_lorenz63(n_ctx, dt=0.01, spinup=2000, seed=seed)
    obs_res = make_corrupted_observations(
        traj, mask_regime="iid_time",
        sparsity=sc["sparsity"], noise_std_frac=sc["noise_std_frac"],
        attractor_std=attr_std, seed=seed * 100 + 7, patch_length=PATCH_LEN,
    )
    observed = obs_res.observed
    linear = impute(observed, kind="linear").astype(np.float32)
    csdi = impute(observed, kind="csdi").astype(np.float32)
    delay_vecs = build_delay_vectors(traj.astype(np.float32))

    # Patches in ambient space; delay = vectors directly
    patches = {
        "clean":  make_patches(traj.astype(np.float32)),
        "linear": make_patches(linear),
        "csdi":   make_patches(csdi),
        "delay":  delay_vecs,
    }
    metrics = {}
    for name, P in patches.items():
        if len(P) == 0:
            metrics[name] = {"curvature": np.array([]), "chord": np.array([])}
            continue
        metrics[name] = {
            "curvature": patch_curvature(P),
            "chord": patch_chord_efficiency(P),
        }
    return patches, metrics, obs_res.metadata


def aggregate_seeds(per_seed: list[tuple]) -> dict:
    """Concatenate metric arrays across seeds for one scenario."""
    out = {}
    for ctx_name, _ in CONTEXTS:
        cur_all = np.concatenate([m[ctx_name]["curvature"] for _, m, _ in per_seed
                                    if len(m[ctx_name]["curvature"])])
        cho_all = np.concatenate([m[ctx_name]["chord"] for _, m, _ in per_seed
                                    if len(m[ctx_name]["chord"])])
        out[ctx_name] = {"curvature": cur_all, "chord": cho_all}
    return out


def compute_wasserstein(agg: dict) -> dict:
    """Wasserstein-1 distance from each context's metric distribution to clean."""
    ref_cur = agg["clean"]["curvature"]
    ref_cho = agg["clean"]["chord"]
    out = {}
    for ctx_name, _ in CONTEXTS:
        if ctx_name == "clean":
            out[ctx_name] = {"W1_curvature": 0.0, "W1_chord": 0.0}
            continue
        c = agg[ctx_name]["curvature"]
        ch = agg[ctx_name]["chord"]
        out[ctx_name] = {
            "W1_curvature": float(wasserstein_distance(ref_cur, c)) if len(c) else float("nan"),
            "W1_chord": float(wasserstein_distance(ref_cho, ch)) if len(ch) else float("nan"),
        }
    return out


def low_curvature_fraction(agg: dict) -> dict:
    """Fraction of patches whose curvature is below tau_lc · median(clean)."""
    ref_med = float(np.median(agg["clean"]["curvature"])) if len(agg["clean"]["curvature"]) else 1.0
    tau = LOW_CURVATURE_TAU_FRAC * ref_med
    out = {"tau_lc": tau, "by_context": {}}
    for ctx_name, _ in CONTEXTS:
        c = agg[ctx_name]["curvature"]
        out["by_context"][ctx_name] = float((c < tau).mean()) if len(c) else float("nan")
    return out


# ----------------------------- plotting --------------------------------------

def plot_panel(scenario_name: str, all_patches_per_seed: list,
                agg: dict, w1: dict, low_curv: dict, out_png: Path):
    """One PNG per scenario: 2x2 panel.
       Top-left:  PCA scatter ambient (clean / linear / csdi)
       Top-right: PCA scatter delay (delay only) — different geometry callout
       Bottom-left: curvature histogram (4 colors)
       Bottom-right: chord-length histogram (4 colors)
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    # --- collect ambient patches across seeds ---
    amb = {n: [] for n, _ in CONTEXTS if n != "delay"}
    delay_all = []
    for patches, _, _ in all_patches_per_seed:
        for name in amb:
            if len(patches[name]) > 0:
                amb[name].append(patches[name].reshape(len(patches[name]), -1))
        if len(patches["delay"]) > 0:
            delay_all.append(patches["delay"])
    for n in amb:
        amb[n] = np.concatenate(amb[n], axis=0) if amb[n] else np.zeros((0, 0))
    delay_flat = np.concatenate(delay_all, axis=0) if delay_all else np.zeros((0, 0))

    # --- ambient PCA fit on clean, project all ---
    ax = axes[0, 0]
    if len(amb["clean"]) > 0:
        pca = PCA(n_components=2).fit(amb["clean"])
        for name, color in CONTEXTS:
            if name == "delay" or name not in amb or len(amb[name]) == 0:
                continue
            proj = pca.transform(amb[name])
            ax.scatter(proj[:, 0], proj[:, 1], s=4, alpha=0.35, color=color, label=name)
        ax.set_title(f"{scenario_name} — ambient patch PCA (fit on clean)")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=8, markerscale=2.5, loc="best")
        ax.grid(alpha=0.3)

    # --- delay PCA fit on delay ---
    ax = axes[0, 1]
    if len(delay_flat) > 0:
        pca_d = PCA(n_components=2).fit(delay_flat)
        proj = pca_d.transform(delay_flat)
        ax.scatter(proj[:, 0], proj[:, 1], s=4, alpha=0.35, color="C3", label="delay (Takens)")
        ax.set_title(f"{scenario_name} — delay-coord patch PCA (own basis)")
        ax.set_xlabel("PC1 (delay)"); ax.set_ylabel("PC2 (delay)")
        ax.legend(fontsize=8, markerscale=2.5, loc="best")
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.98,
                "different geometric object\nfrom ambient patches\n→ separate OOD-mitigation channel",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    # --- curvature histogram ---
    ax = axes[1, 0]
    for name, color in CONTEXTS:
        c = agg[name]["curvature"]
        if len(c) == 0:
            continue
        ax.hist(c, bins=50, density=True, alpha=0.45, color=color, label=name)
    ax.set_title(f"{scenario_name} — patch curvature distribution")
    ax.set_xlabel("mean ‖x[t+1] − 2 x[t] + x[t−1]‖₂ per patch")
    ax.set_ylabel("density")
    if len(agg["clean"]["curvature"]) > 0:
        ref = float(np.median(agg["clean"]["curvature"]))
        ax.axvline(LOW_CURVATURE_TAU_FRAC * ref, color="k", linestyle=":",
                    linewidth=1, label=f"low-curv τ = 10% × clean median")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- chord-length histogram ---
    ax = axes[1, 1]
    for name, color in CONTEXTS:
        c = agg[name]["chord"]
        if len(c) == 0:
            continue
        # clip extreme tails for readability
        c_plot = c[c < np.quantile(c, 0.99)] if len(c) > 50 else c
        ax.hist(c_plot, bins=50, density=True, alpha=0.45, color=color, label=name)
    ax.set_title(f"{scenario_name} — chord efficiency (arc length / endpoint dist)")
    ax.set_xlabel("chord ratio (1 = perfectly straight)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- annotate W1 + low-curv summary on figure ---
    parts = [f"{scenario_name}  s={SCENARIOS_BY_NAME[scenario_name]['sparsity']}, σ=0"]
    parts.append("Wasserstein-1 vs clean (curvature):")
    for name, _ in CONTEXTS:
        if name == "clean":
            continue
        parts.append(f"  {name:8s}  {w1[name]['W1_curvature']:.4f}")
    parts.append(f"low-curv frac (τ={low_curv['tau_lc']:.4f}):")
    for name, _ in CONTEXTS:
        parts.append(f"  {name:8s}  {low_curv['by_context'][name]:.3f}")
    fig.text(0.99, 0.99, "\n".join(parts), ha="right", va="top",
             fontsize=8, family="monospace",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    plt.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"[saved] {out_png}")


SCENARIOS_BY_NAME = {sc["name"]: sc for sc in SCENARIOS}


def main():
    from methods.csdi_impute_adapter import set_csdi_checkpoint, set_csdi_attractor_std
    set_csdi_checkpoint(str(CSDI_CKPT_L63))
    attr_std = float(l63_attractor_std())
    set_csdi_attractor_std(attr_std)
    print(f"[diagnostic] CSDI ckpt: {CSDI_CKPT_L63}")
    print(f"[diagnostic] L63 attractor_std: {attr_std:.4f}")
    print(f"[diagnostic] patch_len={PATCH_LEN}, delay_lags={DELAY_LAGS}, seeds={SEEDS}")

    big_summary = {}
    fig, axes = plt.subplots(len(SCENARIOS), 4, figsize=(18, 4 * len(SCENARIOS)),
                              squeeze=False)

    for sc in SCENARIOS:
        print(f"\n=== scenario {sc['name']}  s={sc['sparsity']}  σ={sc['noise_std_frac']} ===")
        per_seed = []
        for seed in SEEDS:
            patches, metrics, meta = run_one_seed(seed, sc, attr_std)
            per_seed.append((patches, metrics, meta))
            print(f"  seed={seed}  keep_frac={meta['keep_frac']:.3f}  "
                  f"max_gap_steps={meta.get('all_missing_gap_max_steps', 'NA')}  "
                  f"obs/patch={meta.get('expected_obs_per_patch', 'NA'):.2f}")
            for ctx_name, _ in CONTEXTS:
                c = metrics[ctx_name]["curvature"]
                ch = metrics[ctx_name]["chord"]
                print(f"    {ctx_name:8s}  N={len(c):4d}  "
                      f"curv μ={c.mean():.4f}  chord μ={ch.mean() if len(ch) else 0:.3f}")

        agg = aggregate_seeds(per_seed)
        w1 = compute_wasserstein(agg)
        lc = low_curvature_fraction(agg)
        plot_panel(sc["name"], per_seed, agg, w1, lc,
                   FIGS / f"l63_patch_ood_{sc['name']}.png")

        big_summary[sc["name"]] = {
            "config": sc,
            "n_seeds": len(SEEDS),
            "patch_length": PATCH_LEN,
            "delay_lags": list(DELAY_LAGS),
            "wasserstein_to_clean": w1,
            "low_curvature_fraction": lc,
            "metrics_per_context": {
                ctx: {
                    "curvature_mean": float(agg[ctx]["curvature"].mean()) if len(agg[ctx]["curvature"]) else float("nan"),
                    "curvature_median": float(np.median(agg[ctx]["curvature"])) if len(agg[ctx]["curvature"]) else float("nan"),
                    "curvature_std": float(agg[ctx]["curvature"].std()) if len(agg[ctx]["curvature"]) else float("nan"),
                    "chord_mean": float(agg[ctx]["chord"].mean()) if len(agg[ctx]["chord"]) else float("nan"),
                    "chord_median": float(np.median(agg[ctx]["chord"])) if len(agg[ctx]["chord"]) else float("nan"),
                    "n_patches": int(len(agg[ctx]["curvature"])),
                }
                for ctx, _ in CONTEXTS
            },
        }

    out_json = RESULTS / "l63_patch_ood_sp65_sp82.json"
    out_json.write_text(json.dumps(big_summary, indent=2))
    print(f"\n[diagnostic] summary saved → {out_json}")

    # Print quick verdict table
    print("\n[verdict] curvature mean & W1-to-clean per context, per scenario:")
    print(f"  {'scenario':10s}  {'context':8s}  {'curv μ':>8s}  {'W1(curv)':>9s}  {'low-curv %':>10s}")
    for sc_name, summ in big_summary.items():
        for ctx in [c[0] for c in CONTEXTS]:
            cm = summ["metrics_per_context"][ctx]["curvature_mean"]
            w1c = summ["wasserstein_to_clean"][ctx]["W1_curvature"]
            lcv = summ["low_curvature_fraction"]["by_context"][ctx]
            print(f"  {sc_name:10s}  {ctx:8s}  {cm:8.4f}  {w1c:9.4f}  {100*lcv:9.1f}%")


if __name__ == "__main__":
    main()
