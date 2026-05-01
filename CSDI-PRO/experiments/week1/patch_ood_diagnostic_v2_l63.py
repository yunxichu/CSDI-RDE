"""Patch-OOD diagnostic v2 for L63 under the Figure-1 protocol.

This supersedes the earlier raw-patch diagnostics that used
``compute_attractor_std()`` and scenario-independent corruption seeds.  Those
settings do not match the L63 v2 frontier runner.  Here we use the same
calibration as ``phase_transition_grid_l63_v2.py``:

  * ``LORENZ63_ATTRACTOR_STD = 8.51``
  * corruption seed ``1000 * seed + 5000 + grid_index``
  * ``dt = 0.025`` and ``n_ctx = 512``

We measure local raw-patch statistics that can be compared against the v2
Panda representation-space diagnostic:

  1. Per-patch local-stdev distribution (4-step sliding window inside each
     16-step patch), reported as both per-patch median and per-patch fraction
     of low-variance windows.
  2. Per-patch lag-1 autocorrelation.
  3. Per-patch spectral power in the mid frequency band [1/16, 1/4] · Nyquist.

Plus a side-by-side raw trajectory overlay on a single seed window.

Output:
  results/l63_patch_ood_v2_v2protocol_sp65_sp82_10seed.json
  figures/l63_patch_ood_v2_v2protocol_traj_overlay_SP65.png
  figures/l63_patch_ood_v2_v2protocol_traj_overlay_SP82.png
  figures/l63_patch_ood_v2_v2protocol_metrics_SP65.png
  figures/l63_patch_ood_v2_v2protocol_metrics_SP82.png

CPU-only except CSDI imputation (1 GPU). Run:
  CUDA_VISIBLE_DEVICES=1 python -u -m experiments.week1.patch_ood_diagnostic_v2_l63
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

from experiments.week1.corruption import make_corrupted_observations
from experiments.week1.lorenz63_utils import (
    integrate_lorenz63, LORENZ63_ATTRACTOR_STD,
)
from methods.dynamics_impute import impute

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "experiments" / "week1" / "results"
FIGS = REPO / "experiments" / "week1" / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

CSDI_CKPT_L63 = REPO / "experiments/week2_modules/ckpts/dyn_csdi_full_v6_center_ep20.pt"

PATCH_LEN = 16
LOCAL_WIN = 4    # sliding window for local stdev
LOW_VAR_TAU_FRAC = 0.10  # fraction of clean median local stdev that counts as "low-var"
DT = 0.025
N_CTX = 512
LYAP_L63 = 0.906
GRID_INDEX = {"SP65": 4, "SP82": 6}

SCENARIOS = [
    {"name": "SP65", "sparsity": 0.65, "noise_std_frac": 0.0},
    {"name": "SP82", "sparsity": 0.82, "noise_std_frac": 0.0},
]
SEEDS = tuple(range(10))
CONTEXTS = [
    ("clean",  "C0"),
    ("linear", "C1"),
    ("csdi",   "C2"),
]


# ----------------------------- patch metrics ---------------------------------

def make_patches(traj: np.ndarray, patch_len: int = PATCH_LEN,
                  stride: int = 1) -> np.ndarray:
    T, D = traj.shape
    N = max(0, (T - patch_len) // stride + 1)
    out = np.stack([traj[i * stride : i * stride + patch_len]
                    for i in range(N)], axis=0)
    return out.astype(np.float32)


def local_stdev_per_patch(patches: np.ndarray, win: int = LOCAL_WIN) -> dict:
    """Per-patch: aggregate stdev across W-step sliding windows.

    Returns:
      median_stdev: [N] median local stdev per patch
      low_var_fraction: [N] fraction of windows with stdev < tau (set later)
      raw: [N, n_windows] all local stdev values per patch (for tau-aware threshold)
    """
    N, L, D = patches.shape
    n_win = L - win + 1
    raw = np.empty((N, n_win), dtype=np.float32)
    for s in range(n_win):
        w = patches[:, s:s+win, :]  # [N, win, D]
        # Stdev across `win` axis, then norm across D
        sd = w.std(axis=1)  # [N, D]
        raw[:, s] = np.linalg.norm(sd, axis=1) / np.sqrt(D)  # mean per-dim stdev
    return {
        "raw": raw,
        "median_stdev": np.median(raw, axis=1),
    }


def lag1_autocorr_per_patch(patches: np.ndarray) -> np.ndarray:
    """Per-patch lag-1 Pearson autocorrelation, averaged across dims.
    Linear interpolation in gaps → ρ₁ → 1; chaotic dynamics → ρ₁ < 1."""
    N, L, D = patches.shape
    rho = np.zeros(N, dtype=np.float32)
    for d in range(D):
        x = patches[:, :, d]
        x_centered = x - x.mean(axis=1, keepdims=True)
        num = (x_centered[:, :-1] * x_centered[:, 1:]).sum(axis=1)
        den = np.sqrt((x_centered[:, :-1] ** 2).sum(axis=1) *
                      (x_centered[:, 1:] ** 2).sum(axis=1) + 1e-12)
        rho += num / den
    return rho / D


def midfreq_power_per_patch(patches: np.ndarray) -> np.ndarray:
    """Per-patch mid-frequency power (band [1/L, 1/4] · Nyquist), normalized
    by total power. Linear interpolation has near-zero mid-freq content in
    its long flat segments. Average across channels."""
    N, L, D = patches.shape
    out = np.zeros(N, dtype=np.float32)
    # FFT bins 1..L/2; "mid" defined as bins 2 to L/4 (exclude DC and very low freq)
    lo = 2
    hi = max(lo + 1, L // 4)
    for d in range(D):
        x = patches[:, :, d] - patches[:, :, d].mean(axis=1, keepdims=True)
        spec = np.abs(np.fft.rfft(x, axis=1)) ** 2  # [N, L//2 + 1]
        total = spec[:, 1:].sum(axis=1) + 1e-12
        midband = spec[:, lo:hi].sum(axis=1)
        out += midband / total
    return out / D


# ----------------------------- main pipeline ---------------------------------

def run_one_seed(seed: int, sc: dict, attr_std: float):
    traj = integrate_lorenz63(N_CTX, dt=DT, spinup=2000, seed=seed)
    corrupt_seed = 1000 * seed + 5000 + GRID_INDEX[sc["name"]]
    obs_res = make_corrupted_observations(
        traj.astype(np.float32), mask_regime="iid_time",
        sparsity=sc["sparsity"], noise_std_frac=sc["noise_std_frac"],
        attractor_std=attr_std, seed=corrupt_seed, dt=DT, lyap=LYAP_L63,
        patch_length=PATCH_LEN,
    )
    observed = obs_res.observed
    linear = impute(observed, kind="linear").astype(np.float32)
    csdi = impute(
        observed, kind="csdi",
        sigma_override=float(sc["noise_std_frac"]) * attr_std,
    ).astype(np.float32)

    trajs = {"clean": traj.astype(np.float32), "linear": linear, "csdi": csdi}
    patches = {n: make_patches(t) for n, t in trajs.items()}
    metrics = {}
    for n, P in patches.items():
        ls = local_stdev_per_patch(P)
        metrics[n] = {
            "local_stdev_raw": ls["raw"],          # [N, n_windows]
            "local_stdev_median": ls["median_stdev"],
            "lag1_rho": lag1_autocorr_per_patch(P),
            "midfreq_power": midfreq_power_per_patch(P),
        }
    return trajs, patches, metrics, obs_res


def aggregate_seeds(per_seed: list[tuple]) -> dict:
    out = {}
    for ctx, _ in CONTEXTS:
        out[ctx] = {
            "local_stdev_raw": np.concatenate([m[ctx]["local_stdev_raw"]
                                                for _, _, m, _ in per_seed], axis=0),
            "local_stdev_median": np.concatenate([m[ctx]["local_stdev_median"]
                                                    for _, _, m, _ in per_seed]),
            "lag1_rho": np.concatenate([m[ctx]["lag1_rho"]
                                         for _, _, m, _ in per_seed]),
            "midfreq_power": np.concatenate([m[ctx]["midfreq_power"]
                                              for _, _, m, _ in per_seed]),
        }
    return out


def w1_to_clean(agg: dict, key: str) -> dict:
    ref = agg["clean"][key]
    return {ctx: float(wasserstein_distance(ref, agg[ctx][key]))
            for ctx, _ in CONTEXTS}


def low_var_window_fraction(agg: dict) -> dict:
    """Fraction of all sliding windows (across all patches × seeds) whose
    local stdev is below tau_lv = LOW_VAR_TAU_FRAC × median(clean local stdev).
    Linear-fill should be high; clean and CSDI low."""
    ref_med = float(np.median(agg["clean"]["local_stdev_raw"]))
    tau = LOW_VAR_TAU_FRAC * ref_med
    return {
        "tau_lv": tau,
        "by_context": {
            ctx: float((agg[ctx]["local_stdev_raw"] < tau).mean())
            for ctx, _ in CONTEXTS
        },
    }


# ----------------------------- plotting --------------------------------------

def plot_trajectory_overlay(scenario_name: str, trajs_seed0: dict, mask_seed0,
                              out_png: Path, dim: int = 0,
                              window: tuple[int, int] = (200, 280)):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    t0, t1 = window
    xs = np.arange(t0, t1)
    ax.plot(xs, trajs_seed0["clean"][t0:t1, dim], color="C0",
             linewidth=2.5, alpha=0.85, label="clean")
    ax.plot(xs, trajs_seed0["linear"][t0:t1, dim], color="C1",
             linewidth=1.4, alpha=0.95, linestyle="--", label="linear-fill")
    ax.plot(xs, trajs_seed0["csdi"][t0:t1, dim], color="C2",
             linewidth=1.4, alpha=0.95, linestyle=":", label="CSDI-fill")
    # Mark missing timesteps
    if mask_seed0 is not None:
        miss = np.where(~mask_seed0[t0:t1, dim])[0]
        if len(miss):
            ymin = min(trajs_seed0["clean"][t0:t1, dim].min(),
                       trajs_seed0["linear"][t0:t1, dim].min())
            ymax = max(trajs_seed0["clean"][t0:t1, dim].max(),
                       trajs_seed0["linear"][t0:t1, dim].max())
            for i in miss:
                ax.axvspan(t0 + i - 0.5, t0 + i + 0.5,
                           color="grey", alpha=0.07, linewidth=0)
            ax.text(0.99, 0.02, "grey: missing timesteps",
                     transform=ax.transAxes, ha="right", va="bottom",
                     fontsize=8, color="grey")
    ax.set_title(f"{scenario_name} — raw trajectory overlay (seed 0, dim {dim})")
    ax.set_xlabel("time step")
    ax.set_ylabel(f"x_{dim}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"[saved] {out_png}")


def plot_metrics(scenario_name: str, agg: dict, w1_lv, w1_rho, w1_mf, lvw,
                  out_png: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1. Local stdev distribution
    ax = axes[0]
    for ctx, color in CONTEXTS:
        v = agg[ctx]["local_stdev_raw"].ravel()
        # log-scale handles dynamic range; clip 0 to small
        v_pos = np.clip(v, 1e-4, None)
        ax.hist(np.log10(v_pos), bins=60, density=True, alpha=0.45,
                 color=color, label=ctx)
    ax.axvline(np.log10(max(lvw["tau_lv"], 1e-4)), color="k", linestyle=":",
                label=f"τ_lv = 10% × clean median")
    ax.set_xlabel("log₁₀(local stdev, 4-step window)")
    ax.set_ylabel("density")
    ax.set_title(f"{scenario_name} — local stdev (W₁: lin={w1_lv['linear']:.3f}, "
                 f"csdi={w1_lv['csdi']:.3f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 2. lag-1 autocorrelation
    ax = axes[1]
    for ctx, color in CONTEXTS:
        v = agg[ctx]["lag1_rho"]
        ax.hist(v, bins=60, density=True, alpha=0.45, color=color, label=ctx)
    ax.set_xlabel("per-patch lag-1 ρ")
    ax.set_ylabel("density")
    ax.set_title(f"{scenario_name} — lag-1 ρ (W₁: lin={w1_rho['linear']:.4f}, "
                 f"csdi={w1_rho['csdi']:.4f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 3. mid-frequency power
    ax = axes[2]
    for ctx, color in CONTEXTS:
        v = agg[ctx]["midfreq_power"]
        ax.hist(v, bins=60, density=True, alpha=0.45, color=color, label=ctx)
    ax.set_xlabel("mid-frequency power fraction")
    ax.set_ylabel("density")
    ax.set_title(f"{scenario_name} — mid-freq power "
                 f"(W₁: lin={w1_mf['linear']:.4f}, csdi={w1_mf['csdi']:.4f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"{scenario_name} — patch-OOD v2 metrics  "
                 f"(low-var-window frac: clean={100*lvw['by_context']['clean']:.1f}%, "
                 f"lin={100*lvw['by_context']['linear']:.1f}%, "
                 f"csdi={100*lvw['by_context']['csdi']:.1f}%)",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"[saved] {out_png}")


def main():
    from methods.csdi_impute_adapter import set_csdi_checkpoint, set_csdi_attractor_std
    set_csdi_checkpoint(str(CSDI_CKPT_L63))
    attr_std = float(LORENZ63_ATTRACTOR_STD)
    set_csdi_attractor_std(attr_std)
    print(f"[v2] CSDI ckpt: {CSDI_CKPT_L63}")
    print(f"[v2] attractor_std: {attr_std:.4f}")
    print(f"[v2] protocol: dt={DT} n_ctx={N_CTX} seeds={SEEDS[0]}..{SEEDS[-1]} "
          f"corrupt_seed=1000*seed+5000+grid_index")

    big_summary = {}
    for sc in SCENARIOS:
        print(f"\n=== {sc['name']}  s={sc['sparsity']}  σ={sc['noise_std_frac']} ===")
        per_seed = []
        for seed in SEEDS:
            trajs, patches, metrics, obs_res = run_one_seed(seed, sc, attr_std)
            per_seed.append((trajs, patches, metrics, obs_res))
            print(f"  seed={seed}  keep_frac={obs_res.metadata['keep_frac']:.3f}")
            for ctx, _ in CONTEXTS:
                m = metrics[ctx]
                print(f"    {ctx:8s}  local_stdev_med μ={np.median(m['local_stdev_median']):.4f}  "
                      f"ρ₁ μ={m['lag1_rho'].mean():.4f}  "
                      f"midfreq μ={m['midfreq_power'].mean():.4f}")

        agg = aggregate_seeds(per_seed)
        w1_lv = w1_to_clean(agg, "local_stdev_median")
        w1_rho = w1_to_clean(agg, "lag1_rho")
        w1_mf = w1_to_clean(agg, "midfreq_power")
        lvw = low_var_window_fraction(agg)

        # Trajectory overlay (seed 0 only, time window where there are gaps)
        trajs_s0, _, _, obs_res_s0 = per_seed[0]
        plot_trajectory_overlay(
            sc["name"], trajs_s0, obs_res_s0.mask,
            FIGS / f"l63_patch_ood_v2_v2protocol_traj_overlay_{sc['name']}.png",
        )
        plot_metrics(sc["name"], agg, w1_lv, w1_rho, w1_mf, lvw,
                      FIGS / f"l63_patch_ood_v2_v2protocol_metrics_{sc['name']}.png")

        big_summary[sc["name"]] = {
            "config": sc,
            "n_seeds": len(SEEDS),
            "dt": DT,
            "n_ctx": N_CTX,
            "attractor_std": attr_std,
            "seed_protocol": "1000 * seed + 5000 + grid_index",
            "grid_index": GRID_INDEX[sc["name"]],
            "patch_length": PATCH_LEN,
            "local_window": LOCAL_WIN,
            "low_var_window_fraction": lvw,
            "wasserstein_local_stdev_to_clean": w1_lv,
            "wasserstein_lag1_to_clean": w1_rho,
            "wasserstein_midfreq_to_clean": w1_mf,
            "metrics_summary": {
                ctx: {
                    "local_stdev_median_overall": float(np.median(agg[ctx]["local_stdev_median"])),
                    "lag1_rho_mean": float(agg[ctx]["lag1_rho"].mean()),
                    "midfreq_power_mean": float(agg[ctx]["midfreq_power"].mean()),
                    "n_patches": int(len(agg[ctx]["lag1_rho"])),
                }
                for ctx, _ in CONTEXTS
            },
        }

    out_json = RESULTS / "l63_patch_ood_v2_v2protocol_sp65_sp82_10seed.json"
    out_json.write_text(json.dumps(big_summary, indent=2))
    print(f"\n[v2] summary saved → {out_json}")

    print("\n[verdict] sharper W₁ contrasts (linear vs csdi vs clean):")
    print(f"  {'scen':6s}  {'metric':14s}  {'W₁(lin)':>10s}  {'W₁(csdi)':>10s}  "
          f"{'lin/csdi':>10s}")
    for sc_name, summ in big_summary.items():
        for key, label in [("wasserstein_local_stdev_to_clean", "local stdev"),
                            ("wasserstein_lag1_to_clean", "lag-1 ρ"),
                            ("wasserstein_midfreq_to_clean", "mid-freq pwr")]:
            wl = summ[key]["linear"]
            wc = summ[key]["csdi"]
            ratio = wl / max(wc, 1e-9)
            print(f"  {sc_name:6s}  {label:14s}  {wl:10.4f}  {wc:10.4f}  "
                  f"{ratio:10.2f}×")


if __name__ == "__main__":
    main()
