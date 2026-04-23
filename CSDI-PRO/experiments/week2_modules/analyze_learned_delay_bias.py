"""A4 lightweight: extract learned delay_bias from full_v6_center checkpoint
and quantify its "effective τ" — test the "training absorbs τ" hypothesis
that explains the §5.X1 τ-coupling null result.

Strategy (no new GPU training needed):
  1. Load full_v6_center_ep20.pt
  2. Extract delay_bias matrix (shape [L_diffusion_seq, L_diffusion_seq] = [128, 128])
  3. Aggregate along anti-diagonals to get attention-by-offset profile: A(k) = mean of bias[i, i-k]
  4. Extract peaks of A(k) → the "effective τ" the model attends to
  5. Load τ-coupling JSON → extract τ_B values M2 selected on test trajectories
  6. Compare: does learned-τ align with M2-selected τ?

Outcomes to interpret:
  - If learned-τ peaks ≈ median τ_B  → training absorbed τ (hypothesis confirmed)
  - If learned-τ peaks differ         → training learned a different/universal pattern
  - Either way: positive evidence for §5.X1 interpretation

Usage:
    python -m experiments.week2_modules.analyze_learned_delay_bias \\
        --ckpt experiments/week2_modules/ckpts/dyn_csdi_full_v6_center_ep20.pt \\
        --tau_coupling_json experiments/week2_modules/results/tau_coupling_S3_n3_v1.json \\
        --out_fig experiments/week2_modules/figures/learned_delay_bias.png
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


def extract_delay_bias(ckpt_path: str) -> tuple[np.ndarray, float]:
    """Load checkpoint and return (delay_bias matrix [L,L], delay_alpha scalar)."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # wrapped format: {cfg: ..., state: OrderedDict} OR {model_state: ...} OR raw state_dict
    state = sd.get("state", sd.get("model_state", sd.get("state_dict", sd)))
    bias_key = next(k for k in state.keys() if k.endswith("delay_bias"))
    alpha_key = next(k for k in state.keys() if k.endswith("delay_alpha"))
    bias = state[bias_key].cpu().numpy()
    alpha = float(state[alpha_key].cpu().numpy().item())
    return bias, alpha


def antidiagonal_profile(bias: np.ndarray, max_offset: int = 30) -> np.ndarray:
    """Aggregate bias[i, j] by offset k = i - j (anti-diagonal).

    Returns A[k] = mean of bias[i, i - k] for k in [-max_offset, max_offset].
    Symmetric; k=0 is the main diagonal.
    """
    L = bias.shape[0]
    offsets = np.arange(-max_offset, max_offset + 1)
    profile = np.zeros_like(offsets, dtype=np.float64)
    for idx, k in enumerate(offsets):
        # i - j = k → j = i - k, valid when 0 <= i - k < L
        # take mean over all valid (i, j) pairs
        diag = np.diagonal(bias, offset=-k)  # offset arg: j - i; we want i - j = k → offset=-k
        profile[idx] = diag.mean() if len(diag) > 0 else 0.0
    return offsets, profile


def extract_peaks(offsets: np.ndarray, profile: np.ndarray, n_peaks: int = 4,
                   min_offset: int = 1, max_offset: int = 30) -> list[int]:
    """Find n_peaks largest bias offsets in positive range, ignoring |k| < min_offset."""
    # Restrict to positive offsets above min_offset
    mask = (offsets >= min_offset) & (offsets <= max_offset)
    pos_off = offsets[mask]
    pos_prof = profile[mask]
    # Take top n_peaks indices by value
    top_idx = np.argsort(pos_prof)[::-1][:n_peaks]
    peaks = sorted([int(pos_off[i]) for i in top_idx])
    return peaks


def collect_M2_tau(tau_coupling_json: str) -> list[list[int]]:
    """Load the τ-coupling JSON and extract the tau_B values (M2's selection on each seed)."""
    data = json.loads(Path(tau_coupling_json).read_text())
    tau_Bs = []
    for r in data["records"]:
        if r.get("mode") == "default" and "tau_B" in r:
            tau_Bs.append(r["tau_B"])
    return tau_Bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tau_coupling_json", default=None,
                    help="Optional: τ-coupling JSON to cross-reference M2-selected τ")
    ap.add_argument("--out_fig", default=None)
    ap.add_argument("--max_offset", type=int, default=30)
    ap.add_argument("--n_peaks", type=int, default=4)
    args = ap.parse_args()

    # 1. Extract
    bias, alpha = extract_delay_bias(args.ckpt)
    L = bias.shape[0]
    print(f"=== learned delay_bias analysis ===")
    print(f"  ckpt: {args.ckpt}")
    print(f"  shape: {bias.shape}  dtype: {bias.dtype}")
    print(f"  delay_alpha (post-training): {alpha:.4f}  (init was 0.01)")
    print(f"  bias abs mean: {np.abs(bias).mean():.4f}  max: {bias.max():.4f}  min: {bias.min():.4f}")

    # 2. Antidiagonal profile
    offsets, profile = antidiagonal_profile(bias, max_offset=args.max_offset)
    print(f"\n  antidiagonal profile (offset → mean bias):")
    # Print only interesting range
    for k, p in zip(offsets, profile):
        if abs(k) <= 15 or abs(p) > profile.std():
            marker = " <<" if abs(p) > profile.std() else ""
            print(f"    k={k:+3d}  bias_mean={p:+.4f}{marker}")

    # 3. Peaks
    peaks = extract_peaks(offsets, profile, n_peaks=args.n_peaks,
                           min_offset=1, max_offset=args.max_offset)
    print(f"\n  Top-{args.n_peaks} offsets by learned bias (effective τ): {peaks}")

    # 4. Compare to M2-selected τ_B
    if args.tau_coupling_json:
        tau_Bs = collect_M2_tau(args.tau_coupling_json)
        print(f"\n  M2-selected τ_B across {len(tau_Bs)} default-mode seeds:")
        for t in tau_Bs:
            print(f"    τ_B = {t}")
        all_tau = [v for t in tau_Bs for v in t]
        tau_counter = Counter(all_tau)
        median_tau = sorted(all_tau)[len(all_tau) // 2] if all_tau else None
        print(f"  τ_B distribution: {dict(tau_counter.most_common())}")
        print(f"  median τ_B: {median_tau}")
        # Overlap
        peaks_set = set(peaks)
        m2_set = set(all_tau)
        overlap = peaks_set & m2_set
        print(f"\n  Overlap between learned-effective-τ and M2 τ_B: {sorted(overlap)} "
              f"({len(overlap)}/{len(peaks)} peaks match)")

    # 5. Figure
    if args.out_fig:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        # (a) heatmap
        im = axes[0].imshow(bias, cmap="RdBu_r", vmin=-np.abs(bias).max(), vmax=np.abs(bias).max(),
                            aspect="auto")
        axes[0].set_title(f"learned delay_bias (L={L}×{L})\nα={alpha:.3f}, |bias|_mean={np.abs(bias).mean():.3f}")
        axes[0].set_xlabel("j (key position)")
        axes[0].set_ylabel("i (query position)")
        plt.colorbar(im, ax=axes[0], shrink=0.8)
        # (b) antidiagonal profile
        axes[1].plot(offsets, profile, "-o", ms=3)
        axes[1].axhline(0, color="gray", lw=0.5)
        for p in peaks:
            axes[1].axvline(p, color="red", lw=0.8, alpha=0.5, ls="--",
                            label=f"peak at τ={p}" if p == peaks[0] else None)
        axes[1].set_xlabel("offset k = i − j")
        axes[1].set_ylabel("mean bias")
        axes[1].set_title(f"antidiagonal profile\neffective τ ≈ {peaks}")
        axes[1].legend(loc="upper right")
        axes[1].grid(alpha=0.3)
        out_path = Path(args.out_fig)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\n  [saved fig] {out_path}")

    # 6. Save summary JSON
    summary_json = Path(args.ckpt).parent.parent / "results" / "learned_delay_bias_analysis.json"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary = dict(
        ckpt=args.ckpt,
        delay_alpha=alpha,
        bias_stats=dict(abs_mean=float(np.abs(bias).mean()),
                        max=float(bias.max()), min=float(bias.min())),
        effective_tau_peaks=peaks,
        antidiagonal_profile={int(k): float(p) for k, p in zip(offsets, profile)},
    )
    if args.tau_coupling_json:
        summary["m2_tau_B_samples"] = tau_Bs
        summary["m2_tau_B_counter"] = dict(tau_counter.most_common())
        summary["overlap"] = sorted(overlap)
    summary_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  [saved summary] {summary_json}")


if __name__ == "__main__":
    main()
