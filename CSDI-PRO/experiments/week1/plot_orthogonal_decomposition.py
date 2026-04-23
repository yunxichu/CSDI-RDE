"""A6: Generate §5.X3 figure — 2D (s, σ) NRMSE heatmap for Ours and Panda.

Reads:
  - ssgrid_v1_ours_csdi_seeds{0..4}.json (5 separate files from GPU split)
  - ssgrid_v1_panda.json (single file, all 5 seeds)

Produces:
  - Figure X3a: 2 heatmap panels (Ours | Panda) over (s, σ) grid with NRMSE contours
  - Figure X3b: Panda/Ours ratio heatmap with argmax annotation
  - Figure X3c (inset): failure frontier directions — NRMSE(s, σ=0) and NRMSE(s=0, σ)
  - Summary table (printed + saved to ssgrid_summary.json): means/stds for each (s, σ) cell
  - Power-law fits for Prop 5: (α_s, α_σ, α_s', α_σ') estimates
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_all_records() -> list[dict]:
    """Load all ssgrid result JSONs and merge records."""
    all_recs = []
    # 5 ours files
    for s in range(5):
        fp = RESULTS_DIR / f"ssgrid_v1_ours_csdi_seeds{s}.json"
        if not fp.exists():
            print(f"WARNING: missing {fp}")
            continue
        data = json.loads(fp.read_text())
        all_recs.extend(data["records"])
    # 1 panda file
    pp = RESULTS_DIR / "ssgrid_v1_panda.json"
    data = json.loads(pp.read_text())
    all_recs.extend(data["records"])
    return all_recs


def aggregate(records: list[dict], s_vals, sigma_vals) -> dict:
    """Aggregate NRMSE@h=1 by (method, s, sigma)."""
    by = defaultdict(list)
    for r in records:
        if "error" in r or r.get("metrics") is None:
            continue
        m = r.get("method")
        s = r.get("sparsity")
        sig = r.get("noise")
        h1 = r.get("metrics", {}).get(1) or r.get("metrics", {}).get("1")
        if h1 is None or h1.get("nrmse") is None:
            continue
        key = (m, round(float(s), 3), round(float(sig), 3))
        by[key].append(float(h1["nrmse"]))

    # Build 3x3 matrix for each method
    methods = sorted({m for (m, _, _) in by.keys()})
    grids = {}
    for m in methods:
        mat_mean = np.full((len(s_vals), len(sigma_vals)), np.nan)
        mat_std = np.full_like(mat_mean, np.nan)
        mat_n = np.zeros_like(mat_mean, dtype=int)
        for i, s in enumerate(s_vals):
            for j, sig in enumerate(sigma_vals):
                vals = by.get((m, round(float(s), 3), round(float(sig), 3)), [])
                if vals:
                    mat_mean[i, j] = np.mean(vals)
                    mat_std[i, j] = np.std(vals)
                    mat_n[i, j] = len(vals)
        grids[m] = dict(mean=mat_mean, std=mat_std, n=mat_n)
    return grids


def fit_powerlaw(s_vals, sigma_vals, nrmse_mat, dominant_axis: str):
    """Fit log(NRMSE) ≈ log c + α_primary log(primary) + α_secondary log(1+c'·secondary).

    Returns dict with fitted exponents.
    dominant_axis: 's' for Panda, 'sigma' for Ours.
    """
    from scipy.optimize import curve_fit

    S_mat, SIG_mat = np.meshgrid(s_vals, sigma_vals, indexing="ij")
    y = nrmse_mat.flatten()
    s_flat = S_mat.flatten()
    sig_flat = SIG_mat.flatten()

    # Filter out NaNs and any y <= 0
    good = (~np.isnan(y)) & (y > 0)
    y = y[good]
    s_flat = s_flat[good]
    sig_flat = sig_flat[good]
    log_y = np.log(y)

    if dominant_axis == "sigma":
        # Ours: log y ≈ log c_sigma + alpha_sigma * log(sigma_shifted) + alpha_s_prime * log(1 + c_s_prime * s)
        # To avoid log(0), use sigma + eps and s_shifted too. Fit on log scale.
        def model(X, log_c_sig, alpha_sig, alpha_s_prime):
            s, sig = X
            eps = 1e-3
            return log_c_sig + alpha_sig * np.log(sig + eps) + alpha_s_prime * np.log(1 + 2.0 * s)

        try:
            popt, pcov = curve_fit(model, (s_flat, sig_flat), log_y, p0=[0.0, 1.5, 0.3],
                                    maxfev=20000)
            log_c, alpha_sig, alpha_s_prime = popt
            return dict(alpha_primary=float(alpha_sig),      # σ
                        alpha_secondary=float(alpha_s_prime),# s (as saturating)
                        log_c=float(log_c),
                        ratio=float(alpha_sig / max(abs(alpha_s_prime), 1e-6)))
        except Exception as e:
            return dict(error=str(e))
    else:  # s-dominant (Panda)
        def model(X, log_c_s, alpha_s, alpha_sig_prime):
            s, sig = X
            eps = 1e-3
            return log_c_s + alpha_s * np.log(s + eps) + alpha_sig_prime * np.log(1 + sig)

        try:
            popt, pcov = curve_fit(model, (s_flat, sig_flat), log_y, p0=[0.0, 1.0, 0.3],
                                    maxfev=20000)
            log_c, alpha_s, alpha_sig_prime = popt
            return dict(alpha_primary=float(alpha_s),
                        alpha_secondary=float(alpha_sig_prime),
                        log_c=float(log_c),
                        ratio=float(alpha_s / max(abs(alpha_sig_prime), 1e-6)))
        except Exception as e:
            return dict(error=str(e))


def plot_figure(grids, s_vals, sigma_vals, fits, out_path: Path):
    methods = ["ours_csdi", "panda"]
    method_names = {"ours_csdi": "Ours (manifold)", "panda": "Panda (ambient)"}

    # panel layout: 2 heatmaps (ours, panda) + 1 ratio heatmap
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    vmin = min(np.nanmin(grids[m]["mean"]) for m in methods)
    vmax = max(np.nanmax(grids[m]["mean"]) for m in methods)

    for ax, m in zip(axes[:2], methods):
        mat = grids[m]["mean"]
        std = grids[m]["std"]
        im = ax.imshow(mat, cmap="YlOrRd", vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
        ax.set_xticks(range(len(sigma_vals)))
        ax.set_xticklabels([f"{v:.2f}" for v in sigma_vals])
        ax.set_yticks(range(len(s_vals)))
        ax.set_yticklabels([f"{v:.2f}" for v in s_vals])
        ax.set_xlabel("σ / σ_attr")
        ax.set_ylabel("s (sparsity)")
        # Annotate each cell with mean±std
        for i in range(len(s_vals)):
            for j in range(len(sigma_vals)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.2f}\n±{std[i,j]:.2f}",
                            ha="center", va="center",
                            color="white" if mat[i, j] > (vmin + vmax) / 2 else "black",
                            fontsize=8)
        fit = fits.get(m, {})
        alpha_prim = fit.get("alpha_primary", float("nan"))
        alpha_sec = fit.get("alpha_secondary", float("nan"))
        ratio = fit.get("ratio", float("nan"))
        ax.set_title(f"{method_names[m]}  NRMSE@h=1\n"
                     f"α_primary={alpha_prim:.2f}, α_secondary={alpha_sec:.2f}, ratio={ratio:.1f}×")
        plt.colorbar(im, ax=ax, shrink=0.85)

    # ratio panel
    ratio_mat = grids["panda"]["mean"] / grids["ours_csdi"]["mean"]
    ax = axes[2]
    im = ax.imshow(ratio_mat, cmap="viridis", origin="lower", aspect="auto")
    ax.set_xticks(range(len(sigma_vals)))
    ax.set_xticklabels([f"{v:.2f}" for v in sigma_vals])
    ax.set_yticks(range(len(s_vals)))
    ax.set_yticklabels([f"{v:.2f}" for v in s_vals])
    ax.set_xlabel("σ / σ_attr")
    ax.set_ylabel("s (sparsity)")
    for i in range(len(s_vals)):
        for j in range(len(sigma_vals)):
            r = ratio_mat[i, j]
            if not np.isnan(r):
                ax.text(j, i, f"{r:.2f}×", ha="center", va="center",
                        color="white" if r < np.nanmean(ratio_mat) else "black",
                        fontsize=9, weight="bold")
    # Highlight argmax
    if not np.all(np.isnan(ratio_mat)):
        imax, jmax = np.unravel_index(np.nanargmax(ratio_mat), ratio_mat.shape)
        ax.add_patch(plt.Rectangle((jmax - 0.45, imax - 0.45), 0.9, 0.9,
                                     fill=False, edgecolor="red", lw=2.5))
    ax.set_title(f"Panda / Ours ratio\n(max={np.nanmax(ratio_mat):.2f}× at highlighted cell)")
    plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle("§5.X3: (s, σ) 2D NRMSE grid — orthogonal failure channels", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")
    plt.close()


def print_tables(grids, s_vals, sigma_vals):
    methods = ["ours_csdi", "panda"]
    for m in methods:
        print(f"\n=== {m} NRMSE@h=1 (mean ± std over 5 seeds) ===")
        print(f"{'s \\ σ':<10s}", end="")
        for sig in sigma_vals:
            print(f"  σ={sig:<6.2f}", end="")
        print()
        for i, s in enumerate(s_vals):
            print(f"s={s:<8.2f}", end="")
            for j in range(len(sigma_vals)):
                mean = grids[m]["mean"][i, j]
                std = grids[m]["std"][i, j]
                if not np.isnan(mean):
                    print(f"  {mean:.3f}±{std:.3f}", end="")
                else:
                    print(f"  {'--':<12s}", end="")
            print()

    print(f"\n=== Panda / Ours ratio ===")
    ratio = grids["panda"]["mean"] / grids["ours_csdi"]["mean"]
    print(f"{'s \\ σ':<10s}", end="")
    for sig in sigma_vals:
        print(f"  σ={sig:<6.2f}", end="")
    print()
    for i, s in enumerate(s_vals):
        print(f"s={s:<8.2f}", end="")
        for j in range(len(sigma_vals)):
            r = ratio[i, j]
            if not np.isnan(r):
                print(f"  {r:.2f}×       ", end="")
            else:
                print(f"  {'--':<12s}", end="")
        print()


def main():
    s_vals = [0.0, 0.35, 0.70]
    sigma_vals = [0.0, 0.50, 1.53]

    records = load_all_records()
    print(f"loaded {len(records)} total records")

    grids = aggregate(records, s_vals, sigma_vals)
    print(f"methods found: {list(grids.keys())}")

    # Power-law fits (Prop 5 validation)
    fits = {}
    fits["ours_csdi"] = fit_powerlaw(s_vals, sigma_vals, grids["ours_csdi"]["mean"], "sigma")
    fits["panda"] = fit_powerlaw(s_vals, sigma_vals, grids["panda"]["mean"], "s")
    print(f"\n=== Power-law fits (Prop 5) ===")
    for m, f in fits.items():
        print(f"  {m}: {f}")

    print_tables(grids, s_vals, sigma_vals)

    # Save summary
    summary_json = RESULTS_DIR / "ssgrid_summary.json"
    summary = dict(
        s_values=s_vals, sigma_values=sigma_vals,
        methods={m: dict(mean=grids[m]["mean"].tolist(),
                          std=grids[m]["std"].tolist(),
                          n=grids[m]["n"].tolist())
                  for m in grids},
        ratio=(grids["panda"]["mean"] / grids["ours_csdi"]["mean"]).tolist(),
        powerlaw_fits=fits,
    )
    summary_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[saved summary] {summary_json}")

    # Figure
    fig_path = FIG_DIR / "ssgrid_orthogonal_decomposition.png"
    plot_figure(grids, s_vals, sigma_vals, fits, fig_path)


if __name__ == "__main__":
    main()
