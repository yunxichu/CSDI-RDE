# -*- coding: utf-8 -*-
"""
Post-process for PM2.5:
- Load outputs from pm25_RDEnew.py
- Generate plots and save locally
- Compute and save metrics (forecast + optional imputation)

Default outputs:
  out_dir/
    plots/
      forecast_dim0.png, forecast_dim1.png, ...
      residual_dim0.png, ...
      residual_hist.png
    forecast_metrics_overall.json
    forecast_metrics_per_dim.csv
    forecast_residual_summary.csv
    (optional) impute_metrics_overall.json
    (optional) impute_metrics_per_dim.csv
    (optional) std_nan_report.json
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Utils
# -----------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_rmse_mae(y_true: np.ndarray, y_pred: np.ndarray):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan}
    diff = y_true[mask] - y_pred[mask]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    return {"rmse": rmse, "mae": mae}


def summarize_residuals(residual: np.ndarray):
    # residual: (T, D)
    res = residual[np.isfinite(residual)]
    if res.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "p05": np.nan,
            "p50": np.nan,
            "p95": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(res.size),
        "mean": float(np.mean(res)),
        "std": float(np.std(res)),
        "p05": float(np.quantile(res, 0.05)),
        "p50": float(np.quantile(res, 0.50)),
        "p95": float(np.quantile(res, 0.95)),
        "min": float(np.min(res)),
        "max": float(np.max(res)),
    }


def load_args_if_exists(out_dir: str):
    p = os.path.join(out_dir, "args.json")
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return None


def load_pm25_files(ground_path: str, missing_path: str):
    df_full = pd.read_csv(ground_path, index_col="datetime", parse_dates=True).sort_index()
    df_miss = pd.read_csv(missing_path, index_col="datetime", parse_dates=True).sort_index()
    if not df_full.index.equals(df_miss.index):
        raise ValueError("ground 与 missing 的 datetime 索引不一致。")
    if df_full.shape[1] != df_miss.shape[1]:
        raise ValueError("ground 与 missing 的列数不一致。")
    return df_full, df_miss


def time_split_df(df_full: pd.DataFrame, df_miss: pd.DataFrame, split_ratio: float):
    total_len = len(df_full)
    split_point = int(total_len * split_ratio)
    hist_full = df_full.iloc[:split_point].copy()
    hist_miss = df_miss.iloc[:split_point].copy()
    fut_full = df_full.iloc[split_point:].copy()
    fut_miss = df_miss.iloc[split_point:].copy()
    return hist_full, hist_miss, fut_full, fut_miss


# -----------------------------
# Plotting
# -----------------------------
def plot_forecast_one_dim(
    out_path: str,
    time_index,
    y_true_1d: np.ndarray,
    y_pred_1d: np.ndarray,
    y_std_1d: np.ndarray | None = None,
    title: str = "",
):
    plt.figure(figsize=(14, 5))
    plt.plot(time_index, y_true_1d, label="True", linewidth=1.5)
    plt.plot(time_index, y_pred_1d, label="Pred", linewidth=1.5)

    if y_std_1d is not None:
        # ±2 std band (ignore non-finite)
        std = np.array(y_std_1d, dtype=float)
        std[~np.isfinite(std)] = np.nan
        upper = y_pred_1d + 2.0 * std
        lower = y_pred_1d - 2.0 * std
        plt.fill_between(time_index, lower, upper, alpha=0.2, label="±2 std")

    plt.title(title if title else "Forecast (True vs Pred)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_residual_one_dim(out_path: str, time_index, residual_1d: np.ndarray, title: str = ""):
    plt.figure(figsize=(14, 4))
    plt.plot(time_index, residual_1d, linewidth=1.2)
    plt.title(title if title else "Residual (True - Pred)")
    plt.xlabel("Time")
    plt.ylabel("Residual")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_residual_hist(out_path: str, residual_all: np.ndarray, title: str = ""):
    r = residual_all[np.isfinite(residual_all)]
    plt.figure(figsize=(8, 5))
    plt.hist(r, bins=80)
    plt.title(title if title else "Residual Histogram")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="PM2.5 postprocess: plots + metrics")
    parser.add_argument("--out_dir", type=str, required=True, help="Output dir produced by pm25_RDEnew.py")
    parser.add_argument("--ground_path", type=str, default="./data/pm25/Code/STMVL/SampleData/pm25_ground.txt")
    parser.add_argument("--missing_path", type=str, default="./data/pm25/Code/STMVL/SampleData/pm25_missing.txt")
    parser.add_argument("--split_ratio", type=float, default=None, help="If None, try read from out_dir/args.json")
    parser.add_argument("--dims", type=str, default="0,1,2", help="Dims to plot, e.g. '0,1,2'")
    parser.add_argument("--plot_max_points", type=int, default=2000, help="Downsample for plotting speed (0=disable)")

    args = parser.parse_args()
    out_dir = args.out_dir

    # Resolve split_ratio
    run_args = load_args_if_exists(out_dir)
    split_ratio = args.split_ratio
    if split_ratio is None:
        if run_args and "split_ratio" in run_args:
            split_ratio = float(run_args["split_ratio"])
        else:
            raise ValueError("split_ratio not provided and args.json not found in out_dir.")
    dims = [int(x) for x in args.dims.split(",") if x.strip() != ""]

    # Load original data for truth / missing pattern
    df_full, df_miss = load_pm25_files(args.ground_path, args.missing_path)
    hist_full, hist_miss, fut_full, _ = time_split_df(df_full, df_miss, split_ratio)

    # Load predictions
    pred_path = os.path.join(out_dir, "future_pred.csv")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"missing {pred_path}")
    df_pred = pd.read_csv(pred_path, index_col="datetime", parse_dates=True).sort_index()

    std_path = os.path.join(out_dir, "future_pred_std.csv")
    df_std = None
    if os.path.exists(std_path):
        df_std = pd.read_csv(std_path, index_col="datetime", parse_dates=True).sort_index()

    # Align indices (important)
    df_true = fut_full.copy()
    df_true = df_true.loc[df_pred.index]

    if df_std is not None:
        df_std = df_std.loc[df_pred.index]

    y_true = df_true.values.astype(np.float64)
    y_pred = df_pred.values.astype(np.float64)

    residual = y_true - y_pred  # (T,D)

    # Output folders
    plot_dir = os.path.join(out_dir, "plots")
    ensure_dir(plot_dir)

    # Optional downsample for plots
    time_index = df_pred.index
    if args.plot_max_points and len(time_index) > args.plot_max_points:
        idx = np.linspace(0, len(time_index) - 1, args.plot_max_points).astype(int)
        time_plot = time_index[idx]
        y_true_plot = y_true[idx]
        y_pred_plot = y_pred[idx]
        if df_std is not None:
            y_std_plot = df_std.values.astype(np.float64)[idx]
        else:
            y_std_plot = None
    else:
        time_plot = time_index
        y_true_plot = y_true
        y_pred_plot = y_pred
        y_std_plot = df_std.values.astype(np.float64) if df_std is not None else None

    # ---- Forecast plots per selected dim
    for d in dims:
        if d < 0 or d >= y_true.shape[1]:
            print(f"[skip] dim {d} out of range")
            continue

        std_1d = (y_std_plot[:, d] if y_std_plot is not None else None)
        plot_forecast_one_dim(
            out_path=os.path.join(plot_dir, f"forecast_dim{d}.png"),
            time_index=time_plot,
            y_true_1d=y_true_plot[:, d],
            y_pred_1d=y_pred_plot[:, d],
            y_std_1d=std_1d,
            title=f"Forecast dim {d} (True vs Pred)",
        )

        plot_residual_one_dim(
            out_path=os.path.join(plot_dir, f"residual_dim{d}.png"),
            time_index=time_plot,
            residual_1d=(y_true_plot[:, d] - y_pred_plot[:, d]),
            title=f"Residual dim {d} (True - Pred)",
        )

    # Residual histogram (all dims)
    plot_residual_hist(
        out_path=os.path.join(plot_dir, "residual_hist.png"),
        residual_all=residual,
        title="Residual Histogram (All dims)",
    )

    # ---- Forecast metrics: overall + per-dim
    overall = compute_rmse_mae(y_true, y_pred)
    with open(os.path.join(out_dir, "forecast_metrics_overall.json"), "w") as f:
        json.dump(overall, f, indent=4, ensure_ascii=False)

    per_dim = []
    for j, col in enumerate(df_pred.columns):
        m = compute_rmse_mae(y_true[:, j], y_pred[:, j])
        per_dim.append({"dim": j, "name": str(col), "rmse": m["rmse"], "mae": m["mae"]})
    pd.DataFrame(per_dim).to_csv(os.path.join(out_dir, "forecast_metrics_per_dim.csv"), index=False)

    # Residual summary
    res_sum = summarize_residuals(residual)
    pd.DataFrame([res_sum]).to_csv(os.path.join(out_dir, "forecast_residual_summary.csv"), index=False)

    # ---- Optional: report NaN/Inf in std
    if df_std is not None:
        std_vals = df_std.values.astype(np.float64)
        report = {
            "std_total": int(std_vals.size),
            "std_nan": int(np.isnan(std_vals).sum()),
            "std_inf": int(np.isinf(std_vals).sum()),
            "std_nonfinite_ratio": float((~np.isfinite(std_vals)).mean()),
        }
        with open(os.path.join(out_dir, "std_nan_report.json"), "w") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

    # ---- Optional: imputation metrics on first-half missing positions (if history_imputed.csv exists)
    hist_imputed_path = os.path.join(out_dir, "history_imputed.csv")
    if os.path.exists(hist_imputed_path):
        df_hist_imp = pd.read_csv(hist_imputed_path, index_col="datetime", parse_dates=True).sort_index()
        df_hist_imp = df_hist_imp.loc[hist_full.index]  # align

        # Only evaluate on positions that were NaN in hist_miss (i.e., truly missing in missing-version)
        miss_mask = np.isnan(hist_miss.values.astype(np.float64))
        y_imp = df_hist_imp.values.astype(np.float64)
        y_hist_true = hist_full.values.astype(np.float64)

        # overall imputation metrics on missing positions
        # flatten masked values
        true_flat = y_hist_true[miss_mask]
        imp_flat = y_imp[miss_mask]
        imp_overall = compute_rmse_mae(true_flat, imp_flat)
        with open(os.path.join(out_dir, "impute_metrics_overall.json"), "w") as f:
            json.dump(imp_overall, f, indent=4, ensure_ascii=False)

        # per-dim imputation metrics on missing positions
        per_dim_imp = []
        for j, col in enumerate(hist_full.columns):
            msk = miss_mask[:, j]
            m = compute_rmse_mae(y_hist_true[msk, j], y_imp[msk, j])
            per_dim_imp.append({"dim": j, "name": str(col), "rmse": m["rmse"], "mae": m["mae"], "missing_count": int(msk.sum())})
        pd.DataFrame(per_dim_imp).to_csv(os.path.join(out_dir, "impute_metrics_per_dim.csv"), index=False)

    print(f"[done] plots saved to: {plot_dir}")
    print(f"[done] metrics saved to: {out_dir}")


if __name__ == "__main__":
    main()
