# -*- coding: utf-8 -*-
"""
Compute relative error for RDE-GPR PM2.5 outputs.

========
使用方法
python pm25_calc_relative_error.py \
  --pred_csv ./save/pm25_rdegpr_debug_split0.5_seed42_20260129_131341/future_pred.csv \
  --ground_path ./data/pm25/Code/STMVL/SampleData/pm25_ground.txt \
  --split_ratio 0.5

========

Inputs:
  - --pred_csv: path to future_pred.csv (with datetime index)
  - --ground_path: path to pm25_ground.txt (csv with datetime index)
  - --split_ratio: same split_ratio used in prediction script
Outputs (saved under --out_dir, default: same folder as pred_csv):
  - relative_error_ape.csv        (point-wise APE per dim)
  - relative_error_sape.csv       (point-wise sAPE per dim)
  - relative_error_summary.json   (overall + per-dim summary)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd


def safe_json_dump(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def time_split_df(df_full: pd.DataFrame, split_ratio: float):
    total_len = len(df_full)
    split_point = int(total_len * float(split_ratio))
    hist = df_full.iloc[:split_point].copy()
    fut = df_full.iloc[split_point:].copy()
    meta = {
        "total_len": total_len,
        "split_ratio": float(split_ratio),
        "split_point": split_point,
        "hist_len": len(hist),
        "fut_len": len(fut),
        "hist_start": str(hist.index.min()),
        "hist_end": str(hist.index.max()),
        "fut_start": str(fut.index.min()) if len(fut) else None,
        "fut_end": str(fut.index.max()) if len(fut) else None,
    }
    return hist, fut, meta


def summarize_err(err_df: pd.DataFrame):
    # err_df: shape [T, D] with NaNs allowed
    arr = err_df.to_numpy(dtype=float)
    flat = arr[np.isfinite(arr)]
    out = {}

    def stats(x):
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {"count": 0, "mean": np.nan, "median": np.nan, "p90": np.nan, "max": np.nan}
        return {
            "count": int(x.size),
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "p90": float(np.quantile(x, 0.9)),
            "max": float(np.max(x)),
        }

    out["overall"] = stats(flat)

    per_dim = {}
    for col in err_df.columns:
        per_dim[str(col)] = stats(err_df[col].to_numpy(dtype=float))
    out["per_dim"] = per_dim
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", type=str, required=True, help="Path to future_pred.csv")
    ap.add_argument("--ground_path", type=str, required=True, help="Path to pm25_ground.txt (csv with datetime index)")
    ap.add_argument("--split_ratio", type=float, default=0.5, help="Same split_ratio used in prediction")
    ap.add_argument("--eps", type=float, default=1e-6, help="Epsilon to avoid divide-by-zero")
    ap.add_argument("--out_dir", type=str, default="", help="Output directory (default: folder of pred_csv)")
    args = ap.parse_args()

    pred_csv = args.pred_csv
    out_dir = args.out_dir.strip() or os.path.dirname(os.path.abspath(pred_csv))
    os.makedirs(out_dir, exist_ok=True)

    # load prediction
    df_pred = pd.read_csv(pred_csv, index_col="datetime", parse_dates=True).sort_index()

    # load ground and split
    df_full = pd.read_csv(args.ground_path, index_col="datetime", parse_dates=True).sort_index()
    _, fut_full, meta = time_split_df(df_full, args.split_ratio)

    # align true future to prediction index
    # 1) keep only predicted horizon
    fut_true = fut_full.loc[df_pred.index.intersection(fut_full.index)].copy()

    # 2) ensure columns match
    missing_cols = [c for c in df_pred.columns if c not in fut_true.columns]
    if missing_cols:
        raise ValueError(f"Ground truth missing columns: {missing_cols}")

    fut_true = fut_true[df_pred.columns]

    # 3) reindex to pred index (keep NaN if any timestamps missing)
    fut_true = fut_true.reindex(df_pred.index)

    y_true = fut_true.to_numpy(dtype=float)
    y_pred = df_pred.to_numpy(dtype=float)

    eps = float(args.eps)

    # APE: |e| / (|y| + eps)
    ape = np.abs(y_pred - y_true) / (np.abs(y_true) + eps)

    # sAPE: 2|e| / (|y|+|yhat|+eps)
    sape = 2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)

    df_ape = pd.DataFrame(ape, index=df_pred.index, columns=df_pred.columns)
    df_sape = pd.DataFrame(sape, index=df_pred.index, columns=df_pred.columns)

    ape_path = os.path.join(out_dir, "relative_error_ape.csv")
    sape_path = os.path.join(out_dir, "relative_error_sape.csv")
    df_ape.to_csv(ape_path)
    df_sape.to_csv(sape_path)

    summary = {
        "meta": meta,
        "pred_csv": os.path.abspath(pred_csv),
        "ground_path": os.path.abspath(args.ground_path),
        "aligned_rows": int(df_pred.shape[0]),
        "nan_true_count": int(np.isnan(y_true).sum()),
        "nan_pred_count": int(np.isnan(y_pred).sum()),
        "ape": summarize_err(df_ape),
        "sape": summarize_err(df_sape),
    }
    summary_path = os.path.join(out_dir, "relative_error_summary.json")
    safe_json_dump(summary, summary_path)

    print("Saved:")
    print(" ", ape_path)
    print(" ", sape_path)
    print(" ", summary_path)
    print("\nOverall APE mean:", summary["ape"]["overall"]["mean"])
    print("Overall sAPE mean:", summary["sape"]["overall"]["mean"])


if __name__ == "__main__":
    main()
