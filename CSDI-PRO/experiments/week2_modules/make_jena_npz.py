"""Convert Jena Climate 2009–2016 CSV → length-128 windowed npz suitable for
SAITS pretraining (P2.1 real-sensor case study).

Source: https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip
14 numeric features sampled every 10 minutes from the Max Planck Institute
weather station in Jena, Germany.

Output an npz with:
  clean        : (N, T=128, D=14) float32, train-set z-scored per feature.
  raw          : (N, T, D)        float32, hourly resampled but unscaled.
  feature_mean : (D,) train-set means (for unscaling).
  feature_std  : (D,) train-set stds.
  attractor_std: scalar = mean(feature_std) — used by the existing SAITS
                 training runner for normalized-MAE logging only; not a
                 physical attractor radius.
  split        : (N,)   int8 — 0=train, 1=val, 2=test.

Default split: train 2009–2014, val 2015, test 2016. Resample 10-min → 1-hour
mean. Stride 8 windows (75 % overlap) for training-window count.

Run:
  python -u -m experiments.week2_modules.make_jena_npz \
      --src experiments/week2_modules/data/real/jena_climate_2009_2016.csv \
      --out experiments/week2_modules/data/real/jena_clean_hourly_L128.npz
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Jena Climate CSV path")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--stride", type=int, default=8,
                    help="window stride in hourly samples; 8 = 75% overlap at L=128")
    ap.add_argument("--resample_factor", type=int, default=6,
                    help="6 = 10-min → hourly")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[jena] reading {src} ({src.stat().st_size/1e6:.1f} MB)")
    # Use numpy.genfromtxt — robust to mixed-delim CSVs.
    # The first column is "Date Time" string; we parse year only for splitting.
    import csv
    rows = []
    with open(src, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            rows.append(row)
    n_raw = len(rows)
    print(f"[jena] header: {header}")
    print(f"[jena] raw rows: {n_raw}  ({n_raw / (6*24*365):.2f} years)")

    # Date format: "DD.MM.YYYY HH:MM:SS"
    def year_of(date_str: str) -> int:
        return int(date_str.split(".")[2].split(" ")[0])

    years = np.array([year_of(r[0]) for r in rows], dtype=np.int16)
    feats = np.array([[float(x) for x in r[1:]] for r in rows], dtype=np.float32)
    print(f"[jena] feature matrix shape: {feats.shape}")

    # Resample 10-min → hourly mean.
    rf = args.resample_factor
    n_hour = (feats.shape[0] // rf) * rf
    feats_h = feats[:n_hour].reshape(-1, rf, feats.shape[1]).mean(axis=1)
    years_h = years[:n_hour:rf]
    print(f"[jena] hourly shape: {feats_h.shape}")

    # Train/val/test split by year.
    train_mask = years_h <= 2014
    val_mask = years_h == 2015
    test_mask = years_h == 2016
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    print(f"[jena] train hours: {train_idx.size}  val: {val_idx.size}  test: {test_idx.size}")

    # Train-set z-score.
    train_feats = feats_h[train_idx]
    feat_mean = train_feats.mean(axis=0).astype(np.float32)
    feat_std = train_feats.std(axis=0).astype(np.float32) + 1e-6
    feats_z = (feats_h - feat_mean) / feat_std

    # Window builder: stride = args.stride, length = args.seq_len.
    def build_windows(idx_block, label):
        if idx_block.size == 0:
            return np.zeros((0, args.seq_len, feats_h.shape[1]), dtype=np.float32)
        # Require contiguous block. Jena is contiguous within year; we just
        # take the first run of the index block (years are sorted).
        starts = np.arange(0, idx_block.size - args.seq_len + 1, args.stride)
        windows = np.stack([feats_z[idx_block[s:s + args.seq_len]] for s in starts])
        return windows.astype(np.float32)

    train_W = build_windows(train_idx, "train")
    val_W = build_windows(val_idx, "val")
    test_W = build_windows(test_idx, "test")
    print(f"[jena] windows: train={train_W.shape[0]}  val={val_W.shape[0]}  test={test_W.shape[0]}")

    clean = np.concatenate([train_W, val_W, test_W], axis=0)
    split = np.concatenate([
        np.full(train_W.shape[0], 0, dtype=np.int8),
        np.full(val_W.shape[0], 1, dtype=np.int8),
        np.full(test_W.shape[0], 2, dtype=np.int8),
    ])
    raw_unscaled = clean * feat_std + feat_mean

    attractor_std = float(np.mean(feat_std))
    print(f"[jena] surrogate attractor_std (mean per-feature train std) = {attractor_std:.4f}")

    # Also save the contiguous hourly stream for the test year (2016) so the
    # eval script can pull arbitrary 576-hour chunks without window-stitching.
    test_hourly_z = feats_z[test_idx]  # (8709, 14), z-scored
    print(f"[jena] test hourly stream (z-scored): {test_hourly_z.shape}")

    print(f"[jena] writing {out}")
    np.savez(out,
             clean=clean,
             raw=raw_unscaled.astype(np.float32),
             feature_mean=feat_mean,
             feature_std=feat_std,
             attractor_std=np.array(attractor_std, dtype=np.float32),
             split=split,
             test_hourly_z=test_hourly_z.astype(np.float32))
    print(f"[jena] done. clean shape {clean.shape}; size {out.stat().st_size/1e6:.1f} MB.")


if __name__ == "__main__":
    main()
