"""
EEG Missing Data Generator - Simple version
"""
import os
import sys
import numpy as np
import pandas as pd
import json
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def generate_random_missing(data, missing_ratio, seed=42):
    np.random.seed(seed)
    observed = data.copy()
    masks = np.ones_like(data, dtype=float)

    n_timesteps, n_features = data.shape
    n_total = n_timesteps * n_features
    n_missing = int(n_total * missing_ratio)

    all_indices = np.arange(n_total)
    np.random.shuffle(all_indices)
    missing_indices = all_indices[:n_missing]

    rows = missing_indices // n_features
    cols = missing_indices % n_features

    observed[rows, cols] = -9999.0
    masks[rows, cols] = 0

    return observed, masks

def main():
    parser = argparse.ArgumentParser(description="EEG Missing Data Generator")
    parser.add_argument("--data_path", type=str, default="./data/data_extra/Dataset_3-EEG.xlsx")
    parser.add_argument("--out_dir", type=str, default="./data/eeg")
    parser.add_argument("--missing_ratios", type=str, default="0.1,0.3,0.5")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.out_dir + "/images", exist_ok=True)

    print("=" * 60)
    print("EEG Missing Data Generator")
    print("=" * 60)

    df = pd.read_excel(args.data_path, header=None)
    data = df.values.astype(np.float32)
    print(f"Original data shape: {data.shape}")
    print(f"Original data range: [{data.min():.2f}, {data.max():.2f}]")

    np.random.seed(args.seed)

    missing_ratios = [float(x) for x in args.missing_ratios.split(",")]

    for ratio in missing_ratios:
        observed, masks = generate_random_missing(data, ratio, args.seed)
        actual_ratio = 1 - masks.mean()
        print(f"Missing ratio={ratio:.1f}: actual={actual_ratio:.4f}")

        base_name = f"eeg_random_ratio{ratio}_seed{args.seed}"
        np.save(os.path.join(args.out_dir, f"eeg_ground.npy"), data)
        np.save(os.path.join(args.out_dir, f"{base_name}.npy"), observed)
        np.save(os.path.join(args.out_dir, f"{base_name}_mask.npy"), masks)

    print(f"\nSaved to {args.out_dir}/")

if __name__ == "__main__":
    main()