#!/usr/bin/env python3
"""
运行RDE-GPR 100次试验
"""
import os
import subprocess
import json
import time
from tqdm import tqdm

IMPUTE_META = "./save/weather_history_imputed_random_ratio0.1_split0.5_seed42_20260328_155218/impute_meta.json"
GROUND_PATH = "./data/weather/weather_ground.npy"
L = 7
S = 100
TRAINLENGTH = 36
N_JOBS = 2
N_TRIALS = 100

def get_split_point(meta_path):
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return meta.get('split_point', 0)

def get_total_len(meta_path):
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return meta.get('total_len', 0)

def run_trial(seed):
    out_dir = f"./save/weather_rdegpr_L{L}_trial{seed:03d}"
    os.makedirs(out_dir, exist_ok=True)

    split_point = get_split_point(IMPUTE_META)
    total_len = get_total_len(IMPUTE_META)

    cmd = [
        "python", "rde_gpr/weather_CSDIimpute_after-RDEgpr.py",
        "--imputed_history_path", f"./save/weather_history_imputed_random_ratio0.1_split0.5_seed42_20260328_155218/history_imputed.npy",
        "--impute_meta_path", IMPUTE_META,
        "--ground_path", GROUND_PATH,
        "--future_truth_path", "",
        "--split_ratio", "0.5",
        "--total_len", str(total_len),
        "--hist_split_point", str(split_point),
        "--horizon_steps", "24",
        "--history_timesteps", "72",
        "--target_indices", "",
        "--L", str(L),
        "--s", str(S),
        "--trainlength", str(TRAINLENGTH),
        "--n_jobs", str(N_JOBS),
        "--seed", str(seed),
        "--out_dir", out_dir
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, out_dir

def main():
    print(f"RDE-GPR {N_TRIALS} Trials (L={L}, s={S})")
    print("=" * 60)

    results = []
    for i in tqdm(range(N_TRIALS), desc="RDE-GPR trials"):
        seed = i + 1
        success, out_dir = run_trial(seed)
        if success:
            metrics_path = os.path.join(out_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                rmse = metrics.get('metrics', {}).get('rmse', None)
                mae = metrics.get('metrics', {}).get('mae', None)
                results.append({'seed': seed, 'rmse': rmse, 'mae': mae, 'dir': out_dir})

    print(f"\nCompleted {len(results)}/{N_TRIALS} trials successfully")

    if results:
        rmses = [r['rmse'] for r in results if r['rmse'] is not None]
        maes = [r['mae'] for r in results if r['mae'] is not None]

        print("\n" + "=" * 60)
        print("RDE-GPR Results Summary")
        print("=" * 60)
        print(f"L={L}, s={S}, trainlength={TRAINLENGTH}")
        print(f"Mean RMSE: {sum(rmses)/len(rmses):.4f} +/- { (sum((r-sum(rmses)/len(rmses))**2 for r in rmses)/len(rmses))**0.5:.4f}")
        print(f"Mean MAE:  {sum(maes)/len(maes):.4f} +/- { (sum((r-sum(maes)/len(maes))**2 for r in maes)/len(maes))**0.5:.4f}")
        print(f"Min RMSE:  {min(rmses):.4f}")
        print(f"Max RMSE:  {max(rmses):.4f}")

        summary = {
            'L': L, 's': S, 'trainlength': TRAINLENGTH,
            'n_trials': len(results),
            'mean_rmse': sum(rmses)/len(rmses),
            'std_rmse': (sum((r-sum(rmses)/len(rmses))**2 for r in rmses)/len(rmses))**0.5,
            'mean_mae': sum(maes)/len(maes),
            'std_mae': (sum((r-sum(maes)/len(maes))**2 for r in maes)/len(maes))**0.5,
            'min_rmse': min(rmses),
            'max_rmse': max(rmses),
            'trials': results
        }
        with open(f"./save/rdegpr_L{L}_100trials_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: ./save/rdegpr_L{L}_100trials_summary.json")

if __name__ == "__main__":
    main()