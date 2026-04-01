#!/usr/bin/env python3
"""
RDE-GPR参数搜索脚本
测试不同L和s的组合
"""
import os
import subprocess
import json
import time
from tqdm import tqdm

IMPUTE_META = "./save/weather_history_imputed_random_ratio0.1_split0.5_seed42_20260328_155218/impute_meta.json"
GROUND_PATH = "./data/weather/weather_ground.npy"
TRAINLENGTH = 36
N_JOBS = 2
SEED = 42

L_VALUES = [5, 7, 10, 12, 15]
S_VALUES = [20, 30, 50]

def get_split_point(meta_path):
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return meta.get('split_point', 0)

def get_total_len(meta_path):
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return meta.get('total_len', 0)

def run_trial(L, s, seed):
    out_dir = f"./save/weather_rdegpr_L{L}_s{s}"
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
        "--s", str(s),
        "--trainlength", str(TRAINLENGTH),
        "--n_jobs", str(N_JOBS),
        "--seed", str(seed),
        "--out_dir", out_dir
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, out_dir

def main():
    print("RDE-GPR 参数搜索")
    print("=" * 60)
    print(f"L values: {L_VALUES}")
    print(f"s values: {S_VALUES}")
    print("=" * 60)

    results = []
    total_trials = len(L_VALUES) * len(S_VALUES)

    for L in tqdm(L_VALUES, desc="L values"):
        for s in tqdm(S_VALUES, desc=f"  s values for L={L}"):
            print(f"\n--- Testing L={L}, s={s} ---")
            start_time = time.time()

            success, out_dir = run_trial(L, s, SEED)

            elapsed = time.time() - start_time

            if success:
                metrics_path = os.path.join(out_dir, "metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    rmse = metrics.get('overall', {}).get('rmse', None)
                    mae = metrics.get('overall', {}).get('mae', None)
                    results.append({
                        'L': L, 's': s, 'rmse': rmse, 'mae': mae,
                        'elapsed': elapsed, 'dir': out_dir
                    })
                    print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f} ({elapsed/60:.1f}min)")
                else:
                    print(f"  Failed: metrics.json not found")
                    results.append({'L': L, 's': s, 'rmse': None, 'mae': None, 'elapsed': elapsed})
            else:
                print(f"  Failed: return code != 0")
                results.append({'L': L, 's': s, 'rmse': None, 'mae': None, 'elapsed': elapsed})

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    valid_results = [r for r in results if r['rmse'] is not None]
    valid_results.sort(key=lambda x: x['rmse'])

    print(f"\n{'L':>4} {'s':>4} {'RMSE':>10} {'MAE':>10} {'Time':>10}")
    print("-" * 40)
    for r in valid_results:
        print(f"{r['L']:>4} {r['s']:>4} {r['rmse']:>10.4f} {r['mae']:>10.4f} {r['elapsed']/60:>8.1f}min")

    if valid_results:
        best = valid_results[0]
        print(f"\n*** Best: L={best['L']}, s={best['s']}, RMSE={best['rmse']:.4f} ***")

        summary = {
            'L_values': L_VALUES,
            's_values': S_VALUES,
            'results': results,
            'best': best
        }
        with open("./save/rdegpr_param_search_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: ./save/rdegpr_param_search_summary.json")

if __name__ == "__main__":
    main()