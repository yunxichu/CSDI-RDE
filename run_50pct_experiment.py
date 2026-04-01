#!/usr/bin/env python3
"""
Weather 50% Missing Rate - 方法对比实验
1. CSDI-RDE: 先用CSDI补值，再用RDE-GPR预测
2. 基线方法（不补值）: GRU, LSTM, 直接用含缺失值的数据预测
"""
import os
import numpy as np
import pandas as pd
import json
import subprocess
import sys
from tqdm import tqdm

MISSING_RATIO = 0.5
MISSING_MODE = "random"
SEED = 42
SPLIT_RATIO = 0.5

DATA_DIR = "./data/weather"
SAVE_DIR = "./save"
CSDI_RUN_FOLDER = f"{SAVE_DIR}/weather_csdi_ratio{MISSING_RATIO}_fold0"
IMPUTED_DIR = f"{SAVE_DIR}/weather_history_imputed_{MISSING_MODE}_ratio{MISSING_RATIO}_split{SPLIT_RATIO}_seed{SEED}_20260330"

def step1_generate_missing():
    print("=" * 60)
    print("Step 1: 生成50%缺失率数据")
    print("=" * 60)

    if not os.path.exists(f"{DATA_DIR}/weather_missing_{MISSING_MODE}_ratio{MISSING_RATIO}_seed{SEED}.npy"):
        cmd = [
            "python", "imputation/weather_generate_missing.py",
            "--missing_ratios", str(MISSING_RATIO),
            "--modes", MISSING_MODE,
            "--seed", str(SEED),
            "--out_dir", DATA_DIR,
            "--skip_visualization"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"生成缺失数据失败: {result.stderr[-500:]}")
            return False
        print("缺失数据生成完成")
    else:
        print(f"缺失数据已存在: {DATA_DIR}/weather_missing_{MISSING_MODE}_ratio{MISSING_RATIO}_seed{SEED}.npy")

    return True

def get_latest_csdi_folder():
    import glob
    pattern = f"{SAVE_DIR}/weather_{MISSING_MODE}_ratio{MISSING_RATIO}_fold0_*/"
    folders = glob.glob(pattern)
    if folders:
        return sorted(folders)[-1]
    return None

def step2_train_csdi():
    print("\n" + "=" * 60)
    print("Step 2: 训练CSDI模型")
    print("=" * 60)

    csdi_folder = get_latest_csdi_folder()

    if csdi_folder and os.path.exists(f"{csdi_folder}/model.pth"):
        print(f"CSDI模型已存在: {csdi_folder}/model.pth")
        return csdi_folder

    cmd = [
        "python", "csdi/weather_train.py",
        "--missing_ratio", str(MISSING_RATIO),
        "--missing_mode", MISSING_MODE,
        "--epochs", "20",
        "--seed", str(SEED),
        "--device", "cpu"
    ]
    print(f"训练CSDI: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"CSDI训练失败: {result.stderr[-1000:]}")
        return None

    csdi_folder = get_latest_csdi_folder()
    if csdi_folder:
        print(f"CSDI训练完成: {csdi_folder}")
    return csdi_folder

def step3_impute(csdi_folder):
    print("\n" + "=" * 60)
    print("Step 3: CSDI补值")
    print("=" * 60)

    ensure_dir(IMPUTED_DIR)

    if not os.path.exists(f"{IMPUTED_DIR}/history_imputed.npy"):
        cmd = [
            "python", "imputation/weather_CSDIimpute.py",
            "--run_folder", csdi_folder,
            "--missing_ratio", str(MISSING_RATIO),
            "--missing_mode", MISSING_MODE,
            "--seed", str(SEED),
            "--split_ratio", str(SPLIT_RATIO),
            "--device", "cpu",
            "--out_dir", IMPUTED_DIR
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"CSDI补值失败: {result.stderr[-500:]}")
            return False
        print(f"CSDI补值完成: {IMPUTED_DIR}")
    else:
        print(f"CSDI补值数据已存在: {IMPUTED_DIR}/history_imputed.npy")

    return True

def step4_rdegpr_forecast():
    print("\n" + "=" * 60)
    print("Step 4: RDE-GPR预测")
    print("=" * 60)

    out_dir = f"{SAVE_DIR}/weather_rdegpr_ratio{MISSING_RATIO}_forecast"

    cmd = [
        "python", "rde_gpr/weather_CSDIimpute_after-RDEgpr.py",
        "--imputed_history_path", f"{IMPUTED_DIR}/history_imputed.npy",
        "--impute_meta_path", f"{IMPUTED_DIR}/impute_meta.json",
        "--ground_path", f"{DATA_DIR}/weather_ground.npy",
        "--L", "10",
        "--s", "50",
        "--trainlength", "24",
        "--seed", str(SEED),
        "--out_dir", out_dir
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"RDE-GPR失败: {result.stderr[-500:]}")
        return None

    metrics_path = f"{out_dir}/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def step5_gru_with_impute():
    print("\n" + "=" * 60)
    print("Step 5: GRU预测（用补值数据）")
    print("=" * 60)

    out_dir = f"{SAVE_DIR}/weather_gru_ratio{MISSING_RATIO}_forecast"

    cmd = [
        "python", "baselines/weather_simple_rnn_forecast.py",
        "--imputed_history_path", f"{IMPUTED_DIR}/history_imputed.npy",
        "--impute_meta_path", f"{IMPUTED_DIR}/impute_meta.json",
        "--ground_path", f"{DATA_DIR}/weather_ground.npy",
        "--horizon_steps", "24",
        "--history_timesteps", "72",
        "--hidden_size", "64",
        "--num_layers", "2",
        "--epochs", "50",
        "--batch_size", "64",
        "--lr", "1e-3",
        "--seed", str(SEED),
        "--model", "gru",
        "--train_window", "48",
        "--out_dir", out_dir
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"GRU失败: {result.stderr[-500:]}")
        return None

    metrics_path = f"{out_dir}/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def step6_gru_no_impute():
    print("\n" + "=" * 60)
    print("Step 6: GRU预测（不补值，直接用含缺失数据）")
    print("=" * 60)

    missing_data_path = f"{DATA_DIR}/weather_missing_{MISSING_MODE}_ratio{MISSING_RATIO}_seed{SEED}.npy"
    out_dir = f"{SAVE_DIR}/weather_gru_noimpute_ratio{MISSING_RATIO}_forecast"

    cmd = [
        "python", "baselines/weather_simple_rnn_forecast.py",
        "--imputed_history_path", missing_data_path,
        "--impute_meta_path", f"{IMPUTED_DIR}/impute_meta.json",
        "--ground_path", f"{DATA_DIR}/weather_ground.npy",
        "--horizon_steps", "24",
        "--history_timesteps", "72",
        "--hidden_size", "64",
        "--num_layers", "2",
        "--epochs", "50",
        "--batch_size", "64",
        "--lr", "1e-3",
        "--seed", str(SEED),
        "--model", "gru",
        "--train_window", "48",
        "--out_dir", out_dir,
        "--no_impute"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"不补值GRU失败: {result.stderr[-500:]}")
        return None

    metrics_path = f"{out_dir}/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main():
    print("Weather 50% Missing Rate - 方法对比实验")
    print("=" * 60)
    print(f"Missing ratio: {MISSING_RATIO}")
    print(f"Missing mode: {MISSING_MODE}")
    print(f"Seed: {SEED}")

    results = {}

    if not step1_generate_missing():
        return

    csdi_folder = step2_train_csdi()
    if not csdi_folder:
        print("CSDI训练失败")
        return

    if not step3_impute(csdi_folder):
        return

    rdegpr_metrics = step4_rdegpr_forecast()
    if rdegpr_metrics:
        results['CSDI-RDE'] = rdegpr_metrics.get('overall', {})

    gru_impute_metrics = step5_gru_with_impute()
    if gru_impute_metrics:
        results['GRU (with impute)'] = gru_impute_metrics.get('overall', {})

    gru_no_impute_metrics = step6_gru_no_impute()
    if gru_no_impute_metrics:
        results['GRU (no impute)'] = gru_no_impute_metrics.get('overall', {})

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    for method, metrics in results.items():
        rmse = metrics.get('rmse', 'N/A')
        mae = metrics.get('mae', 'N/A')
        print(f"{method:25s}: RMSE={rmse:.4f}, MAE={mae:.4f}" if isinstance(rmse, float) else f"{method:25s}: {metrics}")

    summary = {
        'missing_ratio': MISSING_RATIO,
        'missing_mode': MISSING_MODE,
        'seed': SEED,
        'results': results
    }
    summary_path = f"{SAVE_DIR}/weather_50pct_comparison_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n保存摘要: {summary_path}")

if __name__ == "__main__":
    main()