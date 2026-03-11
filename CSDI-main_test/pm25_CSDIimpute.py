# -*- coding: utf-8 -*-
"""
PM2.5 历史段一次性补值脚本：
- 前 split_ratio 部分：用 CSDI_PM25 对 missing 进行补值
- 输出 history_imputed.csv（后续预测统一使用它）

运行示例：
python pm25_CSDIimpute.py \
  --run_folder ./save/pm25_imputation_split0.5_train0.7_valid0.15_mask0.1_seed42_20260127_151505 \
  --ground_path ./data/pm25/Code/STMVL/SampleData/pm25_ground.txt \
  --missing_path ./data/pm25/Code/STMVL/SampleData/pm25_missing.txt \
  --meanstd_path ./data/pm25/pm25_meanstd.pk \
  --split_ratio 0.5 \
  --device cuda:0 \
  --seed 42 \
  --impute_n_samples 50 \
  --chunk_len 36 --stride 36
"""

import os
import json
import random
import argparse
import datetime
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from main_model import CSDI_PM25


def set_global_seed(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


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
    split_point = int(total_len * float(split_ratio))

    hist_full = df_full.iloc[:split_point].copy()
    hist_miss = df_miss.iloc[:split_point].copy()

    fut_full = df_full.iloc[split_point:].copy()
    fut_miss = df_miss.iloc[split_point:].copy()

    meta = {
        "total_len": total_len,
        "split_ratio": float(split_ratio),
        "split_point": split_point,
        "hist_len": len(hist_full),
        "fut_len": len(fut_full),
        "hist_start": str(hist_full.index.min()),
        "hist_end": str(hist_full.index.max()),
        "fut_start": str(fut_full.index.min()) if len(fut_full) else None,
        "fut_end": str(fut_full.index.max()) if len(fut_full) else None,
    }
    return hist_full, hist_miss, fut_full, fut_miss, meta


def load_csdi_pm25(model_path: str, config_json_path: str, device: str):
    with open(config_json_path, "r") as f:
        full_cfg = json.load(f)
    config = full_cfg["model_config"]

    model = CSDI_PM25(config, device).to(device)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model, config, full_cfg


def csdi_impute_chunk_pm25(
    model,
    chunk_values_LK: np.ndarray,   # (L,K) with NaN
    mean_K: np.ndarray,            # (K,)
    std_K: np.ndarray,             # (K,)
    device: str,
    n_samples: int,
    seed: int,
):
    L, K = chunk_values_LK.shape
    cond_mask_LK = (~np.isnan(chunk_values_LK)).astype(np.float32)

    x0_LK = np.nan_to_num(chunk_values_LK, nan=0.0).astype(np.float32)
    x_norm_LK = ((x0_LK - mean_K) / std_K) * cond_mask_LK  # (L,K)

    observed_data = torch.from_numpy(x_norm_LK).unsqueeze(0).to(device)   # (1,L,K)
    observed_data = observed_data.permute(0, 2, 1).contiguous()           # (1,K,L)

    cond_mask = torch.from_numpy(cond_mask_LK).unsqueeze(0).to(device)    # (1,L,K)
    cond_mask = cond_mask.permute(0, 2, 1).contiguous()                   # (1,K,L)

    observed_tp = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)  # (1,L)

    torch.manual_seed(int(seed))

    with torch.no_grad():
        side_info = model.get_side_info(observed_tp, cond_mask)
        samples = model.impute(observed_data, cond_mask, side_info, n_samples)  # (1,n,K,L)

    samples_nLK = samples[0].permute(0, 2, 1).contiguous().cpu().numpy()  # (n,L,K)
    pred_norm_LK = samples_nLK.mean(axis=0)  # (L,K)
    pred_LK = pred_norm_LK * std_K + mean_K

    out = chunk_values_LK.copy()
    miss = np.isnan(out)
    out[miss] = pred_LK[miss]
    return out


def csdi_impute_history_long(
    model,
    df_hist_missing: pd.DataFrame,
    mean_K: np.ndarray,
    std_K: np.ndarray,
    device: str,
    n_samples: int,
    chunk_len: int,
    stride: int,
    seed: int,
):
    values = df_hist_missing.values.astype(np.float32)  # (T,K)
    T, K = values.shape
    out = values.copy()
    nan_mask = np.isnan(values)

    for start in tqdm(range(0, T, int(stride)), desc="CSDI imputing history"):
        end = min(start + int(chunk_len), T)
        chunk = out[start:end].copy()

        if not np.isnan(chunk).any():
            continue

        chunk_seed = int(seed + start)
        filled = csdi_impute_chunk_pm25(
            model=model,
            chunk_values_LK=chunk,
            mean_K=mean_K,
            std_K=std_K,
            device=device,
            n_samples=int(n_samples),
            seed=chunk_seed,
        )
        out[start:end] = filled

    out[~nan_mask] = values[~nan_mask]  # 再保险
    return pd.DataFrame(out, index=df_hist_missing.index, columns=df_hist_missing.columns)


def main():
    parser = argparse.ArgumentParser(description="PM2.5 History Imputation Once (CSDI_PM25)")

    parser.add_argument("--run_folder", type=str, required=True,
                        help="训练脚本生成的文件夹（里面应有 model.pth / config.json）")
    parser.add_argument("--model_path", type=str, default="", help="可选：显式指定 model.pth")
    parser.add_argument("--config_json", type=str, default="", help="可选：显式指定 config.json")

    parser.add_argument("--ground_path", type=str, required=True)
    parser.add_argument("--missing_path", type=str, required=True)
    parser.add_argument("--meanstd_path", type=str, required=True)

    parser.add_argument("--split_ratio", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--impute_n_samples", type=int, default=50)
    parser.add_argument("--chunk_len", type=int, default=36)
    parser.add_argument("--stride", type=int, default=36)

    parser.add_argument("--out_dir", type=str, default="", help="输出目录（可选）")
    args = parser.parse_args()

    set_global_seed(args.seed)

    # 输出目录
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"./save/pm25_history_imputed_split{args.split_ratio}_seed{args.seed}_{now}/"
    ensure_dir(out_dir)

    # 读数据切分
    df_full, df_miss = load_pm25_files(args.ground_path, args.missing_path)
    hist_full, hist_miss, fut_full, fut_miss, meta = time_split_df(df_full, df_miss, args.split_ratio)

    # mean/std
    with open(args.meanstd_path, "rb") as f:
        mean_K, std_K = pickle.load(f)
    mean_K = np.asarray(mean_K, dtype=np.float32)
    std_K = np.asarray(std_K, dtype=np.float32)
    std_K = np.where(std_K == 0, 1.0, std_K).astype(np.float32)

    # 模型路径
    model_path = args.model_path or os.path.join(args.run_folder, "model.pth")
    config_json = args.config_json or os.path.join(args.run_folder, "config.json")

    print("=" * 80)
    print("PM2.5 历史段补值（一次性）")
    print("=" * 80)
    print("model_path =", model_path)
    print("config_json =", config_json)
    print(json.dumps(meta, indent=4, ensure_ascii=False))

    model, _, _ = load_csdi_pm25(model_path, config_json, args.device)

    # 补值
    df_hist_imputed = csdi_impute_history_long(
        model=model,
        df_hist_missing=hist_miss,
        mean_K=mean_K,
        std_K=std_K,
        device=args.device,
        n_samples=args.impute_n_samples,
        chunk_len=args.chunk_len,
        stride=args.stride,
        seed=args.seed,
    )

    # 保存
    hist_csv = os.path.join(out_dir, "history_imputed.csv")
    df_hist_imputed.to_csv(hist_csv)

    meta_out = {
        **meta,
        "history_imputed_csv": "history_imputed.csv",
        "columns": list(df_hist_imputed.columns),
        "missing_count_in_hist": int(np.isnan(hist_miss.values).sum()),
        "missing_ratio_in_hist": float(np.isnan(hist_miss.values).mean()),
    }
    with open(os.path.join(out_dir, "impute_meta.json"), "w") as f:
        json.dump(meta_out, f, indent=4, ensure_ascii=False)

    print("\n完成！输出目录：", out_dir)
    print("history_imputed.csv =", hist_csv)
    print("impute_meta.json =", os.path.join(out_dir, "impute_meta.json"))


if __name__ == "__main__":
    main()
