# -*- coding: utf-8 -*-
"""
Physio 历史段一次性补值脚本：
- 前 split_ratio 部分：用 CSDI_Physio 对 missing 进行补值
- 输出 3 个文件：
    1) history_full.csv                  # 原始 ground 历史段
    2) history_missing_positions.csv     # 历史段缺失位置
    3) history_imputed.csv               # 历史段补值后的完整矩阵

运行示例：
python imputation/physio_CSDIimpute.py \
  --run_folder ./csdi/save/physio_fold0_20260324_110537 \
  --data_path ./data/physio/ \
  --split_ratio 0.5 \
  --device cuda:0 \
  --seed 1 \
  --impute_n_samples 50
"""

import os
import json
import random
import argparse
import datetime
import pickle
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'csdi'))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from main_model import CSDI_Physio


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


def load_csdi_physio(model_path: str, config_json_path: str, device: str):
    with open(config_json_path, "r") as f:
        full_cfg = json.load(f)
    config = full_cfg.get("model_config", full_cfg)

    model = CSDI_Physio(config, device).to(device)
    sd = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    return model, config, full_cfg


def csdi_impute_physio_batch(
    model,
    observed_data: np.ndarray,   # (L, K) with missing
    observed_mask: np.ndarray,   # (L, K) 1=observed, 0=missing
    gt_mask: np.ndarray,         # (L, K) 1=ground truth, 0=to impute
    mean_K: np.ndarray,          # (K,)
    std_K: np.ndarray,           # (K,)
    device: str,
    n_samples: int,
    seed: int,
):
    L, K = observed_data.shape
    
    # observed_data已经是标准化后的数据，直接使用
    x_norm_LK = observed_data * observed_mask

    observed_data_t = torch.from_numpy(x_norm_LK).unsqueeze(0).to(device).float()
    observed_data_t = observed_data_t.permute(0, 2, 1).contiguous()

    cond_mask = torch.from_numpy(observed_mask).unsqueeze(0).to(device).float()
    cond_mask = cond_mask.permute(0, 2, 1).contiguous()

    observed_tp = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)

    torch.manual_seed(int(seed))

    with torch.no_grad():
        side_info = model.get_side_info(observed_tp, cond_mask)
        samples = model.impute(observed_data_t, cond_mask, side_info, n_samples)

    samples_nLK = samples[0].permute(0, 2, 1).contiguous().cpu().numpy()
    pred_norm_LK = samples_nLK.mean(axis=0)
    pred_LK = pred_norm_LK * std_K + mean_K

    # observed_data已经是标准化后的，需要反标准化
    observed_data_unnorm = observed_data * std_K + mean_K
    
    out = observed_data_unnorm.copy()
    missing_mask = (gt_mask == 0)
    out[missing_mask] = pred_LK[missing_mask]
    return out


def main():
    parser = argparse.ArgumentParser(description="Physio History Imputation Once (CSDI_Physio)")

    parser.add_argument("--run_folder", type=str, required=True,
                        help="训练脚本生成的文件夹（里面应有 model.pth / config.json）")
    parser.add_argument("--model_path", type=str, default="", help="可选：显式指定 model.pth")
    parser.add_argument("--config_json", type=str, default="", help="可选：显式指定 config.json")

    parser.add_argument("--data_path", type=str, default="./data/physio/")
    parser.add_argument("--missing_ratio", type=float, default=0.1)
    parser.add_argument("--nfold", type=int, default=0)

    parser.add_argument("--split_ratio", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--impute_n_samples", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="", help="输出目录（可选）")
    args = parser.parse_args()

    set_global_seed(args.seed)

    print("=" * 80)
    print("Physio 历史段补值（一次性）")
    print("=" * 80)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets'))
    from dataset_physio import Physio_Dataset

    dataset = Physio_Dataset(missing_ratio=args.missing_ratio, seed=args.seed)

    total_len = len(dataset)
    split_point = int(total_len * args.split_ratio)

    hist_indices = list(range(split_point))
    fut_indices = list(range(split_point, total_len))

    meta = {
        "total_len": total_len,
        "split_ratio": args.split_ratio,
        "split_point": split_point,
        "hist_len": len(hist_indices),
        "fut_len": len(fut_indices),
    }
    print(json.dumps(meta, indent=4, ensure_ascii=False))

    mean_K = dataset.train_mean
    std_K = dataset.train_std
    std_K = np.where(std_K == 0, 1.0, std_K).astype(np.float32)

    model_path = args.model_path or os.path.join(args.run_folder, "model.pth")
    config_json = args.config_json or os.path.join(args.run_folder, "config.json")

    print("model_path =", model_path)
    print("config_json =", config_json)

    model, _, _ = load_csdi_physio(model_path, config_json, args.device)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"./save/physio_history_imputed_split{args.split_ratio}_seed{args.seed}_{now}/"
    ensure_dir(out_dir)

    all_imputed = []
    all_full = []
    all_missing_positions = []

    print("\nCSDI imputing history...")
    for idx in tqdm(hist_indices):
        sample = dataset[idx]
        observed_data = sample['observed_data']
        observed_mask = sample['observed_mask']
        gt_mask = sample['gt_mask']
        gt_data = sample['gt_data']

        imputed = csdi_impute_physio_batch(
            model=model,
            observed_data=observed_data,
            observed_mask=observed_mask,
            gt_mask=gt_mask,
            mean_K=mean_K,
            std_K=std_K,
            device=args.device,
            n_samples=args.impute_n_samples,
            seed=args.seed + idx,
        )

        # gt_data已经是原始值，不需要反标准化
        gt_data_unnorm = gt_data

        all_imputed.append(imputed)
        all_full.append(gt_data_unnorm)

        missing_mask = (gt_mask == 0)
        if missing_mask.any():
            miss_rows, miss_cols = np.where(missing_mask)
            for r, c in zip(miss_rows, miss_cols):
                all_missing_positions.append({
                    "sample_idx": idx,
                    "time_step": r,
                    "feature": c,
                    "original_value": gt_data_unnorm[r, c],  # 使用反标准化后的真实值
                    "imputed_value": imputed[r, c],
                })

    hist_imputed = np.array(all_imputed)
    hist_full = np.array(all_full)

    np.save(os.path.join(out_dir, "history_imputed.npy"), hist_imputed)
    np.save(os.path.join(out_dir, "history_full.npy"), hist_full)

    if all_missing_positions:
        miss_df = pd.DataFrame(all_missing_positions)
        miss_df.to_csv(os.path.join(out_dir, "history_missing_positions.csv"), index=False)
    else:
        pd.DataFrame(columns=["sample_idx", "time_step", "feature", "original_value", "imputed_value"]).to_csv(
            os.path.join(out_dir, "history_missing_positions.csv"), index=False
        )

    meta_out = {
        **meta,
        "history_imputed_npy": "history_imputed.npy",
        "history_full_npy": "history_full.npy",
        "history_missing_positions_csv": "history_missing_positions.csv",
        "missing_count_in_hist": int((hist_full != hist_imputed).sum()),
        "missing_samples_exported": len(all_missing_positions),
    }
    with open(os.path.join(out_dir, "impute_meta.json"), "w") as f:
        json.dump(meta_out, f, indent=4, ensure_ascii=False)

    print("\n完成！输出目录：", out_dir)
    print("history_imputed.npy =", os.path.join(out_dir, "history_imputed.npy"))
    print("history_full.npy =", os.path.join(out_dir, "history_full.npy"))
    print("history_missing_positions.csv =", os.path.join(out_dir, "history_missing_positions.csv"))
    print("impute_meta.json =", os.path.join(out_dir, "impute_meta.json"))


if __name__ == "__main__":
    main()
