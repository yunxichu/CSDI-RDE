# -*- coding: utf-8 -*-
"""
Weather 历史段一次性补值脚本（含可视化）

运行示例：
python imputation/weather_CSDIimpute.py \
  --run_folder ./save/weather_random_ratio0.1_fold0_XXXXXXXX \
  --missing_ratio 0.1 \
  --missing_mode random \
  --split_ratio 0.5 \
  --device cuda:0 \
  --seed 42 \
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
import matplotlib.pyplot as plt

from main_model import CSDI_Weather


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


def load_csdi_weather(model_path: str, config_json_path: str, device: str, target_dim: int):
    with open(config_json_path, "r") as f:
        full_cfg = json.load(f)
    config = full_cfg.get("model_config", full_cfg.get("config", full_cfg))
    
    model = CSDI_Weather(config, device, target_dim=target_dim).to(device)
    sd = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    return model, config, full_cfg


def csdi_impute_weather_batch(
    model,
    observed_data: np.ndarray,
    observed_mask: np.ndarray,
    gt_mask: np.ndarray,
    mean_K: np.ndarray,
    std_K: np.ndarray,
    device: str,
    n_samples: int,
    seed: int,
):
    L, K = observed_data.shape
    
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
    
    observed_data_unnorm = observed_data * std_K + mean_K
    
    out = observed_data_unnorm.copy()
    missing_mask = (gt_mask == 0)
    out[missing_mask] = pred_LK[missing_mask]
    return out


def visualize_imputation_results(missing_positions, out_dir, n_features=21, features_to_show=None):
    """
    可视化补值结果
    """
    if features_to_show is None:
        features_to_show = list(range(min(6, n_features)))
    
    df = pd.DataFrame(missing_positions)
    
    # 1. 补值误差分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    errors = df['original_value'] - df['imputed_value']
    
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Error (Original - Imputed)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Imputation Error Distribution')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(df['original_value'], df['imputed_value'], alpha=0.3, s=10)
    max_val = max(df['original_value'].max(), df['imputed_value'].max())
    min_val = min(df['original_value'].min(), df['imputed_value'].min())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect fit')
    axes[1].set_xlabel('Original Value')
    axes[1].set_ylabel('Imputed Value')
    axes[1].set_title('Original vs Imputed Values')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'imputation_error_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存补值误差分布图: {os.path.join(out_dir, 'imputation_error_distribution.png')}")
    
    # 2. 各特征的RMSE
    feature_rmse = []
    for feat in range(n_features):
        feat_df = df[df['feature'] == feat]
        if len(feat_df) > 0:
            rmse = np.sqrt(np.mean((feat_df['original_value'] - feat_df['imputed_value'])**2))
            mae = np.mean(np.abs(feat_df['original_value'] - feat_df['imputed_value']))
            feature_rmse.append({'feature': feat, 'rmse': rmse, 'mae': mae, 'count': len(feat_df)})
    
    if feature_rmse:
        rmse_df = pd.DataFrame(feature_rmse)
        
        fig, ax = plt.subplots(figsize=(14, 5))
        bars = ax.bar(rmse_df['feature'], rmse_df['rmse'], color='steelblue', alpha=0.7)
        ax.set_xlabel('Feature Index')
        ax.set_ylabel('RMSE')
        ax.set_title('Imputation RMSE per Feature')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars, rmse_df['count']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'n={count}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'imputation_rmse_per_feature.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"保存各特征RMSE图: {os.path.join(out_dir, 'imputation_rmse_per_feature.png')}")
    
    # 3. 选定特征的时间序列对比
    n_show = len(features_to_show)
    fig, axes = plt.subplots(n_show, 1, figsize=(16, 4 * n_show))
    if n_show == 1:
        axes = [axes]
    
    for i, feat in enumerate(features_to_show):
        if feat >= n_features:
            continue
        feat_df = df[df['feature'] == feat].sort_values('time_step')
        
        ax = axes[i]
        
        if len(feat_df) > 0:
            ax.scatter(feat_df['time_step'], feat_df['original_value'], 
                      c='blue', s=30, alpha=0.7, label='Original', marker='o')
            ax.scatter(feat_df['time_step'], feat_df['imputed_value'], 
                      c='red', s=30, alpha=0.7, label='Imputed', marker='x')
            
            for _, row in feat_df.iterrows():
                ax.plot([row['time_step'], row['time_step']], 
                       [row['original_value'], row['imputed_value']], 
                       'gray', alpha=0.3, linewidth=0.5)
            
            if len(feat_df) > 0:
                rmse = np.sqrt(np.mean((feat_df['original_value'] - feat_df['imputed_value'])**2))
                ax.text(0.02, 0.95, f'Feature {feat}: RMSE={rmse:.2f}, n={len(feat_df)}', 
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.set_title(f'Feature {feat} - Original vs Imputed')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'imputation_timeseries_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存时间序列对比图: {os.path.join(out_dir, 'imputation_timeseries_comparison.png')}")


def visualize_imputation_heatmap(hist_imputed, hist_full, hist_gt_masks, out_dir, n_timesteps_show=500):
    """
    可视化补值结果热力图
    """
    n_timesteps = min(n_timesteps_show, len(hist_imputed) * hist_imputed.shape[1])
    
    if len(hist_imputed.shape) == 3:
        imputed_flat = hist_imputed.reshape(-1, hist_imputed.shape[-1])[:n_timesteps]
        full_flat = hist_full.reshape(-1, hist_full.shape[-1])[:n_timesteps]
        masks_flat = hist_gt_masks.reshape(-1, hist_gt_masks.shape[-1])[:n_timesteps]
    else:
        imputed_flat = hist_imputed[:n_timesteps]
        full_flat = hist_full[:n_timesteps]
        masks_flat = hist_gt_masks[:n_timesteps]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    im0 = axes[0].imshow(full_flat.T, aspect='auto', cmap='viridis')
    axes[0].set_title('Ground Truth', fontsize=12)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Feature')
    plt.colorbar(im0, ax=axes[0], shrink=0.6)
    
    im1 = axes[1].imshow(imputed_flat.T, aspect='auto', cmap='viridis')
    axes[1].set_title('Imputed Data', fontsize=12)
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Feature')
    plt.colorbar(im1, ax=axes[1], shrink=0.6)
    
    error = np.abs(full_flat - imputed_flat)
    im2 = axes[2].imshow(error.T, aspect='auto', cmap='Reds')
    axes[2].set_title('Absolute Error', fontsize=12)
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Feature')
    plt.colorbar(im2, ax=axes[2], shrink=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'imputation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存补值热力图: {os.path.join(out_dir, 'imputation_heatmap.png')}")


def main():
    parser = argparse.ArgumentParser(description="Weather History Imputation Once (CSDI_Weather)")
    
    parser.add_argument("--run_folder", type=str, required=True,
                        help="训练脚本生成的文件夹（里面应有 model.pth / config.json）")
    parser.add_argument("--model_path", type=str, default="", help="可选：显式指定 model.pth")
    parser.add_argument("--config_json", type=str, default="", help="可选：显式指定 config.json")
    
    parser.add_argument("--missing_ratio", type=float, default=0.1)
    parser.add_argument("--missing_mode", type=str, default="random")
    parser.add_argument("--split_ratio", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--impute_n_samples", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="", help="输出目录（可选）")
    parser.add_argument("--skip_visualization", action="store_true", help="跳过可视化")
    parser.add_argument("--visualize_timesteps", type=int, default=500, help="可视化显示的时间步数")
    parser.add_argument("--visualize_features", type=str, default="0,1,2,3,4,5", help="可视化的特征索引")
    args = parser.parse_args()
    
    set_global_seed(args.seed)
    
    print("=" * 80)
    print("Weather 历史段补值（一次性）")
    print("=" * 80)
    
    sys.path.insert(0, os.path.join(project_root, 'datasets'))
    from dataset_weather import Weather_Dataset
    
    dataset = Weather_Dataset(
        missing_ratio=args.missing_ratio, 
        seed=args.seed,
        missing_mode=args.missing_mode,
        use_generated_missing=True,
    )
    
    total_len = len(dataset.gt_values)
    split_point = int(total_len * args.split_ratio)
    
    meta = {
        "total_len": total_len,
        "split_ratio": args.split_ratio,
        "split_point": split_point,
        "hist_len": split_point,
        "fut_len": total_len - split_point,
        "missing_ratio": args.missing_ratio,
        "missing_mode": args.missing_mode,
    }
    print(json.dumps(meta, indent=4, ensure_ascii=False))
    
    mean_K = dataset.train_mean
    std_K = dataset.train_std
    std_K = np.where(std_K == 0, 1.0, std_K).astype(np.float32)
    
    model_path = args.model_path or os.path.join(args.run_folder, "model.pth")
    config_json = args.config_json or os.path.join(args.run_folder, "config.json")
    
    print("model_path =", model_path)
    print("config_json =", config_json)
    
    model, _, _ = load_csdi_weather(model_path, config_json, args.device, target_dim=21)
    
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"./save/weather_history_imputed_{args.missing_mode}_ratio{args.missing_ratio}_split{args.split_ratio}_seed{args.seed}_{now}/"
    ensure_dir(out_dir)
    
    all_imputed = []
    all_full = []
    all_missing_positions = []
    
    hist_gt_values = dataset.gt_values[:split_point]
    hist_gt_masks = dataset.gt_masks[:split_point]
    hist_observed_data = dataset.observed_data[:split_point]
    hist_observed_masks = dataset.observed_masks[:split_point]
    
    print("\nCSDI imputing history...")
    
    chunk_len = 36
    stride = 36
    
    for start in tqdm(range(0, split_point - chunk_len + 1, stride), desc="CSDI imputing"):
        end = start + chunk_len
        
        observed_data = hist_observed_data[start:end]
        observed_mask = hist_observed_masks[start:end]
        gt_mask = hist_gt_masks[start:end]
        gt_data = hist_gt_values[start:end]
        
        imputed = csdi_impute_weather_batch(
            model=model,
            observed_data=observed_data,
            observed_mask=observed_mask,
            gt_mask=gt_mask,
            mean_K=mean_K,
            std_K=std_K,
            device=args.device,
            n_samples=args.impute_n_samples,
            seed=args.seed + start,
        )
        
        all_imputed.append(imputed)
        all_full.append(gt_data)
        
        missing_mask = (gt_mask == 0)
        if missing_mask.any():
            miss_rows, miss_cols = np.where(missing_mask)
            for r, c in zip(miss_rows, miss_cols):
                all_missing_positions.append({
                    "time_step": start + r,
                    "feature": c,
                    "original_value": gt_data[r, c],
                    "imputed_value": imputed[r, c],
                })
    
    hist_imputed = np.array(all_imputed)
    hist_full = np.array(all_full)
    
    hist_imputed_merged = hist_imputed.reshape(-1, hist_imputed.shape[-1])
    hist_full_merged = hist_full.reshape(-1, hist_full.shape[-1])
    
    np.save(os.path.join(out_dir, "history_imputed.npy"), hist_imputed_merged)
    np.save(os.path.join(out_dir, "history_full.npy"), hist_full_merged)
    
    if all_missing_positions:
        miss_df = pd.DataFrame(all_missing_positions)
        miss_df.to_csv(os.path.join(out_dir, "history_missing_positions.csv"), index=False)
    else:
        pd.DataFrame(columns=["time_step", "feature", "original_value", "imputed_value"]).to_csv(
            os.path.join(out_dir, "history_missing_positions.csv"), index=False
        )
    
    meta_out = {
        **meta,
        "history_imputed_npy": "history_imputed.npy",
        "history_full_npy": "history_full.npy",
        "history_missing_positions_csv": "history_missing_positions.csv",
        "missing_count_in_hist": len(all_missing_positions),
    }
    with open(os.path.join(out_dir, "impute_meta.json"), "w") as f:
        json.dump(meta_out, f, indent=4, ensure_ascii=False)
    
    print("\n完成！输出目录：", out_dir)
    print("history_imputed.npy =", os.path.join(out_dir, "history_imputed.npy"))
    print("history_full.npy =", os.path.join(out_dir, "history_full.npy"))
    print("history_missing_positions.csv =", os.path.join(out_dir, "history_missing_positions.csv"))
    
    # 可视化
    if not args.skip_visualization and all_missing_positions:
        print("\n" + "=" * 80)
        print("生成可视化...")
        print("=" * 80)
        
        features_to_show = [int(x) for x in args.visualize_features.split(",")]
        
        visualize_imputation_results(all_missing_positions, out_dir, n_features=21, features_to_show=features_to_show)
        
        visualize_imputation_heatmap(hist_imputed, hist_full, hist_gt_masks, out_dir, 
                                     n_timesteps_show=args.visualize_timesteps)
        
        print("\n生成的可视化文件:")
        print("  - imputation_error_distribution.png (误差分布)")
        print("  - imputation_rmse_per_feature.png (各特征RMSE)")
        print("  - imputation_timeseries_comparison.png (时间序列对比)")
        print("  - imputation_heatmap.png (热力图)")


if __name__ == "__main__":
    main()
