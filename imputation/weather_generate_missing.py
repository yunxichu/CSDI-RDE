# -*- coding: utf-8 -*-
"""
Weather 数据缺失生成脚本
- 生成多种缺失模式的数据：均匀、随机、时间块、特征缺失
- 输出：ground.npy (原始数据), missing_X.npy (缺失数据), 可视化图片

运行示例：
python imputation/weather_generate_missing.py \
  --data_path ./data/weather/weather.npy \
  --missing_ratios 0.1,0.2 \
  --modes uniform,random \
  --seed 42 \
  --visualize_timesteps 200
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def generate_uniform_missing(data, missing_ratio, seed=42):
    """
    均匀缺失：按照固定间隔挖去数据
    """
    np.random.seed(seed)
    n_timesteps, n_features = data.shape
    
    missing_mask = np.ones_like(data, dtype=np.float32)
    
    interval = int(1.0 / missing_ratio)
    
    for f in range(n_features):
        for t in range(0, n_timesteps, interval):
            missing_mask[t, f] = 0
    
    missing_data = data.copy()
    missing_data[missing_mask == 0] = -9999.0
    
    return missing_data, missing_mask


def generate_random_missing(data, missing_ratio, seed=42):
    """
    随机缺失：随机选择位置挖去数据
    """
    np.random.seed(seed)
    n_timesteps, n_features = data.shape
    
    total_points = n_timesteps * n_features
    n_missing = int(total_points * missing_ratio)
    
    all_indices = np.arange(total_points)
    np.random.shuffle(all_indices)
    missing_indices = all_indices[:n_missing]
    
    missing_mask = np.ones(total_points, dtype=np.float32)
    missing_mask[missing_indices] = 0
    missing_mask = missing_mask.reshape(n_timesteps, n_features)
    
    missing_data = data.copy()
    missing_data[missing_mask == 0] = -9999.0
    
    return missing_data, missing_mask


def generate_temporal_missing(data, missing_ratio, seed=42):
    """
    时间块缺失：挖去连续的时间段
    """
    np.random.seed(seed)
    n_timesteps, n_features = data.shape
    
    missing_mask = np.ones_like(data, dtype=np.float32)
    
    total_missing = int(n_timesteps * n_features * missing_ratio)
    block_size = max(1, int(n_timesteps * 0.05))
    n_blocks = total_missing // (block_size * n_features) + 1
    
    for _ in range(n_blocks):
        start_t = np.random.randint(0, max(1, n_timesteps - block_size))
        end_t = min(start_t + block_size, n_timesteps)
        missing_mask[start_t:end_t, :] = 0
    
    missing_data = data.copy()
    missing_data[missing_mask == 0] = -9999.0
    
    return missing_data, missing_mask


def generate_feature_missing(data, missing_ratio, seed=42):
    """
    特征缺失：随机选择某些特征完全或部分缺失
    """
    np.random.seed(seed)
    n_timesteps, n_features = data.shape
    
    missing_mask = np.ones_like(data, dtype=np.float32)
    
    n_missing_features = max(1, int(n_features * missing_ratio))
    missing_features = np.random.choice(n_features, n_missing_features, replace=False)
    
    for f in missing_features:
        missing_ratio_f = np.random.uniform(0.3, 1.0)
        n_missing_t = int(n_timesteps * missing_ratio_f)
        missing_t = np.random.choice(n_timesteps, n_missing_t, replace=False)
        missing_mask[missing_t, f] = 0
    
    missing_data = data.copy()
    missing_data[missing_mask == 0] = -9999.0
    
    return missing_data, missing_mask


def visualize_missing_patterns(data, missing_masks, modes, ratios, out_dir, n_timesteps_show=200, n_features_show=6):
    """
    可视化不同缺失模式
    """
    n_modes = len(modes)
    n_ratios = len(ratios)
    
    fig, axes = plt.subplots(n_ratios + 1, n_modes, figsize=(5 * n_modes, 4 * (n_ratios + 1)))
    if n_modes == 1:
        axes = axes.reshape(-1, 1)
    if n_ratios == 0:
        axes = axes.reshape(1, -1)
    
    data_show = data[:n_timesteps_show, :n_features_show]
    
    ax = axes[0, 0] if n_modes > 1 else axes[0]
    im = ax.imshow(data_show.T, aspect='auto', cmap='viridis')
    ax.set_title('Original Data (Ground Truth)', fontsize=12)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Feature')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    for j in range(1, n_modes):
        axes[0, j].axis('off')
    
    mode_names = {
        "uniform": "Uniform Missing",
        "random": "Random Missing",
        "temporal": "Temporal Block Missing",
        "feature": "Feature Missing"
    }
    
    for i, ratio in enumerate(ratios):
        for j, mode in enumerate(modes):
            ax = axes[i + 1, j] if n_modes > 1 else axes[i + 1]
            
            mask = missing_masks[(mode, ratio)][:n_timesteps_show, :n_features_show]
            
            display_data = np.where(mask == 1, data_show, np.nan)
            
            im = ax.imshow(display_data.T, aspect='auto', cmap='viridis')
            ax.set_title(f'{mode_names.get(mode, mode)}\nRatio={ratio:.0%}', fontsize=11)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Feature')
            
            missing_positions = np.where(mask == 0)
            if len(missing_positions[0]) > 0:
                ax.scatter(missing_positions[0], missing_positions[1], 
                          c='red', s=1, alpha=0.3, marker='.')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'missing_patterns_overview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存缺失模式概览图: {os.path.join(out_dir, 'missing_patterns_overview.png')}")


def visualize_missing_timeseries(data, missing_data, missing_mask, mode, ratio, out_dir, 
                                  n_timesteps_show=200, features_to_show=[0, 5, 10, 15, 20]):
    """
    可视化单个缺失模式的时间序列
    """
    n_features = len(features_to_show)
    fig, axes = plt.subplots(n_features, 1, figsize=(16, 3 * n_features))
    if n_features == 1:
        axes = [axes]
    
    time_idx = np.arange(min(n_timesteps_show, len(data)))
    
    for i, feat_idx in enumerate(features_to_show):
        if feat_idx >= data.shape[1]:
            continue
            
        ax = axes[i]
        
        ground_values = data[:len(time_idx), feat_idx]
        mask = missing_mask[:len(time_idx), feat_idx]
        
        ax.plot(time_idx, ground_values, 'b-', linewidth=1, alpha=0.7, label='Ground Truth')
        
        observed_idx = np.where(mask == 1)[0]
        ax.scatter(time_idx[observed_idx], ground_values[observed_idx], 
                  c='blue', s=20, alpha=0.8, label='Observed', zorder=3)
        
        missing_idx = np.where(mask == 0)[0]
        if len(missing_idx) > 0:
            ax.scatter(time_idx[missing_idx], ground_values[missing_idx], 
                      c='red', s=40, marker='x', alpha=0.8, label='Missing (to impute)', zorder=4)
        
        ax.set_title(f'Feature {feat_idx} - {mode} missing (ratio={ratio:.0%})', fontsize=11)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        n_missing = len(missing_idx)
        ax.text(0.02, 0.95, f'Missing points: {n_missing} ({100*n_missing/len(time_idx):.1f}%)', 
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filename = f'missing_timeseries_{mode}_ratio{ratio:.1f}.png'
    plt.savefig(os.path.join(out_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存时间序列图: {os.path.join(out_dir, filename)}")


def visualize_missing_heatmap(missing_mask, mode, ratio, out_dir):
    """
    可视化缺失位置热力图
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    missing_ratio_per_time = 1 - missing_mask.mean(axis=1)
    missing_ratio_per_feature = 1 - missing_mask.mean(axis=0)
    
    gs = fig.add_gridspec(2, 2, width_ratios=[20, 1], height_ratios=[1, 20],
                          hspace=0.05, wspace=0.05)
    
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    
    ax_top.bar(np.arange(len(missing_ratio_per_time)), missing_ratio_per_time, 
               color='coral', alpha=0.7, width=1.0)
    ax_top.set_xlim(0, len(missing_ratio_per_time))
    ax_top.set_ylabel('Missing %')
    ax_top.set_title(f'Missing Pattern: {mode} (ratio={ratio:.0%})', fontsize=12)
    ax_top.set_xticks([])
    
    im = ax_main.imshow(missing_mask.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax_main.set_xlabel('Time Step')
    ax_main.set_ylabel('Feature')
    
    ax_right.barh(np.arange(len(missing_ratio_per_feature)), missing_ratio_per_feature,
                  color='coral', alpha=0.7, height=1.0)
    ax_right.set_ylim(0, len(missing_ratio_per_feature))
    ax_right.set_xlabel('Missing %')
    ax_right.set_yticks([])
    
    plt.savefig(os.path.join(out_dir, f'missing_heatmap_{mode}_ratio{ratio:.1f}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存热力图: {os.path.join(out_dir, f'missing_heatmap_{mode}_ratio{ratio:.1f}.png')}")


def main():
    parser = argparse.ArgumentParser(description="Weather Missing Data Generator with Visualization")
    
    parser.add_argument("--data_path", type=str, default="./data/weather/weather.npy")
    parser.add_argument("--out_dir", type=str, default="", help="输出目录")
    parser.add_argument("--missing_ratios", type=str, default="0.1,0.2",
                        help="缺失率列表，逗号分隔")
    parser.add_argument("--modes", type=str, default="uniform,random",
                        help="缺失模式：uniform(均匀), random(随机), temporal(时间块), feature(特征)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visualize_timesteps", type=int, default=200, help="可视化显示的时间步数")
    parser.add_argument("--skip_visualization", action="store_true", help="跳过可视化")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Weather 缺失数据生成器（含可视化）")
    print("=" * 80)
    
    data = np.load(args.data_path)
    print(f"原始数据形状: {data.shape}")
    print(f"原始数据范围: [{data.min():.2f}, {data.max():.2f}]")
    
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.dirname(args.data_path)
    os.makedirs(out_dir, exist_ok=True)
    
    ground_path = os.path.join(out_dir, "weather_ground.npy")
    np.save(ground_path, data)
    print(f"\n保存原始数据: {ground_path}")
    
    missing_ratios = [float(x) for x in args.missing_ratios.split(",")]
    modes = args.modes.split(",")
    
    mode_functions = {
        "uniform": generate_uniform_missing,
        "random": generate_random_missing,
        "temporal": generate_temporal_missing,
        "feature": generate_feature_missing,
    }
    
    print(f"\n缺失率: {missing_ratios}")
    print(f"缺失模式: {modes}")
    print()
    
    all_results = []
    all_missing_masks = {}
    
    for mode in modes:
        if mode not in mode_functions:
            print(f"警告: 未知模式 '{mode}'，跳过")
            continue
        
        func = mode_functions[mode]
        
        for ratio in tqdm(missing_ratios, desc=f"生成 {mode} 缺失"):
            missing_data, missing_mask = func(data, ratio, args.seed)
            
            actual_ratio = 1.0 - missing_mask.mean()
            
            filename = f"weather_missing_{mode}_ratio{ratio:.1f}_seed{args.seed}.npy"
            filepath = os.path.join(out_dir, filename)
            np.save(filepath, missing_data)
            
            mask_filename = f"weather_mask_{mode}_ratio{ratio:.1f}_seed{args.seed}.npy"
            mask_filepath = os.path.join(out_dir, mask_filename)
            np.save(mask_filepath, missing_mask)
            
            all_missing_masks[(mode, ratio)] = missing_mask
            
            all_results.append({
                "mode": mode,
                "ratio": float(ratio),
                "actual_ratio": float(actual_ratio),
                "filename": filename,
                "mask_filename": mask_filename,
            })
            
            print(f"  {mode} ratio={ratio:.1f}: 实际缺失率={actual_ratio:.4f}, 文件={filename}")
    
    if not args.skip_visualization:
        print("\n" + "=" * 80)
        print("生成可视化...")
        print("=" * 80)
        
        visualize_missing_patterns(data, all_missing_masks, modes, missing_ratios, out_dir, 
                                   n_timesteps_show=args.visualize_timesteps)
        
        for mode in modes:
            for ratio in missing_ratios:
                if (mode, ratio) in all_missing_masks:
                    missing_mask = all_missing_masks[(mode, ratio)]
                    missing_filename = f"weather_missing_{mode}_ratio{ratio:.1f}_seed{args.seed}.npy"
                    missing_data = np.load(os.path.join(out_dir, missing_filename))
                    
                    visualize_missing_timeseries(data, missing_data, missing_mask, mode, ratio, out_dir,
                                                n_timesteps_show=args.visualize_timesteps)
                    
                    visualize_missing_heatmap(missing_mask, mode, ratio, out_dir)
    
    print("\n" + "=" * 80)
    print("生成完成！")
    print("=" * 80)
    print(f"\n输出目录: {out_dir}")
    print(f"原始数据: weather_ground.npy")
    print("\n生成的缺失数据文件:")
    for r in all_results:
        print(f"  - {r['filename']} (模式: {r['mode']}, 缺失率: {r['actual_ratio']:.2%})")
    
    print("\n生成的可视化文件:")
    print("  - missing_patterns_overview.png (缺失模式概览)")
    for mode in modes:
        for ratio in missing_ratios:
            print(f"  - missing_timeseries_{mode}_ratio{ratio:.1f}.png (时间序列)")
            print(f"  - missing_heatmap_{mode}_ratio{ratio:.1f}.png (热力图)")
    
    import json
    meta_path = os.path.join(out_dir, "missing_data_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "ground_file": "weather_ground.npy",
            "original_shape": list(data.shape),
            "seed": args.seed,
            "results": all_results,
        }, f, indent=4)
    print(f"\n元数据文件: {meta_path}")


if __name__ == "__main__":
    main()
