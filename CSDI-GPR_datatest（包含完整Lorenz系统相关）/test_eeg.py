#!/usr/bin/env python3
# test_eeg.py - Use trained CSDI model for EEG data interpolation/densification

import torch
import numpy as np
import yaml
import argparse
import matplotlib.pyplot as plt
from main_model import CSDI_EEG
import pandas as pd
import os
from dataset_EEG import EEG_Dataset  # 假设您有这个数据集类

def load_model(model_path, config_path, device='cuda:0'):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = CSDI_EEG(config, device, target_dim=64).to(device)  # EEG有64个通道
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def load_eeg_data(data_path, seq_len=100):
    """加载EEG数据并准备测试样本"""
    # 读取EEG数据
    df = pd.read_excel(data_path, header=None)
    eeg_data = df.values
    
    # 标准化数据
    mean = np.mean(eeg_data, axis=0)
    std = np.std(eeg_data, axis=0)
    normalized_data = (eeg_data - mean) / (std + 1e-8)
    
    # 选择一个测试样本
    sample_idx = 0  # 使用第一个样本
    start_idx = 0   # 从开始位置
    test_sample = normalized_data[start_idx:start_idx+seq_len, :]
    
    return test_sample, mean, std

# 数据插值函数 - 将数据密度增加一倍
def interpolate_eeg(model, original_data, device='cuda:0', n_samples=10):
    """
    使用CSDI模型对EEG数据进行插值，将数据密度增加一倍
    
    参数:
        model: 训练好的CSDI模型
        original_data: 原始EEG数据
        device: 运行设备
        n_samples: 生成样本数量
    
    返回:
        densified_data: 密度增加一倍后的数据
    """
    # 准备数据
    seq_len, num_channels = original_data.shape
    
    # 创建插值掩码 - 在每两个原始点之间插入一个新点
    # 原始数据点作为已知条件
    known_mask = np.ones_like(original_data)
    
    # 创建一个扩展的序列，用于插值
    # 新序列长度将是原来的2倍-1 (在每两个点之间插入一个点)
    new_seq_len = seq_len * 2 - 1
    
    # 创建扩展的观察数据，将插值位置设为0
    extended_data = np.zeros((new_seq_len, num_channels))
    extended_mask = np.zeros((new_seq_len, num_channels))
    
    # 将原始数据放在偶数位置 (0, 2, 4, ...)
    extended_data[::2] = original_data
    extended_mask[::2] = 1  # 标记原始数据点为已知
    
    # Convert to tensors and add batch dimension
    observed_data = torch.tensor(extended_data, dtype=torch.float32).unsqueeze(0).to(device)
    observed_mask = torch.tensor(extended_mask, dtype=torch.float32).unsqueeze(0).to(device)
    cond_mask = observed_mask.clone()  # 使用原始数据点作为条件
    observed_tp = torch.arange(new_seq_len, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Reshape for model (model expects (batch, channels, time))
    observed_data = observed_data.permute(0, 2, 1)
    observed_mask = observed_mask.permute(0, 2, 1)
    cond_mask = cond_mask.permute(0, 2, 1)
    
    with torch.no_grad():
        print("Generating side information...")
        # Get side information
        side_info = model.get_side_info(observed_tp, cond_mask)
        
        print(f"Generating {n_samples} interpolation samples...")
        # Generate interpolated samples
        samples = model.impute(observed_data, cond_mask, side_info, n_samples)
        
        # Convert back to (batch, time, channels) format
        samples = samples.permute(0, 1, 3, 2)  # (1, n_samples, new_seq_len, num_channels)
        
        # Remove batch dimension
        samples = samples.squeeze(0)  # (n_samples, new_seq_len, num_channels)
        
        # Convert to numpy
        interpolated_samples = samples.cpu().numpy()
        
        # Calculate mean interpolation
        result = np.mean(interpolated_samples, axis=0)

        # 保留原始数据点
        result[::2] = extended_data[::2]

        return result, new_seq_len

def main():
    parser = argparse.ArgumentParser(description="Test CSDI model for EEG interpolation/densification")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model.pth")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to EEG data file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of interpolation samples")
    parser.add_argument("--seq_len", type=int, default=100, help="Original sequence length")
    parser.add_argument("--channel_idx", type=int, default=0, help="Channel index to visualize")
    
    args = parser.parse_args()
    
    # 加载模型
    model = load_model(args.model_path, args.config_path, args.device)
    
    # 加载EEG数据
    test_data, mean, std = load_eeg_data(args.data_path, args.seq_len)
    
    # 使用模型进行插值，增加数据密度
    densified_data, new_seq_len = interpolate_eeg(
        model,
        test_data,
        device=args.device,
        n_samples=args.n_samples
    )
    
    # 反标准化
    densified_data = densified_data * std + mean
    test_data = test_data * std + mean
    
    # 可视化特定通道的结果
    channel_idx = args.channel_idx
    plt.figure(figsize=(14, 8))
    
    # 创建时间索引
    original_time = np.arange(args.seq_len)
    densified_time = np.linspace(0, args.seq_len-1, new_seq_len)
    
    # 绘制原始数据点
    plt.plot(original_time, test_data[:, channel_idx], 'bo-', label='Original Data', 
             linewidth=1.5, markersize=6, alpha=0.7)
    
    # 绘制密度增加后的数据
    plt.plot(densified_time, densified_data[:, channel_idx], 'r-', label='Densified Data', 
             linewidth=1.2, alpha=0.8)
    
    # 标记原始数据点
    plt.scatter(original_time, test_data[:, channel_idx], 
                color='blue', s=40, label='Original Points', zorder=5)
    
    # 标记插值点（奇数位置）
    interpolated_indices = np.arange(1, new_seq_len, 2)
    plt.scatter(densified_time[interpolated_indices], densified_data[interpolated_indices, channel_idx], 
                color='red', marker='x', s=60, label='Interpolated Points', zorder=5)
    
    plt.xlabel('Time Step')
    plt.ylabel(f'Value (Channel {channel_idx})')
    plt.title(f'EEG Data Densification: Original vs Densified (Channel {channel_idx})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图像
    os.makedirs("./results", exist_ok=True)
    plt.savefig(f"./results/eeg_densification_channel_{channel_idx}.png", dpi=300)
    plt.show()
    
    # 打印统计信息
    print(f"\nStatistics for channel {channel_idx}:")
    print(f"Original sequence length: {args.seq_len}")
    print(f"Densified sequence length: {new_seq_len}")
    print(f"Data density increased by: {(new_seq_len/args.seq_len - 1)*100:.1f}%")
    print(f"Original sequence range: [{test_data[:, channel_idx].min():.4f}, {test_data[:, channel_idx].max():.4f}]")
    print(f"Densified sequence range: [{densified_data[:, channel_idx].min():.4f}, {densified_data[:, channel_idx].max():.4f}]")
    
    # 计算原始点位置的值差异（应该很小，因为我们保留了原始点）
    original_positions = np.arange(0, new_seq_len, 2)
    original_differences = np.abs(test_data[:, channel_idx] - densified_data[original_positions, channel_idx])
    print(f"Mean absolute difference at original points: {np.mean(original_differences):.6f}")
    
    # 保存结果
    np.savez("./results/eeg_densification_results.npz", 
             original=test_data, 
             densified=densified_data,
             original_time=original_time,
             densified_time=densified_time)
    
    print(f"\nResults saved to ./results/eeg_densification_results.npz")
    print(f"Original data shape: {test_data.shape}")
    print(f"Densified data shape: {densified_data.shape}")

if __name__ == "__main__":
    main()

'''
使用代码：
python test_eeg.py --model_path ./save/eeg_20250829_163713/model.pth --config_path config/eeg.yaml --data_path /home/rhl/CSDI-GPR_project/CSDI-GPR_datatest/data/Dataset_3-EEG.xlsx

输出文件：
- ./results/eeg_densification_channel_{N}.png: 通道N的可视化结果
- ./results/eeg_densification_results.npz: 包含原始数据和密度增加后的数据
'''