#!/usr/bin/env python3
# test_eeg_densification.py - Use trained CSDI model for EEG data densification
"""
# 基本使用
python test_eeg_densification.py --model_path ./save/eeg_20251025_185411/model.pth --config_path config/eeg.yaml --data_path /home/rhl/CSDI-GPR_project/CSDI-GPR_datatest/data/Dataset_3-EEG.xlsx

# 生成更多样本以进行不确定性分析
python test_eeg_densification.py --model_path ./save/eeg_20251025_185411/model.pth --config_path config/eeg.yaml --data_path /home/rhl/CSDI-GPR_project/CSDI-GPR_datatest/data/Dataset_3-EEG.xlsx --n_samples 50

# 增加密度增强倍数
python test_eeg_densification.py --model_path ./save/eeg_20251025_185411/model.pth --config_path config/eeg.yaml --data_path /home/rhl/CSDI-GPR_project/CSDI-GPR_datatest/data/Dataset_3-EEG.xlsx --densification_factor 3
"""
import torch
import numpy as np
import yaml
import argparse
import matplotlib.pyplot as plt
from main_model import CSDI_EEG
import pandas as pd
import os
from dataset_EEG import EEG_Dataset
import seaborn as sns
from scipy import signal

def load_model(model_path, config_path, device='cuda:0'):
    """
    加载训练好的CSDI模型
    """
    # 加载配置
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 初始化模型
    model = CSDI_EEG(config, device, target_dim=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def load_eeg_data(data_path, seq_len=100):
    """
    加载EEG数据并准备测试样本
    """
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

def densify_eeg(model, original_data, device='cuda:0', n_samples=10, densification_factor=2):
    """
    使用CSDI模型对EEG数据进行密度增强
    
    参数:
        model: 训练好的CSDI模型
        original_data: 原始EEG数据
        device: 运行设备
        n_samples: 生成样本数量
        densification_factor: 密度增强倍数
    
    返回:
        densified_data: 密度增强后的数据
        new_seq_len: 新序列长度
    """
    # 准备数据
    seq_len, num_channels = original_data.shape
    
    # 创建扩展序列：在原始点之间插入待填补位置
    new_seq_len = seq_len * densification_factor - (densification_factor - 1)
    extended_data = np.zeros((new_seq_len, num_channels))
    extended_mask = np.zeros((new_seq_len, num_channels))
    
    # 放置原始数据点（作为已知条件）
    extended_data[::densification_factor] = original_data
    extended_mask[::densification_factor] = 1.0
    
    # 转换为张量并添加批次维度
    observed_data = torch.tensor(extended_data, dtype=torch.float32).unsqueeze(0).to(device)
    observed_mask = torch.tensor(extended_mask, dtype=torch.float32).unsqueeze(0).to(device)
    cond_mask = observed_mask.clone()  # 使用已知位置作为条件
    observed_tp = torch.arange(new_seq_len, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 为模型调整形状
    observed_data = observed_data.permute(0, 2, 1)
    observed_mask = observed_mask.permute(0, 2, 1)
    cond_mask = cond_mask.permute(0, 2, 1)
    
    with torch.no_grad():
        print("生成辅助信息...")
        # 获取辅助信息
        side_info = model.get_side_info(observed_tp, cond_mask)
        
        print(f"生成 {n_samples} 个密度增强样本...")
        # 生成填补样本
        samples = model.impute(observed_data, cond_mask, side_info, n_samples)
        
        # 转换回(batch, time, channels)格式
        samples = samples.permute(0, 1, 3, 2)  # (1, n_samples, new_seq_len, num_channels)
        
        # 移除批次维度
        samples = samples.squeeze(0)  # (n_samples, new_seq_len, num_channels)
        
        # 转换为numpy数组
        densified_samples = samples.cpu().numpy()
        
        # 计算平均填补结果
        result = np.mean(densified_samples, axis=0)

        # 保留原始数据点
        result[::densification_factor] = extended_data[::densification_factor]

        return result, new_seq_len, densified_samples

def plot_multiple_channels(original_data, densified_data, channel_indices, original_time, densified_time, save_path):
    """
    绘制多个通道的对比图
    """
    n_channels = len(channel_indices)
    fig, axes = plt.subplots(n_channels, 1, figsize=(14, 3*n_channels))
    
    if n_channels == 1:
        axes = [axes]
    
    for i, channel_idx in enumerate(channel_indices):
        ax = axes[i]
        
        # 绘制原始数据点
        ax.plot(original_time, original_data[:, channel_idx], 'bo-', 
                label='Original Data', linewidth=1.5, markersize=4, alpha=0.7)
        
        # 绘制密度增强后的数据
        ax.plot(densified_time, densified_data[:, channel_idx], 'r-', 
                label='Densified Data', linewidth=1.2, alpha=0.8)
        
        # 标记插值点
        interpolated_indices = [i for i in range(len(densified_time)) if i % 2 != 0]
        ax.scatter(densified_time[interpolated_indices], densified_data[interpolated_indices, channel_idx], 
                  color='red', marker='x', s=40, label='Interpolated Points', zorder=5)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'Channel {channel_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_error_analysis(original_data, densified_data, densified_time, save_path):
    """
    绘制误差分析图
    """
    # 计算插值点的误差（使用线性插值作为基准）
    original_positions = np.arange(0, len(densified_time), 2)
    interpolated_positions = np.arange(1, len(densified_time), 2)
    
    # 线性插值作为对比
    linear_interp = np.zeros_like(densified_data)
    linear_interp[original_positions] = original_data
    
    for channel in range(original_data.shape[1]):
        for i in interpolated_positions:
            prev_idx = i - 1
            next_idx = i + 1
            linear_interp[i, channel] = (linear_interp[prev_idx, channel] + linear_interp[next_idx, channel]) / 2
    
    # 计算CSDI插值和线性插值的误差
    csdi_errors = np.abs(linear_interp[interpolated_positions] - densified_data[interpolated_positions])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 误差分布直方图
    axes[0, 0].hist(csdi_errors.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Absolute Error')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of CSDI Interpolation Errors')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 通道平均误差
    channel_errors = np.mean(csdi_errors, axis=0)
    axes[0, 1].bar(range(len(channel_errors)), channel_errors, color='skyblue', edgecolor='black')
    axes[0, 1].set_xlabel('Channel Index')
    axes[0, 1].set_ylabel('Mean Absolute Error')
    axes[0, 1].set_title('Mean Error per Channel')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 时间点平均误差
    time_errors = np.mean(csdi_errors, axis=1)
    axes[1, 0].plot(densified_time[interpolated_positions], time_errors, 'o-', color='green')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Mean Absolute Error')
    axes[1, 0].set_title('Error Variation Over Time')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 误差热力图
    im = axes[1, 1].imshow(csdi_errors.T, aspect='auto', cmap='hot', interpolation='nearest')
    axes[1, 1].set_xlabel('Time Point Index')
    axes[1, 1].set_ylabel('Channel Index')
    axes[1, 1].set_title('Error Heatmap (Channel vs Time)')
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_spectral_analysis(original_data, densified_data, original_time, densified_time, save_path):
    """
    绘制频谱分析图
    """
    # 选择几个代表性通道进行分析
    channel_indices = [0, 16, 32, 48]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, channel_idx in enumerate(channel_indices):
        if i >= len(axes):
            break
            
        # 计算原始数据和密度增强数据的功率谱
        fs_original = 1.0 / (original_time[1] - original_time[0])  # 原始采样率
        fs_densified = 1.0 / (densified_time[1] - densified_time[0])  # 密度增强后采样率
        
        f_original, Pxx_original = signal.welch(original_data[:, channel_idx], fs_original, nperseg=min(32, len(original_data)))
        f_densified, Pxx_densified = signal.welch(densified_data[:, channel_idx], fs_densified, nperseg=min(64, len(densified_data)))
        
        axes[i].semilogy(f_original, Pxx_original, 'b-', label='Original', linewidth=2)
        axes[i].semilogy(f_densified, Pxx_densified, 'r--', label='Densified', linewidth=2)
        axes[i].set_xlabel('Frequency [Hz]')
        axes[i].set_ylabel('Power Spectral Density')
        axes[i].set_title(f'Power Spectrum - Channel {channel_idx}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Test CSDI model for EEG densification")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model.pth")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to EEG data file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of imputation samples")
    parser.add_argument("--seq_len", type=int, default=100, help="Original sequence length")
    parser.add_argument("--channel_idx", type=int, default=0, help="Channel index to visualize")
    parser.add_argument("--densification_factor", type=int, default=2, help="How many times to increase density")
    
    args = parser.parse_args()
    
    # 加载模型
    print("正在加载模型...")
    model = load_model(args.model_path, args.config_path, args.device)
    
    # 加载EEG数据
    print("正在加载EEG数据...")
    test_data, mean, std = load_eeg_data(args.data_path, args.seq_len)
    
    # 使用模型进行密度增强
    print("正在进行数据密度增强...")
    densified_data, new_seq_len, all_samples = densify_eeg(
        model,
        test_data,
        device=args.device,
        n_samples=args.n_samples,
        densification_factor=args.densification_factor
    )
    
    # 反标准化
    densified_data = densified_data * std + mean
    test_data = test_data * std + mean
    all_samples = all_samples * std + mean  # 所有样本也反标准化
    
    # 创建时间索引
    original_time = np.arange(args.seq_len)
    densified_time = np.linspace(0, args.seq_len-1, new_seq_len)
    
    # 创建结果目录
    os.makedirs("./results", exist_ok=True)
    
    # 可视化1: 单个通道详细对比
    print("生成单个通道详细对比图...")
    channel_idx = args.channel_idx
    plt.figure(figsize=(14, 8))
    
    # 绘制原始数据点
    plt.plot(original_time, test_data[:, channel_idx], 'bo-', label='Original Data', 
             linewidth=1.5, markersize=6, alpha=0.7)
    
    # 绘制密度增强后的数据
    plt.plot(densified_time, densified_data[:, channel_idx], 'r-', label='Densified Data', 
             linewidth=1.2, alpha=0.8)
    
    # 标记原始数据点
    plt.scatter(original_time, test_data[:, channel_idx], 
                color='blue', s=40, label='Original Points', zorder=5)
    
    # 标记插值点
    interpolated_indices = [i for i in range(new_seq_len) if i % args.densification_factor != 0]
    plt.scatter(densified_time[interpolated_indices], densified_data[interpolated_indices, channel_idx], 
                color='red', marker='x', s=60, label='Interpolated Points', zorder=5)
    
    plt.xlabel('Time Step')
    plt.ylabel(f'Value (Channel {channel_idx})')
    plt.title(f'EEG Data Densification: {args.seq_len} → {new_seq_len} points (Channel {channel_idx})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"./results/eeg_densification_channel_{channel_idx}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 可视化2: 多个通道对比
    print("生成多个通道对比图...")
    channel_indices = [0, 1, 2, 3]  # 选择前4个通道进行对比
    plot_multiple_channels(
        test_data, densified_data, channel_indices, 
        original_time, densified_time,
        "./results/eeg_densification_multiple_channels.png"
    )
    
    # 可视化3: 误差分析
    print("生成误差分析图...")
    plot_error_analysis(
        test_data, densified_data, densified_time,
        "./results/eeg_densification_error_analysis.png"
    )
    
    # 可视化4: 频谱分析
    print("生成频谱分析图...")
    plot_spectral_analysis(
        test_data, densified_data, original_time, densified_time,
        "./results/eeg_densification_spectral_analysis.png"
    )
    
    # 可视化5: 不确定性分析（如果n_samples > 1）
    if args.n_samples > 1:
        print("生成不确定性分析图...")
        plt.figure(figsize=(14, 8))
        
        # 计算每个时间点的均值和标准差
        mean_vals = np.mean(all_samples, axis=0)
        std_vals = np.std(all_samples, axis=0)
        
        # 绘制均值线
        plt.plot(densified_time, mean_vals[:, channel_idx], 'r-', label='Mean', linewidth=2)
        
        # 绘制不确定性区域（±1标准差）
        plt.fill_between(
            densified_time, 
            mean_vals[:, channel_idx] - std_vals[:, channel_idx],
            mean_vals[:, channel_idx] + std_vals[:, channel_idx],
            alpha=0.3, color='red', label='±1 Std Dev'
        )
        
        # 绘制原始数据点
        plt.scatter(original_time, test_data[:, channel_idx], 
                   color='blue', s=50, label='Original Points', zorder=5)
        
        plt.xlabel('Time Step')
        plt.ylabel(f'Value (Channel {channel_idx})')
        plt.title(f'CSDI Uncertainty Analysis (Channel {channel_idx}, {args.n_samples} samples)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f"./results/eeg_densification_uncertainty_channel_{channel_idx}.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    # 打印统计信息
    print(f"\n通道 {channel_idx} 的统计信息:")
    print(f"原始序列长度: {args.seq_len}")
    print(f"密度增强后序列长度: {new_seq_len}")
    print(f"数据密度增加: {(new_seq_len/args.seq_len - 1)*100:.1f}%")
    print(f"原始序列范围: [{test_data[:, channel_idx].min():.4f}, {test_data[:, channel_idx].max():.4f}]")
    print(f"密度增强后序列范围: [{densified_data[:, channel_idx].min():.4f}, {densified_data[:, channel_idx].max():.4f}]")
    
    # 计算原始点位置的值差异（应该很小，因为我们保留了原始点）
    original_positions = np.arange(0, new_seq_len, args.densification_factor)
    original_differences = np.abs(test_data[:, channel_idx] - densified_data[original_positions, channel_idx])
    print(f"原始点位置的平均绝对差异: {np.mean(original_differences):.6f}")
    
    # 保存结果
    np.savez("./results/eeg_densification_results.npz", 
             original=test_data, 
             densified=densified_data,
             all_samples=all_samples,
             original_time=original_time,
             densified_time=densified_time)
    
    print(f"\n结果已保存至 ./results/eeg_densification_results.npz")
    print(f"原始数据形状: {test_data.shape}")
    print(f"密度增强后数据形状: {densified_data.shape}")
    print(f"所有样本形状: {all_samples.shape}")
    
    # ============ 新增：图片保存信息汇总 ============
    print(f"\n=== 图片保存汇总 ===")
    print(f"✅ 单个通道详细对比图: ./results/eeg_densification_channel_{channel_idx}.png")
    print(f"✅ 多通道对比图: ./results/eeg_densification_multiple_channels.png")
    print(f"✅ 误差分析图: ./results/eeg_densification_error_analysis.png")
    print(f"✅ 频谱分析图: ./results/eeg_densification_spectral_analysis.png")
    
    if args.n_samples > 1:
        print(f"✅ 不确定性分析图: ./results/eeg_densification_uncertainty_channel_{channel_idx}.png")
    
    # 检查所有图片是否成功保存
    print(f"\n=== 图片保存状态检查 ===")
    saved_images = []
    expected_images = [
        f"./results/eeg_densification_channel_{channel_idx}.png",
        "./results/eeg_densification_multiple_channels.png",
        "./results/eeg_densification_error_analysis.png",
        "./results/eeg_densification_spectral_analysis.png"
    ]
    
    if args.n_samples > 1:
        expected_images.append(f"./results/eeg_densification_uncertainty_channel_{channel_idx}.png")
    
    for img_path in expected_images:
        if os.path.exists(img_path):
            file_size = os.path.getsize(img_path) / 1024  # KB
            status = "✅ 成功保存"
            saved_images.append(img_path)
        else:
            status = "❌ 保存失败"
        print(f"{os.path.basename(img_path)}: {status} ({file_size:.1f} KB)" if 'file_size' in locals() else f"{os.path.basename(img_path)}: {status}")
    
    print(f"\n📊 总计生成图片: {len(saved_images)}/{len(expected_images)}")
    print(f"📁 图片保存目录: ./results/")
    
    # ============ 新增代码结束 ============

if __name__ == "__main__":
    main()