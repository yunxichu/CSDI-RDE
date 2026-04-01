#!/usr/bin/env python3
# test.py - Use trained CSDI model for actual imputation 生成洛伦兹系统并csdi补值
import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csdi_dir = os.path.join(base_dir, 'CSDI-GPR_Lorenz_fullpy')
datasets_dir = os.path.join(base_dir, 'datasets')
sys.path.insert(0, base_dir)
sys.path.insert(0, csdi_dir)
sys.path.insert(0, datasets_dir)
os.chdir(csdi_dir)

import torch
import numpy as np
import yaml
import argparse
import matplotlib.pyplot as plt
from main_model import CSDI_Lorenz
from dataset_lorenz import generate_coupled_lorenz

def load_model(model_path, config_path, device='cuda:0'):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = CSDI_Lorenz(config, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

# 数据填补函数
def impute(model, partial_data, device='cuda:0', n_samples=10): 
    # 准备数据
    partial_len, num_features = partial_data.shape  # 获取部分数据的长度和特征数
    seq_len = partial_len * 2  # 完整序列长度是部分数据的两倍（每隔一个点缺失）
    data = np.zeros((seq_len, num_features))  # 创建全零数组作为初始数据
    data[1::2] = partial_data  # 将部分数据填充到奇数索引位置（偶数索引位置为缺失值）
    known_mask = np.zeros_like(data)  # 创建与数据形状相同的全零掩码
    known_mask[1::2] = 1  # 将已知数据位置标记为1
    
    # Convert to tensors and add batch dimension
    observed_data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, num_features)
    observed_mask = torch.tensor(known_mask, dtype=torch.float32).unsqueeze(0).to(device)   # (1, seq_len, num_features)
    cond_mask = observed_mask.clone()  # Use known positions as conditioning
    observed_tp = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len)
    
    # Reshape for model (model expects (batch, features, time))
    observed_data = observed_data.permute(0, 2, 1)  # (1, num_features, seq_len)
    observed_mask = observed_mask.permute(0, 2, 1)  # (1, num_features, seq_len)
    cond_mask = cond_mask.permute(0, 2, 1)          # (1, num_features, seq_len)
    
    with torch.no_grad():
        print("Generating side information...")
        # Get side information
        side_info = model.get_side_info(observed_tp, cond_mask)
        
        print(f"Generating {n_samples} imputation samples...")
        # Generate imputed samples
        samples = model.impute(observed_data, cond_mask, side_info, n_samples)
        
        # Convert back to (batch, time, features) format
        samples = samples.permute(0, 1, 3, 2)  # (1, n_samples, seq_len, num_features)
        
        # Remove batch dimension
        samples = samples.squeeze(0)  # (n_samples, seq_len, num_features)
        
        # Convert to numpy
        imputed_samples = samples.cpu().numpy()
        
        # Calculate mean imputation
        result = np.mean(imputed_samples, axis=0)

        result[1::2] = partial_data

        return result

def main():
    parser = argparse.ArgumentParser(description="Test CSDI model for imputation")
    parser.add_argument("--model_path", type=str, default="save/model.pth", help="Path to saved model.pth")
    parser.add_argument("--config_path", type=str, default="config/lorenz.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of imputation samples")
    
    args = parser.parse_args()
    
    model = load_model(args.model_path, args.config_path, args.device)
    
    # Create test data
    full_sequence, _ = generate_coupled_lorenz(N=5, L=100, stepsize=4)  # (100, 15)
    partial_data = full_sequence[1::2]
    
    result = impute(
        model,
        partial_data,
        device=args.device,
        n_samples=args.n_samples
    )

    print(result)

    # Plot comparison of full_sequence[:,0] and result[:,0]
    plt.figure(figsize=(12, 6))
    
    # Create time indices
    time_full = np.arange(len(full_sequence))
    time_result = np.arange(len(result))
    
    # Plot original full sequence (first dimension)
    plt.plot(time_full, full_sequence[:, 0], 'b-', label='Original Full Sequence', linewidth=2)
    
    # Plot imputed result (first dimension)
    plt.plot(time_result, result[:, 0], 'r--', label='Imputed Result', linewidth=2)
    
    # Add markers for the known data points (every other point in result)
    known_indices = np.arange(1, len(result), 2)
    plt.scatter(known_indices, result[known_indices, 0], color='green', s=50, 
                label='Known Data Points', zorder=5)
    
    plt.xlabel('Time Step')
    plt.ylabel('Value (First Dimension)')
    plt.title('Comparison: Original vs Imputed Sequence (First Dimension)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print some statistics for comparison
    print(f"\nStatistics for first dimension:")
    print(f"Original sequence range: [{full_sequence[:, 0].min():.4f}, {full_sequence[:, 0].max():.4f}]")
    print(f"Imputed result range: [{result[:, 0].min():.4f}, {result[:, 0].max():.4f}]")
    print(f"Mean absolute difference: {np.mean(np.abs(full_sequence[:, 0] - result[:, 0])):.4f}")

if __name__ == "__main__":
    main()