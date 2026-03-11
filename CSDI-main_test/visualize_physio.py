import argparse
import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from main_model import CSDI_Physio
from dataset_physio import get_dataloader

# PhysioNet数据集的35个生理指标名称
attributes = ['DiasABP', 'HR', 'Na', 'Lactate', 'NIDiasABP', 'PaO2', 'WBC', 'pH', 'Albumin', 'ALT', 'Glucose', 'SaO2',
              'Temp', 'AST', 'Bilirubin', 'HCO3', 'BUN', 'RespRate', 'Mg', 'HCT', 'SysABP', 'FiO2', 'K', 'GCS',
              'Cholesterol', 'NISysABP', 'TroponinT', 'MAP', 'TroponinI', 'PaCO2', 'Platelets', 'Urine', 'NIMAP',
              'Creatinine', 'ALP']

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize CSDI Physio Results")
    parser.add_argument("--path", type=str, required=True, help="模型保存文件夹路径 (e.g., save/physio_fold0_...)")
    parser.add_argument("--device", default="cuda:0", help="运行设备")
    parser.add_argument("--nsample", type=int, default=100, help="采样次数")
    parser.add_argument("--batch_idx", type=int, default=0, help="要可视化的batch索引")
    parser.add_argument("--save_png", action="store_true", help="保存为PNG图片")
    return parser.parse_args()

def visualize(args):
    # 1. 加载配置文件
    config_path = os.path.join(args.path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    print(f"Loaded config from {config_path}")
    
    # 2. 加载数据（保持与训练时一致的参数）
    print("Loading test data...")
    # 从config读取训练时使用的参数
    seed = config.get("seed", 1)  # 如果config里没有seed，默认用1
    nfold = config.get("nfold", 0)
    missing_ratio = config["model"]["test_missing_ratio"]
    
    _, _, test_loader = get_dataloader(
        seed=seed,  # 使用与训练一致的seed
        nfold=nfold,
        batch_size=1,  # batch_size设为1便于可视化
        missing_ratio=missing_ratio
    )
    
    # 获取指定的batch
    for i, batch in enumerate(test_loader):
        if i == args.batch_idx:
            break
    
    print(f"Visualizing batch {args.batch_idx}")
    
    # 3. 加载模型
    print("Loading model...")
    model = CSDI_Physio(config, args.device).to(args.device)
    model_path = os.path.join(args.path, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model.eval()
    
    # 4. 执行推理
    print("Running imputation...")
    with torch.no_grad():
        # evaluate返回: samples, observed_data, target_mask, observed_mask, observed_tp
        output = model.evaluate(batch, args.nsample)
        
        samples = output[0]  # (B, n_samples, K, L)
        observed_data = output[1]  # (B, K, L)
        target_mask = output[2]  # (B, K, L) - 需要预测的位置
        observed_mask = output[3]  # (B, K, L) - 实际有数据的位置
    
    # 5. 数据转换
    samples = samples[0].cpu().numpy()  # (n_samples, K, L)
    observed_data = observed_data[0].cpu().numpy()  # (K, L)
    target_mask = target_mask[0].cpu().numpy()  # (K, L)
    observed_mask = observed_mask[0].cpu().numpy()  # (K, L)
    
    # **关键修正**: 计算条件掩码（模型实际看到的输入）
    cond_mask = observed_mask - target_mask  # gt_mask
    
    # 计算统计量
    p50 = np.median(samples, axis=0)  # 中位数
    p05 = np.percentile(samples, 5, axis=0)  # 5%分位数
    p95 = np.percentile(samples, 95, axis=0)  # 95%分位数
    
    # 6. 绘图
    print("Plotting...")
    K = observed_data.shape[0]  # 35个特征
    L = observed_data.shape[1]  # 48小时
    time_x = np.arange(L)
    
    fig, axes = plt.subplots(5, 7, figsize=(24, 16))
    axes = axes.flatten()
    
    for k in range(K):
        ax = axes[k]
        
        # 绘制预测的置信区间 (灰色区域)
        ax.fill_between(time_x, p05[k, :], p95[k, :], 
                        color='lightblue', alpha=0.4, label="90% CI")
        
        # 绘制预测的中位数 (绿色线)
        ax.plot(time_x, p50[k, :], 
               color='green', linewidth=2, label="Imputed (Median)")
        
        # **修正**: 绘制模型看到的观测点 (红色圆点)
        obs_idx = cond_mask[k, :] == 1  # 使用cond_mask而不是observed_mask
        if np.sum(obs_idx) > 0:
            ax.scatter(time_x[obs_idx], observed_data[k, obs_idx], 
                      color='red', s=30, zorder=5, label="Input (Observed)", marker='o')
        
        # 绘制需要预测的真实值 (蓝色叉)
        target_idx = target_mask[k, :] == 1
        if np.sum(target_idx) > 0:
            ax.scatter(time_x[target_idx], observed_data[k, target_idx], 
                      color='blue', s=40, marker='x', linewidths=2, 
                      zorder=6, label="Target (Ground Truth)")
        
        ax.set_title(f"{attributes[k]}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time (hours)", fontsize=9)
        ax.set_ylabel("Normalized Valued Value", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, L-1)
        
        # 只在第一个子图显示图例
        if k == 0:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.suptitle(f"CSDI Imputation Results - Batch {args.batch_idx}", 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if args.save_png:
        save_file = os.path.join(args.path, f"visualization_batch{args.batch_idx}.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save_file}")
    else:
        plt.show()
    
    # 7. 打印统计信息
    print("\n" + "="*60)
    print("统计信息:")
    print(f"  - 观测点数量: {int(cond_mask.sum())} / {K*L} ({100*cond_mask.sum()/(K*L):.1f}%)")
    print(f"  - 预测点数量: {int(target_mask.sum())} / {K*L} ({100*target_mask.sum()/(K*L):.1f}%)")
    print("="*60)

if __name__ == "__main__":
    args = parse_args()
    visualize(args)
