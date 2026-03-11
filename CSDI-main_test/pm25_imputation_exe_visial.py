# -*- coding: utf-8 -*-这是训练部分代码的可视化版本，用于展示PM2.5
#使用方法/home/rhl/CSDI-main_test/save/pm25_imputation_split0.5_train0.7_valid0.15_mask0.1_seed42_20260127_142505
#python pm25_imputation_exe_visial.py --folder "/home/rhl/CSDI-main_test/save/pm25_imputation_split0.5_train0.7_valid0.15_mask0.1_seed42_20260127_151505" --idx 0
import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os

def get_quantile(samples, q, dim=1):
    """计算分位数"""
    return torch.quantile(samples, q, dim=dim).cpu().numpy()

def load_data(foldername, nsample, mean_std_path='data/pm25/pm25_meanstd.pk'):
    """加载模型生成的数据并进行反归一化处理"""
    
    # 1. 构建文件路径
    path = os.path.join(foldername, f'generated_outputs_nsample{nsample}.pk')
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}\n请检查 nsample 参数或确认 evaluate 是否成功保存了 .pk 文件。")

    print(f"正在加载数据: {path}")
    with open(path, 'rb') as f:
        samples, all_target, all_evalpoint, all_observed, all_observed_time, scaler, mean_scaler = pickle.load(f)

    # 转换为 Numpy 并移到 CPU
    all_target_np = all_target.cpu().numpy()
    all_evalpoint_np = all_evalpoint.cpu().numpy()
    all_observed_np = all_observed.cpu().numpy()
    
    # 计算 "Given" (输入给模型的观测值) = 总观测 - 待评估点
    all_given_np = all_observed_np - all_evalpoint_np

    # 2. 加载 PM2.5 的均值和方差进行反归一化 (还原为真实数值)
    # 如果找不到均值文件，尝试使用 scaler (如果是数值的话)
    if os.path.exists(mean_std_path):
        print(f"加载归一化参数: {mean_std_path}")
        with open(mean_std_path, 'rb') as f:
            train_mean, train_std = pickle.load(f)
            
        # 扩展维度以匹配数据
        # target: (Batch, Time, Feature) -> 均值需要是 (Feature,) 或 (1, 1, Feature)
        # 这里假设 train_mean 是 (Feature,) 维度
        all_target_np = (all_target_np * train_std + train_mean)
        
        # Samples: (Batch, Nsample, Time, Feature)
        # 需要将 tensor 转为 cpu 进行计算
        train_std_tensor = torch.from_numpy(train_std).float().cpu()
        train_mean_tensor = torch.from_numpy(train_mean).float().cpu()
        samples = samples.cpu() * train_std_tensor + train_mean_tensor
    else:
        print("警告: 未找到 pm25_meanstd.pk，将尝试使用 evaluate 中保存的 scaler 进行反归一化")
        # 简单的回退处理
        if isinstance(scaler, (int, float)) and scaler != 1:
             samples = samples.cpu() * scaler + mean_scaler
             all_target_np = all_target_np * scaler + mean_scaler

    return samples, all_target_np, all_evalpoint_np, all_given_np

def plot_all_features(samples, target, evalpoint, given, sample_idx, save_dir):
    """
    绘制指定样本的所有特征（站点）
    类似 Notebook 中的 9x4 网格图
    """
    K = samples.shape[-1] # Feature 数量 (通常是 36)
    L = samples.shape[-2] # Time 长度
    
    # 计算分位数 (Batch, Time, Feature)
    qlist = [0.05, 0.25, 0.5, 0.75, 0.95]
    quantiles_imp = []
    
    # 这里的逻辑是：如果是 given 的点，就用真实值；如果是缺失点，用生成的分位数
    # 注意：samples 已经是反归一化后的 tensor
    for q in qlist:
        imp = get_quantile(samples, q, dim=1) # (Batch, Time, Feature)
        # 组合: 观测部分用真实值，缺失部分用预测值
        final = imp * (1 - given) + target * given
        quantiles_imp.append(final)

    # 设置绘图
    plt.rcParams["font.size"] = 12
    # 自动计算行列: 假设 K=36, 9行4列
    ncols = 4
    nrows = (K + ncols - 1) // ncols 
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 4 * nrows))
    # 展平 axes 方便遍历，处理多余的子图
    axes_flat = axes.flatten()

    for k in range(K):
        ax = axes_flat[k]
        
        # 准备数据点 (用于散点图)
        # 1. 待评估点 (Ground Truth for Missing) - 蓝色圆点
        df_eval = pd.DataFrame({"x": np.arange(0, L), "val": target[sample_idx, :, k], "y": evalpoint[sample_idx, :, k]})
        df_eval = df_eval[df_eval.y != 0]
        
        # 2. 输入观测点 (Observed Input) - 红色叉号
        df_given = pd.DataFrame({"x": np.arange(0, L), "val": target[sample_idx, :, k], "y": given[sample_idx, :, k]})
        df_given = df_given[df_given.y != 0]

        # 绘图逻辑
        # 绘制中位数 (CSDI 预测) - 绿色实线
        ax.plot(range(0, L), quantiles_imp[2][sample_idx, :, k], color='g', linestyle='solid', label='CSDI Pred')
        
        # 绘制置信区间 (5% - 95%) - 绿色阴影
        ax.fill_between(range(0, L), 
                        quantiles_imp[0][sample_idx, :, k], 
                        quantiles_imp[4][sample_idx, :, k],
                        color='g', alpha=0.3, label='90% Conf')
        
        # 绘制真实值点
        ax.plot(df_eval.x, df_eval.val, color='b', marker='o', linestyle='None', markersize=4, label='Target (GT)')
        ax.plot(df_given.x, df_given.val, color='r', marker='x', linestyle='None', markersize=6, label='Observed')

        ax.set_title(f"Station {k}")
        
        # 只在第一列显示 Y 轴标签，最后一行显示 X 轴标签
        if k % ncols == 0:
            ax.set_ylabel('PM2.5 Value')
        if k // ncols == nrows - 1:
            ax.set_xlabel('Time')
            
        # 图例只在第一个图显示，避免杂乱
        if k == 0:
            ax.legend(loc='upper right', fontsize='small')

    # 删除多余的空子图
    for k in range(K, len(axes_flat)):
        fig.delaxes(axes_flat[k])

    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(save_dir, f"vis_sample_{sample_idx}_all_stations.png")
    plt.savefig(save_path)
    print(f"图表已保存: {save_path}")
    # plt.show() # 如果在服务器运行，请注释掉这一行
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize CSDI PM2.5 Results")
    
    parser.add_argument("--folder", type=str, required=True, help="模型保存的文件夹路径 (例如 ./save/pm25_...)")
    parser.add_argument("--nsample", type=int, default=100, help="生成时的采样次数 (默认: 100)")
    parser.add_argument("--idx", type=int, default=0, help="要可视化的测试集样本索引 (默认: 0)")
    parser.add_argument("--meanstd_path", type=str, default="data/pm25/pm25_meanstd.pk", help="PM2.5均值方差文件路径")

    args = parser.parse_args()

    # 运行可视化
    try:
        samples, target, evalpoint, given = load_data(args.folder, args.nsample, args.meanstd_path)
        plot_all_features(samples, target, evalpoint, given, args.idx, args.folder)
    except Exception as e:
        print(f"发生错误: {e}")
