import numpy as np
import matplotlib.pyplot as plt
from dataset_lorenz import generate_coupled_lorenz
from test import impute, load_model
from test2 import predict
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具

def main():
    # --- Parameters ---
    # Data generation - 数据生成参数
    lorenz_N = 5          # 洛伦兹系统的节点数量
    lorenz_L = 50         # 生成的时间序列长度
    lorenz_stepsize = 8   # 采样步长（每隔多少步采样一次）

    # Model and config paths - 模型路径
    model_path = 'save/model.pth'      # 训练好的CSDI模型路径
    config_path = 'config/lorenz.yaml' # 模型配置文件路径

    # Prediction 1 (Original data) - 原始数据预测参数
    trainlength1 = 20      # 减少训练序列长度，确保有足够的数据进行预测
    steps_ahead1 = 1       # 预测步长（预测未来多少步）
    predict_L = 4          # 嵌入维度（用于预测的特征组合维度）
    predict_s = 10         # 随机组合数量（用于集成预测）
    predict_n_jobs = 4     # 并行计算进程数

    # Prediction 2 (Imputed data) - 补全数据预测参数
    trainlength2_multiplier = 1.5  # 减少训练长度倍增因子
    steps_ahead2 = 2             # 预测步长（可以预测更远的未来）

    # --- Code ---

    # Generate Lorenz data - 生成耦合洛伦兹系统数据
    lorenz_data, full_data = generate_coupled_lorenz(
        N=lorenz_N, L=lorenz_L, stepsize=lorenz_stepsize
    )
    print(f"Generated lorenz_data shape: {lorenz_data.shape}")  # 采样后的稀疏数据
    print(f"Generated full_data shape: {full_data.shape}")      # 完整的高分辨率数据

    # 先进行数据补全（只需要一次）
    model = load_model(model_path, config_path)  # 加载预训练模型
    lorenz_data_imputed = impute(model, lorenz_data)  # 对稀疏数据进行补全
    print(f"Imputed data shape: {lorenz_data_imputed.shape}")

    # 计算第二个预测的训练长度
    trainlength2 = int(trainlength1 * trainlength2_multiplier)
    
    # 确保训练长度不超过数据长度
    if trainlength2 >= lorenz_data_imputed.shape[0]:
        trainlength2 = lorenz_data_imputed.shape[0] - 5  # 保留至少5个点用于预测
        print(f"Adjusted trainlength2 to {trainlength2}")

    # 初始化结果数组
    result1 = np.zeros((lorenz_data.shape[0] - trainlength1, 3))
    result2 = np.zeros((lorenz_data_imputed.shape[0] - trainlength2, 3))

    # 对每个维度进行预测
    for dim in range(3):
        # Prediction 1: Original data - 使用原始稀疏数据进行预测
        result1_dim = predict(
            seq=lorenz_data,           # 输入序列（稀疏采样）
            trainlength=trainlength1,  # 训练长度
            L=predict_L,               # 嵌入维度
            s=predict_s,               # 随机组合数
            j=dim,                     # 目标维度
            n_jobs=predict_n_jobs,     # 并行数
            steps_ahead=steps_ahead1   # 预测步长
        )[0]  # 取第一个返回值（预测结果）
        result1[:, dim] = result1_dim
        
        # Prediction 2: Imputed data - 使用补全后的数据进行预测
        result2_dim = predict(
            seq=lorenz_data_imputed,   # 输入序列（补全后的数据）
            trainlength=trainlength2,  # 训练长度
            L=predict_L,               # 相同的嵌入维度
            s=predict_s,               # 相同的随机组合数
            j=dim,                     # 相同的目标维度
            n_jobs=predict_n_jobs,     # 相同的并行数
            steps_ahead=steps_ahead2   # 预测步长
        )[0]
        result2[:, dim] = result2_dim

    print(f"Prediction 1 (original) result shape: {result1.shape}")
    print(f"Prediction 2 (imputed) result shape: {result2.shape}")

    # 获取预测时间点索引
    pred_start_idx1 = trainlength1
    pred_end_idx1 = pred_start_idx1 + len(result1)
    pred_time_indices1 = np.arange(0, full_data.shape[0], lorenz_stepsize)[pred_start_idx1 : pred_end_idx1]
    
    pred_start_idx2 = trainlength2
    pred_end_idx2 = pred_start_idx2 + len(result2)
    
    # 获取真实值
    ground_truth1 = lorenz_data[pred_start_idx1:pred_end_idx1, :3]
    ground_truth2 = lorenz_data_imputed[pred_start_idx2:pred_end_idx2, :3]
    
    # 计算误差
    error1 = result1 - ground_truth1
    error2 = result2 - ground_truth2
    
    max_error1 = np.max(np.abs(error1), axis=0)
    rms_error1 = np.sqrt(np.mean(error1**2, axis=0))
    
    max_error2 = np.max(np.abs(error2), axis=0)
    rms_error2 = np.sqrt(np.mean(error2**2, axis=0))
    
    # 打印误差结果
    print("\n--- Error Analysis ---")
    for dim in range(3):
        print(f"Dimension {dim}:")
        print(f"  Prediction 1 (Original):")
        print(f"    Max Error: {max_error1[dim]:.6f}")
        print(f"    RMS Error: {rms_error1[dim]:.6f}")
        print(f"  Prediction 2 (Imputed):")
        print(f"    Max Error: {max_error2[dim]:.6f}")
        print(f"    RMS Error: {rms_error2[dim]:.6f}")
        print()

    # --- 绘制相图比较 ---
    fig = plt.figure(figsize=(15, 10))
    
    # 第一个子图：原始数据和预测1
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    # 绘制完整洛伦兹吸引子
    ax1.plot(full_data[:, 0], full_data[:, 1], full_data[:, 2], 'g-', alpha=0.3, label='Full Lorenz Attractor')
    # 绘制稀疏采样点
    ax1.scatter(lorenz_data[:, 0], lorenz_data[:, 1], lorenz_data[:, 2], 
               c='blue', marker='o', label='Sampled Data')
    # 绘制预测1结果
    ax1.scatter(result1[:, 0], result1[:, 1], result1[:, 2], 
               c='red', marker='x', s=50, label='Prediction 1 (Original)')
    ax1.set_title('Original Data and Prediction 1')
    ax1.legend()
    
    # 第二个子图：补全数据和预测2
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.plot(full_data[:, 0], full_data[:, 1], full_data[:, 2], 'g-', alpha=0.3, label='Full Lorenz Attractor')
    # 绘制补全数据点
    ax2.scatter(lorenz_data_imputed[:, 0], lorenz_data_imputed[:, 1], lorenz_data_imputed[:, 2], 
               c='cyan', marker='o', alpha=0.5, label='Imputed Data')
    # 绘制预测2结果
    ax2.scatter(result2[:, 0], result2[:, 1], result2[:, 2], 
               c='magenta', marker='+', s=50, label='Prediction 2 (Imputed)')
    ax2.set_title('Imputed Data and Prediction 2')
    ax2.legend()
    
    # 第三个子图：两个预测结果对比
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.plot(full_data[:, 0], full_data[:, 1], full_data[:, 2], 'g-', alpha=0.3, label='Full Lorenz Attractor')
    ax3.scatter(result1[:, 0], result1[:, 1], result1[:, 2], 
               c='red', marker='x', s=50, label='Prediction 1 (Original)')
    ax3.scatter(result2[:, 0], result2[:, 1], result2[:, 2], 
               c='magenta', marker='+', s=50, label='Prediction 2 (Imputed)')
    ax3.set_title('Comparison of Both Predictions')
    ax3.legend()
    
    # 第四个子图：误差比较
    ax4 = fig.add_subplot(2, 2, 4)
    dimensions = ['Dim 0', 'Dim 1', 'Dim 2']
    x = np.arange(len(dimensions))
    width = 0.35
    
    ax4.bar(x - width/2, rms_error1, width, label='Prediction 1 (Original)')
    ax4.bar(x + width/2, rms_error2, width, label='Prediction 2 (Imputed)')
    ax4.set_xlabel('Dimension')
    ax4.set_ylabel('RMS Error')
    ax4.set_title('RMS Error by Dimension')
    ax4.set_xticks(x)
    ax4.set_xticklabels(dimensions)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('phase_space_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()