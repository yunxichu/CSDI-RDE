#test_comb.py
import numpy as np
import matplotlib.pyplot as plt
from dataset_lorenz import generate_coupled_lorenz
from test import impute, load_model
from test2 import predict

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
    trainlength1 = 30      # 训练序列长度（用于预测的历史数据点数）
    steps_ahead1 = 1       # 预测步长（预测未来多少步）
    predict_L = 4          # 嵌入维度（用于预测的特征组合维度）
    predict_s = 10         # 随机组合数量（用于集成预测）
    predict_j = 0          # 目标变量索引（预测哪个维度，0表示第一个维度）
    predict_n_jobs = 4     # 并行计算进程数

    # Prediction 2 (Imputed data) - 补全数据预测参数
    trainlength2_multiplier = 2  # 训练长度倍增因子（补全后可用更多数据）
    steps_ahead2 = 2             # 预测步长（可以预测更远的未来）

    # --- Code ---

    # Generate Lorenz data - 生成耦合洛伦兹系统数据
    lorenz_data, full_data = generate_coupled_lorenz(
        N=lorenz_N, L=lorenz_L, stepsize=lorenz_stepsize
    )
    print(f"Generated lorenz_data shape: {lorenz_data.shape}")  # 采样后的稀疏数据
    print(f"Generated full_data shape: {full_data.shape}")      # 完整的高分辨率数据

    # Prediction 1: Original data - 使用原始稀疏数据进行预测
    result1 = predict(
        seq=lorenz_data,           # 输入序列（稀疏采样）
        trainlength=trainlength1,  # 训练长度
        L=predict_L,               # 嵌入维度
        s=predict_s,               # 随机组合数
        j=predict_j,               # 目标维度
        n_jobs=predict_n_jobs,     # 并行数
        steps_ahead=steps_ahead1   # 预测步长
    )[0]  # 取第一个返回值（预测结果）
    print(f"Prediction 1 (original) result shape: {result1.shape}")

    # Impute data - 使用CSDI模型进行数据补全
    model = load_model(model_path, config_path)  # 加载预训练模型
    lorenz_data_imputed = impute(model, lorenz_data)  # 对稀疏数据进行补全
    print(f"Imputed data shape: {lorenz_data_imputed.shape}")

    # Prediction 2: Imputed data - 使用补全后的数据进行预测
    trainlength2 = trainlength1 * trainlength2_multiplier + 1  # 计算新的训练长度
    result2 = predict(
        seq=lorenz_data_imputed,   # 输入序列（补全后的数据）
        trainlength=trainlength2,  # 更长的训练长度（因为数据更多）
        L=predict_L,               # 相同的嵌入维度
        s=predict_s,               # 相同的随机组合数
        j=predict_j,               # 相同的目标维度
        n_jobs=predict_n_jobs,     # 相同的并行数
        steps_ahead=steps_ahead2   # 更长的预测步长
    )[0]
    result2 = result2[::2]  # 下采样以匹配原始数据的时间分辨率
    print(f"Prediction 2 (imputed) result shape: {result2.shape}")

    # --- Plotting --- 可视化部分
    plt.figure(figsize=(15, 8))

    # 1. Plot full_data as a line - 绘制完整的高分辨率数据作为参考线
    full_data_time_indices = np.arange(full_data.shape[0])
    plt.plot(full_data_time_indices, full_data[:, 0], 'g-', label='Full Lorenz Data (dim 0)', alpha=0.6)

    # 2. Plot lorenz_data as points - 绘制原始稀疏采样数据点
    lorenz_data_time_indices = np.arange(0, full_data.shape[0], lorenz_stepsize)
    plt.plot(lorenz_data_time_indices, lorenz_data[:, 0], 'bo', label='Sampled Lorenz Data (dim 0)')
    
    # 2b. Plot imputed data points - 绘制补全的数据点
    imputed_points = lorenz_data_imputed[0::2, 0]
    imputed_time_indices = lorenz_data_time_indices - (lorenz_stepsize / 2.0)
    plt.plot(imputed_time_indices, imputed_points, 'g+', markersize=8, label='Imputed Data Points')

    # 3. Plot prediction 1 (original) - 绘制基于原始数据的预测结果
    pred_start_idx = trainlength1
    pred_end_idx = pred_start_idx + len(result1)
    pred_time_indices = lorenz_data_time_indices[pred_start_idx : pred_end_idx]
    plt.plot(pred_time_indices, result1, 'rx', markersize=8, label='Prediction 1 (Original)')

    # 4. Plot prediction 2 (imputed) - 绘制基于补全数据的预测结果

    plt.plot(pred_time_indices, result2, 'm+', markersize=10, label='Prediction 2 (Imputed)')

    # --- Error Calculation --- 误差计算
    # Ground truth data at prediction time indices - 获取真实值
    ground_truth = lorenz_data[pred_start_idx:pred_end_idx, 0]

    # Calculate errors for Prediction 1 (Original) - 计算原始数据预测误差
    error1 = result1 - ground_truth  # 预测误差
    max_error1 = np.max(np.abs(error1))  # 最大绝对误差
    rms_error1 = np.sqrt(np.mean(error1**2))  # 均方根误差

    # Calculate errors for Prediction 2 (Imputed) - 计算补全数据预测误差
    error2 = result2 - ground_truth
    max_error2 = np.max(np.abs(error2))
    rms_error2 = np.sqrt(np.mean(error2**2))
    
    # Print error results - 输出误差分析结果
    print("\n--- Error Analysis ---")
    print(f"Prediction 1 (Original):")
    print(f"  Max Error: {max_error1:.6f}")
    print(f"  RMS Error: {rms_error1:.6f}")
    print(f"Prediction 2 (Imputed):")
    print(f"  Max Error: {max_error2:.6f}")
    print(f"  RMS Error: {rms_error2:.6f}")
    
    plt.title('Lorenz Data and Predictions')
    plt.xlabel('Time Step')
    plt.ylabel('Value (Dimension 0)')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
