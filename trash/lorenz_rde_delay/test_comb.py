import os
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
lorenz_dir = os.path.join(base_dir, 'lorenz')
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(lorenz_dir, 'models'))
sys.path.insert(0, os.path.join(lorenz_dir, 'data'))
sys.path.insert(0, os.path.join(lorenz_dir, 'inference'))

import numpy as np
import matplotlib.pyplot as plt
from dataset_lorenz import generate_coupled_lorenz
import torch

# 直接导入模块，避免多进程 pickle 问题
from test import impute, load_model
from test2 import predict
import numpy as np
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def train_lstm(model, X_train, y_train, epochs=100, batch_size=32, learning_rate=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i in range(0, len(X_train), batch_size):
            batch_X = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32).to(device)
            batch_y = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32).to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}')
    return model

def lstm_predict(model, seq, trainlength=30, steps_ahead=1, target_idx=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    total_steps = len(seq) - trainlength
    predictions = []
    stds = []
    with torch.no_grad():
        for step in range(total_steps):
            traindata = seq[step:step + trainlength, :]
            X = traindata[:-steps_ahead].reshape(1, -1, traindata.shape[1])
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            pred = model(X_tensor).cpu().numpy()[0, 0]
            predictions.append(pred)
            stds.append(0.0)
    result = np.zeros((3, total_steps))
    result[0] = predictions
    result[1] = stds
    result[2] = seq[trainlength:, target_idx] - predictions
    return result


def main():
    # --- Parameters ---
    # Data generation - 数据生成参数
    lorenz_N = 5          # 洛伦兹系统的节点数量
    lorenz_L = 50         # 生成的时间序列长度
    lorenz_stepsize = 8   # 采样步长（每隔多少步采样一次）

    # Model and config paths - 模型路径
    model_path = os.path.join(lorenz_dir, 'save', 'model.pth')      # 训练好的CSDI模型路径
    config_path = os.path.join(lorenz_dir, 'config', 'lorenz.yaml') # 模型配置文件路径

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

    # Prediction 1: Original data - 使用原始稀疏数据进行GPR预测
    result1_full = predict(
        seq=lorenz_data,           # 输入序列（稀疏采样）
        trainlength=trainlength1,  # 训练长度
        L=predict_L,               # 嵌入维度
        s=predict_s,               # 随机组合数
        j=predict_j,               # 目标维度
        n_jobs=predict_n_jobs,     # 并行数
        steps_ahead=steps_ahead1   # 预测步长
    )
    # 提取预测值和不确定性
    result1 = result1_full[0]      # 预测值
    std1 = result1_full[1]         # 标准差（不确定性）
    print(f"Prediction 1 (GPR) result shape: {result1.shape}")
    print(f"Prediction 1 std shape: {std1.shape}")
    
    # Prediction 3: LSTM预测（作为baseline）
    print("\nTraining LSTM model...")
    # 准备LSTM训练数据
    input_size = lorenz_data.shape[1]
    hidden_size = 64
    num_layers = 2
    output_size = 1
    
    # 创建LSTM模型
    lstm_model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    
    # 准备训练数据
    X_train = []
    y_train = []
    for i in range(len(lorenz_data) - trainlength1):
        X_train.append(lorenz_data[i:i+trainlength1-1])
        y_train.append([lorenz_data[i+trainlength1-1, predict_j]])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # 训练LSTM模型
    lstm_model = train_lstm(lstm_model, X_train, y_train, epochs=50, batch_size=8)
    
    # 使用LSTM进行预测
    result3_full = lstm_predict(
        lstm_model,
        seq=lorenz_data,
        trainlength=trainlength1,
        steps_ahead=steps_ahead1,
        target_idx=predict_j
    )
    result3 = result3_full[0]      # 预测值
    std3 = result3_full[1]         # 标准差（LSTM不提供不确定性）
    print(f"Prediction 3 (LSTM) result shape: {result3.shape}")

    # Impute data - 使用CSDI模型进行数据补全
    lorenz_data_imputed = None
    result2 = None
    std2 = None
    
    if os.path.exists(model_path) and os.path.exists(config_path):
        try:
            model = load_model(model_path, config_path)  # 加载预训练模型
            lorenz_data_imputed = impute(model, lorenz_data)  # 对稀疏数据进行补全
            print(f"Imputed data shape: {lorenz_data_imputed.shape}")

            # Prediction 2: Imputed data - 使用补全后的数据进行预测
            trainlength2 = trainlength1 * trainlength2_multiplier + 1  # 计算新的训练长度
            result2_full = predict(
                seq=lorenz_data_imputed,   # 输入序列（补全后的数据）
                trainlength=trainlength2,  # 更长的训练长度（因为数据更多）
                L=predict_L,               # 相同的嵌入维度
                s=predict_s,               # 相同的随机组合数
                j=predict_j,               # 相同的目标维度
                n_jobs=predict_n_jobs,     # 相同的并行数
                steps_ahead=steps_ahead2   # 更长的预测步长
            )
            # 提取预测值和不确定性，并下采样
            result2 = result2_full[0][::2]  # 下采样预测值以匹配原始数据的时间分辨率
            std2 = result2_full[1][::2]     # 下采样标准差
            print(f"Prediction 2 (imputed) result shape: {result2.shape}")
            print(f"Prediction 2 std shape: {std2.shape}")
        except Exception as e:
            print(f"补值失败: {e}")
            print("将只进行原始数据的预测")
    else:
        print("模型文件或配置文件不存在，将只进行原始数据的预测")

    # --- Plotting with Uncertainty --- 带不确定性的可视化部分
    plt.figure(figsize=(15, 10))

    # 1. Plot full_data as a line - 绘制完整的高分辨率数据作为参考线
    full_data_time_indices = np.arange(full_data.shape[0])
    plt.plot(full_data_time_indices, full_data[:, 0], 'g-', label='Full Lorenz Data (dim 0)', alpha=0.6)

    # 2. Plot lorenz_data as points - 绘制原始稀疏采样数据点
    lorenz_data_time_indices = np.arange(0, full_data.shape[0], lorenz_stepsize)
    plt.plot(lorenz_data_time_indices, lorenz_data[:, 0], 'bo', label='Sampled Lorenz Data (dim 0)')

    # 2b. Plot imputed data points - 绘制补全的数据点
    if lorenz_data_imputed is not None:
        imputed_points = lorenz_data_imputed[0::2, 0]
        imputed_time_indices = lorenz_data_time_indices - (lorenz_stepsize / 2.0)
        plt.plot(imputed_time_indices, imputed_points, 'g+', markersize=8, label='Imputed Data Points')

    # 3. Plot prediction 1 (GPR) with uncertainty - 绘制基于原始数据的GPR预测结果和不确定性
    pred_start_idx = trainlength1
    pred_end_idx = pred_start_idx + len(result1)
    pred_time_indices = lorenz_data_time_indices[pred_start_idx : pred_end_idx]

    # GPR预测值
    plt.plot(pred_time_indices, result1, 'rx', markersize=8, label='Prediction 1 (GPR)', linewidth=2)

    # GPR不确定性区域 (±2σ)
    plt.fill_between(pred_time_indices,
                     result1 - 2*std1,
                     result1 + 2*std1,
                     alpha=0.3, color='red', label='GPR ±2σ uncertainty')

    # 4. Plot prediction 2 (imputed) with uncertainty - 绘制基于补全数据的预测结果和不确定性
    if result2 is not None and std2 is not None:
        plt.plot(pred_time_indices, result2, 'm+', markersize=10, label='Prediction 2 (Imputed GPR)', linewidth=2)

        # 不确定性区域 (±2σ)
        plt.fill_between(pred_time_indices,
                         result2 - 2*std2,
                         result2 + 2*std2,
                         alpha=0.3, color='magenta', label='Imputed GPR ±2σ uncertainty')
    
    # 5. Plot prediction 3 (LSTM) - 绘制LSTM预测结果
    plt.plot(pred_time_indices, result3, 'go', markersize=6, label='Prediction 3 (LSTM)', linewidth=2)

    # --- Error Calculation --- 误差计算
    # Ground truth data at prediction time indices - 获取真实值
    ground_truth = lorenz_data[pred_start_idx:pred_end_idx, 0]

    # Calculate errors for Prediction 1 (GPR) - 计算GPR预测误差
    error1 = result1 - ground_truth  # 预测误差
    max_error1 = np.max(np.abs(error1))  # 最大绝对误差
    rms_error1 = np.sqrt(np.mean(error1**2))  # 均方根误差
    mean_uncertainty1 = np.mean(std1)  # 平均不确定性

    # Calculate errors for Prediction 3 (LSTM) - 计算LSTM预测误差
    error3 = result3 - ground_truth
    max_error3 = np.max(np.abs(error3))
    rms_error3 = np.sqrt(np.mean(error3**2))

    # Print error results - 输出误差分析结果
    print("\n--- Error Analysis ---")
    print(f"Prediction 1 (GPR):")
    print(f"  Max Error: {max_error1:.6f}")
    print(f"  RMS Error: {rms_error1:.6f}")
    print(f"  Mean Uncertainty (±2σ): {2*mean_uncertainty1:.6f}")
    print(f"  Uncertainty Coverage: {np.sum(np.abs(error1) <= 2*std1) / len(error1) * 100:.1f}% within ±2σ")
    
    print(f"\nPrediction 3 (LSTM):")
    print(f"  Max Error: {max_error3:.6f}")
    print(f"  RMS Error: {rms_error3:.6f}")
    
    if result2 is not None and std2 is not None:
        # Calculate errors for Prediction 2 (Imputed GPR) - 计算补全数据预测误差
        error2 = result2 - ground_truth
        max_error2 = np.max(np.abs(error2))
        rms_error2 = np.sqrt(np.mean(error2**2))
        mean_uncertainty2 = np.mean(std2)  # 平均不确定性
        
        print(f"\nPrediction 2 (Imputed GPR):")
        print(f"  Max Error: {max_error2:.6f}")
        print(f"  RMS Error: {rms_error2:.6f}")
        print(f"  Mean Uncertainty (±1σ): {mean_uncertainty2:.6f}")
        print(f"  Uncertainty Coverage: {np.sum(np.abs(error2) <= std2) / len(error2) * 100:.1f}% within ±1σ")

    plt.title('Lorenz Data and Predictions with Uncertainty Quantification')
    plt.xlabel('Time Step')
    plt.ylabel('Value (Dimension 0)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在右侧避免遮挡
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('prediction_comparison_with_uncertainty.png', dpi=300, bbox_inches='tight')
    plt.show()

    # --- Additional Uncertainty Analysis Plot --- 额外的不确定性分析图
    plt.figure(figsize=(15, 8))

    # 上子图：预测结果对比
    plt.subplot(2, 1, 1)
    plt.plot(pred_time_indices, ground_truth, 'k-', label='Ground Truth', linewidth=2)
    plt.plot(pred_time_indices, result1, 'r--', label='Prediction 1 (GPR)', linewidth=2)
    plt.fill_between(pred_time_indices, result1 - 2*std1, result1 + 2*std1, alpha=0.3, color='red')
    plt.plot(pred_time_indices, result3, 'g--', label='Prediction 3 (LSTM)', linewidth=2)
    
    if result2 is not None and std2 is not None:
        plt.plot(pred_time_indices, result2, 'm--', label='Prediction 2 (Imputed GPR)', linewidth=2)
        plt.fill_between(pred_time_indices, result2 - 2*std2, result2 + 2*std2, alpha=0.3, color='magenta')
    
    plt.ylabel('Value')
    plt.title('Predictions vs Ground Truth with ±2σ Uncertainty')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 下子图：不确定性对比
    plt.subplot(2, 1, 2)
    plt.plot(pred_time_indices, 2*std1, 'r-', label='GPR Uncertainty (±2σ)', linewidth=2)
    plt.plot(pred_time_indices, np.abs(error1), 'r:', label='|GPR Error|', alpha=0.7)
    plt.plot(pred_time_indices, np.abs(error3), 'g:', label='|LSTM Error|', alpha=0.7)
    
    if result2 is not None and std2 is not None:
        plt.plot(pred_time_indices, 2*std2, 'm-', label='Imputed GPR Uncertainty (±2σ)', linewidth=2)
        plt.plot(pred_time_indices, np.abs(error2), 'm:', label='|Imputed GPR Error|', alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel('Uncertainty / Error')
    plt.title('Uncertainty vs Actual Error')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
