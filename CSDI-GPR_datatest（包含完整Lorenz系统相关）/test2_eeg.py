# test2_eeg_comparison_main.py - 修复的EEG数据对比主代码
import numpy as np
import time
import multiprocessing as mp
from functools import partial
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import itertools
import matplotlib.pyplot as plt
import matplotlib
# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from gpr_module_eeg import GaussianProcessRegressor
from tqdm import tqdm
import pandas as pd
import os
import argparse

def _parallel_predict(comb, traindata, target_idx, steps_ahead=1):
    """
    修复的并行预测函数，兼容现有GPR模块
    """
    start_time = time.time()
    try:
        trainlength = len(traindata)
        trainX = traindata[:trainlength-steps_ahead, list(comb)]
        trainy = traindata[steps_ahead:trainlength, target_idx]
        testX = traindata[trainlength-steps_ahead, list(comb)].reshape(1, -1)

        # 检查数据有效性
        if len(trainX) == 0 or len(trainy) == 0:
            return np.nan, np.nan
            
        if np.any(np.isnan(trainX)) or np.any(np.isnan(trainy)) or np.any(np.isnan(testX)):
            return np.nan, np.nan
            
        # 检查方差
        if np.var(trainy) < 1e-10:
            return np.nan, np.nan
            
        # 检查输入变量方差
        X_vars = np.var(trainX, axis=0)
        if np.any(X_vars < 1e-10):
            return np.nan, np.nan

        # 手动标准化（兼容现有GPR模块）
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        # 合并训练和测试X用于一致的标准化
        combined_X = np.vstack([trainX, testX])
        combined_X_scaled = scaler_X.fit_transform(combined_X)
        trainX_scaled = combined_X_scaled[:-1]
        testX_scaled = combined_X_scaled[-1:]

        trainy_scaled = scaler_y.fit_transform(trainy.reshape(-1, 1)).flatten()

        # 使用现有的GPR模块
        gp = GaussianProcessRegressor(noise=1e-6)
        gp.fit(trainX_scaled, trainy_scaled, init_params=(1.0, 0.1, 0.1), optimize=True)
        pred_scaled, std_scaled = gp.predict(testX_scaled, return_std=True)

        # 反标准化
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        std = std_scaled[0] * scaler_y.scale_[0]
        
        return pred, std
    except Exception as e:
        return np.nan, np.nan

def predict(seq, trainlength=30, L=5, s=600, j=0, n_jobs=4, steps_ahead=1, desc_prefix=""):
    """
    修复的预测函数
    """
    # 数据预处理
    noise_strength = 1e-6
    x = seq + noise_strength * np.random.randn(*seq.shape)

    total_steps = len(seq) - trainlength
    if total_steps <= 0:
        print(f"警告：序列长度 {len(seq)} 小于等于训练长度 {trainlength}")
        return np.zeros((3, 1))

    # 结果存储矩阵 [预测值, 标准差, 残差]
    result = np.zeros((3, total_steps))
    
    # 创建进程池
    pool = mp.Pool(processes=n_jobs)
    
    # 初始化进度条
    with tqdm(total=total_steps, desc=f"{desc_prefix}Processing Steps") as pbar:
        for step in range(total_steps):
            try:
                # 1. Data Slicing
                traindata = x[step: step + trainlength, :]
                real_value = x[step + trainlength, j]
                
                # 2. 生成随机嵌入基组合
                D = traindata.shape[1]
                if D < L:
                    L_actual = D
                    if step == 0:  # 只在第一步打印警告
                        print(f"警告：通道数 {D} 小于嵌入维度 {L}，调整为 {L_actual}")
                else:
                    L_actual = L
                    
                combs = list(itertools.combinations(range(D), L_actual))
                np.random.shuffle(combs)
                selected_combs = combs[:min(s, len(combs))]
                
                if len(selected_combs) == 0:
                    result[0, step] = np.nan
                    result[1, step] = np.nan
                    result[2, step] = np.nan
                    continue
                
                # 3. 并行预测
                predictions = pool.map(
                    partial(_parallel_predict, 
                            traindata=traindata,
                            target_idx=j,
                            steps_ahead=steps_ahead),
                    selected_combs
                )
                
                # 4. 后处理
                pred_values = np.array([p[0] for p in predictions])
                pred_stds = np.array([p[1] for p in predictions]) 
                valid_mask = ~np.isnan(pred_values) & ~np.isnan(pred_stds)
                valid_preds = pred_values[valid_mask]
                
                if len(valid_preds) == 0:
                    final_pred = np.nan
                    final_std = np.nan
                elif len(valid_preds) == 1:
                    final_pred = valid_preds[0]
                    final_std = 0.0
                else:
                    try:
                        kde = gaussian_kde(valid_preds)
                        xi = np.linspace(valid_preds.min(), valid_preds.max(), 1000)
                        density = kde(xi)
                        final_pred = np.sum(xi * density) / np.sum(density)
                        final_std = np.std(valid_preds)
                    except:
                        final_pred = np.mean(valid_preds)
                        final_std = np.std(valid_preds)

                result[0, step] = final_pred
                result[1, step] = final_std
                result[2, step] = real_value - final_pred
                
                # 每20步打印详细信息
                if (step % 20 == 0) or (step == total_steps - 1):
                    valid_count = np.sum(valid_mask)
                    pbar.write(f"Step {step+1}/{total_steps} | Valid predictions: {valid_count}/{len(selected_combs)} | Residual: {result[2, step]:.4f}")

            except Exception as e:
                print(f"Step {step} 处理失败: {str(e)}")
                result[0, step] = np.nan
                result[1, step] = np.nan
                result[2, step] = np.nan

            # 更新进度条
            pbar.update(1)

    pool.close()
    pool.join()
    return result

def load_original_eeg_data(data_path, seq_len=None):
    """
    加载原始EEG数据
    """
    try:
        # 方法1：尝试使用openpyxl引擎
        try:
            df = pd.read_excel(data_path, header=None, engine='openpyxl')
        except:
            # 方法2：如果openpyxl失败，尝试xlrd
            try:
                df = pd.read_excel(data_path, header=None, engine='xlrd')
            except:
                # 方法3：如果都失败，转换为CSV再读取
                print("Excel读取失败，尝试转换为CSV格式...")
                csv_path = data_path.replace('.xlsx', '.csv')
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, header=None)
                else:
                    raise Exception("请将Excel文件转换为CSV格式")
        
        data = df.values.astype(np.float32)
        
        # 检查和处理缺失值
        if np.any(np.isnan(data)):
            print("警告：原始数据包含NaN值，进行前向填充处理")
            df_clean = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill')
            data = df_clean.values.astype(np.float32)
        
        if seq_len and data.shape[0] > seq_len:
            data = data[:seq_len]
            
        print(f"原始EEG数据加载成功，形状: {data.shape}")
        return data
        
    except Exception as e:
        print(f"原始数据加载失败: {str(e)}")
        return None

def load_imputed_eeg_data(data_path, seq_len=None):
    """
    加载imputed EEG数据
    """
    try:
        if not os.path.exists(data_path):
            print(f"文件不存在: {data_path}")
            return None
            
        # 加载.npz文件
        if data_path.endswith('.npz'):
            loaded_data = np.load(data_path)
            print(f"NPZ文件中可用的keys: {list(loaded_data.keys())}")
            
            # 尝试常见的key名称，优先选择 'imputed'
            possible_keys = ['imputed', 'imputed_data', 'data', 'samples', 'generated_samples', 'original']
            data = None
            
            for key in possible_keys:
                if key in loaded_data:
                    data = loaded_data[key]
                    print(f"使用key: {key}")
                    break
            
            if data is None:
                # 如果没有找到常见key，使用第一个
                first_key = list(loaded_data.keys())[0]
                data = loaded_data[first_key]
                print(f"使用第一个可用key: {first_key}")
                
        else:
            raise Exception("Imputed数据必须是.npz格式")
        
        # 确保数据是2D的
        if data.ndim == 3:
            # 如果是3D数据 (batch, time, features)，取第一个batch
            data = data[0]
            print(f"检测到3D数据，使用第一个batch")
        elif data.ndim == 1:
            # 如果是1D数据，转换为单列2D数据
            data = data.reshape(-1, 1)
            print(f"检测到1D数据，转换为2D")
        
        # 检查和处理缺失值
        if np.any(np.isnan(data)):
            print("警告：Imputed数据包含NaN值，进行前向填充处理")
            df_clean = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill')
            data = df_clean.values.astype(np.float32)
        
        if seq_len and data.shape[0] > seq_len:
            data = data[:seq_len]
            
        print(f"Imputed EEG数据加载成功，形状: {data.shape}")
        return data
        
    except Exception as e:
        print(f"Imputed数据加载失败: {str(e)}")
        return None

def calculate_metrics(result):
    """
    计算预测性能指标
    """
    valid_residuals = result[2, ~np.isnan(result[2, :])]
    if len(valid_residuals) == 0:
        return None
    
    metrics = {
        'rmse': np.sqrt(np.mean(np.square(valid_residuals))),
        'mae': np.mean(np.abs(valid_residuals)),
        'std': np.std(valid_residuals),
        'mean': np.mean(valid_residuals),
        'valid_predictions': len(valid_residuals),
        'total_predictions': result.shape[1]
    }
    return metrics

def plot_comparison(original_data, imputed_data, original_result, imputed_result, 
                   channel_idx, trainlength, steps_ahead):
    """
    绘制原始数据和imputed数据的预测对比图
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 计算时间索引
    prediction_start_idx = trainlength + steps_ahead - 1
    
    # 原始数据预测
    total_time_steps = original_data.shape[0]
    prediction_time_indices = np.arange(prediction_start_idx, 
                                      prediction_start_idx + original_result.shape[1])
    
    ax1.plot(np.arange(total_time_steps), original_data[:, channel_idx], 
             'b-', label=f'Actual Original Data (Ch {channel_idx})', alpha=0.7)
    valid_mask = ~np.isnan(original_result[0, :])
    if np.any(valid_mask):
        ax1.plot(prediction_time_indices[valid_mask], original_result[0, valid_mask], 
                 'r-', label='Predictions', alpha=0.8, linewidth=2)
    ax1.axvline(x=prediction_start_idx, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Original Data - GPR Prediction')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Imputed数据预测
    total_time_steps = imputed_data.shape[0]
    prediction_time_indices = np.arange(prediction_start_idx, 
                                      prediction_start_idx + imputed_result.shape[1])
    
    ax2.plot(np.arange(total_time_steps), imputed_data[:, channel_idx], 
             'g-', label=f'Imputed Data (Ch {channel_idx})', alpha=0.7)
    valid_mask = ~np.isnan(imputed_result[0, :])
    if np.any(valid_mask):
        ax2.plot(prediction_time_indices[valid_mask], imputed_result[0, valid_mask], 
                 'r-', label='Predictions', alpha=0.8, linewidth=2)
    ax2.axvline(x=prediction_start_idx, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('Imputed Data - GPR Prediction')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 残差对比
    ax3.plot(original_result[2, :], 'b-', label='Original Data Residuals', alpha=0.7)
    ax3.plot(imputed_result[2, :], 'g-', label='Imputed Data Residuals', alpha=0.7)
    ax3.set_title('Residuals Comparison')
    ax3.set_xlabel('Prediction Step')
    ax3.set_ylabel('Residual')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 残差分布对比
    original_valid_residuals = original_result[2, ~np.isnan(original_result[2, :])]
    imputed_valid_residuals = imputed_result[2, ~np.isnan(imputed_result[2, :])]
    
    if len(original_valid_residuals) > 0:
        ax4.hist(original_valid_residuals, bins=30, alpha=0.7, label='Original Data Residuals', color='blue')
    if len(imputed_valid_residuals) > 0:
        ax4.hist(imputed_valid_residuals, bins=30, alpha=0.7, label='Imputed Data Residuals', color='green')
    ax4.set_title('Residuals Distribution Comparison')
    ax4.set_xlabel('Residual Value')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"eeg_comparison_channel_{channel_idx}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存到: {plot_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='EEG时间序列预测性能对比')
    parser.add_argument('--imputed_data_path', type=str, 
                       default='./results/eeg_imputation_results.npz',
                       help='Imputed数据路径 (.npz文件)')
    parser.add_argument('--original_data_path', type=str, 
                       default='/home/rhl/CSDI-GPR_project/CSDI-GPR_datatest/data/Dataset_3-EEG.xlsx',
                       help='原始EEG数据路径 (.xlsx文件)')
    parser.add_argument('--channel_idx', type=int, default=0,
                       help='要预测的通道索引')
    parser.add_argument('--seq_len', type=int, default=200,
                       help='序列长度')
    parser.add_argument('--trainlength', type=int, default=20,
                       help='训练窗口长度')
    parser.add_argument('--L', type=int, default=4,
                       help='嵌入维度')
    parser.add_argument('--s', type=int, default=50,
                       help='随机组合数量')
    parser.add_argument('--n_jobs', type=int, default=4,
                       help='并行进程数')
    
    args = parser.parse_args()
    
    print("="*50)
    print("EEG数据GPR预测性能对比")
    print("="*50)
    
    # 1. 加载原始EEG数据
    print("\n1. 加载原始EEG数据...")
    original_data = load_original_eeg_data(args.original_data_path, args.seq_len)
    if original_data is None:
        print("原始数据加载失败，退出程序")
        return
    
    # 2. 加载imputed EEG数据
    print("\n2. 加载Imputed EEG数据...")
    imputed_data = load_imputed_eeg_data(args.imputed_data_path, args.seq_len)
    if imputed_data is None:
        print("Imputed数据加载失败，退出程序")
        return
    
    # 检查数据维度匹配
    if original_data.shape != imputed_data.shape:
        min_len = min(original_data.shape[0], imputed_data.shape[0])
        min_channels = min(original_data.shape[1], imputed_data.shape[1])
        original_data = original_data[:min_len, :min_channels]
        imputed_data = imputed_data[:min_len, :min_channels]
        print(f"调整数据形状匹配: {original_data.shape}")
    
    print(f"预测通道: {args.channel_idx}")
    print(f"数据形状: {original_data.shape}")
    
    # 3. 对原始数据进行GPR预测
    print("\n3. 对原始数据进行GPR预测...")
    original_result = predict(
        seq=original_data,
        trainlength=args.trainlength,
        L=args.L,
        s=args.s,
        j=args.channel_idx,
        n_jobs=args.n_jobs,
        steps_ahead=1,
        desc_prefix="原始数据 - "
    )
    
    # 4. 对imputed数据进行GPR预测
    print("\n4. 对Imputed数据进行GPR预测...")
    imputed_result = predict(
        seq=imputed_data,
        trainlength=args.trainlength,
        L=args.L,
        s=args.s,
        j=args.channel_idx,
        n_jobs=args.n_jobs,
        steps_ahead=1,
        desc_prefix="Imputed数据 - "
    )
    
    # 5. 计算性能指标
    print("\n5. 性能对比分析...")
    original_metrics = calculate_metrics(original_result)
    imputed_metrics = calculate_metrics(imputed_result)
    
    # 打印对比结果
    print("\n" + "="*50)
    print("性能对比结果")
    print("="*50)
    
    if original_metrics and imputed_metrics:
        print(f"{'指标':<15} {'原始数据':<15} {'Imputed数据':<15} {'改善程度':<15}")
        print("-" * 60)
        
        for metric in ['rmse', 'mae', 'std', 'mean']:
            orig_val = original_metrics[metric]
            imp_val = imputed_metrics[metric]
            improvement = ((orig_val - imp_val) / orig_val * 100) if orig_val != 0 else 0
            print(f"{metric.upper():<15} {orig_val:<15.4f} {imp_val:<15.4f} {improvement:<15.2f}%")
        
        print(f"{'有效预测数':<15} {original_metrics['valid_predictions']:<15} {imputed_metrics['valid_predictions']:<15}")
        
        # 判断哪个性能更好
        print("\n性能总结:")
        if imputed_metrics['rmse'] < original_metrics['rmse']:
            print("✓ Imputed数据的预测性能更好 (RMSE更低)")
        else:
            print("✓ 原始数据的预测性能更好 (RMSE更低)")
    elif original_metrics:
        print("只有原始数据的指标可用:")
        for metric in ['rmse', 'mae', 'std', 'mean', 'valid_predictions']:
            print(f"{metric.upper()}: {original_metrics[metric]:.4f}")
    elif imputed_metrics:
        print("只有Imputed数据的指标可用:")
        for metric in ['rmse', 'mae', 'std', 'mean', 'valid_predictions']:
            print(f"{metric.upper()}: {imputed_metrics[metric]:.4f}")
    else:
        print("两个数据集都无法获得有效的预测结果")
    
    # 6. 绘制对比图
    print("\n6. 生成对比可视化...")
    plot_comparison(original_data, imputed_data, original_result, imputed_result,
                   args.channel_idx, args.trainlength, 1)
    
    # 7. 保存结果
    output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"eeg_comparison_channel_{args.channel_idx}.npz")
    np.savez(output_path,
             original_predictions=original_result[0, :],
             original_stds=original_result[1, :],
             original_residuals=original_result[2, :],
             imputed_predictions=imputed_result[0, :],
             imputed_stds=imputed_result[1, :],
             imputed_residuals=imputed_result[2, :],
             original_data=original_data[:, args.channel_idx],
             imputed_data=imputed_data[:, args.channel_idx])
    
    print(f"对比结果已保存到: {output_path}")
    
    # 保存性能指标到CSV
    if original_metrics or imputed_metrics:
        if original_metrics and imputed_metrics:
            metrics_df = pd.DataFrame({
                '指标': ['RMSE', 'MAE', 'STD', 'MEAN', '有效预测数'],
                '原始数据': [original_metrics['rmse'], original_metrics['mae'], 
                            original_metrics['std'], original_metrics['mean'],
                            original_metrics['valid_predictions']],
                'Imputed数据': [imputed_metrics['rmse'], imputed_metrics['mae'],
                               imputed_metrics['std'], imputed_metrics['mean'],
                               imputed_metrics['valid_predictions']]
            })
        elif original_metrics:
            metrics_df = pd.DataFrame({
                '指标': ['RMSE', 'MAE', 'STD', 'MEAN', '有效预测数'],
                '原始数据': [original_metrics['rmse'], original_metrics['mae'], 
                            original_metrics['std'], original_metrics['mean'],
                            original_metrics['valid_predictions']]
            })
        else:
            metrics_df = pd.DataFrame({
                '指标': ['RMSE', 'MAE', 'STD', 'MEAN', '有效预测数'],
                'Imputed数据': [imputed_metrics['rmse'], imputed_metrics['mae'],
                               imputed_metrics['std'], imputed_metrics['mean'],
                               imputed_metrics['valid_predictions']]
            })
        
        csv_path = os.path.join(output_dir, f"metrics_comparison_channel_{args.channel_idx}.csv")
        metrics_df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"性能指标已保存到: {csv_path}")

if __name__ == "__main__":
    main()