# final_debug_eeg.py - 最终调试版本，彻底解决问题
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
plt.rcParams['axes.unicode_minus'] = False
from gpr_module_eeg import GaussianProcessRegressor
from tqdm import tqdm
import pandas as pd
import os
import argparse

def debug_single_prediction(traindata, target_idx, comb=(0, 1, 2), steps_ahead=1):
    """
    调试单个预测，详细输出每个步骤
    """
    print(f"\n=== 详细调试单个预测 ===")
    print(f"组合: {comb}")
    print(f"训练数据形状: {traindata.shape}")
    print(f"目标通道索引: {target_idx}")
    
    try:
        trainlength = len(traindata)
        print(f"训练长度: {trainlength}")
        
        if trainlength <= steps_ahead:
            print(f"ERROR: 训练长度 {trainlength} <= 预测步数 {steps_ahead}")
            return np.nan, np.nan
            
        trainX = traindata[:trainlength-steps_ahead, list(comb)]
        trainy = traindata[steps_ahead:trainlength, target_idx]
        testX = traindata[trainlength-steps_ahead, list(comb)].reshape(1, -1)
        
        print(f"trainX 形状: {trainX.shape}")
        print(f"trainy 形状: {trainy.shape}")
        print(f"testX 形状: {testX.shape}")
        
        # 详细统计信息
        print(f"trainX 统计: min={np.min(trainX):.4f}, max={np.max(trainX):.4f}, mean={np.mean(trainX):.4f}, std={np.std(trainX):.4f}")
        print(f"trainy 统计: min={np.min(trainy):.4f}, max={np.max(trainy):.4f}, mean={np.mean(trainy):.4f}, std={np.std(trainy):.4f}")
        
        # 检查数据有效性
        if len(trainX) == 0 or len(trainy) == 0:
            print("ERROR: 数据长度为零")
            return np.nan, np.nan
            
        if np.any(np.isnan(trainX)) or np.any(np.isnan(trainy)) or np.any(np.isnan(testX)):
            print("ERROR: 数据包含NaN")
            nan_in_trainX = np.sum(np.isnan(trainX))
            nan_in_trainy = np.sum(np.isnan(trainy))
            nan_in_testX = np.sum(np.isnan(testX))
            print(f"NaN计数: trainX={nan_in_trainX}, trainy={nan_in_trainy}, testX={nan_in_testX}")
            return np.nan, np.nan
            
        # 检查方差
        trainy_var = np.var(trainy)
        print(f"trainy 方差: {trainy_var}")
        if trainy_var < 1e-10:
            print(f"ERROR: 目标变量方差过小: {trainy_var}")
            return np.nan, np.nan
            
        # 检查输入变量方差
        X_vars = np.var(trainX, axis=0)
        print(f"trainX 各列方差: {X_vars}")
        if np.any(X_vars < 1e-10):
            print(f"ERROR: 输入变量方差过小")
            return np.nan, np.nan

        # 手动标准化
        print("开始标准化...")
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        # 合并训练和测试X用于一致的标准化
        combined_X = np.vstack([trainX, testX])
        print(f"合并X形状: {combined_X.shape}")
        
        try:
            combined_X_scaled = scaler_X.fit_transform(combined_X)
            trainX_scaled = combined_X_scaled[:-1]
            testX_scaled = combined_X_scaled[-1:]
            print(f"X标准化成功")
        except Exception as e:
            print(f"X标准化失败: {e}")
            return np.nan, np.nan

        try:
            trainy_scaled = scaler_y.fit_transform(trainy.reshape(-1, 1)).flatten()
            print(f"y标准化成功")
        except Exception as e:
            print(f"y标准化失败: {e}")
            return np.nan, np.nan
            
        print(f"标准化后统计:")
        print(f"trainX_scaled: mean={np.mean(trainX_scaled):.4f}, std={np.std(trainX_scaled):.4f}")
        print(f"trainy_scaled: mean={np.mean(trainy_scaled):.4f}, std={np.std(trainy_scaled):.4f}")

        # GPR训练
        print("开始GPR训练...")
        try:
            gp = GaussianProcessRegressor(noise=1e-6)
            gp.fit(trainX_scaled, trainy_scaled, init_params=(1.0, 0.1, 0.1), optimize=True)
            print("GPR训练成功")
        except Exception as e:
            print(f"GPR训练失败: {e}")
            import traceback
            traceback.print_exc()
            return np.nan, np.nan

        # GPR预测
        print("开始GPR预测...")
        try:
            pred_scaled, std_scaled = gp.predict(testX_scaled, return_std=True)
            print(f"GPR预测成功: pred_scaled={pred_scaled}, std_scaled={std_scaled}")
        except Exception as e:
            print(f"GPR预测失败: {e}")
            import traceback
            traceback.print_exc()
            return np.nan, np.nan

        # 反标准化
        print("开始反标准化...")
        try:
            pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
            std = std_scaled[0] * scaler_y.scale_[0]
            print(f"反标准化成功: pred={pred:.4f}, std={std:.4f}")
        except Exception as e:
            print(f"反标准化失败: {e}")
            import traceback
            traceback.print_exc()
            return np.nan, np.nan
        
        return pred, std
        
    except Exception as e:
        print(f"整体预测失败: {e}")
        import traceback
        traceback.print_exc()
        return np.nan, np.nan

def _parallel_predict_robust(comb, traindata, target_idx, steps_ahead=1):
    """
    更鲁棒的并行预测函数
    """
    try:
        trainlength = len(traindata)
        if trainlength <= steps_ahead:
            return np.nan, np.nan
            
        trainX = traindata[:trainlength-steps_ahead, list(comb)]
        trainy = traindata[steps_ahead:trainlength, target_idx]
        testX = traindata[trainlength-steps_ahead, list(comb)].reshape(1, -1)

        # 基本检查
        if len(trainX) == 0 or len(trainy) == 0:
            return np.nan, np.nan
            
        if np.any(np.isnan(trainX)) or np.any(np.isnan(trainy)) or np.any(np.isnan(testX)):
            return np.nan, np.nan
            
        if np.var(trainy) < 1e-10:
            return np.nan, np.nan
            
        X_vars = np.var(trainX, axis=0)
        if np.any(X_vars < 1e-10):
            return np.nan, np.nan

        # 标准化
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        combined_X = np.vstack([trainX, testX])
        combined_X_scaled = scaler_X.fit_transform(combined_X)
        trainX_scaled = combined_X_scaled[:-1]
        testX_scaled = combined_X_scaled[-1:]

        trainy_scaled = scaler_y.fit_transform(trainy.reshape(-1, 1)).flatten()

        # GPR
        gp = GaussianProcessRegressor(noise=1e-6)
        gp.fit(trainX_scaled, trainy_scaled, init_params=(1.0, 0.1, 0.1), optimize=True)
        pred_scaled, std_scaled = gp.predict(testX_scaled, return_std=True)

        # 反标准化
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        std = std_scaled[0] * scaler_y.scale_[0]
        
        return pred, std
        
    except Exception as e:
        return np.nan, np.nan

def predict_robust(seq, trainlength=30, L=5, s=50, j=0, n_jobs=1, steps_ahead=1, desc_prefix="", debug_first=True):
    """
    更鲁棒的预测函数，包含详细调试
    """
    print(f"\n=== {desc_prefix}开始鲁棒预测 ===")
    
    # 数据分析
    print(f"输入数据形状: {seq.shape}")
    print(f"数据类型: {seq.dtype}")
    print(f"是否包含NaN: {np.any(np.isnan(seq))}")
    print(f"是否包含inf: {np.any(np.isinf(seq))}")
    print(f"数据范围: [{np.min(seq):.4f}, {np.max(seq):.4f}]")
    print(f"数据均值: {np.mean(seq):.4f}, 标准差: {np.std(seq):.4f}")
    
    # 检查目标通道
    target_channel_data = seq[:, j]
    print(f"目标通道 {j} 统计:")
    print(f"  范围: [{np.min(target_channel_data):.4f}, {np.max(target_channel_data):.4f}]")
    print(f"  均值: {np.mean(target_channel_data):.4f}, 标准差: {np.std(target_channel_data):.4f}")
    print(f"  方差: {np.var(target_channel_data):.6f}")

    # 添加微小噪声
    noise_strength = 1e-8
    x = seq + noise_strength * np.random.randn(*seq.shape)

    total_steps = len(seq) - trainlength
    if total_steps <= 0:
        print(f"错误：序列长度 {len(seq)} 小于等于训练长度 {trainlength}")
        return np.zeros((3, 1))

    print(f"总预测步数: {total_steps}")
    print(f"训练长度: {trainlength}")
    print(f"嵌入维度: {L}")
    print(f"组合数量: {s}")

    # 调试第一步
    if debug_first and total_steps > 0:
        print(f"\n=== 调试第一个预测步骤 ===")
        traindata = x[0: 0 + trainlength, :]
        print(f"第一步训练数据形状: {traindata.shape}")
        
        # 生成组合
        D = traindata.shape[1]
        if D < L:
            L_actual = D
            print(f"调整嵌入维度: {L} -> {L_actual}")
        else:
            L_actual = L
            
        combs = list(itertools.combinations(range(D), L_actual))
        first_comb = combs[0]
        print(f"测试第一个组合: {first_comb}")
        
        # 调试单个预测
        pred, std = debug_single_prediction(traindata, j, first_comb, steps_ahead)
        print(f"调试结果: pred={pred}, std={std}")
        
        if np.isnan(pred):
            print("调试预测失败，停止执行")
            return np.zeros((3, total_steps)) * np.nan

    # 结果存储
    result = np.zeros((3, total_steps))
    
    # 使用单进程便于调试
    print(f"\n=== 开始完整预测 ===")
    for step in range(min(5, total_steps)):  # 只预测前5步进行测试
        print(f"\n--- 预测步骤 {step+1} ---")
        
        traindata = x[step: step + trainlength, :]
        real_value = x[step + trainlength, j]
        
        # 生成组合
        D = traindata.shape[1]
        L_actual = min(L, D)
        combs = list(itertools.combinations(range(D), L_actual))
        np.random.shuffle(combs)
        selected_combs = combs[:min(s, len(combs))]
        
        print(f"组合数量: {len(selected_combs)}")
        
        # 预测
        predictions = []
        for i, comb in enumerate(selected_combs[:5]):  # 只测试前5个组合
            pred, std = _parallel_predict_robust(comb, traindata, j, steps_ahead)
            predictions.append((pred, std))
            if i == 0:  # 只打印第一个组合的结果
                print(f"组合 {comb}: pred={pred}, std={std}")
        
        # 后处理
        pred_values = np.array([p[0] for p in predictions])
        pred_stds = np.array([p[1] for p in predictions]) 
        valid_mask = ~np.isnan(pred_values) & ~np.isnan(pred_stds)
        valid_preds = pred_values[valid_mask]
        
        valid_count = len(valid_preds)
        print(f"有效预测数: {valid_count}/{len(predictions)}")
        
        if valid_count == 0:
            final_pred = np.nan
            final_std = np.nan
        elif valid_count == 1:
            final_pred = valid_preds[0]
            final_std = 0.0
        else:
            final_pred = np.mean(valid_preds)
            final_std = np.std(valid_preds)

        result[0, step] = final_pred
        result[1, step] = final_std
        result[2, step] = real_value - final_pred
        
        print(f"最终预测: {final_pred:.4f}, 残差: {result[2, step]:.4f}")

    return result

def load_data_with_debug(data_path, data_type="原始", seq_len=None):
    """
    加载数据并进行详细调试
    """
    print(f"\n=== 加载{data_type}数据 ===")
    print(f"文件路径: {data_path}")
    print(f"文件是否存在: {os.path.exists(data_path)}")
    
    try:
        if data_path.endswith('.npz'):
            # NPZ文件
            loaded_data = np.load(data_path)
            print(f"NPZ文件keys: {list(loaded_data.keys())}")
            
            # 选择数据
            if 'imputed' in loaded_data:
                data = loaded_data['imputed']
                key_used = 'imputed'
            elif 'original' in loaded_data:
                data = loaded_data['original']
                key_used = 'original'
            else:
                key_used = list(loaded_data.keys())[0]
                data = loaded_data[key_used]
            
            print(f"使用key: {key_used}")
            
        else:
            # Excel文件
            try:
                df = pd.read_excel(data_path, header=None, engine='openpyxl')
            except:
                try:
                    df = pd.read_excel(data_path, header=None, engine='xlrd')
                except:
                    csv_path = data_path.replace('.xlsx', '.csv')
                    df = pd.read_csv(csv_path, header=None)
            
            data = df.values.astype(np.float32)
        
        print(f"原始数据形状: {data.shape}")
        print(f"原始数据类型: {data.dtype}")
        
        # 处理3D数据
        if data.ndim == 3:
            print(f"检测到3D数据，使用第一个batch")
            data = data[0]
        
        # 处理缺失值
        nan_count = np.sum(np.isnan(data))
        if nan_count > 0:
            print(f"发现 {nan_count} 个NaN值，进行填充")
            df_clean = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill')
            data = df_clean.values
        
        # 截断长度
        if seq_len and data.shape[0] > seq_len:
            print(f"截断数据: {data.shape[0]} -> {seq_len}")
            data = data[:seq_len]
        
        print(f"最终数据形状: {data.shape}")
        print(f"数据统计: min={np.min(data):.4f}, max={np.max(data):.4f}, mean={np.mean(data):.4f}, std={np.std(data):.4f}")
        
        return data.astype(np.float32)
        
    except Exception as e:
        print(f"{data_type}数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='最终调试版EEG GPR预测')
    parser.add_argument('--imputed_data_path', type=str, 
                       default='./results/eeg_imputation_results.npz',
                       help='Imputed数据路径')
    parser.add_argument('--original_data_path', type=str, 
                       default='/home/rhl/CSDI-GPR_project/CSDI-GPR_datatest/data/Dataset_3-EEG.xlsx',
                       help='原始EEG数据路径')
    parser.add_argument('--channel_idx', type=int, default=0,
                       help='预测通道索引')
    parser.add_argument('--seq_len', type=int, default=100,
                       help='序列长度')
    parser.add_argument('--trainlength', type=int, default=20,
                       help='训练窗口长度')
    parser.add_argument('--L', type=int, default=3,
                       help='嵌入维度')
    
    args = parser.parse_args()
    
    print("="*50)
    print("最终调试版EEG GPR预测")
    print("="*50)
    
    # 加载数据
    original_data = load_data_with_debug(args.original_data_path, "原始", args.seq_len)
    if original_data is None:
        return
        
    imputed_data = load_data_with_debug(args.imputed_data_path, "Imputed", args.seq_len)
    if imputed_data is None:
        return
    
    # 数据匹配
    if original_data.shape != imputed_data.shape:
        min_len = min(original_data.shape[0], imputed_data.shape[0])
        min_channels = min(original_data.shape[1], imputed_data.shape[1])
        print(f"\n数据形状不匹配，调整为: ({min_len}, {min_channels})")
        original_data = original_data[:min_len, :min_channels]
        imputed_data = imputed_data[:min_len, :min_channels]
    
    print(f"\n最终匹配数据形状: {original_data.shape}")
    
    # 预测原始数据
    print(f"\n" + "="*30)
    print("预测原始数据")
    print("="*30)
    original_result = predict_robust(
        seq=original_data,
        trainlength=args.trainlength,
        L=args.L,
        s=10,
        j=args.channel_idx,
        n_jobs=1,
        steps_ahead=1,
        desc_prefix="原始数据 - ",
        debug_first=True
    )
    
    # 预测imputed数据
    print(f"\n" + "="*30)
    print("预测Imputed数据")
    print("="*30)
    imputed_result = predict_robust(
        seq=imputed_data,
        trainlength=args.trainlength,
        L=args.L,
        s=10,
        j=args.channel_idx,
        n_jobs=1,
        steps_ahead=1,
        desc_prefix="Imputed数据 - ",
        debug_first=True
    )
    
    # 结果对比
    print(f"\n" + "="*50)
    print("结果对比")
    print("="*50)
    
    orig_valid = np.sum(~np.isnan(original_result[0, :]))
    imp_valid = np.sum(~np.isnan(imputed_result[0, :]))
    
    print(f"原始数据有效预测数: {orig_valid}/{original_result.shape[1]}")
    print(f"Imputed数据有效预测数: {imp_valid}/{imputed_result.shape[1]}")
    
    if orig_valid > 0:
        orig_residuals = original_result[2, ~np.isnan(original_result[2, :])]
        print(f"原始数据RMSE: {np.sqrt(np.mean(orig_residuals**2)):.4f}")
    
    if imp_valid > 0:
        imp_residuals = imputed_result[2, ~np.isnan(imputed_result[2, :])]
        print(f"Imputed数据RMSE: {np.sqrt(np.mean(imp_residuals**2)):.4f}")

if __name__ == "__main__":
    main()