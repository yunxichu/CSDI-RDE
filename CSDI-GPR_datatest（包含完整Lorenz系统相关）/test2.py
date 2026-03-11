# test2.py  使用集成高斯过程回归（Gaussian Process Regression）对Lorenz系统进行时间序列预测
import numpy as np
import time
import multiprocessing as mp
from functools import partial
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import itertools
import matplotlib.pyplot as plt
from gpr_module import GaussianProcessRegressor
from tqdm import tqdm
import pandas as pd
import os
from dataset_lorenz import generate_coupled_lorenz

def _parallel_predict(comb, traindata, target_idx, steps_ahead=1):
    start_time = time.time()
    try:
        trainlength = len(traindata)
        trainX = traindata[:trainlength-steps_ahead, list(comb)]
        trainy = traindata[steps_ahead:trainlength, target_idx]
        testX = traindata[trainlength-steps_ahead, list(comb)].reshape(1, -1)

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        # Combine train and test X for scaling
        combined_X = np.vstack([trainX, testX])
        combined_X_scaled = scaler_X.fit_transform(combined_X)
        trainX_scaled = combined_X_scaled[:-1]
        testX_scaled = combined_X_scaled[-1:]

        trainy_scaled = scaler_y.fit_transform(trainy.reshape(-1, 1)).flatten()

        gp = GaussianProcessRegressor(noise=1e-6)
        gp.fit(trainX_scaled, trainy_scaled, init_params=(1.0, 0.1, 0.1), optimize=True)
        pred_scaled, std_scaled = gp.predict(testX_scaled, return_std=True)

        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0, 0]
        end_time = time.time()
        # print(f"预测时间: {end_time - start_time:.4f}s")
        return pred, std_scaled[0]
    except Exception as e:
        print(f"预测失败：{str(e)}")
        return np.nan, np.nan
    

def predict(seq, trainlength=30, L=4, s=600, j=0, n_jobs=4, steps_ahead=1):
    # 数据预处理
    noise_strength = 1e-4
    x = seq + noise_strength * np.random.randn(*seq.shape)

    total_steps = len(seq) - trainlength

    # 结果存储矩阵 [预测值, 标准差, 残差]
    result = np.zeros((3, total_steps))
    
    # 创建4个并行进程的进程池
    pool = mp.Pool(processes=n_jobs)
    
    # 初始化进度条
    with tqdm(total=total_steps, desc="Processing Steps") as pbar:
        for step in range(total_steps):
            step_start_time = time.time()

            # 1. Data Slicing
            slicing_start_time = time.time()
            traindata = x[step: step + trainlength, :]
            real_value = x[step + trainlength, j]  # Get value from (time, feature) indexing
            slicing_time = time.time() - slicing_start_time
            
            # 生成随机嵌入基组合
            comb_gen_start_time = time.time()
            D = traindata.shape[1] #15
            combs = list(itertools.combinations(range(D), L))
            np.random.shuffle(combs)
            selected_combs = combs[:s]
            comb_gen_time = time.time() - comb_gen_start_time
            
            # 并行预测
            parallel_pred_start_time = time.time()
            predictions = pool.map(
                partial(_parallel_predict, 
                        traindata=traindata,
                        target_idx=j,
                        steps_ahead=steps_ahead),
                selected_combs
            )
            parallel_pred_time = time.time() - parallel_pred_start_time
            
            # 后处理
            post_proc_start_time = time.time()
            
            pred_values = np.array([p[0] for p in predictions])
            pred_stds = np.array([p[1] for p in predictions]) 
            valid_mask = ~np.isnan(pred_values) & ~np.isnan(pred_stds)
            valid_preds = pred_values[valid_mask]
            valid_stds = pred_stds[valid_mask]
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
                except:  # 添加异常处理
                    final_pred = np.mean(valid_preds)
                    final_std = np.std(valid_preds)

                # print(valid_preds)
                # print(final_pred)
            
            post_proc_time = time.time() - post_proc_start_time

            result[0, step] = final_pred
            result[1, step] = final_std
            result[2, step] = real_value - final_pred
            
            step_time = time.time() - step_start_time

            # 每10步或最后一步打印详细信息
            if (step % 10 == 0) or (step == total_steps - 1):
                pbar.write(f"Step {step+1}/{total_steps} | Residual: {result[2, step]:.4f}")
            #print(f"  Timings: Total: {step_time:.4f}s | Slicing: {slicing_time:.4f}s | Comb Gen: {comb_gen_time:.4f}s | Parallel Pred: {parallel_pred_time:.4f}s | Post-proc: {post_proc_time:.4f}s")

            # 更新进度条
            pbar.update(1)

    pool.close()
    return result

def main():
    lorenz_data, full_data = generate_coupled_lorenz(N=5, L=100, stepsize=1)
    print(f"Generated data shape: {lorenz_data.shape}")
    print(f"Data range: [{lorenz_data.min():.3f}, {lorenz_data.max():.3f}]")

    trainlength = 30
    steps_ahead = 1
    
    result = predict(
        seq=lorenz_data,
        trainlength=trainlength,
        L=4,
        s=100,
        j=0,
        n_jobs=4,
        steps_ahead=steps_ahead
    )
    
    result = predict(
        seq=lorenz_data,
        trainlength=trainlength,
        L=4,
        s=100,
        j=1,
        n_jobs=4,
        steps_ahead=steps_ahead
    )

    result = predict(
        seq=lorenz_data,
        trainlength=trainlength,
        L=4,
        s=100,
        j=2,
        n_jobs=4,
        steps_ahead=steps_ahead
    )
    for i in range(result.shape[1]):
        print(f"Step {i+1}/{result.shape[1]} | Predicted: {result[0, i]:.4f}, Residual: {result[2, i]:.4f}")

    # Calculate and print the root mean square of residuals
    print(f"RMS of residuals: {np.sqrt(np.mean(np.square(result[2]))):.4f}")

    # Plotting section
    plt.figure(figsize=(12, 6))

    # Calculate the time indices for alignment
    total_time_steps = lorenz_data.shape[0]
    prediction_start_idx = trainlength + steps_ahead - 1
    prediction_time_indices = np.arange(prediction_start_idx, prediction_start_idx + result.shape[1])

    # Plot the actual Lorenz data (first dimension)
    plt.plot(np.arange(total_time_steps), lorenz_data[:, 0], 'b-', label='Actual Lorenz Data (dim 0)', alpha=0.7)

    # Plot the predictions
    plt.plot(prediction_time_indices, result[0, :], 'r-', label='Predictions', alpha=0.8, linewidth=2)

    # Add vertical line to show where predictions start
    plt.axvline(x=prediction_start_idx, color='gray', linestyle='--', alpha=0.5, label='Prediction Start')

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Comparison: Actual Lorenz Data vs Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    

if __name__ == "__main__":
    #np.random.seed(10)
    main()