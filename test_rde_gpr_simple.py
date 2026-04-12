# -*- coding: utf-8 -*-
"""
RDE-GPR 快速测试脚本
测试数据加载和GPR核心功能
"""

import os, sys, json, time, argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# GPR 核心函数（简化版）
def compute_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * dists)

def gpr_predict(X_train, y_train, X_test, length_scale=1.0, sigma_f=1.0, sigma_n=0.1):
    K = compute_kernel(X_train, X_train, length_scale, sigma_f) + sigma_n**2 * np.eye(len(X_train))
    K_s = compute_kernel(X_train, X_test, length_scale, sigma_f)
    K_ss = compute_kernel(X_test, X_test, length_scale, sigma_f)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
    mu = K_s.T.dot(alpha)
    v = np.linalg.solve(L, K_s)
    cov = K_ss - v.T.dot(v)
    return mu, np.sqrt(np.diag(cov))

def main():
    print("="*70)
    print("RDE-GPR 快速测试")
    print("="*70)
    
    # 参数
    imputed_history_path = "./save/pm25_history_imputed_split0.5_seed42_20260324_153521/history_imputed.csv"
    ground_path = "./data/pm25/Code/STMVL/SampleData/pm25_ground.txt"
    split_ratio = 0.5
    L = 4  # 延迟嵌入长度
    s = 20 # 随机延迟采样数
    target_indices = [0, 1, 2, 3]  # 前4个站点
    
    print(f"\n[参数]")
    print(f"  补值历史: {imputed_history_path}")
    print(f"  真实数据: {ground_path}")
    print(f"  站点: {target_indices}")
    print(f"  L={L}, s={s}")
    
    # 加载数据
    t0 = time.time()
    
    # 加载补值历史
    df_hist = pd.read_csv(imputed_history_path, index_col="datetime", parse_dates=True).sort_index()
    history = df_hist.values.astype(np.float64)
    
    # 加载完整数据用于分割
    df_full = pd.read_csv(ground_path, index_col="datetime", parse_dates=True).sort_index()
    total_len = len(df_full)
    split_point = int(total_len * split_ratio)
    
    print(f"\n[数据加载]")
    print(f"  补值历史形状: {history.shape}")
    print(f"  完整数据长度: {total_len}")
    print(f"  分割点: {split_point}")
    print(f"  历史时间范围: {df_hist.index[0]} ~ {df_hist.index[-1]}")
    
    # 标准化
    scaler = StandardScaler()
    history_scaled = scaler.fit_transform(history)
    
    # 延迟嵌入（简化版）
    T, D = history_scaled.shape
    delays = np.random.randint(1, L+1, size=s)
    X = []
    y = []
    for t in range(max(delays), T):
        features = []
        for delay in delays:
            features.extend(history_scaled[t-delay, :])
        X.append(features)
        y.append(history_scaled[t, target_indices])
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n[延迟嵌入]")
    print(f"  样本数: {len(X)}")
    print(f"  特征维度: {X.shape[1]}")
    print(f"  目标维度: {y.shape[1]}")
    
    # 测试GPR
    print(f"\n[测试GPR]")
    t1 = time.time()
    
    # 取最后一个样本作为测试
    X_test = X[-1:]
    y_test = y[-1:]
    X_train = X[:-1]
    y_train = y[:-1]
    
    # 对每个站点单独预测
    for i, dim in enumerate(target_indices):
        mu, std = gpr_predict(X_train, y_train[:, i], X_test, length_scale=1.0, sigma_f=1.0)
        print(f"  站点 {dim}: 预测={mu[0]:.2f}, 标准差={std[0]:.2f}")
    
    gpr_time = time.time() - t1
    total_time = time.time() - t0
    
    print(f"\n[性能]")
    print(f"  总耗时: {total_time:.1f}s")
    print(f"  GPR耗时: {gpr_time:.1f}s")
    
    print(f"\n" + "="*70)
    print("✅ 测试完成！RDE-GPR 核心功能正常")
    print("="*70)

if __name__ == "__main__":
    main()
