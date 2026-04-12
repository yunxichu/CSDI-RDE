# -*- coding: utf-8 -*-
"""
RDE-GPR 数据加载和初始化测试
验证补值数据和GPR初始化是否正常
"""

import os, sys, json, time
import numpy as np
import pandas as pd
from rde_gpr.pm25_CSDIimpute_after_RDEgpr import PM25RDEGPR

if __name__ == "__main__":
    print("="*70)
    print("RDE-GPR 快速测试 - 数据加载和初始化")
    print("="*70)
    
    # 测试参数
    imputed_history_path = "./save/pm25_history_imputed_split0.5_seed42_20260324_153521/history_imputed.csv"
    ground_path = "./data/pm25/Code/STMVL/SampleData/pm25_ground.txt"
    split_ratio = 0.5
    horizon_days = 1
    L = 4
    s = 20
    trainlength = 1000
    target_indices = [0, 1, 2, 3]  # 前4个站点
    n_jobs = 4
    
    print(f"\n[参数]")
    print(f"  补值历史: {imputed_history_path}")
    print(f"  真实数据: {ground_path}")
    print(f"  站点: {target_indices}")
    print(f"  L={L}, s={s}, trainlength={trainlength}")
    
    # 加载数据
    t0 = time.time()
    model = PM25RDEGPR(
        imputed_history_path=imputed_history_path,
        ground_path=ground_path,
        split_ratio=split_ratio,
        horizon_days=horizon_days,
        L=L, s=s, trainlength=trainlength,
        target_indices=target_indices,
        n_jobs=n_jobs
    )
    load_time = time.time() - t0
    print(f"\n[加载完成]")
    print(f"  耗时: {load_time:.1f}s")
    print(f"  历史长度: {model.hist_len}")
    print(f"  未来长度: {model.fut_len}")
    print(f"  预测步数: {model.horizon}")
    print(f"  站点数: {len(model.target_indices)}")
    
    # 测试第一步预测
    print(f"\n[测试第一步预测]")
    t1 = time.time()
    y0, std0 = model.predict_one_step(model.X_hist[-model.L:])
    pred_time = time.time() - t1
    print(f"  耗时: {pred_time:.1f}s")
    print(f"  预测值: {y0}")
    print(f"  标准差: {std0}")
    
    print(f"\n[数据质量检查]")
    print(f"  历史数据形状: {model.X_hist.shape}")
    print(f"  历史数据范围: [{model.X_hist.min():.2f}, {model.X_hist.max():.2f}]")
    print(f"  缺失值: {np.isnan(model.X_hist).sum()}")
    
    print(f"\n" + "="*70)
    print("✅ 测试完成！RDE-GPR 初始化正常")
    print("="*70)
