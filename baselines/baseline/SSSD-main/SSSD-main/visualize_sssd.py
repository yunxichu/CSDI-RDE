# -*- coding: utf-8 -*-
"""
PM2.5 SSSD 预测结果可视化脚本
生成预测值与真实值的对比图
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def load_data(pred_path, ground_path, split_ratio=0.5):
    """加载预测数据和真实数据"""
    # 加载预测数据
    df_pred = pd.read_csv(pred_path, index_col="datetime", parse_dates=True)
    
    # 加载真实数据
    df_ground = pd.read_csv(ground_path, index_col="datetime", parse_dates=True)
    
    # 按 split_ratio 分割
    split_point = int(len(df_ground) * split_ratio)
    df_fut = df_ground.iloc[split_point:]
    
    # 截取与预测结果相同长度的数据
    df_true = df_fut.iloc[:len(df_pred)]
    
    return df_pred, df_true


def compute_metrics(y_true, y_pred):
    """计算评估指标"""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan}
    diff = y_true[mask] - y_pred[mask]
    return {"rmse": float(np.sqrt(np.mean(diff**2))), "mae": float(np.mean(np.abs(diff)))} 

def plot_forecast(pred, true, out_dir, plot_dim=0):
    """绘制预测与真实值对比图"""
    # 选择维度
    d = max(0, min(plot_dim, pred.shape[1] - 1))
    col_name = pred.columns[d]
    
    plt.figure(figsize=(14, 6))
    plt.plot(true.index, true.iloc[:, d], label="真实值", color="steelblue", linewidth=2)
    plt.plot(pred.index, pred.iloc[:, d], label="SSSD预测", color="tomato", linewidth=2, linestyle="--")
    plt.title(f"PM2.5 预测 vs 真实值 ({col_name})", fontsize=14)
    plt.xlabel("时间", fontsize=12)
    plt.ylabel("PM2.5 浓度", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"forecast_dim{d}.png"), dpi=150)
    plt.close()

def plot_rmse_bar(pred, true, out_dir):
    """绘制各维度RMSE柱状图"""
    rmse_list = []
    for col in pred.columns:
        y_true = true[col].values
        y_pred = pred[col].values
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if mask.sum() > 0:
            rmse = np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2))
        else:
            rmse = np.nan
        rmse_list.append(rmse)
    
    plt.figure(figsize=(16, 6))
    plt.bar(np.arange(len(rmse_list)), rmse_list)
    plt.title("各站点 RMSE", fontsize=14)
    plt.xlabel("站点ID", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.xticks(np.arange(len(rmse_list)), pred.columns, rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rmse_per_dim.png"), dpi=150)
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, default="./save/pm25_sssd_0.5_42_20260310_210817/future_pred.csv")
    parser.add_argument("--ground_path", type=str, default="/home/rhl/CSDI-main_test/data/pm25/Code/STMVL/SampleData/pm25_ground.txt")
    parser.add_argument("--split_ratio", type=float, default=0.5)
    parser.add_argument("--plot_dim", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=".")
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 加载数据
    print("加载数据...")
    df_pred, df_true = load_data(args.pred_path, args.ground_path, args.split_ratio)
    
    # 计算整体指标
    y_true = df_true.values
    y_pred = df_pred.values
    metrics = compute_metrics(y_true, y_pred)
    print(f"整体 RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    
    # 绘制预测对比图
    print("绘制预测对比图...")
    plot_forecast(df_pred, df_true, args.out_dir, args.plot_dim)
    
    # 绘制RMSE柱状图
    print("绘制RMSE柱状图...")
    plot_rmse_bar(df_pred, df_true, args.out_dir)
    
    print(f"\n可视化完成！结果保存在: {args.out_dir}")


if __name__ == "__main__":
    main()
