#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PM2.5数据集统计分析脚本
分析数据的形状、时间范围、频率等基本信息
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def analyze_pm25_data():
    """分析PM2.5数据集的详细信息"""
    
    print("=" * 80)
    print("PM2.5 数据集统计分析")
    print("=" * 80)
    
    # 读取数据文件
    try:
        df_ground = pd.read_csv(
            "./data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
            index_col="datetime",
            parse_dates=True,
        )
        print("\n✓ 成功读取 pm25_ground.txt")
    except Exception as e:
        print(f"\n✗ 读取 pm25_ground.txt 失败: {e}")
        return
    
    try:
        df_missing = pd.read_csv(
            "./data/pm25/Code/STMVL/SampleData/pm25_missing.txt",
            index_col="datetime",
            parse_dates=True,
        )
        print("✓ 成功读取 pm25_missing.txt")
    except Exception as e:
        print(f"✗ 读取 pm25_missing.txt 失败: {e}")
        return
    
    print("\n" + "=" * 80)
    print("1. 基本信息")
    print("=" * 80)
    
    # 数据形状
    print(f"\n【数据形状】")
    print(f"  Ground Truth 数据: {df_ground.shape[0]} 行 × {df_ground.shape[1]} 列")
    print(f"  Missing 数据:      {df_missing.shape[0]} 行 × {df_missing.shape[1]} 列")
    
    # 列名（站点）
    print(f"\n【站点信息】")
    print(f"  站点数量: {df_ground.shape[1]} 个")
    print(f"  站点列表: {list(df_ground.columns)}")
    
    # 时间范围
    print(f"\n【时间范围】")
    print(f"  开始时间: {df_ground.index.min()}")
    print(f"  结束时间: {df_ground.index.max()}")
    print(f"  时间跨度: {(df_ground.index.max() - df_ground.index.min()).days} 天")
    
    # 时间频率
    print(f"\n【时间频率】")
    time_diff = df_ground.index.to_series().diff()
    freq_counts = time_diff.value_counts().head(5)
    print(f"  最常见的时间间隔:")
    for interval, count in freq_counts.items():
        if pd.notna(interval):
            hours = interval.total_seconds() / 3600
            print(f"    {hours:.1f} 小时: {count} 次")
    
    # 判断是否是每小时数据
    mode_interval = time_diff.mode()[0]
    if pd.notna(mode_interval):
        mode_hours = mode_interval.total_seconds() / 3600
        if mode_hours == 1.0:
            print(f"  → 数据频率: 每小时一次")
        elif mode_hours == 24.0:
            print(f"  → 数据频率: 每天一次")
        else:
            print(f"  → 数据频率: 每 {mode_hours} 小时一次")
    
    print("\n" + "=" * 80)
    print("2. 月份分布")
    print("=" * 80)
    
    # 按月份统计
    print(f"\n【各月份数据量】")
    month_counts = df_ground.groupby(df_ground.index.month).size()
    for month, count in month_counts.items():
        print(f"  {month:2d}月: {count:5d} 条记录")
    
    # 按年份和月份统计
    print(f"\n【按年份-月份统计】")
    year_month_counts = df_ground.groupby([df_ground.index.year, df_ground.index.month]).size()
    print(year_month_counts.to_string())
    
    print("\n" + "=" * 80)
    print("3. 缺失值分析")
    print("=" * 80)
    
    # Ground Truth 缺失值
    print(f"\n【Ground Truth 数据缺失情况】")
    ground_null_count = df_ground.isnull().sum()
    ground_null_ratio = (ground_null_count / len(df_ground) * 100)
    print(f"  总缺失值: {df_ground.isnull().sum().sum()} 个")
    print(f"  缺失率: {df_ground.isnull().sum().sum() / df_ground.size * 100:.2f}%")
    print(f"\n  各站点缺失情况:")
    for col in df_ground.columns:
        null_count = ground_null_count[col]
        null_pct = ground_null_ratio[col]
        print(f"    {col}: {null_count:5d} ({null_pct:6.2f}%)")
    
    # Missing 数据缺失值
    print(f"\n【Missing 数据缺失情况】")
    missing_null_count = df_missing.isnull().sum()
    missing_null_ratio = (missing_null_count / len(df_missing) * 100)
    print(f"  总缺失值: {df_missing.isnull().sum().sum()} 个")
    print(f"  缺失率: {df_missing.isnull().sum().sum() / df_missing.size * 100:.2f}%")
    print(f"\n  各站点缺失情况:")
    for col in df_missing.columns:
        null_count = missing_null_count[col]
        null_pct = missing_null_ratio[col]
        print(f"    {col}: {null_count:5d} ({null_pct:6.2f}%)")
    
    print("\n" + "=" * 80)
    print("4. 数值统计")
    print("=" * 80)
    
    # 基本统计信息
    print(f"\n【Ground Truth 数据统计】")
    print(df_ground.describe().to_string())
    
    print("\n" + "=" * 80)
    print("5. 样本数据预览")
    print("=" * 80)
    
    print(f"\n【前5条记录 - Ground Truth】")
    print(df_ground.head().to_string())
    
    print(f"\n【前5条记录 - Missing】")
    print(df_missing.head().to_string())
    
    print("\n" + "=" * 80)
    print("6. 数据切分预览（按 split_ratio=0.5）")
    print("=" * 80)
    
    # 按月份展示切分情况
    print(f"\n【各月份切分后的数据量】")
    for month in df_ground.index.month.unique():
        month_data = df_ground[df_ground.index.month == month]
        total_len = len(month_data)
        split_point = int(total_len * 0.5)
        
        print(f"\n  {month:2d}月:")
        print(f"    总长度:     {total_len:5d}")
        print(f"    前50%(补值): {split_point:5d}")
        print(f"    后50%(预测): {total_len - split_point:5d}")
        print(f"    时间范围:   {month_data.index.min()} 至 {month_data.index.max()}")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    analyze_pm25_data()

