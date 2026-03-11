# -*- coding: utf-8 -*-
"""
网格搜索结果分析脚本
用于深入分析网格搜索的结果，生成额外的可视化和统计报告

用法:
python analyze_grid_results.py <grid_search_output_dir>

示例:
python analyze_grid_results.py ./save/pm25_grid_search_20260129_120000/
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_results(output_dir):
    """加载网格搜索结果"""
    results_path = os.path.join(output_dir, "grid_search_results.csv")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"找不到结果文件: {results_path}")
    
    df = pd.read_csv(results_path)
    return df


def analyze_parameter_importance(df, output_dir):
    """分析参数重要性"""
    print("\n=== 参数重要性分析 ===")
    
    successful = df[df['status'] == 'success'].copy()
    if len(successful) == 0:
        print("没有成功的运行结果")
        return
    
    # 计算每个L值的平均性能
    L_stats = successful.groupby('L').agg({
        'rmse': ['mean', 'std', 'min', 'max'],
        'mae': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print("\n按L分组的统计:")
    print(L_stats)
    
    # 计算每个trainlength的平均性能
    trainlen_stats = successful.groupby('trainlength').agg({
        'rmse': ['mean', 'std', 'min', 'max'],
        'mae': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print("\n按trainlength分组的统计:")
    print(trainlen_stats)
    
    # 保存统计结果
    L_stats.to_csv(os.path.join(output_dir, "analysis_L_statistics.csv"))
    trainlen_stats.to_csv(os.path.join(output_dir, "analysis_trainlen_statistics.csv"))
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # L vs RMSE (箱线图)
    successful.boxplot(column='rmse', by='L', ax=axes[0, 0])
    axes[0, 0].set_title('RMSE Distribution by L', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('L')
    axes[0, 0].set_ylabel('RMSE')
    plt.sca(axes[0, 0])
    plt.xticks(rotation=0)
    
    # trainlength vs RMSE (箱线图)
    successful.boxplot(column='rmse', by='trainlength', ax=axes[0, 1])
    axes[0, 1].set_title('RMSE Distribution by trainlength', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('trainlength')
    axes[0, 1].set_ylabel('RMSE')
    plt.sca(axes[0, 1])
    plt.xticks(rotation=45)
    
    # L vs MAE (箱线图)
    successful.boxplot(column='mae', by='L', ax=axes[1, 0])
    axes[1, 0].set_title('MAE Distribution by L', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('L')
    axes[1, 0].set_ylabel('MAE')
    plt.sca(axes[1, 0])
    plt.xticks(rotation=0)
    
    # trainlength vs MAE (箱线图)
    successful.boxplot(column='mae', by='trainlength', ax=axes[1, 1])
    axes[1, 1].set_title('MAE Distribution by trainlength', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('trainlength')
    axes[1, 1].set_ylabel('MAE')
    plt.sca(axes[1, 1])
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'analysis_parameter_boxplots.png'), dpi=200)
    plt.close()
    
    print(f"\n箱线图已保存: analysis_parameter_boxplots.png")


def analyze_interactions(df, output_dir):
    """分析参数交互效应"""
    print("\n=== 参数交互效应分析 ===")
    
    successful = df[df['status'] == 'success'].copy()
    if len(successful) == 0:
        return
    
    # 创建交互图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # RMSE交互效应
    pivot_rmse = successful.pivot_table(values='rmse', index='trainlength', columns='L')
    sns.heatmap(pivot_rmse, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=axes[0], cbar_kws={'label': 'RMSE'})
    axes[0].set_title('RMSE: Parameter Interaction', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('L')
    axes[0].set_ylabel('trainlength')
    
    # MAE交互效应
    pivot_mae = successful.pivot_table(values='mae', index='trainlength', columns='L')
    sns.heatmap(pivot_mae, annot=True, fmt='.2f', cmap='RdYlGn_r', ax=axes[1], cbar_kws={'label': 'MAE'})
    axes[1].set_title('MAE: Parameter Interaction', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('L')
    axes[1].set_ylabel('trainlength')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'analysis_interaction_heatmaps.png'), dpi=200)
    plt.close()
    
    print(f"交互效应热力图已保存: analysis_interaction_heatmaps.png")


def analyze_performance_distribution(df, output_dir):
    """分析性能分布"""
    print("\n=== 性能分布分析 ===")
    
    successful = df[df['status'] == 'success'].copy()
    if len(successful) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # RMSE直方图
    axes[0, 0].hist(successful['rmse'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(successful['rmse'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 0].axvline(successful['rmse'].median(), color='green', linestyle='--', linewidth=2, label='Median')
    axes[0, 0].set_xlabel('RMSE')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('RMSE Distribution', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE直方图
    axes[0, 1].hist(successful['mae'], bins=20, color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(successful['mae'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 1].axvline(successful['mae'].median(), color='green', linestyle='--', linewidth=2, label='Median')
    axes[0, 1].set_xlabel('MAE')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('MAE Distribution', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSE vs MAE散点图
    axes[1, 0].scatter(successful['rmse'], successful['mae'], alpha=0.6, s=100, c=successful['L'], cmap='viridis')
    axes[1, 0].set_xlabel('RMSE')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('RMSE vs MAE (colored by L)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0])
    cbar.set_label('L')
    
    # 计算时间分布
    axes[1, 1].scatter(successful['L'], successful['elapsed_time'], alpha=0.6, s=100, c=successful['trainlength'], cmap='plasma')
    axes[1, 1].set_xlabel('L')
    axes[1, 1].set_ylabel('Elapsed Time (seconds)')
    axes[1, 1].set_title('Computation Time (colored by trainlength)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('trainlength')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'analysis_performance_distribution.png'), dpi=200)
    plt.close()
    
    print(f"性能分布图已保存: analysis_performance_distribution.png")
    
    # 打印统计摘要
    print(f"\nRMSE统计:")
    print(f"  Mean: {successful['rmse'].mean():.4f}")
    print(f"  Median: {successful['rmse'].median():.4f}")
    print(f"  Std: {successful['rmse'].std():.4f}")
    print(f"  Min: {successful['rmse'].min():.4f}")
    print(f"  Max: {successful['rmse'].max():.4f}")
    
    print(f"\nMAE统计:")
    print(f"  Mean: {successful['mae'].mean():.4f}")
    print(f"  Median: {successful['mae'].median():.4f}")
    print(f"  Std: {successful['mae'].std():.4f}")
    print(f"  Min: {successful['mae'].min():.4f}")
    print(f"  Max: {successful['mae'].max():.4f}")


def find_pareto_frontier(df, output_dir):
    """找出帕累托前沿（在RMSE-计算时间权衡下的最优解集）"""
    print("\n=== 帕累托前沿分析 ===")
    
    successful = df[df['status'] == 'success'].copy()
    if len(successful) == 0:
        return
    
    # 找出帕累托最优解
    pareto_optimal = []
    for idx, row in successful.iterrows():
        is_dominated = False
        for _, other_row in successful.iterrows():
            # 如果另一个解在RMSE和时间上都不差于当前解
            if (other_row['rmse'] <= row['rmse'] and 
                other_row['elapsed_time'] <= row['elapsed_time'] and
                (other_row['rmse'] < row['rmse'] or other_row['elapsed_time'] < row['elapsed_time'])):
                is_dominated = True
                break
        if not is_dominated:
            pareto_optimal.append(idx)
    
    pareto_df = successful.loc[pareto_optimal].sort_values('rmse')
    
    print(f"\n找到 {len(pareto_df)} 个帕累托最优解:")
    print(pareto_df[['L', 'trainlength', 'rmse', 'mae', 'elapsed_time']])
    
    # 保存帕累托最优解
    pareto_df.to_csv(os.path.join(output_dir, 'analysis_pareto_optimal.csv'), index=False)
    
    # 可视化帕累托前沿
    plt.figure(figsize=(12, 7))
    plt.scatter(successful['elapsed_time'], successful['rmse'], 
               alpha=0.5, s=100, c='lightblue', edgecolors='black', label='All solutions')
    plt.scatter(pareto_df['elapsed_time'], pareto_df['rmse'], 
               alpha=0.9, s=150, c='red', marker='*', edgecolors='darkred', label='Pareto optimal')
    
    # 连接帕累托前沿点
    pareto_sorted = pareto_df.sort_values('elapsed_time')
    plt.plot(pareto_sorted['elapsed_time'], pareto_sorted['rmse'], 
            'r--', linewidth=2, alpha=0.5)
    
    # 标注帕累托点
    for _, row in pareto_df.iterrows():
        plt.annotate(f"L={int(row['L'])}\nT={int(row['trainlength'])}", 
                    xy=(row['elapsed_time'], row['rmse']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=8, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    plt.xlabel('Computation Time (seconds)', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Pareto Frontier: RMSE vs Computation Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'analysis_pareto_frontier.png'), dpi=200)
    plt.close()
    
    print(f"\n帕累托前沿图已保存: analysis_pareto_frontier.png")


def analyze_sensitivity(df, output_dir):
    """敏感性分析：参数变化对性能的影响"""
    print("\n=== 敏感性分析 ===")
    
    successful = df[df['status'] == 'success'].copy()
    if len(successful) == 0:
        return
    
    # 计算参数变化的影响
    L_sensitivity = []
    for L in sorted(successful['L'].unique()):
        subset = successful[successful['L'] == L]
        rmse_range = subset['rmse'].max() - subset['rmse'].min()
        rmse_std = subset['rmse'].std()
        L_sensitivity.append({
            'L': L,
            'rmse_range': rmse_range,
            'rmse_std': rmse_std,
            'rmse_cv': rmse_std / subset['rmse'].mean() if subset['rmse'].mean() > 0 else 0
        })
    
    trainlen_sensitivity = []
    for trainlen in sorted(successful['trainlength'].unique()):
        subset = successful[successful['trainlength'] == trainlen]
        rmse_range = subset['rmse'].max() - subset['rmse'].min()
        rmse_std = subset['rmse'].std()
        trainlen_sensitivity.append({
            'trainlength': trainlen,
            'rmse_range': rmse_range,
            'rmse_std': rmse_std,
            'rmse_cv': rmse_std / subset['rmse'].mean() if subset['rmse'].mean() > 0 else 0
        })
    
    L_sens_df = pd.DataFrame(L_sensitivity)
    trainlen_sens_df = pd.DataFrame(trainlen_sensitivity)
    
    print("\nL的敏感性 (固定trainlength时RMSE的变化):")
    print(L_sens_df)
    
    print("\ntrainlength的敏感性 (固定L时RMSE的变化):")
    print(trainlen_sens_df)
    
    # 保存敏感性分析结果
    L_sens_df.to_csv(os.path.join(output_dir, 'analysis_L_sensitivity.csv'), index=False)
    trainlen_sens_df.to_csv(os.path.join(output_dir, 'analysis_trainlen_sensitivity.csv'), index=False)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar(L_sens_df['L'], L_sens_df['rmse_range'], color='steelblue', alpha=0.7)
    axes[0].set_xlabel('L', fontsize=11)
    axes[0].set_ylabel('RMSE Range', fontsize=11)
    axes[0].set_title('Sensitivity of L\n(RMSE range when trainlength varies)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(trainlen_sens_df['trainlength'], trainlen_sens_df['rmse_range'], color='coral', alpha=0.7)
    axes[1].set_xlabel('trainlength', fontsize=11)
    axes[1].set_ylabel('RMSE Range', fontsize=11)
    axes[1].set_title('Sensitivity of trainlength\n(RMSE range when L varies)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'analysis_sensitivity.png'), dpi=200)
    plt.close()
    
    print(f"\n敏感性分析图已保存: analysis_sensitivity.png")


def generate_recommendations(df, output_dir):
    """生成参数选择建议"""
    print("\n=== 参数选择建议 ===")
    
    successful = df[df['status'] == 'success'].copy()
    if len(successful) == 0:
        print("没有成功的运行结果")
        return
    
    recommendations = {}
    
    # 1. 最佳性能（不考虑计算时间）
    best_rmse = successful.loc[successful['rmse'].idxmin()]
    recommendations['best_performance'] = {
        'L': int(best_rmse['L']),
        'trainlength': int(best_rmse['trainlength']),
        'rmse': float(best_rmse['rmse']),
        'mae': float(best_rmse['mae']),
        'time': float(best_rmse['elapsed_time']),
        'reason': '最低RMSE，适合追求最佳预测精度'
    }
    
    # 2. 最快速度（在RMSE < 平均值+0.5std的范围内）
    threshold = successful['rmse'].mean() + 0.5 * successful['rmse'].std()
    good_enough = successful[successful['rmse'] <= threshold]
    if len(good_enough) > 0:
        fastest = good_enough.loc[good_enough['elapsed_time'].idxmin()]
        recommendations['fastest_good'] = {
            'L': int(fastest['L']),
            'trainlength': int(fastest['trainlength']),
            'rmse': float(fastest['rmse']),
            'mae': float(fastest['mae']),
            'time': float(fastest['elapsed_time']),
            'reason': '在保证良好性能的前提下计算最快'
        }
    
    # 3. 平衡性能和速度
    # 计算标准化分数（RMSE越小越好，时间越短越好）
    rmse_normalized = (successful['rmse'] - successful['rmse'].min()) / (successful['rmse'].max() - successful['rmse'].min())
    time_normalized = (successful['elapsed_time'] - successful['elapsed_time'].min()) / (successful['elapsed_time'].max() - successful['elapsed_time'].min())
    successful['composite_score'] = rmse_normalized + time_normalized
    
    balanced = successful.loc[successful['composite_score'].idxmin()]
    recommendations['balanced'] = {
        'L': int(balanced['L']),
        'trainlength': int(balanced['trainlength']),
        'rmse': float(balanced['rmse']),
        'mae': float(balanced['mae']),
        'time': float(balanced['elapsed_time']),
        'reason': '性能和计算时间的最佳平衡'
    }
    
    # 4. 稳健选择（中位数性能）
    median_rmse = successful['rmse'].median()
    robust = successful.iloc[(successful['rmse'] - median_rmse).abs().argsort()[:1]].iloc[0]
    recommendations['robust'] = {
        'L': int(robust['L']),
        'trainlength': int(robust['trainlength']),
        'rmse': float(robust['rmse']),
        'mae': float(robust['mae']),
        'time': float(robust['elapsed_time']),
        'reason': '中等性能，较为稳健的选择'
    }
    
    # 保存建议
    with open(os.path.join(output_dir, 'analysis_recommendations.json'), 'w') as f:
        json.dump(recommendations, f, indent=4)
    
    # 打印建议
    print("\n推荐参数组合:")
    for key, rec in recommendations.items():
        print(f"\n[{key.upper().replace('_', ' ')}]")
        print(f"  L = {rec['L']}, trainlength = {rec['trainlength']}")
        print(f"  RMSE = {rec['rmse']:.4f}, MAE = {rec['mae']:.4f}")
        print(f"  计算时间 = {rec['time']:.2f} 秒")
        print(f"  原因: {rec['reason']}")
    
    print(f"\n详细建议已保存: analysis_recommendations.json")


def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_grid_results.py <grid_search_output_dir>")
        print("\n示例:")
        print("  python analyze_grid_results.py ./save/pm25_grid_search_20260129_120000/")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    if not os.path.exists(output_dir):
        print(f"错误: 目录不存在: {output_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("网格搜索结果深度分析")
    print("=" * 80)
    print(f"\n分析目录: {output_dir}")
    
    try:
        # 加载结果
        df = load_results(output_dir)
        print(f"\n加载了 {len(df)} 条结果记录")
        print(f"成功: {(df['status']=='success').sum()}, 失败: {(df['status']!='success').sum()}")
        
        # 执行各项分析
        analyze_parameter_importance(df, output_dir)
        analyze_interactions(df, output_dir)
        analyze_performance_distribution(df, output_dir)
        find_pareto_frontier(df, output_dir)
        analyze_sensitivity(df, output_dir)
        generate_recommendations(df, output_dir)
        
        print("\n" + "=" * 80)
        print("分析完成！")
        print("=" * 80)
        print(f"\n所有分析结果已保存到: {output_dir}")
        print("\n生成的文件:")
        print("  - analysis_*.csv (统计数据)")
        print("  - analysis_*.png (可视化图表)")
        print("  - analysis_recommendations.json (参数建议)")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
