#!/usr/bin/env python3
"""
EEG预测结果对比可视化脚本 - Lorenz96高质量风格版
参考 lorenz96_rde_delay/inference/test_comb_rde.py 的可视化风格

特点：
- 使用 gridspec 复杂布局
- 预测曲线带置信区间
- 清晰的图例、标签和标题
- 高分辨率输出 (dpi=200)
- 支持从ground_path加载真实值
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import os
import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class EEGComparisonVisualizer:
    def __init__(self, out_dir='./save/eeg_comparison_visualization'):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def load_data(self, results_dir, ground_path=None):
        """加载预测结果数据，使用pandas处理表头"""
        data = {}

        # 加载RDE-GPR结果
        rdegpr_path = os.path.join(results_dir, 'rdegpr', 'future_pred.csv')
        if os.path.exists(rdegpr_path):
            df = pd.read_csv(rdegpr_path)
            # 只取目标维度（前3列）
            data['rdegpr'] = df.iloc[:, :min(3, df.shape[1])].values

        # 加载GRU结果
        gru_path = os.path.join(results_dir, 'gru', 'future_pred.csv')
        if os.path.exists(gru_path):
            df = pd.read_csv(gru_path)
            data['gru'] = df.iloc[:, :min(3, df.shape[1])].values

        # 加载LSTM结果
        lstm_path = os.path.join(results_dir, 'lstm', 'future_pred.csv')
        if os.path.exists(lstm_path):
            df = pd.read_csv(lstm_path)
            data['lstm'] = df.iloc[:, :min(3, df.shape[1])].values

        # 加载GRU-ODE-Bayes结果
        gruode_path = os.path.join(results_dir, 'gruodebayes', 'future_pred.csv')
        if os.path.exists(gruode_path):
            df = pd.read_csv(gruode_path)
            data['gruodebayes'] = df.iloc[:, :min(3, df.shape[1])].values
        else:
            data['gruodebayes'] = None

        # 加载真实值
        data['true'] = None

        # 方法1: 从ground_path加载（.npy文件）
        if ground_path and os.path.exists(ground_path):
            print(f"  ✓ 从 {ground_path} 加载真实值")
            gt_data = np.load(ground_path)
            history_timesteps = 100
            horizon_steps = 24
            target_dims = [0, 1, 2]
            data['true'] = gt_data[history_timesteps:history_timesteps+horizon_steps, target_dims]
            print(f"    真实值形状: {data['true'].shape}")

        # 方法2: 从JSON加载
        if data['true'] is None:
            json_path = os.path.join(results_dir, 'comparison_results.json')
            if os.path.exists(json_path):
                import json
                with open(json_path, 'r') as f:
                    results = json.load(f)
                if 'y_true' in results:
                    data['true'] = np.array(results['y_true'])
                    print(f"  ✓ 从JSON加载真实值")

        # 方法3: 尝试从CSV加载
        if data['true'] is None:
            truth_path = os.path.join(results_dir, 'rdegpr', 'future_truth.csv')
            if os.path.exists(truth_path):
                df = pd.read_csv(truth_path)
                data['true'] = df.values
                print(f"  ✓ 从CSV加载真实值")

        if data['true'] is not None:
            print(f"  ✓ 真实值已加载，形状: {data['true'].shape}")
        else:
            print("  ⚠ 未找到真实值，将只绘制预测轨迹对比图")

        return data

    def generate_full_comparison(self, data, dim=0):
        """生成完整的对比图 - Lorenz96风格"""

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.30,
                               height_ratios=[1, 1, 0.8])

        t = np.arange(len(data['rdegpr']))
        has_true = data.get('true') is not None

        # ========== Plot 1: 预测轨迹对比 ==========
        ax1 = fig.add_subplot(gs[0, 0])

        # 真实值
        if has_true:
            ax1.plot(t, data['true'][:, dim], 'k-', lw=2.5, label='Ground Truth', alpha=0.9)

        # RDE-GPR
        ax1.plot(t, data['rdegpr'][:, dim], 'r-', lw=2, label='RDE-GPR', alpha=0.9)

        # GRU
        if 'gru' in data:
            ax1.plot(t, data['gru'][:, dim], 'b--', lw=1.8, label='GRU', alpha=0.85)

        # LSTM
        if 'lstm' in data:
            ax1.plot(t, data['lstm'][:, dim], 'g:', lw=2, label='LSTM', alpha=0.85)

        # GRU-ODE-Bayes
        if data.get('gruodebayes') is not None:
            ax1.plot(t, data['gruodebayes'][:, dim], 'm-.', lw=1.8, label='GRU-ODE-Bayes', alpha=0.85)

        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Value', fontsize=12)
        ax1.set_title(f'EEG Prediction Comparison (dim={dim})', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)

        # ========== Plot 2: 预测误差对比 ==========
        ax2 = fig.add_subplot(gs[0, 1])

        if has_true:
            rdegpr_err = np.abs(data['rdegpr'][:, dim] - data['true'][:, dim])
            ax2.fill_between(t, 0, rdegpr_err, alpha=0.25, color='red', label='RDE-GPR Error')
            ax2.plot(t, rdegpr_err, 'r-', lw=1.5, alpha=0.8)

            if 'gru' in data:
                gru_err = np.abs(data['gru'][:, dim] - data['true'][:, dim])
                ax2.plot(t, gru_err, 'b--', lw=1.5, alpha=0.8, label='GRU Error')

            if 'lstm' in data:
                lstm_err = np.abs(data['lstm'][:, dim] - data['true'][:, dim])
                ax2.plot(t, lstm_err, 'g:', lw=1.5, alpha=0.8, label='LSTM Error')

            if data.get('gruodebayes') is not None:
                gruode_err = np.abs(data['gruodebayes'][:, dim] - data['true'][:, dim])
                ax2.plot(t, gruode_err, 'm-.', lw=1.5, alpha=0.8, label='GRU-ODE-Bayes Error')

            ax2.set_xlabel('Time Step', fontsize=12)
            ax2.set_ylabel('Absolute Error', fontsize=12)
            ax2.set_title('Prediction Error Comparison', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=9, loc='best')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, '需要真实值数据\n请添加 --ground_path 参数',
                     ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Prediction Error (No Ground Truth)', fontsize=14)

        # ========== Plot 3: 散点图（预测值 vs 真实值）==========
        ax3 = fig.add_subplot(gs[1, 0])

        if has_true:
            ax3.scatter(data['true'][:, dim], data['rdegpr'][:, dim],
                       alpha=0.7, color='red', s=40, label='RDE-GPR', edgecolors='white', linewidths=0.5)

            if 'gru' in data:
                ax3.scatter(data['true'][:, dim], data['gru'][:, dim],
                           alpha=0.6, color='blue', s=35, marker='^', label='GRU', edgecolors='white', linewidths=0.5)

            if 'lstm' in data:
                ax3.scatter(data['true'][:, dim], data['lstm'][:, dim],
                           alpha=0.6, color='green', s=35, marker='s', label='LSTM', edgecolors='white', linewidths=0.5)

            if data.get('gruodebayes') is not None:
                ax3.scatter(data['true'][:, dim], data['gruodebayes'][:, dim],
                           alpha=0.6, color='purple', s=35, marker='D', label='GRU-ODE-Bayes', edgecolors='white', linewidths=0.5)

            # y=x 参考线
            min_val = min(data['true'][:, dim].min(), data['rdegpr'][:, dim].min())
            max_val = max(data['true'][:, dim].max(), data['rdegpr'][:, dim].max())
            margin = (max_val - min_val) * 0.05
            ax3.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin],
                    'k--', lw=1.5, alpha=0.6, label='y=x Reference')

            ax3.set_xlabel('Ground Truth', fontsize=12)
            ax3.set_ylabel('Predicted Value', fontsize=12)
            ax3.set_title('Predicted vs Ground Truth', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=8, loc='upper left')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, '需要真实值数据\n请添加 --ground_path 参数',
                     ha='center', va='center', transform=ax3.transAxes, fontsize=14)
            ax3.set_title('Predicted vs Ground Truth (No Data)', fontsize=14)

        # ========== Plot 4: RMSE柱状图 ==========
        ax4 = fig.add_subplot(gs[1, 1])

        methods = ['RDE-GPR']
        rmse_values = []
        mae_values = []
        colors = ['#DD4444']

        if has_true:
            rmse_rdegpr = np.sqrt(np.mean((data['rdegpr'][:, dim] - data['true'][:, dim])**2))
            mae_rdegpr = np.mean(np.abs(data['rdegpr'][:, dim] - data['true'][:, dim]))
            rmse_values.append(rmse_rdegpr)
            mae_values.append(mae_rdegpr)

            if 'gru' in data:
                methods.append('GRU')
                colors.append('#4477DD')
                rmse_gru = np.sqrt(np.mean((data['gru'][:, dim] - data['true'][:, dim])**2))
                mae_gru = np.mean(np.abs(data['gru'][:, dim] - data['true'][:, dim]))
                rmse_values.append(rmse_gru)
                mae_values.append(mae_gru)

            if 'lstm' in data:
                methods.append('LSTM')
                colors.append('#44BB77')
                rmse_lstm = np.sqrt(np.mean((data['lstm'][:, dim] - data['true'][:, dim])**2))
                mae_lstm = np.mean(np.abs(data['lstm'][:, dim] - data['true'][:, dim]))
                rmse_values.append(rmse_lstm)
                mae_values.append(mae_lstm)

            if data.get('gruodebayes') is not None:
                methods.append('GRU-\nODE-Bayes')
                colors.append('#AA4499')
                rmse_gruode = np.sqrt(np.mean((data['gruodebayes'][:, dim] - data['true'][:, dim])**2))
                mae_gruode = np.mean(np.abs(data['gruodebayes'][:, dim] - data['true'][:, dim]))
                rmse_values.append(rmse_gruode)
                mae_values.append(mae_gruode)

        x_pos = np.arange(len(methods))

        if rmse_values:
            bars = ax4.bar(x_pos, rmse_values, color=colors, alpha=0.75, edgecolor='white', linewidth=1.5)

            for bar, v in zip(bars, rmse_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(rmse_values)*0.02,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(methods, fontsize=10)
            ax4.set_ylabel('RMSE', fontsize=12)
            ax4.grid(True, alpha=0.3, axis='y')
        else:
            ax4.text(0.5, 0.5, '需要真实值数据\n请添加 --ground_path 参数',
                     ha='center', va='center', transform=ax4.transAxes, fontsize=14)

        ax4.set_title('RMSE Comparison', fontsize=14, fontweight='bold')

        # ========== Plot 5: MAE柱状图 ==========
        ax5 = fig.add_subplot(gs[2, :])

        if mae_values:
            bars2 = ax5.bar(x_pos, mae_values, color=colors, alpha=0.75, edgecolor='white', linewidth=1.5)

            for bar, v in zip(bars2, mae_values):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + max(mae_values)*0.02,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(methods, fontsize=10)
            ax5.set_ylabel('MAE', fontsize=12)
            ax5.grid(True, alpha=0.3, axis='y')
        else:
            ax5.text(0.5, 0.5, '需要真实值数据\n请添加 --ground_path 参数',
                     ha='center', va='center', transform=ax5.transAxes, fontsize=14)

        ax5.set_title('MAE Comparison', fontsize=14, fontweight='bold')

        plt.suptitle('EEG Forecast Comparison: RDE-GPR vs Baselines',
                     fontsize=16, fontweight='bold', y=1.01)

        output_path = os.path.join(self.out_dir, f'full_comparison_dim{dim}_{timestamp}.png')
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'✓ 生成完整对比图: {output_path}')

        return {
            'rmse': dict(zip(methods, rmse_values)) if rmse_values else {},
            'mae': dict(zip(methods, mae_values)) if mae_values else {}
        }

    def generate_metrics_summary(self, all_metrics):
        """生成指标汇总表格"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        rows = []
        for dim_metrics in all_metrics:
            row = {}
            for metric_name, values in dim_metrics.items():
                for method, value in values.items():
                    key = f'{method}_{metric_name}'
                    row[key] = value
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            csv_path = os.path.join(self.out_dir, f'metrics_summary_{timestamp}.csv')
            df.to_csv(csv_path, index=False)
            print(f'✓ 生成指标汇总: {csv_path}')

        return rows

    def run(self, results_dir, ground_path=None):
        """运行完整的可视化流程"""
        print('=' * 60)
        print('EEG预测结果对比可视化 (Lorenz96风格)')
        print('=' * 60)
        print(f'\n加载数据: {results_dir}')

        data = self.load_data(results_dir, ground_path=ground_path)
        print(f'  ✓ 加载方法数: {len(data)}')

        all_metrics = []

        # 为每个目标维度生成对比图
        n_dims = min(3, data['rdegpr'].shape[1]) if 'rdegpr' in data else 0

        for dim in range(n_dims):
            print(f'\n--- 生成维度 {dim} 的对比图 ---')
            metrics = self.generate_full_comparison(data, dim=dim)
            all_metrics.append(metrics)

        # 生成指标汇总
        if all_metrics:
            self.generate_metrics_summary(all_metrics)

        print('\n' + '=' * 60)
        print(f'可视化完成! 结果保存在: {self.out_dir}')
        print('=' * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='EEG预测结果对比可视化 (Lorenz96风格)')
    parser.add_argument('--results_dir', type=str, default='./save/eeg_comparison',
                        help='预测结果目录')
    parser.add_argument('--out_dir', type=str, default='./save/eeg_comparison_visualization',
                        help='输出目录')
    parser.add_argument('--ground_path', type=str, default=None,
                        help='真实值路径 (.npy 文件)')

    args = parser.parse_args()

    visualizer = EEGComparisonVisualizer(out_dir=args.out_dir)
    visualizer.run(args.results_dir, ground_path=args.ground_path)
