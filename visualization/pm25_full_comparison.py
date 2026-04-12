#!/usr/bin/env python3
"""
PM2.5预测结果对比可视化脚本 - Lorenz96高质量风格版 v2
参考 lorenz96_rde_delay/inference/test_comb_rde.py 的可视化风格

修复：
- 按split_ratio正确匹配ground truth（而非取最后N行）
- 支持置信区间（future_pred_std）
- 5面板布局：数据概览 | 预测轨迹(带CI) | 散点图 | 误差分析 | RMSE/MAE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import os
import datetime
import json

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class PM25ComparisonVisualizer:
    def __init__(self, out_dir='./save/pm25_comparison_visualization'):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def _load_csv_numeric(self, path):
        df = pd.read_csv(path)
        if df.shape[1] > 1:
            try:
                float(df.iloc[0, 0])
            except (ValueError, TypeError):
                df = df.iloc[:, 1:]
        return df.values.astype(float), df.columns.tolist()

    def load_data(self, results_dir, ground_path=None, split_ratio=0.5, horizon_steps=24):
        data = {}
        
        rdegpr_path = os.path.join(results_dir, 'rdegpr', 'future_pred.csv')
        if os.path.exists(rdegpr_path):
            pred_vals, pred_cols = self._load_csv_numeric(rdegpr_path)
            data['rdegpr'] = pred_vals
            data['pred_cols'] = pred_cols
            
            rdegpr_std_path = os.path.join(results_dir, 'rdegpr', 'future_pred_std.csv')
            if os.path.exists(rdegpr_std_path):
                std_vals, _ = self._load_csv_numeric(rdegpr_std_path)
                data['rdegpr_std'] = std_vals
        
        for method in ['gru', 'lstm', 'neuralcde', 'gruodebayes']:
            m_path = os.path.join(results_dir, method, 'future_pred.csv')
            if os.path.exists(m_path):
                vals, _ = self._load_csv_numeric(m_path)
                data[method] = vals
        
        if ground_path is not None and os.path.exists(ground_path):
            gt_df = pd.read_csv(ground_path)
            if gt_df.shape[1] > 1:
                try:
                    float(gt_df.iloc[0, 0])
                except (ValueError, TypeError):
                    gt_df = gt_df.iloc[:, 1:]
            
            total_len = len(gt_df)
            split_point = int(total_len * split_ratio)
            
            gt_future = gt_df.values[split_point:split_point + horizon_steps].astype(float)
            
            actual_horizon = min(horizon_steps, len(gt_df) - split_point, 
                                 pred_vals.shape[0] if 'rdegpr' in data else horizon_steps)
            gt_future = gt_df.values[split_point:split_point + actual_horizon].astype(float)
            
            data['true'] = gt_future
            data['split_info'] = {
                'total_len': total_len,
                'split_point': split_point,
                'horizon': actual_horizon,
                'gt_start_idx': split_point,
                'gt_end_idx': split_point + actual_horizon
            }
            
            print(f'  ✓ Ground truth: 总{total_len}行, 分割点={split_point}, 取[{split_point}:{split_point+actual_horizon}]共{actual_horizon}行')
        else:
            json_path = os.path.join(results_dir, 'comparison_results.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    results = json.load(f)
                if 'y_true' in results:
                    data['true'] = np.array(results['y_true'])
        
        return data

    def generate_full_comparison(self, data, dim=0, station_name='Station', is_persistence=False):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        has_std = 'rdegpr_std' in data and data['rdegpr_std'] is not None
        has_true = 'true' in data and data['true'] is not None
        
        n_rows = 3
        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.38, wspace=0.30)
        
        t = np.arange(len(data['rdegpr']))
        
        ax1 = fig.add_subplot(gs[0, :])
        
        if has_true:
            ax1.plot(t, data['true'][:, dim], 'k-', lw=2.5, label='Ground Truth', alpha=0.9, zorder=5)
        
        method_label = 'RDE-GPR' if not is_persistence else 'RDE-GPR (Persistence Fallback)'
        
        if has_std and not is_persistence:
            std_dim = data['rdegpr_std'][:, dim] if dim < data['rdegpr_std'].shape[1] else np.zeros(len(t))
            ax1.plot(t, data['rdegpr'][:, dim], 'r-', lw=2.2, label=method_label, alpha=0.9, zorder=4)
            ax1.fill_between(t, 
                           data['rdegpr'][:, dim] - 2*std_dim,
                           data['rdegpr'][:, dim] + 2*std_dim,
                           alpha=0.18, color='red', label='95% CI (±2σ)')
        elif is_persistence:
            ax1.plot(t, data['rdegpr'][:, dim], 'r--', lw=1.8, label=method_label, alpha=0.6, zorder=4)
            ax1.set_title(f'⚠️ {station_name} (dim={dim}) - PERSISTENCE (非GPR预测)\n'
                         f'该维度target_indices未包含，显示的是naive persistence baseline',
                         fontsize=12, fontweight='bold', color='darkred')
        else:
            ax1.plot(t, data['rdegpr'][:, dim], 'r-', lw=2.2, label=method_label, alpha=0.9)
        
        for method, style in [('gru', ('b--', 1.8, '^')), ('lstm', ('g:', 2, 's')),
                              ('neuralcde', ('m-.', 1.8, 'D')), ('gruodebayes', ('c--', 1.8, 'v'))]:
            if method in data:
                ls, lw, mk = style
                ax1.plot(t, data[method][:, dim], ls, lw=lw, label=method.upper(), alpha=0.8)
        
        ax1.set_xlabel('Time Step (Hours)', fontsize=12)
        ax1.set_ylabel('PM2.5 Concentration (μg/m³)', fontsize=12)
        if not is_persistence:
            ax1.set_title(f'PM2.5 Forecast Comparison - {station_name} (dim={dim})', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10, loc='best', ncol=2 if (has_std and not is_persistence) else 1)
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[1, 0])
        
        if has_true:
            colors_map = {'rdegpr': '#DD4444', 'gru': '#4477DD', 'lstm': '#44BB77',
                         'neuralcde': '#AA44BB', 'gruodebayes': '#44BBBB'}
            marker_map = {'rdegpr': 'o', 'gru': '^', 'lstm': 's',
                         'neuralcde': 'D', 'gruodebayes': 'v'}
            
            for method in ['rdegpr', 'gru', 'lstm', 'neuralcde', 'gruodebayes']:
                if method in data:
                    mask = ~np.isnan(data[method][:, dim]) & ~np.isnan(data['true'][:, dim])
                    if mask.sum() > 0:
                        ax2.scatter(data['true'][mask, dim], data[method][mask, dim],
                                   alpha=0.7, color=colors_map[method], s=50,
                                   label=method.upper(), marker=marker_map[method],
                                   edgecolors='white', linewidths=0.5)
            
            all_valid = ~np.isnan(data['rdegpr'][:, dim]) & ~np.isnan(data['true'][:, dim])
            min_val = min(data['true'][all_valid, dim].min(), data['rdegpr'][all_valid, dim].min())
            max_val = max(data['true'][all_valid, dim].max(), data['rdegpr'][all_valid, dim].max())
            margin = (max_val - min_val) * 0.05
            ax2.plot([min_val-margin, max_val+margin], [min_val-margin, max_val+margin],
                    'k--', lw=1.5, alpha=0.6, label='y=x Reference')
        
        ax2.set_xlabel('Ground Truth (μg/m³)', fontsize=12)
        ax2.set_ylabel('Predicted Value (μg/m³)', fontsize=12)
        ax2.set_title('Predicted vs Ground Truth', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=9, loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        ax3 = fig.add_subplot(gs[1, 1])
        
        if has_true:
            error = data['rdegpr'][:, dim] - data['true'][:, dim]
            colors_err = ['crimson' if e >= 0 else 'steelblue' for e in error]
            ax3.bar(t, error, color=colors_err, alpha=0.7, edgecolor='white', linewidth=0.5)
            ax3.axhline(0, color='k', lw=1.2)
            
            rmse = np.sqrt(np.nanmean(error**2))
            mae = np.nanmean(np.abs(error))
            ax3.set_title(f'Prediction Error | RMSE={rmse:.2f}, MAE={mae:.2f}', fontsize=14, fontweight='bold')
        else:
            ax3.set_title('Prediction Error (no ground truth)', fontsize=14, fontweight='bold')
        
        ax3.set_xlabel('Time Step (Hours)', fontsize=12)
        ax3.set_ylabel('Error (μg/m³)', fontsize=12)
        ax3.grid(True, alpha=0.3, axis='y')
        
        ax4 = fig.add_subplot(gs[2, :])
        
        methods_list = ['RDE-GPR']
        rmse_values = []
        mae_values = []
        colors = ['#DD4444']
        
        if has_true:
            rmse_r = np.sqrt(np.nanmean((data['rdegpr'][:, dim] - data['true'][:, dim])**2))
            mae_r = np.nanmean(np.abs(data['rdegpr'][:, dim] - data['true'][:, dim]))
            rmse_values.append(rmse_r)
            mae_values.append(mae_r)
            
            for method, c, name in [('gru', '#4477DD', 'GRU'), ('lstm', '#44BB77', 'LSTM'),
                                    ('neuralcde', '#AA44BB', 'NeuralCDE'),
                                    ('gruodebayes', '#44BBBB', 'GRU-ODE-Bayes')]:
                if method in data:
                    methods_list.append(name)
                    colors.append(c)
                    rmse_m = np.sqrt(np.nanmean((data[method][:, dim] - data['true'][:, dim])**2))
                    mae_m = np.nanmean(np.abs(data[method][:, dim] - data['true'][:, dim]))
                    rmse_values.append(rmse_m)
                    mae_values.append(mae_m)
        
        x_pos = np.arange(len(methods_list))
        width = 0.35
        
        bars1 = ax4.bar(x_pos - width/2, rmse_values, width, label='RMSE',
                       color=colors, alpha=0.75, edgecolor='white', linewidth=1.5)
        bars2 = ax4.bar(x_pos + width/2, mae_values, width, label='MAE',
                       color=[c for c in colors], alpha=0.45, edgecolor='white',
                       linewidth=1.5, hatch='///')
        
        for bar, v in zip(bars1, rmse_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(rmse_values)*0.03,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        for bar, v in zip(bars2, mae_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + max(mae_values)*0.03,
                    f'{v:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(methods_list, fontsize=11)
        ax4.set_ylabel('Error (μg/m³)', fontsize=12)
        ax4.set_title('RMSE & MAE Comparison Across Methods', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10, loc='upper right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'PM2.5 Forecast: RDE-GPR vs Baselines ({station_name})',
                     fontsize=16, fontweight='bold', y=1.01)
        
        output_path = os.path.join(self.out_dir, f'full_comparison_{station_name}_dim{dim}_{timestamp}.png')
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'✓ 生成完整对比图: {output_path}')
        
        return {
            'rmse': dict(zip(methods_list, rmse_values)),
            'mae': dict(zip(methods_list, mae_values))
        }

    def generate_metrics_summary(self, all_metrics, station_name='pm25'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        rows = []
        for i, dim_metrics in enumerate(all_metrics):
            row = {'dim': i}
            for metric_name, values in dim_metrics.items():
                for method, value in values.items():
                    row[f'{method}_{metric_name}'] = value
            rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            csv_path = os.path.join(self.out_dir, f'metrics_summary_{station_name}_{timestamp}.csv')
            df.to_csv(csv_path, index=False)
            print(f'✓ 生成指标汇总: {csv_path}')
        
        return rows

    def _detect_gpr_dims(self, data):
        """检测哪些维度有真正的RDE-GPR预测（std不全为0）"""
        gpr_dims = []
        persistence_dims = []
        
        if 'rdegpr_std' in data and data['rdegpr_std'] is not None:
            std_data = data['rdegpr_std']
            for d in range(std_data.shape[1]):
                col_std = std_data[:, d]
                if np.all(col_std == 0) or np.all(np.isnan(col_std)):
                    persistence_dims.append(d)
                else:
                    gpr_dims.append(d)
        else:
            n_dims = data['rdegpr'].shape[1]
            gpr_dims = list(range(n_dims))
        
        return gpr_dims, persistence_dims

    def run(self, results_dir, ground_path=None, split_ratio=0.5, horizon_steps=24,
            target_dims=None):
        print('=' * 60)
        print('PM2.5预测结果对比可视化 (Lorenz96风格 v2)')
        print('=' * 60)
        print(f'\n加载数据: {results_dir}')
        
        data = self.load_data(results_dir, ground_path=ground_path,
                             split_ratio=split_ratio, horizon_steps=horizon_steps)
        print(f'  ✓ 加载方法数: {len([k for k in data.keys() if k not in ["pred_cols", "split_info", "rdegpr_std"]])}')
        
        if 'rdegpr' not in data:
            print('✗ 未找到 RDE-GPR 结果，请先运行预测实验')
            return
        
        if 'true' in data and 'split_info' in data:
            si = data['split_info']
            print(f'  ✓ 真实值维度: {data["true"].shape}')
            print(f'    分割信息: total={si["total_len"]}, split@{si["split_point"]}, horizon={si["horizon"]}')
        elif 'true' in data:
            print(f'  ✓ 真实值维度: {data["true"].shape}')
        else:
            print('  ⚠ 未找到真实值，部分图表将缺少对比')
        
        gpr_dims, persistence_dims = self._detect_gpr_dims(data)
        
        station_names = []
        if 'pred_cols' in data:
            station_names = data['pred_cols']
        
        print(f'\n  📊 维度分析:')
        print(f'    ✅ RDE-GPR真正预测维度 ({len(gpr_dims)}个): ', end='')
        for d in gpr_dims:
            name = station_names[d] if d < len(station_names) else f'Station{d:03d}'
            print(f'dim{d}({name})', end=' ')
        print()
        
        if persistence_dims:
            print(f'    ⚠️  Persistence/回退维度 ({len(persistence_dims)}个, std=0): ', end='')
            for d in persistence_dims[:10]:
                name = station_names[d] if d < len(station_names) else f'Station{d:03d}'
                print(f'dim{d}({name})', end=' ')
            if len(persistence_dims) > 10:
                print(f'...等共{len(persistence_dims)}个', end='')
            print()
            print(f'    💡 这些维度未做GPR预测(可能是target_indices未包含)，将跳过')
        
        if target_dims is None:
            target_dims = gpr_dims[:min(3, len(gpr_dims))] if gpr_dims else list(range(min(3, data['rdegpr'].shape[1])))
        
        all_metrics = []
        
        for dim in target_dims:
            station_name = station_names[dim] if dim < len(station_names) else f'Station{dim:03d}'
            is_persistence = dim in persistence_dims
            print(f'\n--- 生成维度 {dim} ({station_name}) {"[⚠️ Persistence]" if is_persistence else "[✅ GPR]"} ---')
            metrics = self.generate_full_comparison(data, dim=dim, station_name=station_name,
                                                   is_persistence=is_persistence)
            all_metrics.append(metrics)
        
        if all_metrics:
            self.generate_metrics_summary(all_metrics)
        
        print('\n' + '=' * 60)
        print(f'可视化完成! 结果保存在: {self.out_dir}')
        print(f'共生成 {len(all_metrics)} 张图 (GPR: {sum(1 for d in target_dims if d in gpr_dims)}, Persistence: {sum(1 for d in target_dims if d in persistence_dims)})')
        print('=' * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PM2.5预测结果对比可视化 (Lorenz96风格 v2)')
    parser.add_argument('--results_dir', type=str, default='./save/pm25_comparison',
                        help='预测结果目录')
    parser.add_argument('--out_dir', type=str, default='./save/visualization_results/pm25',
                        help='输出目录')
    parser.add_argument('--ground_path', type=str, default=None,
                        help='真实值文件路径 (支持CSV/TXT格式)')
    parser.add_argument('--split_ratio', type=float, default=0.5,
                        help='训练/测试分割比例 (默认0.5)')
    parser.add_argument('--horizon_steps', type=int, default=24,
                        help='预测步数 (默认24小时)')
    parser.add_argument('--target_dims', type=str, default=None,
                        help='目标维度，逗号分隔 (默认0,1,2)')
    
    args = parser.parse_args()
    
    target_dims = None
    if args.target_dims is not None:
        target_dims = [int(x.strip()) for x in args.target_dims.split(',')]
    
    visualizer = PM25ComparisonVisualizer(out_dir=args.out_dir)
    visualizer.run(
        results_dir=args.results_dir,
        ground_path=args.ground_path,
        split_ratio=args.split_ratio,
        horizon_steps=args.horizon_steps,
        target_dims=target_dims
    )
