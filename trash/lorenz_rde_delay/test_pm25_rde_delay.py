# test_pm25_rde_delay.py - 使用已补全的PM2.5数据进行RDE-Delay预测
import os
import sys
import numpy as np
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(base_dir, 'models'))

from rde_module import RandomlyDelayEmbedding


def load_pm25_imputed_data(csv_path):
    """加载已补全的PM2.5数据，跳过第一列（日期时间）"""
    df = pd.read_csv(csv_path)
    # 跳过第一列（日期时间），只保留数值数据
    data = df.iloc[:, 1:].values.astype(float)
    return data


def run_rde_delay_prediction(seq, target_idx=0, trainlength=30, max_delay=50, M=4, num_samples=100, steps_ahead=1):
    """使用RDE-Delay方法进行预测"""
    print(f"\n使用 RDE-Delay 进行预测...")
    print(f"参数: max_delay={max_delay}, M={M}, num_samples={num_samples}, trainlength={trainlength}")

    rde = RandomlyDelayEmbedding(max_delay=max_delay, M=M, num_samples=num_samples)

    predictions, stds, all_preds = rde.ensemble_predict(
        seq=seq,
        target_idx=target_idx,
        trainlength=trainlength,
        steps_ahead=steps_ahead,
        return_uncertainty=True
    )

    print(f"RDE-Delay 预测完成，形状: {predictions.shape}")
    return predictions, stds, all_preds


def main():
    # PM2.5已补全数据路径
    pm25_imputed_path = "/home/rhl/Github/pm25_history_imputed_split0.5_seed42_20260128_101132/history_imputed.csv"

    # RDE-Delay参数
    rde_trainlength = 30
    rde_max_delay = 50
    rde_M = 4
    rde_num_samples = 100
    rde_steps_ahead = 1
    rde_target_idx = 0  # 预测第一个站点

    print("=" * 60)
    print("加载 PM2.5 已补全数据...")
    if os.path.exists(pm25_imputed_path):
        pm25_data = load_pm25_imputed_data(pm25_imputed_path)
        print(f"PM2.5 数据形状: {pm25_data.shape}")
        print(f"数据范围: [{np.nanmin(pm25_data):.2f}, {np.nanmax(pm25_data):.2f}]")
        print(f"包含NaN数量: {np.sum(np.isnan(pm25_data))}")
    else:
        print(f"文件不存在: {pm25_imputed_path}")
        return

    print("\n" + "=" * 60)
    print("使用 RDE-Delay 对PM2.5数据进行预测")
    pred_rde, std_rde, all_preds = run_rde_delay_prediction(
        seq=pm25_data,
        target_idx=rde_target_idx,
        trainlength=rde_trainlength,
        max_delay=rde_max_delay,
        M=rde_M,
        num_samples=rde_num_samples,
        steps_ahead=rde_steps_ahead
    )

    # 结果分析
    print("\n" + "=" * 60)
    print("结果分析")
    print("=" * 60)

    pred_start_idx = rde_trainlength
    ground_truth = pm25_data[pred_start_idx:, rde_target_idx]

    error = pred_rde - ground_truth
    max_error = np.max(np.abs(error))
    rms_error = np.sqrt(np.mean(error**2))
    mean_uncertainty = np.mean(std_rde)

    print(f"\nRDE-Delay 预测结果:")
    print(f"  最大误差: {max_error:.6f}")
    print(f"  RMS误差: {rms_error:.6f}")
    print(f"  平均不确定性: {mean_uncertainty:.6f}")
    print(f"  不确定性覆盖率: {np.sum(np.abs(error) <= 2*std_rde) / len(error) * 100:.1f}%")

    print("\n" + "=" * 60)
    print("完成！")


if __name__ == '__main__':
    main()
