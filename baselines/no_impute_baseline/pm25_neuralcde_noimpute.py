#!/usr/bin/env python3
"""
No-Impute Baseline: NeuralCDE 直接用原始缺失数据进行预测
对比方案：不经过补值，直接用原始数据预测

# 检查是否有 nohup.out
cat /home/rhl/Github/nohup.out 2>/dev/null | tail -50

运行
CUDA_VISIBLE_DEVICES=2 python experiments/no_impute_baseline/pm25_neuralcde_noimpute.py \
  --missing_path ./data/pm25/Code/STMVL/SampleData/pm25_missing.txt \
  --ground_path ./data/pm25/Code/STMVL/SampleData/pm25_ground.txt \
  --split_ratio 0.5 --horizon_days 1 \
  --device cuda
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baselines.pm25_neuralcde_forecast import (
    time_split_df, basic_array_stats, assert_or_raise, infer_steps_per_day,
    StandardScaler, NeuralCDE, CDEFunc, train_model, forecast, make_cde_input
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--missing_path", type=str, required=True,
                        help="原始缺失数据路径 (pm25_missing.txt)")
    parser.add_argument("--ground_path", type=str, required=True,
                        help="真实值数据路径 (pm25_ground.txt)")
    parser.add_argument("--split_ratio", type=float, default=0.5)
    parser.add_argument("--horizon_days", type=float, default=1.0)
    parser.add_argument("--window_size", type=int, default=48)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use_fixed_solver", action="store_true")
    parser.add_argument("--no_verbose", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    print(f"[设备] {device}")

    # ================================================================
    # 1. 读数据 - 使用原始缺失数据
    # ================================================================
    df_full = pd.read_csv(args.ground_path, index_col="datetime",
                          parse_dates=True).sort_index()
    hist_full, fut_full, meta = time_split_df(df_full, args.split_ratio)
    
    # 读取原始缺失数据
    df_missing = pd.read_csv(args.missing_path, index_col="datetime",
                            parse_dates=True).sort_index()
    df_hist = df_missing.iloc[:meta["hist_len"]].copy()

    assert_or_raise(df_hist.index.equals(hist_full.index),
        "missing_path datetime 与 ground 前半段不一致，请检查 split_ratio。")
    assert_or_raise(list(df_hist.columns) == list(hist_full.columns),
        "missing_path 列与 ground 列不一致。")

    # 处理缺失值：用前向填充 + 后向填充
    history = df_hist.values.astype(np.float64)
    print(f"原始数据缺失数: {np.isnan(history).sum()}")
    
    # 前向填充
    df_hist_filled = df_hist.ffill()
    # 后向填充（处理开头缺失）
    df_hist_filled = df_hist_filled.bfill()
    history = df_hist_filled.values.astype(np.float64)
    print(f"填充后缺失数: {np.isnan(history).sum()}")
    
    # 如果还有缺失，用0填充
    history = np.nan_to_num(history, nan=0.0)
    
    D = history.shape[1]
    columns = list(df_full.columns)

    report = {"meta": meta, "args": vars(args), "checks": {},
              "history_stats": basic_array_stats(history, "history_missing_filled")}

    # ================================================================
    # 2. Horizon
    # ================================================================
    full_horizon = meta["fut_len"]
    horizon = full_horizon
    if args.horizon_days > 0:
        spd = infer_steps_per_day(fut_full.index)
        horizon = int(round(args.horizon_days * spd))
    fut_full = fut_full.iloc[:horizon].copy()
    report["checks"]["final_horizon_steps"] = int(horizon)

    # ================================================================
    # 3. 标准化
    # ================================================================
    scaler = StandardScaler()
    history_scaled = scaler.fit_transform(history).astype(np.float32)
    fut_true = fut_full.values.astype(np.float64)
    fut_true_scaled = scaler.transform(fut_true).astype(np.float32)

    print(f"\n{'='*70}")
    print(f"PM2.5 预测 —— NeuralCDE (No-Impute Baseline)")
    print(f"{'='*70}")
    print(f"输入数据: 原始缺失数据 (ffill + bfill + 0填充)")
    print(f"历史长度: {history_scaled.shape[0]} 步")
    print(f"预测步数: {horizon} 步")
    print(f"特征维度: {D}")

    # ================================================================
    # 4. 构建训练数据
    # ================================================================
    W = args.window_size
    stride = 1
    N = (len(history_scaled) - W - horizon) // stride + 1
    print(f"训练样本: {N}")

    X_windows = []
    y_windows = []
    for i in range(N):
        X_windows.append(history_scaled[i:i+W])
        y_windows.append(history_scaled[i+W:i+W+horizon, :])

    X_train_np = np.stack(X_windows, axis=0)
    y_train_np = np.stack(y_windows, axis=0)[:, 0, :]
    
    X_train_np = make_cde_input(X_train_np)

    print(f"X_train_np: {X_train_np.shape}, y_train_np: {y_train_np.shape}")

    # ================================================================
    # 5. 训练
    # ================================================================
    model = NeuralCDE(
        input_channels=D+1,
        hidden_channels=args.hidden_channels,
        output_channels=D,
        num_layers=args.num_layers,
        step_size=1.0/args.window_size if args.use_fixed_solver else None
    )

    loss_hist, best_loss = train_model(
        model, X_train_np, y_train_np,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        verbose=not args.no_verbose
    )

    print(f"\n训练完成，最优 MSE: {best_loss:.6f}")

    # ================================================================
    # 6. 预测
    # ================================================================
    print("\n[预测阶段]（批量推理）")
    preds_scaled, _ = forecast(
        model, history_scaled, fut_true_scaled, horizon,
        args.window_size, device, scaler,
        batch_size=args.batch_size,
        verbose=not args.no_verbose
    )

    # ================================================================
    # 7. 评估
    # ================================================================
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # 处理 NaN：预测值和真实值都可能包含 NaN
    nan_count_pred = np.isnan(preds_scaled).sum()
    nan_count_true = np.isnan(fut_true).sum()
    
    if nan_count_pred > 0:
        print(f"  [警告] 预测结果包含 {nan_count_pred} 个 NaN")
        preds_scaled = np.nan_to_num(preds_scaled, nan=0.0)
    
    if nan_count_true > 0:
        print(f"  [警告] 真实值包含 {nan_count_true} 个 NaN（在预测区间内）")
        # 创建 mask，只计算非 NaN 位置
        mask = ~np.isnan(fut_true)
        if mask.sum() > 0:
            rmse = np.sqrt(np.mean((fut_true[mask] - preds_scaled[mask])**2))
            mae = np.mean(np.abs(fut_true[mask] - preds_scaled[mask]))
        else:
            rmse = np.nan
            mae = np.nan
        print(f"\n整体评估（NeuralCDE No-Impute）:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  (基于 {mask.sum()}/{fut_true.size} 个有效数据点)")
    else:
        rmse = np.sqrt(mean_squared_error(fut_true, preds_scaled))
        mae = mean_absolute_error(fut_true, preds_scaled)
        print(f"\n整体评估（NeuralCDE No-Impute）:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")

    # 保存结果
    save_dir = f"./save/pm25_neuralcde_noimpute_{args.seed}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    pred_df = pd.DataFrame(preds_scaled, index=fut_full.index)
    pred_df.to_csv(f"{save_dir}/future_pred.csv")
    
    # 保存 metrics.json（与 best_record 格式一致）
    import json
    metrics = {
        "overall": {
            "rmse": float(rmse),
            "mae": float(mae)
        },
        "horizon": int(horizon)
    }
    with open(f"{save_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # 可视化
    import matplotlib.pyplot as plt
    
    # 绘制第一个维度的预测结果
    d = 0
    plt.figure(figsize=(14, 5))
    plt.plot(fut_full.index, fut_true[:, d], label=f"True (dim {d})", color="steelblue")
    plt.plot(fut_full.index, preds_scaled[:, d], label=f"NeuralCDE No-Impute (dim {d})", color="tomato")
    plt.xlabel("Time"); plt.ylabel("PM2.5")
    plt.title(f"NeuralCDE No-Impute Forecast vs True (dim {d})")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{save_dir}/plot_forecast_dim{d}.png", dpi=150)
    plt.close()
    
    # 每个维度的 RMSE
    rmse_list = []
    for j in range(fut_true.shape[1]):
        mask_j = ~np.isnan(fut_true[:, j]) & ~np.isnan(preds_scaled[:, j])
        if mask_j.sum() > 0:
            rmse = np.sqrt(np.mean((fut_true[mask_j, j] - preds_scaled[mask_j, j])**2))
        else:
            rmse = np.nan
        rmse_list.append(rmse)
    
    plt.figure(figsize=(14, 5))
    plt.bar(np.arange(len(rmse_list)), rmse_list)
    plt.xlabel("Dimension"); plt.ylabel("RMSE")
    plt.title("NeuralCDE No-Impute RMSE per Dimension")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/plot_rmse_per_dim.png", dpi=150)
    plt.close()

    print(f"输出目录: {save_dir}")
    return rmse, mae


if __name__ == "__main__":
    main()
