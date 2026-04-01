#!/usr/bin/env python3
"""
No-Impute Baseline: GRU-ODE-Bayes 直接用原始缺失数据进行预测
对比方案：不经过补值，直接用原始数据预测

CUDA_VISIBLE_DEVICES=3 python baselines/no_impute_baseline/pm25_gruodebayes_noimpute.py \
  --missing_path ./data/pm25/Code/STMVL/SampleData/pm25_missing.txt \
  --ground_path ./data/pm25/Code/STMVL/SampleData/pm25_ground.txt \
  --split_ratio 0.5 --horizon_days 1 --device cuda
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baselines.pm25_gruodebayes_forecast import (
    time_split_df, basic_array_stats, assert_or_raise, infer_steps_per_day,
    StandardScaler, NNFOwithBayesianJumps, ForecastHead, build_windows, windows_to_gruode_format
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
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--p_hidden", type=int, default=32)
    parser.add_argument("--prep_hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--delta_t", type=float, default=0.1)
    parser.add_argument("--time_scale", type=float, default=0.02)
    parser.add_argument("--solver", type=str, default="euler")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
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

    # ================================================================
    # 2. Horizon
    # ================================================================
    full_horizon = meta["fut_len"]
    horizon = full_horizon
    if args.horizon_days > 0:
        spd = infer_steps_per_day(fut_full.index)
        horizon = int(round(args.horizon_days * spd))
    fut_full = fut_full.iloc[:horizon].copy()

    # ================================================================
    # 3. 标准化
    # ================================================================
    scaler = StandardScaler()
    history_scaled = scaler.fit_transform(history).astype(np.float32)
    fut_true = fut_full.values.astype(np.float64)
    fut_true_scaled = scaler.transform(fut_true).astype(np.float32)

    print(f"\n{'='*70}")
    print(f"PM2.5 预测 —— GRU-ODE-Bayes (No-Impute Baseline)")
    print(f"{'='*70}")
    print(f"输入数据: 原始缺失数据 (ffill + bfill + 0填充)")
    print(f"历史长度: {history_scaled.shape[0]} 步")
    print(f"预测步数: {horizon} 步")
    print(f"特征维度: {D}")

    # ================================================================
    # 4. 构建训练数据
    # ================================================================
    W = args.window_size
    steps_ahead = 1
    X_train, y_train = build_windows(history_scaled, W, steps_ahead)
    print(f"训练样本: {X_train.shape[0]}")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")

    # ================================================================
    # 5. 构建模型
    # ================================================================
    gruode_model = NNFOwithBayesianJumps(
        input_size=D,
        hidden_size=args.hidden_size,
        p_hidden=args.p_hidden,
        prep_hidden=args.prep_hidden,
        logvar=True,
        mixing=0.0001,
        full_gru_ode=False,
        impute=False,
        solver=args.solver,
        cov_size=1,
    ).to(device)

    head = ForecastHead(args.hidden_size, args.p_hidden, D).to(device)

    n_params = (sum(p.numel() for p in gruode_model.parameters() if p.requires_grad)
                + sum(p.numel() for p in head.parameters() if p.requires_grad))
    print(f"模型参数量：{n_params:,}")

    # ================================================================
    # 6. 训练
    # ================================================================
    params = list(gruode_model.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False)

    N = X_train.shape[0]
    dataset = TensorDataset(torch.arange(N))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    X_np = X_train
    y_t = torch.tensor(y_train, dtype=torch.float32)

    best_loss = float("inf")
    best_state = None
    loss_history = []

    pred_step = max(float(steps_ahead), float(args.window_size) * 0.2)
    time_scale = args.time_scale
    delta_t = args.delta_t

    for epoch in tqdm(range(args.epochs), desc="  训练", disable=args.no_verbose):
        gruode_model.train()
        head.train()
        ep_loss = 0.0
        ep_n = 0
        
        for (idx,) in loader:
            optimizer.zero_grad()
            
            idx_list = idx.tolist()
            X_batch = X_np[idx_list]
            y_batch = y_t[idx].to(device)
            x_last = torch.tensor(X_batch[:, -1, :], dtype=torch.float32, device=device)
            B = len(idx_list)
            
            times, time_ptr, X_flat, M_flat, obs_idx, cov, T_total = \
                windows_to_gruode_format(X_batch, device, time_scale)
            
            try:
                h, nll_loss, p_out, loss_1 = gruode_model(
                    times, time_ptr, X_flat, M_flat, obs_idx,
                    delta_t=delta_t, T=T_total, cov=cov, pred_step=pred_step * time_scale
                )
                
                delta = head(h, x_last)
                pred = x_last + delta
                
                mse_loss = nn.functional.mse_loss(pred, y_batch)
                loss = mse_loss + 0.01 * nll_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                optimizer.step()
                
                ep_loss += mse_loss.item() * B
                ep_n += B
            except Exception as ex:
                tqdm.write(f"  [跳过 batch] {ex}")
                continue
        
        if ep_n > 0:
            ep_loss /= ep_n
        loss_history.append(ep_loss)
        scheduler.step(ep_loss)

        if ep_loss < best_loss:
            best_loss = ep_loss
            gruode_state = {k: v.cpu().clone() for k, v in gruode_model.state_dict().items()}
            head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            best_state = {"gruode": gruode_state, "head": head_state}

        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {ep_loss:.6f}")

    if best_state:
        gruode_model.load_state_dict({k: v.to(device) for k, v in best_state["gruode"].items()})
        head.load_state_dict({k: v.to(device) for k, v in best_state["head"].items()})

    print(f"\n训练完成，最优 MSE: {best_loss:.6f}")

    # ================================================================
    # 7. 预测 (批量推理)
    # ================================================================
    print("\n[预测阶段]（批量推理）")
    
    T_hist = history_scaled.shape[0]
    W = args.window_size
    
    full = np.concatenate([history_scaled, fut_true_scaled], axis=0).astype(np.float32)
    
    windows = np.stack([full[T_hist - W + i : T_hist + i] for i in range(horizon)])
    
    gruode_model.eval()
    head.eval()
    preds_list = []
    n_batches = (horizon + args.batch_size - 1) // args.batch_size
    
    for b in tqdm(range(n_batches), desc="  批量推理", disable=args.no_verbose):
        sl = slice(b * args.batch_size, (b+1) * args.batch_size)
        X_batch = windows[sl]
        x_last = torch.tensor(X_batch[:, -1, :], dtype=torch.float32, device=device)
        
        times, time_ptr, X_flat, M_flat, obs_idx, cov, T_total = \
            windows_to_gruode_format(X_batch, device, time_scale)
        
        with torch.no_grad():
            h, _, p, _ = gruode_model(
                times, time_ptr, X_flat, M_flat, obs_idx,
                delta_t=delta_t, T=T_total, cov=cov, pred_step=pred_step * time_scale
            )
            pred = head(h, x_last).cpu().numpy()
        
        preds_list.append(pred)
    
    preds_scaled = np.concatenate(preds_list, axis=0)
    
    last_vals = windows[:, -1, :]
    nan_count = 0
    for i in range(preds_scaled.shape[0]):
        nan_cols = np.isnan(preds_scaled[i])
        if nan_cols.any():
            nan_count += 1
            preds_scaled[i, nan_cols] = last_vals[i, nan_cols]
    
    if nan_count > 0:
        print(f"  [警告] 批量推理中出现 {nan_count} 次 NaN")
    
    preds = scaler.inverse_transform(preds_scaled.astype(np.float64))
    preds_scaled = preds

    # ================================================================
    # 8. 评估
    # ================================================================
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # 处理 NaN
    nan_count_pred = np.isnan(preds_scaled).sum()
    nan_count_true = np.isnan(fut_true).sum()
    
    if nan_count_pred > 0:
        print(f"  [警告] 预测结果包含 {nan_count_pred} 个 NaN")
        preds_scaled = np.nan_to_num(preds_scaled, nan=0.0)
    
    if nan_count_true > 0:
        print(f"  [警告] 真实值包含 {nan_count_true} 个 NaN（在预测区间内）")
        mask = ~np.isnan(fut_true)
        if mask.sum() > 0:
            rmse = np.sqrt(np.mean((fut_true[mask] - preds_scaled[mask])**2))
            mae = np.mean(np.abs(fut_true[mask] - preds_scaled[mask]))
        else:
            rmse = np.nan
            mae = np.nan
        print(f"\n整体评估（GRU-ODE-Bayes No-Impute）:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  (基于 {mask.sum()}/{fut_true.size} 个有效数据点)")
    else:
        rmse = np.sqrt(mean_squared_error(fut_true, preds_scaled))
        mae = mean_absolute_error(fut_true, preds_scaled)
        print(f"\n整体评估（GRU-ODE-Bayes No-Impute）:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")

    # 保存结果
    save_dir = f"./save/pm25_gruodebayes_noimpute_{args.seed}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    pred_df = pd.DataFrame(preds_scaled, index=fut_full.index)
    pred_df.to_csv(f"{save_dir}/future_pred.csv")
    
    # 保存 metrics.json
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
    plt.plot(fut_full.index, preds_scaled[:, d], label=f"GRU-ODE-Bayes No-Impute (dim {d})", color="tomato")
    plt.xlabel("Time"); plt.ylabel("PM2.5")
    plt.title(f"GRU-ODE-Bayes No-Impute Forecast vs True (dim {d})")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{save_dir}/plot_forecast_dim{d}.png", dpi=150)
    plt.close()
    
    # 每个维度的 RMSE
    rmse_list = []
    for j in range(fut_true.shape[1]):
        mask_j = ~np.isnan(fut_true[:, j]) & ~np.isnan(preds_scaled[:, j])
        if mask_j.sum() > 0:
            r = np.sqrt(np.mean((fut_true[mask_j, j] - preds_scaled[mask_j, j])**2))
        else:
            r = np.nan
        rmse_list.append(r)
    
    plt.figure(figsize=(14, 5))
    plt.bar(np.arange(len(rmse_list)), rmse_list)
    plt.xlabel("Dimension"); plt.ylabel("RMSE")
    plt.title("GRU-ODE-Bayes No-Impute RMSE per Dimension")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/plot_rmse_per_dim.png", dpi=150)
    plt.close()

    print(f"输出目录: {save_dir}")
    return rmse, mae


if __name__ == "__main__":
    main()
