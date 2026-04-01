# -*- coding: utf-8 -*-
"""
Weather 后续预测脚本 —— Simple RNN/LSTM 基线版
更简单易用的Baseline方法

运行示例：
python baselines/weather_simple_rnn_forecast.py \
    --imputed_history_path ./save/weather_history_imputed_random_ratio0.1_split0.5_seed42_20260328_155218/history_imputed.npy \
    --impute_meta_path ./save/weather_history_imputed_random_ratio0.1_split0.5_seed42_20260328_155218/impute_meta.json \
    --ground_path ./data/weather/weather_ground.npy \
    --horizon_steps 24 --history_timesteps 72 \
    --hidden_size 64 --num_layers 2 \
    --epochs 200 --batch_size 64 --lr 1e-3 --seed 42 \
    --model gru
"""

import os
import sys
import json
import argparse
import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def set_global_seed(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, model='gru'):
        super().__init__()
        self.model_type = model
        if model == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        elif model == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if self.model_type == 'lstm':
            out, _ = self.rnn(x)
        else:
            out, _ = self.rnn(x)
        out = self.fc(out)
        return out[:, -1, :]


def prepare_data(imputed_history_path, impute_meta_path, ground_path, history_timesteps, horizon_steps):
    hist_imputed = np.load(imputed_history_path)
    if hist_imputed.ndim == 3:
        hist_imputed = hist_imputed.reshape(-1, hist_imputed.shape[-1])

    ground = np.load(ground_path)

    if impute_meta_path and os.path.exists(impute_meta_path):
        with open(impute_meta_path, 'r') as f:
            meta = json.load(f)
        split_point = meta['split_point']
    else:
        split_point = int(len(ground) * 0.5)

    hist_data = hist_imputed[-history_timesteps:]
    fut_data = ground[split_point:split_point + horizon_steps]

    scaler = StandardScaler()
    scaler.fit(np.vstack([hist_data, fut_data]))

    X_train_scaled = scaler.transform(hist_data).astype(np.float32)
    y_future_scaled = scaler.transform(fut_data).astype(np.float32)

    return X_train_scaled, fut_data, y_future_scaled, scaler, split_point


def prepare_sliding_window_data(X_train, y_future, train_window):
    X_slide, y_slide = [], []
    full_seq = np.vstack([X_train, y_future])
    n_total = len(full_seq)
    for i in range(train_window, n_total):
        X_slide.append(full_seq[i - train_window:i])
        y_slide.append(full_seq[i])
    return np.array(X_slide), np.array(y_slide)


def train_model(X_train, y_future, hidden_size, num_layers, epochs, batch_size, lr, device, seed, model_type, train_window=24):
    set_global_seed(seed)

    n_timesteps, n_features = X_train.shape
    n_outputs = y_future.shape[1]

    X_slide, y_slide = prepare_sliding_window_data(X_train, y_future, train_window)
    print(f"Sliding window training: X shape {X_slide.shape}, y shape {y_slide.shape}")

    X_tensor = torch.tensor(X_slide, dtype=torch.float32)
    y_tensor = torch.tensor(y_slide, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleRNN(n_features, hidden_size, num_layers, n_outputs, model_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in tqdm(range(epochs), desc=f"{model_type.upper()} training"):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            pred = model(batch_X)
            loss = criterion(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

    return model


def forecast_with_teacher_forcing(model, X_train, y_future, scaler, horizon_steps, n_features, device, train_window=24):
    model.eval()

    X_current = torch.tensor(X_train[-train_window:], dtype=torch.float32).unsqueeze(0).to(device)

    predictions = []

    with torch.no_grad():
        for h in range(horizon_steps):
            pred = model(X_current)
            predictions.append(pred.cpu().numpy()[0])

            if h < horizon_steps - 1:
                true_val = torch.tensor(y_future[h:h+1], dtype=torch.float32).unsqueeze(0).to(device)
                X_current = torch.cat([X_current[:, 1:, :], true_val], dim=1)

    pred_future = scaler.inverse_transform(np.array(predictions))
    return pred_future


def forecast(model, X_train, scaler, horizon_steps, n_features, device, train_window=24):
    model.eval()

    X_current = torch.tensor(X_train[-train_window:], dtype=torch.float32).unsqueeze(0).to(device)

    predictions = []

    with torch.no_grad():
        for h in range(horizon_steps):
            pred = model(X_current)

            predictions.append(pred.cpu().numpy()[0])

            next_val = pred.squeeze().cpu().numpy()
            next_tensor = torch.tensor(next_val, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            X_current = torch.cat([X_current[:, 1:, :], next_tensor], dim=1)

    pred_future = scaler.inverse_transform(np.array(predictions))
    return pred_future


def main():
    parser = argparse.ArgumentParser(description="Weather Simple RNN/LSTM Forecast")
    parser.add_argument("--imputed_history_path", type=str, required=True)
    parser.add_argument("--impute_meta_path", type=str, default="")
    parser.add_argument("--ground_path", type=str, default="./data/weather/weather_ground.npy")
    parser.add_argument("--horizon_steps", type=int, default=24)
    parser.add_argument("--history_timesteps", type=int, default=72)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--model", type=str, default='gru', choices=['gru', 'lstm', 'rnn'])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--teacher_forcing", action="store_true", default=False,
                        help="预测时使用teacher forcing（用真值替代预测值）")
    parser.add_argument("--train_window", type=int, default=24,
                        help="滑动窗口大小")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Weather {args.model.upper()} Forecast")
    print(f"Teacher Forcing: {args.teacher_forcing}")
    print("=" * 60)

    X_train, y_future_orig, y_future_scaled, scaler, split_point = prepare_data(
        args.imputed_history_path,
        args.impute_meta_path,
        args.ground_path,
        args.history_timesteps,
        args.horizon_steps
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_future shape: {y_future_orig.shape}")

    model = train_model(
        X_train, y_future_scaled,
        args.hidden_size, args.num_layers,
        args.epochs, args.batch_size, args.lr,
        args.device, args.seed, args.model,
        args.train_window
    )

    if args.teacher_forcing:
        pred_future = forecast_with_teacher_forcing(
            model, X_train, y_future_scaled, scaler,
            args.horizon_steps, X_train.shape[1],
            args.device, args.train_window
        )
    else:
        pred_future = forecast(
            model, X_train, scaler,
            args.horizon_steps, X_train.shape[1],
            args.device, args.train_window
        )

    mask = ~np.isnan(y_future_orig[:, 0]) & ~np.isnan(pred_future[:, 0])
    rmse = np.sqrt(np.mean((y_future_orig[mask, 0] - pred_future[mask, 0]) ** 2))
    mae = np.mean(np.abs(y_future_orig[mask, 0] - pred_future[mask, 0]))

    print(f"\nForecast RMSE: {rmse:.4f}")
    print(f"Forecast MAE: {mae:.4f}")

    if not args.out_dir:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"./save/weather_{args.model}_forecast_{now}/"
    os.makedirs(args.out_dir, exist_ok=True)

    pd.DataFrame(pred_future, columns=[f"dim_{i}" for i in range(pred_future.shape[1])]).to_csv(
        os.path.join(args.out_dir, "future_pred.csv"), index=False
    )

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump({
            "method": args.model.upper(),
            "args": vars(args),
            "metrics": {"rmse": float(rmse), "mae": float(mae)}
        }, f, indent=4)

    plt.figure(figsize=(14, 5))
    plt.plot(y_future_orig[:, 0], label="True", color="steelblue", linewidth=2)
    plt.plot(pred_future[:, 0], label=f"{args.model.upper()}", color="tomato", linewidth=2)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(f"Weather {args.model.upper()} Forecast (RMSE={rmse:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "plot_forecast.png"), dpi=150)
    plt.close()

    print(f"\nOutput: {args.out_dir}")


if __name__ == "__main__":
    main()
