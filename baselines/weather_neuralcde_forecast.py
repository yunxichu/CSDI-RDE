# -*- coding: utf-8 -*-
"""
Weather 后续预测脚本 —— NeuralCDE 基线版

参考：https://github.com/patrick-kidger/torchcde

数据接口与 RDE-GPR 脚本完全对齐：
- 输入：history_imputed.npy + weather_ground.npy
- 输出：future_pred.csv / metrics.json / plot_*.png

安装：
  pip install torch torchdiffeq torchcde

示例：
python baselines/weather_neuralcde_forecast.py \
    --imputed_history_path ./save/weather_history_imputed_random_ratio0.1_split0.5_seed42_20260328_155218/history_imputed.npy \
    --impute_meta_path ./save/weather_history_imputed_random_ratio0.1_split0.5_seed42_20260328_155218/impute_meta.json \
    --ground_path ./data/weather/weather_ground.npy \
    --horizon_steps 24 --history_timesteps 72 \
    --hidden_channels 64 --num_layers 3 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42

"""

import os
import sys
import json
import time
import random
import argparse
import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def set_global_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def safe_json_dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False, default=str)


def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan}
    diff = y_true[mask] - y_pred[mask]
    return {"rmse": float(np.sqrt(np.mean(diff**2))), "mae": float(np.mean(np.abs(diff)))}


class CDEFunc(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.input_channels = input_channels

        layers = []
        in_dim = hidden_channels
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_channels), nn.Softplus()]
            in_dim = hidden_channels
        layers += [nn.Linear(hidden_channels, hidden_channels * input_channels), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, t, z):
        return self.net(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)


class NeuralCDE(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers=3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.initial = nn.Linear(input_channels, hidden_channels)
        self.func = CDEFunc(input_channels, hidden_channels, num_layers)
        self.readout = nn.Linear(hidden_channels, 1)

    def forward(self, coeffs, t_span):
        from torchcde import CubicSpline, cdeint
        X = CubicSpline(coeffs, t_span)
        z0 = self.initial(X.evaluate(t_span[0]))
        zT = cdeint(X, z0, t_span, func=self.func, method='rk4', options={'step_size': 0.25})
        return self.readout(zT).squeeze(-1)


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

    y_future = ground[split_point:split_point + horizon_steps]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    hist_data = hist_imputed[-history_timesteps:]
    scaler_X.fit(hist_data)
    scaler_y.fit(y_future[:, :1])

    X_train = scaler_X.transform(hist_data)
    y_train = scaler_y.transform(y_future[:, :1])

    return X_train, y_train, scaler_X, scaler_y, y_future


def train_neuralcde(X_train, y_train, hidden_channels, num_layers, epochs, batch_size, lr, device, seed):
    set_global_seed(seed)

    n_timesteps, n_features = X_train.shape
    n_outputs = y_train.shape[1]

    times = torch.linspace(0, 1, n_timesteps)

    from torchcde import natural_cubic_coeffs
    coeffs_np = natural_cubic_coeffs(
        torch.tensor(X_train, dtype=torch.float32),
        times
    ).numpy()

    model = NeuralCDE(n_features, hidden_channels, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in tqdm(range(epochs), desc="NeuralCDE training"):
        perm = torch.randperm(n_timesteps - 1)
        epoch_loss = 0
        n_batches = 0

        for i in range(0, n_timesteps - 1, batch_size):
            batch_idx = perm[i:i + batch_size]
            if len(batch_idx) < 2:
                continue

            batch_coeffs = torch.tensor(coeffs_np[batch_idx], dtype=torch.float32).to(device)
            t_start = times[batch_idx[0]].item()
            t_end = times[batch_idx[-1]].item()
            t_span = torch.tensor([t_start, t_end], dtype=torch.float32).to(device)

            pred = model(batch_coeffs.unsqueeze(0), t_span)
            target = torch.tensor(y_train[batch_idx + 1], dtype=torch.float32).to(device)

            loss = criterion(pred.squeeze(), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / max(1, n_batches):.4f}")

    return model, times, coeffs_np


def forecast(model, X_train, y_future, scaler_X, scaler_y, horizon_steps, hidden_channels, num_layers, device, coeffs_np):
    model.eval()

    n_timesteps, n_features = X_train.shape
    times = torch.linspace(0, 1, n_timesteps)

    from torchcde import natural_cubic_coeffs
    coeffs = natural_cubic_coeffs(
        torch.tensor(X_train, dtype=torch.float32),
        times
    ).to(device)

    t_span = torch.tensor([0.0, 1.0], dtype=torch.float32).to(device)

    pred_future_normalized = []

    with torch.no_grad():
        for h in range(horizon_steps):
            pred = model(coeffs.unsqueeze(0), t_span)
            pred_future_normalized.append(pred.item())

            if h < horizon_steps - 1:
                pred_val = pred.cpu().numpy()[0, 0]
                new_row = X_train[-1:].copy()
                new_row[0, 0] = pred_val
                X_train = np.vstack([X_train[1:], new_row])

                coeffs = natural_cubic_coeffs(
                    torch.tensor(X_train, dtype=torch.float32),
                    times
                ).to(device)

    pred_future_normalized = np.array(pred_future_normalized).reshape(-1, 1)
    pred_future = scaler_y.inverse_transform(pred_future_normalized)

    return pred_future


def main():
    parser = argparse.ArgumentParser(description="Weather NeuralCDE Forecast")
    parser.add_argument("--imputed_history_path", type=str, required=True)
    parser.add_argument("--impute_meta_path", type=str, default="")
    parser.add_argument("--ground_path", type=str, default="./data/weather/weather_ground.npy")
    parser.add_argument("--horizon_steps", type=int, default=24)
    parser.add_argument("--history_timesteps", type=int, default=72)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    print("=" * 60)
    print("Weather NeuralCDE Forecast")
    print("=" * 60)

    X_train, y_train, scaler_X, scaler_y, y_future = prepare_data(
        args.imputed_history_path,
        args.impute_meta_path,
        args.ground_path,
        args.history_timesteps,
        args.horizon_steps
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_future shape: {y_future.shape}")

    model, times, coeffs_np = train_neuralcde(
        X_train, y_train,
        args.hidden_channels, args.num_layers,
        args.epochs, args.batch_size, args.lr,
        args.device, args.seed
    )

    pred_future = forecast(
        model, X_train.copy(), y_future,
        scaler_X, scaler_y,
        args.horizon_steps,
        args.hidden_channels, args.num_layers,
        args.device, coeffs_np
    )

    metrics = compute_metrics(y_future[:, 0], pred_future[:, 0])
    print(f"\nForecast RMSE: {metrics['rmse']:.4f}")
    print(f"Forecast MAE: {metrics['mae']:.4f}")

    if not args.out_dir:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"./save/weather_neuralcde_forecast_{now}/"
    ensure_dir(args.out_dir)

    pd.DataFrame(pred_future, columns=[f"dim_{i}" for i in range(pred_future.shape[1])]).to_csv(
        os.path.join(args.out_dir, "future_pred.csv"), index=False
    )

    safe_json_dump({
        "method": "NeuralCDE",
        "args": vars(args),
        "metrics": metrics
    }, os.path.join(args.out_dir, "metrics.json"))

    plt.figure(figsize=(14, 5))
    plt.plot(y_future[:, 0], label="True", color="steelblue")
    plt.plot(pred_future[:, 0], label="NeuralCDE", color="tomato")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(f"Weather NeuralCDE Forecast (RMSE={metrics['rmse']:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "plot_forecast.png"), dpi=150)
    plt.close()

    print(f"\nOutput: {args.out_dir}")


if __name__ == "__main__":
    main()
