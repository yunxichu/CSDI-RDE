# -*- coding: utf-8 -*-
"""
Weather 后续预测脚本 —— GRU-ODE-Bayes 基线版

基于：https://github.com/edebrouwer/gru_ode_bayes

运行示例：
python baselines/weather_gruodebayes_forecast.py \
    --imputed_history_path ./save/weather_history_imputed_random_ratio0.1_split0.5_seed42_20260328_155218/history_imputed.npy \
    --impute_meta_path ./save/weather_history_imputed_random_ratio0.1_split0.5_seed42_20260328_155218/impute_meta.json \
    --ground_path ./data/weather/weather_ground.npy \
    --horizon_steps 24 --history_timesteps 72 \
    --hidden_size 64 --p_hidden 32 \
    --delta_t 0.1 --time_scale 0.02 --solver euler \
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
import math

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


class GRUODECell(nn.Module):
    def __init__(self, input_size, hidden_size, p_hidden):
        super().__init__()
        self.hidden_size = hidden_size

        self.lin_xn = nn.Linear(input_size, hidden_size)
        self.lin_hn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_xp = nn.Linear(input_size, p_hidden)
        self.lin_hp = nn.Linear(hidden_size, p_hidden, bias=False)
        self.lin_p = nn.Linear(p_hidden, hidden_size)

        self.act_xn = nn.Tanh()
        self.act_hn = nn.Tanh()

    def forward(self, x, h, p):
        xn = self.act_xn(self.lin_xn(x))
        zn = torch.sigmoid(self.lin_hn(h) + 1.0)
        h_new = zn * h + (1 - zn) * xn

        xp = self.lin_xp(x)
        hp = self.lin_hp(h)
        p_new = xp + hp
        p_new = torch.relu(p_new) + 1e-6

        return h_new, p_new


class GRUODEBayes(nn.Module):
    def __init__(self, input_size, hidden_size, p_hidden, delta_t=0.1, time_scale=0.02, solver='euler'):
        super().__init__()
        self.hidden_size = hidden_size
        self.delta_t = delta_t
        self.time_scale = time_scale
        self.solver = solver

        self.gru_cell = GRUODECell(input_size, hidden_size, p_hidden)
        self.p_init = nn.Parameter(torch.ones(p_hidden) * 0.5)

    def forward(self, times, X, h0=None):
        """
        times: (batch, n_timesteps)
        X: (batch, n_timesteps, input_size)
        """
        batch_size, n_steps, input_size = X.shape

        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=X.device)
        else:
            h = h0

        p = self.p_init.unsqueeze(0).expand(batch_size, -1)

        h_history = [h]
        for t in range(n_steps - 1):
            dt = times[:, t + 1] - times[:, t]
            dt = dt.clamp(min=1e-6)

            n_steps_ode = max(1, int((dt.item() * self.time_scale / self.delta_t) + 0.5))

            for _ in range(n_steps_ode):
                x_t = X[:, t, :]
                h, p = self.gru_cell(x_t, h, p)

            h_history.append(h)

        return torch.stack(h_history, dim=1)


class GRUODEBayesForecast(nn.Module):
    def __init__(self, input_size, hidden_size, p_hidden, output_size=1):
        super().__init__()
        self.gru_ode = GRUODEBayes(input_size, hidden_size, p_hidden)
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, times, X, h0=None):
        h = self.gru_ode(times, X, h0)
        out = self.readout(h)
        return out[:, -1:, :]


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

    X_train = scaler.transform(hist_data).astype(np.float32)
    y_train = scaler.transform(fut_data).astype(np.float32)

    return X_train, y_train, scaler, fut_data, split_point


def train_gruodebayes(X_train, y_train, hidden_size, p_hidden, delta_t, time_scale, solver,
                      epochs, batch_size, lr, device, seed):
    set_global_seed(seed)

    n_timesteps, n_features = X_train.shape
    n_outputs = y_train.shape[1]

    model = GRUODEBayesForecast(n_features, hidden_size, p_hidden, n_outputs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(0)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    times = torch.linspace(0, 1, n_timesteps).unsqueeze(0)

    model.train()
    for epoch in tqdm(range(epochs), desc="GRU-ODE-Bayes training"):
        optimizer.zero_grad()

        pred = model(times.to(device), X_tensor.to(device))
        target = y_tensor.to(device)

        loss = criterion(pred.squeeze(), target)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model


def forecast(model, X_train, scaler, horizon_steps, n_features, device):
    model.eval()

    X_current = torch.tensor(X_train[-1:], dtype=torch.float32).unsqueeze(0).to(device)
    times = torch.linspace(0, 1, 72).unsqueeze(0).to(device)

    predictions = []

    model.eval()
    with torch.no_grad():
        for h in range(horizon_steps):
            pred = model(times, X_current)
            pred_vals = pred[0, -1, :].cpu().numpy()
            predictions.append(pred_vals)

            new_row = torch.zeros(1, 1, n_features).to(device)
            new_row[0, 0, :] = torch.tensor(pred_vals)
            X_current = torch.cat([X_current[:, 1:, :], new_row], dim=1)

    pred_future = np.array(predictions) * scaler.scale_ + scaler.mean_
    return pred_future


def main():
    parser = argparse.ArgumentParser(description="Weather GRU-ODE-Bayes Forecast")
    parser.add_argument("--imputed_history_path", type=str, required=True)
    parser.add_argument("--impute_meta_path", type=str, default="")
    parser.add_argument("--ground_path", type=str, default="./data/weather/weather_ground.npy")
    parser.add_argument("--horizon_steps", type=int, default=24)
    parser.add_argument("--history_timesteps", type=int, default=72)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--p_hidden", type=int, default=32)
    parser.add_argument("--delta_t", type=float, default=0.1)
    parser.add_argument("--time_scale", type=float, default=0.02)
    parser.add_argument("--solver", type=str, default='euler', choices=['euler', 'rk4'])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    print("=" * 60)
    print("Weather GRU-ODE-Bayes Forecast")
    print("=" * 60)

    X_train, y_train, scaler, y_future, split_point = prepare_data(
        args.imputed_history_path,
        args.impute_meta_path,
        args.ground_path,
        args.history_timesteps,
        args.horizon_steps
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_future shape: {y_future.shape}")

    model = train_gruodebayes(
        X_train, y_train,
        args.hidden_size, args.p_hidden,
        args.delta_t, args.time_scale, args.solver,
        args.epochs, args.batch_size, args.lr,
        args.device, args.seed
    )

    pred_future = forecast(
        model, X_train, scaler,
        args.horizon_steps,
        X_train.shape[1],
        args.device
    )

    metrics = compute_metrics(y_future[:, 0], pred_future[:, 0])
    print(f"\nForecast RMSE: {metrics['rmse']:.4f}")
    print(f"Forecast MAE: {metrics['mae']:.4f}")

    if not args.out_dir:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"./save/weather_gruodebayes_forecast_{now}/"
    ensure_dir(args.out_dir)

    pd.DataFrame(pred_future, columns=[f"dim_{i}" for i in range(pred_future.shape[1])]).to_csv(
        os.path.join(args.out_dir, "future_pred.csv"), index=False
    )

    safe_json_dump({
        "method": "GRU-ODE-Bayes",
        "args": vars(args),
        "metrics": metrics
    }, os.path.join(args.out_dir, "metrics.json"))

    plt.figure(figsize=(14, 5))
    plt.plot(y_future[:, 0], label="True", color="steelblue")
    plt.plot(pred_future[:, 0], label="GRU-ODE-Bayes", color="tomato")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title(f"Weather GRU-ODE-Bayes Forecast (RMSE={metrics['rmse']:.4f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "plot_forecast.png"), dpi=150)
    plt.close()

    print(f"\nOutput: {args.out_dir}")


if __name__ == "__main__":
    main()
