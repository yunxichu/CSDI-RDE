# -*- coding: utf-8 -*-
"""
EEG 基线预测脚本（GRU/LSTM）

支持直接从EEG补值数据进行单步滚动预测
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def set_global_seed(seed):
    import random
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def prepare_data(imputed_path, ground_path, history_timesteps, target_dims=None, use_ground_truth_train=False):
    data = np.load(imputed_path)
    ground = np.load(ground_path)

    if data.ndim == 3:
        data = data.reshape(-1, data.shape[-1])
    if ground.ndim == 3:
        ground = ground.reshape(-1, ground.shape[-1])

    if target_dims is not None:
        data = data[:, target_dims]
        ground = ground[:, target_dims]

    if use_ground_truth_train:
        hist_data = ground[:history_timesteps]
    else:
        hist_data = data[:history_timesteps]
    fut_data = ground[history_timesteps:history_timesteps + 24]

    scaler = StandardScaler()
    scaler.fit(hist_data)

    X_train_scaled = scaler.transform(hist_data).astype(np.float32)
    y_future_scaled = scaler.transform(fut_data).astype(np.float32)

    return X_train_scaled, fut_data, y_future_scaled, scaler

def prepare_sliding_window_data(data, train_window):
    X_slide, y_slide = [], []
    n_total = len(data)
    for i in range(train_window, n_total):
        X_slide.append(data[i - train_window:i])
        y_slide.append(data[i])
    return np.array(X_slide), np.array(y_slide)

def train_model(X_train, hidden_size, num_layers, epochs, batch_size, lr, device, seed, model_type, train_window=48):
    set_global_seed(seed)
    n_timesteps, n_features = X_train.shape

    X_slide, y_slide = prepare_sliding_window_data(X_train, train_window)

    model = SimpleGRU(n_features, hidden_size, num_layers, n_features) if model_type == 'gru' else SimpleLSTM(n_features, hidden_size, num_layers, n_features)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tensor = torch.tensor(X_slide, dtype=torch.float32)
    y_tensor = torch.tensor(y_slide, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model

def forecast(model, X_train, scaler, horizon_steps, n_features, device, train_window=48, y_future_scaled=None):
    model.eval()
    X_current = torch.tensor(X_train[-train_window:], dtype=torch.float32).unsqueeze(0).to(device)

    predictions = []
    with torch.no_grad():
        for h in range(horizon_steps):
            pred = model(X_current)
            predictions.append(pred.cpu().numpy())
            if y_future_scaled is not None:
                next_input = torch.tensor(y_future_scaled[h], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            else:
                next_input = pred.unsqueeze(1)
            X_current = torch.cat([X_current[:, 1:, :], next_input], dim=1)

    pred_future = np.concatenate(predictions, axis=0)
    pred_future = scaler.inverse_transform(pred_future)
    return pred_future

def main():
    parser = argparse.ArgumentParser(description="EEG RNN Forecast")
    parser.add_argument("--imputed_path", type=str, required=True)
    parser.add_argument("--ground_path", type=str, required=True)
    parser.add_argument("--horizon_steps", type=int, default=24)
    parser.add_argument("--history_timesteps", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="gru", choices=["gru", "lstm"])
    parser.add_argument("--train_window", type=int, default=48)
    parser.add_argument("--target_dims", type=str, default=None, help="e.g., 0,1,2")
    parser.add_argument("--out_dir", type=str, default="./save/eeg_baseline_gru")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_teacher_forcing", action="store_true", help="Use ground truth for sliding window")
    parser.add_argument("--use_ground_truth_train", action="store_true", help="Use ground truth for training (fair comparison)")

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    target_dims = None
    if args.target_dims:
        target_dims = [int(x) for x in args.target_dims.split(',')]

    print("=" * 60)
    print(f"EEG {args.model.upper()} Forecast (Baseline)")
    if args.use_ground_truth_train:
        print("Training with GROUND TRUTH (fair comparison mode)")
    else:
        print("Training with IMPUTED data")
    print("=" * 60)

    X_train, y_future_orig, y_future_scaled, scaler = prepare_data(
        args.imputed_path, args.ground_path,
        args.history_timesteps, target_dims, args.use_ground_truth_train
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_future shape: {y_future_orig.shape}")
    print(f"Train window: {args.train_window}, Epochs: {args.epochs}")

    model = train_model(
        X_train, args.hidden_size, args.num_layers,
        args.epochs, args.batch_size, args.lr, args.device, args.seed, args.model, args.train_window
    )

    y_for_forecast = y_future_scaled if args.use_teacher_forcing else None
    pred_future = forecast(model, X_train, scaler, args.horizon_steps, X_train.shape[1], args.device, args.train_window, y_for_forecast)

    rmse = np.sqrt(np.mean((y_future_orig - pred_future) ** 2))
    mae = np.mean(np.abs(y_future_orig - pred_future))

    print(f"\nForecast RMSE: {rmse:.4f}")
    print(f"Forecast MAE: {mae:.4f}")

    pred_df = pd.DataFrame(pred_future, columns=[f"dim_{i}" for i in range(pred_future.shape[1])])
    pred_df.to_csv(os.path.join(args.out_dir, "future_pred.csv"), index=False)

    results = {
        "method": f"EEG_{args.model.upper()}_Baseline",
        "rmse": float(rmse),
        "mae": float(mae),
        "horizon_steps": args.horizon_steps,
        "history_timesteps": args.history_timesteps,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "epochs": args.epochs,
        "train_window": args.train_window,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {args.out_dir}")
    return results

if __name__ == "__main__":
    main()