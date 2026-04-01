# -*- coding: utf-8 -*-
"""
EEG GRU-ODE-Bayes 预测脚本

简化版本，用于时间序列预测
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def set_global_seed(seed):
    import random
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return {"rmse": np.nan, "mae": np.nan}
    diff = y_true[mask] - y_pred[mask]
    return {"rmse": float(np.sqrt(np.mean(diff**2))), "mae": float(np.mean(np.abs(diff)))}

class GRUODECell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lin_xz = nn.Linear(input_size, hidden_size)
        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_xr = nn.Linear(input_size, hidden_size)
        self.lin_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_xh = nn.Linear(input_size, hidden_size)
        self.lin_hh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h))
        r = torch.sigmoid(self.lin_xr(x) + self.lin_hr(h))
        h_tilde = torch.tanh(self.lin_xh(x) + self.lin_hh(r * h))
        h_new = (1 - z) * h + z * h_tilde
        return h_new

class GRUODEBayesModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru_cells = nn.ModuleList([
            GRUODECell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        batch_size, n_steps, input_size = X.shape
        h = [torch.zeros(batch_size, self.hidden_size, device=X.device) for _ in range(self.num_layers)]
        outputs = []
        for t in range(n_steps):
            x_t = X[:, t, :]
            for i, cell in enumerate(self.gru_cells):
                h[i] = cell(x_t, h[i])
                x_t = h[i]
            outputs.append(self.readout(h[-1]))
        return torch.stack(outputs, dim=1)

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

    X_train = scaler.transform(hist_data).astype(np.float32)
    y_future_scaled = scaler.transform(fut_data).astype(np.float32)

    return X_train, fut_data, y_future_scaled, scaler

def train_gruodebayes(X_train, hidden_size, epochs, lr, device, seed, train_window=48):
    set_global_seed(seed)
    n_timesteps, n_features = X_train.shape

    model = GRUODEBayesModel(n_features, hidden_size, n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_slide = []
    y_slide = []
    for i in range(train_window, n_timesteps):
        X_slide.append(X_train[i - train_window:i])
        y_slide.append(X_train[i])
    X_slide = np.array(X_slide)
    y_slide = np.array(y_slide)

    X_tensor = torch.tensor(X_slide, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_slide, dtype=torch.float32).to(device)

    model.train()
    for epoch in tqdm(range(epochs), desc="GRU-ODE-Bayes training"):
        perm = torch.randperm(len(X_slide))
        total_loss = 0
        n_batches = 0
        for i in range(0, len(X_slide), 32):
            batch_idx = perm[i:i+32]
            batch_X = X_tensor[batch_idx]
            batch_y = y_tensor[batch_idx]
            optimizer.zero_grad()
            pred = model(batch_X)
            pred_last = pred[:, -1, :]
            loss = criterion(pred_last, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")

    return model, train_window

def forecast(model, X_train, scaler, horizon_steps, device, train_window, y_future_scaled=None):
    model.eval()
    X_current = torch.tensor(X_train[-train_window:], dtype=torch.float32).unsqueeze(0).to(device)

    predictions = []
    with torch.no_grad():
        for h in range(horizon_steps):
            pred = model(X_current)
            pred_vals = pred[0, -1, :].cpu().numpy()
            predictions.append(pred_vals)

            if y_future_scaled is not None:
                next_input = torch.tensor(y_future_scaled[h], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            else:
                next_input = torch.tensor(pred_vals, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            X_current = torch.cat([X_current[:, 1:, :], next_input], dim=1)

    pred_future = np.array(predictions)
    pred_future = scaler.inverse_transform(pred_future)
    return pred_future

def main():
    parser = argparse.ArgumentParser(description="EEG GRU-ODE-Bayes Forecast")
    parser.add_argument("--imputed_path", type=str, required=True)
    parser.add_argument("--ground_path", type=str, required=True)
    parser.add_argument("--horizon_steps", type=int, default=24)
    parser.add_argument("--history_timesteps", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_dims", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="./save/eeg_gruodebayes")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_teacher_forcing", action="store_true", help="Use ground truth for sliding window")
    parser.add_argument("--use_ground_truth_train", action="store_true", help="Use ground truth for training (fair comparison)")

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    target_dims = None
    if args.target_dims:
        target_dims = [int(x) for x in args.target_dims.split(',')]

    print("=" * 60)
    print("EEG GRU-ODE-Bayes Forecast")
    if args.use_ground_truth_train:
        print("Training with GROUND TRUTH (fair comparison mode)")
    else:
        print("Training with IMPUTED data")
    print("=" * 60)

    X_train, y_future, y_future_scaled, scaler = prepare_data(
        args.imputed_path, args.ground_path,
        args.history_timesteps, target_dims, args.use_ground_truth_train
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_future shape: {y_future.shape}")

    model, train_window = train_gruodebayes(
        X_train, args.hidden_size,
        args.epochs, args.lr, args.device, args.seed
    )

    y_for_forecast = y_future_scaled if args.use_teacher_forcing else None
    pred_future = forecast(model, X_train, scaler, args.horizon_steps, args.device, train_window, y_for_forecast)

    metrics = compute_metrics(y_future, pred_future)
    print(f"\nForecast RMSE: {metrics['rmse']:.4f}")
    print(f"Forecast MAE: {metrics['mae']:.4f}")

    pred_df = pd.DataFrame(pred_future, columns=[f"dim_{i}" for i in range(pred_future.shape[1])])
    pred_df.to_csv(os.path.join(args.out_dir, "future_pred.csv"), index=False)

    results = {
        "method": "GRU-ODE-Bayes",
        "rmse": float(metrics['rmse']),
        "mae": float(metrics['mae']),
        "args": {k: str(v) for k, v in vars(args).items() if k != 'device'}
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to {args.out_dir}")
    return results

if __name__ == "__main__":
    main()
