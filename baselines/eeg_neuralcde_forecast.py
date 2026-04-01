# -*- coding: utf-8 -*-
"""
EEG NeuralCDE 预测脚本

基于 torchcde 的神经控制微分方程方法
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

class CDEFunc(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels * input_channels),
        )

    def forward(self, t, z):
        return self.net(z).view(-1, self.hidden_channels, self.input_channels)

class NeuralCDEModel(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.initial = nn.Linear(input_channels, hidden_channels)
        self.func = CDEFunc(input_channels, hidden_channels)
        self.readout = nn.Linear(hidden_channels, input_channels)

    def forward(self, coeffs, t_span):
        from torchcde import CubicSpline, cdeint
        X = CubicSpline(coeffs, t_span)
        z0 = self.initial(X.evaluate(t_span[0]))
        zT = cdeint(X, self.func, z0, t_span, method='rk4', options={'step_size': 0.05})
        return self.readout(zT)

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

def train_neuralcde(X_train, hidden_channels, epochs, lr, device, seed):
    set_global_seed(seed)
    n_timesteps, n_features = X_train.shape

    from torchcde import natural_cubic_coeffs
    times = torch.linspace(0, 1, n_timesteps)
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    coeffs = natural_cubic_coeffs(X_tensor, times)

    model = NeuralCDEModel(n_features, hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    t_span = torch.tensor([0.0, 1.0], dtype=torch.float32)

    model.train()
    for epoch in tqdm(range(epochs), desc="NeuralCDE training"):
        optimizer.zero_grad()
        pred = model(coeffs.to(device), t_span.to(device))
        target = X_tensor.to(device)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model

def forecast(model, X_train, scaler, horizon_steps, device, y_future_scaled=None):
    model.eval()
    n_timesteps, n_features = X_train.shape
    times = torch.linspace(0, 1, n_timesteps)

    from torchcde import natural_cubic_coeffs

    predictions = []
    X_current = X_train.copy()

    with torch.no_grad():
        for h in range(horizon_steps):
            X_tensor = torch.tensor(X_current, dtype=torch.float32)
            coeffs = natural_cubic_coeffs(X_tensor, times)
            t_span = torch.tensor([0.0, 1.0], dtype=torch.float32)

            pred = model(coeffs.to(device), t_span.to(device))
            pred_val = pred[-1, :].cpu().numpy()
            predictions.append(pred_val)

            if h < horizon_steps - 1:
                if y_future_scaled is not None:
                    new_row = y_future_scaled[h]
                else:
                    new_row = pred_val
                X_current = np.vstack([X_current[1:], new_row.reshape(1, -1)])

    pred_future = np.array(predictions)
    pred_future = scaler.inverse_transform(pred_future)
    return pred_future

def main():
    parser = argparse.ArgumentParser(description="EEG NeuralCDE Forecast")
    parser.add_argument("--imputed_path", type=str, required=True)
    parser.add_argument("--ground_path", type=str, required=True)
    parser.add_argument("--horizon_steps", type=int, default=24)
    parser.add_argument("--history_timesteps", type=int, default=100)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_dims", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="./save/eeg_neuralcde")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_teacher_forcing", action="store_true", help="Use ground truth for sliding window")
    parser.add_argument("--use_ground_truth_train", action="store_true", help="Use ground truth for training (fair comparison)")

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    target_dims = None
    if args.target_dims:
        target_dims = [int(x) for x in args.target_dims.split(',')]

    print("=" * 60)
    print("EEG NeuralCDE Forecast")
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

    model = train_neuralcde(
        X_train, args.hidden_channels,
        args.epochs, args.lr,
        args.device, args.seed
    )

    y_for_forecast = y_future_scaled if args.use_teacher_forcing else None
    pred_future = forecast(model, X_train, scaler, args.horizon_steps, args.device, y_for_forecast)

    metrics = compute_metrics(y_future, pred_future)
    print(f"\nForecast RMSE: {metrics['rmse']:.4f}")
    print(f"Forecast MAE: {metrics['mae']:.4f}")

    pred_df = pd.DataFrame(pred_future, columns=[f"dim_{i}" for i in range(pred_future.shape[1])])
    pred_df.to_csv(os.path.join(args.out_dir, "future_pred.csv"), index=False)

    results = {
        "method": "NeuralCDE",
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
