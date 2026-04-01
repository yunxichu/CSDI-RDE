"""
EEG GRU/LSTM Forecast
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

def prepare_data(imputed_history_path, impute_meta_path, ground_path, history_timesteps, horizon_steps, target_dims=None):
    data = np.load(imputed_history_path)
    ground = np.load(ground_path)

    if data.ndim == 3:
        data = data.reshape(-1, data.shape[-1])

    if target_dims is not None:
        data = data[:, target_dims]
        ground = ground[:, target_dims]

    with open(impute_meta_path, 'r') as f:
        meta = json.load(f)
    split_point = meta.get('split_point', int(len(ground) * 0.5))

    hist_data = data[split_point - history_timesteps:split_point]
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

    X_slide, y_slide = prepare_sliding_window_data(X_train, y_future, train_window)

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

def forecast(model, X_train, scaler, horizon_steps, n_features, device, train_window=24):
    model.eval()
    X_current = torch.tensor(X_train[-train_window:], dtype=torch.float32).unsqueeze(0).to(device)

    predictions = []
    with torch.no_grad():
        for _ in range(horizon_steps):
            pred = model(X_current)
            predictions.append(pred.cpu().numpy())
            X_current = torch.cat([X_current[:, 1:, :], pred.unsqueeze(1)], dim=1)

    pred_future = np.concatenate(predictions, axis=0)
    pred_future = scaler.inverse_transform(pred_future)
    return pred_future

def main():
    parser = argparse.ArgumentParser(description="EEG RNN Forecast")
    parser.add_argument("--imputed_history_path", type=str, required=True)
    parser.add_argument("--impute_meta_path", type=str, required=True)
    parser.add_argument("--ground_path", type=str, required=True)
    parser.add_argument("--horizon_steps", type=int, default=24)
    parser.add_argument("--history_timesteps", type=int, default=72)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="gru", choices=["gru", "lstm"])
    parser.add_argument("--train_window", type=int, default=48)
    parser.add_argument("--target_dims", type=str, default=None, help="e.g., 0,1,2,3")
    parser.add_argument("--out_dir", type=str, default="./save/eeg_forecast")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    target_dims = None
    if args.target_dims:
        target_dims = [int(x) for x in args.target_dims.split(',')]

    print("=" * 60)
    print(f"EEG {args.model.upper()} Forecast")
    print("=" * 60)

    X_train, y_future_orig, y_future_scaled, scaler, split_point = prepare_data(
        args.imputed_history_path, args.impute_meta_path, args.ground_path,
        args.history_timesteps, args.horizon_steps, target_dims
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_future shape: {y_future_orig.shape}")
    print(f"Sliding window training: X shape ({args.train_window}, {X_train.shape[0] - args.train_window}, {X_train.shape[1]}), y shape ({X_train.shape[0] - args.train_window}, {X_train.shape[1]})")

    model = train_model(
        X_train, y_future_scaled, args.hidden_size, args.num_layers,
        args.epochs, args.batch_size, args.lr, args.device, args.seed, args.model, args.train_window
    )

    pred_future = forecast(model, X_train, scaler, args.horizon_steps, X_train.shape[1], args.device, args.train_window)

    rmse = np.sqrt(np.mean((y_future_orig - pred_future) ** 2))
    mae = np.mean(np.abs(y_future_orig - pred_future))

    print(f"\nForecast RMSE: {rmse:.4f}")
    print(f"Forecast MAE: {mae:.4f}")

    pred_df = pd.DataFrame(pred_future, columns=[f"dim_{i}" for i in range(pred_future.shape[1])])
    pred_df.to_csv(os.path.join(args.out_dir, "future_pred.csv"), index=False)

    metrics = {"rmse": float(rmse), "mae": float(mae), "horizon": args.horizon_steps}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nOutput: {args.out_dir}")

if __name__ == "__main__":
    main()