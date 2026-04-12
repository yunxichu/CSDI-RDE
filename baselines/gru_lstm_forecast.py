# -*- coding: utf-8 -*-
"""
通用 GRU/LSTM 预测脚本
============================================================
支持数据集：PM2.5, EEG, Lorenz63, Lorenz96

示例：
  # PM2.5
  python baselines/gru_lstm_forecast.py \
    --dataset pm25 --model gru \
    --imputed_history_path ./save/pm25_history_imputed_split0.5_seed42_20260324_154400/history_imputed.csv \
    --ground_path ./data/pm25/Code/STMVL/SampleData/pm25_ground.txt \
    --split_ratio 0.5 --horizon_days 1 \
    --window_size 48 --hidden_size 64 --num_layers 2 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42

  # EEG
  python baselines/gru_lstm_forecast.py \
    --dataset eeg --model lstm \
    --imputed_path ./save/eeg_csdi_imputed.npy \
    --ground_path ./data/eeg/eeg_ground.npy \
    --history_timesteps 100 --horizon_steps 24 \
    --target_dims 0,1,2 \
    --window_size 48 --hidden_size 64 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42
"""

import os, sys, json, time, random, argparse, datetime, warnings
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
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def safe_json_dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False, default=str)

def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0: return {"rmse": np.nan, "mae": np.nan}
    diff = y_true[mask] - y_pred[mask]
    return {"rmse": float(np.sqrt(np.mean(diff**2))), "mae": float(np.mean(np.abs(diff)))}


class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def build_windows(data, window_size, steps_ahead=1):
    T, D = data.shape
    xs, ys = [], []
    for i in range(T - window_size - steps_ahead + 1):
        xs.append(data[i: i + window_size])
        ys.append(data[i + window_size + steps_ahead - 1])
    return np.stack(xs), np.stack(ys)


def load_pm25_data(args):
    df_full = pd.read_csv(args.ground_path, index_col="datetime", parse_dates=True).sort_index()
    sp = int(len(df_full) * args.split_ratio)
    hist_full = df_full.iloc[:sp]
    fut_full = df_full.iloc[sp:]
    df_hist = pd.read_csv(args.imputed_history_path, index_col="datetime", parse_dates=True).sort_index()
    history = df_hist.values.astype(np.float64)
    columns = list(df_full.columns)
    fut_true = fut_full.values.astype(np.float64)
    if args.horizon_days > 0:
        dt = fut_full.index.to_series().diff().dropna().median()
        spd = max(1, int(round(pd.Timedelta(days=1) / dt))) if dt > pd.Timedelta(0) else 24
        horizon = int(round(args.horizon_days * spd))
    elif args.horizon_steps > 0:
        horizon = int(args.horizon_steps)
    else:
        horizon = len(fut_full)
    horizon = max(1, min(horizon, len(fut_full)))
    return history, fut_true[:horizon], columns, horizon, fut_full.index[:horizon]


def load_npy_data(args):
    data = np.load(args.imputed_path)
    ground = np.load(args.ground_path)
    if data.ndim == 3:
        data = data.reshape(-1, data.shape[-1])
    if ground.ndim == 3:
        ground = ground.reshape(-1, ground.shape[-1])
    if args.target_dims:
        dims = [int(x) for x in args.target_dims.split(',')]
        data = data[:, dims]
        ground = ground[:, dims]
    hist_len = args.history_timesteps
    history = data[:hist_len].astype(np.float64)
    fut_true = ground[hist_len:hist_len + args.horizon_steps].astype(np.float64)
    columns = [f"dim_{i}" for i in range(history.shape[1])]
    return history, fut_true, columns, args.horizon_steps, None


def load_csv_data(args):
    gt = np.loadtxt(args.ground_path, delimiter=',')
    if args.data_path and os.path.exists(args.data_path):
        data = np.loadtxt(args.data_path, delimiter=',')
    else:
        data = gt.copy()
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if gt.ndim == 1:
        gt = gt.reshape(-1, 1)
    hist_len = args.trainlength
    history = data[:hist_len].astype(np.float64)
    horizon = min(args.horizon_steps, len(gt) - hist_len)
    fut_true = gt[hist_len:hist_len + horizon].astype(np.float64)
    columns = [f"dim_{i}" for i in range(history.shape[1])]
    return history, fut_true, columns, horizon, None


def train_model(model, X_train, y_train, epochs, batch_size, lr, device, patience=15):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=max(1, patience // 2), factor=0.5)
    criterion = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    N = len(y_train)
    loader = DataLoader(TensorDataset(torch.arange(N)), batch_size=batch_size, shuffle=True)

    best_loss, best_state, no_imp = float("inf"), None, 0
    loss_history = []

    for epoch in tqdm(range(epochs), desc="  训练"):
        model.train()
        ep_loss = 0.0
        for (idx,) in loader:
            optimizer.zero_grad()
            pred = model(X_t[idx].to(device))
            loss = criterion(pred, y_t[idx].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= N
        loss_history.append(ep_loss)
        scheduler.step(ep_loss)
        if ep_loss < best_loss:
            best_loss = ep_loss
            no_imp = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)
    return loss_history, best_loss


def forecast(model, history_scaled, fut_true_scaled, horizon, window_size, device, scaler):
    T_hist = history_scaled.shape[0]
    D = history_scaled.shape[1]
    full = np.concatenate([history_scaled, fut_true_scaled], axis=0).astype(np.float32)
    W = window_size
    windows = np.stack([full[T_hist - W + i: T_hist + i] for i in range(horizon)])

    model.eval()
    X_all = torch.tensor(windows, dtype=torch.float32)
    with torch.no_grad():
        preds_s = model(X_all.to(device)).cpu().numpy()
    preds = scaler.inverse_transform(preds_s.astype(np.float64))
    stds = np.zeros_like(preds)
    return preds, stds


def main():
    parser = argparse.ArgumentParser(description="GRU/LSTM Forecast (Universal)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["pm25", "eeg", "lorenz63", "lorenz96"])
    parser.add_argument("--model", type=str, required=True, choices=["gru", "lstm"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--imputed_history_path", type=str, default="")
    parser.add_argument("--ground_path", type=str, required=True)
    parser.add_argument("--split_ratio", type=float, default=0.5)
    parser.add_argument("--horizon_days", type=float, default=0.0)

    parser.add_argument("--imputed_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--history_timesteps", type=int, default=100)
    parser.add_argument("--horizon_steps", type=int, default=24)
    parser.add_argument("--trainlength", type=int, default=60)
    parser.add_argument("--target_dims", type=str, default=None)

    parser.add_argument("--window_size", type=int, default=48)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--steps_ahead", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--plot_dim", type=int, default=0)
    parser.add_argument("--skip_metrics", action="store_true")
    args = parser.parse_args()

    set_global_seed(args.seed)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = f"./save/{args.dataset}_{args.model}_{now}/"
    ensure_dir(out_dir)
    safe_json_dump(vars(args), os.path.join(out_dir, "args.json"))

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    if args.dataset == "pm25":
        history, fut_true, columns, horizon, fut_index = load_pm25_data(args)
    elif args.dataset == "eeg":
        history, fut_true, columns, horizon, fut_index = load_npy_data(args)
    else:
        history, fut_true, columns, horizon, fut_index = load_csv_data(args)

    D = history.shape[1]
    print(f"\n{'='*70}\n{args.dataset.upper()} 预测 - {args.model.upper()}\n{'='*70}")
    print(f"history: {history.shape}, future: {fut_true.shape}, horizon: {horizon}")

    scaler = StandardScaler()
    history_scaled = scaler.fit_transform(history).astype(np.float32)
    fut_true_scaled = scaler.transform(fut_true).astype(np.float32)

    W = int(args.window_size)
    sa = int(args.steps_ahead)
    X_win, y_win = build_windows(history_scaled, W, sa)

    ModelClass = SimpleGRU if args.model == "gru" else SimpleLSTM
    model = ModelClass(D, args.hidden_size, args.num_layers, D)

    print(f"train_samples={X_win.shape[0]}, hidden={args.hidden_size}, layers={args.num_layers}")

    t0 = time.time()
    loss_hist, best_loss = train_model(
        model, X_win, y_win, args.epochs, args.batch_size, args.lr, device, args.patience)
    train_time = time.time() - t0
    print(f"  训练耗时 {train_time:.1f}s, best MSE={best_loss:.6f}")

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    t1 = time.time()
    preds, stds = forecast(model, history_scaled, fut_true_scaled,
                           horizon, W, device, scaler)
    print(f"  预测耗时 {time.time()-t1:.1f}s")

    if fut_index is not None:
        df_pred = pd.DataFrame(preds, index=fut_index, columns=columns)
        df_std = pd.DataFrame(stds, index=fut_index, columns=columns)
    else:
        df_pred = pd.DataFrame(preds, columns=columns)
        df_std = pd.DataFrame(stds, columns=columns)
    df_pred.to_csv(os.path.join(out_dir, "future_pred.csv"))
    df_std.to_csv(os.path.join(out_dir, "future_pred_std.csv"))
    np.save(os.path.join(out_dir, "future_pred.npy"), preds)
    np.save(os.path.join(out_dir, "future_pred_std.npy"), stds)

    if not args.skip_metrics:
        overall = compute_metrics(fut_true, preds)
        safe_json_dump({"overall": overall, "horizon": horizon},
                       os.path.join(out_dir, "metrics.json"))
        per_dim = [{"dim": j, "name": str(col),
                    **compute_metrics(fut_true[:, j], preds[:, j])}
                   for j, col in enumerate(columns)]
        pd.DataFrame(per_dim).to_csv(os.path.join(out_dir, "metrics_per_dim.csv"), index=False)

        d = max(0, min(int(args.plot_dim), D - 1))
        plt.figure(figsize=(14, 5))
        plt.plot(fut_true[:, d], label=f"True (dim {d})", color="steelblue")
        plt.plot(preds[:, d], label=f"{args.model.upper()} (dim {d})", color="tomato")
        plt.xlabel("Time"); plt.ylabel("Value")
        plt.title(f"{args.model.upper()} Forecast vs True ({args.dataset}, dim {d})")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"plot_forecast_dim{d}.png"), dpi=150)
        plt.close()

        print(f"\n整体评估（{args.model.upper()}, {args.dataset}）：")
        print(json.dumps(overall, indent=4, ensure_ascii=False))

    print(f"\n输出目录：{out_dir}")


if __name__ == "__main__":
    main()
