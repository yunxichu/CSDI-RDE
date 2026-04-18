# -*- coding: utf-8 -*-
"""
通用 NeuralCDE 预测脚本（修复版）
============================================================
基于 PM2.5 正确版本，修复了以下问题：
1. 训练目标：用滑动窗口预测下一时刻（而非重建自身）
2. CubicSpline API：正确使用 hermite_cubic_coefficients
3. cdeint 调用：使用关键字参数避免顺序问题
4. 预测策略：真值滑窗 + 批量推理

支持数据集：PM2.5, EEG, Lorenz63, Lorenz96

安装：
  pip install torch torchdiffeq torchcde

示例：
  # PM2.5
  python baselines/neuralcde_forecast.py \
    --dataset pm25 \
    --imputed_history_path ./save/pm25_history_imputed_split0.5_seed42_20260324_154400/history_imputed.csv \
    --ground_path ./data/pm25/Code/STMVL/SampleData/pm25_ground.txt \
    --split_ratio 0.5 --horizon_days 1 \
    --window_size 48 --hidden_channels 64 --num_layers 3 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42

  # EEG
  python baselines/neuralcde_forecast.py \
    --dataset eeg \
    --imputed_path ./save/eeg_csdi_imputed.npy \
    --ground_path ./data/eeg/eeg_ground.npy \
    --history_timesteps 100 --horizon_steps 24 \
    --target_dims 0,1,2 \
    --window_size 48 --hidden_channels 64 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42

  # Lorenz96
  python baselines/neuralcde_forecast.py \
    --dataset lorenz96 \
    --data_path ./lorenz96_rde_delay/results/imputed_100_*.csv \
    --ground_path ./lorenz96_rde_delay/results/gt_100_*.csv \
    --trainlength 60 --horizon_steps 40 \
    --window_size 20 --hidden_channels 64 \
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
    def __init__(self, input_channels, hidden_channels, output_channels,
                 num_layers=3, step_size=None):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.step_size = step_size
        self.initial = nn.Linear(input_channels, hidden_channels)
        self.func = CDEFunc(input_channels, hidden_channels, num_layers)
        self.readout = nn.Linear(hidden_channels, output_channels)

    def forward(self, coeffs, x_last=None):
        import torchcde
        X = torchcde.CubicSpline(coeffs)
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)
        if self.step_size is not None:
            z_T = torchcde.cdeint(X=X, func=self.func, z0=z0,
                                  t=X.interval, adjoint=False,
                                  method='rk4',
                                  options=dict(step_size=self.step_size))
        else:
            z_T = torchcde.cdeint(X=X, func=self.func, z0=z0,
                                  t=X.interval, adjoint=False)
        z_final = z_T[:, -1]
        delta = self.readout(z_final)
        if x_last is not None:
            return x_last + delta
        return delta


def build_windows(data, window_size, steps_ahead=1):
    T, D = data.shape
    xs, ys = [], []
    for i in range(T - window_size - steps_ahead + 1):
        xs.append(data[i: i + window_size])
        ys.append(data[i + window_size + steps_ahead - 1])
    return np.stack(xs), np.stack(ys)


def make_cde_input(X_win):
    N, W, D = X_win.shape
    t_ = np.linspace(0., 1., W, dtype=np.float32)[None, :, None]
    t_ = np.broadcast_to(t_, (N, W, 1)).copy()
    return np.concatenate([t_, X_win.astype(np.float32)], axis=-1)


def train_model(model, X_train_np, y_train_np, epochs, batch_size, lr,
                device, patience=15, verbose=True):
    import torchcde
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=max(1, patience // 2), factor=0.5, verbose=False)

    X_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    with torch.no_grad():
        coeffs_cpu = torchcde.hermite_cubic_coefficients_with_backward_differences(X_tensor)
    y_np = y_train_np.astype(np.float32)
    y_mask_np = (~np.isnan(y_np)).astype(np.float32)
    y_clean = np.nan_to_num(y_np, nan=0.0)
    x_last_raw = X_train_np[:, -1, 1:].astype(np.float32)
    x_last_clean = np.nan_to_num(x_last_raw, nan=0.0)
    y_tensor = torch.tensor(y_clean, dtype=torch.float32)
    y_mask = torch.tensor(y_mask_np, dtype=torch.float32)
    x_last_cpu = torch.tensor(x_last_clean, dtype=torch.float32)
    N = len(y_train_np)

    loader = DataLoader(TensorDataset(torch.arange(N)),
                        batch_size=batch_size, shuffle=True, drop_last=False)
    best_loss, best_state, no_imp = float("inf"), None, 0
    loss_history = []

    for epoch in tqdm(range(epochs), desc="  训练", disable=not verbose):
        model.train()
        ep_loss = 0.0
        for (idx,) in loader:
            optimizer.zero_grad()
            b_coeffs = coeffs_cpu[idx].to(device)
            b_y = y_tensor[idx].to(device)
            b_m = y_mask[idx].to(device)
            b_x_last = x_last_cpu[idx].to(device)
            pred = model(b_coeffs, b_x_last)
            sq = (pred - b_y) ** 2 * b_m
            loss = sq.sum() / b_m.sum().clamp_min(1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= N
        loss_history.append(ep_loss)
        scheduler.step(ep_loss)
        # 修复: 当 ep_loss 是 NaN/Inf 时, best_loss=inf 比较返回 False
        # 导致 best_state 永远是 None, 且 no_imp 一直累加 → 提前 early stop
        if np.isfinite(ep_loss) and ep_loss < best_loss:
            best_loss = ep_loss
            no_imp = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        if no_imp >= patience:
            if verbose:
                tqdm.write(f"  Early stopping @ epoch {epoch+1}, best MSE={best_loss:.6f}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)
    return loss_history, best_loss


def forecast(model, history_scaled, fut_true_scaled, horizon, window_size,
             device, scaler, batch_size=256, verbose=True):
    import torchcde
    T_hist = history_scaled.shape[0]
    D = history_scaled.shape[1]
    fut_clean = fut_true_scaled.copy()
    nan_mask = np.isnan(fut_clean)
    if nan_mask.any():
        if verbose:
            print(f"  检测到fut_true中有{nan_mask.sum()}个NaN，使用前向填充")
        for j in range(fut_clean.shape[1]):
            col = fut_clean[:, j]
            nan_idx = np.where(np.isnan(col))[0]
            for idx in nan_idx:
                last_valid = history_scaled[-1, j]
                for k in range(idx - 1, -1, -1):
                    if not np.isnan(fut_clean[k, j]):
                        last_valid = fut_clean[k, j]
                        break
                col[idx] = last_valid
            fut_clean[:, j] = col
    full = np.concatenate([history_scaled, fut_clean], axis=0).astype(np.float32)
    W = window_size
    windows = np.stack([full[T_hist - W + i: T_hist + i] for i in range(horizon)])
    X_all_np = make_cde_input(windows)
    X_all = torch.tensor(X_all_np, dtype=torch.float32)
    x_last_np = np.nan_to_num(windows[:, -1, :].astype(np.float32), nan=0.0)
    x_last_all = torch.tensor(x_last_np, dtype=torch.float32)
    with torch.no_grad():
        coeffs_all = torchcde.hermite_cubic_coefficients_with_backward_differences(X_all)

    model.eval()
    preds_list = []
    n_batches = (horizon + batch_size - 1) // batch_size
    for b in tqdm(range(n_batches), desc="  批量推理", disable=not verbose):
        sl = slice(b * batch_size, (b + 1) * batch_size)
        b_coeffs = coeffs_all[sl].to(device)
        b_x_last = x_last_all[sl].to(device)
        with torch.no_grad():
            p = model(b_coeffs, b_x_last).cpu().numpy()
        preds_list.append(p)

    preds_s = np.concatenate(preds_list, axis=0)
    preds = scaler.inverse_transform(preds_s.astype(np.float64))
    stds = np.zeros_like(preds)
    return preds, stds


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


def main():
    parser = argparse.ArgumentParser(description="NeuralCDE Forecast (Fixed Universal Version)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["pm25", "eeg", "lorenz63", "lorenz96"],
                        help="Dataset name")
    parser.add_argument("--seed", type=int, default=42)

    # PM2.5 specific
    parser.add_argument("--imputed_history_path", type=str, default="")
    parser.add_argument("--ground_path", type=str, required=True)
    parser.add_argument("--split_ratio", type=float, default=0.5)
    parser.add_argument("--horizon_days", type=float, default=0.0)

    # EEG / Lorenz specific
    parser.add_argument("--imputed_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--history_timesteps", type=int, default=100)
    parser.add_argument("--horizon_steps", type=int, default=24)
    parser.add_argument("--trainlength", type=int, default=60)
    parser.add_argument("--target_dims", type=str, default=None)

    # Model
    parser.add_argument("--window_size", type=int, default=48)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--steps_ahead", type=int, default=1)
    parser.add_argument("--use_fixed_solver", action="store_true")
    parser.add_argument("--rk4_nsteps", type=int, default=0)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)

    # Other
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--plot_dim", type=int, default=0)
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--no_verbose", action="store_true")
    args = parser.parse_args()

    verbose = not args.no_verbose
    set_global_seed(args.seed)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = f"./save/{args.dataset}_neuralcde_{now}/"
    ensure_dir(out_dir)
    safe_json_dump(vars(args), os.path.join(out_dir, "args.json"))

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            device = torch.device(args.device)
            if device.type == "cuda":
                torch.cuda.device_count()
        except (RuntimeError, AssertionError):
            print(f"[警告] CUDA不可用，回退到CPU")
            device = torch.device("cpu")
    print(f"[设备] {device}")

    # Load data
    if args.dataset == "pm25":
        history, fut_true, columns, horizon, fut_index = load_pm25_data(args)
    elif args.dataset == "eeg":
        history, fut_true, columns, horizon, fut_index = load_npy_data(args)
    else:
        history, fut_true, columns, horizon, fut_index = load_csv_data(args)

    D = history.shape[1]
    print(f"\n{'='*70}\n{args.dataset.upper()} 预测 - NeuralCDE (修复版)\n{'='*70}")
    print(f"history: {history.shape}, future: {fut_true.shape}, horizon: {horizon}")

    # Scale
    scaler = StandardScaler()
    history_scaled = scaler.fit_transform(history).astype(np.float32)
    fut_true_scaled = scaler.transform(fut_true).astype(np.float32)

    # Build training windows
    W = int(args.window_size)
    sa = int(args.steps_ahead)
    assert W >= 2, "window_size must >= 2"
    assert history_scaled.shape[0] > W + sa, f"history too short for window_size={W}"

    X_win, y_win = build_windows(history_scaled, W, sa)
    X_win_t = make_cde_input(X_win)
    input_channels = D + 1
    output_channels = D

    print(f"train_samples={X_win_t.shape[0]}, input_ch={input_channels}, "
          f"hidden_ch={args.hidden_channels}, layers={args.num_layers}")

    # Model
    if args.use_fixed_solver:
        nsteps = args.rk4_nsteps if args.rk4_nsteps > 0 else (W - 1)
        step_size = 1.0 / nsteps
    else:
        step_size = None

    model = NeuralCDE(input_channels, args.hidden_channels, output_channels,
                      num_layers=args.num_layers, step_size=step_size)

    # Train
    print("\n[训练阶段]")
    t0 = time.time()
    loss_hist, best_loss = train_model(
        model, X_win_t, y_win,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=device, patience=args.patience, verbose=verbose)
    train_time = time.time() - t0
    print(f"  耗时 {train_time:.1f}s, best MSE={best_loss:.6f}")

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    safe_json_dump(loss_hist, os.path.join(out_dir, "loss_history.json"))

    # Forecast
    print("\n[预测阶段]")
    t1 = time.time()
    preds, stds = forecast(
        model, history_scaled, fut_true_scaled,
        horizon=horizon, window_size=W,
        device=device, scaler=scaler,
        batch_size=args.batch_size, verbose=verbose)
    print(f"  耗时 {time.time()-t1:.1f}s")

    # Save
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

    # Metrics
    if not args.skip_metrics:
        overall = compute_metrics(fut_true, preds)
        safe_json_dump({"overall": overall, "horizon": horizon},
                       os.path.join(out_dir, "metrics.json"))
        per_dim = [{"dim": j, "name": str(col),
                    **compute_metrics(fut_true[:, j], preds[:, j])}
                   for j, col in enumerate(columns)]
        pd.DataFrame(per_dim).to_csv(
            os.path.join(out_dir, "metrics_per_dim.csv"), index=False)

        d = max(0, min(int(args.plot_dim), D - 1))
        plt.figure(figsize=(14, 5))
        plt.plot(fut_true[:, d], label=f"True (dim {d})", color="steelblue")
        plt.plot(preds[:, d], label=f"NeuralCDE (dim {d})", color="tomato")
        plt.xlabel("Time"); plt.ylabel("Value")
        plt.title(f"NeuralCDE Forecast vs True ({args.dataset}, dim {d})")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"plot_forecast_dim{d}.png"), dpi=150)
        plt.close()

        print(f"\n整体评估（NeuralCDE, {args.dataset}）：")
        print(json.dumps(overall, indent=4, ensure_ascii=False))

    print(f"\n输出目录：{out_dir}")


if __name__ == "__main__":
    main()
