# -*- coding: utf-8 -*-
"""
通用 GRU-ODE-Bayes 预测脚本
============================================================
基于 PM2.5 完整复现版，支持所有数据集

示例：
  # PM2.5
  python baselines/gruodebayes_forecast.py \
    --dataset pm25 \
    --imputed_history_path ./save/pm25_history_imputed_split0.5_seed42_20260324_154400/history_imputed.csv \
    --ground_path ./data/pm25/Code/STMVL/SampleData/pm25_ground.txt \
    --split_ratio 0.5 --horizon_days 1 \
    --hidden_size 64 --p_hidden 32 --prep_hidden 32 \
    --window_size 48 --delta_t 0.1 --time_scale 0.02 --solver euler \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42

  # Lorenz96
  python baselines/gruodebayes_forecast.py \
    --dataset lorenz96 \
    --data_path ./lorenz96_rde_delay/results/imputed_100_20260323_192045.csv \
    --ground_path ./lorenz96_rde_delay/results/gt_100_20260323_192045.csv \
    --trainlength 60 --horizon_steps 40 \
    --window_size 20 --hidden_size 64 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42

  # EEG
  python baselines/gruodebayes_forecast.py \
    --dataset eeg \
    --imputed_path ./save/eeg_csdi_imputed.npy \
    --ground_path ./data/eeg/eeg_ground.npy \
    --history_timesteps 100 --horizon_steps 24 \
    --target_dims 0,1,2 \
    --window_size 48 --hidden_size 64 \
    --epochs 100 --batch_size 128 --lr 1e-3 --seed 42
"""

import os, sys, json, time, random, argparse, datetime, warnings, math
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


class GRUODECell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.lin_xz = nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_xn = nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_hn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h))
        n = torch.tanh(self.lin_xn(x) + self.lin_hn(z * h))
        return (1 - z) * (n - h)


class GRUObservationCellLogvar(nn.Module):
    def __init__(self, input_size, hidden_size, prep_hidden, bias=True):
        super().__init__()
        self.gru_d = nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)
        std = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep = nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))
        self.input_size = input_size
        self.prep_hidden = prep_hidden

    def forward(self, h, p, X_obs, M_obs, i_obs):
        p_obs = p[i_obs]
        mean, logvar = torch.chunk(p_obs, 2, dim=1)
        sigma = torch.exp(0.5 * logvar)
        error = (X_obs - mean) / sigma
        log_lik_c = np.log(np.sqrt(2 * np.pi))
        losses = 0.5 * ((torch.pow(error, 2) + logvar + 2 * log_lik_c) * M_obs)
        gru_input = torch.stack([X_obs, mean, logvar, error], dim=2).unsqueeze(2)
        gru_input = torch.matmul(gru_input, self.w_prep).squeeze(2) + self.bias_prep
        gru_input.relu_()
        gru_input = gru_input.permute(2, 0, 1)
        gru_input = (gru_input * M_obs).permute(1, 2, 0).contiguous().view(-1, self.prep_hidden * self.input_size)
        temp = h.clone()
        temp[i_obs] = self.gru_d(gru_input, h[i_obs])
        return temp, losses


def gaussian_KL(mu_1, mu_2, sigma_1, sigma_2):
    return (torch.log(sigma_2) - torch.log(sigma_1) +
            (torch.pow(sigma_1, 2) + torch.pow((mu_1 - mu_2), 2)) / (2 * sigma_2**2) - 0.5)


class NNFOwithBayesianJumps(nn.Module):
    def __init__(self, input_size, hidden_size, p_hidden, prep_hidden,
                 bias=True, cov_size=1, cov_hidden=1, solver="euler"):
        super().__init__()
        self.p_model = nn.Sequential(
            nn.Linear(hidden_size, p_hidden, bias=bias), nn.ReLU(),
            nn.Linear(p_hidden, 2 * input_size, bias=bias))
        self.gru_c = GRUODECell(2 * input_size, hidden_size, bias=bias)
        self.gru_obs = GRUObservationCellLogvar(input_size, hidden_size, prep_hidden, bias=bias)
        self.covariates_map = nn.Sequential(
            nn.Linear(cov_size, cov_hidden, bias=bias), nn.ReLU(),
            nn.Linear(cov_hidden, hidden_size, bias=bias), nn.Tanh())
        self.solver = solver
        self.input_size = input_size

    def ode_step(self, h, p, delta_t):
        if self.solver == "euler":
            h = h + delta_t * self.gru_c(p, h)
        elif self.solver == "midpoint":
            k = h + delta_t / 2 * self.gru_c(p, h)
            pk = self.p_model(k)
            h = h + delta_t * self.gru_c(pk, k)
        p = self.p_model(h)
        return h, p

    def forward(self, times, time_ptr, X, M, obs_idx, delta_t, T, cov, pred_step=0.0):
        h = self.covariates_map(cov)
        p = self.p_model(h)
        current_time = 0.0
        loss_1 = 0
        for i, obs_time in enumerate(times):
            while current_time < (obs_time - 0.001 * delta_t):
                h, p = self.ode_step(h, p, delta_t)
                current_time += delta_t
            start = time_ptr[i]
            end = time_ptr[i + 1]
            X_obs = X[start:end]
            M_obs = M[start:end]
            i_obs = obs_idx[start:end]
            h, losses = self.gru_obs(h, p, X_obs, M_obs, i_obs)
            loss_1 = loss_1 + losses.sum()
            p = self.p_model(h)
        while current_time < T - 0.001 * delta_t:
            h, p = self.ode_step(h, p, delta_t)
            current_time += delta_t
        if pred_step > 0:
            n_pred = int(pred_step / delta_t)
            for _ in range(n_pred):
                h, p = self.ode_step(h, p, delta_t)
        return h, p, loss_1


def build_gob_batches(data_scaled, window_size, delta_t, time_scale, batch_size, device):
    N_samples = data_scaled.shape[0] - window_size
    D = data_scaled.shape[1]
    all_times, all_time_ptr, all_X, all_M, all_obs_idx, all_cov, all_y = [], [], [], [], [], [], []
    ptr = 0
    for i in range(N_samples):
        win = data_scaled[i:i+window_size+1]
        times_i = (np.arange(window_size) * time_scale).tolist()
        time_ptr_i = list(range(window_size))
        X_i = torch.tensor(win[:window_size], dtype=torch.float32)
        M_i = torch.ones(window_size, D, dtype=torch.float32)
        obs_idx_i = torch.arange(1, dtype=torch.long).repeat(window_size)
        cov_i = torch.ones(1, 1, dtype=torch.float32)
        y_i = win[window_size]
        all_times.append(times_i)
        all_time_ptr.append(time_ptr_i)
        all_X.append(X_i)
        all_M.append(M_i)
        all_obs_idx.append(obs_idx_i)
        all_cov.append(cov_i)
        all_y.append(y_i)
    return all_times, all_time_ptr, all_X, all_M, all_obs_idx, all_cov, np.array(all_y)


def load_pm25_data(args):
    df_full = pd.read_csv(args.ground_path, index_col="datetime", parse_dates=True).sort_index()
    sp = int(len(df_full) * args.split_ratio)
    df_hist = pd.read_csv(args.imputed_history_path, index_col="datetime", parse_dates=True).sort_index()
    history = df_hist.values.astype(np.float64)
    columns = list(df_full.columns)
    fut_full = df_full.iloc[sp:]
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
    if data.ndim == 3: data = data.reshape(-1, data.shape[-1])
    if ground.ndim == 3: ground = ground.reshape(-1, ground.shape[-1])
    if args.target_dims:
        dims = [int(x) for x in args.target_dims.split(',')]
        data = data[:, dims]; ground = ground[:, dims]
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
    if data.ndim == 1: data = data.reshape(-1, 1)
    if gt.ndim == 1: gt = gt.reshape(-1, 1)
    hist_len = args.trainlength
    history = data[:hist_len].astype(np.float64)
    horizon = min(args.horizon_steps, len(gt) - hist_len)
    fut_true = gt[hist_len:hist_len + horizon].astype(np.float64)
    columns = [f"dim_{i}" for i in range(history.shape[1])]
    return history, fut_true, columns, horizon, None


def main():
    parser = argparse.ArgumentParser(description="GRU-ODE-Bayes Forecast (Universal)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["pm25", "eeg", "lorenz63", "lorenz96"])
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
    parser.add_argument("--p_hidden", type=int, default=32)
    parser.add_argument("--prep_hidden", type=int, default=32)
    parser.add_argument("--delta_t", type=float, default=0.1)
    parser.add_argument("--time_scale", type=float, default=0.02)
    parser.add_argument("--solver", type=str, default="euler", choices=["euler", "midpoint"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--plot_dim", type=int, default=0)
    args = parser.parse_args()

    set_global_seed(args.seed)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or f"./save/{args.dataset}_gruodebayes_{now}/"
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

    if args.dataset == "pm25":
        history, fut_true, columns, horizon, fut_index = load_pm25_data(args)
    elif args.dataset == "eeg":
        history, fut_true, columns, horizon, fut_index = load_npy_data(args)
    else:
        history, fut_true, columns, horizon, fut_index = load_csv_data(args)

    D = history.shape[1]
    print(f"\n{'='*70}\n{args.dataset.upper()} 预测 - GRU-ODE-Bayes\n{'='*70}")
    print(f"history: {history.shape}, future: {fut_true.shape}, horizon: {horizon}")

    scaler = StandardScaler()
    history_scaled = scaler.fit_transform(history).astype(np.float32)
    fut_true_scaled = scaler.transform(fut_true).astype(np.float32)

    W = int(args.window_size)
    model = NNFOwithBayesianJumps(
        input_size=D, hidden_size=args.hidden_size,
        p_hidden=args.p_hidden, prep_hidden=args.prep_hidden,
        solver=args.solver).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print(f"\n[训练阶段] window={W}, delta_t={args.delta_t}, time_scale={args.time_scale}")
    t0 = time.time()

    N_samples = len(history_scaled) - W
    best_loss, best_state, no_imp = float("inf"), None, 0

    for epoch in tqdm(range(args.epochs), desc="训练"):
        model.train()
        epoch_loss = 0.0
        indices = np.random.permutation(N_samples)
        bs = args.batch_size

        for b_start in range(0, N_samples, bs):
            b_idx = indices[b_start:b_start + bs]
            b_loss = 0.0
            optimizer.zero_grad()

            for i in b_idx:
                win = history_scaled[i:i+W+1]
                times_i = (np.arange(W) * args.time_scale).tolist()
                X_i = torch.tensor(win[:W], dtype=torch.float32, device=device)
                M_i = torch.ones(W, D, dtype=torch.float32, device=device)
                obs_idx_i = torch.zeros(W, dtype=torch.long, device=device)
                cov_i = torch.ones(1, 1, device=device)
                time_ptr_i = list(range(W + 1))
                T_val = W * args.time_scale

                h, p, loss = model(times_i, time_ptr_i, X_i, M_i, obs_idx_i,
                                   args.delta_t, T_val, cov_i)
                mean_pred = p[:, :D]
                target = torch.tensor(win[W], dtype=torch.float32, device=device).unsqueeze(0)
                b_loss = b_loss + nn.functional.mse_loss(mean_pred, target)

            b_loss = b_loss / len(b_idx)
            b_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += b_loss.item() * len(b_idx)

        epoch_loss /= N_samples
        scheduler.step(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
        if no_imp >= args.patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)
    print(f"  训练耗时 {time.time()-t0:.1f}s, best loss={best_loss:.6f}")
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    print("\n[预测阶段]")
    t1 = time.time()
    full = np.concatenate([history_scaled, fut_true_scaled], axis=0).astype(np.float32)
    preds_list = []

    model.eval()
    for step in tqdm(range(horizon), desc="预测"):
        win = full[len(history_scaled) - W + step: len(history_scaled) + step]
        times_i = (np.arange(W) * args.time_scale).tolist()
        X_i = torch.tensor(win[:W], dtype=torch.float32, device=device)
        M_i = torch.ones(W, D, dtype=torch.float32, device=device)
        obs_idx_i = torch.zeros(W, dtype=torch.long, device=device)
        cov_i = torch.ones(1, 1, device=device)
        time_ptr_i = list(range(W + 1))
        T_val = W * args.time_scale

        with torch.no_grad():
            h, p, _ = model(times_i, time_ptr_i, X_i, M_i, obs_idx_i,
                            args.delta_t, T_val, cov_i)
            mean_pred = p[:, :D].cpu().numpy()[0]
        preds_list.append(mean_pred)

    preds_s = np.array(preds_list)
    preds = scaler.inverse_transform(preds_s.astype(np.float64))
    stds = np.zeros_like(preds)
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

    overall = compute_metrics(fut_true, preds)
    safe_json_dump({"overall": overall, "horizon": horizon}, os.path.join(out_dir, "metrics.json"))
    per_dim = [{"dim": j, "name": str(col), **compute_metrics(fut_true[:, j], preds[:, j])}
               for j, col in enumerate(columns)]
    pd.DataFrame(per_dim).to_csv(os.path.join(out_dir, "metrics_per_dim.csv"), index=False)

    d = max(0, min(int(args.plot_dim), D - 1))
    plt.figure(figsize=(14, 5))
    plt.plot(fut_true[:, d], label=f"True (dim {d})", color="steelblue")
    plt.plot(preds[:, d], label=f"GRU-ODE-Bayes (dim {d})", color="tomato")
    plt.xlabel("Time"); plt.ylabel("Value")
    plt.title(f"GRU-ODE-Bayes Forecast ({args.dataset}, dim {d})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"plot_forecast_dim{d}.png"), dpi=150)
    plt.close()

    print(f"\n整体评估（GRU-ODE-Bayes, {args.dataset}）：")
    print(json.dumps(overall, indent=4, ensure_ascii=False))
    print(f"\n输出目录：{out_dir}")


if __name__ == "__main__":
    main()
