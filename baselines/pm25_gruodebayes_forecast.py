# -*- coding: utf-8 -*-
"""
PM2.5 后续预测脚本 —— GRU-ODE-Bayes 基线版
============================================================
基于：https://github.com/edebrouwer/gru_ode_bayes
完全复现官方 NNFOwithBayesianJumps 模型。

关键修正：
1. GRUODECell: n = tanh(lin_xn(x) + lin_hn(z * h))  ← 官方是 z*h
2. 观测更新需要 p（预测分布参数）作为输入
3. ODE演化时 p 作为输入，每步更新 p = p_model(h)
4. 时间尺度参数 time_scale：控制ODE演化的速度

运行示例：
  python baselines/pm25_gruodebayes_forecast.py \
    --imputed_history_path ./pm25_history_imputed_split0.5_seed42_20260128_101132/history_imputed.csv \
    --ground_path ./data/pm25/Code/STMVL/SampleData/pm25_ground.txt \
    --split_ratio 0.5 --horizon_days 1 \
    --hidden_size 64 --p_hidden 32 --prep_hidden 32 \
    --window_size 48 --delta_t 0.1 --time_scale 0.02 --solver euler \
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


# =============================================================================
# 工具函数
# =============================================================================

def set_global_seed(seed):
    seed = int(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def safe_json_dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False, default=str)

def basic_array_stats(x, name):
    x = np.asarray(x)
    out = {"name": name, "shape": list(x.shape),
           "nan_count": int(np.isnan(x).sum()),
           "inf_count": int(np.isinf(x).sum())}
    if not np.all(np.isnan(x)):
        out.update({"min": float(np.nanmin(x)), "max": float(np.nanmax(x)),
                    "mean": float(np.nanmean(x)), "std": float(np.nanstd(x))})
    return out

def assert_or_raise(cond, msg):
    if not cond: raise ValueError(msg)

def infer_steps_per_day(idx, default=24):
    if len(idx) < 2: return default
    dt = idx.to_series().diff().dropna().median()
    return max(1, int(round(pd.Timedelta(days=1) / dt))) if dt > pd.Timedelta(0) else default

def time_split_df(df_full, split_ratio):
    sp = int(len(df_full) * float(split_ratio))
    hist, fut = df_full.iloc[:sp].copy(), df_full.iloc[sp:].copy()
    meta = {"total_len": len(df_full), "split_ratio": float(split_ratio),
            "split_point": sp, "hist_len": len(hist), "fut_len": len(fut),
            "hist_start": str(hist.index.min()), "hist_end": str(hist.index.max()),
            "fut_start": str(fut.index.min()) if len(fut) else None,
            "fut_end":   str(fut.index.max()) if len(fut) else None}
    return hist, fut, meta

def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0: return {"rmse": np.nan, "mae": np.nan}
    diff = y_true[mask] - y_pred[mask]
    return {"rmse": float(np.sqrt(np.mean(diff**2))), "mae": float(np.mean(np.abs(diff)))}

def save_plots(out_dir, fut_index, y_true, y_pred, plot_dim):
    ensure_dir(out_dir)
    d = max(0, min(int(plot_dim), y_true.shape[1] - 1))
    plt.figure(figsize=(14, 5))
    plt.plot(fut_index, y_true[:, d], label=f"True (dim {d})", color="steelblue")
    plt.plot(fut_index, y_pred[:, d], label=f"GRU-ODE-Bayes (dim {d})", color="tomato")
    plt.xlabel("Time"); plt.ylabel("PM2.5")
    plt.title(f"GRU-ODE-Bayes Forecast vs True (dim {d})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"plot_forecast_dim{d}.png"), dpi=150)
    plt.close()

    rmse_list = [compute_metrics(y_true[:, j], y_pred[:, j])["rmse"]
                 for j in range(y_true.shape[1])]
    plt.figure(figsize=(14, 5))
    plt.bar(np.arange(len(rmse_list)), rmse_list)
    plt.xlabel("Dimension"); plt.ylabel("RMSE")
    plt.title("GRU-ODE-Bayes RMSE per Dimension"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_rmse_per_dim.png"), dpi=150)
    plt.close()


# =============================================================================
# GRU-ODE-Bayes 核心组件（完全复现官方仓库）
# =============================================================================

class GRUODECell(nn.Module):
    """
    GRU-ODE 连续时间单元（官方实现）
    forward(x, h) -> dh
    其中 x 是 p（预测分布参数，2*input_size 维）
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lin_xz = nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_xn = nn.Linear(input_size, hidden_size, bias=bias)
        self.lin_hn = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        z = torch.sigmoid(self.lin_xz(x) + self.lin_hz(h))
        n = torch.tanh(self.lin_xn(x) + self.lin_hn(z * h))
        dh = (1 - z) * (n - h)
        return dh


class FullGRUODECell(nn.Module):
    """
    完整GRU-ODE单元（官方实现，包含reset gate）
    forward(x, h) -> dh
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.lin_x = nn.Linear(input_size, hidden_size * 3, bias=bias)
        self.lin_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lin_hr = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, h):
        xr, xz, xh = torch.chunk(self.lin_x(x), 3, dim=1)
        r = torch.sigmoid(xr + self.lin_hr(h))
        z = torch.sigmoid(xz + self.lin_hz(h))
        u = torch.tanh(xh + self.lin_hh(r * h))
        dh = (1 - z) * (u - h)
        return dh


class GRUObservationCellLogvar(nn.Module):
    """
    GRU 贝叶斯观测更新单元（官方实现）
    forward(h, p, X_obs, M_obs, i_obs) -> h_new, losses
    """
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
        h = temp

        return h, losses


def compute_KL_loss(p_obs, X_obs, M_obs, obs_noise_std=1e-2):
    mean, logvar = torch.chunk(p_obs, 2, dim=1)
    std = torch.exp(0.5 * logvar)
    obs_noise_std_t = torch.tensor(obs_noise_std, device=mean.device, dtype=mean.dtype)
    return (gaussian_KL(mu_1=mean, mu_2=X_obs, sigma_1=std, sigma_2=obs_noise_std_t) * M_obs).sum()


def gaussian_KL(mu_1, mu_2, sigma_1, sigma_2):
    return (torch.log(sigma_2) - torch.log(sigma_1) + 
            (torch.pow(sigma_1, 2) + torch.pow((mu_1 - mu_2), 2)) / (2 * sigma_2**2) - 0.5)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.05)


# =============================================================================
# NNFOwithBayesianJumps（官方模型完整复现）
# =============================================================================

class NNFOwithBayesianJumps(nn.Module):
    """
    Neural Negative Feedback ODE with Bayesian Jumps
    完全复现官方仓库的 forward 逻辑
    """
    def __init__(self, input_size, hidden_size, p_hidden, prep_hidden,
                 bias=True, cov_size=1, cov_hidden=1, logvar=True, mixing=1,
                 full_gru_ode=False, solver="euler", impute=True):
        super().__init__()
        self.impute = impute
        self.full_gru_ode = full_gru_ode
        self.p_model = nn.Sequential(
            nn.Linear(hidden_size, p_hidden, bias=bias),
            nn.ReLU(),
            nn.Linear(p_hidden, 2 * input_size, bias=bias),
        )
        if full_gru_ode:
            self.gru_c = FullGRUODECell(2 * input_size, hidden_size, bias=bias)
        else:
            self.gru_c = GRUODECell(2 * input_size, hidden_size, bias=bias)
        self.gru_obs = GRUObservationCellLogvar(input_size, hidden_size, prep_hidden, bias=bias)
        self.covariates_map = nn.Sequential(
            nn.Linear(cov_size, cov_hidden, bias=bias),
            nn.ReLU(),
            nn.Linear(cov_hidden, hidden_size, bias=bias),
            nn.Tanh()
        )
        self.solver = solver
        self.input_size = input_size
        self.logvar = logvar
        self.mixing = mixing
        self.apply(init_weights)

    def ode_step(self, h, p, delta_t):
        if self.solver == "euler":
            h = h + delta_t * self.gru_c(p, h)
            p = self.p_model(h)
        elif self.solver == "midpoint":
            k = h + delta_t / 2 * self.gru_c(p, h)
            pk = self.p_model(k)
            h = h + delta_t * self.gru_c(pk, k)
            p = self.p_model(h)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
        return h, p

    def forward(self, times, time_ptr, X, M, obs_idx, delta_t, T, cov, pred_step=0.0):
        """
        Args:
            times:    观测时间点数组
            time_ptr: 每个时间点的数据起始索引
            X:        数据张量
            M:        掩码张量 (1=观测, 0=缺失)
            obs_idx:  每条观测属于哪个样本
            delta_t:  ODE 积分步长
            T:        总时间长度
            cov:      协变量
            pred_step: 预测步长（在T之后继续演化的时间）

        Returns:
            h:    最终隐状态
            loss: 总损失
        """
        h = self.covariates_map(cov)
        p = self.p_model(h)
        current_time = 0.0

        loss_1 = 0
        loss_2 = 0

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
            loss_2 = loss_2 + compute_KL_loss(p_obs=p[i_obs], X_obs=X_obs, M_obs=M_obs)

        while current_time < T:
            h, p = self.ode_step(h, p, delta_t)
            current_time += delta_t

        if pred_step > 0:
            T_pred = T + pred_step
            while current_time < T_pred:
                h, p = self.ode_step(h, p, delta_t)
                current_time += delta_t

        loss = loss_1 + self.mixing * loss_2
        return h, loss, p, loss_1


# =============================================================================
# 预测头：从 h 映射到预测值
# =============================================================================

class ForecastHead(nn.Module):
    def __init__(self, hidden_size, p_hidden, output_size):
        super().__init__()
        self.delta_net = nn.Sequential(
            nn.Linear(hidden_size, p_hidden),
            nn.ReLU(),
            nn.Linear(p_hidden, output_size),
        )

    def forward(self, h, x_last):
        delta = self.delta_net(h)
        return x_last + delta


# =============================================================================
# 数据格式转换：滑动窗口 -> GRU-ODE-Bayes 稀疏时间格式
# =============================================================================

def windows_to_gruode_format(windows, device, time_scale=1.0):
    """
    将规则采样的滑动窗口 (B, W, D) 转换为 NNFOwithBayesianJumps 的输入格式。
    
    time_scale: 时间缩放因子。
                如果 W=48, time_scale=1.0，则时间点为 0,1,2,...,47
                如果 time_scale=1.0/(W-1)，则时间点归一化到 [0,1]
                较小的 time_scale 会让 ODE 演化更快，模型更"激进"
    """
    B, W, D = windows.shape
    times = torch.arange(W, dtype=torch.float32, device=device) * time_scale
    time_ptr = torch.arange(0, W * B + 1, B, dtype=torch.long, device=device)
    X_flat = torch.tensor(windows, dtype=torch.float32, device=device)
    X_flat = X_flat.permute(1, 0, 2).reshape(W * B, D)
    M_flat = torch.ones_like(X_flat)
    obs_idx = torch.arange(B, dtype=torch.long, device=device).repeat(W)
    cov = torch.zeros(B, 1, dtype=torch.float32, device=device)
    T_total = float(W - 1) * time_scale
    return times, time_ptr, X_flat, M_flat, obs_idx, cov, T_total


# =============================================================================
# 数据准备
# =============================================================================

def build_windows(data, window_size, steps_ahead=1):
    T, D = data.shape
    xs, ys = [], []
    for i in range(T - window_size - steps_ahead + 1):
        xs.append(data[i: i + window_size])
        ys.append(data[i + window_size + steps_ahead - 1])
    return np.stack(xs), np.stack(ys)


# =============================================================================
# 训练（带增量损失）
# =============================================================================

def train_model(gruode_model, head, X_train, y_train,
                epochs, batch_size, lr, device, delta_t,
                patience=15, verbose=True, time_scale=1.0, pred_step=1.0,
                lambda_increment=0.5):
    """
    lambda_increment: 增量损失权重。损失函数为：
        L = MSE(pred, y) + lambda_increment * MSE(pred - x_last, y - x_last) + 0.01 * nll_loss
    这样可以防止模型学习"复制上一帧"。
    """
    params = list(gruode_model.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=max(1, patience // 2), factor=0.5, verbose=False)

    N, W, D = X_train.shape
    dataset = TensorDataset(torch.arange(N))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    X_np = X_train
    y_t = torch.tensor(y_train, dtype=torch.float32)

    best_loss, best_enc_state, best_head_state, no_imp = float("inf"), None, None, 0
    loss_history = []

    for epoch in tqdm(range(epochs), desc="  训练", disable=not verbose):
        gruode_model.train()
        head.train()
        ep_loss, ep_n = 0.0, 0
        ep_mse, ep_inc = 0.0, 0.0

        for (idx,) in loader:
            idx_list = idx.tolist()
            X_batch = X_np[idx_list]
            y_batch = y_t[idx].to(device)
            x_last = torch.tensor(X_batch[:, -1, :], dtype=torch.float32, device=device)
            B = len(idx_list)

            times, time_ptr, X_flat, M_flat, obs_idx, cov, T_total = \
                windows_to_gruode_format(X_batch, device, time_scale)

            optimizer.zero_grad()
            try:
                h, nll_loss, p_out, loss_1 = gruode_model(
                    times, time_ptr, X_flat, M_flat, obs_idx,
                    delta_t=delta_t, T=T_total, cov=cov, pred_step=pred_step * time_scale
                )
                pred = head(h, x_last)
                
                mse_loss = nn.functional.mse_loss(pred, y_batch)
                
                delta_pred = pred - x_last
                delta_true = y_batch - x_last
                inc_loss = nn.functional.mse_loss(delta_pred, delta_true)
                
                loss = mse_loss + lambda_increment * inc_loss + 0.01 * nll_loss

                loss.backward()
                nn.utils.clip_grad_norm_(params, max_norm=5.0)
                optimizer.step()
                ep_loss += mse_loss.item() * B
                ep_mse += mse_loss.item() * B
                ep_inc += inc_loss.item() * B
                ep_n += B
            except Exception as ex:
                tqdm.write(f"  [跳过 batch] {ex}")
                continue

        if ep_n > 0:
            ep_loss /= ep_n
            ep_mse /= ep_n
            ep_inc /= ep_n
        loss_history.append(ep_loss)
        scheduler.step(ep_loss)

        if verbose and (epoch + 1) % 5 == 0:
            tqdm.write(f"  epoch {epoch+1}: MSE={ep_mse:.6f}, Inc={ep_inc:.6f}")

        if ep_loss < best_loss:
            best_loss = ep_loss
            no_imp = 0
            best_enc_state = {k: v.cpu().clone() for k, v in gruode_model.state_dict().items()}
            best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
        else:
            no_imp += 1

        if no_imp >= patience:
            if verbose:
                tqdm.write(f"  Early stopping @ epoch {epoch+1}，best MSE={best_loss:.6f}")
            break

    if best_enc_state:
        gruode_model.load_state_dict(best_enc_state)
        head.load_state_dict(best_head_state)
    gruode_model.to(device)
    head.to(device)
    return loss_history, best_loss


# =============================================================================
# 预测（自回归预测：用预测值替换真值）
# =============================================================================

def forecast_autoregressive(gruode_model, head, history_scaled, horizon, 
                            window_size, delta_t, device, scaler,
                            batch_size=64, verbose=True, time_scale=1.0, pred_step=1.0):
    """
    自回归预测：每次预测后，将预测值加入窗口，用于下一次预测。
    使用 head(h, x_last) 作为预测输出。
    """
    gruode_model.eval()
    head.eval()
    
    W = window_size
    current_window = history_scaled[-W:].copy()
    preds_scaled = []
    nan_count = 0
    
    for i in tqdm(range(horizon), desc="  自回归预测", disable=not verbose):
        X_batch = current_window[np.newaxis, :, :]
        x_last = torch.tensor(X_batch[:, -1, :], dtype=torch.float32, device=device)
        
        times, time_ptr, X_flat, M_flat, obs_idx, cov, T_total = \
            windows_to_gruode_format(X_batch, device, time_scale)
        
        with torch.no_grad():
            h, _, p, _ = gruode_model(
                times, time_ptr, X_flat, M_flat, obs_idx,
                delta_t=delta_t, T=T_total, cov=cov, pred_step=pred_step * time_scale
            )
            pred = head(h, x_last).cpu().numpy()[0]
        
        if np.isnan(pred).any():
            nan_count += 1
            pred[np.isnan(pred)] = current_window[-1, np.isnan(pred)]
        
        preds_scaled.append(pred)
        current_window = np.vstack([current_window[1:], pred[np.newaxis, :]])
    
    if nan_count > 0:
        print(f"  [警告] 推理过程中出现 {nan_count} 次 NaN")
    
    preds_scaled = np.array(preds_scaled)
    preds = scaler.inverse_transform(preds_scaled.astype(np.float64))
    stds = np.zeros_like(preds)
    return preds, stds


def forecast(gruode_model, head, history_scaled, fut_true_scaled,
             horizon, window_size, delta_t, device, scaler,
             batch_size=64, verbose=True, time_scale=1.0, use_autoregressive=True, pred_step=1.0):
    """
    预测函数。
    use_autoregressive=True: 自回归预测（推荐，不使用真值）
    use_autoregressive=False: 真值滑窗预测（用于调试）
    使用 head(h, x_last) 作为预测输出。
    """
    if use_autoregressive:
        return forecast_autoregressive(
            gruode_model, head, history_scaled, horizon,
            window_size, delta_t, device, scaler,
            batch_size, verbose, time_scale, pred_step
        )
    
    T_hist = history_scaled.shape[0]
    W = window_size
    full = np.concatenate([history_scaled, fut_true_scaled], axis=0).astype(np.float32)

    windows = np.stack([full[T_hist - W + i: T_hist + i] for i in range(horizon)])

    gruode_model.eval()
    head.eval()
    preds_list = []
    n_batches = (horizon + batch_size - 1) // batch_size

    for b in tqdm(range(n_batches), desc="  批量推理", disable=not verbose):
        sl = slice(b * batch_size, (b + 1) * batch_size)
        X_batch = windows[sl]
        x_last = torch.tensor(X_batch[:, -1, :], dtype=torch.float32, device=device)

        times, time_ptr, X_flat, M_flat, obs_idx, cov, T_total = \
            windows_to_gruode_format(X_batch, device, time_scale)

        with torch.no_grad():
            h, _, p, _ = gruode_model(
                times, time_ptr, X_flat, M_flat, obs_idx,
                delta_t=delta_t, T=T_total, cov=cov, pred_step=pred_step * time_scale
            )
            pred = head(h, x_last).cpu().numpy()
        preds_list.append(pred)

    preds_s = np.concatenate(preds_list, axis=0)

    last_vals = windows[:, -1, :]
    nan_count = 0
    for i in range(preds_s.shape[0]):
        nan_cols = np.isnan(preds_s[i])
        if nan_cols.any():
            nan_count += 1
            preds_s[i, nan_cols] = last_vals[i, nan_cols]
    
    if nan_count > 0:
        print(f"  [警告] 批量推理中出现 {nan_count} 次 NaN")
    
    preds = scaler.inverse_transform(preds_s.astype(np.float64))
    stds = np.zeros_like(preds)
    return preds, stds


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PM2.5: GRU-ODE-Bayes baseline (Official Reproduction)")

    parser.add_argument("--imputed_history_path", type=str, required=True)
    parser.add_argument("--ground_path", type=str, required=True)
    parser.add_argument("--split_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--horizon_days", type=float, default=0.0)
    parser.add_argument("--horizon_steps", type=int, default=0)

    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--p_hidden", type=int, default=32)
    parser.add_argument("--prep_hidden", type=int, default=32)
    parser.add_argument("--solver", type=str, default="euler", choices=["euler", "midpoint"])
    parser.add_argument("--delta_t", type=float, default=0.1,
                        help="ODE 积分步长。官方默认 0.1")
    parser.add_argument("--time_scale", type=float, default=0.02,
                        help="时间缩放因子。默认 0.02，即归一化到 [0,1]")
    parser.add_argument("--full_gru_ode", action="store_true",
                        help="使用完整GRU-ODE（官方推荐）")
    parser.add_argument("--impute", action="store_true",
                        help="启用插值模式")
    parser.add_argument("--mixing", type=float, default=1e-4,
                        help="KL loss权重。官方默认 1e-4")

    parser.add_argument("--window_size", type=int, default=48)
    parser.add_argument("--steps_ahead", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--plot_dim", type=int, default=0)
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--no_verbose", action="store_true")
    parser.add_argument("--use_true_window", action="store_true",
                        help="使用真值滑窗预测（默认False，使用自回归预测）")
    parser.add_argument("--lambda_increment", type=float, default=0.5,
                        help="增量损失权重，防止模型学习'复制上一帧'。默认0.5")

    args = parser.parse_args()
    verbose = not args.no_verbose
    set_global_seed(args.seed)

    print("[GRU-ODE-Bayes] 官方模型完整复现版")
    print(f"[参数] time_scale={args.time_scale}, delta_t={args.delta_t}, full_gru_ode={args.full_gru_ode}, impute={args.impute}, mixing={args.mixing}")

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or \
        f"./save/pm25_gruodebayes_split{args.split_ratio}_seed{args.seed}_{now}/"
    ensure_dir(out_dir)
    safe_json_dump(vars(args), os.path.join(out_dir, "args.json"))

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    print(f"[设备] {device}")

    # ================================================================
    # 1. 读数据
    # ================================================================
    df_full = pd.read_csv(args.ground_path, index_col="datetime",
                          parse_dates=True).sort_index()
    hist_full, fut_full, meta = time_split_df(df_full, args.split_ratio)
    df_hist = pd.read_csv(args.imputed_history_path, index_col="datetime",
                          parse_dates=True).sort_index()

    assert_or_raise(df_hist.index.equals(hist_full.index),
        "imputed_history datetime 与 ground 前半段不一致，请检查 split_ratio。")
    assert_or_raise(list(df_hist.columns) == list(hist_full.columns),
        "imputed_history 列与 ground 列不一致。")

    history = df_hist.values.astype(np.float64)
    D = history.shape[1]
    columns = list(df_full.columns)

    report = {"meta": meta, "args": vars(args), "checks": {},
              "history_stats": basic_array_stats(history, "history_imputed")}
    assert_or_raise(not np.isnan(history).any(), "history_imputed 含 NaN。")
    assert_or_raise(not np.isinf(history).any(), "history_imputed 含 Inf。")

    # ================================================================
    # 2. Horizon
    # ================================================================
    full_horizon = meta["fut_len"]
    horizon = full_horizon
    if args.horizon_days > 0:
        spd = infer_steps_per_day(fut_full.index)
        horizon = int(round(args.horizon_days * spd))
    elif args.horizon_steps > 0:
        horizon = int(args.horizon_steps)
    horizon = max(1, min(horizon, full_horizon))
    fut_full = fut_full.iloc[:horizon].copy()
    report["checks"]["final_horizon_steps"] = int(horizon)

    # ================================================================
    # 3. 标准化
    # ================================================================
    scaler = StandardScaler()
    history_scaled = scaler.fit_transform(history).astype(np.float32)
    fut_true = fut_full.values.astype(np.float64)
    fut_true_scaled = scaler.transform(fut_true).astype(np.float32)

    # ================================================================
    # 4. 滑动窗口
    # ================================================================
    W = int(args.window_size)
    sa = int(args.steps_ahead)
    assert_or_raise(W >= 2, "window_size 必须 >= 2")
    assert_or_raise(history_scaled.shape[0] > W + sa,
        f"history 太短，请减小 window_size({W})。")

    X_win, y_win = build_windows(history_scaled, W, sa)

    report["checks"].update({
        "num_train_samples": int(X_win.shape[0]),
        "D": int(D), "window_size": W,
    })

    print(f"\n{'='*70}")
    print("PM2.5 预测 —— GRU-ODE-Bayes（官方复现版）")
    print(f"{'='*70}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nhorizon={horizon}步  训练样本={X_win.shape[0]}")
    print(f"D={D}  hidden={args.hidden_size}  solver={args.solver}")
    print(f"time_scale={args.time_scale}  delta_t={args.delta_t}")

    # ================================================================
    # 5. 构建模型
    # ================================================================
    gruode_model = NNFOwithBayesianJumps(
        input_size=D,
        hidden_size=args.hidden_size,
        p_hidden=args.p_hidden,
        prep_hidden=args.prep_hidden,
        logvar=True,
        mixing=args.mixing,
        full_gru_ode=args.full_gru_ode,
        impute=args.impute,
        solver=args.solver,
        cov_size=1,
    ).to(device)

    head = ForecastHead(args.hidden_size, args.p_hidden, D).to(device)

    n_params = (sum(p.numel() for p in gruode_model.parameters() if p.requires_grad)
                + sum(p.numel() for p in head.parameters() if p.requires_grad))
    print(f"模型参数量：{n_params:,}")

    # ================================================================
    # 6. 训练
    # ================================================================
    print("\n[训练阶段]")
    t0 = time.time()
    pred_step = max(float(args.steps_ahead), float(args.window_size) * 0.2)
    print(f"  [info] pred_step={pred_step:.2f} (window_size*0.2={float(args.window_size)*0.2:.2f})")
    loss_hist, best_loss = train_model(
        gruode_model, head, X_win, y_win,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=device, delta_t=args.delta_t,
        patience=args.patience, verbose=verbose, time_scale=args.time_scale,
        pred_step=pred_step, lambda_increment=args.lambda_increment)
    train_time = time.time() - t0
    print(f"  耗时 {train_time:.1f}s，最优 MSE（标准化）={best_loss:.6f}")

    torch.save({"gruode": gruode_model.state_dict(), "head": head.state_dict()},
               os.path.join(out_dir, "model.pt"))
    report["training"] = {
        "train_time_sec": round(train_time, 2),
        "best_mse_normalized": float(best_loss),
        "epochs_run": int(len(loss_hist)),
        "n_params": int(n_params),
    }
    safe_json_dump(loss_hist, os.path.join(out_dir, "loss_history.json"))

    # ================================================================
    # 7. 预测
    # ================================================================
    use_ar = not args.use_true_window
    pred_mode = "自回归预测" if use_ar else "真值滑窗预测"
    print(f"\n[预测阶段]（{pred_mode}）")
    t1 = time.time()
    preds, stds = forecast(
        gruode_model, head, history_scaled, fut_true_scaled,
        horizon=horizon, window_size=W, delta_t=args.delta_t,
        device=device, scaler=scaler,
        batch_size=args.batch_size, verbose=verbose, time_scale=args.time_scale,
        use_autoregressive=use_ar, pred_step=pred_step)
    print(f"  耗时 {time.time()-t1:.1f}s")

    n_nan = int(np.isnan(preds).sum())
    report["pred_stats"] = basic_array_stats(preds, "preds")
    report["checks"]["pred_nan_count"] = n_nan
    if n_nan > 0:
        print(f"  [警告] 仍有 {n_nan} 个 NaN（已持久性回填）")

    # ================================================================
    # 8. 保存
    # ================================================================
    df_pred = pd.DataFrame(preds, index=fut_full.index, columns=columns)
    df_std = pd.DataFrame(stds, index=fut_full.index, columns=columns)
    pred_csv = os.path.join(out_dir, "future_pred.csv")
    std_csv = os.path.join(out_dir, "future_pred_std.csv")
    df_pred.to_csv(pred_csv)
    df_std.to_csv(std_csv)
    np.save(os.path.join(out_dir, "future_pred.npy"), preds)
    np.save(os.path.join(out_dir, "future_pred_std.npy"), stds)
    safe_json_dump(report, os.path.join(out_dir, "debug_report.json"))
    print(f"\n预测保存：\n  {pred_csv}\n  {std_csv}")

    # ================================================================
    # 9. 评估
    # ================================================================
    if not args.skip_metrics:
        overall = compute_metrics(fut_true, preds)
        safe_json_dump({"overall": overall, "horizon": horizon},
                       os.path.join(out_dir, "metrics.json"))
        per_dim = [{"dim": j, "name": str(col),
                    **compute_metrics(fut_true[:, j], preds[:, j])}
                   for j, col in enumerate(columns)]
        pd.DataFrame(per_dim).to_csv(
            os.path.join(out_dir, "metrics_per_dim.csv"), index=False)
        save_plots(out_dir, fut_full.index, fut_true, preds, args.plot_dim)
        print("\n整体评估（GRU-ODE-Bayes）：")
        print(json.dumps(overall, indent=4, ensure_ascii=False))
    else:
        print("\n已跳过 metrics（--skip_metrics）")

    print(f"\n输出目录：{out_dir}")


if __name__ == "__main__":
    main()
