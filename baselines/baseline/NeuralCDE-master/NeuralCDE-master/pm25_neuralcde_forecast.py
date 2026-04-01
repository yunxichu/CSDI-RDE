# -*- coding: utf-8 -*-
"""
PM2.5 后续预测脚本 —— NeuralCDE 基线版（严格按照官方 torchcde API）
============================================================
参考：https://github.com/patrick-kidger/torchcde

数据接口与 RDE-GPR 脚本完全对齐：
- 输入：history_imputed.csv + pm25_ground.txt
- 输出：future_pred.csv / future_pred_std.csv / metrics.json / plot_*.png

安装：
  pip install torch torchdiffeq torchcde

示例：
python pm25_neuralcde_forecast.py \
    --imputed_history_path /home/rhl/CSDI-main_test/save/pm25_history_imputed_split0.5_seed42_20260128_101132/history_imputed.csv \
    --ground_path /home/rhl/CSDI-main_test/data/pm25/Code/STMVL/SampleData/pm25_ground.txt \
    --split_ratio 0.5 --horizon_days 1 \
    --window_size 48 --hidden_channels 64 --num_layers 3 \
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
    d = max(0, min(int(plot_dim), y_true.shape[1]-1))
    plt.figure(figsize=(14, 5))
    plt.plot(fut_index, y_true[:, d], label=f"True (dim {d})", color="steelblue")
    plt.plot(fut_index, y_pred[:, d], label=f"NeuralCDE (dim {d})", color="tomato")
    plt.xlabel("Time"); plt.ylabel("PM2.5")
    plt.title(f"NeuralCDE Forecast vs True (dim {d})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"plot_forecast_dim{d}.png"), dpi=150)
    plt.close()

    rmse_list = [compute_metrics(y_true[:, j], y_pred[:, j])["rmse"]
                 for j in range(y_true.shape[1])]
    plt.figure(figsize=(14, 5))
    plt.bar(np.arange(len(rmse_list)), rmse_list)
    plt.xlabel("Dimension"); plt.ylabel("RMSE")
    plt.title("NeuralCDE RMSE per Dimension"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "plot_rmse_per_dim.png"), dpi=150)
    plt.close()


# =============================================================================
# NeuralCDE 模型（严格按官方 API）
# =============================================================================

class CDEFunc(nn.Module):
    """
    向量场 f_theta(t, z) -> (batch, hidden, input_channels)
    严格遵循官方：func(t, z) 返回形状 (..., hidden_channels, input_channels)
    """
    def __init__(self, input_channels, hidden_channels, num_layers=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.input_channels  = input_channels

        # 构建 MLP，最后线性层输出 hidden * input
        layers = []
        in_dim = hidden_channels
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_channels), nn.Softplus()]
            in_dim = hidden_channels
        layers += [nn.Linear(hidden_channels, hidden_channels * input_channels), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, t, z):
        # z: (..., hidden_channels)
        # 返回: (..., hidden_channels, input_channels)
        return self.net(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)


class NeuralCDE(nn.Module):
    """
    严格按照官方 torchcde 示例：
    https://github.com/patrick-kidger/torchcde

    关键点：
    - coeffs 在外部预计算，forward 里调用 CubicSpline(coeffs)
    - z0 由第一个观测点经线性层得到：z0 = initial(X.evaluate(X.interval[0]))
    - cdeint 使用 t=X.interval（即 [t_start, t_end]，两点）
    - rk4 + step_size 保证精度
    """
    def __init__(self, input_channels, hidden_channels, output_channels,
                 num_layers=3, step_size=None):
        super().__init__()
        self.input_channels  = input_channels
        self.hidden_channels = hidden_channels
        self.step_size       = step_size   # rk4 步长，None=自适应 dopri5

        self.initial = nn.Linear(input_channels, hidden_channels)
        self.func    = CDEFunc(input_channels, hidden_channels, num_layers)
        self.readout = nn.Linear(hidden_channels, output_channels)

    def forward(self, coeffs):
        import torchcde

        # 官方用法：在 forward 里构造 CubicSpline
        X = torchcde.CubicSpline(coeffs)

        # 初始隐状态：从第一个观测点线性映射
        # X.evaluate(X.interval[0]) -> (batch, input_channels)
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)           # (batch, hidden_channels)

        # ODE 积分，t=X.interval 即 [t_start, t_end]
        # rk4 + step_size 保证积分精度；adjoint=False 训练更快
        if self.step_size is not None:
            z_T = torchcde.cdeint(X=X, func=self.func, z0=z0,
                                  t=X.interval,
                                  adjoint=False,
                                  method='rk4',
                                  options=dict(step_size=self.step_size))
        else:
            # 默认自适应 dopri5（精度最高，速度较慢）
            z_T = torchcde.cdeint(X=X, func=self.func, z0=z0,
                                  t=X.interval,
                                  adjoint=False)

        # z_T: (batch, 2, hidden_channels) -> 取最后时刻
        z_final = z_T[:, -1]           # (batch, hidden_channels)
        return self.readout(z_final)   # (batch, output_channels)


# =============================================================================
# 数据准备
# =============================================================================

def build_windows(data: np.ndarray, window_size: int, steps_ahead: int = 1):
    """
    滑动窗口切片。
    data: (T, D)
    返回 X: (N, W, D),  y: (N, D)
    """
    T, D = data.shape
    xs, ys = [], []
    for i in range(T - window_size - steps_ahead + 1):
        xs.append(data[i: i + window_size])
        ys.append(data[i + window_size + steps_ahead - 1])
    return np.stack(xs), np.stack(ys)


def make_cde_input(X_win: np.ndarray) -> np.ndarray:
    """
    按照官方示例，把时间作为第一个通道拼入数据：
      t_ shape (N, W, 1)，值域 [0,1]
      x_ shape (N, W, D)
      x  shape (N, W, D+1)   <- 这就是送入 hermite_cubic 的格式

    官方代码：
      t_ = t.unsqueeze(0).unsqueeze(-1).expand(batch, length, 1)
      x_ = torch.rand(batch, length, input_channels - 1)
      x  = torch.cat([t_, x_], dim=2)
    """
    N, W, D = X_win.shape
    t_ = np.linspace(0., 1., W, dtype=np.float32)[None, :, None]   # (1, W, 1)
    t_ = np.broadcast_to(t_, (N, W, 1)).copy()
    return np.concatenate([t_, X_win.astype(np.float32)], axis=-1)  # (N, W, D+1)


# =============================================================================
# 训练
# =============================================================================

def train_model(model, X_train_np, y_train_np, epochs, batch_size, lr,
                device, patience=15, verbose=True):
    """
    X_train_np: (N, W, D+1) — 已含时间通道，numpy
    y_train_np: (N, D)       — 目标，numpy

    训练流程：
    1. 一次性预计算所有 coeffs（hermite cubic）
    2. 按 batch index 取对应 coeffs 和 y 送入模型
    """
    import torchcde

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=max(1, patience // 2), factor=0.5, verbose=False)

    # --- 预计算样条系数（CPU，省显存）---
    print("  预计算 Hermite 三次样条系数…", flush=True)
    X_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    with torch.no_grad():
        # 官方：coeffs = hermite_cubic_coefficients_with_backward_differences(x)
        # x 形状: (batch, length, channels)，时间已在第0通道
        coeffs_cpu = torchcde.hermite_cubic_coefficients_with_backward_differences(X_tensor)
    # coeffs_cpu: (N, W-1, (D+1)*4) Tensor，可直接索引

    y_tensor = torch.tensor(y_train_np, dtype=torch.float32)
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
            # 取这个 batch 的 coeffs 和 y，送到 GPU
            b_coeffs = coeffs_cpu[idx].to(device)
            b_y      = y_tensor[idx].to(device)

            pred = model(b_coeffs)          # (B, D)
            loss = nn.functional.mse_loss(pred, b_y)
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
            if verbose:
                tqdm.write(f"  Early stopping @ epoch {epoch+1}，best MSE={best_loss:.6f}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device)
    return loss_history, best_loss


# =============================================================================
# 预测（真值滑窗 + 批量推理）
# =============================================================================

def forecast(model, history_scaled, fut_true_scaled, horizon, window_size,
             device, scaler, batch_size=256, verbose=True):
    """
    预测策略：窗口用真实值滑动，每步互相独立，全部批量推理。

    预测 fut[i] 时：
      window = concat(history, fut_true)[T_hist - W + i : T_hist + i]
    即：窗口的前 (W-i) 步来自历史，后 i 步来自未来真实值。

    全部 horizon 个窗口一次性构建，按 batch_size 批量送 GPU。
    """
    import torchcde

    T_hist = history_scaled.shape[0]
    D      = history_scaled.shape[1]

    # 拼接 [历史 | 未来真值]
    full = np.concatenate([history_scaled, fut_true_scaled], axis=0).astype(np.float32)

    # 构建所有窗口 (horizon, W, D)
    W = window_size
    windows = np.stack([full[T_hist - W + i : T_hist + i] for i in range(horizon)])

    # 添加时间通道 -> (horizon, W, D+1)
    X_all_np = make_cde_input(windows)
    X_all = torch.tensor(X_all_np, dtype=torch.float32)

    # 预计算系数
    with torch.no_grad():
        coeffs_all = torchcde.hermite_cubic_coefficients_with_backward_differences(X_all)

    model.eval()
    preds_list = []
    n_batches = (horizon + batch_size - 1) // batch_size
    for b in tqdm(range(n_batches), desc="  批量推理", disable=not verbose):
        sl = slice(b * batch_size, (b+1) * batch_size)
        b_coeffs = coeffs_all[sl].to(device)
        with torch.no_grad():
            p = model(b_coeffs).cpu().numpy()   # (B, D)
        preds_list.append(p)

    preds_s = np.concatenate(preds_list, axis=0)   # (horizon, D)
    preds   = scaler.inverse_transform(preds_s.astype(np.float64))
    stds    = np.zeros_like(preds)                  # 不做 MC Dropout，保持简洁
    return preds, stds


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    # 数据（与 RDE-GPR 完全对齐）
    parser.add_argument("--imputed_history_path", type=str, required=True)
    parser.add_argument("--ground_path",          type=str, required=True)
    parser.add_argument("--split_ratio",  type=float, default=0.5)
    parser.add_argument("--seed",         type=int,   default=42)

    # Horizon
    parser.add_argument("--horizon_days",  type=float, default=0.0)
    parser.add_argument("--horizon_steps", type=int,   default=0)

    # 模型
    parser.add_argument("--window_size",      type=int,   default=48)
    parser.add_argument("--hidden_channels",  type=int,   default=64)
    parser.add_argument("--num_layers",       type=int,   default=3)
    parser.add_argument("--steps_ahead",      type=int,   default=1)
    # rk4 步长：None=自适应 dopri5；推荐值=1/(window_size-1)（精确）
    # 也可以传更大的步长加速，如 1/12（4倍加速，精度略降）
    parser.add_argument("--use_fixed_solver", action="store_true",
                        help="使用 rk4 固定步长（更快）；默认用自适应 dopri5（更精确）")
    parser.add_argument("--rk4_nsteps", type=int, default=0,
                        help="rk4 积分步数（use_fixed_solver 时有效）；0=等于 window_size-1")

    # 训练
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int,   default=15)

    # 其他
    parser.add_argument("--device",       type=str, default="auto")
    parser.add_argument("--out_dir",      type=str, default="")
    parser.add_argument("--plot_dim",     type=int, default=0)
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--no_verbose",   action="store_true")
    args = parser.parse_args()

    verbose = not args.no_verbose
    set_global_seed(args.seed)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or \
        f"./save/pm25_neuralcde_split{args.split_ratio}_seed{args.seed}_{now}/"
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
    D       = history.shape[1]
    columns = list(df_full.columns)

    report = {"meta": meta, "args": vars(args), "checks": {},
              "history_stats": basic_array_stats(history, "history_imputed")}
    assert_or_raise(not np.isnan(history).any(), "history_imputed 含 NaN。")
    assert_or_raise(not np.isinf(history).any(), "history_imputed 含 Inf。")

    # ================================================================
    # 2. Horizon
    # ================================================================
    full_horizon = meta["fut_len"]
    assert_or_raise(full_horizon > 0, "后半段长度为 0，请检查 split_ratio。")
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
    history_scaled   = scaler.fit_transform(history).astype(np.float32)
    fut_true         = fut_full.values.astype(np.float64)
    fut_true_scaled  = scaler.transform(fut_true).astype(np.float32)

    # ================================================================
    # 4. 构建训练集
    # ================================================================
    W  = int(args.window_size)
    sa = int(args.steps_ahead)
    assert_or_raise(W >= 2, "window_size 必须 >= 2")
    assert_or_raise(history_scaled.shape[0] > W + sa,
        f"history 太短({history_scaled.shape[0]})，请减小 window_size({W})。")

    X_win, y_win = build_windows(history_scaled, W, sa)  # (N, W, D), (N, D)
    X_win_t = make_cde_input(X_win)                       # (N, W, D+1)，含时间通道

    input_channels  = D + 1   # 时间 + D 维观测
    output_channels = D

    report["checks"].update({
        "num_train_samples": int(X_win_t.shape[0]),
        "input_channels": int(input_channels),
        "window_size": W,
    })

    print(f"\n{'='*70}\nPM2.5 预测 —— NeuralCDE（官方 API）\n{'='*70}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nhorizon={horizon}步  训练样本={X_win_t.shape[0]}")
    print(f"input_ch={input_channels}  hidden_ch={args.hidden_channels}  layers={args.num_layers}")

    # ================================================================
    # 5. 模型
    # ================================================================
    # rk4 步长：t 归一化在 [0,1]，W 个点 → 相邻间距 = 1/(W-1)
    if args.use_fixed_solver:
        nsteps    = args.rk4_nsteps if args.rk4_nsteps > 0 else (W - 1)
        step_size = 1.0 / nsteps
        print(f"求解器：rk4，积分步数={nsteps}，step_size={step_size:.4f}")
    else:
        step_size = None
        print("求解器：dopri5（自适应，精度最高）")

    model = NeuralCDE(input_channels, args.hidden_channels, output_channels,
                      num_layers=args.num_layers, step_size=step_size)

    # ================================================================
    # 6. 训练
    # ================================================================
    print("\n[训练阶段]")
    t0 = time.time()
    loss_hist, best_loss = train_model(
        model, X_win_t, y_win,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, device=device, patience=args.patience, verbose=verbose)
    train_time = time.time() - t0
    print(f"  耗时 {train_time:.1f}s，最优归一化 MSE={best_loss:.6f}")

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    report["training"] = {"train_time_sec": round(train_time, 2),
                          "best_loss_normalized_mse": float(best_loss),
                          "epochs_run": int(len(loss_hist))}
    safe_json_dump(loss_hist, os.path.join(out_dir, "loss_history.json"))

    # ================================================================
    # 7. 预测
    # ================================================================
    print("\n[预测阶段]（真值滑窗 + 批量推理）")
    t1 = time.time()
    preds, stds = forecast(
        model, history_scaled, fut_true_scaled,
        horizon=horizon, window_size=W,
        device=device, scaler=scaler,
        batch_size=args.batch_size, verbose=verbose)
    print(f"  耗时 {time.time()-t1:.1f}s")

    report["pred_stats"] = basic_array_stats(preds, "preds")
    report["checks"]["pred_has_nan"]  = bool(np.isnan(preds).any())
    report["checks"]["pred_all_zero"] = bool(np.allclose(preds, 0.0))

    # ================================================================
    # 8. 保存（与 RDE-GPR 格式完全对齐）
    # ================================================================
    df_pred = pd.DataFrame(preds, index=fut_full.index, columns=columns)
    df_std  = pd.DataFrame(stds,  index=fut_full.index, columns=columns)
    pred_csv = os.path.join(out_dir, "future_pred.csv")
    std_csv  = os.path.join(out_dir, "future_pred_std.csv")
    df_pred.to_csv(pred_csv); df_std.to_csv(std_csv)
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
        print("\n整体评估（NeuralCDE）：")
        print(json.dumps(overall, indent=4, ensure_ascii=False))
    else:
        print("\n已跳过 metrics（--skip_metrics）")

    if report["checks"]["pred_has_nan"]:
        print("\n[诊断] preds 含 NaN：请检查 history 是否有常数列。")
    print(f"\n输出目录：{out_dir}")


if __name__ == "__main__":
    main()
