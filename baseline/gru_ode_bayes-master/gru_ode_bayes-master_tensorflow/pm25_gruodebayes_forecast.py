# -*- coding: utf-8 -*-
"""
PM2.5 后续预测脚本 —— GRU-ODE-Bayes 基线版
============================================================
直接使用本地仓库 gru_ode_bayes 的 NNFOwithBayesianJumps 模型。
tensorflow 仅用于 logging，通过 mock 跳过，不影响模型本身。

数据接口与 RDE-GPR 脚本完全对齐：
- 输入：history_imputed.csv + pm25_ground.txt
- 输出：future_pred.csv / future_pred_std.csv / metrics.json / plot_*.png

运行示例：
python pm25_gruodebayes_forecast.py \
    --imputed_history_path /home/rhl/CSDI-main_test/save/pm25_history_imputed_split0.5_seed42_20260128_101132/history_imputed.csv \
    --ground_path /home/rhl/CSDI-main_test/data/pm25/Code/STMVL/SampleData/pm25_ground.txt \
    --gruodebayes_repo_path /home/rhl/baseline/gru_ode_bayes-master/gru_ode_bayes-master_tensorflow \
    --split_ratio 0.5 --horizon_days 1 \
    --hidden_size 64 --p_hidden 32 --prep_hidden 32 \
    --window_size 48 --delta_t 1.0 \
    --load_model /home/rhl/baseline/gru_ode_bayes-master/gru_ode_bayes-master_tensorflow/save/pm25_gruodebayes_split0.5_seed42_20260228_160721/model.pt \
    --seed 42
"""


import os, sys, json, time, random, argparse, datetime, warnings, types
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn

# torch._dynamo.trace_rules 是懒加载模块：首次调用 Adam() 时才会 import，
# 届时它会扫描 sys.modules 里所有已知第三方包（含 tensorflow）的 __spec__，
# 若 __spec__ is None 则抛 ValueError。
# 解决方案：在注入 tensorflow mock 之前，先强制触发 _dynamo 初始化。
try:
    import torch._dynamo  # noqa — 强制触发 trace_rules 模块级初始化
except Exception:
    pass
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
# 导入本地 gru_ode_bayes（mock tensorflow 绕过 logging 依赖）
# =============================================================================

def load_gruode_model(repo_path):
    """
    从本地仓库加载 NNFOwithBayesianJumps。
    tensorflow 仅用于 TensorBoard logging，用 mock 模块替换，不影响模型。
    """
    # Mock tensorflow，避免 ImportError
    # 必须设置 __spec__，否则 torch._dynamo.trace_rules 扫描模块时会因 __spec__ is None 而崩溃
    if "tensorflow" not in sys.modules:
        import importlib.util
        tf_mock = types.ModuleType("tensorflow")
        tf_mock.__spec__ = importlib.util.spec_from_loader("tensorflow", loader=None)
        tf_mock.__path__ = []   # 让它看起来像一个包

        tf_summary = types.ModuleType("tensorflow.summary")
        tf_summary.__spec__ = importlib.util.spec_from_loader("tensorflow.summary", loader=None)
        tf_summary.FileWriter = lambda *a, **kw: None
        tf_mock.summary = tf_summary
        tf_mock.Summary = lambda *a, **kw: None

        sys.modules["tensorflow"] = tf_mock
        sys.modules["tensorflow.summary"] = tf_summary

    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        import gru_ode_bayes
        NNFOwithBayesianJumps = gru_ode_bayes.NNFOwithBayesianJumps
        print(f"[GRU-ODE-Bayes] 从本地仓库加载成功：{repo_path}")
        return NNFOwithBayesianJumps
    except Exception as e:
        raise ImportError(
            f"无法从 {repo_path} 加载 gru_ode_bayes：{e}\n"
            "请确认 --gruodebayes_repo_path 指向仓库根目录（含 gru_ode_bayes/ 子目录）。"
        )


# =============================================================================
# 数据格式转换：滑动窗口 -> GRU-ODE-Bayes 稀疏时间格式
# =============================================================================

def windows_to_gruode_format(windows: np.ndarray, device: torch.device):
    """
    将规则采样的滑动窗口 (B, W, D) 转换为 NNFOwithBayesianJumps 的输入格式。

    GRU-ODE-Bayes 设计用于不规则时序，其 forward 接口为：
      model(times, time_ptr, X, M, obs_idx, delta_t, T, cov)

    对规则采样，每个时间步 t=0,1,...,W-1 所有 B 个样本都有观测：
      times:    (W,)       每个唯一时间点，[0,1,...,W-1]
      time_ptr: (W+1,)     time_ptr[t] 到 time_ptr[t+1] 是时间 t 的观测索引
                           = [0, B, 2B, ..., W*B]
      X:        (W*B, D)   所有观测值，按时间主序排列
      M:        (W*B, D)   掩码，全 1（无缺失）
      obs_idx:  (W*B,)     每条观测属于哪个样本，= [0..B-1, 0..B-1, ...]
      delta_t:  float      ODE 积分步长
      T:        float      总时间长度 = W-1
      cov:      (B, 1)     协变量，用常数 0 占位
    """
    B, W, D = windows.shape

    times    = torch.arange(W, dtype=torch.float32, device=device)          # (W,)
    time_ptr = torch.arange(0, W * B + 1, B, dtype=torch.long, device=device)  # (W+1,)

    # 按时间主序展开：[t=0 的所有样本, t=1 的所有样本, ...]
    X_flat   = torch.tensor(windows, dtype=torch.float32, device=device)    # (B, W, D)
    X_flat   = X_flat.permute(1, 0, 2).reshape(W * B, D)                   # (W*B, D)
    M_flat   = torch.ones_like(X_flat)                                      # (W*B, D)

    obs_idx  = torch.arange(B, dtype=torch.long, device=device).repeat(W)  # (W*B,)
    cov      = torch.zeros(B, 1, dtype=torch.float32, device=device)        # (B, 1)

    return times, time_ptr, X_flat, M_flat, obs_idx, cov


# =============================================================================
# 读出层：把 hT 映射到预测值
# =============================================================================

class ForecastHead(nn.Module):
    """在 NNFOwithBayesianJumps 输出的 hT 上加一个小 MLP 做预测。"""
    def __init__(self, hidden_size, p_hidden, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, p_hidden),
            nn.ReLU(),
            nn.Linear(p_hidden, output_size),
        )
    def forward(self, hT):
        return self.net(hT)


# =============================================================================
# 数据准备
# =============================================================================

def build_windows(data: np.ndarray, window_size: int, steps_ahead: int = 1):
    """滑动窗口。data: (T, D) -> X: (N, W, D), y: (N, D)"""
    T, D = data.shape
    xs, ys = [], []
    for i in range(T - window_size - steps_ahead + 1):
        xs.append(data[i: i + window_size])
        ys.append(data[i + window_size + steps_ahead - 1])
    return np.stack(xs), np.stack(ys)


# =============================================================================
# 训练
# =============================================================================

def train_model(gruode_model, head, X_train, y_train,
                epochs, batch_size, lr, device, delta_t,
                patience=15, verbose=True):
    """
    联合训练 NNFOwithBayesianJumps（编码器）+ ForecastHead（预测头）。
    X_train: (N, W, D),  y_train: (N, D)
    """
    params = list(gruode_model.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=max(1, patience // 2), factor=0.5, verbose=False)

    N, W, D = X_train.shape
    dataset  = TensorDataset(torch.arange(N))
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    X_np = X_train  # numpy，按需取 batch
    y_t  = torch.tensor(y_train, dtype=torch.float32)

    best_loss, best_enc_state, best_head_state, no_imp = float("inf"), None, None, 0
    loss_history = []

    T_total = float(W - 1)   # 总时间长度（时间步数 -1）

    for epoch in tqdm(range(epochs), desc="  训练", disable=not verbose):
        gruode_model.train(); head.train()
        ep_loss, ep_n = 0.0, 0

        for (idx,) in loader:
            idx_list = idx.tolist()
            X_batch  = X_np[idx_list]              # (B, W, D) numpy
            y_batch  = y_t[idx].to(device)         # (B, D)
            B        = len(idx_list)

            times, time_ptr, X_flat, M_flat, obs_idx, cov = \
                windows_to_gruode_format(X_batch, device)

            optimizer.zero_grad()
            try:
                # NNFOwithBayesianJumps.forward 返回 (hT, loss, p_list, ...)
                # hT: (B, hidden_size)，loss 是模型内部的 NLL 损失
                result = gruode_model(
                    times, time_ptr, X_flat, M_flat, obs_idx,
                    delta_t=delta_t, T=T_total, cov=cov
                )
                hT = result[0]                     # (B, hidden_size)
                nll_loss = result[1]               # 模型内部 NLL（可选用于正则）

                pred     = head(hT)                # (B, D)
                mse_loss = nn.functional.mse_loss(pred, y_batch)
                # 总损失：MSE 为主，NLL 作轻微正则
                loss = mse_loss + 0.01 * nll_loss

                loss.backward()
                nn.utils.clip_grad_norm_(params, max_norm=5.0)
                optimizer.step()
                ep_loss += mse_loss.item() * B
                ep_n    += B
            except Exception as ex:
                tqdm.write(f"  [跳过 batch] {ex}")
                continue

        if ep_n > 0:
            ep_loss /= ep_n
        loss_history.append(ep_loss)
        scheduler.step(ep_loss)

        if ep_loss < best_loss:
            best_loss = ep_loss; no_imp = 0
            best_enc_state  = {k: v.cpu().clone() for k, v in gruode_model.state_dict().items()}
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
    gruode_model.to(device); head.to(device)
    return loss_history, best_loss


# =============================================================================
# 预测（真值滑窗 + 批量推理）
# =============================================================================

def forecast(gruode_model, head, history_scaled, fut_true_scaled,
             horizon, window_size, delta_t, device, scaler,
             batch_size=64, verbose=True):
    """
    真值滑窗：预测 fut[i] 时，窗口末步之前填真值。
    所有窗口批量推理。
    """
    T_hist  = history_scaled.shape[0]
    W       = window_size
    T_total = float(W - 1)
    full    = np.concatenate([history_scaled, fut_true_scaled], axis=0).astype(np.float32)

    windows  = np.stack([full[T_hist - W + i : T_hist + i] for i in range(horizon)])
    # windows: (horizon, W, D)

    gruode_model.eval(); head.eval()
    preds_list = []
    n_batches  = (horizon + batch_size - 1) // batch_size

    for b in tqdm(range(n_batches), desc="  批量推理", disable=not verbose):
        sl      = slice(b * batch_size, (b + 1) * batch_size)
        X_batch = windows[sl]              # (B, W, D)

        times, time_ptr, X_flat, M_flat, obs_idx, cov = \
            windows_to_gruode_format(X_batch, device)

        with torch.no_grad():
            result = gruode_model(
                times, time_ptr, X_flat, M_flat, obs_idx,
                delta_t=delta_t, T=T_total, cov=cov
            )
            hT   = result[0]              # (B, hidden_size)
            pred = head(hT).cpu().numpy() # (B, D)
        preds_list.append(pred)

    preds_s = np.concatenate(preds_list, axis=0)   # (horizon, D)

    # NaN 回退
    last_vals = windows[:, -1, :]
    for i in range(preds_s.shape[0]):
        nan_cols = np.isnan(preds_s[i])
        if nan_cols.any():
            preds_s[i, nan_cols] = last_vals[i, nan_cols]

    preds = scaler.inverse_transform(preds_s.astype(np.float64))
    stds  = np.zeros_like(preds)
    return preds, stds


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PM2.5: GRU-ODE-Bayes baseline")

    # 数据（与 RDE-GPR 完全对齐）
    parser.add_argument("--imputed_history_path", type=str, required=True)
    parser.add_argument("--ground_path",          type=str, required=True)
    parser.add_argument("--split_ratio",  type=float, default=0.5)
    parser.add_argument("--seed",         type=int,   default=42)

    # Horizon
    parser.add_argument("--horizon_days",  type=float, default=0.0)
    parser.add_argument("--horizon_steps", type=int,   default=0)

    # 模型超参（与 gru_ode_bayes 仓库接口对齐）
    parser.add_argument("--hidden_size",  type=int,   default=64)
    parser.add_argument("--p_hidden",     type=int,   default=32)
    parser.add_argument("--prep_hidden",  type=int,   default=32)
    parser.add_argument("--solver",       type=str,   default="euler",
                        choices=["euler", "midpoint", "dopri5"])
    parser.add_argument("--delta_t",      type=float, default=1.0,
                        help="ODE 积分步长（时间单位与 window 步长一致，默认 1.0）")
    parser.add_argument("--full_gru_ode", action="store_true",
                        help="使用 full GRU-ODE（默认 False，与仓库默认一致）")

    # 窗口
    parser.add_argument("--window_size",  type=int,   default=48)
    parser.add_argument("--steps_ahead",  type=int,   default=1)

    # 训练
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int,   default=15)

    # 路径
    parser.add_argument("--gruodebayes_repo_path", type=str, required=True,
                        help="gru_ode_bayes 本地仓库根目录（含 gru_ode_bayes/ 子包）")
    parser.add_argument("--device",      type=str,   default="auto")
    parser.add_argument("--out_dir",     type=str,   default="")
    parser.add_argument("--plot_dim",    type=int,   default=0)
    parser.add_argument("--load_model",   type=str,   default="",
                        help="跳过训练，直接从此路径加载 model.pt，e.g. .../model.pt")
    parser.add_argument("--skip_metrics",action="store_true")
    parser.add_argument("--no_verbose",  action="store_true")

    args    = parser.parse_args()
    verbose = not args.no_verbose
    set_global_seed(args.seed)

    # ── 加载模型类 ──
    NNFOwithBayesianJumps = load_gruode_model(args.gruodebayes_repo_path)

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
    horizon = full_horizon
    if args.horizon_days > 0:
        spd = infer_steps_per_day(fut_full.index)
        horizon = int(round(args.horizon_days * spd))
    elif args.horizon_steps > 0:
        horizon = int(args.horizon_steps)
    horizon  = max(1, min(horizon, full_horizon))
    fut_full = fut_full.iloc[:horizon].copy()
    report["checks"]["final_horizon_steps"] = int(horizon)

    # ================================================================
    # 3. 标准化
    # ================================================================
    scaler          = StandardScaler()
    history_scaled  = scaler.fit_transform(history).astype(np.float32)
    fut_true        = fut_full.values.astype(np.float64)
    fut_true_scaled = scaler.transform(fut_true).astype(np.float32)

    # ================================================================
    # 4. 滑动窗口
    # ================================================================
    W  = int(args.window_size)
    sa = int(args.steps_ahead)
    assert_or_raise(W >= 2, "window_size 必须 >= 2")
    assert_or_raise(history_scaled.shape[0] > W + sa,
        f"history 太短，请减小 window_size({W})。")

    X_win, y_win = build_windows(history_scaled, W, sa)
    # X_win: (N, W, D),  y_win: (N, D)

    report["checks"].update({
        "num_train_samples": int(X_win.shape[0]),
        "D": int(D), "window_size": W,
    })

    print(f"\n{'='*70}")
    print("PM2.5 预测 —— GRU-ODE-Bayes（直接使用本地仓库模型）")
    print(f"{'='*70}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nhorizon={horizon}步  训练样本={X_win.shape[0]}")
    print(f"D={D}  hidden={args.hidden_size}  solver={args.solver}  delta_t={args.delta_t}")

    # ================================================================
    # 5. 构建模型
    #    cov_size=1：协变量维度（我们用常数占位）
    # ================================================================
    gruode_model = NNFOwithBayesianJumps(
        input_size   = D,
        hidden_size  = args.hidden_size,
        p_hidden     = args.p_hidden,
        prep_hidden  = args.prep_hidden,
        logvar       = True,
        mixing        = 1,
        full_gru_ode = args.full_gru_ode,
        impute       = False,   # 我们数据无缺失，不需要 imputation
        solver       = args.solver,
        cov_size     = 1,
    ).to(device)

    head = ForecastHead(args.hidden_size, args.p_hidden, D).to(device)

    n_params = (sum(p.numel() for p in gruode_model.parameters() if p.requires_grad)
                + sum(p.numel() for p in head.parameters() if p.requires_grad))
    print(f"模型参数量：{n_params:,}")

    # ================================================================
    # 6. 训练 or 加载
    # ================================================================
    if args.load_model:
        # ── 直接加载已有 checkpoint，跳过训练 ──
        ckpt_path = args.load_model
        assert_or_raise(os.path.isfile(ckpt_path), f"找不到 model.pt：{ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        gruode_model.load_state_dict(ckpt["gruode"])
        head.load_state_dict(ckpt["head"])
        print(f"\n[跳过训练] 已从 {ckpt_path} 加载模型权重")
        report["training"] = {"loaded_from": ckpt_path, "n_params": int(n_params)}
    else:
        print("\n[训练阶段]")
        t0 = time.time()
        loss_hist, best_loss = train_model(
            gruode_model, head, X_win, y_win,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, device=device, delta_t=args.delta_t,
            patience=args.patience, verbose=verbose)
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
    print("\n[预测阶段]（真值滑窗 + 批量推理）")
    t1 = time.time()
    preds, stds = forecast(
        gruode_model, head, history_scaled, fut_true_scaled,
        horizon=horizon, window_size=W, delta_t=args.delta_t,
        device=device, scaler=scaler,
        batch_size=args.batch_size, verbose=verbose)
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
        print("\n整体评估（GRU-ODE-Bayes）：")
        print(json.dumps(overall, indent=4, ensure_ascii=False))
    else:
        print("\n已跳过 metrics（--skip_metrics）")

    print(f"\n输出目录：{out_dir}")


if __name__ == "__main__":
    main()
