# -*- coding: utf-8 -*-
"""
PM2.5 后续预测脚本 —— GRU-ODE-Bayes（贴近原始仓库逻辑）
============================================================
核心思路（与原仓库 paper_plotting.py / climate_gruode.py 对齐）：
  1. 训练：用模型自身 NLL 损失（result[1]），无额外 ForecastHead
  2. 预测：把未来时间步以掩码 M=0 附加到窗口末尾，调用
           model(..., return_path=True) 从 p_vec 直接读出未来均值

运行示例（加载已有模型）：
  python pm25_gruodebayes_forecast.py \
    --imputed_history_path .../history_imputed.csv \
    --ground_path .../pm25_ground.txt \
    --gruodebayes_repo_path /home/rhl/baseline/gru_ode_bayes-master/gru_ode_bayes-master_tensorflow \
    --split_ratio 0.5 --horizon_days 1 \
    --hidden_size 64 --p_hidden 32 --prep_hidden 32 \
    --window_size 48 --delta_t 1.0 \
    --load_model .../model.pt \
    --seed 42

运行示例（从头训练）：
  python pm25_gruodebayes_forecast.py \
    --imputed_history_path .../history_imputed.csv \
    --ground_path .../pm25_ground.txt \
    --gruodebayes_repo_path /home/rhl/baseline/gru_ode_bayes-master/gru_ode_bayes-master_tensorflow \
    --split_ratio 0.5 --horizon_days 1 \
    --hidden_size 64 --p_hidden 32 --prep_hidden 32 \
    --window_size 48 --delta_t 1.0 \
    --epochs 100 --batch_size 64 --lr 1e-3 --seed 42
"""

import os, sys, json, time, random, argparse, datetime, warnings, types
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn

# ── torch._dynamo.trace_rules 懒加载时会扫描 sys.modules 里所有模块的 __spec__，
#    若 tensorflow mock 的 __spec__ 为 None 则报 ValueError。
#    解决：在注入 mock 之前先强制触发 _dynamo 初始化。
try:
    import torch._dynamo  # noqa
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
    tensorflow 仅用于 TensorBoard logging，mock 替换后不影响模型。
    """
    if "tensorflow" not in sys.modules:
        import importlib.machinery
        tf_mock = types.ModuleType("tensorflow")
        tf_mock.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
        tf_mock.__path__ = []

        tf_summary = types.ModuleType("tensorflow.summary")
        tf_summary.__spec__ = importlib.machinery.ModuleSpec("tensorflow.summary", loader=None)
        tf_summary.FileWriter = lambda *a, **kw: None
        tf_mock.summary   = tf_summary
        tf_mock.Summary   = lambda *a, **kw: None

        sys.modules["tensorflow"]         = tf_mock
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
            "请确认 --gruodebayes_repo_path 指向仓库根目录（含 gru_ode_bayes/ 子包）。"
        )


# =============================================================================
# 数据格式构建（对齐原始仓库接口）
# =============================================================================

def build_gruode_batch(windows: np.ndarray, masks: np.ndarray, device: torch.device):
    """
    将规则采样窗口转换为 NNFOwithBayesianJumps.forward 所需的稀疏格式。

    原始仓库接口（参考 paper_plotting.py）：
      model(times, time_ptr, X, M, obs_idx, delta_t=..., T=..., cov=..., return_path=...)
      -> hT, loss, _, t_vec, p_vec, _, eval_times, eval_vals

    参数
    ----
    windows : (B, W, D)  —— 观测值，未来 masked 步可为 0
    masks   : (B, W, D)  —— 1=有效观测，0=无效（掩码）

    规则采样下的稀疏格式：
      times    : (W,)    每个时间点 [0, 1, ..., W-1]
      time_ptr : (W+1,)  第 t 个时间点的样本范围 [0, B, 2B, ..., W*B]
      X        : (W*B, D) 按时间主序展开的观测
      M        : (W*B, D) 对应掩码
      obs_idx  : (W*B,)  每条观测归属的样本序号
      cov      : (B, 1)  协变量（常数 0 占位）
    """
    B, W, D = windows.shape

    times    = torch.arange(W, dtype=torch.float32, device=device)
    time_ptr = torch.arange(0, W * B + 1, B, dtype=torch.long, device=device)

    # 按时间主序展开：[t=0 的 B 个样本, t=1 的 B 个样本, ...]
    X_flat = (torch.tensor(windows, dtype=torch.float32, device=device)
              .permute(1, 0, 2).reshape(W * B, D))
    M_flat = (torch.tensor(masks,   dtype=torch.float32, device=device)
              .permute(1, 0, 2).reshape(W * B, D))

    obs_idx = torch.arange(B, dtype=torch.long, device=device).repeat(W)
    cov     = torch.zeros(B, 1, dtype=torch.float32, device=device)

    return times, time_ptr, X_flat, M_flat, obs_idx, cov


# =============================================================================
# 数据准备：滑动窗口（训练用，全部掩码=1）
# =============================================================================

def build_train_windows(data: np.ndarray, window_size: int, steps_ahead: int = 1):
    """
    训练窗口：[i, i+W) 为输入，目标 = 下 steps_ahead 步（不用于 NLL 训练，仅保留接口一致）。
    由于 NLL 训练用的是窗口内所有观测，steps_ahead 主要影响窗口切分。
    返回：windows (N, W, D), target (N, D)
    """
    T, D = data.shape
    xs, ys = [], []
    for i in range(T - window_size - steps_ahead + 1):
        xs.append(data[i: i + window_size])
        ys.append(data[i + window_size + steps_ahead - 1])
    return np.stack(xs), np.stack(ys)


# =============================================================================
# 训练（原始仓库逻辑：直接用 NLL 损失，无 ForecastHead）
# =============================================================================

def train_model(gruode_model, X_train, epochs, batch_size, lr,
                device, delta_t, patience=15, verbose=True):
    """
    用模型自身 NLL 损失训练（对齐原仓库 climate_gruode.py 的做法）。
    X_train: (N, W, D) —— 均为有效观测，掩码全 1。
    """
    optimizer = torch.optim.Adam(gruode_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=max(1, patience // 2), factor=0.5, verbose=False)

    N, W, D = X_train.shape
    dataset  = TensorDataset(torch.arange(N))
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    T_total  = float(W - 1)

    best_loss, best_state, no_imp = float("inf"), None, 0
    loss_history = []

    for epoch in tqdm(range(epochs), desc="  训练", disable=not verbose):
        gruode_model.train()
        ep_loss, ep_n = 0.0, 0

        for (idx,) in loader:
            idx_list = idx.tolist()
            X_batch  = X_train[idx_list]                      # (B, W, D) numpy
            B        = len(idx_list)
            masks    = np.ones_like(X_batch)                  # 全 1（无缺失）

            times, time_ptr, X_flat, M_flat, obs_idx, cov = \
                build_gruode_batch(X_batch, masks, device)

            optimizer.zero_grad()
            try:
                # 原始仓库 forward 返回：(hT, loss, ...)
                # loss = mixing * loss_1 + loss_2
                # loss_1: 观测 NLL，loss_2: KL / 平滑项
                result   = gruode_model(
                    times, time_ptr, X_flat, M_flat, obs_idx,
                    delta_t=delta_t, T=T_total, cov=cov
                )
                nll_loss = result[1]       # 模型内部 NLL（原仓库即以此为目标）

                nll_loss.backward()
                nn.utils.clip_grad_norm_(gruode_model.parameters(), max_norm=5.0)
                optimizer.step()

                ep_loss += nll_loss.item() * B
                ep_n    += B
            except Exception as ex:
                tqdm.write(f"  [跳过 batch] {ex}")
                continue

        if ep_n > 0:
            ep_loss /= ep_n
        loss_history.append(ep_loss)
        scheduler.step(ep_loss)

        if ep_loss < best_loss:
            best_loss  = ep_loss; no_imp = 0
            best_state = {k: v.cpu().clone()
                          for k, v in gruode_model.state_dict().items()}
        else:
            no_imp += 1

        if no_imp >= patience:
            if verbose:
                tqdm.write(f"  Early stopping @ epoch {epoch+1}，best NLL={best_loss:.6f}")
            break

    if best_state:
        gruode_model.load_state_dict(best_state)
    gruode_model.to(device)
    return loss_history, best_loss


# =============================================================================
# 预测（对齐原始仓库 return_path=True，从 p_vec 读出未来均值）
# =============================================================================

def _extract_mean_from_pvec(p_vec, input_size, logvar):
    """
    原仓库：若 logvar=True，p_vec 形状 (T, B, 2*D)，前 D 为均值，后 D 为对数方差。
           若 logvar=False，p_vec 形状 (T, B, D)，直接为均值。
    返回均值 (T, B, D)。
    """
    if logvar:
        return p_vec[..., :input_size]
    return p_vec


def forecast_step(gruode_model, window_obs: np.ndarray,
                  steps_ahead: int, delta_t: float,
                  device: torch.device, input_size: int, logvar: bool):
    """
    对单个样本（B=1）做一次预测：
    - window_obs: (W, D) 历史观测（已标准化），全部掩码=1
    - 在末尾附加 steps_ahead 个掩码=0 的时间步（无观测）
    - 调用 model(..., return_path=True) 拿 p_vec
    - 返回第 W+steps_ahead-1 时刻的均值预测 (D,)

    原始仓库 paper_plotting.py 的做法：
      hT, loss, _, t_vec, p_vec, _, eval_times, eval_vals =
          model(times, time_ptr, X, M, obs_idx, delta_t=delta_t, T=T, cov=cov, return_path=True)
    """
    W, D = window_obs.shape
    total_len = W + steps_ahead    # 观测 + 未来
    T_total   = float(total_len - 1)

    # 构建扩展窗口（未来步 = 0，掩码 = 0）
    ext_obs  = np.zeros((1, total_len, D), dtype=np.float32)
    ext_mask = np.zeros((1, total_len, D), dtype=np.float32)
    ext_obs[0, :W, :]  = window_obs
    ext_mask[0, :W, :] = 1.0      # 历史有效，未来掩码=0

    times, time_ptr, X_flat, M_flat, obs_idx, cov = \
        build_gruode_batch(ext_obs, ext_mask, device)

    with torch.no_grad():
        result = gruode_model(
            times, time_ptr, X_flat, M_flat, obs_idx,
            delta_t=delta_t, T=T_total, cov=cov,
            return_path=True
        )
    # result: (hT, loss, _, t_vec, p_vec, _, eval_times, eval_vals)
    # p_vec shape: (total_len, B, 2*D) or (total_len, B, D)
    p_vec = result[4]   # (total_len, 1, *)
    mean  = _extract_mean_from_pvec(p_vec, input_size, logvar)  # (total_len, 1, D)
    return mean[-1, 0, :].cpu().numpy()   # 最后一个时刻（即 steps_ahead 步后）的预测


def forecast(gruode_model, history_scaled, fut_true_scaled,
             horizon, window_size, steps_ahead, delta_t,
             device, scaler, input_size, logvar,
             batch_size=64, verbose=True):
    """
    真值滑窗（oracle）预测：
    预测第 i 步时，窗口 = history[-W+i:] 全为真值（已标准化）。
    """
    T_hist = history_scaled.shape[0]
    D      = history_scaled.shape[1]
    W      = window_size
    full   = np.concatenate([history_scaled, fut_true_scaled], axis=0).astype(np.float32)

    gruode_model.eval()
    preds_list = []

    # 按 batch_size 批量处理以加速
    n_batches = (horizon + batch_size - 1) // batch_size

    for b in tqdm(range(n_batches), desc="  批量推理", disable=not verbose):
        sl      = slice(b * batch_size, min((b + 1) * batch_size, horizon))
        indices = list(range(sl.start, sl.stop))
        B_cur   = len(indices)

        total_len = W + steps_ahead
        T_total   = float(total_len - 1)

        ext_obs  = np.zeros((B_cur, total_len, D), dtype=np.float32)
        ext_mask = np.zeros((B_cur, total_len, D), dtype=np.float32)

        for j, i in enumerate(indices):
            obs_win = full[T_hist - W + i : T_hist + i]   # (W, D)
            ext_obs[j, :W, :]  = obs_win
            ext_mask[j, :W, :] = 1.0

        times, time_ptr, X_flat, M_flat, obs_idx, cov = \
            build_gruode_batch(ext_obs, ext_mask, device)

        with torch.no_grad():
            result = gruode_model(
                times, time_ptr, X_flat, M_flat, obs_idx,
                delta_t=delta_t, T=T_total, cov=cov,
                return_path=True
            )
        p_vec = result[4]                                          # (total_len, B_cur, *)
        mean  = _extract_mean_from_pvec(p_vec, input_size, logvar)  # (total_len, B_cur, D)
        pred  = mean[-1, :, :].cpu().numpy()                       # (B_cur, D)
        preds_list.append(pred)

    preds_s = np.concatenate(preds_list, axis=0)   # (horizon, D)

    # NaN 回退：用窗口末步值填补
    last_obs = np.stack([
        full[T_hist - W + i + W - 1] for i in range(horizon)
    ])  # (horizon, D)
    for i in range(horizon):
        nan_cols = np.isnan(preds_s[i])
        if nan_cols.any():
            preds_s[i, nan_cols] = last_obs[i, nan_cols]

    preds = scaler.inverse_transform(preds_s.astype(np.float64))
    stds  = np.zeros_like(preds)
    return preds, stds


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PM2.5: GRU-ODE-Bayes（贴近原仓库逻辑）")

    # 数据
    parser.add_argument("--imputed_history_path", type=str, required=True)
    parser.add_argument("--ground_path",          type=str, required=True)
    parser.add_argument("--split_ratio",  type=float, default=0.5)
    parser.add_argument("--seed",         type=int,   default=42)

    # Horizon
    parser.add_argument("--horizon_days",  type=float, default=0.0)
    parser.add_argument("--horizon_steps", type=int,   default=0)

    # 模型超参（与仓库接口对齐）
    parser.add_argument("--hidden_size",  type=int,   default=64)
    parser.add_argument("--p_hidden",     type=int,   default=32)
    parser.add_argument("--prep_hidden",  type=int,   default=32)
    parser.add_argument("--cov_hidden",   type=int,   default=1,
                        help="协变量映射隐层（仓库默认 1）")
    parser.add_argument("--logvar",       action="store_true", default=True,
                        help="p_vec 是否含 logvar（仓库默认 True）")
    parser.add_argument("--mixing",       type=float, default=1.0,
                        help="loss_1 和 loss_2 的混合权重（仓库默认 1）")
    parser.add_argument("--solver",       type=str,   default="euler",
                        choices=["euler", "midpoint", "dopri5"])
    parser.add_argument("--delta_t",      type=float, default=1.0)
    parser.add_argument("--full_gru_ode", action="store_true")
    parser.add_argument("--impute",       action="store_true",
                        help="是否开启 impute 模式（仓库默认 True，数据无缺失时可关）")
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--obs_noise_std",type=float, default=0.1,
                        help="观测噪声标准差（影响 KL 项）")

    # 窗口
    parser.add_argument("--window_size",  type=int,   default=48,
                        help="历史观测窗口长度（步数）")
    parser.add_argument("--steps_ahead",  type=int,   default=1,
                        help="预测步数（ODE 在窗口末尾继续运行的步数）")

    # 训练
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int,   default=15)

    # 路径
    parser.add_argument("--gruodebayes_repo_path", type=str, required=True)
    parser.add_argument("--load_model",  type=str,   default="",
                        help="跳过训练，直接从此路径加载 model.pt")
    parser.add_argument("--device",     type=str,   default="auto")
    parser.add_argument("--out_dir",    type=str,   default="")
    parser.add_argument("--plot_dim",   type=int,   default=0)
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--no_verbose",   action="store_true")

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
    df_full  = pd.read_csv(args.ground_path, index_col="datetime",
                           parse_dates=True).sort_index()
    hist_full, fut_full, meta = time_split_df(df_full, args.split_ratio)
    df_hist  = pd.read_csv(args.imputed_history_path, index_col="datetime",
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
        spd     = infer_steps_per_day(fut_full.index)
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
    # 4. 训练窗口（用于 NLL 训练）
    # ================================================================
    W  = int(args.window_size)
    sa = int(args.steps_ahead)
    assert_or_raise(W >= 2, "window_size 必须 >= 2")
    assert_or_raise(history_scaled.shape[0] > W + sa,
        f"history 太短，请减小 window_size({W})。")

    X_win, _ = build_train_windows(history_scaled, W, sa)
    # X_win: (N, W, D)

    report["checks"].update({
        "num_train_samples": int(X_win.shape[0]),
        "D": int(D), "window_size": W, "steps_ahead": sa,
    })

    print(f"\n{'='*70}")
    print("PM2.5 预测 —— GRU-ODE-Bayes（贴近原仓库逻辑，NLL 训练 + return_path 预测）")
    print(f"{'='*70}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\nhorizon={horizon}步  训练样本={X_win.shape[0]}")
    print(f"D={D}  hidden={args.hidden_size}  solver={args.solver}  delta_t={args.delta_t}")

    # ================================================================
    # 5. 构建模型（参数对齐原仓库 NNFOwithBayesianJumps）
    # ================================================================
    gruode_model = NNFOwithBayesianJumps(
        input_size    = D,
        hidden_size   = args.hidden_size,
        p_hidden      = args.p_hidden,
        prep_hidden   = args.prep_hidden,
        logvar        = args.logvar,
        mixing        = args.mixing,
        full_gru_ode  = args.full_gru_ode,
        impute        = args.impute,
        solver        = args.solver,
        cov_size      = 1,
        cov_hidden    = args.cov_hidden,
        dropout_rate  = args.dropout_rate,
        obs_noise_std = args.obs_noise_std,
    ).to(device)

    n_params = sum(p.numel() for p in gruode_model.parameters() if p.requires_grad)
    print(f"模型参数量：{n_params:,}")

    # ================================================================
    # 6. 训练 or 加载
    # ================================================================
    if args.load_model:
        ckpt_path = args.load_model
        assert_or_raise(os.path.isfile(ckpt_path), f"找不到 model.pt：{ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        # 兼容旧格式（含 ForecastHead）和新格式（仅 gruode）
        state = ckpt.get("gruode", ckpt)
        gruode_model.load_state_dict(state)
        print(f"\n[跳过训练] 已从 {ckpt_path} 加载模型权重")
        report["training"] = {"loaded_from": ckpt_path, "n_params": int(n_params)}
    else:
        print("\n[训练阶段]（NLL 损失，无 ForecastHead）")
        t0 = time.time()
        loss_hist, best_loss = train_model(
            gruode_model, X_win,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, device=device, delta_t=args.delta_t,
            patience=args.patience, verbose=verbose)
        train_time = time.time() - t0
        print(f"  耗时 {train_time:.1f}s，最优 NLL={best_loss:.6f}")

        torch.save({"gruode": gruode_model.state_dict()},
                   os.path.join(out_dir, "model.pt"))
        report["training"] = {
            "train_time_sec": round(train_time, 2),
            "best_nll":       float(best_loss),
            "epochs_run":     int(len(loss_hist)),
            "n_params":       int(n_params),
        }
        safe_json_dump(loss_hist, os.path.join(out_dir, "loss_history.json"))

    # ================================================================
    # 7. 预测（return_path=True，从 p_vec 读均值）
    # ================================================================
    print("\n[预测阶段]（真值滑窗 + return_path=True，从 p_vec 读均值）")
    t1 = time.time()
    preds, stds = forecast(
        gruode_model, history_scaled, fut_true_scaled,
        horizon=horizon, window_size=W, steps_ahead=sa,
        delta_t=args.delta_t, device=device, scaler=scaler,
        input_size=D, logvar=args.logvar,
        batch_size=args.batch_size, verbose=verbose)
    print(f"  耗时 {time.time()-t1:.1f}s")

    n_nan = int(np.isnan(preds).sum())
    report["pred_stats"] = basic_array_stats(preds, "preds")
    report["checks"]["pred_nan_count"] = n_nan
    if n_nan > 0:
        print(f"  [警告] 仍有 {n_nan} 个 NaN（已回填）")

    # ================================================================
    # 8. 保存
    # ================================================================
    df_pred  = pd.DataFrame(preds, index=fut_full.index, columns=columns)
    df_std   = pd.DataFrame(stds,  index=fut_full.index, columns=columns)
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
