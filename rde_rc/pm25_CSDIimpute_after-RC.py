#!/usr/bin/env python3
"""
PM2.5 预测：使用 CSDI 补值 + RC (Reservoir Computing) 预测
基于 rde_gpr/pm25_CSDIimpute_after-RDEgpr.py 的结构，将 GPR 替换为 RC

RDE (Random Embedding Ensemble) + RC 方法：
1. 随机选择 L 个变量作为嵌入
2. 训练 RC 模型
3. 多次随机初始化，取平均
"""
import os
import sys
import json
import time
import random
import argparse
import datetime
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_json_dump(obj, path: str):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def time_split_df(df_full: pd.DataFrame, split_ratio: float):
    N = len(df_full)
    split_point = int(N * split_ratio)
    hist_df = df_full.iloc[:split_point]
    fut_df = df_full.iloc[split_point:]
    meta = {
        "total_len": N,
        "split_ratio": split_ratio,
        "split_point": split_point,
        "hist_len": len(hist_df),
        "fut_len": len(fut_df),
    }
    return hist_df, fut_df, meta


class ReservoirComputer:
    """
    简化的回声状态网络 (ESN)
    """
    def __init__(self, reservoir_size=100, spectral_radius=0.9, 
                 input_scaling=1.0, leaking_rate=0.3, regularization=1e-4):
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leaking_rate = leaking_rate
        self.regularization = regularization
        
        self.W_in = None
        self.W = None
        self.W_out = None
        self.last_state = None
        
    def _initialize(self, n_inputs, rng):
        """初始化权重矩阵"""
        self.W_in = rng.randn(self.reservoir_size, n_inputs) * self.input_scaling
        
        # 储备池权重
        density = 0.1
        W_dense = rng.randn(self.reservoir_size, self.reservoir_size)
        W_dense[rng.rand(*W_dense.shape) > density] = 0
        
        eigenvalues = np.linalg.eigvals(W_dense)
        max_eig = np.max(np.abs(eigenvalues))
        if max_eig > 0:
            self.W = W_dense * (self.spectral_radius / max_eig)
        else:
            self.W = W_dense
            
    def compute_states(self, X, discard=0, save_last_state=True):
        """计算储备池状态"""
        T, n_inputs = X.shape
        if self.W_in is None:
            self._initialize(n_inputs, np.random.RandomState(42))
            
        states = np.zeros((T, self.reservoir_size))
        state = np.zeros(self.reservoir_size)
        
        for t in range(T):
            pre_activation = self.W @ state + self.W_in @ X[t]
            state = (1 - self.leaking_rate) * state + self.leaking_rate * np.tanh(pre_activation)
            states[t] = state
        
        if save_last_state:
            self.last_state = state.copy()
            
        return states[discard:]
    
    def fit_predict(self, X_train, y_train, X_test, discard=0):
        """
        训练并预测
        X_train: (T_train, n_inputs)
        y_train: (T_train,) 目标变量
        X_test: (1, n_inputs) 测试点
        """
        # 确保 discard 不会超过训练数据长度
        discard = min(discard, X_train.shape[0] - 1)
        discard = max(discard, 0)
        
        states = self.compute_states(X_train, discard=discard, save_last_state=True)
        y_train = y_train[discard:]
        
        # 检查是否有足够的数据
        if states.shape[0] < 2:
            return np.nan
        
        # 线性回归: y = W_out @ state + b
        n_samples = states.shape[0]
        X_aug = np.concatenate([states, np.ones((n_samples, 1))], axis=1)
        
        # 岭回归
        regularization_matrix = self.regularization * np.eye(X_aug.shape[1])
        XTX = X_aug.T @ X_aug + regularization_matrix
        XTy = X_aug.T @ y_train
        
        try:
            W_out = np.linalg.solve(XTX, XTy)
        except:
            return np.nan
        
        # 使用测试点计算新状态并预测
        if self.last_state is None:
            return np.nan
        
        # 从 last_state 继续计算测试点的状态
        state = self.last_state.copy()
        pre_activation = self.W @ state + self.W_in @ X_test[0]
        state = (1 - self.leaking_rate) * state + self.leaking_rate * np.tanh(pre_activation)
        
        test_aug = np.concatenate([state.reshape(1, -1), np.ones((1, 1))], axis=1)
        pred = test_aug @ W_out
        
        if np.ndim(pred) == 2:
            return pred[0, 0]
        else:
            return pred[0]


def rc_predict_for_target(traindata, target_idx, L, s, steps_ahead, rng, debug=False):
    """
    对单个目标变量使用 RC + RDE 进行预测
    
    traindata: (T, D) 训练数据
    target_idx: int 目标维度索引
    L: int 嵌入维度（随机选择的变量数）
    s: int 随机组合数量（集成数量）
    steps_ahead: int 预测步数
    """
    T, D = traindata.shape
    
    # 检查训练数据长度是否足够
    if T < steps_ahead + 2:
        if debug:
            print(f"  目标 {target_idx}: 训练数据太短 (T={T}), 回退到持久性")
        return np.nan, np.nan, 2
    
    # RDE: 随机选择 L 个变量的组合
    all_cols = list(range(D))
    available_cols = [c for c in all_cols if c != target_idx]
    
    if len(available_cols) < L - 1:
        L = len(available_cols) + 1
    
    # 生成所有可能的组合（或随机采样）
    if s >= 100 or len(available_cols) > 10:
        # 随机采样 s 个组合
        combinations = []
        for _ in range(s):
            selected = rng.choice(available_cols, size=L-1, replace=False)
            comb = [int(target_idx)] + [int(x) for x in selected]
            combinations.append(comb)
    else:
        # 枚举所有组合
        combs = list(itertools.combinations(available_cols, L-1))
        rng.shuffle(combs)
        combinations = [[int(target_idx)] + [int(x) for x in c] for c in combs[:s]]
    
    predictions = []
    
    # 确保 reservoir_size 不会太小
    reservoir_size = min(100, max(10, T // 10))
    
    for i, comb in enumerate(combinations):
        if debug and i == 0:
            print(f"    comb {i}: {comb}, target_idx={target_idx}")
        try:
            X = traindata[:-steps_ahead, comb]
            if debug and i == 0:
                print(f"      X shape: {X.shape}")
            y = traindata[steps_ahead:, target_idx]
            if debug and i == 0:
                print(f"      y shape: {y.shape}, std: {np.std(y):.6f}")
            x_test = traindata[-steps_ahead, comb].reshape(1, -1)
            
            if np.isnan(X).any() or np.isnan(y).any() or np.isnan(x_test).any():
                if debug and i == 0:
                    print(f"    comb {i}: 数据包含 NaN")
                continue
            if np.std(y) < 1e-6:
                if debug and i == 0:
                    print(f"    comb {i}: y 标准差太小 ({np.std(y):.6f})")
                continue
            
            # 随机初始化 RC
            rng_rc = np.random.RandomState(rng.randint(0, 10000) + i)
            rc = ReservoirComputer(
                reservoir_size=reservoir_size,
                spectral_radius=0.9,
                input_scaling=1.0,
                leaking_rate=0.3,
                regularization=1e-2
            )
            
            pred = rc.fit_predict(X, y, x_test)
            
            if np.isnan(pred):
                if debug and i == 0:
                    print(f"    comb {i}: RC 预测为 NaN")
                continue
            
            predictions.append(pred)
            
        except Exception as e:
            if debug and i == 0:
                print(f"    comb {i}: 异常 {e}")
            continue
    
    if len(predictions) == 0:
        if debug:
            print(f"  目标 {target_idx}: 无有效预测，回退到持久性")
        return np.nan, np.nan, 2
    
    if debug:
        print(f"  目标 {target_idx}: {len(predictions)} 个有效预测, mean={np.mean(predictions):.4f}, std={np.std(predictions):.4f}")
    
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    
    return pred_mean, pred_std, 0


def rc_forecast_multivariate(history, future_truth, horizon, trainlength, L, s, steps_ahead, n_jobs=4, debug=False):
    """
    使用 RC + RDE 对多变量时间序列进行滚动预测
    """
    history = np.asarray(history, dtype=np.float64)
    future_truth = np.asarray(future_truth, dtype=np.float64)
    
    T_hist, D = history.shape
    
    print(f"RC+RDE 滚动预测: T_hist={T_hist}, D={D}, horizon={horizon}, trainlength={trainlength}, L={L}, s={s}")
    
    seq_true = np.vstack([history, future_truth[:horizon]])
    
    preds = np.zeros((horizon, D), dtype=np.float64)
    stds = np.zeros((horizon, D), dtype=np.float64)
    
    rng = np.random.RandomState(42)
    
    for step in tqdm(range(horizon), desc="RC+RDE 滚动预测"):
        if step == 0:
            debug_step = True
        else:
            debug_step = False
            
        t_pred = T_hist + step
        start = t_pred - trainlength
        end = t_pred
        
        if debug_step:
            print(f"  step={step}, t_pred={t_pred}, start={start}, end={end}")
            
        traindata = seq_true[start:end].copy()
        
        if debug_step:
            print(f"  traindata shape: {traindata.shape}")
        
        prev_true = seq_true[t_pred - 1].copy()
        next_vec = prev_true.copy()
        next_std = np.zeros((D,), dtype=np.float64)
        
        for j in range(D):
            rng_j = np.random.RandomState(rng.randint(0, 100000) + step * 1000 + j)
            
            pred_j, std_j, status = rc_predict_for_target(
                traindata=traindata,
                target_idx=j,
                L=L,
                s=s,
                steps_ahead=steps_ahead,
                rng=rng_j,
                debug=debug_step
            )
            
            if np.isnan(pred_j):
                pred_j = next_vec[j]
                std_j = 0.0
            
            next_vec[j] = float(pred_j)
            next_std[j] = float(std_j)
        
        preds[step] = next_vec
        stds[step] = next_std
    
    return preds, stds


def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    
    if len(y_true_valid) == 0:
        return {"rmse": np.nan, "mae": np.nan}
    
    rmse = np.sqrt(np.mean((y_true_valid - y_pred_valid) ** 2))
    mae = np.mean(np.abs(y_true_valid - y_pred_valid))
    
    return {"rmse": rmse, "mae": mae}


def save_plots(out_dir: str, fut_index: pd.DatetimeIndex, y_true: np.ndarray, y_pred: np.ndarray, plot_dim: int = 0):
    ensure_dir(out_dir)
    
    plt.figure(figsize=(14, 5))
    plt.plot(fut_index, y_true[:, plot_dim], label="True", color="steelblue")
    plt.plot(fut_index, y_pred[:, plot_dim], label="RC+RDE Forecast", color="tomato", alpha=0.8)
    plt.xlabel("Time")
    plt.ylabel("PM2.5")
    plt.title(f"RC+RDE Forecast vs True (dim {plot_dim})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/plot_forecast_dim{plot_dim}.png", dpi=150)
    plt.close()
    
    rmses = []
    for j in range(y_true.shape[1]):
        mask = ~np.isnan(y_true[:, j]) & ~np.isnan(y_pred[:, j])
        if mask.sum() > 0:
            rmse = np.sqrt(np.mean((y_true[mask, j] - y_pred[mask, j]) ** 2))
        else:
            rmse = np.nan
        rmses.append(rmse)
    
    plt.figure(figsize=(14, 5))
    plt.bar(range(len(rmses)), rmses)
    plt.xlabel("Dimension")
    plt.ylabel("RMSE")
    plt.title("RC+RDE RMSE per Dimension")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/plot_rmse_per_dim.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="PM2.5: CSDI补值 + RC+RDE预测")
    
    parser.add_argument("--imputed_history_path", type=str, required=True)
    parser.add_argument("--ground_path", type=str, required=True)
    parser.add_argument("--split_ratio", type=float, default=0.5)
    
    # RC+RDE 参数
    parser.add_argument("--L", type=int, default=7,
                        help="嵌入维度（随机选择的变量数）")
    parser.add_argument("--s", type=int, default=100,
                        help="随机组合数量（集成数量）")
    parser.add_argument("--steps_ahead", type=int, default=1)
    parser.add_argument("--trainlength", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=24)
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    set_global_seed(args.seed)
    
    print(f"\n{'='*60}")
    print("PM2.5 预测: CSDI补值 + RC+RDE (Random Embedding Ensemble)")
    print(f"{'='*60}")
    
    # 加载数据
    print(f"\n[1] 加载数据: {args.imputed_history_path}")
    df_imputed = pd.read_csv(args.imputed_history_path, index_col="datetime", parse_dates=True)
    
    df_full = pd.read_csv(args.ground_path, index_col="datetime", parse_dates=True)
    hist_df, fut_df, meta = time_split_df(df_full, args.split_ratio)
    
    print(f"  历史: {meta['hist_len']} 步, 未来: {meta['fut_len']} 步")
    
    df_hist = df_imputed.iloc[:meta['hist_len']].copy()
    
    # RC+RDE 预测
    print(f"\n[2] RC+RDE 预测 (L={args.L}, s={args.s}, steps_ahead={args.steps_ahead}, horizon={args.horizon})")
    
    history_values = df_hist.values.astype(np.float64)
    future_values = fut_df.values.astype(np.float64)
    
    y_pred, y_std = rc_forecast_multivariate(
        history_values,
        future_values,
        horizon=args.horizon,
        trainlength=args.trainlength,
        L=args.L,
        s=args.s,
        steps_ahead=args.steps_ahead,
        debug=args.debug
    )
    
    print(f"  预测完成，形状: {y_pred.shape}")
    
    # 评估
    if not args.skip_metrics:
        print(f"\n[3] 评估")
        
        y_true = future_values[:args.horizon, :]
        
        metrics = compute_metrics(y_true, y_pred)
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
    else:
        metrics = {"rmse": None, "mae": None}
    
    # 保存
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"./save/pm25_rc_rde_{args.split_ratio}_{args.seed}_{timestamp}"
    ensure_dir(out_dir)
    
    pred_df = pd.DataFrame(y_pred, index=fut_df.index[:len(y_pred)], 
                          columns=df_full.columns)
    pred_df.to_csv(f"{out_dir}/future_pred.csv")
    
    save_metrics = {
        "overall": metrics,
        "L": args.L,
        "s": args.s,
        "steps_ahead": args.steps_ahead,
        "trainlength": args.trainlength,
        "horizon": args.horizon
    }
    safe_json_dump(save_metrics, f"{out_dir}/metrics.json")
    safe_json_dump(vars(args), f"{out_dir}/args.json")
    
    if not args.skip_metrics:
        save_plots(out_dir, fut_df.index[:len(y_pred)], y_true, y_pred)
    
    print(f"\n输出目录: {out_dir}")
    
    return metrics


if __name__ == "__main__":
    main()
