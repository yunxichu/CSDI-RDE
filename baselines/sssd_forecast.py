# -*- coding: utf-8 -*-
"""
通用 SSSD (Structured State Space Diffusion) 预测脚本
============================================================
基于 PM2.5 版本，支持所有数据集

示例：
  # PM2.5
  python baselines/sssd_forecast.py \
    --dataset pm25 \
    --imputed_history_path ./save/pm25_history_imputed_split0.5_seed42_20260324_154400/history_imputed.csv \
    --ground_path ./data/pm25/Code/STMVL/SampleData/pm25_ground.txt \
    --split_ratio 0.5 --horizon_days 1 \
    --d_model 64 --n_layers 4 --diffusion_steps 100 \
    --window_size 48 --epochs 100 --batch_size 16 --lr 1e-4 --seed 42

  # Lorenz96
  python baselines/sssd_forecast.py \
    --dataset lorenz96 \
    --data_path ./lorenz96_rde_delay/results/imputed_100_20260323_192045.csv \
    --ground_path ./lorenz96_rde_delay/results/gt_100_20260323_192045.csv \
    --trainlength 60 --horizon_steps 40 \
    --d_model 64 --n_layers 4 --diffusion_steps 50 \
    --window_size 20 --epochs 100 --batch_size 16 --lr 1e-4 --seed 42

  # EEG
  python baselines/sssd_forecast.py \
    --dataset eeg \
    --imputed_path ./save/eeg_csdi_imputed.npy \
    --ground_path ./data/eeg/eeg_ground.npy \
    --history_timesteps 100 --horizon_steps 24 \
    --target_dims 0,1,2 \
    --d_model 64 --n_layers 4 --diffusion_steps 50 \
    --window_size 48 --epochs 100 --batch_size 16 --lr 1e-4 --seed 42
"""

import os, json, time, random, argparse, datetime, warnings, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def compute_metrics(y_true, y_pred):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0: return {"rmse": np.nan, "mae": np.nan}
    diff = y_true[mask] - y_pred[mask]
    return {"rmse": float(np.sqrt(np.mean(diff**2))), "mae": float(np.mean(np.abs(diff)))}


def calc_dh(T, b0, bT):
    Beta = torch.linspace(b0, bT, T)
    Alpha = 1 - Beta
    Alpha_bar = Alpha.clone()
    for t in range(1, T): Alpha_bar[t] *= Alpha_bar[t-1]
    Sigma = torch.sqrt(Beta * (1 - torch.cat([torch.ones(1), Alpha_bar[:-1]])) / (1 - Alpha_bar))
    return {"T": T, "Alpha": Alpha, "Alpha_bar": Alpha_bar, "Sigma": Sigma}

def t_embed(steps, dim):
    half = dim // 2
    emb = torch.exp(torch.arange(half, device=steps.device) * -np.log(10000) / (half - 1))
    return torch.cat([torch.sin(steps.float().unsqueeze(-1) * emb),
                      torch.cos(steps.float().unsqueeze(-1) * emb)], dim=-1)


class S4Layer(nn.Module):
    def __init__(self, d, ds=64, drop=0.0):
        super().__init__()
        self.A = nn.Parameter(torch.randn(d, ds, ds) * 0.01)
        self.B = nn.Parameter(torch.randn(d, ds) * 0.01)
        self.C = nn.Parameter(torch.randn(d, ds) * 0.01)
        self.D = nn.Parameter(torch.ones(d))
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, D, L = x.shape
        h = torch.zeros(B, D, self.A.shape[1], device=x.device)
        out = []
        for t in range(L):
            h = torch.tanh(torch.einsum('dnn,bdn->bdn', self.A, h) +
                           torch.einsum('dn,bd->bdn', self.B, x[:,:,t]))
            out.append(torch.einsum('dn,bdn->bd', self.C, h) + self.D * x[:,:,t])
        return self.drop(self.proj(torch.stack(out, -1).transpose(1,2)).transpose(1,2))


class ResBlock(nn.Module):
    def __init__(self, d, ic, dd, drop=0.0):
        super().__init__()
        self.s4 = S4Layer(d, drop=drop)
        self.norm = nn.LayerNorm(d)
        self.fc_t = nn.Linear(dd, d)
        self.cond = nn.Conv1d(2*ic, d, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x, cond, te):
        z = x + self.fc_t(te).unsqueeze(2)
        z = self.norm(z.transpose(1,2)).transpose(1,2)
        return self.drop(self.s4(z) + self.cond(cond)) + x


class SSSD(nn.Module):
    def __init__(self, d=64, nl=4, pl=[2,2], ex=2, ic=1, oc=1, dd=512, drop=0.0):
        super().__init__()
        H = d
        self.dl = nn.ModuleList()
        for p in pl:
            for _ in range(nl): self.dl.append(ResBlock(H, ic, dd, drop))
            self.dl.append(nn.Sequential(nn.Conv1d(H, H*ex, 1), nn.ReLU()))
            H *= ex
        self.cl = nn.ModuleList([ResBlock(H, ic, dd, drop) for _ in range(nl)])
        self.ul = nn.ModuleList()
        for p in pl[::-1]:
            H //= ex
            self.ul.append(nn.ModuleList([nn.Conv1d(H*ex, H, 1)] +
                          [ResBlock(H, ic, dd, drop) for _ in range(nl)]))
        self.norm = nn.LayerNorm(H)
        self.init = nn.Sequential(nn.Conv1d(ic, d, 1), nn.ReLU())
        self.final = nn.Sequential(nn.Conv1d(d, d, 1), nn.ReLU(), nn.Conv1d(d, oc, 1))
        self.fc_t = nn.Sequential(nn.Linear(128, dd), nn.ReLU(), nn.Linear(dd, dd), nn.ReLU())

    def forward(self, noise, cond, mask, steps):
        cond = torch.cat([cond * mask, mask.float()], 1)
        te = self.fc_t(t_embed(steps, 128))
        x = self.init(noise)
        outs = [x]
        for L in self.dl:
            if isinstance(L, ResBlock): x = L(x, cond, te)
            else: x = L(x)
            outs.append(x)
        for L in self.cl: x = L(x, cond, te)
        x = x + outs.pop()
        for block in self.ul:
            for L in block:
                if isinstance(L, ResBlock): x = L(x, cond, te)
                else: x = L(x)
            x = x + outs.pop()
        return self.final(self.norm(x.transpose(1,2)).transpose(1,2))


def train_loss(net, x, cond, mask, dh, dev):
    B = x.shape[0]
    T = dh["T"]
    t = torch.randint(0, T, (B,), device=dev)
    z = torch.randn_like(x)
    ab = dh["Alpha_bar"].to(dev)[t].view(B,1,1)
    xt = torch.sqrt(ab) * x + torch.sqrt(1-ab) * z
    eps = net(xt, cond, mask, t)
    lm = mask.bool()
    return F.mse_loss(eps[lm], z[lm])


def sample(net, cond, mask, dh, dev):
    B, C, L = cond.shape
    x = torch.randn(B, C, L, device=dev)
    Alpha = dh["Alpha"].to(dev)
    Alpha_bar = dh["Alpha_bar"].to(dev)
    Sigma = dh["Sigma"].to(dev)
    with torch.no_grad():
        for t in range(dh["T"]-1, -1, -1):
            x = x * (1 - mask) + cond * mask
            eps = net(x, cond, mask, torch.full((B,), t, device=dev))
            a, ab, s = Alpha[t], Alpha_bar[t], Sigma[t]
            x = (x - (1-a)/torch.sqrt(1-ab)*eps) / torch.sqrt(a)
            if t > 0: x = x + s * torch.randn_like(x)
            x = x * (1 - mask) + cond * mask
    return x


def build_win(data, W, sa=1):
    T, D = data.shape
    X, Y = [], []
    for i in range(T - W - sa + 1):
        X.append(data[i:i+W])
        Y.append(data[i+W+sa-1])
    return np.stack(X), np.stack(Y)


def load_pm25_data(args):
    df_full = pd.read_csv(args.ground_path, index_col="datetime", parse_dates=True).sort_index()
    sp = int(len(df_full) * args.split_ratio)
    hist_df, fut_df = df_full.iloc[:sp], df_full.iloc[sp:]
    hist = pd.read_csv(args.imputed_history_path, index_col="datetime", parse_dates=True).sort_index().values
    D = hist.shape[1]
    hor = len(fut_df)
    if args.horizon_days > 0: hor = int(args.horizon_days * 24)
    elif args.horizon_steps > 0: hor = int(args.horizon_steps)
    hor = min(hor, len(fut_df))
    return hist.astype(np.float64), fut_df.values[:hor].astype(np.float64), list(df_full.columns), hor, fut_df.index[:hor]


def load_npy_data(args):
    data = np.load(args.imputed_path)
    ground = np.load(args.ground_path)
    if data.ndim == 3: data = data.reshape(-1, data.shape[-1])
    if ground.ndim == 3: ground = ground.reshape(-1, ground.shape[-1])
    if args.target_dims:
        dims = [int(x) for x in args.target_dims.split(',')]
        data = data[:, dims]; ground = ground[:, dims]
    hist_len = args.history_timesteps
    return data[:hist_len].astype(np.float64), ground[hist_len:hist_len+args.horizon_steps].astype(np.float64), \
           [f"dim_{i}" for i in range(data.shape[1])], args.horizon_steps, None


def load_csv_data(args):
    gt = np.loadtxt(args.ground_path, delimiter=',')
    data = np.loadtxt(args.data_path, delimiter=',') if args.data_path and os.path.exists(args.data_path) else gt.copy()
    if data.ndim == 1: data = data.reshape(-1, 1)
    if gt.ndim == 1: gt = gt.reshape(-1, 1)
    hist_len = args.trainlength
    horizon = min(args.horizon_steps, len(gt) - hist_len)
    return data[:hist_len].astype(np.float64), gt[hist_len:hist_len+horizon].astype(np.float64), \
           [f"dim_{i}" for i in range(data.shape[1])], horizon, None


def main():
    pa = argparse.ArgumentParser(description="SSSD Forecast (Universal)")
    pa.add_argument("--dataset", type=str, required=True,
                    choices=["pm25", "eeg", "lorenz63", "lorenz96"])
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--imputed_history_path", type=str, default="")
    pa.add_argument("--ground_path", type=str, required=True)
    pa.add_argument("--split_ratio", type=float, default=0.5)
    pa.add_argument("--horizon_days", type=float, default=0.0)
    pa.add_argument("--imputed_path", type=str, default="")
    pa.add_argument("--data_path", type=str, default="")
    pa.add_argument("--history_timesteps", type=int, default=100)
    pa.add_argument("--horizon_steps", type=int, default=24)
    pa.add_argument("--trainlength", type=int, default=60)
    pa.add_argument("--target_dims", type=str, default=None)
    pa.add_argument("--d_model", type=int, default=64)
    pa.add_argument("--n_layers", type=int, default=4)
    pa.add_argument("--dropout", type=float, default=0.1)
    pa.add_argument("--diffusion_steps", type=int, default=100)
    pa.add_argument("--window_size", type=int, default=48)
    pa.add_argument("--epochs", type=int, default=100)
    pa.add_argument("--batch_size", type=int, default=16)
    pa.add_argument("--lr", type=float, default=1e-4)
    pa.add_argument("--patience", type=int, default=15)
    pa.add_argument("--device", type=str, default="auto")
    pa.add_argument("--out_dir", type=str, default="")
    pa.add_argument("--plot_dim", type=int, default=0)
    a = pa.parse_args()

    set_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else torch.device(a.device)
    out_dir = a.out_dir or f"./save/{a.dataset}_sssd_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    ensure_dir(out_dir)

    if a.dataset == "pm25":
        history, fut_true, columns, horizon, fut_index = load_pm25_data(a)
    elif a.dataset == "eeg":
        history, fut_true, columns, horizon, fut_index = load_npy_data(a)
    else:
        history, fut_true, columns, horizon, fut_index = load_csv_data(a)

    D = history.shape[1]
    print(f"\n{'='*70}\n{a.dataset.upper()} 预测 - SSSD\n{'='*70}")
    print(f"history: {history.shape}, future: {fut_true.shape}, horizon: {horizon}")

    sc = StandardScaler()
    hist_s = sc.fit_transform(history).astype(np.float32)
    fut_true_scaled = sc.transform(fut_true).astype(np.float32)

    X, Y = build_win(hist_s, a.window_size)
    dh = calc_dh(a.diffusion_steps, 1e-4, 0.02)

    net = SSSD(d=a.d_model, nl=a.n_layers, ic=D, oc=D, drop=a.dropout).to(dev)
    print(f"SSSD params: {sum(p.numel() for p in net.parameters()):,}")

    print("\n[Training]")
    t0 = time.time()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    Xt = torch.tensor(X); Yt = torch.tensor(Y)
    ld = DataLoader(TensorDataset(torch.arange(len(X))), batch_size=a.batch_size, shuffle=True)
    best, state, ni = float("inf"), None, 0

    for e in tqdm(range(a.epochs), desc="Train"):
        net.train()
        el = 0
        for (idx,) in ld:
            x = Xt[idx].to(dev).transpose(1,2)
            mask = torch.zeros_like(x); mask[:,:,-1] = 1
            cond = x.clone(); cond[:,:,-1] = 0
            opt.zero_grad()
            l = train_loss(net, x, cond, mask, dh, dev)
            l.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            el += l.item() * len(idx)
        el /= len(X)
        if el < best: best, state, ni = el, {k:v.cpu().clone() for k,v in net.state_dict().items()}, 0
        else: ni += 1
        if ni >= a.patience: break

    if state: net.load_state_dict(state); net.to(dev)
    print(f"Done in {time.time()-t0:.1f}s, best loss={best:.6f}")
    torch.save(net.state_dict(), os.path.join(out_dir, "model.pt"))

    print("\n[Forecasting with true value filling]")
    t1 = time.time()
    net.eval()
    cur = hist_s[-a.window_size:].copy()
    preds = []
    for i in tqdm(range(horizon), desc="Forecast"):
        x = torch.tensor(cur[np.newaxis], dtype=torch.float32, device=dev).transpose(1,2)
        mask = torch.zeros_like(x); mask[:,:,-1] = 1
        cond = x.clone(); cond[:,:,-1] = 0
        with torch.no_grad():
            pred = sample(net, cond, mask, dh, dev)[:,:, -1].cpu().numpy()[0]
        preds.append(pred)
        cur = np.vstack([cur[1:], fut_true_scaled[i][np.newaxis]])

    preds = sc.inverse_transform(np.array(preds))
    print(f"Done in {time.time()-t1:.1f}s")

    if fut_index is not None:
        pd.DataFrame(preds, index=fut_index, columns=columns).to_csv(os.path.join(out_dir, "future_pred.csv"))
    else:
        pd.DataFrame(preds, columns=columns).to_csv(os.path.join(out_dir, "future_pred.csv"))
    np.save(os.path.join(out_dir, "future_pred.npy"), preds)

    overall = compute_metrics(fut_true, preds)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"overall": overall, "horizon": horizon}, f, indent=2)

    d = max(0, min(int(a.plot_dim), D - 1))
    plt.figure(figsize=(14, 5))
    plt.plot(fut_true[:, d], label=f"True (dim {d})", color="steelblue")
    plt.plot(preds[:, d], label=f"SSSD (dim {d})", color="tomato")
    plt.xlabel("Time"); plt.ylabel("Value")
    plt.title(f"SSSD Forecast ({a.dataset}, dim {d})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"plot_forecast_dim{d}.png"), dpi=150)
    plt.close()

    print(f"\nRMSE: {overall['rmse']:.4f}, MAE: {overall['mae']:.4f}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
