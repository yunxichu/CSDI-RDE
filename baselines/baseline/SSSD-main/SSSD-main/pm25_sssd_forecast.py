# -*- coding: utf-8 -*-
"""PM2.5 Forecasting with SSSD (Structured State Space Diffusion)"""
import os, json, time, random, argparse, datetime, warnings, math, numpy as np, pandas as pd
from tqdm import tqdm
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def ensure_dir(p): os.makedirs(p, exist_ok=True)

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
    steps = steps.float().unsqueeze(-1)
    return torch.cat([torch.sin(steps * emb), torch.cos(steps * emb)], dim=-1)

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
            h = torch.tanh(torch.einsum('dnn,bdn->bdn', self.A, h) + torch.einsum('dn,bd->bdn', self.B, x[:,:,t]))
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
            self.ul.append(nn.ModuleList([nn.Conv1d(H*ex, H, 1)] + [ResBlock(H, ic, dd, drop) for _ in range(nl)]))
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
    t = torch.randint(0, dh["T"], (B,), device='cpu')
    z = torch.randn_like(x)
    z = x * mask + z * (1 - mask)
    ab = dh["Alpha_bar"][t].view(B,1,1).to(dev)
    xt = torch.sqrt(ab) * x + torch.sqrt(1-ab) * z
    t = t.to(dev)
    eps = net(xt, cond, mask, t)
    lm = (1 - mask).bool()
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
    return x

def build_win(data, W, sa=1):
    T, D = data.shape
    X, Y = [], []
    for i in range(T - W - sa + 1):
        X.append(data[i:i+W])
        Y.append(data[i+W+sa-1])
    return np.stack(X), np.stack(Y)

def train(net, X, Y, ep, bs, lr, dev, dh, pat=15):
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    ld = DataLoader(TensorDataset(torch.arange(len(X))), batch_size=bs, shuffle=True)
    best, state, ni = float("inf"), None, 0
    for e in tqdm(range(ep), desc="Train"):
        net.train()
        el = 0
        for (idx,) in ld:
            x = Xt[idx].to(dev).transpose(1,2)
            m = torch.ones_like(x); m[:,:,-1] = 0
            c = x.clone(); c[:,:,-1] = 0
            opt.zero_grad()
            l = train_loss(net, x, c, m, dh, dev)
            l.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            el += l.item() * len(idx)
        el /= len(X)
        if el < best: best, state, ni = el, {k:v.cpu().clone() for k,v in net.state_dict().items()}, 0
        else: ni += 1
        if ni >= pat: break
    if state: net.load_state_dict(state); net.to(dev)
    return best

def forecast(net, hist, fut_true_scaled, hor, W, dev, sc, dh):
    """
    使用真值滑窗预测：每次预测后用真实值更新窗口
    """
    net.eval()
    cur = hist[-W:].copy()
    preds = []
    for i in tqdm(range(hor), desc="Forecast"):
        x = torch.tensor(cur[np.newaxis], dtype=torch.float32, device=dev).transpose(1,2)
        m = torch.ones_like(x); m[:,:,-1] = 0
        c = x.clone(); c[:,:,-1] = 0
        p = sample(net, c, m, dh, dev)[:,:, -1].cpu().numpy()[0]
        preds.append(p)
        # 使用真实值更新窗口
        true_val = fut_true_scaled[i]
        cur = np.vstack([cur[1:], true_val[np.newaxis]])
    return sc.inverse_transform(np.array(preds))

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--imputed_history_path", required=True)
    pa.add_argument("--ground_path", required=True)
    pa.add_argument("--split_ratio", type=float, default=0.5)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--horizon_days", type=float, default=0.0)
    pa.add_argument("--d_model", type=int, default=64)
    pa.add_argument("--n_layers", type=int, default=4)
    pa.add_argument("--dropout", type=float, default=0.1)
    pa.add_argument("--diffusion_steps", type=int, default=100)
    pa.add_argument("--window_size", type=int, default=48)
    pa.add_argument("--epochs", type=int, default=100)
    pa.add_argument("--batch_size", type=int, default=16)
    pa.add_argument("--lr", type=float, default=1e-4)
    pa.add_argument("--device", type=str, default="auto")
    pa.add_argument("--out_dir", type=str, default="")
    a = pa.parse_args()
    set_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else torch.device(a.device)
    
    df = pd.read_csv(a.ground_path, index_col="datetime", parse_dates=True).sort_index()
    sp = int(len(df) * a.split_ratio)
    hist_df, fut_df = df.iloc[:sp], df.iloc[sp:]
    hist = pd.read_csv(a.imputed_history_path, index_col="datetime", parse_dates=True).sort_index().values
    D = hist.shape[1]
    sc = StandardScaler()
    hist_s = sc.fit_transform(hist).astype(np.float32)
    
    hor = len(fut_df)
    if a.horizon_days > 0: hor = int(a.horizon_days * 24)
    hor = min(hor, len(fut_df))
    fut_df = fut_df.iloc[:hor]
    fut_true = fut_df.values
    # 对未来真实值进行标准化
    fut_true_scaled = sc.transform(fut_true).astype(np.float32)
    
    X, Y = build_win(hist_s, a.window_size)
    dh = calc_dh(a.diffusion_steps, 1e-4, 0.02)
    
    net = SSSD(d=a.d_model, nl=a.n_layers, ic=D, oc=D, drop=a.dropout).to(dev)
    print(f"SSSD params: {sum(p.numel() for p in net.parameters()):,}")
    
    print("\n[Training]")
    t0 = time.time()
    best = train(net, X, Y, a.epochs, a.batch_size, a.lr, dev, dh)
    print(f"Done in {time.time()-t0:.1f}s, loss={best:.6f}")
    
    print("\n[Forecasting]")
    t1 = time.time()
    preds = forecast(net, hist_s, fut_true_scaled, hor, a.window_size, dev, sc, dh)
    print(f"Done in {time.time()-t1:.1f}s")
    
    out = a.out_dir or f"./save/pm25_sssd_{a.split_ratio}_{a.seed}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    ensure_dir(out)
    pd.DataFrame(preds, index=fut_df.index, columns=df.columns).to_csv(os.path.join(out, "future_pred.csv"))
    
    m = ~np.isnan(fut_true) & ~np.isnan(preds)
    rmse = np.sqrt(np.mean((fut_true[m] - preds[m])**2)) if m.any() else float('nan')
    mae = np.mean(np.abs(fut_true[m] - preds[m])) if m.any() else float('nan')
    print(f"\nRMSE: {rmse:.4f}, MAE: {mae:.4f}")
    print(f"Output: {out}")

if __name__ == "__main__":
    main()
