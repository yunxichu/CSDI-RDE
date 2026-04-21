"""Minimal faithful reimplementation of Panda-72M arch.

Matches HF state_dict at ``GilpinLab/panda-72M``. Covers:
  - DynamicsEmbedder: RFF of patches with 2 harmonic orders + raw passthrough.
  - Dual attention per layer: temporal (RoPE 75%) + channel (no RoPE).
  - Pre-RMSNorm × 3 per block; ffn_dim=d_model.
  - Head: mean-pool over patches + linear to prediction_length.

Weight key layout (280 tensors):
  model.encoder.embedder.{freq_weights [16,188], freq_biases [1,1,1,188], projection.weight [768,768]}
  model.encoder.layers.{0..11}.{temporal_self_attn,channel_self_attn}.{q,k,v,out}_proj.{weight,bias}
  model.encoder.layers.{0..11}.norm_sublayer{1,2,3}.weight  (RMSNorm scale-only)
  model.encoder.layers.{0..11}.ff.{0,3}.{weight,bias}  (Linear-GELU-Dropout-Linear, ffn_dim=768)
  head.projection.weight  [128,768]

No distribution head; ``loss="mse"`` → deterministic point forecast. The 100 parallel samples
advertised in config are never realised by these weights.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PandaConfig:
    d_model: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    ffn_dim: int = 768
    context_length: int = 512
    prediction_length: int = 128
    patch_length: int = 16
    patch_stride: int = 16
    num_poly_feats: int = 188
    poly_degrees: int = 2
    num_rff: int = 376   # = 2 * num_poly_feats (cos+sin) — not used directly, sanity only
    rope_percent: float = 0.75
    max_wavelength: int = 500
    norm_eps: float = 1e-5
    scaling: str = "std"

    @classmethod
    def from_json(cls, path: str | Path) -> "PandaConfig":
        d = json.loads(Path(path).read_text())
        known = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**known)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


def _build_rope_cache(seq_len: int, head_dim: int, pct: float, base: float, device, dtype):
    """Build cos/sin cache covering the first ``rope_dim`` of each head.

    ``rope_dim`` is the even-sized prefix equal to ``floor(pct·head_dim/2)·2``.
    """
    rope_dim = int(head_dim * pct) // 2 * 2
    if rope_dim == 0:
        return None
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin, rope_dim


def _apply_rope(x: torch.Tensor, rope_cache) -> torch.Tensor:
    """x: [*, S, D] — rotate the first rope_dim channels by position-dependent θ."""
    cos, sin, rope_dim = rope_cache
    x_rot = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    x_rot = x_rot.reshape(*x_rot.shape[:-1], rope_dim // 2, 2)
    x1, x2 = x_rot[..., 0], x_rot[..., 1]
    # broadcast cos/sin over leading dims
    shape = [1] * (x.ndim - 2) + [cos.shape[0], cos.shape[1]]
    c = cos.view(shape); s = sin.view(shape)
    y1 = x1 * c - x2 * s
    y2 = x1 * s + x2 * c
    y = torch.stack([y1, y2], dim=-1).flatten(-2)
    return torch.cat([y, x_pass], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, use_rope: bool, rope_pct: float, rope_base: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.use_rope = use_rope
        self.rope_pct = rope_pct
        self.rope_base = rope_base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B', S, D]
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # [B,H,S,Hd]
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        if self.use_rope:
            rope = _build_rope_cache(S, self.head_dim, self.rope_pct, self.rope_base, x.device, x.dtype)
            if rope is not None:
                q = _apply_rope(q, rope)
                k = _apply_rope(k, rope)
        # scaled dot-product, no mask (bidirectional encoder)
        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)  # [B,H,S,Hd]
        attn = attn.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(attn)


class DynamicsEmbedder(nn.Module):
    """RFF(x^k @ W + b) for k = 1..poly_degrees, concatenated with raw patch, then linearly projected.

    Input :  x_patched [B, C, P, patch_len]
    Output:  h          [B, C, P, d_model]

    The projection matrix is [d_model, d_model] (no bias) — so the fused feature dim
    (raw + 2·poly·num_poly_feats) must equal d_model. For the shipped config:
        patch_len + 2 · poly_degrees · num_poly_feats = 16 + 2·2·188 = 768 = d_model ✓
    """

    def __init__(self, patch_len: int, d_model: int, num_poly_feats: int, poly_degrees: int):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.num_poly_feats = num_poly_feats
        self.poly_degrees = poly_degrees
        self.freq_weights = nn.Parameter(torch.randn(patch_len, num_poly_feats))
        self.freq_biases = nn.Parameter(torch.zeros(1, 1, 1, num_poly_feats))
        self.projection = nn.Linear(d_model, d_model, bias=False)
        fused_dim = patch_len + 2 * poly_degrees * num_poly_feats
        assert fused_dim == d_model, f"fused dim {fused_dim} != d_model {d_model}"

    def forward(self, x_patched: torch.Tensor) -> torch.Tensor:
        # z = x W + b   [B, C, P, num_poly_feats]
        z = torch.einsum("bcpl,lf->bcpf", x_patched, self.freq_weights) + self.freq_biases
        features = [x_patched]
        for k in range(1, self.poly_degrees + 1):
            zk = k * z
            features.append(torch.cos(zk))
            features.append(torch.sin(zk))
        fused = torch.cat(features, dim=-1)  # [B, C, P, d_model]
        return self.projection(fused)


class PandaLayer(nn.Module):
    def __init__(self, cfg: PandaConfig):
        super().__init__()
        self.norm_sublayer1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.norm_sublayer2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.norm_sublayer3 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.temporal_self_attn = MultiHeadAttention(
            cfg.d_model, cfg.num_attention_heads, use_rope=True,
            rope_pct=cfg.rope_percent, rope_base=cfg.max_wavelength,
        )
        self.channel_self_attn = MultiHeadAttention(
            cfg.d_model, cfg.num_attention_heads, use_rope=False,
            rope_pct=0.0, rope_base=1.0,
        )
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(cfg.ffn_dim, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, P, D]
        B, C, P, D = x.shape

        # Temporal attention over patches
        h = self.norm_sublayer1(x)
        h_t = h.reshape(B * C, P, D)
        x = x + self.temporal_self_attn(h_t).view(B, C, P, D)

        # Channel attention over channels
        h = self.norm_sublayer2(x)
        h_c = h.permute(0, 2, 1, 3).reshape(B * P, C, D)
        x = x + self.channel_self_attn(h_c).view(B, P, C, D).permute(0, 2, 1, 3).contiguous()

        # Feed-forward
        h = self.norm_sublayer3(x)
        x = x + self.ff(h)
        return x


def _sincos_positional(num_patches: int, d_model: int, device, dtype):
    """Standard Transformer sincos PE over patches: [P, d_model]."""
    pos = torch.arange(num_patches, device=device, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
    inv_freq = torch.exp(-math.log(10000.0) * i / d_model)
    ang = pos * inv_freq
    pe = torch.zeros(num_patches, d_model, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang)
    return pe.to(dtype)


class PandaEncoder(nn.Module):
    def __init__(self, cfg: PandaConfig):
        super().__init__()
        self.embedder = DynamicsEmbedder(cfg.patch_length, cfg.d_model, cfg.num_poly_feats, cfg.poly_degrees)
        self.layers = nn.ModuleList([PandaLayer(cfg) for _ in range(cfg.num_hidden_layers)])

    def forward(self, x_patched: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x_patched)
        # Add sincos positional encoding over patch dim (config: positional_encoding_type="sincos")
        P = x.shape[2]
        pe = _sincos_positional(P, x.shape[-1], x.device, x.dtype)  # [P, D]
        x = x + pe.view(1, 1, P, -1)
        for layer in self.layers:
            x = layer(x)
        return x


class PandaHead(nn.Module):
    """Mean-pool over patches, linear to prediction_length."""

    def __init__(self, d_model: int, prediction_length: int):
        super().__init__()
        self.projection = nn.Linear(d_model, prediction_length, bias=False)
        self.prediction_length = prediction_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, P, D] → mean over P → [B, C, D]
        pooled = x.mean(dim=2)
        return self.projection(pooled)  # [B, C, H]


class PandaModel(nn.Module):
    """Wraps encoder + head; exposes ``forecast(context)`` in the user API.

    ``context`` may be shorter than ``context_length``; we left-pad with the first observed value
    (Panda-style) and patch on the right.
    """

    def __init__(self, cfg: PandaConfig):
        super().__init__()
        self.cfg = cfg
        self.model = nn.ModuleDict({"encoder": PandaEncoder(cfg)})
        self.head = PandaHead(cfg.d_model, cfg.prediction_length)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "PandaModel":
        model_dir = Path(model_dir)
        cfg = PandaConfig.from_json(model_dir / "config.json")
        m = cls(cfg)
        from safetensors.torch import load_file
        sd = load_file(str(model_dir / "model.safetensors"))
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected keys: {unexpected[:5]}")
        for k in missing:
            if "freq_weights" in k or "freq_biases" in k or "projection.weight" in k or "norm" in k or "ff." in k or "_attn" in k:
                raise RuntimeError(f"critical weight missing: {k}")
        return m

    @torch.no_grad()
    def forecast(self, context: torch.Tensor) -> torch.Tensor:
        """context: [B, C, L] — any L. Returns [B, C, prediction_length] (de-standardised)."""
        cfg = self.cfg
        B, C, L = context.shape

        # (1) Take the last ``context_length`` steps; left-pad with first value if shorter.
        if L >= cfg.context_length:
            x = context[:, :, -cfg.context_length:]
        else:
            pad_len = cfg.context_length - L
            pad = context[:, :, :1].expand(B, C, pad_len)
            x = torch.cat([pad, context], dim=-1)

        # (2) Per-instance-per-channel std scaling.
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-5)
        x_n = (x - mean) / std

        # (3) Patch: [B, C, P=L/stride, patch_length].
        p_len, p_stride = cfg.patch_length, cfg.patch_stride
        P = cfg.context_length // p_stride
        x_p = x_n.unfold(-1, p_len, p_stride)  # [B, C, P, patch_len]
        assert x_p.shape[2] == P, f"patch count {x_p.shape[2]} != {P}"

        # (4) Encoder + head.
        enc = self.model["encoder"](x_p)   # [B, C, P, d_model]
        y_n = self.head(enc)               # [B, C, prediction_length]

        # (5) De-standardise.
        return y_n * std + mean


def autoregressive_forecast(
    model: PandaModel, context: torch.Tensor, horizon: int, block: Optional[int] = None,
) -> torch.Tensor:
    """Repeat ``model.forecast`` to reach arbitrary horizon ≥ prediction_length.

    Each block takes the *latest* context window (sliding) and emits ``prediction_length``
    steps, which are appended to the context before the next block.
    """
    pred_len = model.cfg.prediction_length
    block = block or pred_len
    assert block <= pred_len
    out = []
    ctx = context
    n_done = 0
    while n_done < horizon:
        y = model.forecast(ctx)  # [B, C, pred_len]
        take = min(block, horizon - n_done)
        out.append(y[:, :, :take])
        ctx = torch.cat([ctx, y[:, :, :take]], dim=-1)
        n_done += take
    return torch.cat(out, dim=-1)
