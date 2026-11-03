"""
Kuzgun'un düşük seviye işlemleri (GPU/ROCm odaklı, torch.compile dostu).

- RMSNorm (FP32 indirgeme)
- causal_short_conv: kaydırmalı toplam ile depthwise nedensel conv (compile tek çekirdeğe birleştirir)
- rope: uzun bağlamda da hassas (açılar float64'te hesaplanır)
- sliding_window_attention: blok-yerel SDPA (her yerde) veya flex_attention (varsa)
- chunk_gated_delta_rule: Muninn'in chunk-paralel eğitimi (UT dönüşümü, FP32)
- gated_delta_step: tek token güncellemesi (üretim)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # PyTorch >= 2.5
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    HAS_FLEX = True
except Exception:  # pragma: no cover
    HAS_FLEX = False


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


# --------------------------------------------------------------------------- kısa conv
def causal_short_conv(x: torch.Tensor, weight: torch.Tensor, prefix: Optional[torch.Tensor] = None) -> torch.Tensor:
    """x: [B, T, C], weight: [C, K] (weight[:, -1] = mevcut token). prefix: [B, K-1, C] önceki girdiler.

    y_t = Σ_j weight[:, K-1-j] · x_{t-j}. Kaydırmalı toplam; torch.compile bunu tek çekirdeğe birleştirir.
    """
    k = weight.size(-1)
    t = x.size(1)
    if prefix is None:
        full = F.pad(x, (0, 0, k - 1, 0))
    else:
        full = torch.cat((prefix.to(x.dtype), x), dim=1)
    w = weight.to(x.dtype)
    y = full[:, k - 1:k - 1 + t] * w[:, k - 1]
    for j in range(1, k):
        y = y + full[:, k - 1 - j:k - 1 - j + t] * w[:, k - 1 - j]
    return y


# --------------------------------------------------------------------------- RoPE
def rope_cos_sin(positions: torch.Tensor, dim: int, base: float, dtype: torch.dtype):
    """positions: [T] (int). Açılar CPU'da float64: 1M+ pozisyonda da hassas; DirectML float64 desteklemez."""
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    ang = positions.cpu().to(torch.float64)[:, None] * inv[None, :]
    return ang.cos().to(positions.device, dtype), ang.sin().to(positions.device, dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D], cos/sin: [T, D/2]  (yarım-döndürme düzeni)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


# --------------------------------------------------------------------------- pencere dikkati
def _block_local_sdpa(q, k, v, window: int) -> torch.Tensor:
    """Her W'lik blok yalnız kendisine ve bir önceki bloğa bakar: O(T·2W). q önceden ölçeklenmiş."""
    b, h, t, d = q.shape
    w = window
    pad = (-t) % w
    if pad:
        q, k, v = (F.pad(z, (0, 0, 0, pad)) for z in (q, k, v))
    n = (t + pad) // w
    qb, kb, vb = (z.view(b, h, n, w, d) for z in (q, k, v))
    kk = torch.cat((F.pad(kb, (0, 0, 0, 0, 1, 0))[:, :, :-1], kb), dim=3)
    vv = torch.cat((F.pad(vb, (0, 0, 0, 0, 1, 0))[:, :, :-1], vb), dim=3)
    i = torch.arange(w, device=q.device)[:, None]
    j = torch.arange(2 * w, device=q.device)[None, :]
    band = (j > i) & (j <= i + w)                           # uzaklık = W + i − j ∈ [0, W)
    first = band & (j >= w)                                 # ilk bloğun "önceki bloğu" yok
    mask = torch.cat((first[None], band[None].expand(n - 1, w, 2 * w)), dim=0) if n > 1 else first[None]
    # 4-D düzen [B, H·N, W, D]: ROCm/CUDA'nın hızlı SDPA çekirdekleri 4-D girdi ister
    out = F.scaled_dot_product_attention(qb.reshape(b, h * n, w, d), kk.reshape(b, h * n, 2 * w, d),
                                         vv.reshape(b, h * n, 2 * w, d), attn_mask=mask.repeat(h, 1, 1), scale=1.0)
    return out.reshape(b, h, n * w, d)[:, :, :t]


_FLEX_MASKS = {}


@torch.compiler.disable
def _flex_mask(t: int, window: int, device):
    key = (t, window, str(device))
    if key not in _FLEX_MASKS:
        def mask_mod(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (q_idx - kv_idx < window)
        _FLEX_MASKS[key] = create_block_mask(mask_mod, B=None, H=None, Q_LEN=t, KV_LEN=t, device=device)
    return _FLEX_MASKS[key]


def sliding_window_attention(q, k, v, window: int, backend: str = "sdpa") -> torch.Tensor:
    """Nedensel pencere dikkati; her sorgu kendisi dahil son `window` anahtara bakar. [B,H,T,D]."""
    t = q.size(2)
    if t <= window:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
    if backend == "flex" and HAS_FLEX and q.is_cuda:
        return flex_attention(q, k, v, block_mask=_flex_mask(t, window, q.device), scale=1.0)
    return _block_local_sdpa(q, k, v, window)


# --------------------------------------------------------------------------- Muninn (delta hafıza)
def _unit_lower_inverse(a_strict: torch.Tensor) -> torch.Tensor:
    """(I + A)^{-1}, A kesin alt üçgen. Kararlı üçgen çözüm; destek yoksa ileri yerine koyma."""
    c = a_strict.size(-1)
    eye = torch.eye(c, dtype=a_strict.dtype, device=a_strict.device)
    try:
        return torch.linalg.solve_triangular(eye + a_strict, eye.expand_as(a_strict).contiguous(),
                                             upper=False, unitriangular=True)
    except (RuntimeError, NotImplementedError):
        inv = -a_strict.clone()
        for i in range(1, c):
            inv[..., i, :i] = inv[..., i, :i] + (inv[..., i, :, None] * inv[..., :, :i]).sum(-2)
        return inv + eye


def chunk_gated_delta_rule(q, k, v, g, beta, chunk_size: int = 64,
                           initial_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Kapılı delta kuralı, chunk-paralel. Token-token referansla birebir aynı sonucu verir.

    q, k: [B,H,T,Dk] (k L2-normalize, q ölçeklenmiş), v: [B,H,T,Dv],
    g: [B,H,T] log α (≤ 0), beta: [B,H,T]. Hesap FP32'de yapılır (autocast kapalı).
    Döndürür: o [B,H,T,Dv] (FP32), son durum S [B,H,Dk,Dv] (FP32).
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, g, beta = (z.float() for z in (q, k, v, g, beta))
        b, h, t, dk = k.shape
        dv = v.shape[-1]
        c = chunk_size
        pad = (-t) % c
        if pad:  # β = 0 ve g = 0 dolgu durumu değiştirmez
            q, k, v = (F.pad(z, (0, 0, 0, pad)) for z in (q, k, v))
            g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
        n = (t + pad) // c
        q, k, v = (z.view(b, h, n, c, -1) for z in (q, k, v))
        g, beta = g.view(b, h, n, c), beta.view(b, h, n, c)

        g_cum = g.cumsum(-1)
        decay = (g_cum[..., :, None] - g_cum[..., None, :]).tril().exp().tril()  # exp(g_i − g_j), i ≥ j
        k_beta = k * beta[..., None]
        w_inv = _unit_lower_inverse((k_beta @ k.transpose(-1, -2) * decay).tril(-1))
        u = w_inv @ (v * beta[..., None])
        w = w_inv @ (k_beta * g_cum.exp()[..., None])
        attn = q @ k.transpose(-1, -2) * decay                                   # chunk içi (i ≥ j)
        q_decay = q * g_cum.exp()[..., None]
        k_tail = k * (g_cum[..., -1:] - g_cum).exp()[..., None]
        chunk_decay = g_cum[..., -1].exp()[..., None, None]

        state = initial_state.float() if initial_state is not None else q.new_zeros(b, h, dk, dv)
        outs = []
        for i in range(n):
            v_new = u[:, :, i] - w[:, :, i] @ state
            outs.append(q_decay[:, :, i] @ state + attn[:, :, i] @ v_new)
            state = state * chunk_decay[:, :, i] + k_tail[:, :, i].transpose(-1, -2) @ v_new
        o = torch.stack(outs, dim=2).view(b, h, n * c, dv)[:, :, :t]
    return o, state


def gated_delta_step(q, k, v, g, beta, state):
    """Tek token: q, k: [B,H,Dk], v: [B,H,Dv], g/beta: [B,H], state: [B,H,Dk,Dv] (FP32)."""
    q, k, v, g, beta = (z.float() for z in (q, k, v, g, beta))
    state = state * g.exp()[..., None, None]
    delta = (v - torch.einsum("bhk,bhkv->bhv", k, state)) * beta[..., None]
    state = state + k[..., :, None] * delta[..., None, :]
    return torch.einsum("bhk,bhkv->bhv", q, state), state
