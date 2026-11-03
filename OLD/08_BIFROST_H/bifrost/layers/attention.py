"""
CausalAttention — Transformer++ referans dikkati (RoPE + SDPA, GQA, opsiyonel QK-norm
ve kayan pencere). Baseline (B0) ve Heimdall'ın pencere kolu için kullanılır.

Cache: pencere yoksa tüm K/V (O(T)), pencere varsa son W token (O(1)).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norms import RMSNorm


def rope_tables(positions: torch.Tensor, head_dim: int, base: float, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim))
    freqs = torch.outer(positions.float(), inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, Dh]; cos/sin: [T, Dh/2]  (yarım-döndürme düzeni)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class CausalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: Optional[int] = None, head_dim: Optional[int] = None,
                 window: Optional[int] = None, rope_base: float = 10_000.0, qk_norm: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        if num_heads % self.num_kv_heads:
            raise ValueError("num_heads, num_kv_heads'in katı olmalı")
        self.head_dim = head_dim or dim // num_heads
        self.window = window
        self.rope_base = rope_base
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def _qkv(self, x: torch.Tensor, positions: torch.Tensor):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv_proj(x).view(b, t, 2, self.num_kv_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = rope_tables(positions, self.head_dim, self.rope_base, x.dtype)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def _attend(self, q, k, v, mask) -> torch.Tensor:
        groups = self.num_heads // self.num_kv_heads
        if groups > 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        if mask is None:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self._qkv(x, torch.arange(t, device=x.device))
        mask = None
        if self.window is not None and self.window < t:
            i = torch.arange(t, device=x.device)
            dist = i[:, None] - i[None, :]
            mask = (dist >= 0) & (dist < self.window)
        y = self._attend(q, k, v, mask)
        return self.o_proj(y.transpose(1, 2).reshape(b, t, -1))

    def forward_stateful(self, x: torch.Tensor, state: dict) -> Tuple[torch.Tensor, dict]:
        b, t, _ = x.shape
        pos = state["pos"]
        q, k, v = self._qkv(x, torch.arange(pos, pos + t, device=x.device))
        k = torch.cat((state["k"], k), dim=2)
        v = torch.cat((state["v"], v), dim=2)
        i = torch.arange(pos, pos + t, device=x.device)[:, None]
        j = torch.arange(k.size(2), device=x.device)[None, :] + (pos + t - k.size(2))
        mask = (j <= i) if self.window is None else (j <= i) & (i - j < self.window)
        y = self._attend(q, k, v, mask)
        if self.window is not None:
            k, v = k[:, :, -self.window:], v[:, :, -self.window:]
        return self.o_proj(y.transpose(1, 2).reshape(b, t, -1)), {"k": k, "v": v, "pos": pos + t}

    def init_state(self, batch: int, device, dtype) -> dict:
        shape = (batch, self.num_kv_heads, 0, self.head_dim)
        return {"k": torch.zeros(shape, device=device, dtype=dtype),
                "v": torch.zeros(shape, device=device, dtype=dtype), "pos": 0}

    def step(self, x: torch.Tensor, state: dict) -> Tuple[torch.Tensor, dict]:
        b = x.size(0)
        pos = state["pos"]
        q, k, v = self._qkv(x.unsqueeze(1), torch.tensor([pos], device=x.device))
        k = torch.cat((state["k"], k), dim=2)
        v = torch.cat((state["v"], v), dim=2)
        if self.window is not None:
            k, v = k[:, :, -self.window:], v[:, :, -self.window:]
        groups = self.num_heads // self.num_kv_heads
        kk = k.repeat_interleave(groups, dim=1) if groups > 1 else k
        vv = v.repeat_interleave(groups, dim=1) if groups > 1 else v
        y = F.scaled_dot_product_attention(q, kk, vv)
        return self.o_proj(y.transpose(1, 2).reshape(b, -1)), {"k": k, "v": v, "pos": pos + 1}
