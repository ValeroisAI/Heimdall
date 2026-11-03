"""
Kuzgun — pencere dikkati (Huginn) + delta hafıza (Muninn), tek katmanda paralel.

Odin'in iki kuzgunu: Huginn "düşünce", Muninn "hafıza".

    q, k, v = SiLU(ShortConv(W_qkv x))            ortak anahtarlar (Canon-B conv)
    q̂, k̂    = L2Norm(q), L2Norm(k)

    Huginn : son W token üzerinde tam softmax (RoPE, kafa başına öğrenilen sıcaklık)
             -> yakın geçmiş TAM token hassasiyetiyle, maliyet O(T·W), cache O(W)
    Muninn : kapılı delta kuralı  S = α S + β k̂ (v − α Sᵀk̂)ᵀ,  o = Sᵀ q̂
             -> pencerenin ötesindeki tüm geçmiş sabit boyutlu durumda, cache O(1)
             coupled_decay: log α ← β·log α — "yazmıyorsan unutma" (GCAM-v3'teki bağlı unutma
             fikri). Önemsiz tokenler (β≈0) hafızayı aşındırmaz; 1M token boyunca bilgi korunur.

    o = σ(m_H)·RMSNorm(o_Huginn) + σ(m_M)·RMSNorm(o_Muninn)     (kafa ve token başına kapı)
    y = W_o( o ⊙ SiLU(W_g x) )

İki kol aynı q̂/k̂/v'yi kullanır: pencerede aranan anahtar, hafızada aranan anahtarla
aynıdır; bilgi pencereden çıkınca aynı anahtarla hafızadan bulunur. Global dikkat yoktur.

branches: "both" (Kuzgun), "window" (yalnız Huginn), "memory" (yalnız Muninn) — ablation için.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import apply_rope, rope_tables
from .csl import ShortConv
from .mimir import gated_delta_chunk
from .norms import RMSNorm


def local_window_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int) -> torch.Tensor:
    """Nedensel pencere dikkati: her sorgu kendisi dahil son `window` anahtara bakar.

    q, k, v: [B, H, T, D]; q önceden ölçeklenmiş olmalı (scale=1 kullanılır).
    Blok-yerel uygulama: her W'lik blok yalnız kendisine ve bir önceki bloğa bakar -> O(T·2W).
    """
    b, h, t, d = q.shape
    w = window
    pad = (-t) % w
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    n = (t + pad) // w
    qb, kb, vb = (x.view(b, h, n, w, d) for x in (q, k, v))
    kk = torch.cat((F.pad(kb, (0, 0, 0, 0, 1, 0))[:, :, :-1], kb), dim=3)  # [B,H,N,2W,D]: önceki + mevcut blok
    vv = torch.cat((F.pad(vb, (0, 0, 0, 0, 1, 0))[:, :, :-1], vb), dim=3)
    i = torch.arange(w, device=q.device)[:, None]
    j = torch.arange(2 * w, device=q.device)[None, :]
    mask = ((j > i) & (j <= i + w)).expand(n, w, 2 * w).clone()  # uzaklık = W + i − j ∈ [0, W)
    mask[0, :, :w] = False  # ilk bloğun "önceki bloğu" dolgudur
    out = F.scaled_dot_product_attention(qb, kk, vv, attn_mask=mask, scale=1.0)
    return out.reshape(b, h, n * w, d)[:, :, :t]


class Kuzgun(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int = 64, window: int = 64, conv_kernel: int = 4,
                 chunk_size: int = 64, branches: str = "both", negative_eigen: bool = False,
                 rope_base: float = 10_000.0, coupled_decay: bool = False, archival_heads: int = 0) -> None:
        super().__init__()
        if branches not in ("both", "window", "memory"):
            raise ValueError("branches: both | window | memory")
        self.h, self.dh = num_heads, head_dim
        self.window = window
        self.chunk_size = chunk_size
        self.branches = branches
        self.use_window = branches in ("both", "window")
        self.use_memory = branches in ("both", "memory")
        self.rope_base = rope_base
        self.beta_scale = 2.0 if negative_eigen else 1.0
        self.coupled_decay = coupled_decay
        # Arşiv kafaları: ilk `archival_heads` kafada unutma yok (α ≡ 1). Delta kuralı yalnız yazılan
        # anahtarın yönünü değiştirdiği için önemsiz tokenler diğer anahtarları silemez -> 1M+ token hafıza.
        decay_mask = torch.ones(num_heads)
        decay_mask[:archival_heads] = 0.0
        self.register_buffer("decay_mask", decay_mask, persistent=False)
        inner = num_heads * head_dim

        self.qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.conv = ShortConv(3 * inner, conv_kernel, activation="silu")
        self.g_proj = nn.Linear(dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, dim, bias=False)
        n_mix = 2 if branches == "both" else 0
        self.mix = nn.Linear(dim, n_mix * num_heads, bias=True) if n_mix else None
        if self.use_window:
            self.log_temp = nn.Parameter(torch.full((num_heads,), math.log(math.sqrt(head_dim))))
            self.norm_w = RMSNorm(head_dim)
        if self.use_memory:
            self.ab = nn.Linear(dim, 2 * num_heads, bias=True)  # unutma (a) ve yazma gücü (b)
            self.norm_m = RMSNorm(head_dim)
            # Gated DeltaNet başlatması: A ∈ [1, 16], dt ∈ [1e-3, 1e-1]
            self.A_log = nn.Parameter(torch.empty(num_heads).uniform_(1.0, 16.0).log())
            dt = torch.exp(torch.empty(num_heads).uniform_(math.log(1e-3), math.log(1e-1)))
            self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

    # ------------------------------------------------------------------ yardımcılar
    def _split(self, x: torch.Tensor, conv_state: Optional[torch.Tensor]):
        z = self.qkv(x)
        if conv_state is None:
            z, new_conv = self.conv(z), None
        else:
            z, new_conv = self.conv.forward_stateful(z, conv_state)
        b, t, _ = z.shape
        q, k, v = z.view(b, t, 3, self.h, self.dh).unbind(2)  # [B,T,H,Dh]
        return F.normalize(q, dim=-1), F.normalize(k, dim=-1), v, new_conv

    def _memory_gates(self, x: torch.Tensor):
        a, bt = self.ab(x).float().chunk(2, dim=-1)
        g = -self.A_log.float().exp() * F.softplus(a + self.dt_bias.float())
        write = torch.sigmoid(bt)
        if self.coupled_decay:
            g = g * write  # "yazmıyorsan unutma": β→0 olan token hafızayı aşındırmaz (α→1)
        g = g * self.decay_mask  # arşiv kafaları: α ≡ 1
        return g, self.beta_scale * write  # [B,T,H]

    def _combine(self, x: torch.Tensor, o_w: Optional[torch.Tensor], o_m: Optional[torch.Tensor]) -> torch.Tensor:
        # o_*: [B,T,H,Dh]
        if self.branches == "both":
            m_w, m_m = torch.sigmoid(self.mix(x)).unsqueeze(-1).chunk(2, dim=-2)
            o = m_w * self.norm_w(o_w) + m_m * self.norm_m(o_m)
        elif self.use_window:
            o = self.norm_w(o_w)
        else:
            o = self.norm_m(o_m)
        b, t = x.shape[:2]
        o = o * F.silu(self.g_proj(x)).view(b, t, self.h, self.dh)
        return self.o_proj(o.reshape(b, t, -1))

    # ------------------------------------------------------------------ ileri geçiş
    def _run(self, x: torch.Tensor, state: Optional[dict]) -> Tuple[torch.Tensor, Optional[dict]]:
        b, t, _ = x.shape
        q, k, v, conv_state = self._split(x, None if state is None else state["conv"])
        new = {"conv": conv_state} if state is not None else None
        tr = lambda z: z.transpose(1, 2)  # [B,T,H,D] -> [B,H,T,D]
        o_w = o_m = None

        if self.use_window:
            pos0 = 0 if state is None else state["pos"]
            cos, sin = rope_tables(torch.arange(pos0, pos0 + t, device=x.device), self.dh, self.rope_base, x.dtype)
            qr = apply_rope(tr(q), cos, sin) * self.log_temp.exp().view(1, -1, 1, 1).to(x.dtype)
            kr = apply_rope(tr(k), cos, sin)
            vv = tr(v)
            if state is None:
                o_w = local_window_attention(qr, kr, vv, self.window)
            else:
                ck, cv = state["k"], state["v"]
                c = ck.size(2)
                k_all, v_all = torch.cat((ck, kr), 2), torch.cat((cv, vv), 2)
                q_all = torch.cat((qr.new_zeros(b, self.h, c, self.dh), qr), 2)
                o_w = local_window_attention(q_all, k_all, v_all, self.window)[:, :, c:]
                keep = self.window - 1
                new["k"], new["v"] = k_all[:, :, -keep:], v_all[:, :, -keep:]
                new["pos"] = pos0 + t
            o_w = tr(o_w)

        if self.use_memory:
            g, beta = self._memory_gates(x)
            s0 = None if state is None else state["S"]
            o_m, s_new = gated_delta_chunk(tr(q).float() / math.sqrt(self.dh), tr(k).float(), tr(v).float(),
                                           tr(g), tr(beta), self.chunk_size, initial_state=s0)
            o_m = tr(o_m).to(x.dtype)
            if state is not None:
                new["S"] = s_new

        return self._combine(x, o_w, o_m), new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._run(x, None)[0]

    def init_state(self, batch: int, device, dtype) -> dict:
        state = {"conv": self.conv.init_state(batch, device, dtype)}
        if self.use_window:
            empty = torch.zeros(batch, self.h, 0, self.dh, device=device, dtype=dtype)
            state.update(k=empty, v=empty.clone(), pos=0)
        if self.use_memory:
            state["S"] = torch.zeros(batch, self.h, self.dh, self.dh, device=device, dtype=torch.float32)
        return state

    def forward_stateful(self, x: torch.Tensor, state: dict) -> Tuple[torch.Tensor, dict]:
        return self._run(x, state)

    def step(self, x: torch.Tensor, state: dict) -> Tuple[torch.Tensor, dict]:
        y, state = self._run(x.unsqueeze(1), state)
        return y.squeeze(1), state
