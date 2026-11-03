"""
Kuzgun dil modeli.

Blok:   x ← x + Kuzgun(RMSNorm(x));   [x ← x + Hafıza(RMSNorm(x))];   x ← x + SwiGLU(RMSNorm(x))
        (Hafıza: seçilen bloklarda product-key katmanı. cfg.ternary: gizli matrisler üçlü, bkz. uclu.py)

Kuzgun katmanı (global dikkat yok, çıkarım belleği O(1)):
    [q;k;v] = SiLU(ShortConv(W_qkv x)),  q̂ = q/‖q‖,  k̂ = k/‖k‖       (ortak anahtarlar)
    Huginn : son W token üzerinde tam softmax (RoPE, kafa başına sıcaklık)
    Muninn : S = α S + β k̂ (v − α Sᵀk̂)ᵀ,  o = Sᵀq̂     (arşiv kafalarında α ≡ 1)
    o = σ(m_H)·RMSNorm(o_H) + σ(m_M)·RMSNorm(o_M);   y = W_o(o ⊙ SiLU(W_g x))

API:
    model(ids, targets)                 -> (loss, logs)     eğitim
    model(ids)                          -> logits
    model.new_cache(B)                  -> cache (sabit boyutlu)
    model.prefill(ids, cache)           -> son logit, cache (uzun istemler parça parça)
    model.step(id, cache)               -> logit, cache       (token token üretim)
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import KuzgunConfig
from .hafiza import ProductKeyMemory
from .ops import (RMSNorm, apply_rope, causal_short_conv, chunk_gated_delta_rule, gated_delta_step,
                  rope_cos_sin, sliding_window_attention)
from .uclu import TernaryLinear


def make_linear(cfg: KuzgunConfig, d_in: int, d_out: int) -> nn.Module:
    """Gizli matrisler: yoğun (nn.Linear) ya da gizli ağırlıksız üçlü (TernaryLinear)."""
    if cfg.ternary:
        return TernaryLinear(d_in, d_out, use_int8=cfg.ternary_int8)
    return nn.Linear(d_in, d_out, bias=False)


class Kuzgun(nn.Module):
    def __init__(self, cfg: KuzgunConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, h, dh = cfg.d_model, cfg.n_heads, cfg.head_dim
        inner = h * dh
        self.h, self.dh, self.inner = h, dh, inner
        self.qkv = make_linear(cfg, d, 3 * inner)
        self.conv_w = nn.Parameter(torch.zeros(3 * inner, cfg.conv_kernel))
        self.gates = nn.Linear(d, 4 * h, bias=True)       # [unutma a, yazma b, karışım m_H, karışım m_M]
        self.g_proj = make_linear(cfg, d, inner)
        self.o_proj = make_linear(cfg, inner, d)
        self.norm_h = RMSNorm(dh, cfg.norm_eps)
        self.norm_m = RMSNorm(dh, cfg.norm_eps)
        self.log_temp = nn.Parameter(torch.full((h,), math.log(math.sqrt(dh))))
        self.A_log = nn.Parameter(torch.empty(h).uniform_(1.0, 16.0).log())
        dt = torch.exp(torch.empty(h).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))    # softplus⁻¹(dt)
        decay_mask = torch.ones(h)
        decay_mask[:cfg.archival_heads] = 0.0                               # arşiv kafaları: α ≡ 1
        self.register_buffer("decay_mask", decay_mask, persistent=False)
        self.beta_scale = 2.0 if cfg.negative_eigen else 1.0

    def reset_conv(self) -> None:
        with torch.no_grad():
            self.conv_w.normal_(0, 0.02)
            self.conv_w[:, -1] += 1.0   # başta kimliğe yakın

    # ------------------------------------------------------------------ ortak hazırlık
    def _qkv(self, x, conv_prefix=None):
        b, t, _ = x.shape
        z = self.qkv(x)
        z_conv = F.silu(causal_short_conv(z, self.conv_w, conv_prefix))
        q, k, v = z_conv.view(b, t, 3, self.h, self.dh).unbind(2)
        q = F.normalize(q.float(), dim=-1)
        k = F.normalize(k.float(), dim=-1)
        return q, k, v, z

    def _gates(self, x):
        a, bw, mh, mm = self.gates(x).float().split(self.h, dim=-1)
        g = -self.A_log.float().exp() * F.softplus(a + self.dt_bias.float()) * self.decay_mask
        beta = self.beta_scale * torch.sigmoid(bw)
        return g, beta, torch.sigmoid(mh), torch.sigmoid(mm)

    def _mix(self, x, o_h, o_m, mh, mm):
        # o_*: [B,T,H,Dh]
        o = mh[..., None] * self.norm_h(o_h) + mm[..., None] * self.norm_m(o_m)
        b, t = x.shape[:2]
        o = o.to(x.dtype) * F.silu(self.g_proj(x)).view(b, t, self.h, self.dh)
        return self.o_proj(o.reshape(b, t, self.inner))

    # ------------------------------------------------------------------ eğitim (tam dizi)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v, _ = self._qkv(x)
        g, beta, mh, mm = self._gates(x)
        tr = lambda z: z.transpose(1, 2)   # [B,T,H,D] -> [B,H,T,D]
        cos, sin = rope_cos_sin(torch.arange(t, device=x.device), self.dh, self.cfg.rope_base, torch.float32)
        temp = self.log_temp.float().exp().view(1, -1, 1, 1)
        qh = (apply_rope(tr(q), cos, sin) * temp).to(x.dtype)
        kh = apply_rope(tr(k), cos, sin).to(x.dtype)
        o_h = sliding_window_attention(qh, kh, tr(v).to(x.dtype), self.cfg.window, self.cfg.attn_backend)
        o_m, _ = chunk_gated_delta_rule(tr(q) / math.sqrt(self.dh), tr(k), tr(v), tr(g), tr(beta),
                                        self.cfg.chunk_size)
        return self._mix(x, tr(o_h), tr(o_m), mh, mm)

    # ------------------------------------------------------------------ çıkarım (cache)
    def new_cache(self, batch: int, device, dtype) -> dict:
        k = self.conv_w.size(1)
        return {
            "conv": torch.zeros(batch, k - 1, 3 * self.inner, device=device, dtype=dtype),
            "k": torch.zeros(batch, self.h, 0, self.dh, device=device, dtype=dtype),
            "v": torch.zeros(batch, self.h, 0, self.dh, device=device, dtype=dtype),
            "pos": 0,
            "S": torch.zeros(batch, self.h, self.dh, self.dh, device=device, dtype=torch.float32),
        }

    def prefill(self, x: torch.Tensor, cache: dict) -> Tuple[torch.Tensor, dict]:
        """x: [B,T,D] — önceki cache'e devam eder (uzun istemler parça parça verilebilir)."""
        b, t, _ = x.shape
        q, k, v, z = self._qkv(x, cache["conv"])
        conv_all = torch.cat((cache["conv"], z.to(cache["conv"].dtype)), dim=1)
        g, beta, mh, mm = self._gates(x)
        tr = lambda y: y.transpose(1, 2)
        pos = cache["pos"]
        cos, sin = rope_cos_sin(torch.arange(pos, pos + t, device=x.device), self.dh, self.cfg.rope_base, torch.float32)
        temp = self.log_temp.float().exp().view(1, -1, 1, 1)
        qh = (apply_rope(tr(q), cos, sin) * temp).to(x.dtype)
        kh = apply_rope(tr(k), cos, sin).to(x.dtype)
        c = cache["k"].size(2)
        k_all = torch.cat((cache["k"], kh.to(cache["k"].dtype)), dim=2)
        v_all = torch.cat((cache["v"], tr(v).to(cache["v"].dtype)), dim=2)
        q_all = torch.cat((qh.new_zeros(b, self.h, c, self.dh), qh), dim=2)
        o_h = sliding_window_attention(q_all, k_all, v_all, self.cfg.window, "sdpa")[:, :, c:]
        o_m, S = chunk_gated_delta_rule(tr(q) / math.sqrt(self.dh), tr(k), tr(v), tr(g), tr(beta),
                                        self.cfg.chunk_size, initial_state=cache["S"])
        keep = self.cfg.window - 1
        new_cache = {"conv": conv_all[:, conv_all.size(1) - cache["conv"].size(1):], "k": k_all[:, :, -keep:],
                     "v": v_all[:, :, -keep:], "pos": pos + t, "S": S}
        return self._mix(x, tr(o_h), tr(o_m), mh, mm), new_cache

    def step(self, x: torch.Tensor, cache: dict) -> Tuple[torch.Tensor, dict]:
        """x: [B,D] — tek token. Tüm işler bağlam uzunluğundan bağımsız."""
        b = x.size(0)
        q, k, v, z = self._qkv(x[:, None], cache["conv"])
        conv_all = torch.cat((cache["conv"], z.to(cache["conv"].dtype)), dim=1)[:, 1:]
        g, beta, mh, mm = self._gates(x[:, None])
        pos = cache["pos"]
        cos, sin = rope_cos_sin(torch.tensor([pos], device=x.device), self.dh, self.cfg.rope_base, torch.float32)
        temp = self.log_temp.float().exp().view(1, -1, 1, 1)
        qh = (apply_rope(q.transpose(1, 2), cos, sin) * temp).to(x.dtype)          # [B,H,1,D]
        kh = apply_rope(k.transpose(1, 2), cos, sin).to(x.dtype)
        k_all = torch.cat((cache["k"], kh.to(cache["k"].dtype)), dim=2)[:, :, -self.cfg.window:]
        v_all = torch.cat((cache["v"], v.transpose(1, 2).to(cache["v"].dtype)), dim=2)[:, :, -self.cfg.window:]
        o_h = F.scaled_dot_product_attention(qh, k_all, v_all, scale=1.0)            # pencere içi tüm tokenler
        o_m, S = gated_delta_step(q[:, 0] / math.sqrt(self.dh), k[:, 0], v[:, 0], g[:, 0], beta[:, 0], cache["S"])
        y = self._mix(x[:, None], o_h.transpose(1, 2), o_m[:, None], mh, mm)
        keep = self.cfg.window - 1
        return y[:, 0], {"conv": conv_all, "k": k_all[:, :, -keep:], "v": v_all[:, :, -keep:], "pos": pos + 1, "S": S}


class SwiGLU(nn.Module):
    def __init__(self, cfg: KuzgunConfig) -> None:
        super().__init__()
        self.w12 = make_linear(cfg, cfg.d_model, 2 * cfg.ffn_hidden)
        self.w3 = make_linear(cfg, cfg.ffn_hidden, cfg.d_model)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, cfg: KuzgunConfig, idx: int = 0) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mixer = Kuzgun(cfg)
        self.memory = None
        if idx in cfg.memory_layers:  # token başına bakılan dev hafıza (cache gerektirmez)
            self.norm_mem = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.memory = ProductKeyMemory(cfg.d_model, cfg.memory_n_sub, cfg.memory_heads,
                                           cfg.memory_topk, cfg.memory_dq)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def _tail(self, x):
        if self.memory is not None:
            x = x + self.memory(self.norm_mem(x))
        return x + self.ffn(self.norm2(x))

    def forward(self, x):
        return self._tail(x + self.mixer(self.norm1(x)))

    def prefill(self, x, cache):
        y, cache = self.mixer.prefill(self.norm1(x), cache)
        return self._tail(x + y), cache

    def step(self, x, cache):
        y, cache = self.mixer.step(self.norm1(x), cache)
        return self._tail(x + y), cache


class KuzgunLM(nn.Module):
    def __init__(self, cfg: KuzgunConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.mtp:
            # t+2 tahmini: [h_t ; emb(x_{t+1})] -> küçük FFN -> ortak LM başı (DeepSeek-V3 MTP'nin hafif hali)
            self.mtp_norm_h = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.mtp_norm_e = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.mtp_proj = make_linear(cfg, 2 * cfg.d_model, cfg.d_model)
            self.mtp_ffn = SwiGLU(cfg)
            self.mtp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.grad_ckpt = False
        self._init_weights()
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def _init_weights(self) -> None:
        for name, m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, Kuzgun):
                m.reset_conv()
        for name, m in self.named_modules():  # artık dalların çıkışları sıfırdan: derin ağda temiz gradyan
            if name.endswith(("o_proj", "w3")):
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                elif isinstance(m, TernaryLinear):
                    m.zero_()

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if non_embedding else n

    def memory_params(self) -> int:
        """Hafıza değer tabloları: parametre çok, token başına hesap küçük (yalnız top-k satır okunur)."""
        return sum(b.memory.values.weight.numel() for b in self.blocks if b.memory is not None)

    def _logits(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(self.norm(h)).float()
        cap = self.cfg.logit_softcap
        return cap * torch.tanh(logits / cap) if cap else logits

    def forward(self, ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        x = self.embed(ids)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if (self.grad_ckpt and self.training) else block(x)
        logits = self._logits(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        logs = {"loss_lm": loss.detach()}
        if self.cfg.mtp and ids.size(1) > 1:
            # h_t ve bir sonraki gerçek token x_{t+1} ile x_{t+2}'yi tahmin et (hedefler targets'tan kaydırılır)
            e_next = self.embed(targets[:, :-1])
            m = self.mtp_proj(torch.cat((self.mtp_norm_h(x[:, :-1]), self.mtp_norm_e(e_next)), dim=-1))
            m = m + self.mtp_ffn(self.mtp_norm(m))
            mtp_logits = self._logits(m)
            mtp_loss = F.cross_entropy(mtp_logits.view(-1, mtp_logits.size(-1)), targets[:, 1:].reshape(-1))
            logs["loss_mtp"] = mtp_loss.detach()
            loss = loss + self.cfg.mtp_weight * mtp_loss
        return loss, logs

    # ------------------------------------------------------------------ çıkarım
    def new_cache(self, batch: int, device=None, dtype=None) -> List[dict]:
        device = device or self.embed.weight.device
        dtype = dtype or self.embed.weight.dtype
        return [b.mixer.new_cache(batch, device, dtype) for b in self.blocks]

    @torch.no_grad()
    def prefill(self, ids: torch.Tensor, cache: List[dict], chunk: int = 4096) -> Tuple[torch.Tensor, List[dict]]:
        """Uzun istemi parça parça işler; bellek sabit kalır. Son tokenin logit'ini döndürür."""
        logits = None
        for s in range(0, ids.size(1), chunk):
            x = self.embed(ids[:, s:s + chunk])
            new = []
            for block, c in zip(self.blocks, cache):
                x, c = block.prefill(x, c)
                new.append(c)
            cache = new
            logits = self._logits(x[:, -1])
        return logits, cache

    @torch.no_grad()
    def step(self, ids: torch.Tensor, cache: List[dict]) -> Tuple[torch.Tensor, List[dict]]:
        x = self.embed(ids)
        new = []
        for block, c in zip(self.blocks, cache):
            x, c = block.step(x, c)
            new.append(c)
        return self._logits(x), new

    @staticmethod
    def cache_bytes(cache: List[dict]) -> int:
        return sum(v.numel() * v.element_size() for c in cache for v in c.values() if torch.is_tensor(v))
