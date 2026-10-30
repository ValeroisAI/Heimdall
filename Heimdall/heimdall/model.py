"""
Heimdall dil modeli.

Blok (pre-norm):  x ← x + Karıştırıcı(RMSNorm(x));  x ← x + SwiGLU(RMSNorm(x))
Karıştırıcı, katman düzenine göre:

  D — Gated DeltaNet
      [q;k;v] = SiLU(ShortConv(W x)),  q̂ = q/‖q‖·d^-½,  k̂ = k/‖k‖
      α = exp(−e^{A}·softplus(W_a x + dt))   (arşiv kafalarında α ≡ 1),   β = σ(W_b x)
      S_t = α S_{t-1} + β k̂ (v − α S_{t-1}ᵀ k̂)ᵀ,   o = S_tᵀ q̂
      y = W_o( RMSNorm(o) ⊙ SiLU(W_g x) )
  A — Dikkat (GQA, QK-norm, çıkış kapısı)
      y = W_o( SDPA(q, k, v) ⊙ σ(W_gate x) );  hibritte konum kodlaması yok (NoPE)

Cache (çıkarım): {"lens": [int]*B, "layers": [...]}. DeltaNet katmanı sabit boyutlu durum tutar;
dikkat katmanı her anahtarın mutlak konumunu (`kpos`, boş = −1) saklar. Bu sayede farklı
uzunluktaki diziler aynı batch'te birleştirilebilir (sürekli batch'leme).
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .bit import BitLinear
from .config import HeimdallConfig
from .kernels import RMSNorm, causal_attention, causal_conv, delta_rule, delta_rule_step, rope


def make_linear(bits: int, d_in: int, d_out: int, rank: int = 0) -> nn.Module:
    """Gizli matrisler: yoğun nn.Linear ya da gizli ağırlıksız BitLinear (bit.py)."""
    return BitLinear(d_in, d_out, bits, rank) if bits else nn.Linear(d_in, d_out, bias=False)


def _lin(cfg: HeimdallConfig, role: str, d_in: int, d_out: int) -> nn.Module:
    """role: "mix" (DeltaNet/dikkat projeksiyonları) | "ffn". cfg.bit_scope hangi rollerin bit olacağını seçer."""
    bits = cfg.bits if (cfg.bits and cfg.bit_scope in ("all", role)) else 0
    return make_linear(bits, d_in, d_out, cfg.bit_rank)


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: HeimdallConfig) -> None:
        super().__init__()
        d, h, dh = cfg.d_model, cfg.n_heads, cfg.head_dim
        self.cfg, self.h, self.dh, self.inner = cfg, h, dh, h * dh
        self.qkv = _lin(cfg, "mix", d, 3 * self.inner)
        self.qkv.weight.muon_split = (self.inner,) * 3
        self.conv = nn.Parameter(torch.zeros(3 * self.inner, cfg.conv_kernel))
        self.ab = nn.Linear(d, 2 * h, bias=True)
        self.g_proj = _lin(cfg, "mix", d, self.inner)
        self.o_norm = RMSNorm(dh, cfg.norm_eps)
        self.o_proj = _lin(cfg, "mix", self.inner, d)
        self.A_log = nn.Parameter(torch.empty(h).uniform_(1.0, 16.0).log())
        dt = torch.exp(torch.empty(h).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        mask = torch.ones(h)
        mask[:cfg.archival_heads] = 0.0
        self.register_buffer("decay_mask", mask, persistent=False)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.conv.normal_(0, 0.02)
            self.conv[:, -1] += 1.0

    def empty_cache(self, batch: int, device, dtype) -> dict:
        return {"conv": torch.zeros(batch, self.cfg.conv_kernel - 1, 3 * self.inner, device=device, dtype=dtype),
                "S": torch.zeros(batch, self.h, self.dh, self.dh, device=device, dtype=torch.float32)}

    def forward(self, x, cache: Optional[dict] = None, pos=None, lens=None):
        b, t, _ = x.shape
        z = self.qkv(x)
        zc = F.silu(causal_conv(z, self.conv, None if cache is None else cache["conv"]))
        q, k, v = zc.view(b, t, 3, self.h, self.dh).unbind(2)
        q = F.normalize(q.float(), dim=-1) * self.dh ** -0.5
        k = F.normalize(k.float(), dim=-1)
        a, bb = self.ab(x).float().chunk(2, dim=-1)
        g = -self.A_log.float().exp() * F.softplus(a + self.dt_bias.float()) * self.decay_mask
        beta = torch.sigmoid(bb)
        if cache is None:
            o, _ = delta_rule(q, k, v, g, beta, self.cfg.chunk_size, backend=self.cfg.kernel)
        else:
            cache["conv"] = torch.cat((cache["conv"], z.to(cache["conv"].dtype)), 1)[:, -(self.cfg.conv_kernel - 1):]
            if t == 1:
                o, cache["S"] = delta_rule_step(q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0], cache["S"])
                o = o[:, None]
            else:
                o, cache["S"] = delta_rule(q, k, v, g, beta, self.cfg.chunk_size, cache["S"], self.cfg.kernel)
        o = self.o_norm(o.float()).to(x.dtype) * F.silu(self.g_proj(x)).view(b, t, self.h, self.dh)
        return self.o_proj(o.reshape(b, t, self.inner))


class Attention(nn.Module):
    def __init__(self, cfg: HeimdallConfig) -> None:
        super().__init__()
        d, h, hkv, dh = cfg.d_model, cfg.attn_heads, cfg.attn_kv_heads, cfg.attn_head_dim
        self.h, self.hkv, self.dh, self.window = h, hkv, dh, cfg.attn_window
        self.rope_base = cfg.rope_base if cfg.use_rope else 0.0
        self.qkv = _lin(cfg, "mix", d, (h + 2 * hkv) * dh)
        self.qkv.weight.muon_split = (h * dh, hkv * dh, hkv * dh)
        self.q_norm = RMSNorm(dh, cfg.norm_eps)
        self.k_norm = RMSNorm(dh, cfg.norm_eps)
        self.gate = _lin(cfg, "mix", d, h * dh) if cfg.attn_gate else None
        self.o_proj = _lin(cfg, "mix", h * dh, d)

    def empty_cache(self, batch: int, device, dtype) -> dict:
        cap = self.window  # pencere: halka tampon (sabit); global: gerektikçe büyür
        return {"k": torch.zeros(batch, self.hkv, cap, self.dh, device=device, dtype=dtype),
                "v": torch.zeros(batch, self.hkv, cap, self.dh, device=device, dtype=dtype),
                "kpos": torch.full((batch, cap), -1, device=device, dtype=torch.long)}

    def _expand(self, z):  # GQA: [B,Hkv,L,D] → [B,H,L,D]
        if self.hkv == self.h:
            return z
        b, _, n, d = z.shape
        return z[:, :, None].expand(b, self.hkv, self.h // self.hkv, n, d).reshape(b, self.h, n, d)

    def forward(self, x, cache: Optional[dict] = None, pos=None, lens=None):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).split((self.h * self.dh, self.hkv * self.dh, self.hkv * self.dh), dim=-1)
        q = self.q_norm(q.view(b, t, self.h, self.dh)).transpose(1, 2)
        k = self.k_norm(k.view(b, t, self.hkv, self.dh)).transpose(1, 2)
        v = v.view(b, t, self.hkv, self.dh).transpose(1, 2)
        ar = torch.arange(t, device=x.device)
        positions = ar if pos is None else pos[:, None] + ar           # [T] ya da [B,T]
        if self.rope_base:
            q, k = rope(q, positions, self.rope_base), rope(k, positions, self.rope_base)
        if cache is None:
            o = causal_attention(q, self._expand(k), self._expand(v), self.window)
        else:
            o = self._cached(q, k, v, positions, cache, lens)
        o = o.transpose(1, 2).reshape(b, t, self.h * self.dh)
        if self.gate is not None:
            o = o * torch.sigmoid(self.gate(x))
        return self.o_proj(o)

    def _write(self, cache, k, v, positions):
        slots = positions % self.window if self.window else positions
        idx = slots[:, None, :, None].expand(-1, self.hkv, -1, self.dh)
        cache["k"].scatter_(2, idx, k.to(cache["k"].dtype))
        cache["v"].scatter_(2, idx, v.to(cache["v"].dtype))
        cache["kpos"].scatter_(1, slots, positions)

    def _cached(self, q, k, v, positions, cache, lens: List[int]):
        b, t = q.size(0), q.size(2)
        if max(lens) == 0:                                             # boş cache: hızlı nedensel yol
            o = causal_attention(q, self._expand(k), self._expand(v), self.window)
        elif self.window:                                              # halka + yeni anahtarlar
            k_all = torch.cat((cache["k"], k.to(cache["k"].dtype)), 2)
            v_all = torch.cat((cache["v"], v.to(cache["v"].dtype)), 2)
            o = self._masked(q, k_all, v_all, torch.cat((cache["kpos"], positions), 1), positions)
        if not self.window:
            need = max(lens) + t
            cap = cache["kpos"].size(1)
            if need > cap:
                grow = -(-need // 256) * 256 - cap
                cache["k"] = F.pad(cache["k"], (0, 0, 0, grow))
                cache["v"] = F.pad(cache["v"], (0, 0, 0, grow))
                cache["kpos"] = F.pad(cache["kpos"], (0, grow), value=-1)
            self._write(cache, k, v, positions)
            if max(lens) > 0:
                o = self._masked(q, cache["k"], cache["v"], cache["kpos"], positions)
        else:
            m = min(t, self.window)
            self._write(cache, k[:, :, -m:], v[:, :, -m:], positions[:, -m:])
        return o

    def _masked(self, q, k, v, kpos, qpos):
        kp, qp = kpos[:, None, :], qpos[:, :, None]                    # [B,1,L], [B,T,1]
        mask = (kp >= 0) & (kp <= qp)
        if self.window:
            mask = mask & (qp - kp < self.window)
        return F.scaled_dot_product_attention(q, self._expand(k).to(q.dtype), self._expand(v).to(q.dtype),
                                              attn_mask=mask[:, None])


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int, bits: int = 0, rank: int = 0) -> None:
        super().__init__()
        self.w12 = make_linear(bits, d, 2 * hidden, rank)
        self.w12.weight.muon_split = (hidden, hidden)
        self.w3 = make_linear(bits, hidden, d, rank)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class MoE(nn.Module):
    """İnce taneli uzman karışımı (DeepSeek-V3 tarzı): sigmoid yönlendirici, top-k, paylaşılan uzman,
    yardımcı kayıpsız yük dengeleme (yönlendirme sapması, yalnız seçimi etkiler)."""

    def __init__(self, cfg: HeimdallConfig) -> None:
        super().__init__()
        d, e = cfg.d_model, cfg.moe_experts
        h = cfg.moe_hidden or max(64, cfg.ffn_hidden // cfg.moe_topk)
        self.e, self.k, self.h = e, cfg.moe_topk, h
        self.router = nn.Linear(d, e, bias=False)
        self.w12 = nn.Parameter(torch.randn(e, 2 * h, d) * 0.02)
        self.w3 = nn.Parameter(torch.zeros(e, d, h))
        self.shared = SwiGLU(d, h) if cfg.moe_shared else None
        self.register_buffer("route_bias", torch.zeros(e))
        self.balance_rate = 1e-3

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        scores = torch.sigmoid(self.router(flat).float())
        idx = (scores + self.route_bias).topk(self.k, dim=-1).indices                # [N,k]
        w = scores.gather(-1, idx)
        w = (w / w.sum(-1, keepdim=True)).to(x.dtype)
        out = self._experts(flat, idx, w)
        if self.shared is not None:
            out = out + self.shared(flat)
        if self.training:  # az kullanılan uzmanın sapmasını artır, çok kullanılanınkini azalt
            with torch.no_grad():
                load = torch.bincount(idx.flatten(), minlength=self.e).float()
                self.route_bias += self.balance_rate * torch.sign(load.mean() - load)
        return out.reshape(shape)

    @torch.compiler.disable  # veri bağımlı döngü: derleyici dışında
    def _experts(self, flat, idx, w):
        n = flat.size(0)
        order = idx.flatten().argsort()
        tok = order // self.k
        counts = torch.bincount(idx.flatten(), minlength=self.e).tolist()
        wt = w.flatten()[order]
        out = torch.zeros_like(flat)
        start = 0
        for e, c in enumerate(counts):
            if c:
                t = tok[start:start + c]
                a, b = F.linear(flat[t], self.w12[e].to(flat.dtype)).chunk(2, dim=-1)
                y = F.linear(F.silu(a) * b, self.w3[e].to(flat.dtype)) * wt[start:start + c, None]
                out.index_add_(0, t, y.to(out.dtype))
                start += c
        return out


class Block(nn.Module):
    def __init__(self, cfg: HeimdallConfig, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mixer = GatedDeltaNet(cfg) if kind == "D" else Attention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = MoE(cfg) if cfg.moe_experts else SwiGLU(cfg.d_model, cfg.ffn_hidden, cfg.bits if cfg.bit_scope in ("all", "ffn") else 0, cfg.bit_rank)

    def forward(self, x, cache=None, pos=None, lens=None):
        x = x + self.mixer(self.norm1(x), cache, pos, lens)
        return x + self.ffn(self.norm2(x))


class MTPModule(nn.Module):
    """h' = Blok(W [norm(emb(x_{i+k})) ; norm(h_i)]), çıkış normu + paylaşılan lm_head → x_{i+k+1}."""

    def __init__(self, cfg: HeimdallConfig) -> None:
        super().__init__()
        self.norm_e, self.norm_h = RMSNorm(cfg.d_model, cfg.norm_eps), RMSNorm(cfg.d_model, cfg.norm_eps)
        self.fc = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.block = Block(cfg, "D")
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(self, e, h):
        return self.block(self.fc(torch.cat((self.norm_e(e), self.norm_h(h)), -1)))


class HeimdallLM(nn.Module):
    def __init__(self, cfg: HeimdallConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, kind) for kind in cfg.kinds])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.mtp = nn.ModuleList([MTPModule(cfg) for _ in range(cfg.mtp_depth)])
        self.grad_ckpt = False
        self._init_weights()
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, GatedDeltaNet):
                m.reset_parameters()
        for name, m in self.named_modules():  # artık dal çıkışları sıfırdan başlar
            if name.endswith(("o_proj", "w3")):
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                elif isinstance(m, BitLinear):
                    m.zero_()

    def active_params(self) -> int:
        """Token başına kullanılan parametre (MoE'de yalnız seçilen uzmanlar)."""
        n = self.num_params(non_embedding=True)
        for b in self.blocks:
            if isinstance(b.ffn, MoE):
                n -= (b.ffn.e - b.ffn.k) * 3 * b.ffn.h * self.cfg.d_model
        return n

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if non_embedding else n

    def _logits(self, h):
        logits = self.lm_head(self.norm(h)).float()
        cap = self.cfg.logit_softcap
        return cap * torch.tanh(logits / cap) if cap else logits

    def forward(self, ids: torch.Tensor, targets: Optional[torch.Tensor] = None,
                cache: Optional[dict] = None, last_only: bool = False):
        """Eğitim: model(ids, targets) → loss.  Çıkarım: model(ids, cache=c) → logits (cache yerinde güncellenir)."""
        x = self.embed(ids)
        if cache is None:
            for blk in self.blocks:
                x = checkpoint(blk, x, use_reentrant=False) if (self.grad_ckpt and self.training) else blk(x)
        else:
            lens = cache["lens"]
            pos = torch.tensor(lens, device=ids.device)
            for blk, c in zip(self.blocks, cache["layers"]):
                x = blk(x, c, pos, lens)
            cache["lens"] = [n + ids.size(1) for n in lens]
        if last_only:
            x = x[:, -1:]
        logits = self._logits(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        if self.training and len(self.mtp):
            loss = loss + self.cfg.mtp_weight * self._mtp_loss(x, targets)
        return loss

    def _mtp_loss(self, x, targets):
        """targets[:, i] = x_{i+1}. Modül k (1'den): girdi h_i ve x_{i+k}, hedef x_{i+k+1}."""
        h, t, losses = self.norm(x), targets.size(1), []
        safe = targets.clamp(min=0)
        for k, mod in enumerate(self.mtp, start=1):
            n = t - k
            if n <= 0:
                break
            h = mod(self.embed(safe[:, k - 1:k - 1 + n]), h[:, :n])
            logits = self._logits_from(mod.norm(h))
            losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets[:, k:k + n].reshape(-1),
                                          ignore_index=-100))
        return sum(losses) / len(losses)

    def _logits_from(self, hn):
        logits = self.lm_head(hn).float()
        cap = self.cfg.logit_softcap
        return cap * torch.tanh(logits / cap) if cap else logits

    # ------------------------------------------------------------------ cache yönetimi
    def new_cache(self, batch: int = 1) -> dict:
        w = self.embed.weight
        return {"lens": [0] * batch, "layers": [b.mixer.empty_cache(batch, w.device, w.dtype) for b in self.blocks]}

    @staticmethod
    def cache_select(cache: dict, keep: List[int]) -> dict:
        idx = torch.tensor(keep, device=cache["layers"][0][next(iter(cache["layers"][0]))].device)
        return {"lens": [cache["lens"][i] for i in keep],
                "layers": [{k: v.index_select(0, idx) for k, v in c.items()} for c in cache["layers"]]}

    @staticmethod
    def cache_merge(caches: List[dict]) -> dict:
        layers = []
        for parts in zip(*(c["layers"] for c in caches)):
            if "kpos" in parts[0]:  # dikkat: kapasiteyi eşitle (boş yuva kpos = −1)
                cap = max(p["kpos"].size(1) for p in parts)
                parts = [{"k": F.pad(p["k"], (0, 0, 0, cap - p["k"].size(2))),
                          "v": F.pad(p["v"], (0, 0, 0, cap - p["v"].size(2))),
                          "kpos": F.pad(p["kpos"], (0, cap - p["kpos"].size(1)), value=-1)} for p in parts]
            layers.append({k: torch.cat([p[k] for p in parts], 0) for k in parts[0]})
        return {"lens": [n for c in caches for n in c["lens"]], "layers": layers}

    @staticmethod
    def cache_bytes(cache: dict) -> int:
        return sum(v.numel() * v.element_size() for c in cache["layers"] for v in c.values())
