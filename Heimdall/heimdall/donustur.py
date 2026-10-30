"""
donustur.py — Hazır bir Transformer'ı (Llama / Qwen2 / Qwen3 / SmolLM ailesi, BitNet b1.58) sabit bellekli modele dönüştürür.

Fikir: 1000 kat hesap farkını sıfırdan eğitimle kapatamayız, ama o hesap açık ağırlıklarda zaten harcanmış.
Her dikkat katmanı şu katmana dönüştürülür (öğretmenin q/k/v/o ağırlıkları aynen kullanılır):

    o_t = o_pencere + γ_t ⊙ (ρ ⊙ o_hafıza − o_pencere)

    o_pencere : softmax dikkat, yalnız son W token + ilk S "çapa" token (StreamingLLM). t < W iken öğretmenle birebir aynı.
    o_hafıza  : kapılı delta kuralı. Token j, pencereden çıktığı anda (t = j + W) hafızaya yazılır; hafıza ve pencere
                ayrık, hiçbir token iki kez sayılmaz.  S = αS + β k̂(v − αSᵀk̂)ᵀ,  k̂ = norm(F_k k),  q̂ = norm(F_q q)
    γ_t       : pencere dışına düşen dikkat kütlesinin tahmini; girdiye ve pencere içi log-normalizöre (lse) bakar.
                t < W iken 0.

Çıkarım belleği: katman başına W+S anahtar + sabit boyutlu S matrisi. Bağlam uzunluğundan bağımsız.

Eğitim iki aşamalıdır (yalnız yeni parametreler, model boyutunun ~%2'si):
    A) Katman katman: her katman öğretmenin gizli durumlarıyla beslenir, öğretmenin dikkat çıktısını taklit eder.
    B) Uçtan uca: öğretmenin token olasılıklarına KL damıtma.

    # model ve veri otomatik iner (Hugging Face), GPU varsa bf16 ile GPU'da çalışır:
    python -m heimdall.donustur --model HuggingFaceTB/SmolLM2-135M --wikitext veri/wikitext --out donusum
"""

import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .kernels import RMSNorm, delta_rule, delta_rule_step, rope


class TeacherConfig:
    def __init__(self, hf: dict) -> None:
        self.d = hf["hidden_size"]
        self.n_layers = hf["num_hidden_layers"]
        self.n_heads = hf["num_attention_heads"]
        self.n_kv = hf.get("num_key_value_heads", self.n_heads)
        self.head_dim = hf.get("head_dim") or self.d // self.n_heads
        self.inter = hf["intermediate_size"]
        self.rope_theta = hf.get("rope_theta", 10000.0)
        self.eps = hf.get("rms_norm_eps", 1e-6)
        self.vocab = hf["vocab_size"]
        self.tie = hf.get("tie_word_embeddings", False)
        self.qk_norm = hf.get("model_type") == "qwen3"
        self.attn_bias = hf.get("attention_bias", hf.get("model_type") == "qwen2")
        # BitNet b1.58: üçlü ağırlık + 8-bit aktivasyon, relu² FFN, dikkat/FFN çıkışından önce ek RMSNorm (sub-norm)
        self.bitnet = hf.get("model_type") == "bitnet"
        if self.bitnet and (hf.get("quantization_config") or {}).get("quantization_mode", "online") != "online":
            raise NotImplementedError("BitNet: yalnız bf16 ana ağırlıklı ('online') sürüm destekleniyor (ör. -bf16 reposu)")
        if hf.get("hidden_act", "silu") not in ("silu", "relu2"):
            raise NotImplementedError(f"aktivasyon {hf['hidden_act']} desteklenmiyor")
        self.act = hf.get("hidden_act", "silu")
        if hf.get("rope_scaling"):
            raise NotImplementedError("rope_scaling henüz desteklenmiyor")


class Q4Linear(nn.Module):
    """Donuk nicemlenmiş doğrusal katman (4 ya da 8 bit). 32'lik gruplar, grup başına fp16 ölçek + sıfır noktası
    (asimetrik); her grup için en düşük hatayı veren kırpma oranı aranır. Çarpımdan önce açılır."""

    CLIPS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.7)

    def __init__(self, weight: torch.Tensor, group: int = 32, rows: int = 2048, bits: int = 4) -> None:
        super().__init__()
        out_f, in_f = weight.shape
        assert in_f % group == 0, (in_f, group)
        packed, scales, zeros = [], [], []
        top = 2 ** bits - 1
        self.bits = bits
        for r in range(0, out_f, rows):  # satır parçaları: büyük matrislerde bellek tepe noktası küçük kalır
            w = weight[r:r + rows].float().view(-1, in_f // group, group)
            wmax, wmin = w.amax(-1, keepdim=True), w.amin(-1, keepdim=True)
            best_err = best_s = best_z = None
            for c in self.CLIPS:
                sc = ((wmax - wmin) * c / top).clamp(min=1e-8)
                z = wmin * c
                err = ((((w - z) / sc).round().clamp(0, top) * sc + z - w) ** 2).sum(-1, keepdim=True)
                if best_err is None:
                    best_err, best_s, best_z = err, sc, z
                else:
                    better = err < best_err
                    best_err = torch.where(better, err, best_err)
                    best_s, best_z = torch.where(better, sc, best_s), torch.where(better, z, best_z)
            q = ((w - best_z) / best_s).round().clamp(0, top).to(torch.uint8).view(w.size(0), in_f)
            packed.append(q[:, 0::2] | (q[:, 1::2] << 4) if bits == 4 else q)
            scales.append(best_s.squeeze(-1).half())
            zeros.append(best_z.squeeze(-1).half())
        self.register_buffer("packed", torch.cat(packed))                           # [out, in/2]
        self.register_buffer("scale", torch.cat(scales))                            # [out, in/group]
        self.register_buffer("zero", torch.cat(zeros))
        self.in_features, self.out_features, self.group = in_f, out_f, group
        self.bias = None

    @classmethod
    def empty(cls, out_f: int, in_f: int, group: int = 32, bits: int = 4, device=None) -> "Q4Linear":
        """Boş iskelet (kayıtlı nicemlenmiş ağırlıklar load_state_dict ile doldurulur; yeniden nicemleme yok)."""
        m = cls.__new__(cls)
        nn.Module.__init__(m)
        m.bits, m.in_features, m.out_features, m.group, m.bias = bits, in_f, out_f, group, None
        m.register_buffer("packed", torch.empty(out_f, in_f // 2 if bits == 4 else in_f, dtype=torch.uint8, device=device))
        m.register_buffer("scale", torch.empty(out_f, in_f // group, dtype=torch.half, device=device))
        m.register_buffer("zero", torch.empty(out_f, in_f // group, dtype=torch.half, device=device))
        return m

    def dequant(self, dtype) -> torch.Tensor:
        q = torch.stack((self.packed & 15, self.packed >> 4), -1) if self.bits == 4 else self.packed
        q = q.view(self.out_features, -1, self.group).to(dtype)
        return (q * self.scale.to(dtype)[..., None] + self.zero.to(dtype)[..., None]).view(
            self.out_features, self.in_features)

    def forward(self, x):
        if torch.is_autocast_enabled(x.device.type):
            x = x.to(torch.get_autocast_dtype(x.device.type))
        with torch.autocast(x.device.type, enabled=False):
            y = _DequantMM.apply(x, self)
        return y if self.bias is None else y + self.bias.to(y.dtype)


class _DequantMM(torch.autograd.Function):
    """x · Wᵀ, W = Q4Linear. Açılmış ağırlık geri yayılım için SAKLANMAZ, gerektiğinde yeniden açılır
    (27B öğretmende katman başına ~0.5 GB tasarruf). Ağırlığa gradyan yok (donuk öğretmen)."""

    @staticmethod
    def forward(ctx, x, mod):
        ctx.mod = mod
        return F.linear(x, mod.dequant(x.dtype))

    @staticmethod
    def backward(ctx, g):
        return g @ ctx.mod.dequant(g.dtype), None


class _TernaryMM(torch.autograd.Function):
    """x · Wᵀ, W = int8 üçlü. Geri yayılım için açılmış ağırlık kopyası SAKLANMAZ, int8'den yeniden açılır
    (2.4B öğretmende ~4-8 GB tasarruf). Ağırlığa gradyan yok (donuk öğretmen)."""

    @staticmethod
    def forward(ctx, x, wq):
        ctx.save_for_backward(wq)
        ctx.x_dtype = x.dtype
        return F.linear(x, wq.to(x.dtype))

    @staticmethod
    def backward(ctx, g):
        (wq,) = ctx.saved_tensors
        return (g @ wq.to(g.dtype)).to(ctx.x_dtype), None


class TernaryLinear(nn.Module):
    """Donuk BitNet b1.58 doğrusal katmanı (transformers AutoBitLinear 'online' ile aynı hesap): ağırlık tensör başına
    absmean ölçekli {−1,0,+1}, girdi token başına absmax 8-bit. Ağırlık int8 tutulur (bf16'nın yarısı)."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        w = weight.float()
        mean_abs = w.abs().mean().clamp(min=1e-5)
        self.register_buffer("wq", (w / mean_abs).round().clamp(-1, 1).to(torch.int8))
        self.register_buffer("w_scale", mean_abs.reshape(1))
        self.out_features, self.in_features = w.shape
        self.bias = None

    def forward(self, x):
        xf = x.float()
        s = 127 / xf.abs().amax(-1, keepdim=True).clamp(min=1e-5)
        xq = (xf * s).round().clamp(-128, 127) / s
        xq = (xf + (xq - xf).detach()).to(x.dtype)        # düz-geçiş (STE): yeni parametrelere gradyan akar
        if torch.is_autocast_enabled(x.device.type):
            xq = xq.to(torch.get_autocast_dtype(x.device.type))
        with torch.autocast(x.device.type, enabled=False):
            y = _TernaryMM.apply(xq, self.wq)
        return y * self.w_scale.to(y.dtype)


class ConvAttention(nn.Module):
    def __init__(self, tc: TeacherConfig) -> None:
        super().__init__()
        self.tc, self.h, self.hkv, self.dh = tc, tc.n_heads, tc.n_kv, tc.head_dim
        self.q_proj = nn.Linear(tc.d, self.h * self.dh, bias=tc.attn_bias)
        self.k_proj = nn.Linear(tc.d, self.hkv * self.dh, bias=tc.attn_bias)
        self.v_proj = nn.Linear(tc.d, self.hkv * self.dh, bias=tc.attn_bias)
        self.o_proj = nn.Linear(self.h * self.dh, tc.d, bias=False)
        if tc.qk_norm:
            self.q_norm = RMSNorm(self.dh, tc.eps)
            self.k_norm = RMSNorm(self.dh, tc.eps)
        self.attn_sub_norm = RMSNorm(self.h * self.dh, tc.eps) if tc.bitnet else nn.Identity()
        self.converted = False
        self.mode = "teacher"          # teacher | student | layer (aşama A) | window (yalnız pencere, karşılaştırma)
        self.aux = None

    def convert(self, window: int, sinks: int, archival: int) -> None:
        h, dh = self.h, self.dh
        self.window, self.sinks = window, sinks
        self.fq = nn.Parameter(torch.eye(dh).repeat(h, 1, 1))
        self.fk = nn.Parameter(torch.eye(dh).repeat(h, 1, 1))
        self.ab = nn.Linear(self.tc.d, 2 * h)
        self.mix = nn.Linear(self.tc.d, h)
        for lin in (self.ab, self.mix):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        nn.init.constant_(self.mix.bias, -2.0)
        self.mix_lse = nn.Parameter(torch.zeros(h))
        self.mem_scale = nn.Parameter(torch.ones(h))
        self.A_log = nn.Parameter(torch.empty(h).uniform_(1.0, 16.0).log())
        dt = torch.exp(torch.empty(h).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        mask = torch.ones(h)
        mask[:archival] = 0.0
        self.register_buffer("decay_mask", mask, persistent=False)
        self.converted = True

    def new_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith(("q_proj", "k_proj", "v_proj", "o_proj",
                                                                           "q_norm", "k_norm", "attn_sub_norm"))]

    def _qkv(self, x, start: int = 0):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.h, self.dh)
        k = self.k_proj(x).view(b, t, self.hkv, self.dh)
        v = self.v_proj(x).view(b, t, self.hkv, self.dh)
        if self.tc.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        rep = self.h // self.hkv
        k, v = k.repeat_interleave(rep, dim=2), v.repeat_interleave(rep, dim=2)
        pos = torch.arange(start, start + t, device=x.device)
        qr = rope(q.transpose(1, 2), pos, self.tc.rope_theta)
        kr = rope(k.transpose(1, 2), pos, self.tc.rope_theta)
        return q, k, v, qr, kr

    def _full(self, qr, kr, v):
        return F.scaled_dot_product_attention(qr, kr, v.transpose(1, 2), is_causal=True).transpose(1, 2)

    def _window_attn(self, qr, kr, v):
        """Pencere + çapa softmax dikkati, blok-yerel: bellek O(T·(2W+S)). Döndürür o [B,T,H,D], lse [B,H,T]."""
        b, h, t, d = qr.shape
        w, s = self.window, min(self.sinks, t)
        pad = (-t) % w
        n = (t + pad) // w
        vt = v.transpose(1, 2)
        qb, kb, vb = (F.pad(z.float(), (0, 0, 0, pad)).view(b, h, n, w, d) for z in (qr, kr, vt))
        prev = lambda z: F.pad(z, (0, 0, 0, 0, 1, 0))[:, :, :-1]
        sink = lambda z: z[:, :, None, :s].float().expand(b, h, n, s, d)
        keys = torch.cat((sink(kr), prev(kb), kb), dim=3)                           # [B,H,N,S+2W,D]
        vals = torch.cat((sink(vt), prev(vb), vb), dim=3)
        blk = torch.arange(n, device=qr.device)[:, None, None]
        qpos = blk * w + torch.arange(w, device=qr.device)[None, :, None]           # [N,W,1]
        j = torch.arange(w, device=qr.device)
        kpos = torch.cat((torch.arange(s, device=qr.device).expand(n, s),
                          (blk[:, 0] - 1) * w + j, blk[:, 0] * w + j), dim=1)[:, None, :]   # [N,1,S+2W]
        is_sink = torch.arange(s + 2 * w, device=qr.device) < s
        ok = (kpos >= 0) & (kpos <= qpos) & (is_sink | ((qpos - kpos < w) & (kpos >= s)))
        scores = (qb @ keys.transpose(-1, -2)) * self.dh ** -0.5
        scores = scores.masked_fill(~ok, float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)
        o = torch.exp(scores - lse[..., None]) @ vals
        o = o.reshape(b, h, n * w, d)[:, :, :t].transpose(1, 2)
        return o, lse.reshape(b, h, n * w)[:, :, :t]

    def _mem_inputs(self, x, q, k, start: int = 0):
        t = x.size(1)
        i = torch.arange(start, start + t, device=x.device)
        qm = F.normalize(torch.einsum("bthd,hde->bthe", q.float(), self.fq), dim=-1) * self.dh ** -0.5
        km = F.normalize(torch.einsum("bthd,hde->bthe", k.float(), self.fk), dim=-1)
        a, bb = self.ab(x).float().chunk(2, dim=-1)
        g = -self.A_log.exp() * F.softplus(a + self.dt_bias) * self.decay_mask
        beta = torch.sigmoid(bb) * (i >= self.sinks).float()[None, :, None]         # çapa tokenler yazılmaz
        return qm, km, g, beta

    def _hybrid(self, x, q, k, v, qr, kr, memory: bool = True, cache=None):
        b, t = x.shape[:2]
        w = self.window
        i = torch.arange(t, device=x.device)
        o_win, lse = self._window_attn(qr, kr, v)
        if cache is None and (not memory or t <= w):
            return o_win
        qm, km, g, beta = self._mem_inputs(x, q, k)
        shift = lambda z: F.pad(z, (0, 0) * (z.dim() - 2) + (w, 0))[:, :t]           # j, t = j + W'de yazılır
        o_mem, S = delta_rule(qm, shift(km), shift(v.float()), shift(g), shift(beta), 64, backend="torch")
        if cache is not None:  # prefill: halka, çapa ve hafıza durumunu doldur
            self._fill_cache(cache, kr, v, km, g, beta, S)
        gamma = torch.sigmoid(self.mix(x).float() + self.mix_lse * (lse.transpose(1, 2) - math.log(w)))
        gamma = gamma * (i >= w).float()[None, :, None]
        return o_win + gamma[..., None] * (self.mem_scale[:, None] * o_mem - o_win)

    # ------------------------------------------------------------------ sabit bellekli çıkarım
    def _fill_cache(self, c, kr, v, km, g, beta, S):
        b, h, t, d = kr.shape
        w, s = self.window, self.sinks
        dev = kr.device
        c.update(kind="ring", S=S,
                 ring_k=kr.new_zeros(b, h, w, d), ring_v=kr.new_zeros(b, h, w, d),
                 ring_km=km.new_zeros(b, w, h, d), ring_g=g.new_zeros(b, w, h), ring_beta=beta.new_zeros(b, w, h),
                 ring_pos=torch.full((w,), -1, device=dev, dtype=torch.long),
                 sink_k=kr[:, :, :s].clone(), sink_v=v.transpose(1, 2)[:, :, :s].clone())
        m = min(t, w)
        pos = torch.arange(t - m, t, device=dev)
        slot = pos % w
        c["ring_k"][:, :, slot] = kr[:, :, -m:]
        c["ring_v"][:, :, slot] = v.transpose(1, 2)[:, :, -m:].to(kr.dtype)
        c["ring_km"][:, slot], c["ring_g"][:, slot], c["ring_beta"][:, slot] = km[:, -m:], g[:, -m:], beta[:, -m:]
        c["ring_pos"][slot] = pos

    def _step(self, x, c, t: int):
        """Tek token (konum t). Bellek ve iş, bağlam uzunluğundan bağımsız."""
        w, s = self.window, self.sinks
        q, k, v, qr, kr = self._qkv(x, start=t)
        qm, km, g, beta = self._mem_inputs(x, q, k, start=t)
        slot = t % w
        o_mem = None
        if t >= w:  # pencereden çıkan token (t − W) şimdi hafızaya yazılır, sonra okunur
            o_mem, c["S"] = delta_rule_step(qm[:, 0], c["ring_km"][:, slot], c["ring_v"][:, :, slot],
                                            c["ring_g"][:, slot], c["ring_beta"][:, slot], c["S"])
        c["ring_k"][:, :, slot], c["ring_v"][:, :, slot] = kr[:, :, 0], v[:, 0].to(kr.dtype)
        c["ring_km"][:, slot], c["ring_g"][:, slot], c["ring_beta"][:, slot] = km[:, 0], g[:, 0], beta[:, 0]
        c["ring_pos"][slot] = t
        if t < s:
            c["sink_k"] = torch.cat((c["sink_k"], kr), 2)
            c["sink_v"] = torch.cat((c["sink_v"], v.transpose(1, 2).to(kr.dtype)), 2)
        rp = c["ring_pos"]
        ok = torch.cat((torch.ones(c["sink_k"].size(2), dtype=torch.bool, device=x.device),
                        (rp >= s) & (t - rp < w)))
        keys = torch.cat((c["sink_k"], c["ring_k"]), 2).float()
        vals = torch.cat((c["sink_v"], c["ring_v"]), 2).float()
        scores = ((qr.float() @ keys.transpose(-1, -2)) * self.dh ** -0.5).masked_fill(~ok, float("-inf"))
        lse = torch.logsumexp(scores, dim=-1)                                        # [B,H,1]
        o = (torch.exp(scores - lse[..., None]) @ vals).transpose(1, 2)              # [B,1,H,D]
        if o_mem is not None:
            gamma = torch.sigmoid(self.mix(x).float() + self.mix_lse * (lse.transpose(1, 2) - math.log(w)))
            o = o + gamma[..., None] * (self.mem_scale[:, None] * o_mem[:, None] - o)
        return o

    def _full_cached(self, x, c, start: int):
        q, k, v, qr, kr = self._qkv(x, start=start)
        vt = v.transpose(1, 2)
        if start == 0:
            c.update(kind="full", k=kr, v=vt)
            return self._full(qr, kr, v)
        c["k"], c["v"] = torch.cat((c["k"], kr), 2), torch.cat((c["v"], vt), 2)
        return F.scaled_dot_product_attention(qr, c["k"], c["v"]).transpose(1, 2)

    def forward(self, x, cache=None, start: int = 0):
        b, t, _ = x.shape
        if cache is not None:
            if not self.converted or self.mode == "teacher":
                o = self._full_cached(x, cache, start)
            elif start == 0:
                q, k, v, qr, kr = self._qkv(x)
                o = self._hybrid(x, q, k, v, qr, kr, cache=cache)
            else:
                assert t == 1, "dönüşüm cache'i: prefill tek parça, sonra token token"
                o = self._step(x, cache, start)
            return self.o_proj(self.attn_sub_norm(o.reshape(b, t, -1).to(x.dtype)))
        q, k, v, qr, kr = self._qkv(x)
        if not self.converted or self.mode == "teacher":
            o = self._full(qr, kr, v)
        elif self.mode == "layer":  # artık akış öğretmenden; öğrenci dalı yan hesaplanır
            with torch.no_grad():
                o = self._full(qr, kr, v)
            o_s = self._hybrid(x, q, k, v, qr, kr)
            self.aux = (o_s - o.float()).square().mean() / o.float().square().mean().clamp(min=1e-8)
        else:
            o = self._hybrid(x, q, k, v, qr, kr, memory=self.mode == "student")
        return self.o_proj(self.attn_sub_norm(o.reshape(b, t, -1).to(x.dtype)))


class Layer(nn.Module):
    def __init__(self, tc: TeacherConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(tc.d, tc.eps)
        self.self_attn = ConvAttention(tc)
        self.post_attention_layernorm = RMSNorm(tc.d, tc.eps)
        self.gate_proj = nn.Linear(tc.d, tc.inter, bias=False)
        self.up_proj = nn.Linear(tc.d, tc.inter, bias=False)
        self.down_proj = nn.Linear(tc.inter, tc.d, bias=False)
        self.ffn_sub_norm = RMSNorm(tc.inter, tc.eps) if tc.bitnet else nn.Identity()
        self.act = (lambda z: F.relu(z).square()) if tc.act == "relu2" else F.silu

    def forward(self, x, cache=None, start: int = 0):
        x = x + self.self_attn(self.input_layernorm(x), cache, start)
        h = self.post_attention_layernorm(x)
        return x + self.down_proj(self.ffn_sub_norm(self.act(self.gate_proj(h)) * self.up_proj(h)))


class ConvLM(nn.Module):
    def __init__(self, tc: TeacherConfig) -> None:
        super().__init__()
        self.tc = tc
        self.grad_ckpt = False         # eğitimde katman başına yeniden hesap: aktivasyon belleği ~L kat azalır, hız ~%30 düşer
        self.embed_tokens = nn.Embedding(tc.vocab, tc.d)
        self.layers = nn.ModuleList([Layer(tc) for _ in range(tc.n_layers)])
        self.norm = RMSNorm(tc.d, tc.eps)
        self.lm_head = nn.Linear(tc.d, tc.vocab, bias=False)
        if tc.tie:
            self.lm_head.weight = self.embed_tokens.weight

    @classmethod
    def from_pretrained(cls, path, q4: bool = False, group: int = 32, sensitive_bits: int = 8) -> "ConvLM":
        """HF ağırlıklarını tensör tensör yükler (tam fp32 kopya oluşmaz). q4: katman matrisleri 4-bit tutulur."""
        from safetensors import safe_open
        p = Path(path)
        with torch.device("meta"):
            model = cls(TeacherConfig(json.loads((p / "config.json").read_text())))
        mods = dict(model.named_modules())
        biases = {}
        for f in sorted(p.glob("*.safetensors")):
            with safe_open(str(f), framework="pt") as fh:
                for key in fh.keys():
                    name = key.removeprefix("model.").replace("mlp.", "")
                    mod_name, attr = name.rsplit(".", 1)
                    if mod_name not in mods:
                        raise KeyError(f"beklenmeyen ağırlık: {key}")
                    t = fh.get_tensor(key)
                    mod = mods[mod_name]
                    quant = q4 and isinstance(mod, nn.Linear) and (mod_name.startswith("layers.") or
                                                                  (mod_name == "lm_head" and not model.tc.tie))
                    if model.tc.bitnet and isinstance(mod, nn.Linear) and mod_name.startswith("layers."):
                        assert attr == "weight", key
                        q = TernaryLinear(t)
                        parent, child = mod_name.rsplit(".", 1)
                        setattr(mods[parent], child, q)
                        mods[mod_name] = q
                    elif quant and attr == "weight":
                        # Q4_K_M gibi: en hassas matrisler (v_proj, down_proj) daha yüksek bitte
                        bits = sensitive_bits if mod_name.endswith(("v_proj", "down_proj")) else 4
                        q = Q4Linear(t, group, bits=bits)
                        parent, child = mod_name.rsplit(".", 1) if "." in mod_name else ("", mod_name)
                        setattr(mods[parent] if parent else model, child, q)
                        mods[mod_name] = q
                    elif quant or (isinstance(mods[mod_name], Q4Linear) and attr == "bias"):
                        biases[mod_name] = t.float()
                    else:
                        setattr(mod, attr, nn.Parameter(t.float(), requires_grad=False))
        for mod_name, b in biases.items():
            mods[mod_name].bias = nn.Parameter(b, requires_grad=False)
        if model.tc.tie:
            model.lm_head.weight = model.embed_tokens.weight
        meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
        assert not meta, f"eksik ağırlıklar: {meta[:5]}"
        for q in model.parameters():
            q.requires_grad_(True)
        return model

    def n_params(self) -> int:
        return sum(q.numel() for q in self.parameters()) + sum(
            m.packed.numel() * (2 if m.bits == 4 else 1) for m in self.modules() if isinstance(m, Q4Linear)) + sum(
            m.wq.numel() for m in self.modules() if isinstance(m, TernaryLinear))

    def convert(self, window: int = 64, sinks: int = 4, archival_frac: float = 0.25, keep_global=()) -> None:
        for i, layer in enumerate(self.layers):
            if i not in keep_global:
                layer.self_attn.convert(window, sinks, int(archival_frac * self.tc.n_heads))

    def set_mode(self, mode: str) -> None:
        for layer in self.layers:
            layer.self_attn.mode = mode

    def new_params(self):
        return [p for layer in self.layers if layer.self_attn.converted for p in layer.self_attn.new_params()]

    def forward(self, ids, cache=None, last_only: bool = False):
        """cache=None: tam dizi. cache verilirse: ilk çağrı prefill, sonrakiler tek token (sabit bellek)."""
        x = self.embed_tokens(ids)
        start = cache["pos"] if cache is not None else 0
        for i, layer in enumerate(self.layers):
            if self.grad_ckpt and cache is None and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(layer, x, None, start, use_reentrant=False)
            else:
                x = layer(x, None if cache is None else cache["layers"][i], start)
        if cache is not None:
            cache["pos"] += ids.size(1)
        if last_only:
            x = x[:, -1:]
        return self.lm_head(self.norm(x)).float()

    def new_cache(self) -> dict:
        return {"pos": 0, "layers": [{} for _ in self.layers]}

    @staticmethod
    def cache_bytes(cache) -> int:
        return sum(v.numel() * v.element_size() for c in cache["layers"] for v in c.values() if torch.is_tensor(v))

    @torch.no_grad()
    def generate(self, ids, max_new: int = 32, stop=None):
        """Açgözlü üretim. ids: [1, T]. Döndürür (yeni tokenler, cache)."""
        cache = self.new_cache()
        logits = self(ids, cache, last_only=True)
        out = []
        for _ in range(max_new):
            nxt = int(logits[0, -1].argmax())
            if stop is not None and nxt == stop:
                break
            out.append(nxt)
            logits = self(torch.tensor([[nxt]], device=ids.device), cache)
        return out, cache

    def save_conversion(self, path, meta=None) -> None:
        conv = [dict(i=i, window=l.self_attn.window, sinks=l.self_attn.sinks,
                     archival=int((l.self_attn.decay_mask == 0).sum())) for i, l in enumerate(self.layers)
                if l.self_attn.converted]
        torch.save({"conv": conv, "params": {n: q.detach().cpu() for n, q in self.named_parameters()
                                             if any(n.startswith(f"layers.{c['i']}.self_attn.") for c in conv)
                                             and not n.split(".")[-2].endswith(("_proj", "_norm"))},
                    "meta": meta or {}}, path)

    @classmethod
    def load_converted(cls, model_dir, conv_path, q4: bool = False) -> "ConvLM":
        model = cls.from_pretrained(model_dir, q4=q4)
        ck = torch.load(conv_path, map_location="cpu", weights_only=False)
        for c in ck["conv"]:
            model.layers[c["i"]].self_attn.convert(c["window"], c["sinks"], c["archival"])
        missing = [n for n in ck["params"] if n not in dict(model.named_parameters())]
        assert not missing, missing
        model.load_state_dict(ck["params"], strict=False)
        model.set_mode("student")
        return model

    def aux_loss(self):
        auxes = [l.self_attn.aux for l in self.layers if l.self_attn.aux is not None]
        return sum(auxes) / len(auxes)


# --------------------------------------------------------------------------- veri ve ölçüm
def eval_windows(path, t: int, n: int, sep: int, dtype=np.uint16):
    """Makale başından başlayan, en az t+1 uzunluklu pencereler (bağlam gerçekten büyür)."""
    data = np.fromfile(path, dtype=dtype).astype(np.int64)
    starts = [0] + [i + 1 for i in np.nonzero(data == sep)[0][:-1]]
    ends = list(np.nonzero(data == sep)[0])
    wins = [torch.from_numpy(data[s:s + t + 1]) for s, e in zip(starts, ends) if e - s > t]
    return torch.stack(wins[:n])


@torch.no_grad()
def evaluate(model, wins, mode: str, buckets):
    model.set_mode(mode)
    nll = torch.zeros(wins.size(1) - 1, device=wins.device)
    for w in wins:
        logits = model(w[None, :-1])
        nll += F.cross_entropy(logits[0], w[1:], reduction="none")
    nll /= wins.size(0)
    out = {"ppl": math.exp(nll.mean().item())}
    for a, b in buckets:
        out[f"{a}-{b}"] = math.exp(nll[a:b].mean().item())
    return out


def fetch_model(name: str) -> str:
    if Path(name).exists():
        return name
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=name, allow_patterns=["*.json", "*.safetensors"])


def prepare_wikitext(model_dir: str, out_dir: str, train_tokens: int = 20_000_000, sep: int = 0):
    """wikitext-103'ü indirip öğretmenin tokenizer'ıyla .bin dosyalarına çevirir (makaleler sep ile ayrılır)."""
    import re

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {"test": ["test-00000-of-00001"], "train": ["train-00000-of-00002", "train-00001-of-00002"]}
    if all((out / f"wiki_{k}.bin").exists() for k in files):
        return str(out / "wiki_train.bin"), str(out / "wiki_test.bin")
    tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    for split, names in files.items():
        ids = []
        for name in names:
            path = hf_hub_download("Salesforce/wikitext", f"wikitext-103-raw-v1/{name}.parquet", repo_type="dataset")
            cur = []
            for line in pq.read_table(path).column("text").to_pylist() + [" = END = \n"]:
                if re.match(r"^ = [^=].* = \n$", line) and cur:
                    ids.extend(tok.encode("".join(cur)).ids + [sep])
                    cur = []
                cur.append(line)
            if split == "train" and len(ids) >= train_tokens:
                break
        np.array(ids, dtype=np.uint16 if tok.get_vocab_size() < 65536 else np.uint32).tofile(out / f"wiki_{split}.bin")
        print(f"wikitext {split}: {len(ids):,} token", flush=True)
    return str(out / "wiki_train.bin"), str(out / "wiki_test.bin")


@torch.no_grad()
def verify_against_hf(model_name: str, n_tokens: int = 128) -> float:
    """Öğretmen uygulamamızı transformers'ın resmi modeliyle karşılaştırır (dönüştürmeden önce çalıştır)."""
    from transformers import AutoModelForCausalLM

    path = fetch_model(model_name)
    ids = torch.randint(0, 1000, (1, n_tokens))
    ref = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()(ids).logits
    ours = ConvLM.from_pretrained(path).eval()(ids)
    err = (ref - ours).abs().max().item()
    if json.loads((Path(path) / "config.json").read_text()).get("model_type") == "bitnet":
        # 8-bit aktivasyon yuvarlaması süreksiz: 1e-6'lık toplama sırası farkı bile bazı basamakları çevirir ve 30 katmanda
        # büyür (kendi modelimize 1e-6 gürültü: logit farkı ~1.5, NLL farkı ~0.003). Ölçüt: tahmin uyuşması.
        agree = (ref.argmax(-1) == ours.argmax(-1)).float().mean().item()
        print(f"{model_name}: argmax uyuşması {agree:.3f}, en büyük logit farkı {err:.2e} → "
              f"{'AYNI (aktivasyon yuvarlaması sınırında)' if agree > 0.8 else 'FARKLI'}")
        return err
    print(f"{model_name}: en büyük logit farkı {err:.2e} (logit ölçeği {ref.abs().max().item():.1f}) → "
          f"{'BİREBİR AYNI' if err < 1e-3 * ref.abs().max().item() else 'FARKLI: bu model desteklenmiyor'}")
    return err


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model klasörü ya da HF adı (ör. HuggingFaceTB/SmolLM2-135M)")
    p.add_argument("--train", help="eğitim .bin (öğretmen tokenizer'ı)")
    p.add_argument("--test", help="test .bin (makaleler --sep ile ayrılmış)")
    p.add_argument("--wikitext", help="wikitext-103'ü bu klasöre hazırla ve kullan")
    p.add_argument("--wikitext-tokens", type=float, default=20e6)
    p.add_argument("--device", default=None, help="cuda (ROCm dahil) | cpu; boşsa otomatik")
    p.add_argument("--out", default="donusum")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--sinks", type=int, default=4)
    p.add_argument("--archival-frac", type=float, default=0.25)
    p.add_argument("--keep-global", type=int, nargs="*", default=[], help="dönüştürülmeyecek katmanlar")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--steps-a", type=int, default=100)
    p.add_argument("--steps-b", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--eval-len", type=int, default=2048)
    p.add_argument("--eval-n", type=int, default=12)
    p.add_argument("--sep", type=int, default=0, help="makale ayırıcı token")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--grad-ckpt", action="store_true", help="katman başına yeniden hesap (bellek ↓, hız ~%30 ↓); büyük modeller için")
    p.add_argument("--dogrula", action="store_true", help="yalnız transformers ile birebirlik kontrolü yap")
    p.add_argument("--q4", action="store_true", help="donuk öğretmen matrisleri 4-bit (büyük modeller 16 GB'a sığsın)")
    p.add_argument("--hassas-bit", type=int, default=8, choices=[4, 8],
                   help="--q4 ile v_proj/down_proj bit sayısı (8: daha kaliteli, 4: en küçük)")
    args = p.parse_args()
    if args.dogrula:
        verify_against_hf(args.model)
        return
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    args.model = fetch_model(args.model)
    if args.wikitext:
        args.train, args.test = prepare_wikitext(args.model, args.wikitext, int(args.wikitext_tokens), args.sep)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    gpu = device.type == "cuda"
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if gpu else contextlib.nullcontext()

    model = ConvLM.from_pretrained(args.model, q4=args.q4, sensitive_bits=args.hassas_bit)
    model.convert(args.window, args.sinks, args.archival_frac, set(args.keep_global))
    model.grad_ckpt = args.grad_ckpt
    for q in model.parameters():
        q.requires_grad_(False)
    params = model.new_params()
    new_ids = {id(q) for q in params}
    for q in model.parameters():
        if gpu and id(q) not in new_ids:
            q.data = q.data.to(torch.bfloat16)   # donmuş öğretmen ağırlıkları bf16: yarı bellek, hızlı matmul
    for q in params:
        q.requires_grad_(True)
    model.to(device)
    n_new = sum(q.numel() for q in params)
    tc = model.tc
    kv_teacher = lambda n: tc.n_layers * 2 * tc.n_kv * tc.head_dim * n * 2
    state = sum(tc.n_heads * tc.head_dim ** 2 * 4 + 2 * tc.n_kv * tc.head_dim * (args.window + args.sinks) * 2
                for i in range(tc.n_layers) if i not in args.keep_global)
    print(f"Öğretmen {model.n_params() / 1e6:.0f}M param{' (4-bit)' if args.q4 else ''}, {tc.n_layers} katman | yeni param "
          f"{n_new / 1e6:.2f}M | pencere {args.window} + {args.sinks} çapa | bellek: öğretmen 8K'da "
          f"{kv_teacher(8192) / 2**20:.0f} MB, 1M'de {kv_teacher(2**20) / 2**30:.1f} GB → dönüşüm {state / 2**20:.1f} MB (sabit)",
          flush=True)

    tok_dtype = np.uint16 if model.tc.vocab < 65536 else np.uint32     # prepare_wikitext ile aynı kural
    wins = eval_windows(args.test, args.eval_len, args.eval_n, args.sep, tok_dtype).to(device)
    L = args.eval_len
    buckets = [(a, min(b, L)) for a, b in ((0, args.window), (args.window, 256), (256, 1024), (1024, L)) if a < min(b, L)]
    data = np.memmap(args.train, dtype=tok_dtype, mode="r")
    rng = np.random.default_rng(0)

    def batch():
        s = rng.integers(0, len(data) - args.seq_len - 1, args.batch)
        return torch.from_numpy(np.stack([data[i:i + args.seq_len] for i in s]).astype(np.int64)).to(device)

    results = {"args": vars(args), "eval_windows": int(wins.size(0))}
    t0 = time.time()
    with amp:
        results["ogretmen"] = evaluate(model, wins, "teacher", buckets)
        results["yalniz_pencere"] = evaluate(model, wins, "window", buckets)
        results["donusum_egitimsiz"] = evaluate(model, wins, "student", buckets)
    for k in ("ogretmen", "yalniz_pencere", "donusum_egitimsiz"):
        print(f"{k:22s} " + " | ".join(f"{b}: {v:.2f}" for b, v in results[k].items()), flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    for stage, steps in (("A", args.steps_a), ("B", args.steps_b)):
        model.set_mode("layer" if stage == "A" else "student")
        for g in opt.param_groups:
            g["lr"] = args.lr
        for step in range(1, steps + 1):
            for g in opt.param_groups:  # kosinüs düşüş
                g["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * step / steps))
            x = batch()
            with amp:
                if stage == "A":
                    model(x)
                    loss = model.aux_loss()
                else:
                    with torch.no_grad():
                        model.set_mode("teacher")
                        t_logp = F.log_softmax(model(x), dim=-1)
                        model.set_mode("student")
                    s_logp = F.log_softmax(model(x), dim=-1)
                    loss = F.kl_div(s_logp, t_logp, log_target=True, reduction="batchmean") / x.size(1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step % 10 == 0 or step == steps:
                peak = f" | tepe GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if gpu else ""
                print(f"  aşama {stage} adım {step}/{steps} | kayıp {loss.item():.4f} | "
                      f"{step * x.numel() / (time.time() - t0):,.0f} tok/s{peak}", flush=True)
        t0 = time.time()
        key = f"donusum_asama_{stage}"
        with amp:
            results[key] = evaluate(model, wins, "student", buckets)
        results[key]["egitim_token"] = (args.steps_a + (args.steps_b if stage == "B" else 0)) * args.batch * args.seq_len
        print(f"{key:22s} " + " | ".join(f"{b}: {v:.2f}" for b, v in results[key].items()), flush=True)
        (out / "sonuc.json").write_text(json.dumps(results, indent=2))
    model.save_conversion(out / "donusum_param.pt", {"model": args.model, "results": results})
    print(f"Kaydedildi: {out}", flush=True)


if __name__ == "__main__":
    main()
