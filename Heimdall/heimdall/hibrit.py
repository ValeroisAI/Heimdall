"""
hibrit.py — Zaten hibrit olan modellerde (Qwen3.5 / 3.6 / 3.8 dense ailesi: 3 Gated DeltaNet + 1 kapılı tam dikkat)
yalnız TAM DİKKAT katmanlarını sabit bellekli katmana çevirir. DeltaNet ve FFN katmanlarına dokunulmaz.

Qwen3.8-27B: 64 katman = 48 Gated DeltaNet (zaten sabit bellek) + 16 tam dikkat (24 Q / 4 KV kafa, kafa boyutu 256,
RoPE yalnız ilk 64 boyutta, çıkışta sigmoid kapı). Orijinalde KV cache token başına 64 KB büyür (262K'da 16 GB).
Dönüşümde her tam dikkat katmanı şu olur (donustur.py ile aynı fikir, GQA'ya göre düzenlendi):

    o = o_pencere + γ ⊙ (ρ ⊙ o_hafıza − o_pencere),   y = W_o( o ⊙ σ(kapı) )

    o_pencere : son W token + ilk S "çapa" token üzerinde softmax dikkat (öğretmenin q/k/v'si aynen)
    o_hafıza  : kapılı delta kuralı. Pencereden çıkan token hafızaya yazılır. Hafıza KV kafası başına tutulur
                (4 durum; her biri 6 sorgu kafasınca okunur): katman başına 1 MB, bağlamdan bağımsız.
    γ         : pencere dışına düşen dikkat kütlesinin tahmini (girdi + pencere içi log-normalizör)

Yeni parametreler katman başına ~2M (öğretmenin %0.01'i). Eğitim: donustur_hibrit.py.
Öğretmen uygulaması transformers'ın Qwen3_5 koduyla birebir test edilir (tests/test_hibrit.py).
MTP (çok-token tahmini) katmanı da yüklenir ve aynı şekilde dönüştürülür: spekülatif üretimde taslak token verir.
"""

import json
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .donustur import Q4Linear
from .kernels import causal_conv, delta_rule, delta_rule_step, rope

ATTN, LINEAR = "full_attention", "linear_attention"


class HybridConfig:
    def __init__(self, hf: dict) -> None:
        hf = hf.get("text_config", hf)
        if hf.get("model_type") not in ("qwen3_5_text", "qwen3_5"):
            raise NotImplementedError(f"model_type {hf.get('model_type')}: yalnız Qwen3.5/3.6/3.8 dense (qwen3_5)")
        if hf.get("num_experts"):
            raise NotImplementedError("MoE sürümleri (ör. 35B-A3B) henüz desteklenmiyor")
        if hf.get("hidden_act", "silu") != "silu":
            raise NotImplementedError(f"aktivasyon {hf['hidden_act']}")
        self.d, self.n_layers = hf["hidden_size"], hf["num_hidden_layers"]
        every = hf.get("full_attention_interval", 4)
        self.layer_types = hf.get("layer_types") or [ATTN if (i + 1) % every == 0 else LINEAR
                                                     for i in range(self.n_layers)]
        self.h, self.hkv = hf["num_attention_heads"], hf["num_key_value_heads"]
        self.dh = hf.get("head_dim") or self.d // self.h
        rp = hf.get("rope_parameters") or {}
        if rp.get("rope_type", "default") != "default" or hf.get("rope_scaling"):
            raise NotImplementedError("yalnız varsayılan RoPE")
        self.rope_theta = rp.get("rope_theta", hf.get("rope_theta", 10000.0))
        self.rot = int(self.dh * rp.get("partial_rotary_factor", hf.get("partial_rotary_factor", 1.0)))
        self.gate = hf.get("attn_output_gate", True)
        self.attn_bias = hf.get("attention_bias", False)
        self.lin_hk, self.lin_hv = hf["linear_num_key_heads"], hf["linear_num_value_heads"]
        self.lin_dk, self.lin_dv = hf["linear_key_head_dim"], hf["linear_value_head_dim"]
        self.conv_k = hf.get("linear_conv_kernel_dim", 4)
        self.inter, self.eps = hf["intermediate_size"], hf.get("rms_norm_eps", 1e-6)
        self.vocab, self.tie = hf["vocab_size"], hf.get("tie_word_embeddings", False)
        self.mtp_layers = hf.get("mtp_num_hidden_layers", 0)
        eos = hf.get("eos_token_id", 0)
        self.eos = eos[0] if isinstance(eos, list) else eos

    def kv_bytes_per_token(self) -> int:
        """Orijinal modelin bf16 KV cache'i (yalnız tam dikkat katmanları büyür)."""
        return self.layer_types.count(ATTN) * 2 * self.hkv * self.dh * 2


class ZNorm(nn.Module):
    """Qwen3.5 RMSNorm: ağırlık sıfır merkezli, çarpan (1 + w)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps) * (1.0 + self.weight.float())).type_as(x)


class GatedNorm(nn.Module):
    """DeltaNet çıkış normu: w ⊙ RMSNorm(o) ⊙ SiLU(z)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x, z):
        xf = x.float()
        h = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)).to(x.dtype)
        return (self.weight * h * F.silu(z.float())).to(x.dtype)


def l2norm(x, eps: float = 1e-6):
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + eps)


class _Conv(nn.Module):
    def __init__(self, c: int, k: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(c, 1, k))


class GatedDeltaNet(nn.Module):
    """Qwen3.5 Gated DeltaNet (donuk öğretmen; Heimdall'ınkiyle aynı denklem, QK kafaları V'den az)."""

    def __init__(self, c: HybridConfig) -> None:
        super().__init__()
        self.c = c
        self.kd, self.vd = c.lin_hk * c.lin_dk, c.lin_hv * c.lin_dv
        self.in_proj_qkv = nn.Linear(c.d, 2 * self.kd + self.vd, bias=False)
        self.in_proj_z = nn.Linear(c.d, self.vd, bias=False)
        self.in_proj_b = nn.Linear(c.d, c.lin_hv, bias=False)
        self.in_proj_a = nn.Linear(c.d, c.lin_hv, bias=False)
        self.conv1d = _Conv(2 * self.kd + self.vd, c.conv_k)
        self.dt_bias = nn.Parameter(torch.ones(c.lin_hv))
        self.A_log = nn.Parameter(torch.zeros(c.lin_hv))
        self.norm = GatedNorm(c.lin_dv, c.eps)
        self.out_proj = nn.Linear(self.vd, c.d, bias=False)

    def forward(self, x, cache: Optional[dict] = None, start: int = 0):
        c, (b, t, _) = self.c, x.shape
        k1 = c.conv_k - 1
        z = self.in_proj_qkv(x)
        prefix = None if cache is None else cache.get("conv")
        zc = F.silu(causal_conv(z, self.conv1d.weight[:, 0], prefix))
        if cache is not None:
            full = z if prefix is None else torch.cat((prefix.to(z.dtype), z), 1)
            cache["conv"] = F.pad(full, (0, 0, max(0, k1 - full.size(1)), 0))[:, -k1:]
        q, k, v = zc.split((self.kd, self.kd, self.vd), dim=-1)
        r = c.lin_hv // c.lin_hk
        q = l2norm(q.reshape(b, t, c.lin_hk, c.lin_dk).float()).repeat_interleave(r, 2) * c.lin_dk ** -0.5
        k = l2norm(k.reshape(b, t, c.lin_hk, c.lin_dk).float()).repeat_interleave(r, 2)
        v = v.reshape(b, t, c.lin_hv, c.lin_dv).float()
        beta = torch.sigmoid(self.in_proj_b(x).float())
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(x).float() + self.dt_bias.float())
        s0 = None if cache is None else cache.get("S")
        if s0 is not None and t == 1:
            o, s = delta_rule_step(q[:, 0], k[:, 0], v[:, 0], g[:, 0], beta[:, 0], s0)
            o = o[:, None]
        else:
            o, s = delta_rule(q, k, v, g, beta, 64, s0)
        if cache is not None:
            cache["S"] = s
        o = self.norm(o.to(x.dtype), self.in_proj_z(x).view(b, t, c.lin_hv, c.lin_dv))
        return self.out_proj(o.reshape(b, t, self.vd))


class HybridAttention(nn.Module):
    """Qwen3.5 kapılı tam dikkat (öğretmen) ve sabit bellekli dönüşümü (öğrenci)."""

    NEW = ("fq", "fk", "ab", "mix", "mix_lse", "mem_scale", "A_log", "dt_bias")
    GATES = ("ab", "mix", "mix_lse", "mem_scale", "A_log", "dt_bias")   # aşama B "gates": fq/fk donuk

    def __init__(self, c: HybridConfig) -> None:
        super().__init__()
        self.c = c
        self.q_proj = nn.Linear(c.d, c.h * c.dh * (2 if c.gate else 1), bias=c.attn_bias)
        self.k_proj = nn.Linear(c.d, c.hkv * c.dh, bias=c.attn_bias)
        self.v_proj = nn.Linear(c.d, c.hkv * c.dh, bias=c.attn_bias)
        self.o_proj = nn.Linear(c.h * c.dh, c.d, bias=c.attn_bias)
        self.q_norm, self.k_norm = ZNorm(c.dh, c.eps), ZNorm(c.dh, c.eps)
        self.converted = False
        self.mode = "teacher"       # teacher | student | window (hafızasız, karşılaştırma)

    # ------------------------------------------------------------------ dönüşüm
    def convert(self, window: int, sinks: int, archival: int) -> None:
        c = self.c
        dev = self.k_norm.weight.device
        self.window, self.sinks = window, sinks
        self.fq = nn.Parameter(torch.eye(c.dh, device=dev).repeat(c.h, 1, 1))
        self.fk = nn.Parameter(torch.eye(c.dh, device=dev).repeat(c.hkv, 1, 1))
        self.ab = nn.Linear(c.d, 2 * c.hkv, device=dev)
        self.mix = nn.Linear(c.d, c.h, device=dev)
        for lin in (self.ab, self.mix):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        nn.init.constant_(self.mix.bias, -2.0)
        self.mix_lse = nn.Parameter(torch.zeros(c.h, device=dev))
        self.mem_scale = nn.Parameter(torch.ones(c.h, device=dev))
        self.A_log = nn.Parameter(torch.empty(c.hkv, device=dev).uniform_(1.0, 16.0).log())
        dt = torch.exp(torch.empty(c.hkv, device=dev).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        mask = torch.ones(c.hkv, device=dev)
        mask[:archival] = 0.0
        self.register_buffer("decay_mask", mask, persistent=False)
        self.converted, self.mode = True, "student"

    def new_params(self, which: str = "all"):
        names = self.NEW if which == "all" else self.GATES
        return [p for n, p in self.named_parameters() if n.split(".")[0] in names]

    # ------------------------------------------------------------------ ortak projeksiyonlar
    def _rope(self, x, pos):
        r = self.c.rot
        if r == 0:
            return x
        if r == x.size(-1):
            return rope(x, pos, self.c.rope_theta)
        return torch.cat((rope(x[..., :r], pos, self.c.rope_theta), x[..., r:]), dim=-1)

    def _proj(self, x, start: int):
        c, (b, t, _) = self.c, x.shape
        qg = self.q_proj(x)
        gate = None
        if c.gate:
            q, gate = qg.view(b, t, c.h, 2 * c.dh).chunk(2, dim=-1)
            gate = gate.reshape(b, t, c.h * c.dh)
        else:
            q = qg.view(b, t, c.h, c.dh)
        q = self.q_norm(q)
        k = self.k_norm(self.k_proj(x).view(b, t, c.hkv, c.dh))
        v = self.v_proj(x).view(b, t, c.hkv, c.dh)
        pos = torch.arange(start, start + t, device=x.device)
        return q, k, v, self._rope(q.transpose(1, 2), pos), self._rope(k.transpose(1, 2), pos), gate

    def _out(self, o, gate, x):
        o = o.reshape(*x.shape[:2], -1).to(x.dtype)
        return self.o_proj(o * torch.sigmoid(gate) if gate is not None else o)

    # ------------------------------------------------------------------ öğretmen
    def _teacher(self, qr, kr, v, cache):
        vt = v.transpose(1, 2)
        if cache is not None:
            if "k" in cache:
                kr, vt = torch.cat((cache["k"], kr), 2), torch.cat((cache["v"], vt), 2)
            cache["k"], cache["v"] = kr, vt
        t, n = qr.size(2), kr.size(2)
        if t == n:
            o = F.scaled_dot_product_attention(qr, kr, vt, is_causal=True, enable_gqa=True)
        else:
            mask = torch.arange(n, device=qr.device)[None] <= torch.arange(n - t, n, device=qr.device)[:, None]
            o = F.scaled_dot_product_attention(qr, kr, vt, attn_mask=mask, enable_gqa=True)
        return o.transpose(1, 2)

    # ------------------------------------------------------------------ öğrenci (eğitim ve çıkarım AYNI yol)
    def _mem_inputs(self, x, q, k, pos):
        dh = self.c.dh
        qm = F.normalize(torch.einsum("bthd,hde->bthe", q.float(), self.fq), dim=-1) * dh ** -0.5
        km = F.normalize(torch.einsum("bthd,hde->bthe", k.float(), self.fk), dim=-1)
        a, bb = self.ab(x).float().chunk(2, dim=-1)
        g = -self.A_log.exp() * F.softplus(a + self.dt_bias) * self.decay_mask
        beta = torch.sigmoid(bb) * (pos >= self.sinks).float()[None, :, None]     # çapa tokenler yazılmaz
        return qm, km, g, beta

    def _window(self, qr, ext_k, ext_v, ext_pos, sink_k, sink_v, pos):
        """Pencere + çapa dikkati, W'lik sorgu blokları (bellek O(t·W)). ext_*: [halka (W) ; yeni tokenler]."""
        b, h, t, dh = qr.shape
        hkv, w, s = ext_k.size(1), self.window, self.sinks
        q = qr.float().view(b, hkv, h // hkv, t, dh) * dh ** -0.5
        ns = 0 if sink_k is None else sink_k.size(2)
        spos = torch.arange(ns, device=qr.device)
        outs, lses = [], []
        for o0 in range(0, t, w):
            n = min(w, t - o0)
            kb, vb, kp = ext_k[:, :, o0:o0 + n + w], ext_v[:, :, o0:o0 + n + w], ext_pos[o0:o0 + n + w]
            if ns:
                kb, vb, kp = torch.cat((sink_k, kb), 2), torch.cat((sink_v, vb), 2), torch.cat((spos, kp))
            is_sink = torch.arange(kp.numel(), device=qr.device) < ns
            qp = pos[o0:o0 + n, None]
            ok = (kp >= 0) & (kp <= qp) & (is_sink | ((qp - kp < w) & (kp >= s)))
            sc = torch.einsum("bgrqd,bgkd->bgrqk", q[:, :, :, o0:o0 + n], kb.float()).masked_fill(~ok, float("-inf"))
            lse = sc.logsumexp(-1)
            outs.append(torch.einsum("bgrqk,bgkd->bgrqd", torch.exp(sc - lse[..., None]), vb.float()))
            lses.append(lse)
        return torch.cat(outs, 3).reshape(b, h, t, dh).transpose(1, 2), torch.cat(lses, 3).reshape(b, h, t)

    def _student(self, x, q, k, v, qr, kr, start: int, cache: Optional[dict], memory: bool = True):
        c, (b, t, _) = self.c, x.shape
        w, s, r, dev = self.window, self.sinks, c.h // c.hkv, x.device
        st = cache if cache is not None else {}
        pos = torch.arange(start, start + t, device=dev)
        vt = v.transpose(1, 2)
        sink_k, sink_v = st.get("sink_k"), st.get("sink_v")
        n_new = max(0, min(t, s - start))
        if n_new:
            sink_k = kr[:, :, :n_new] if sink_k is None else torch.cat((sink_k, kr[:, :, :n_new]), 2)
            sink_v = vt[:, :, :n_new] if sink_v is None else torch.cat((sink_v, vt[:, :, :n_new]), 2)
        pp = torch.arange(start - w, start, device=dev)                       # halkanın kapsadığı konumlar
        if "ring_k" in st:
            idx = pp % w
            rk, rv, rkm = st["ring_k"][:, :, idx], st["ring_v"][:, :, idx], st["ring_km"][:, idx]
            rg, rb, valid = st["ring_g"][:, idx], st["ring_beta"][:, idx], st["ring_pos"][idx] == pp
        else:
            rk, rv = kr.new_zeros(b, c.hkv, w, c.dh), vt.new_zeros(b, c.hkv, w, c.dh)
            rkm = torch.zeros(b, w, c.hkv, c.dh, device=dev)
            rg, rb = torch.zeros(b, w, c.hkv, device=dev), torch.zeros(b, w, c.hkv, device=dev)
            valid = torch.zeros(w, dtype=torch.bool, device=dev)
        ext_pos = torch.cat((torch.where(valid, pp, torch.full_like(pp, -1)), pos))
        o_win, lse = self._window(qr, torch.cat((rk.to(kr.dtype), kr), 2), torch.cat((rv.to(vt.dtype), vt), 2),
                                  ext_pos, sink_k, sink_v, pos)
        o = o_win
        use_mem = memory and start + t > w
        if use_mem or cache is not None:
            qm, km, g, beta = self._mem_inputs(x, q, k, pos)
        if use_mem:  # konum i'deki sorgudan önce token i−W yazılır: yazma dizisi = ext'in ilk t elemanı
            vf = valid.float()[None, :, None]
            wk = torch.cat((rkm, km), 1)[:, :t]
            wv = torch.cat((rv.transpose(1, 2).float(), v.float()), 1)[:, :t]
            wg = torch.cat((rg * vf, g), 1)[:, :t]
            wb = torch.cat((rb * vf, beta), 1)[:, :t]
            s0 = st.get("S")
            if t == 1:
                sm = torch.zeros(b, c.hkv, c.dh, c.dh, device=dev) if s0 is None else s0
                sm = sm * wg[:, 0].exp()[..., None, None]
                delta = (wv[:, 0] - torch.einsum("bgk,bgkv->bgv", wk[:, 0], sm)) * wb[:, 0, :, None]
                sm = sm + wk[:, 0, :, :, None] * delta[:, :, None, :]
                o_mem = torch.einsum("bgrk,bgkv->bgrv", qm[:, 0].view(b, c.hkv, r, c.dh), sm).reshape(b, 1, c.h, c.dh)
            else:  # KV kafası başına tek durum: yazmalar grup içinde aynı, okuma 6 sorgu kafasıyla
                rep = lambda z: z.repeat_interleave(r, 2)
                o_mem, sm = delta_rule(qm, rep(wk), rep(wv), rep(wg), rep(wb), 64,
                                       None if s0 is None else s0.repeat_interleave(r, 1), backend="torch")
                sm = sm[:, ::r]
            gamma = torch.sigmoid(self.mix(x).float() + self.mix_lse * (lse.transpose(1, 2) - math.log(w)))
            gamma = gamma * (pos >= w).float()[None, :, None]
            o = o_win + gamma[..., None] * (self.mem_scale[:, None] * o_mem - o_win)
            if cache is not None:
                cache["S"] = sm
        if cache is not None:
            if "ring_k" not in cache:
                cache.update(ring_k=rk, ring_v=rv, ring_km=rkm, ring_g=rg, ring_beta=rb,
                             ring_pos=torch.full((w,), -1, dtype=torch.long, device=dev))
            m = min(t, w)
            slot = pos[-m:] % w
            cache["ring_k"][:, :, slot], cache["ring_v"][:, :, slot] = kr[:, :, -m:], vt[:, :, -m:]
            cache["ring_km"][:, slot], cache["ring_g"][:, slot] = km[:, -m:], g[:, -m:]
            cache["ring_beta"][:, slot], cache["ring_pos"][slot] = beta[:, -m:], pos[-m:]
            cache["sink_k"], cache["sink_v"] = sink_k, sink_v
        return o

    def forward(self, x, cache: Optional[dict] = None, start: int = 0):
        q, k, v, qr, kr, gate = self._proj(x, start)
        if not self.converted or self.mode == "teacher":
            o = self._teacher(qr, kr, v, cache)
        else:
            o = self._student(x, q, k, v, qr, kr, start, cache, memory=self.mode != "window")
        return self._out(o, gate, x)

    def distill_pair(self, x):
        """Aşama A: (öğretmen çıkışı, öğrenci çıkışı), ikisi de W_o'dan sonra (artık akışın gördüğü)."""
        q, k, v, qr, kr, gate = self._proj(x, 0)
        with torch.no_grad():
            yt = self._out(self._teacher(qr, kr, v, None), gate, x)
        return yt, self._out(self._student(x, q, k, v, qr, kr, 0, None, memory=self.mode != "window"), gate, x)


class MLP(nn.Module):
    def __init__(self, c: HybridConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(c.d, c.inter, bias=False)
        self.up_proj = nn.Linear(c.d, c.inter, bias=False)
        self.down_proj = nn.Linear(c.inter, c.d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HLayer(nn.Module):
    def __init__(self, c: HybridConfig, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.input_layernorm = ZNorm(c.d, c.eps)
        if kind == LINEAR:
            self.linear_attn = GatedDeltaNet(c)
        else:
            self.self_attn = HybridAttention(c)
        self.post_attention_layernorm = ZNorm(c.d, c.eps)
        self.mlp = MLP(c)

    @property
    def mixer(self):
        return self.linear_attn if self.kind == LINEAR else self.self_attn

    def forward(self, x, cache: Optional[dict] = None, start: int = 0):
        x = x + self.mixer(self.input_layernorm(x), cache, start)
        return x + self.mlp(self.post_attention_layernorm(x))


class MTP(nn.Module):
    """Qwen3.5 MTP: h' = Katman(W_fc [norm(emb(x_{i+1})) ; norm(h_i)]) → aynı lm_head ile x_{i+2} tahmini."""

    def __init__(self, c: HybridConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(2 * c.d, c.d, bias=False)
        self.pre_fc_norm_embedding, self.pre_fc_norm_hidden = ZNorm(c.d, c.eps), ZNorm(c.d, c.eps)
        self.layers = nn.ModuleList([HLayer(c, ATTN) for _ in range(c.mtp_layers)])
        self.norm = ZNorm(c.d, c.eps)

    def inputs(self, emb_next, hidden):
        return self.fc(torch.cat((self.pre_fc_norm_embedding(emb_next), self.pre_fc_norm_hidden(hidden)), -1))

    def forward(self, emb_next, hidden, cache: Optional[dict] = None, start: int = 0):
        return self.norm(self.layers[0](self.inputs(emb_next, hidden), cache, start))


# --------------------------------------------------------------------------- ağırlık yükleme
class Weights:
    """safetensors ağırlıklarına isimle erişim (Qwen3_5ForConditionalGeneration ve ForCausalLM adları; görsel atlanır)."""

    PREFIXES = ("model.language_model.", "language_model.model.", "model.")

    def __init__(self, path) -> None:
        from safetensors import safe_open
        self._open = safe_open
        self.path = Path(path)
        self.config = json.loads((self.path / "config.json").read_text())
        idx = self.path / "model.safetensors.index.json"
        if idx.exists():
            wm = json.loads(idx.read_text())["weight_map"]
        else:
            wm = {}
            for f in sorted(self.path.glob("*.safetensors")):
                with safe_open(str(f), framework="pt") as fh:
                    wm.update({k: f.name for k in fh.keys()})
        self.map = {}
        for k, f in wm.items():
            n = self.local(k)
            if n is not None:
                self.map[n] = (k, f)
        self._fh = {}

    @classmethod
    def local(cls, key: str):
        if key.startswith(("model.visual.", "visual.")):
            return None
        for p in cls.PREFIXES:
            if key.startswith(p):
                return key[len(p):]
        return key

    def has(self, name: str) -> bool:
        return name in self.map

    def get(self, name: str) -> torch.Tensor:
        key, f = self.map[name]
        if f not in self._fh:
            self._fh[f] = self._open(str(self.path / f), framework="pt")
        return self._fh[f].get_tensor(key)


def _replace(root: nn.Module, name: str, new: nn.Module) -> None:
    parent, _, child = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, child, new)


def q4_eligible(mod: nn.Module, group: int) -> bool:
    return bool(group) and isinstance(mod, nn.Linear) and mod.out_features >= 256 and mod.in_features % group == 0


def materialize(root: nn.Module, prefix: str, w: Weights, device, dtype, q4_group: int = 0, embed_device=None) -> None:
    """meta cihazında kurulmuş modülü ağırlıklarla doldurur. q4_group > 0: büyük matrisler 4-bit (GPU'da nicemlenir)."""
    for name, sub in list(root.named_modules()):
        for pname, _ in list(sub.named_parameters(recurse=False)):
            key = prefix + (f"{name}." if name else "") + pname
            if not w.has(key):
                continue
            t = w.get(key)
            if pname == "weight" and q4_eligible(sub, q4_group):
                q = Q4Linear(t.to(device), q4_group, bits=4)
                if sub.bias is not None:
                    q.bias = nn.Parameter(w.get(key[:-6] + "bias").float().to(device), requires_grad=False)
                _replace(root, name, q)
                break
            big = isinstance(sub, (nn.Linear, nn.Embedding))
            dev = embed_device if (embed_device is not None and isinstance(sub, nn.Embedding)) else device
            setattr(sub, pname, nn.Parameter(t.to(dev, dtype if big else torch.float32), requires_grad=False))


def q4_skeleton(root: nn.Module, group: int, device) -> None:
    """Kayıtlı 4-bit cache'ten yüklemek için: uygun Linear'ları boş Q4Linear ile değiştir."""
    for name, sub in list(root.named_modules()):
        if q4_eligible(sub, group):
            q = Q4Linear.empty(sub.out_features, sub.in_features, group, 4, device)
            if sub.bias is not None:
                q.bias = nn.Parameter(torch.empty(sub.out_features, device=device), requires_grad=False)
            _replace(root, name, q)


# --------------------------------------------------------------------------- model
class HybridLM(nn.Module):
    def __init__(self, c: HybridConfig, mtp: bool = True) -> None:
        super().__init__()
        self.c = c
        self.embed_tokens = nn.Embedding(c.vocab, c.d)
        self.layers = nn.ModuleList([HLayer(c, kind) for kind in c.layer_types])
        self.norm = ZNorm(c.d, c.eps)
        self.lm_head = nn.Linear(c.d, c.vocab, bias=False)
        self.mtp = MTP(c) if (mtp and c.mtp_layers) else None
        self.grad_ckpt = False
        self.mtp_hidden = "normed"     # MTP'ye giren gizli durum: "normed" (son normdan sonra) | "raw"

    @classmethod
    def load(cls, path, device="cpu", dtype=torch.float32, q4_group: int = 0, embed_device=None,
             mtp: bool = True, q4_cache=None) -> "HybridLM":
        gguf = str(path).endswith(".gguf")
        if gguf:
            from .gguf_cevir import GGUFWeights
            w = GGUFWeights(path)
        else:
            w = Weights(path)
        c = HybridConfig(w.config)
        with torch.device("meta"):
            m = cls(c, mtp=mtp and w.has("mtp.fc.weight"))
        if q4_cache is not None and Path(q4_cache).exists():
            q4_skeleton(m, q4_group, device)
            sd = torch.load(q4_cache, map_location="cpu", weights_only=True, mmap=True)
            m.load_state_dict(sd, strict=False, assign=True)
            m.embed_tokens.to(embed_device or device)
            for n, p in list(m.named_parameters()) + list(m.named_buffers()):
                if not n.startswith("embed_tokens"):
                    p.data = p.data.to(device)
        else:
            materialize(m, "", w, device, dtype, q4_group, embed_device)
            if q4_cache is not None:
                torch.save({k: v for k, v in m.state_dict().items()}, q4_cache)
        if c.tie and not w.has("lm_head.weight"):
            m.lm_head.weight = m.embed_tokens.weight
        meta = [n for n, t in list(m.named_parameters()) + list(m.named_buffers()) if t.is_meta]
        assert not meta, f"eksik ağırlıklar: {meta[:5]}"
        for p in m.parameters():
            p.requires_grad_(False)
        m.embedded_conversion = w.conversion_state() if gguf else None   # GGUF dönüşümü kendi içinde taşır
        return m

    @property
    def device(self):
        return self.norm.weight.device

    def attn_modules(self):
        """(ad, HybridAttention) çiftleri: ana model + MTP."""
        out = [(f"layers.{i}", l.self_attn) for i, l in enumerate(self.layers) if l.kind == ATTN]
        if self.mtp is not None:
            out += [(f"mtp.layers.{i}", l.self_attn) for i, l in enumerate(self.mtp.layers)]
        return out

    def convert(self, window: int = 512, sinks: int = 4, archival_frac: float = 0.25, keep_global=()) -> None:
        for name, a in self.attn_modules():
            if name not in keep_global:
                a.convert(window, sinks, int(archival_frac * self.c.hkv))

    def set_mode(self, mode: str) -> None:
        for _, a in self.attn_modules():
            a.mode = mode if a.converted else "teacher"

    def new_params(self, which: str = "all"):
        return [p for _, a in self.attn_modules() if a.converted for p in a.new_params(which)]

    # ------------------------------------------------------------------ ileri
    def embed(self, ids):
        e = self.embed_tokens(ids.to(self.embed_tokens.weight.device))
        return e.to(self.device)

    def hidden(self, ids, cache: Optional[dict] = None):
        """Döndürür (normlu, ham) son gizli durum. cache verilirse yerinde güncellenir (her uzunlukta parça)."""
        x = self.embed(ids)
        start = cache["pos"] if cache is not None else 0
        for i, layer in enumerate(self.layers):
            if self.grad_ckpt and cache is None and torch.is_grad_enabled():
                x = checkpoint(layer, x, None, 0, use_reentrant=False)
            else:
                x = layer(x, None if cache is None else cache["layers"][i], start)
        if cache is not None:
            cache["pos"] += ids.size(1)
        return self.norm(x), x

    def logits(self, h):
        return self.lm_head(h).float()

    def forward(self, ids, cache: Optional[dict] = None, last_only: bool = False):
        h, _ = self.hidden(ids, cache)
        return self.logits(h[:, -1:] if last_only else h)

    def mtp_hidden_in(self, hn, hr):
        return hn if self.mtp_hidden == "normed" else hr

    def mtp_step(self, hn, hr, next_ids, cache: Optional[dict] = None):
        """MTP: konum i'nin gizli durumu + x_{i+1} → x_{i+2} logitleri. cache: {"pos", "layer"}."""
        start = 0 if cache is None else cache.setdefault("pos", 0)
        h = self.mtp(self.embed(next_ids), self.mtp_hidden_in(hn, hr),
                     None if cache is None else cache.setdefault("layer", {}), start)
        if cache is not None:
            cache["pos"] += next_ids.size(1)
        return self.logits(h)

    # ------------------------------------------------------------------ cache
    def new_cache(self) -> dict:
        return {"pos": 0, "layers": [{} for _ in self.layers], "mtp": {}}

    @staticmethod
    def cache_bytes(cache) -> int:
        dicts = list(cache["layers"]) + [cache.get("mtp", {}).get("layer", {})]
        return sum(v.numel() * v.element_size() for d in dicts for v in d.values() if torch.is_tensor(v))

    @staticmethod
    def _snapshot(cache) -> dict:
        return {"pos": cache["pos"], "layers": [{k: v.clone() if torch.is_tensor(v) else v for k, v in d.items()}
                                                for d in cache["layers"]]}

    @torch.no_grad()
    def prefill(self, ids, cache, chunk: int = 2048, mtp_state: Optional[dict] = None):
        """Uzun girdiyi parça parça işler (bellek parça boyutuyla sınırlı). Son konumun logitlerini döndürür.
        mtp_state verilirse MTP cache'i de beslenir; son konumun gizli durumu mtp_state["pending"]'de bekler."""
        hn = hr = None
        for a in range(0, ids.size(1), chunk):
            part = ids[:, a:a + chunk]
            hn, hr = self.hidden(part, cache)
            if mtp_state is not None and self.mtp is not None:
                pend = mtp_state.get("pending")
                ph = hn if pend is None else torch.cat((pend[0], hn), 1)
                pr = hr if pend is None else torch.cat((pend[1], hr), 1)
                nxt = part if pend is not None else part[:, 1:]
                if nxt.size(1):
                    self.mtp_step(ph[:, :nxt.size(1)], pr[:, :nxt.size(1)], nxt, cache["mtp"])
                mtp_state["pending"] = (hn[:, -1:], hr[:, -1:])
        return self.logits(hn[:, -1:])[:, -1]

    @torch.no_grad()
    def generate(self, ids, max_new: int = 64, stop=(), chunk: int = 2048, cache=None, mtp: bool = False,
                 stats: Optional[dict] = None):
        """Açgözlü üretim. mtp=True: MTP taslağı + doğrulama (çıktı mtp=False ile aynıdır, yalnız hızlıdır).
        cache verilirse (sohbet) önceki bağlamın üstüne devam eder. Döndürür (yeni tokenler, cache)."""
        cache = cache if cache is not None else self.new_cache()
        mtp = mtp and self.mtp is not None
        state = cache.setdefault("mtp_state", {}) if mtp else None
        last = self.prefill(ids, cache, chunk, state)
        x = int(last.argmax())
        out = [x]
        stats = stats if stats is not None else {}
        stats.setdefault("taslak", 0), stats.setdefault("kabul", 0)
        dev = self.device
        if not mtp:
            while len(out) < max_new and x not in stop:
                x = int(self(torch.tensor([[x]], device=dev), cache, last_only=True)[0, -1].argmax())
                out.append(x)
            return out, cache
        pn, pr = state.pop("pending")
        draft = int(self.mtp_step(pn, pr, torch.tensor([[x]], device=dev), cache["mtp"])[0, -1].argmax())
        while len(out) < max_new and x not in stop:
            snap = self._snapshot(cache)
            hn, hr = self.hidden(torch.tensor([[x, draft]], device=dev), cache)
            y = self.logits(hn)[0].argmax(-1).tolist()
            stats["taslak"] += 1
            if y[0] == draft:          # taslak doğru: tek ileri geçişte iki token
                stats["kabul"] += 1
                out.append(draft)
                if draft in stop or len(out) >= max_new:
                    break
                x = y[1]
                out.append(x)
                lg = self.mtp_step(hn, hr, torch.tensor([[draft, x]], device=dev), cache["mtp"])
            else:                      # yanlış: cache'i geri al, yalnız x'i işle
                cache["pos"], cache["layers"] = snap["pos"], snap["layers"]
                hn, hr = self.hidden(torch.tensor([[x]], device=dev), cache)
                x = y[0]
                out.append(x)
                lg = self.mtp_step(hn, hr, torch.tensor([[x]], device=dev), cache["mtp"])
            draft = int(lg[0, -1].argmax())
        return out, cache

    # ------------------------------------------------------------------ dönüşüm kaydı
    def conversion_state(self, meta=None) -> dict:
        conv, params = [], {}
        for name, a in self.attn_modules():
            if a.converted:
                conv.append(dict(name=name, window=a.window, sinks=a.sinks, archival=int((a.decay_mask == 0).sum())))
                for n, p in a.named_parameters():
                    if n.split(".")[0] in a.NEW:
                        params[f"{name}.self_attn.{n}"] = p.detach().float().cpu()
        return {"conv": conv, "params": params, "mtp_hidden": self.mtp_hidden, "meta": meta or {}}

    def apply_conversion(self, state: dict) -> None:
        mods = dict(self.attn_modules())
        for cv in state["conv"]:
            if cv["name"] in mods:
                mods[cv["name"]].convert(cv["window"], cv["sinks"], cv["archival"])
        own = dict(self.named_parameters())
        for n, t in state["params"].items():
            if n in own:
                own[n].data.copy_(t.to(own[n].device))
        self.mtp_hidden = state.get("mtp_hidden", "normed")
        self.set_mode("student")
