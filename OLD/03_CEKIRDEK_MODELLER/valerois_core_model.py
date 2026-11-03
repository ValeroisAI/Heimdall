"""
valerois_core_model.py
======================
Valerois-Qwen 1.78B Hibrit Model Mimarisi.
Chunked Causal Attention (Yerel Hassasiyet + RoPE) +
O(1) Valerois Recurrent State Memory (Sinirsiz Baglam + Gereklilik Kapisi).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(torch.bfloat16) * self.weight

class ValeroisProductionGCAM(nn.Module):
    """
    Uretim Sinifi Valerois GCAM Katmani.
    - Intra-chunk: Causal Softmax Attention with RoPE (Yuksek yerel hassasiyet, matematik ve kodlama)
    - Inter-chunk: O(1) Sabit Boyutlu Durum Matrisi S_t = gamma * S_{t-1} + (K * nec)^T V
    - Bellek: 1 Milyon token icin katman basina sadece 393 KB (Tum model: 11 MB sabit durum)!
    """
    def __init__(self, d_model=1536, num_heads=12, num_kv_heads=2, head_dim=128, chunk_size=512):
        super().__init__()
        self.chunk_size = chunk_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.d_model = d_model
        self.num_kv_groups = num_heads // num_kv_heads # 6

        # DeepSeek-R1'den aktarilan projeksiyonlar
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=True, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=True, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=True, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=False, dtype=torch.bfloat16)

        # Valerois Ozgun Kapilar (Gereklilik Puani & Bellek Saklama)
        self.w_necessity = nn.Linear(d_model, num_heads, bias=True, dtype=torch.bfloat16)
        self.w_decay = nn.Linear(d_model, num_heads, bias=True, dtype=torch.bfloat16)

        # Baslangic durumlari: Gecis serbest (necessity ~ 0.95, decay ~ 0.88)
        nn.init.zeros_(self.w_necessity.weight)
        nn.init.constant_(self.w_necessity.bias, 3.0)
        nn.init.zeros_(self.w_decay.weight)
        nn.init.constant_(self.w_decay.bias, 2.0)

        # Chunklar arasi kontrollu hafiza koprusu (Baslangicta 0.0 - Model ilk andan itibaren %100 akillidir)
        self.inter_gate = nn.Parameter(torch.zeros(num_heads, dtype=torch.bfloat16))
        self.register_buffer("causal_mask", torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool)), persistent=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        H, Dh = self.num_heads, self.head_dim

        q_raw = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)
        k_raw = self.k_proj(x).view(B, T, self.num_kv_heads, Dh).transpose(1, 2)
        v_raw = self.v_proj(x).view(B, T, self.num_kv_heads, Dh).transpose(1, 2)

        # Yerel dikkat icin RoPE rotasyonu
        q_rope, k_rope = apply_rotary_pos_emb(q_raw, k_raw, cos, sin)

        k_rope = k_rope.repeat_interleave(self.num_kv_groups, dim=1)
        v = v_raw.repeat_interleave(self.num_kv_groups, dim=1)
        k_unrot = k_raw.repeat_interleave(self.num_kv_groups, dim=1)
        q_unrot = q_raw

        necessity = torch.sigmoid(self.w_necessity(x)).transpose(1, 2).unsqueeze(-1)
        decay = torch.sigmoid(self.w_decay(x)).transpose(1, 2).unsqueeze(-1)

        C = self.chunk_size
        num_chunks = math.ceil(T / C)
        pad_len = num_chunks * C - T

        if pad_len > 0:
            q_rope = F.pad(q_rope, (0, 0, 0, pad_len))
            k_rope = F.pad(k_rope, (0, 0, 0, pad_len))
            q_unrot = F.pad(q_unrot, (0, 0, 0, pad_len))
            k_unrot = F.pad(k_unrot, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            necessity = F.pad(necessity, (0, 0, 0, pad_len))
            decay = F.pad(decay, (0, 0, 0, pad_len), value=1.0)

        state = torch.zeros(B, H, Dh, Dh, device=x.device, dtype=torch.bfloat16)
        out_chunks = []
        causal_mask = self.causal_mask

        gate = torch.tanh(self.inter_gate).view(1, H, 1, 1)

        for i in range(num_chunks):
            start = i * C
            end = start + C

            qr_c = q_rope[:, :, start:end]
            kr_c = k_rope[:, :, start:end]
            qu_c = q_unrot[:, :, start:end]
            ku_c = k_unrot[:, :, start:end]
            v_c = v[:, :, start:end]
            nec_c = necessity[:, :, start:end]
            dec_c = decay[:, :, start:end]

            # 1. Chunk Ici Hassas Dikkat (Softmax + RoPE)
            scores_intra = (qr_c @ kr_c.transpose(-1, -2)) / math.sqrt(Dh)
            scores_intra = scores_intra.masked_fill(~causal_mask, -1e4)
            attn_intra = F.softmax(scores_intra, dim=-1)
            out_intra = attn_intra @ (v_c * nec_c)

            # 2. Chunklar Arasi O(1) Sabit Durum Hafizasi
            if i > 0:
                out_inter = (qu_c @ state) / (Dh * math.sqrt(C))
                out_c = out_intra + gate * out_inter
            else:
                out_c = out_intra

            out_chunks.append(out_c)

            # 3. Rekurent Durum Guncellemesi (Gereklilik Puani Agirlikli)
            kv = (ku_c * nec_c).transpose(-1, -2) @ v_c
            chunk_decay = dec_c.mean(dim=2, keepdim=True).squeeze(2)
            state = state * chunk_decay.unsqueeze(-1) + kv

        out = torch.cat(out_chunks, dim=2)[:, :, :T, :].transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out)

class QwenMLP(nn.Module):
    def __init__(self, hidden_size=1536, intermediate_size=8960):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class ValeroisProductionBlock(nn.Module):
    def __init__(self, layer_idx, chunk_size=512):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(1536)
        self.self_attn = ValeroisProductionGCAM(chunk_size=chunk_size)
        self.post_attention_layernorm = RMSNorm(1536)
        self.mlp = QwenMLP()

    def forward(self, x, cos, sin):
        residual = x
        x = residual + self.self_attn(self.input_layernorm(x), cos, sin)
        residual = x
        x = residual + self.mlp(self.post_attention_layernorm(x))
        return x

class ValeroisQwenProductionModel(nn.Module):
    def __init__(self, vocab_size=151936, num_layers=28, chunk_size=512):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, 1536, dtype=torch.bfloat16)
        self.layers = nn.ModuleList([ValeroisProductionBlock(i, chunk_size=chunk_size) for i in range(num_layers)])
        self.norm = RMSNorm(1536)
        self.lm_head = nn.Linear(1536, vocab_size, bias=False, dtype=torch.bfloat16)

    def forward(self, input_ids, cos=None, sin=None):
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)

