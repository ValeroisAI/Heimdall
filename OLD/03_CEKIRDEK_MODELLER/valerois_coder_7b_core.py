"""
valerois_coder_7b_core.py
=========================
Valerois-Coder-7B Hibrit Mimari Motoru:
- 4-bit Dondurulmuş Qwen2.5-Coder-7B Omurgası (BitsAndBytes Linear4bit)
- O(1) Valerois GCAM Sınırsız Bağlam ve Rekurent Bellek Kapıları (BF16 Eğitilebilir)
- Doğrudan Qwen2ForCausalLM üzerine modüler nakil (In-place Grafting)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

class ValeroisCoderGCAM(nn.Module):
    """
    Qwen2.5-Coder-7B için Üretim Sınıfı GCAM Katmanı.
    - Intra-chunk: Causal Softmax Attention (Yerel Hassasiyet, Sözdizimi & Aritmetik)
    - Inter-chunk: O(1) Sabit Durum Matrisi S_t = decay * S_{t-1} + (K * nec)^T V
    - Bellek: 1 Milyon token için katman başına sadece 917 KB (Tüm 7B model için 25.6 MB sabit durum)!
    """
    def __init__(self, orig_attn, chunk_size=512):
        super().__init__()
        self.chunk_size = chunk_size
        cfg = orig_attn.config
        self.num_heads = cfg.num_attention_heads                     # 28
        self.num_kv_heads = cfg.num_key_value_heads                 # 4
        self.head_dim = orig_attn.head_dim                           # 128
        self.d_model = cfg.hidden_size                               # 3584
        self.num_kv_groups = orig_attn.num_key_value_groups         # 7

        # Orijinal Dondurulmuş 4-bit Projeksiyonlar
        self.q_proj = orig_attn.q_proj
        self.k_proj = orig_attn.k_proj
        self.v_proj = orig_attn.v_proj
        self.o_proj = orig_attn.o_proj

        target_device = orig_attn.q_proj.weight.device

        # Valerois Özgün Bellek Kapıları (Eğitilebilir BF16)
        self.w_necessity = nn.Linear(self.d_model, self.num_heads, bias=True, dtype=torch.bfloat16).to(target_device)
        self.w_decay = nn.Linear(self.d_model, self.num_heads, bias=True, dtype=torch.bfloat16).to(target_device)

        # Başlangıç Durumları:
        nn.init.zeros_(self.w_necessity.weight)
        nn.init.constant_(self.w_necessity.bias, 3.0) # necessity ~ 0.95
        nn.init.zeros_(self.w_decay.weight)
        nn.init.constant_(self.w_decay.bias, 2.0)     # decay ~ 0.88

        # CRITICAL FIX: inter_gate = 1.0 (tanh(1.0) = 0.76) ile başlar!
        # Böylece ilk andan itibaren chunklar arası bilgi akışı güçlü ve açıktır.
        self.inter_gate = nn.Parameter(torch.ones(self.num_heads, dtype=torch.bfloat16, device=target_device) * 1.0)
        self.register_buffer("causal_mask", torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=target_device)), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings=None,
        attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_ids=None,
        **kwargs,
    ):
        B, T, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        # 1. 4-bit Projeksiyonlar (Q, K, V)
        q_raw = self.q_proj(hidden_states).view(B, T, H, Dh).transpose(1, 2)
        k_raw = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, Dh).transpose(1, 2)
        v_raw = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, Dh).transpose(1, 2)

        # 2. RoPE Rotasyonu
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q_rope, k_rope = apply_rotary_pos_emb(q_raw, k_raw, cos, sin)
        else:
            q_rope, k_rope = q_raw, k_raw

        k_rope = k_rope.repeat_interleave(self.num_kv_groups, dim=1)
        v = v_raw.repeat_interleave(self.num_kv_groups, dim=1)
        k_unrot = k_raw.repeat_interleave(self.num_kv_groups, dim=1)
        q_unrot = q_raw

        # 3. Valerois Bellek Kapıları (BF16)
        h_bf16 = hidden_states.to(torch.bfloat16)
        necessity = torch.sigmoid(self.w_necessity(h_bf16)).transpose(1, 2).unsqueeze(-1)
        decay = torch.sigmoid(self.w_decay(h_bf16)).transpose(1, 2).unsqueeze(-1)

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

        state = torch.zeros(B, H, Dh, Dh, device=hidden_states.device, dtype=torch.bfloat16)
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

            # [A] Chunk İçi Hassas Softmax Dikkat (RoPE ile)
            scores_intra = (qr_c @ kr_c.transpose(-1, -2)) / math.sqrt(Dh)
            scores_intra = scores_intra.masked_fill(~causal_mask, -1e4)
            attn_intra = F.softmax(scores_intra, dim=-1)
            out_intra = attn_intra @ (v_c * nec_c)

            # [B] Chunklar Arası O(1) Sabit Durum Belleği
            if i > 0:
                out_inter = (qu_c @ state) / (Dh * math.sqrt(C))
                out_c = out_intra + gate * out_inter
            else:
                out_c = out_intra

            out_chunks.append(out_c)

            # [C] Rekurent Durum Güncellemesi (Gereklilik Puanı Ağırlıklı)
            kv = (ku_c * nec_c).transpose(-1, -2) @ v_c
            chunk_decay = dec_c.mean(dim=2, keepdim=True).squeeze(2)
            state = state * chunk_decay.unsqueeze(-1) + kv

        out = torch.cat(out_chunks, dim=2)[:, :, :T, :].transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out), None

def graft_valerois_to_qwen_coder(model, chunk_size=512):
    """
    Qwen2.5-Coder-7B modelinin tüm 28 katmanındaki self_attn modüllerini
    ValeroisCoderGCAM ile yerinde değiştirir (in-place grafting).
    """
    print("=" * 80)
    print(" 🏥 VALEROIS-CODER-7B HİBRİT GRAFTING BAŞLATILIYOR...")
    print("=" * 80)
    
    # 1. Tüm ana modeli dondur
    for p in model.parameters():
        p.requires_grad = False

    trainable_params = []
    
    # 2. 28 Katmana Valerois GCAM naklet
    for i, layer in enumerate(model.model.layers):
        orig_attn = layer.self_attn
        gcam = ValeroisCoderGCAM(orig_attn, chunk_size=chunk_size)
        layer.self_attn = gcam
        
        # Kapıları eğitilebilir yap
        gcam.w_necessity.weight.requires_grad = True
        gcam.w_necessity.bias.requires_grad = True
        gcam.w_decay.weight.requires_grad = True
        gcam.w_decay.bias.requires_grad = True
        gcam.inter_gate.requires_grad = True
        
        trainable_params.extend([
            gcam.w_necessity.weight,
            gcam.w_necessity.bias,
            gcam.w_decay.weight,
            gcam.w_decay.bias,
            gcam.inter_gate
        ])
        
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"[*] 28 Katmanın Tamamına Valerois GCAM Başarıyla Nakledildi!")
    print(f"[*] Toplam Eğitilebilir Kapı Parametresi: {num_trainable:,} (~{num_trainable/1e6:.2f}M)")
    print(f"[*] Dondurulmuş ve Korunan 4-bit Bilgi: ~7.6 Milyar Parametre (%99.9)")
    print("=" * 80)
    return model, trainable_params
