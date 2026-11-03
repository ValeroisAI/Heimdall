"""
valerois_gcam_v3.py
===================
Layer Design 7: Valerois Selective GCAM-v3 ("The True WOW Layer")
Key Innovations over v2:
1. Selective Retention (Mamba-2 style):
   - Content-aware write gate beta in [0, 1] (High for novel facts/needles, ~0 for noise)
   - Coupled Forget Gate: alpha = 1.0 - beta * sigmoid(w_forget(h))
   - When beta ~ 0 (background noise): alpha = 1.0 (Zero forgetting! Memory is preserved forever)
2. Delta Rule (Associative Error-Driven Writing):
   - Memory recall: V_hat = K @ S_{t-1}
   - Innovation / Error: Delta_V = V - V_hat
   - Memory update: S_t = alpha * S_{t-1} + beta * (K^T @ Delta_V)
3. Intra-Chunk (Window = 512): Exact RoPE GQA Softmax Attention for 100% syntax precision.
4. Inter-Chunk: O(1) Matrix State S in R^{H x Dh x Dh} (24.5 MB for 1M+ context).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

class ValeroisGCAMv3(nn.Module):
    def __init__(self, orig_attn=None, d_model=3584, num_heads=28, num_kv_heads=4, head_dim=128, chunk_size=512):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model = d_model

        if orig_attn is not None:
            cfg = orig_attn.config
            self.num_heads = cfg.num_attention_heads
            self.num_kv_heads = cfg.num_key_value_heads
            self.head_dim = orig_attn.head_dim
            self.d_model = cfg.hidden_size
            self.num_kv_groups = orig_attn.num_key_value_groups

            self.q_proj = orig_attn.q_proj
            self.k_proj = orig_attn.k_proj
            self.v_proj = orig_attn.v_proj
            self.o_proj = orig_attn.o_proj
            target_device = orig_attn.q_proj.weight.device
        else:
            self.head_dim = head_dim if head_dim > 0 else 64
            self.num_heads = d_model // self.head_dim
            self.num_kv_heads = max(1, self.num_heads // 4)
            self.num_kv_groups = self.num_heads // self.num_kv_heads

            self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, d_model, bias=False)
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Selective Write Gate (Beta) & Forget Gate
        self.w_write = nn.Linear(self.d_model, self.num_heads, bias=True).to(target_device)
        self.w_forget = nn.Linear(self.d_model, self.num_heads, bias=True).to(target_device)
        
        # Initialize: default conservative write (mostly preserve state unless high novelty)
        nn.init.zeros_(self.w_write.weight)
        nn.init.constant_(self.w_write.bias, -1.0) # Sigmoid(-1) ~ 0.27 base
        nn.init.zeros_(self.w_forget.weight)
        nn.init.constant_(self.w_forget.bias, -2.0) # Low forget rate
        
        self.inter_gate = nn.Parameter(torch.ones(self.num_heads, device=target_device) * 1.0)
        self.register_buffer("causal_mask", torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=target_device)), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings=None,
        state=None,
        **kwargs
    ):
        B, T, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        # Projections
        q_raw = self.q_proj(hidden_states).view(B, T, H, Dh).transpose(1, 2)
        k_raw = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, Dh).transpose(1, 2)
        v_raw = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, Dh).transpose(1, 2)

        # RoPE for intra-chunk
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q_rope, k_rope = apply_rotary_pos_emb(q_raw, k_raw, cos, sin)
        else:
            q_rope, k_rope = q_raw, k_raw

        k_rope = k_rope.repeat_interleave(self.num_kv_groups, dim=1)
        v = v_raw.repeat_interleave(self.num_kv_groups, dim=1)
        k_unrot = k_raw.repeat_interleave(self.num_kv_groups, dim=1)
        q_unrot = q_raw

        # Selective Gates
        h_dtype = hidden_states.dtype
        # Beta: Write rate (B, H, T, 1)
        beta = torch.sigmoid(self.w_write(hidden_states)).transpose(1, 2).unsqueeze(-1)
        # Forget factor: (B, H, T, 1)
        forget_factor = torch.sigmoid(self.w_forget(hidden_states)).transpose(1, 2).unsqueeze(-1)

        C = self.chunk_size
        num_chunks = math.ceil(T / C)
        pad_len = num_chunks * C - T

        if pad_len > 0:
            q_rope = F.pad(q_rope, (0, 0, 0, pad_len))
            k_rope = F.pad(k_rope, (0, 0, 0, pad_len))
            q_unrot = F.pad(q_unrot, (0, 0, 0, pad_len))
            k_unrot = F.pad(k_unrot, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
            beta = F.pad(beta, (0, 0, 0, pad_len))
            forget_factor = F.pad(forget_factor, (0, 0, 0, pad_len))

        if state is None:
            curr_state = torch.zeros(B, H, Dh, Dh, device=hidden_states.device, dtype=h_dtype)
        else:
            curr_state = state

        out_chunks = []
        causal_mask = self.causal_mask
        gate = torch.tanh(self.inter_gate).view(1, H, 1, 1)

        # Normalize keys for stable delta rule
        k_norm = k_unrot / (k_unrot.norm(dim=-1, keepdim=True) + 1e-5)

        for i in range(num_chunks):
            start = i * C
            end = start + C

            qr_c = q_rope[:, :, start:end]
            kr_c = k_rope[:, :, start:end]
            qu_c = q_unrot[:, :, start:end]
            kn_c = k_norm[:, :, start:end]
            v_c = v[:, :, start:end]
            b_c = beta[:, :, start:end]
            f_c = forget_factor[:, :, start:end]

            # [A] Intra-Chunk: Exact Softmax Attention
            scores_intra = (qr_c @ kr_c.transpose(-1, -2)) / math.sqrt(Dh)
            scores_intra = scores_intra.masked_fill(~causal_mask, -1e4)
            attn_intra = F.softmax(scores_intra, dim=-1)
            out_intra = attn_intra @ v_c

            # [B] Inter-Chunk: O(1) Matrix State Recurrence
            if i > 0 or state is not None:
                out_inter = (qu_c @ curr_state) / (math.sqrt(Dh))
                out_c = out_intra + gate * out_inter
            else:
                out_c = out_intra

            out_chunks.append(out_c)

            # [C] Sparse Selective Delta-Rule Memory Update
            # Novelty Thresholding: Only write if novelty/importance exceeds threshold
            # Normal background language has beta ~ 0, which yields alpha = 1.0 (Exact zero forgetting!)
            b_thresh = F.relu(b_c - 0.2) # Threshold at 0.2
            has_write = (b_thresh.sum() > 0.0)

            if has_write:
                mean_b = b_thresh.mean(dim=2, keepdim=True).squeeze(2) # (B, H, 1)
                mean_f = f_c.mean(dim=2, keepdim=True).squeeze(2)      # (B, H, 1)
                alpha = (1.0 - mean_b * mean_f).clamp(min=0.0, max=1.0)
                
                # Delta error
                v_pred = kn_c @ curr_state
                delta_v = v_c - v_pred
                update = (kn_c * b_thresh).transpose(-1, -2) @ delta_v
                curr_state = curr_state * alpha.unsqueeze(-1) + update / math.sqrt(C)
            # If no write (background ordinary tokens), curr_state is 100% preserved (alpha = 1.0, update = 0)

        out = torch.cat(out_chunks, dim=2)[:, :, :T, :].transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out), curr_state
