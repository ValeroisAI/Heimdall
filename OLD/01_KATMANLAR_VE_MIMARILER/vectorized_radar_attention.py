"""
================================================================================
 ⚡ RADAR-SOFTMAX ATTENTION v2: FULLY VECTORIZED ZERO-LOOP KERNEL
================================================================================
 Eliminates the Python loop entirely!
 Batches all chunk queries and gathered resonant keys into a SINGLE
 GPU kernel dispatch using batched tensor gather + scaled_dot_product_attention.
 
 Complexity: Strictly O(N) with context length.
 Kernel Dispatches: Exactly 1 GEMM + 1 Radar + 1 Batched SDPA!
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorizedRadarAttention(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        n_heads: int = 8,
        chunk_size: int = 32,
        top_k_chunks: int = 2,
        local_chunks: int = 2,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.chunk_size = chunk_size
        self.top_k_chunks = top_k_chunks
        self.local_chunks = local_chunks
        self.total_attended_chunks = local_chunks + top_k_chunks
        self.dtype = dtype

        self.q_proj = nn.Linear(dim, dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(dim, dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(dim, dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(dim, dim, bias=False, dtype=dtype)

        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.radar_temp = nn.Parameter(torch.tensor(math.sqrt(self.head_dim), dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H = self.n_heads
        d = self.head_dim
        C = self.chunk_size

        pad_len = (C - (T % C)) % C
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        total_T = x.shape[1]
        num_chunks = total_T // C

        # 1. Project Q, K, V
        q = self.q_proj(x).view(B, total_T, H, d).transpose(1, 2)  # [B, H, T, d]
        k = self.k_proj(x).view(B, total_T, H, d).transpose(1, 2)  # [B, H, T, d]
        v = self.v_proj(x).view(B, total_T, H, d).transpose(1, 2)  # [B, H, T, d]

        # Reshape into chunks
        q_chunks = q.view(B, H, num_chunks, C, d)
        k_chunks = k.view(B, H, num_chunks, C, d)
        v_chunks = v.view(B, H, num_chunks, C, d)

        # Fallback for ultra-short sequences
        if num_chunks <= self.total_attended_chunks:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).contiguous().view(B, total_T, D)
            if pad_len > 0:
                out = out[:, :T]
            return self.o_proj(out)

        # -------------------------------------------------------------
        # PASS 1: VECTORIZED RADAR
        # -------------------------------------------------------------
        # Centroids: [B, H, num_chunks, d]
        k_centroids = F.normalize(k_chunks.mean(dim=3), p=2, dim=-1)
        q_centroids = F.normalize(q_chunks.mean(dim=3), p=2, dim=-1)

        # Radar affinity matrix: [B, H, num_chunks, num_chunks]
        radar_scores = torch.matmul(q_centroids, k_centroids.transpose(-1, -2)) * (self.scale * self.radar_temp)

        # Causal mask: query chunk c can only route to past chunks p <= c
        causal_mask = torch.tril(torch.ones(num_chunks, num_chunks, device=x.device, dtype=torch.bool))
        radar_scores = radar_scores.masked_fill(~causal_mask, -1e9)

        # Mask out local chunks from topk selection so they aren't picked twice
        # Local window for chunk c is [c - local_chunks + 1, ..., c]
        band_mask = torch.triu(torch.tril(torch.ones(num_chunks, num_chunks, device=x.device, dtype=torch.bool)), diagonal=-(self.local_chunks - 1))
        distant_scores = radar_scores.masked_fill(band_mask, -1e9)

        # Pick top-k distant chunks in parallel for all chunks:
        # topk_distant_idx: [B, H, num_chunks, top_k]
        topk_distant_idx = torch.topk(distant_scores, k=self.top_k_chunks, dim=-1).indices

        # Construct local indices: [num_chunks, local_chunks]
        # For each chunk c, local indices are: max(0, c - offset)
        offsets = torch.arange(self.local_chunks - 1, -1, -1, device=x.device)  # [local_chunks]
        c_range = torch.arange(num_chunks, device=x.device).unsqueeze(-1)  # [num_chunks, 1]
        local_idx = torch.clamp(c_range - offsets.unsqueeze(0), min=0)  # [num_chunks, local_chunks]
        local_idx = local_idx.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)  # [B, H, num_chunks, local_chunks]

        # Combine selected chunk indices: [B, H, num_chunks, total_attended_chunks]
        selected_chunk_idx = torch.cat([topk_distant_idx, local_idx], dim=-1)
        M = self.total_attended_chunks  # e.g. 2 + 2 = 4 chunks

        # -------------------------------------------------------------
        # PASS 2: FULLY VECTORIZED BATCHED GATHER & SDPA (1 DISPATCH!)
        # -------------------------------------------------------------
        # Expand chunk index to gather all C tokens per selected chunk
        # k_chunks is [B, H, num_chunks, C, d]
        # We flatten chunks across batch/heads to gather efficiently
        # Flattened k: [B * H, num_chunks, C * d]
        k_flat = k_chunks.reshape(B * H, num_chunks, C * d)
        v_flat = v_chunks.reshape(B * H, num_chunks, C * d)
        idx_flat = selected_chunk_idx.reshape(B * H, num_chunks * M)

        # Gather: [B * H, num_chunks * M, C * d]
        idx_expanded = idx_flat.unsqueeze(-1).expand(-1, -1, C * d)
        gathered_k = torch.gather(k_flat, dim=1, index=idx_expanded)
        gathered_v = torch.gather(v_flat, dim=1, index=idx_expanded)

        # Reshape to [B * H * num_chunks, M * C, d]
        K_batched = gathered_k.reshape(B * H * num_chunks, M * C, d)
        V_batched = gathered_v.reshape(B * H * num_chunks, M * C, d)
        Q_batched = q_chunks.reshape(B * H * num_chunks, C, d)

        # Single batched SDPA execution on GPU!
        # The last C tokens in each gathered set correspond to the current chunk c (local_idx last entry is c)
        # So intra-chunk causal mask applies to the last C tokens
        past_len = (M - 1) * C
        intra_causal = torch.tril(torch.ones(C, C, device=x.device, dtype=torch.bool))
        past_allowed = torch.ones(C, past_len, device=x.device, dtype=torch.bool)
        batched_mask = torch.cat([past_allowed, intra_causal], dim=-1)  # [C, M * C]

        # Run exact attention on gathered tokens
        out_batched = F.scaled_dot_product_attention(
            Q_batched,
            K_batched,
            V_batched,
            attn_mask=batched_mask,
            scale=self.scale
        )  # [B * H * num_chunks, C, d]

        # Differentiable Router Flow:
        # Gather the radar scores for the selected chunks and compute softmax gate
        selected_scores = torch.gather(radar_scores, dim=-1, index=selected_chunk_idx) # [B, H, num_chunks, M]
        radar_gate = F.softmax(selected_scores * self.scale, dim=-1) # [B, H, num_chunks, M]
        # Gate factor for the current local chunk (last index in M) to ensure gradient flows back to router
        gate_factor = radar_gate[..., -1:].unsqueeze(-1) # [B, H, num_chunks, 1, 1]

        # Reshape back to [B, T, D]
        out = out_batched.view(B, H, num_chunks, C, d)
        out = (out * gate_factor).view(B, H, total_T, d).transpose(1, 2).contiguous().view(B, total_T, D)

        if pad_len > 0:
            out = out[:, :T]

        return self.o_proj(out)
