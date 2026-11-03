"""
VALEROIS CSL-QV5
================

A causal sequence layer intended for *from-scratch* pretraining on normal
PyTorch CUDA/ROCm installations.  It deliberately makes no "infinite context"
or O(1)-memory claim.

The layer has three paths:

  1. causal depth-wise convolution: exact local order/syntax;
  2. chunk-summary cross attention: each token reads completed, compressed
     chunks only; and
  3. a deliberately narrow SwiGLU channel mixer.

For chunk size C and memory width M, its global mixing cost is O(T^2*M/C),
instead of full attention's O(T^2*D), and inference keeps O((T/C)*M) KV state.
That is a real, explicit compression trade-off, not O(1) magic.  Training is
fully parallel over time: there is no Python recurrence and the expensive
operations are Conv1d, GEMM, and scaled_dot_product_attention.

The file is intentionally BF16-first.  Training a model from random 4-bit
frozen weights plus LoRA is not full pretraining.  Quantize this model only
after it has learned useful BF16 weights, or use a true QAT kernel with FP32
master weights and a straight-through quantizer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Numerically stable RMSNorm which keeps its reduction in FP32."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().square().mean(dim=-1, keepdim=True)
        return (x.float() * torch.rsqrt(variance + self.eps) * self.weight.float()).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Small RoPE used only on the compressed-memory attention path."""

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, Dh], positions: [T]
        freqs = torch.outer(positions.float(), self.inv_freq.float())
        cos = freqs.cos().to(x.dtype)[None, None, :, :]
        sin = freqs.sin().to(x.dtype)[None, None, :, :]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


@dataclass(frozen=True)
class CSLQv5Cost:
    """Transparent per-layer operation/state estimates, ignoring constants."""

    local_and_channel: int
    memory_attention: int
    memory_kv_elements: int


class ChunkMemoryAttention(nn.Module):
    """
    Token-to-completed-chunk causal attention.

    A token in chunk b can attend the learned seed and summaries of chunks
    [0, b-1], never its own unfinished chunk.  Thus block averaging is causal;
    it cannot leak a later token to an earlier one.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        head_dim: Optional[int] = None,
        chunk_size: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if chunk_size < 2:
            raise ValueError("chunk_size must be >= 2")
        head_dim = head_dim or max(32, dim // (num_heads * 4))
        if head_dim % 2:
            raise ValueError("head_dim must be even")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.memory_dim = num_heads * head_dim
        self.chunk_size = chunk_size
        self.dropout = dropout

        # A narrow global path: for dim=768 and 4x64 heads, this is 0.79M
        # parameters, vs 2.36M for a conventional four D-by-D attention path.
        self.q_proj = nn.Linear(dim, self.memory_dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * self.memory_dim, bias=False)
        self.o_proj = nn.Linear(self.memory_dim, dim, bias=False)
        self.seed = nn.Parameter(torch.zeros(1, 1, dim))
        self.rope = RotaryEmbedding(head_dim)

    def _summary_memory(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return shifted causal memory and its physical token positions."""
        batch, time, dim = x.shape
        chunks = math.ceil(time / self.chunk_size)
        pad = chunks * self.chunk_size - time
        padded = F.pad(x, (0, 0, 0, pad)) if pad else x
        blocks = padded.view(batch, chunks, self.chunk_size, dim)

        # Do not dilute the final partial block with zero padding.
        counts = torch.full((chunks,), self.chunk_size, device=x.device, dtype=x.dtype)
        if pad:
            counts[-1] = self.chunk_size - pad
        summaries = blocks.sum(dim=2) / counts.view(1, chunks, 1)

        # memory[0] is a learned BOS summary; memory[b] is summary[b-1].
        memory = torch.cat((self.seed.expand(batch, -1, -1), summaries[:, :-1]), dim=1)
        positions = torch.zeros(chunks, device=x.device, dtype=torch.long)
        if chunks > 1:
            positions[1:] = torch.arange(chunks - 1, device=x.device) * self.chunk_size + (self.chunk_size - 1)
        return memory, positions

    def _causal_block_mask(self, time: int, chunks: int, device: torch.device) -> torch.Tensor:
        # True means "may attend" for scaled_dot_product_attention.
        query_block = torch.arange(time, device=device) // self.chunk_size
        memory_index = torch.arange(chunks, device=device)
        return memory_index.unsqueeze(0) <= query_block.unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, _ = x.shape
        memory, memory_pos = self._summary_memory(x)
        chunks = memory.size(1)

        q = self.q_proj(x).view(batch, time, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv_proj(memory).chunk(2, dim=-1)
        k = k.view(batch, chunks, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, chunks, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q, torch.arange(time, device=x.device))
        k = self.rope(k, memory_pos)
        mask = self._causal_block_mask(time, chunks, x.device)[None, None, :, :]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(batch, time, self.memory_dim)
        return self.o_proj(y)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor | int]:
        seed_k, seed_v = self.kv_proj(self.seed.to(device=device, dtype=dtype)).chunk(2, dim=-1)
        seed_k = seed_k.view(1, 1, self.num_heads, self.head_dim).transpose(1, 2).expand(batch_size, -1, -1, -1)
        seed_v = seed_v.view(1, 1, self.num_heads, self.head_dim).transpose(1, 2).expand(batch_size, -1, -1, -1)
        return {
            "k": self.rope(seed_k, torch.zeros(1, device=device, dtype=torch.long)),
            "v": seed_v,
            "summary_sum": torch.zeros(batch_size, self.dim, device=device, dtype=dtype),
            "within_chunk": 0,
            "position": 0,
        }

    def step(self, x: torch.Tensor, state: Dict[str, torch.Tensor | int]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor | int]]:
        """One causal token step; state grows by one tiny KV entry per chunk."""
        batch = x.size(0)
        position = int(state["position"])
        within = int(state["within_chunk"])
        q = self.q_proj(x).view(batch, self.num_heads, 1, self.head_dim)
        q = self.rope(q, torch.tensor([position], device=x.device))
        y = F.scaled_dot_product_attention(q, state["k"], state["v"], dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(batch, self.memory_dim)

        summary_sum = state["summary_sum"] + x
        new_state = dict(state)
        if within + 1 == self.chunk_size:
            summary = (summary_sum / self.chunk_size).unsqueeze(1)
            k, v = self.kv_proj(summary).chunk(2, dim=-1)
            k = k.view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
            key_pos = torch.tensor([position], device=x.device)
            new_state["k"] = torch.cat((state["k"], self.rope(k, key_pos)), dim=2)
            new_state["v"] = torch.cat((state["v"], v), dim=2)
            new_state["summary_sum"] = torch.zeros_like(summary_sum)
            new_state["within_chunk"] = 0
        else:
            new_state["summary_sum"] = summary_sum
            new_state["within_chunk"] = within + 1
        new_state["position"] = position + 1
        return self.o_proj(y), new_state


class CSLQv5Block(nn.Module):
    """A compact causal local-plus-compressed-global pretraining block."""

    def __init__(
        self,
        dim: int,
        num_layers: int,
        chunk_size: int = 64,
        num_memory_heads: int = 4,
        memory_head_dim: Optional[int] = None,
        conv_kernel: int = 7,
        ffn_multiplier: float = 1.5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if conv_kernel < 2:
            raise ValueError("conv_kernel must be >= 2")
        ffn_dim = max(32, int(dim * ffn_multiplier))
        self.dim = dim
        self.conv_kernel = conv_kernel
        self.ffn_dim = ffn_dim
        self.residual_scale = 1.0 / math.sqrt(2.0 * num_layers)

        self.local_norm = RMSNorm(dim)
        self.local_in = nn.Linear(dim, 2 * dim, bias=False)
        self.local_conv = nn.Conv1d(dim, dim, conv_kernel, groups=dim, bias=False)
        self.local_out = nn.Linear(dim, dim, bias=False)

        self.global_norm = RMSNorm(dim)
        self.memory = ChunkMemoryAttention(
            dim, num_memory_heads, memory_head_dim, chunk_size, dropout
        )

        self.ffn_norm = RMSNorm(dim)
        self.ffn_in = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.ffn_out = nn.Linear(ffn_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _local(self, x: torch.Tensor) -> torch.Tensor:
        u, gate = self.local_in(self.local_norm(x)).chunk(2, dim=-1)
        u = self.local_conv(F.pad(u.transpose(1, 2), (self.conv_kernel - 1, 0))).transpose(1, 2)
        return self.local_out(F.silu(gate) * u)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_scale * self.dropout(self._local(x))
        x = x + self.residual_scale * self.dropout(self.memory(self.global_norm(x)))
        u, gate = self.ffn_in(self.ffn_norm(x)).chunk(2, dim=-1)
        x = x + self.residual_scale * self.dropout(self.ffn_out(F.silu(gate) * u))
        return x

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, object]:
        return {
            "conv": torch.zeros(batch_size, self.dim, self.conv_kernel - 1, device=device, dtype=dtype),
            "memory": self.memory.init_state(batch_size, device, dtype),
        }

    def step(self, x: torch.Tensor, state: Dict[str, object]) -> Tuple[torch.Tensor, Dict[str, object]]:
        local_x = self.local_norm(x)
        u, gate = self.local_in(local_x).chunk(2, dim=-1)
        full = torch.cat((state["conv"], u.unsqueeze(-1)), dim=-1)
        conv_u = F.conv1d(full, self.local_conv.weight).squeeze(-1)
        local = self.local_out(F.silu(gate) * conv_u)
        x = x + self.residual_scale * local

        global_x = self.global_norm(x)
        global_y, memory_state = self.memory.step(global_x, state["memory"])
        x = x + self.residual_scale * global_y

        u, gate = self.ffn_in(self.ffn_norm(x)).chunk(2, dim=-1)
        x = x + self.residual_scale * self.ffn_out(F.silu(gate) * u)
        return x, {"conv": full[:, :, 1:], "memory": memory_state}

    def cost(self, seq_len: int, batch_size: int = 1) -> CSLQv5Cost:
        chunks = math.ceil(seq_len / self.memory.chunk_size)
        # Approximate multiply-add counts.  KV projection is performed once per
        # chunk, while Q/O and the channel mixer run once per token.
        local_channel = batch_size * (
            seq_len * (
                3 * self.dim * self.dim
                + self.conv_kernel * self.dim
                + 2 * self.dim * self.memory.memory_dim
                + 3 * self.dim * self.ffn_dim
            )
            + chunks * 2 * self.dim * self.memory.memory_dim
        )
        memory = batch_size * seq_len * chunks * self.memory.memory_dim
        kv_state = batch_size * chunks * 2 * self.memory.memory_dim
        return CSLQv5Cost(local_channel, memory, kv_state)


class ValeroisCSLQv5LM(nn.Module):
    """Small, clean language-model shell for from-scratch CSL-QV5 experiments."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 768,
        num_layers: int = 16,
        chunk_size: int = 64,
        num_memory_heads: int = 4,
        memory_head_dim: Optional[int] = None,
        conv_kernel: int = 7,
        ffn_multiplier: float = 1.5,
        dropout: float = 0.0,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            CSLQv5Block(dim, num_layers, chunk_size, num_memory_heads, memory_head_dim, conv_kernel, ffn_multiplier, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Conv1d):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))

    def init_state(self, batch_size: int, device: torch.device, dtype: Optional[torch.dtype] = None) -> List[Dict[str, object]]:
        dtype = dtype or self.embed.weight.dtype
        return [block.init_state(batch_size, device, dtype) for block in self.blocks]

    def step(self, input_id: torch.Tensor, state: List[Dict[str, object]]) -> Tuple[torch.Tensor, List[Dict[str, object]]]:
        x = self.embed(input_id)
        next_state: List[Dict[str, object]] = []
        for block, block_state in zip(self.blocks, state):
            x, block_state = block.step(x, block_state)
            next_state.append(block_state)
        return self.lm_head(self.norm(x)), next_state
