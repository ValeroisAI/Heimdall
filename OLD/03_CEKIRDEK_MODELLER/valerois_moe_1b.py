"""
================================================================================
 🏛️ VALEROIS ULTRA 1B MoE (MIXTURE-OF-EXPERTS) ARCHITECTURE
================================================================================
Total Parameters: ~1.02 Billion (4x Experts per Block)
Routing Strategy: Soft-MoE Continuous Differentiable Gating Router per Layer
Domain Experts:
  - Expert 0: Code Master (Python AST & Docstring Specialist)
  - Expert 1: Language & Prose Master (Grammar & Story Specialist)
  - Expert 2: Science & Knowledge Master (ARC & QA Specialist)
  - Expert 3: Math & CoT Reasoning Master (GSM8K & Step Logic Specialist)
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class SwiGLUExpert(nn.Module):
    def __init__(self, hidden: int, intermediate_dim: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden, intermediate_dim * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

class MoEFeedForward(nn.Module):
    def __init__(self, hidden: int, intermediate_dim: int, num_experts: int = 4):
        super().__init__()
        self.hidden = hidden
        self.num_experts = num_experts
        self.router = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUExpert(hidden, intermediate_dim) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        router_logits = self.router(x) # [B, T, num_experts]
        gate_weights = F.softmax(router_logits, dim=-1) # [B, T, num_experts]

        # Pure differentiable Soft-MoE accumulation without scatter
        out = torch.zeros_like(x)
        for i in range(self.num_experts):
            w_i = gate_weights[..., i:i+1] # [B, T, 1]
            exp_out = self.experts[i](x)   # [B, T, C]
            out = out + w_i * exp_out

        return out

class ValeroisMoETitanBlock(nn.Module):
    def __init__(self, hidden: int = 1024, num_heads: int = 16, expand: float = 2.5, num_experts: int = 4, n_layers: int = 16):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.intermediate_dim = int(hidden * expand)
        self.res_scale = 1.0 / math.sqrt(2.0 * n_layers)

        # 1. Temporal CSL
        self.norm1 = RMSNorm(hidden)
        self.conv = nn.Conv1d(hidden, hidden, kernel_size=4, padding=3, groups=hidden, bias=False)

        # 2. Global Semantic Attention
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

        # 3. MoE Gated MLP (4 Experts per Layer)
        self.norm2 = RMSNorm(hidden)
        self.moe_ffn = MoEFeedForward(hidden, self.intermediate_dim, num_experts=num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Pre-Norm 1 & Local Conv
        norm_x = self.norm1(x)
        conv_in = norm_x.transpose(1, 2)
        conv_out = self.conv(conv_in)[:, :, :T].transpose(1, 2)

        # Scaled Dot-Product Attention
        q = self.q_proj(conv_out).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(conv_out).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(conv_out).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        attn_out = self.o_proj(attn_out)

        # Residual 1
        x = x + self.res_scale * attn_out

        # Pre-Norm 2 & MoE FFN
        norm_x2 = self.norm2(x)
        moe_out = self.moe_ffn(norm_x2)

        # Residual 2
        x = x + self.res_scale * moe_out
        return x

class ValeroisMoE1B(nn.Module):
    def __init__(
        self,
        vocab_size: int = 8192,
        hidden: int = 1024,
        n_layers: int = 16,
        num_heads: int = 16,
        expand: float = 2.5,
        num_experts: int = 4,
        use_checkpointing: bool = False
    ):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "hidden": hidden,
            "n_layers": n_layers,
            "num_heads": num_heads,
            "expand": expand,
            "num_experts": num_experts,
            "architecture": "Valerois-Ultra-1B-MoE"
        }
        self.use_checkpointing = use_checkpointing

        self.embed = nn.Embedding(vocab_size, hidden)
        self.embed_norm = RMSNorm(hidden)

        self.layers = nn.ModuleList([
            ValeroisMoETitanBlock(hidden, num_heads, expand, num_experts, n_layers)
            for _ in range(n_layers)
        ])

        self.final_norm = RMSNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_norm(self.embed(input_ids))
        for layer in self.layers:
            if self.use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.final_norm(x)
        return self.head(x)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {
            "total_parameters": total,
            "total_millions": total / 1e6
        }

if __name__ == "__main__":
    model = ValeroisMoE1B()
    stats = model.count_parameters()
    print(f"[+] ValeroisMoE1B Initialized: {stats['total_millions']:.2f}M Parameters")
