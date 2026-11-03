"""
================================================================================
 ⚡ VALEROIS NATIVE BITNET b1.58 FUSED TRITON GEMM ENGINE (AMD ROCm 7.2)
================================================================================
 Direct in-register BitNet computing without ANY intermediate BF16 weight allocations!
   - Weight storage: Strictly 2-bit packed ternary uint8 (4 weights per byte).
   - Memory footprint: 0.25 bytes per parameter (8x smaller than BF16!).
   - In-Register Unpacking: Micro-arithmetic logic directly in GPU registers/LDS.
   - Hardware: AMD Radeon RX 9070 XT (RDNA 4 AI Accelerators).
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Optional, Tuple


# ==============================================================================
# SECTION 1: TRITON GPU FORWARD & BACKWARD KERNELS
# ==============================================================================

@triton.jit
def _bitnet_gemm_forward_kernel(
    x_ptr, w_packed_ptr, out_ptr, scale_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk_packed,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K_PACKED: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    K_packed = K // 4
    
    for k_block in range(0, K_packed, BLOCK_K_PACKED):
        offs_kp = k_block + tl.arange(0, BLOCK_K_PACKED)
        
        # 1. Load packed weights: (BLOCK_N, BLOCK_K_PACKED) as uint8
        w_mask = (offs_n[:, None] < N) & (offs_kp[None, :] < K_packed)
        w_packed = tl.load(w_packed_ptr + offs_n[:, None] * stride_wn + offs_kp[None, :] * stride_wk_packed, mask=w_mask, other=0)
        
        # 2. In-register bitwise unpacking to ternary {-1, 0, +1}
        # 00 -> 0, 01 -> +1, 10 -> -1
        p = w_packed.to(tl.int32)
        c0 = p & 3
        c1 = (p >> 2) & 3
        c2 = (p >> 4) & 3
        c3 = (p >> 6) & 3
        
        val0 = ((c0 & 1) - ((c0 >> 1) & 1)).to(tl.bfloat16)
        val1 = ((c1 & 1) - ((c1 >> 1) & 1)).to(tl.bfloat16)
        val2 = ((c2 & 1) - ((c2 >> 1) & 1)).to(tl.bfloat16)
        val3 = ((c3 & 1) - ((c3 >> 1) & 1)).to(tl.bfloat16)
        
        # 3. Load interleaved input slices
        offs_k0 = offs_kp * 4
        offs_k1 = offs_k0 + 1
        offs_k2 = offs_k0 + 2
        offs_k3 = offs_k0 + 3
        
        x_mask0 = (offs_m[:, None] < M) & (offs_k0[None, :] < K)
        x_mask1 = (offs_m[:, None] < M) & (offs_k1[None, :] < K)
        x_mask2 = (offs_m[:, None] < M) & (offs_k2[None, :] < K)
        x_mask3 = (offs_m[:, None] < M) & (offs_k3[None, :] < K)
        
        x0 = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k0[None, :] * stride_xk, mask=x_mask0, other=0.0)
        x1 = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k1[None, :] * stride_xk, mask=x_mask1, other=0.0)
        x2 = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k2[None, :] * stride_xk, mask=x_mask2, other=0.0)
        x3 = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k3[None, :] * stride_xk, mask=x_mask3, other=0.0)
        
        # 4. In-register matrix accumulation
        acc += tl.dot(x0, tl.trans(val0))
        acc += tl.dot(x1, tl.trans(val1))
        acc += tl.dot(x2, tl.trans(val2))
        acc += tl.dot(x3, tl.trans(val3))
        
    # Per-channel scaling
    scale = tl.load(scale_ptr + offs_n, mask=(offs_n < N), other=1.0)
    out = (acc * scale[None, :]).to(tl.bfloat16)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, out, mask=out_mask)


@triton.jit
def _bitnet_gemm_backward_dx_kernel(
    dy_ptr, w_packed_ptr, dx_ptr, scale_ptr,
    M, N, K,
    stride_dym, stride_dyn,
    stride_wn, stride_wk_packed,
    stride_dxm, stride_dxk,
    BLOCK_M: tl.constexpr, BLOCK_K_PACKED: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_kp = pid_k * BLOCK_K_PACKED + tl.arange(0, BLOCK_K_PACKED)
    K_packed = K // 4
    
    acc0 = tl.zeros((BLOCK_M, BLOCK_K_PACKED), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_K_PACKED), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_K_PACKED), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_K_PACKED), dtype=tl.float32)
    
    for n_block in range(0, N, BLOCK_N):
        offs_n = n_block + tl.arange(0, BLOCK_N)
        
        scale = tl.load(scale_ptr + offs_n, mask=(offs_n < N), other=0.0)
        dy_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        dy = tl.load(dy_ptr + offs_m[:, None] * stride_dym + offs_n[None, :] * stride_dyn, mask=dy_mask, other=0.0)
        dy_scaled = (dy * scale[None, :]).to(tl.bfloat16)
        
        w_mask = (offs_n[:, None] < N) & (offs_kp[None, :] < K_packed)
        w_packed = tl.load(w_packed_ptr + offs_n[:, None] * stride_wn + offs_kp[None, :] * stride_wk_packed, mask=w_mask, other=0)
        
        p = w_packed.to(tl.int32)
        c0 = p & 3
        c1 = (p >> 2) & 3
        c2 = (p >> 4) & 3
        c3 = (p >> 6) & 3
        
        val0 = ((c0 & 1) - ((c0 >> 1) & 1)).to(tl.bfloat16)
        val1 = ((c1 & 1) - ((c1 >> 1) & 1)).to(tl.bfloat16)
        val2 = ((c2 & 1) - ((c2 >> 1) & 1)).to(tl.bfloat16)
        val3 = ((c3 & 1) - ((c3 >> 1) & 1)).to(tl.bfloat16)
        
        acc0 += tl.dot(dy_scaled, val0)
        acc1 += tl.dot(dy_scaled, val1)
        acc2 += tl.dot(dy_scaled, val2)
        acc3 += tl.dot(dy_scaled, val3)
        
    offs_k0 = offs_kp * 4
    offs_k1 = offs_k0 + 1
    offs_k2 = offs_k0 + 2
    offs_k3 = offs_k0 + 3
    
    m_mask = offs_m[:, None] < M
    tl.store(dx_ptr + offs_m[:, None] * stride_dxm + offs_k0[None, :] * stride_dxk, acc0.to(tl.bfloat16), mask=(m_mask & (offs_k0[None, :] < K)))
    tl.store(dx_ptr + offs_m[:, None] * stride_dxm + offs_k1[None, :] * stride_dxk, acc1.to(tl.bfloat16), mask=(m_mask & (offs_k1[None, :] < K)))
    tl.store(dx_ptr + offs_m[:, None] * stride_dxm + offs_k2[None, :] * stride_dxk, acc2.to(tl.bfloat16), mask=(m_mask & (offs_k2[None, :] < K)))
    tl.store(dx_ptr + offs_m[:, None] * stride_dxm + offs_k3[None, :] * stride_dxk, acc3.to(tl.bfloat16), mask=(m_mask & (offs_k3[None, :] < K)))


# ==============================================================================
# SECTION 2: BIT PACKING & DISPATCH HELPERS
# ==============================================================================

def pack_ternary(w_ternary: torch.Tensor) -> torch.Tensor:
    """
    Packs a ternary tensor with values in {-1, 0, 1} into 2-bit packed uint8 format.
    Output shape: (N, K // 4).
    """
    N, K = w_ternary.shape
    assert K % 4 == 0, f"Inner dimension K ({K}) must be divisible by 4 for 2-bit packing."
    
    w_code = torch.zeros_like(w_ternary, dtype=torch.uint8)
    w_code[w_ternary == 1] = 1
    w_code[w_ternary == -1] = 2
    
    c0 = w_code[:, 0::4]
    c1 = w_code[:, 1::4] << 2
    c2 = w_code[:, 2::4] << 4
    c3 = w_code[:, 3::4] << 6
    return (c0 | c1 | c2 | c3).contiguous()


def quantize_and_pack(w_float: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes full-precision weights to ternary {-1, 0, 1} and dynamic scale,
    then packs into 2-bit uint8.
    """
    gamma = w_float.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
    w_scaled = torch.clamp(torch.round(w_float / gamma), -1.0, 1.0)
    packed = pack_ternary(w_scaled)
    scale = gamma.squeeze(-1).to(torch.bfloat16).contiguous()
    return packed, scale


# ==============================================================================
# SECTION 3: PYTORCH AUTOGRAD FUNCTION
# ==============================================================================

class ValeroisNativeBitNetFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, packed_w: torch.Tensor, scale: torch.Tensor):
        # x: (..., K) -> flatten to (M, K)
        orig_shape = x.shape
        K = orig_shape[-1]
        x_2d = x.reshape(-1, K).contiguous()
        M = x_2d.shape[0]
        N = packed_w.shape[0]
        
        ctx.save_for_backward(x_2d, packed_w, scale)
        ctx.orig_shape = orig_shape
        ctx.M = M
        ctx.N = N
        ctx.K = K
        
        out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
        
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K_PACKED = 32
        
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _bitnet_gemm_forward_kernel[grid](
            x_2d, packed_w, out, scale,
            M, N, K,
            x_2d.stride(0), x_2d.stride(1),
            packed_w.stride(0), packed_w.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K_PACKED=BLOCK_K_PACKED,
            num_warps=8
        )
        
        return out.reshape(*orig_shape[:-1], N)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_2d, packed_w, scale = ctx.saved_tensors
        M, N, K = ctx.M, ctx.N, ctx.K
        
        dy_2d = grad_output.reshape(-1, N).contiguous()
        
        dx = torch.empty((M, K), device=dy_2d.device, dtype=torch.bfloat16)
        
        BLOCK_M = 64
        BLOCK_K_PACKED = 32
        BLOCK_N = 128
        
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K // 4, BLOCK_K_PACKED))
        _bitnet_gemm_backward_dx_kernel[grid](
            dy_2d, packed_w, dx, scale,
            M, N, K,
            dy_2d.stride(0), dy_2d.stride(1),
            packed_w.stride(0), packed_w.stride(1),
            dx.stride(0), dx.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_K_PACKED=BLOCK_K_PACKED, BLOCK_N=BLOCK_N,
            num_warps=8
        )
        
        # Scale gradient (optional/analytical)
        grad_scale = None
        if scale.requires_grad:
            # d_scale_n = sum_m (dy_m,n * raw_out_m,n / scale_n)
            grad_scale = torch.zeros_like(scale)
            
        dx_reshaped = dx.reshape(ctx.orig_shape)
        return dx_reshaped, None, grad_scale


# ==============================================================================
# SECTION 4: NATIVE BITNET LINEAR MODULE WITH ADAPTER SUBSPACE
# ==============================================================================

class ValeroisNativeBitLinear(nn.Module):
    """
    100% Native BitNet Linear Layer:
    - Base weights stored as 2-bit packed uint8 (0.25 bytes/param).
    - ZERO BF16 weight allocation in global memory.
    - High-rank trainable subspace adapter (Rank-16/32) for full gradient learning.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        lora_alpha: float = 32.0,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = lora_alpha / max(1, rank)
        self.dtype = dtype

        dev = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Native 2-bit Packed Base Weight: 0.25 bytes per parameter!
        self.register_buffer("packed_weight", torch.zeros(out_features, in_features // 4, dtype=torch.uint8, device=dev))
        self.register_buffer("weight_scale", torch.ones(out_features, dtype=dtype, device=dev))

        # 2. Subspace Adapter for High-Velocity Gradient Learning
        if rank > 0:
            self.adapter_A = nn.Parameter(torch.empty(in_features, rank, dtype=dtype, device=dev))
            self.adapter_B = nn.Parameter(torch.zeros(rank, out_features, dtype=dtype, device=dev))
            nn.init.kaiming_uniform_(self.adapter_A, a=math.sqrt(5))
            nn.init.zeros_(self.adapter_B)
        else:
            self.register_parameter("adapter_A", None)
            self.register_parameter("adapter_B", None)

    @torch.no_grad()
    def initialize_from_float(self, float_weight: torch.Tensor):
        """Quantizes and packs full precision weight into native 2-bit format."""
        packed, scale = quantize_and_pack(float_weight)
        self.packed_weight.copy_(packed)
        self.weight_scale.copy_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Direct Native Triton BitNet Forward Pass (No BF16 weight allocation!)
        base_out = ValeroisNativeBitNetFunction.apply(x, self.packed_weight, self.weight_scale)
        
        # 2. Subspace Adapter Addition
        if self.rank > 0 and self.adapter_A is not None:
            adapter_out = torch.matmul(torch.matmul(x, self.adapter_A), self.adapter_B) * self.scaling
            return base_out + adapter_out
            
        return base_out
