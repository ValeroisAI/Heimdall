import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_swish_fwd_kernel(
    X_ptr, Y_ptr, W_ptr,
    stride_x_row, stride_x_col,
    stride_y_row, stride_y_col,
    N, eps,
    BLOCK_N: tl.constexpr
):
    row_idx = tl.program_id(0)
    X_ptr += row_idx * stride_x_row
    Y_ptr += row_idx * stride_y_row

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + cols * stride_x_col, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # RMSNorm
    x_norm = x * rstd * w
    
    # Swish (SiLU): x * sigmoid(x)
    swish = x_norm * tl.sigmoid(x_norm)
    
    tl.store(Y_ptr + cols * stride_y_col, swish.to(X_ptr.dtype.element_ty), mask=mask)

class FusedRMSNormSwish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        M = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        N = x.shape[-1]
        x_flat = x.view(-1, N)
        y = torch.empty_like(x_flat)
        
        BLOCK_N = triton.next_power_of_2(N)
        
        _rmsnorm_swish_fwd_kernel[(M,)](
            x_flat, y, weight,
            x_flat.stride(0), x_flat.stride(1),
            y.stride(0), y.stride(1),
            N, eps,
            BLOCK_N=BLOCK_N
        )
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return y.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Fallback to PyTorch autograd for backward for simplicity and stability on ROCm unless strictly needed
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            weight = weight.detach().requires_grad_(True)
            # Recompute
            var = x.pow(2).mean(-1, keepdim=True)
            x_norm = x * torch.rsqrt(var + eps) * weight
            y = x_norm * torch.sigmoid(x_norm)
            y.backward(grad_output)
        return x.grad, weight.grad, None

def triton_fused_rmsnorm_swish(x, weight, eps=1e-5):
    if x.is_cuda and triton is not None:
        return FusedRMSNormSwish.apply(x, weight, eps)
    else:
        var = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + eps) * weight
        return x_norm * torch.sigmoid(x_norm)
