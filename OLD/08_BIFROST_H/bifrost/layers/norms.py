import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm; indirgeme FP32'de yapılır (bf16 eğitimde kararlılık için)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)
