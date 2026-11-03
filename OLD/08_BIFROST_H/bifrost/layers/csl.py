"""
Bifrost CSL v2 — yerel, nedensel depthwise konvolüsyon katmanları.

ShortConv:   [B, T, D] üzerinde nedensel depthwise conv (Canon katmanı). Çıkarımda
             son (K-1) girdiyi tutan halka tampon kullanır -> O(1) cache.
BifrostCSL:  GLU-conv karıştırıcı (CSL-QV5'in yerel yolu):
             u, g = W_in x ;  u = DWConv(u) ;  y = W_out(SiLU(g) * u)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortConv(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 4, dilation: int = 1, activation: Optional[str] = None) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size >= 1 olmalı")
        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation
        self.weight = nn.Parameter(torch.empty(dim, 1, kernel_size))
        # Başlangıçta son tap'e ağırlık ver: katman kimliğe yakın başlar.
        nn.init.normal_(self.weight, std=0.02)
        with torch.no_grad():
            self.weight[:, 0, -1] += 1.0

    @property
    def span(self) -> int:
        return (self.kernel_size - 1) * self.dilation

    def _act(self, y: torch.Tensor) -> torch.Tensor:
        return F.silu(y) if self.activation == "silu" else y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        t = x.size(1)
        y = F.conv1d(F.pad(x.transpose(1, 2), (self.span, 0)), self.weight.to(x.dtype),
                     dilation=self.dilation, groups=self.dim)
        return self._act(y[..., :t].transpose(1, 2))

    def init_state(self, batch: int, device, dtype) -> torch.Tensor:
        return torch.zeros(batch, self.dim, self.span, device=device, dtype=dtype)

    def forward_stateful(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parça parça işleme (prefill): önceki parçanın son girdileri dolgu yerine kullanılır."""
        t = x.size(1)
        full = torch.cat((state, x.transpose(1, 2)), dim=-1)
        y = F.conv1d(full, self.weight.to(x.dtype), dilation=self.dilation, groups=self.dim)
        new_state = full[..., full.size(-1) - self.span:] if self.span else state
        return self._act(y[..., :t].transpose(1, 2)), new_state

    def step(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, D]; state: [B, D, span] (en eski -> en yeni)
        window = torch.cat((state, x.unsqueeze(-1)), dim=-1)
        taps = window[..., :: self.dilation] if self.dilation > 1 else window
        y = (taps * self.weight.to(x.dtype).squeeze(1)).sum(-1)
        return self._act(y), window[..., 1:]


class BifrostCSL(nn.Module):
    """GLU + nedensel depthwise conv karıştırıcı (normalizasyon blokta yapılır)."""

    def __init__(self, dim: int, kernel_size: int = 4, dilation: int = 1) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.conv = ShortConv(dim, kernel_size, dilation)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, g = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(F.silu(g) * self.conv(u))

    def init_state(self, batch: int, device, dtype) -> dict:
        return {"conv": self.conv.init_state(batch, device, dtype)}

    def forward_stateful(self, x: torch.Tensor, state: dict) -> Tuple[torch.Tensor, dict]:
        u, g = self.in_proj(x).chunk(2, dim=-1)
        u, conv_state = self.conv.forward_stateful(u, state["conv"])
        return self.out_proj(F.silu(g) * u), {"conv": conv_state}

    def step(self, x: torch.Tensor, state: dict) -> Tuple[torch.Tensor, dict]:
        u, g = self.in_proj(x).chunk(2, dim=-1)
        u, conv_state = self.conv.step(u, state["conv"])
        return self.out_proj(F.silu(g) * u), {"conv": conv_state}
