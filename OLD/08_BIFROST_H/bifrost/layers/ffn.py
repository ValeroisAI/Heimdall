import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """w12 tek matris (gate + up birleşik), w3 çıkış."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * up)
