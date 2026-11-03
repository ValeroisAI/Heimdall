"""
pure_csl_layer.py
=================
Layer Design 1: Pure CSL (Causal Sequence Layer - Conv1D)
- O(N) linear time complexity during training.
- O(1) fixed-size buffer during inference (only stores K-1 past tokens).
- Zero quadratic attention matrix.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PureCSLLayer(nn.Module):
    def __init__(self, d_model=3584, kernel_size=4, expansion_factor=2):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.hidden_dim = int(d_model * expansion_factor)
        
        # Input & Output Projections
        self.in_proj = nn.Linear(d_model, self.hidden_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=self.hidden_dim, # Depthwise
            bias=True
        )
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(self.hidden_dim, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, hidden_states, state=None, **kwargs):
        """
        hidden_states: (B, T, D)
        state: Optional (B, hidden_dim, kernel_size - 1) for O(1) generation
        """
        B, T, D = hidden_states.shape
        residual = hidden_states
        h = self.norm(hidden_states)
        
        # In-proj -> Gated Linear Unit
        projected = self.in_proj(h)
        x, gate = projected.chunk(2, dim=-1)
        
        # Conv1d expects (B, C, T)
        x_conv = x.transpose(1, 2) # (B, hidden_dim, T)
        
        if state is not None and T == 1:
            # O(1) Step-by-step inference
            x_cat = torch.cat([state, x_conv], dim=-1) # (B, hidden_dim, K)
            new_state = x_cat[:, :, 1:]
            conv_out = F.conv1d(x_cat, self.conv1d.weight, self.conv1d.bias, groups=self.hidden_dim)
            x_out = conv_out.transpose(1, 2)
        else:
            # Training / Parallel forward: Causal padding
            conv_out = self.conv1d(x_conv)[:, :, :T]
            x_out = conv_out.transpose(1, 2)
            new_state = x_conv[:, :, -(self.kernel_size - 1):] if self.kernel_size > 1 else None

        # Gated activation & Output projection
        out = self.act(x_out) * torch.sigmoid(gate)
        output = self.out_proj(out) + residual
        return output, new_state
