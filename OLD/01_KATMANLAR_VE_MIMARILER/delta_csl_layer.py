"""
delta_csl_layer.py
==================
Layer Design 2: Delta-CSL (Causal Sequence Layer with Learned Delta-Decay)
- Combines depthwise causal 1D convolution with data-dependent exponential decay.
- O(N) training complexity via parallel associative cumulative product.
- O(1) step inference with running state.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeltaCSLLayer(nn.Module):
    def __init__(self, d_model=3584, kernel_size=4, expansion_factor=2):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.hidden_dim = int(d_model * expansion_factor)
        
        self.norm = nn.RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.hidden_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=self.hidden_dim,
            bias=True
        )
        # Delta decay projection: predicts decay per channel based on input
        self.w_delta = nn.Linear(d_model, self.hidden_dim, bias=True)
        nn.init.zeros_(self.w_delta.weight)
        nn.init.constant_(self.w_delta.bias, 2.0) # sigmoid(2.0) ~ 0.88 decay
        
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(self.hidden_dim, d_model, bias=False)

    def forward(self, hidden_states, state=None, **kwargs):
        """
        hidden_states: (B, T, D)
        state: tuple (conv_state, running_accum) for O(1) inference
        """
        B, T, D = hidden_states.shape
        residual = hidden_states
        h = self.norm(hidden_states)
        
        # Projections
        projected = self.in_proj(h)
        x, gate = projected.chunk(2, dim=-1)
        decay = torch.sigmoid(self.w_delta(h)) # (B, T, hidden_dim)
        
        x_conv = x.transpose(1, 2) # (B, hidden_dim, T)
        
        if state is not None and T == 1:
            conv_buf, accum = state
            x_cat = torch.cat([conv_buf, x_conv], dim=-1)
            new_conv_buf = x_cat[:, :, 1:]
            conv_out = F.conv1d(x_cat, self.conv1d.weight, self.conv1d.bias, groups=self.hidden_dim).transpose(1, 2)
            
            # Recurrent delta update: accum_t = decay_t * accum_{t-1} + conv_out_t
            d_step = decay[:, 0:1, :]
            new_accum = accum * d_step + conv_out
            out = self.act(new_accum) * torch.sigmoid(gate)
            new_state = (new_conv_buf, new_accum)
        else:
            # Parallel training
            conv_out = self.conv1d(x_conv)[:, :, :T].transpose(1, 2) # (B, T, hidden_dim)
            
            # Parallel associative scan for 1D decay:
            # Simple chunked cumsum/cumprod or iterative recurrent loop
            accum = torch.zeros(B, 1, self.hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
            accum_list = []
            for t in range(T):
                c_t = conv_out[:, t:t+1, :]
                d_t = decay[:, t:t+1, :]
                accum = accum * d_t + c_t
                accum_list.append(accum)
            accum_seq = torch.cat(accum_list, dim=1)
            
            out = self.act(accum_seq) * torch.sigmoid(gate)
            new_conv_buf = x_conv[:, :, -(self.kernel_size - 1):] if self.kernel_size > 1 else None
            new_state = (new_conv_buf, accum)

        output = self.out_proj(out) + residual
        return output, new_state
