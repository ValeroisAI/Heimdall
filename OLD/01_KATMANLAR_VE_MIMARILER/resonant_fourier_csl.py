"""
resonant_fourier_csl.py
=======================
Layer Design 3: Resonant Fourier CSL (Harmonic Spectral State)
- Evaluates harmonic / periodic resonance for catching repetitive programming structures (loops, scopes, indentation).
- Parameterized with learned complex poles: z_k = r_k * exp(i * theta_k).
- O(N) causal filtering, O(1) state representation (complex phasor state).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResonantFourierCSL(nn.Module):
    def __init__(self, d_model=3584, num_resonators=64, expansion_factor=2):
        super().__init__()
        self.d_model = d_model
        self.num_resonators = num_resonators
        self.hidden_dim = int(d_model * expansion_factor)
        
        self.norm = nn.RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.hidden_dim * 2, bias=False)
        
        # Complex resonant poles: r in (0, 1), theta in [0, pi]
        # Real & Imaginary damping / frequencies
        self.log_r = nn.Parameter(torch.empty(self.hidden_dim).uniform_(-2.0, -0.05)) # r = exp(log_r) < 1
        self.theta = nn.Parameter(torch.empty(self.hidden_dim).uniform_(0.0, math.pi))
        
        self.out_proj = nn.Linear(self.hidden_dim, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, hidden_states, state=None, **kwargs):
        """
        hidden_states: (B, T, D)
        state: complex tensor (B, hidden_dim) for O(1) phasor accumulation
        """
        B, T, D = hidden_states.shape
        residual = hidden_states
        h = self.norm(hidden_states)
        
        projected = self.in_proj(h)
        x, gate = projected.chunk(2, dim=-1) # (B, T, hidden_dim)
        
        # Complex pole: z = r * (cos(theta) + i*sin(theta))
        r = torch.exp(self.log_r).clamp(0.01, 0.999)
        cos_t = torch.cos(self.theta)
        sin_t = torch.sin(self.theta)
        z_real = (r * cos_t).view(1, 1, self.hidden_dim)
        z_imag = (r * sin_t).view(1, 1, self.hidden_dim)
        
        if state is not None and T == 1:
            s_real, s_imag = state
            x_step = x[:, 0:1, :]
            # Complex multiplication: (s_r + i*s_i) * (z_r + i*z_i) + x
            new_s_real = (s_real * z_real - s_imag * z_imag) + x_step
            new_s_imag = (s_real * z_imag + s_imag * z_real)
            out_real = new_s_real
            new_state = (new_s_real, new_s_imag)
        else:
            # Parallel loop / recurrent scan
            s_r = torch.zeros(B, 1, self.hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
            s_i = torch.zeros(B, 1, self.hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype)
            out_list = []
            for t in range(T):
                xt = x[:, t:t+1, :]
                s_r_next = (s_r * z_real - s_i * z_imag) + xt
                s_i_next = (s_r * z_imag + s_i * z_real)
                s_r, s_i = s_r_next, s_i_next
                out_list.append(s_r)
            out_real = torch.cat(out_list, dim=1)
            new_state = (s_r, s_i)
            
        out = self.act(out_real) * torch.sigmoid(gate)
        output = self.out_proj(out) + residual
        return output, new_state
