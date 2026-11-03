"""
gated_linear_attention.py
=========================
Layer Design 5: Gated Linear Attention (GLA / RetNet / Katharopoulos)
- Removes quadratic softmax bottleneck using kernel feature maps phi(Q) and phi(K).
- Uses associative matrix multiplication: O(N) FLOPs during training instead of O(N^2).
- Recurrent matrix state S_t = alpha_t * S_{t-1} + phi(k_t)^T * v_t
- O(1) memory footprint during generation (constant state matrix H x d_k x d_v).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedLinearAttention(nn.Module):
    def __init__(self, d_model=3584, num_heads=16, head_dim=64):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        self.num_heads = d_model // head_dim
        self.val_dim = head_dim
        
        self.norm = nn.RMSNorm(d_model)
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_heads * self.val_dim, bias=False)
        
        # Data-dependent decay gate & Output gate
        self.decay_proj = nn.Linear(d_model, num_heads, bias=True)
        nn.init.zeros_(self.decay_proj.weight)
        nn.init.constant_(self.decay_proj.bias, 2.5) # sigmoid ~ 0.92
        
        self.g_proj = nn.Linear(d_model, num_heads * self.val_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.val_dim, d_model, bias=False)
        
        self.group_norm = nn.GroupNorm(num_heads, num_heads * self.val_dim)

    def phi(self, x):
        """Kernel activation: 1 + ELU(x) guarantees non-negativity without softmax"""
        return F.elu(x) + 1.0

    def forward(self, hidden_states, state=None, **kwargs):
        """
        hidden_states: (B, T, D)
        state: (B, H, head_dim, val_dim)
        """
        B, T, D = hidden_states.shape
        H, Dh, Dv = self.num_heads, self.head_dim, self.val_dim
        residual = hidden_states
        h = self.norm(hidden_states)
        
        q = self.phi(self.q_proj(h)).view(B, T, H, Dh).transpose(1, 2) # (B, H, T, Dh)
        k = self.phi(self.k_proj(h)).view(B, T, H, Dh).transpose(1, 2) # (B, H, T, Dh)
        v = self.v_proj(h).view(B, T, H, Dv).transpose(1, 2)          # (B, H, T, Dv)
        
        decay = torch.sigmoid(self.decay_proj(h)).view(B, T, H).transpose(1, 2) # (B, H, T)
        g = torch.sigmoid(self.g_proj(h)).view(B, T, H, Dv).transpose(1, 2)     # (B, H, T, Dv)
        
        if state is not None and T == 1:
            # O(1) step update
            # state: (B, H, Dh, Dv)
            d_step = decay[:, :, 0].view(B, H, 1, 1)
            k_step = k[:, :, 0, :].unsqueeze(-1) # (B, H, Dh, 1)
            v_step = v[:, :, 0, :].unsqueeze(-2) # (B, H, 1, Dv)
            q_step = q[:, :, 0, :].unsqueeze(-2) # (B, H, 1, Dh)
            
            new_state = state * d_step + (k_step @ v_step) # (B, H, Dh, Dv)
            out_step = (q_step @ new_state).squeeze(-2)    # (B, H, Dv)
            out_gated = out_step * g[:, :, 0, :]
            
            # Reshape & Output
            out_flat = out_gated.transpose(1, 2).contiguous().view(B, 1, H * Dv)
            output = self.out_proj(out_flat) + residual
            return output, new_state
        else:
            # Training / Parallel associative chunk or recurrent scan
            curr_state = torch.zeros(B, H, Dh, Dv, device=hidden_states.device, dtype=hidden_states.dtype)
            out_list = []
            for t in range(T):
                d_t = decay[:, :, t].view(B, H, 1, 1)
                k_t = k[:, :, t, :].unsqueeze(-1)
                v_t = v[:, :, t, :].unsqueeze(-2)
                q_t = q[:, :, t, :].unsqueeze(-2)
                
                curr_state = curr_state * d_t + (k_t @ v_t)
                o_t = (q_t @ curr_state).squeeze(-2) # (B, H, Dv)
                out_list.append(o_t)
                
            out_seq = torch.stack(out_list, dim=2) # (B, H, T, Dv)
            out_gated = out_seq * g
            out_flat = out_gated.transpose(1, 2).contiguous().view(B, T, H * Dv)
            output = self.out_proj(out_flat) + residual
            return output, curr_state
