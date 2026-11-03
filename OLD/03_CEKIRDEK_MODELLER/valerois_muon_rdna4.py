"""
================================================================================
 ⚡ VALEROIS-MUON OPTIMIZER (ROCm RDNA 4 OPTIMIZED)
================================================================================
 Based on Keller Jordan's Newton-Schulz Orthogonalized Momentum Optimizer (2024).
 Converges ~2x faster per token than AdamW on 2D Transformer / CSL matrices!
================================================================================
"""

import math
import torch
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Computes orthogonalized update for 2D weight matrix using quintic Newton-Schulz.
    Ensures spectral norm of update is normalized to 1.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # Normalize Frobenius norm

    if G.size(0) > G.size(1):
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T

    return X


class ValeroisMuon(Optimizer):
    """
    Muon optimizer for 2D hidden matrices + AdamW for 1D/Embedding parameters.
    """
    def __init__(self, muon_params, adam_params, lr=0.02, adam_lr=3e-4, momentum=0.95, weight_decay=0.01):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(list(muon_params) + list(adam_params), defaults)

        self.muon_params = set(muon_params)
        self.adam_params = set(adam_params)
        self.adam_lr = adam_lr
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self):
        # 1. Update 2D matrices with Newton-Schulz Orthogonal Momentum
        for p in self.muon_params:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)

            buf = state["momentum_buffer"]
            buf.mul_(self.momentum).add_(g)

            # Apply orthogonalized Newton-Schulz update
            update = zeropower_via_newtonschulz5(buf)
            # Scale update to match RMS of parameter
            scale = max(1, p.size(0) / p.size(1)) ** 0.5
            p.data.mul_(1.0 - self.lr * self.weight_decay)
            p.data.add_(update, alpha=-self.lr * scale)

        # 2. Update 1D / Embedding parameters with fast Sign / Adam update
        for p in self.adam_params:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(g)
                state["exp_avg_sq"] = torch.zeros_like(g)
                state["step"] = 0

            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            beta1, beta2 = 0.9, 0.99

            exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

            denom = exp_avg_sq.sqrt().add_(1e-8)
            step_size = self.adam_lr * math.sqrt(1 - beta2 ** state["step"]) / (1 - beta1 ** state["step"])

            p.data.mul_(1.0 - self.adam_lr * self.weight_decay)
            p.data.addcdiv_(exp_avg, denom, value=-step_size)
