"""
Optimizasyon: Muon (gizli matrisler) + AdamW (embedding, norm, conv, kapı, skalerler)
ve zamana/adıma bağlı WSD (warmup → sabit → doğrusal düşüş) çizelgesi.

Muon: momentum güncellemesini Newton-Schulz ile yarı-ortogonalleştirir; her yönde benzer
adım büyüklüğü verir. Birleşik matrisler (qkv, gate+up) parça parça ortogonalleştirilir.
"""

import math
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn


def zeropower_ns5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz beşinci derece: G -> yaklaşık U Vᵀ. Son iki boyutta, toplu (batched) çalışır."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.float()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        s = x @ x.mT
        x = a * x + (b * s + c * (s @ s)) @ x
    if transposed:
        x = x.mT
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True,
                 weight_decay: float = 0.0, ns_steps: int = 5) -> None:
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                buf = self.state[p].setdefault("momentum", torch.zeros_like(g))
                buf.lerp_(g, 1 - beta)
                u = g.lerp(buf, beta) if group["nesterov"] else buf
                parts = getattr(p, "muon_split", 1)
                u = u.view(parts, p.size(0) // parts, -1)
                u = zeropower_ns5(u, group["ns_steps"])
                u = u * max(1.0, u.size(-2) / u.size(-1)) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u.view_as(p), alpha=-group["lr"])


def split_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """(muon, adam) grupları. Muon: blok içindeki büyük 2D matrisler. Geri kalan her şey AdamW."""
    embed_ids = {id(p) for n, p in model.named_parameters() if "embed" in n or "lm_head" in n}
    muon, adam, seen = [], [], set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim == 2 and min(p.shape) >= 32 and id(p) not in embed_ids:
            muon.append(p)
        else:
            adam.append(p)
    return muon, adam


def mark_fused(model: nn.Module) -> None:
    """Birleşik projeksiyonları Muon için işaretle (qkv: 3 parça, SwiGLU w12: 2 parça, kv: 2 parça)."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if name.endswith(".qkv"):
                module.weight.muon_split = 3
            elif name.endswith((".w12", ".kv_proj")):
                module.weight.muon_split = 2


def build_optimizers(model: nn.Module, kind: str = "muon", lr_muon: float = 0.02, lr_adam: float = 3e-3,
                     weight_decay: float = 0.0):
    mark_fused(model)
    if kind == "muon":
        muon, adam = split_params(model)
        return [Muon(muon, lr=lr_muon, weight_decay=weight_decay),
                torch.optim.AdamW(adam, lr=lr_adam, betas=(0.9, 0.95), weight_decay=0.0)]
    if kind == "adamw":
        decay = [p for p in model.parameters() if p.ndim >= 2]
        no_decay = [p for p in model.parameters() if p.ndim < 2]
        return [torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=lr_adam, betas=(0.9, 0.95))]
    raise ValueError(kind)


def wsd_factor(progress: float, warmup: float = 0.02, decay_start: float = 0.7) -> float:
    """progress ∈ [0,1] (zaman ya da adım oranı) → LR çarpanı."""
    if progress < warmup:
        return max(progress / warmup, 0.02)
    if progress < decay_start:
        return 1.0
    return max(0.0, (1.0 - progress) / (1.0 - decay_start))


def set_lr(optimizers: Iterable[torch.optim.Optimizer], factor: float) -> None:
    for opt in optimizers:
        for group in opt.param_groups:
            group.setdefault("base_lr", group["lr"])
            group["lr"] = group["base_lr"] * factor
