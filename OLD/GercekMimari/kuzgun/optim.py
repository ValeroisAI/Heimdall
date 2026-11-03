"""
Muon (gizli matrisler) + AdamW (embedding, norm, conv, kapılar, skalerler) ve WSD çizelgesi.

Muon, momentumlu gradyanı Newton-Schulz ile yarı-ortogonalleştirir: her yönde benzer büyüklükte
adım atar; aynı token bütçesinde AdamW'den belirgin hızlı öğrenir. Birleşik matrisler (qkv, w12)
parça parça ortogonalleştirilir. GPU'da Newton-Schulz bf16'da çalışır.
Üçlü ağırlıklar TernaryFlip (uclu.py), hafıza değer tabloları SparseRowRMS (hafiza.py) ile güncellenir.
"""

import torch
import torch.nn as nn


def zeropower_ns5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """G ≈ U Vᵀ (son iki boyutta, toplu). Katsayılar: Keller Jordan'ın beşinci derece Newton-Schulz'u."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16() if g.is_cuda else g.float()
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
            mom = group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                buf = self.state[p].get("momentum_buffer")
                if buf is None:
                    buf = self.state[p]["momentum_buffer"] = torch.zeros_like(g)
                buf.lerp_(g, 1 - mom)
                u = g.lerp(buf, mom) if group["nesterov"] else buf
                parts = getattr(p, "muon_split", 1)
                u = zeropower_ns5(u.view(parts, p.size(0) // parts, -1), group["ns_steps"])
                u = u * max(1.0, u.size(-2) / u.size(-1)) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u.view_as(p), alpha=-group["lr"])


def param_groups(model: nn.Module) -> dict:
    """muon: bloklardaki büyük yoğun 2D matrisler | adam_decay: embedding | adam_plain: geri kalan küçükler
    ternary: üçlü ağırlıklar (TernaryFlip) | sparse: hafıza değer tabloları (SparseRowRMS)."""
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            if name.endswith(".qkv"):
                m.weight.muon_split = 3
            elif name.endswith(".w12"):
                m.weight.muon_split = 2
    embed_ids = {id(model.embed.weight), id(model.lm_head.weight)}
    groups = {k: [] for k in ("muon", "adam_decay", "adam_plain", "ternary", "sparse")}
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if getattr(p, "ternary", False):
            groups["ternary"].append(p)
        elif getattr(p, "sparse_rows", False):
            groups["sparse"].append(p)
        elif id(p) in embed_ids:
            groups["adam_decay"].append(p)
        elif p.ndim == 2 and min(p.shape) >= 64 and "gates" not in name:
            groups["muon"].append(p)
        else:
            groups["adam_plain"].append(p)
    return groups


def build_optimizers(model: nn.Module, lr_muon: float = 0.02, lr_adam: float = 3e-3,
                     weight_decay: float = 0.0, embed_decay: float = 0.0, use_muon: bool = True,
                     flip_rate: float = 2e-3, lr_memory: float = None) -> list:
    """Sıra sabit (checkpoint bu sıraya göre yüklenir): [Muon], AdamW, [TernaryFlip], [SparseRowRMS]."""
    from .hafiza import SparseRowRMS
    from .uclu import TernaryFlip

    g = param_groups(model)
    fused = {"fused": True} if torch.cuda.is_available() else {}
    adam_groups = [{"params": g["adam_decay"], "weight_decay": embed_decay},
                   {"params": g["adam_plain"], "weight_decay": 0.0}]
    if not use_muon:  # karşılaştırma için saf AdamW
        adam_groups.insert(0, {"params": g["muon"], "weight_decay": 0.1})
    try:
        adam = torch.optim.AdamW(adam_groups, lr=lr_adam, betas=(0.9, 0.95), eps=1e-8, **fused)
    except (RuntimeError, TypeError):
        adam = torch.optim.AdamW(adam_groups, lr=lr_adam, betas=(0.9, 0.95), eps=1e-8)
    opts = [Muon(g["muon"], lr=lr_muon, weight_decay=weight_decay), adam] if use_muon and g["muon"] else [adam]
    if g["ternary"]:
        opts.append(TernaryFlip(g["ternary"], lr=flip_rate))
    if g["sparse"]:
        opts.append(SparseRowRMS(g["sparse"], lr=lr_memory or lr_adam))
    return opts


def wsd(progress: float, warmup: float = 0.01, decay_start: float = 0.75, floor: float = 0.0) -> float:
    """Warmup → sabit → doğrusal düşüş. progress ∈ [0, 1]."""
    if progress < warmup:
        return max(progress / warmup, 1e-3)
    if progress < decay_start:
        return 1.0
    return floor + (1.0 - floor) * max(0.0, (1.0 - progress) / (1.0 - decay_start))


def set_lr(optimizers, factor: float) -> None:
    for opt in optimizers:
        for group in opt.param_groups:
            group.setdefault("base_lr", group["lr"])
            group["lr"] = group["base_lr"] * factor
