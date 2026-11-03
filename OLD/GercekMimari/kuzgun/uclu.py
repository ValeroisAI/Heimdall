"""
uclu.py — Gizli ağırlıksız üçlü (−1 / 0 / +1) eğitim
===================================================
BitNet b1.58 eğitimde her ağırlığın fp32 bir "gizli" kopyasını tutar; bellek kazancı yalnız çıkarımdadır.
Burada gizli kopya YOK:

  - Ağırlık bir bf16 parametredir ama değerleri daima −1, 0 veya +1'dir ("sanal bf16": kart sıradan bir
    bf16 matris görür, üçlü değerler bf16'da birebir temsil edilir). Satır başına öğrenilen ölçek α (fp32).
        y = (x · Wᵀ) ⊙ α
  - Geri yayılımın ağırlık gradyanı (düz geçiş) momentumda biriktirilir. TernaryFlip optimizer'ı her adımda
    momentumu en güçlü olan ağırlıkların küçük bir oranını (flip oranı = öğrenme hızının karşılığı) bir
    basamak çevirir: +1 → 0 → −1 ya da tersi. Çevrilen ağırlığın momentumu sıfırlanır.
  - Bellek / parametre: ağırlık 2 + gradyan 2 + momentum 2 = 6 bayt (standart AdamW + fp32 kopya: ~16-18).

İsteğe bağlı INT8 yolu (--uclu-int8): ileri geçişte aktivasyonlar token başına int8'e nicemlenir ve
torch._int_mm ile tamsayı çarpımı yapılır (RDNA4 INT8 ≈ 2× bf16). Desteklenmiyorsa otomatik bf16'ya döner.
Bu yöntem ikili ağlardaki BOP'tan esinlenir; dil modellerinde denenmemiş, deneysel bir yaklaşımdır.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_INT8_OK: Optional[bool] = None


def int8_available(device: torch.device) -> bool:
    """torch._int_mm bu cihazda doğru çalışıyor mu? (derlemeden önce bir kez çağrılır)"""
    global _INT8_OK
    if _INT8_OK is None:
        try:
            a = torch.randint(-8, 8, (32, 64), dtype=torch.int8, device=device)
            b = torch.randint(-1, 2, (64, 32), dtype=torch.int8, device=device)
            ok = torch.equal(torch._int_mm(a, b).cpu(), (a.int().cpu() @ b.int().cpu()))
            _INT8_OK = bool(ok)
        except Exception:
            _INT8_OK = False
    return _INT8_OK


class _Int8TernaryMatmul(torch.autograd.Function):
    """İleri: int8 aktivasyon × üçlü ağırlık (tamsayı çarpım). Geri: düz geçiş, bf16/fp32 formülleri."""

    @staticmethod
    def forward(ctx, x2d, w):
        s = x2d.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
        xq = (x2d / s).round().clamp(-127, 127).to(torch.int8)
        y = torch._int_mm(xq, w.to(torch.int8).t().contiguous()).to(x2d.dtype) * s
        ctx.save_for_backward(x2d, w)
        return y

    @staticmethod
    def backward(ctx, gy):
        x2d, w = ctx.saved_tensors
        gx = gy @ w.to(gy.dtype)
        gw = (gy.t() @ x2d).to(w.dtype)
        return gx, gw


class TernaryLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_int8: bool = False) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=torch.bfloat16))
        self.weight.ternary = True
        # etkin ölçek = base · scale; scale 1 civarında öğrenilir (AdamW adımı göreli kalsın)
        self.register_buffer("base", torch.full((out_features,), 0.02))
        self.scale = nn.Parameter(torch.ones(out_features))
        self.use_int8 = use_int8
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 0.02) -> None:
        w = torch.randn(self.out_features, self.in_features) * std
        gamma = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)            # satır başına absmean (BitNet)
        self.weight.copy_((w / gamma).round().clamp(-1, 1).to(self.weight.dtype))
        self.base.copy_(gamma.squeeze(1))
        self.scale.fill_(1.0)

    @torch.no_grad()
    def zero_(self) -> None:
        """Artık dal çıkışı: ağırlıklar 0 başlar, ölçek makul kalır (çevrilince devreye girer)."""
        self.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_int8 and x.is_cuda and x.numel() // x.size(-1) > 16:
            shape = x.shape
            y = _Int8TernaryMatmul.apply(x.reshape(-1, shape[-1]), self.weight).reshape(*shape[:-1], -1)
        else:
            y = F.linear(x, self.weight.to(x.dtype))
        return y * (self.base * self.scale).to(y.dtype)

    def extra_repr(self) -> str:
        return f"{self.in_features}→{self.out_features}, üçlü, int8={self.use_int8}"


class TernaryFlip(torch.optim.Optimizer):
    """Gizli ağırlıksız üçlü optimizer. lr = adım başına çevrilen ağırlık oranı (ör. 2e-3)."""

    def __init__(self, params, lr: float = 2e-3, momentum: float = 0.9, sample: int = 16384) -> None:
        super().__init__(params, dict(lr=lr, momentum=momentum, sample=sample))
        self.last_flip_frac = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        flipped = total = 0
        for group in self.param_groups:
            beta, rate = group["momentum"], group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                m = st.get("m")
                if m is None:
                    m = st["m"] = torch.zeros_like(p)
                    st["idx"] = torch.randint(0, p.numel(), (min(group["sample"], p.numel()),), device=p.device)
                mf = m.float().mul_(beta).add_(p.grad.float(), alpha=1 - beta)
                if rate > 0:
                    # satır başına normalize et: her satırın ölçeği farklı olabilir
                    r = mf / (mf.square().mean(dim=1, keepdim=True).sqrt() + 1e-12)
                    pf = p.float()
                    # yalnız hareket edebilenler yarışır: +1'i yukarı, −1'i aşağı iten momentum bütçe harcamaz
                    score = torch.where(((pf >= 1) & (r < 0)) | ((pf <= -1) & (r > 0)), 0.0, r.abs())
                    thr = torch.quantile(score.flatten()[st["idx"]], 1.0 - rate).clamp(min=1e-12)
                    new = (pf - torch.sign(r) * (score > thr)).clamp(-1, 1)
                    changed = new != pf
                    p.copy_(new.to(p.dtype))
                    mf = torch.where(changed, torch.zeros_like(mf), mf)   # çevrilenin momentumu sıfırlanır
                    flipped += int(changed.sum())
                total += p.numel()
                m.copy_(mf.to(m.dtype))
        self.last_flip_frac = flipped / max(total, 1)


def set_int8(model: nn.Module, on: bool) -> None:
    for m in model.modules():
        if isinstance(m, TernaryLinear):
            m.use_int8 = on


def ternary_stats(model: nn.Module) -> dict:
    n = zeros = 0
    for m in model.modules():
        if isinstance(m, TernaryLinear):
            n += m.weight.numel()
            zeros += int((m.weight == 0).sum())
    return {"uclu_param": n, "sifir_orani": zeros / max(n, 1)}
