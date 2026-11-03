"""
valerois_muon_hybrid.py — Muon + ValeroisFactoredOpt hibrit optimizer sarmalayici
==================================================================================
Kullanim:
    from valerois_muon_hybrid import make_hybrid_optimizers, HybridOptimizerGroup

    opts = make_hybrid_optimizers(model, lr_muon=0.02, lr_other=3e-4)
    # Egitim dongusunde:
    for opt in opts:
        opt.step()
    for opt in opts:
        opt.zero_grad(set_to_none=True)

Param grubu bolme kurali (Valerois v8/v7 mimarisine ozel):
    Muon  : ndim >= 2 VE (conv, norm, embed, head, lm_head, gate icermeyen) tum 2D matrisler
    Other : embed, head, lm_head, norm weight, bias, conv, gate, scaler -> ValeroisFactoredOpt

Neden bu bolme?
    - Muon Newton-Schulz ile gradyan momentumunu semi-orthogonalize eder.
      Embedding/head matrislerini orthogonalize etmek discrete lookup'ta
      token dagilimini bozabilir.
    - Conv parametreleri depthwise (groups=hidden) -> reshape'te anlamsiz.
    - Norm/bias 1D -> Muon ndim >= 2 sartini zaten karsilamiyor.
"""

import os
import sys
import math
from typing import List, Tuple

import torch
import torch.nn as nn

# Speed pack dizini PATH'e ekle
_KIMI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "KIMI")
if _KIMI_DIR not in sys.path:
    sys.path.insert(0, _KIMI_DIR)

from sp_muon import Muon  # noqa: E402

# Mevcut Valerois optimizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from valerois_optimizer import ValeroisFactoredOpt  # noqa: E402


# ---------------------------------------------------------------------------
# Param grubu bolme
# ---------------------------------------------------------------------------

_SKIP_MUON_KEYWORDS = (
    "embed",
    "head",
    "lm_head",
    "norm",
    "bias",
    "conv",
    "gate",
    "scaler",
    "router",
    "log_lambda",
    "lambda_param",
)


def split_params_for_muon(
    model: nn.Module,
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Model parametrelerini Muon ve ValeroisFactoredOpt gruplarina ayirir.

    Returns:
        muon_params  : 2D+ hidden weight matrisleri (Muon icin)
        other_params : 1D + skip_keyword iceren her sey (ValeroisFactoredOpt icin)
    """
    muon_params, other_params = [], []
    seen = set()

    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))

        if not p.requires_grad:
            continue

        is_skip = any(kw in name for kw in _SKIP_MUON_KEYWORDS)

        if p.ndim >= 2 and not is_skip:
            muon_params.append(p)
        else:
            other_params.append(p)

    return muon_params, other_params


# ---------------------------------------------------------------------------
# Hibrit optimizer grup yoneticisi
# ---------------------------------------------------------------------------

class HybridOptimizerGroup:
    """
    Muon + ValeroisFactoredOpt'u tek bir nesne gibi sunar.
    API: step(), zero_grad(), state_dict(), load_state_dict()
    """

    def __init__(
        self,
        muon: Muon,
        other: ValeroisFactoredOpt,
    ):
        self.muon = muon
        self.other = other
        self._opts = [muon, other]

    def step(self, closure=None):
        for opt in self._opts:
            opt.step(closure)

    def zero_grad(self, set_to_none: bool = True):
        for opt in self._opts:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "muon": self.muon.state_dict(),
            "other": self.other.state_dict(),
        }

    def load_state_dict(self, sd: dict):
        self.muon.load_state_dict(sd["muon"])
        self.other.load_state_dict(sd["other"])

    def set_lr(self, lr_muon: float, lr_other: float):
        """WSD lr takvimi icin her adimda lr guncelleme."""
        for g in self.muon.param_groups:
            g["lr"] = lr_muon
        for g in self.other.param_groups:
            g["lr"] = lr_other

    def param_groups(self):
        """Debug icin tum param_groups listesi."""
        return self.muon.param_groups + self.other.param_groups


# ---------------------------------------------------------------------------
# Ana fabrika fonksiyonu
# ---------------------------------------------------------------------------

def make_hybrid_optimizers(
    model: nn.Module,
    lr_muon: float = 0.02,
    lr_other: float = 3e-4,
    momentum_muon: float = 0.95,
    ns_steps: int = 5,
    weight_decay_other: float = 0.01,
    betas_other: Tuple[float, float] = (0.9, 0.95),
    eps_other: float = 1e-5,
) -> HybridOptimizerGroup:
    """
    Valerois mimarisi icin hibrit optimizer olusturur.

    Args:
        model          : Egitilecek model (ValeroisAeroDriveModel veya Apotheosis)
        lr_muon        : Muon baslangic lr (onerilen: 0.02)
        lr_other       : ValeroisFactoredOpt baslangic lr (onerilen: 3e-4)
        momentum_muon  : Muon momentum (onerilen: 0.95)
        ns_steps       : Newton-Schulz iterasyonu (5 yeterli)
        weight_decay_other : Weight decay (embed/head dahil degil)
        betas_other    : AdamW-style betas ValeroisFactoredOpt icin
        eps_other      : Epsilon

    Returns:
        HybridOptimizerGroup
    """
    muon_params, other_params = split_params_for_muon(model)

    n_muon = sum(p.numel() for p in muon_params)
    n_other = sum(p.numel() for p in other_params)

    print(f"[muon]  {n_muon:,} parametre (2D hidden matrisler — Newton-Schulz orthogonalize)")
    print(f"[other] {n_other:,} parametre (embed/head/norm/bias/conv — ValeroisFactoredOpt)")

    if not muon_params:
        print("[uyari] Muon icin parametre bulunamadi — sadece ValeroisFactoredOpt kullaniliyor.")
        muon_opt = Muon([], lr=lr_muon, momentum=momentum_muon, ns_steps=ns_steps)
    else:
        muon_opt = Muon(muon_params, lr=lr_muon, momentum=momentum_muon, ns_steps=ns_steps)

    other_opt = ValeroisFactoredOpt(
        [
            {"params": other_params, "weight_decay": weight_decay_other},
        ],
        lr=lr_other,
        betas=betas_other,
        eps=eps_other,
    )

    return HybridOptimizerGroup(muon_opt, other_opt)


# ---------------------------------------------------------------------------
# WSD lr takvimi (sp_trainer.py ile uyumlu)
# ---------------------------------------------------------------------------

def wsd_lr_scale(
    step: int,
    total_steps: int,
    warmup: int,
    decay_frac: float = 0.35,
    min_lr_frac: float = 0.1,
) -> float:
    """
    Warmup -> Stable -> Lineer Decay (WSD) lr zamanlayicisi.

    Returns:
        lr_scale: [min_lr_frac, 1.0] araliginda carpan
    """
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    decay_start = int(total_steps * (1.0 - decay_frac))
    if step < decay_start:
        return 1.0
    t = (step - decay_start) / max(total_steps - decay_start, 1)
    return 1.0 - (1.0 - min_lr_frac) * t


# ---------------------------------------------------------------------------
# Birim test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch.nn.functional as F

    torch.manual_seed(42)
    print("=== valerois_muon_hybrid birim testi ===")

    # Kucuk bir Valerois benzeri demo modeli
    class _DemoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(512, 128)
            self.up_proj = nn.Linear(128, 256, bias=False)
            self.down_proj = nn.Linear(256, 128, bias=False)
            self.norm = nn.RMSNorm(128)
            self.head = nn.Linear(128, 512, bias=False)
            self.conv = nn.Conv1d(128, 128, 3, groups=128, bias=False)

    model = _DemoModel()
    opts = make_hybrid_optimizers(model, lr_muon=0.02, lr_other=3e-4)

    # Birkac adim
    x = torch.randint(0, 512, (4, 16))
    for i in range(5):
        h = model.embed(x)                        # [4, 16, 128]
        h = model.norm(h)
        h_c = h.transpose(1, 2)                   # [4, 128, 16]
        h_c = F.pad(h_c, (2, 0))
        h_c = model.conv(h_c)[..., :16].transpose(1, 2)
        h = h + h_c
        h = model.down_proj(F.silu(model.up_proj(h)))
        logits = model.head(h)
        loss = F.cross_entropy(logits.reshape(-1, 512),
                               torch.randint(0, 512, (4 * 16,)))
        opts.zero_grad()
        loss.backward()
        opts.step()

        # WSD lr guncelle
        scale = wsd_lr_scale(i, total_steps=20, warmup=2)
        opts.set_lr(0.02 * scale, 3e-4 * scale)
        print(f"  adim {i}: loss={loss.item():.4f}, lr_scale={scale:.3f}")

    print("valerois_muon_hybrid OK")
