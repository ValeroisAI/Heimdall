"""
bit.py — Gizli ağırlıksız 1-bit / 1.58-bit eğitim motoru.

BitNet eğitimde her ağırlığın fp32 "gizli" kopyasını ve Adam durumunu tutar; 1-bit yalnız çıkarımda olur.
Burada gizli kopya YOK. Ağırlık, değerleri yalnız {−1, +1} (bits=1) ya da {−1, 0, +1} (bits=2) olan bir
bf16 tensördür ("sanal bf16": donanım sıradan bir bf16 matris görür, değerler bf16'da birebir temsil edilir).

    y = (x · Wᵀ) ⊙ (taban · ölçek)        ölçek: satır başına öğrenilen, 1 civarında başlar (AdamW)

BitFlip optimizer'ı gradyan momentumunu biriktirir; her adımda "çevrilmek isteyen" ağırlıkların en
güçlü `flip oranı` kadarını çevirir (ikili: işaret değişir; üçlü: bir basamak kayar). Çevrilen ağırlığın
momentumu sıfırlanır. Öğrenme hızı = çevrilen ağırlık oranı; WSD çizelgesini izler.

Bellek / parametre: ağırlık 2 + gradyan 2 + momentum 2 = 6 bayt (AdamW + fp32 kopya: 16-18 bayt).
Dürüst not: bu, eğitimde ağırlıkların 1 bit tutulduğu anlamına GELMEZ; bellek ~2.5-3× azalır. Paketlenmiş
1-bit depolama ve gradyanı çekirdek içinde çeviren (fused) bir Triton çekirdeği ayrı bir mühendislik işidir.
Gizli ağırlıksız eğitim dil modellerinde yeni ve deneyseldir; eşit boyutta yoğun modelden geride kalması beklenir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _autocast(device_type: str) -> bool:
    try:
        return torch.is_autocast_enabled(device_type)
    except TypeError:                                        # eski PyTorch: yalnız CUDA durumu sorulabilir
        return device_type == "cuda" and torch.is_autocast_enabled()


def _match(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Ağırlığı girdinin türüne çevir; autocast açıkken ÇEVİRME. Artık akış fp32 olduğundan (norm çıkışı) eski kod her
    ileri geçişte ağırlığın fp32 kopyasını çıkarıyor, autocast onu yeniden bf16'ya çeviriyor, geri geçişte de gradyan
    fp32 kopyadan dönüyordu: w12 için parametre başına ~12 bayt gereksiz bellek trafiği. Autocast ikisini de kendi çevirir."""
    return weight if (weight.dtype == x.dtype or _autocast(x.device.type)) else weight.to(x.dtype)


class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bits: int = 1, rank: int = 0) -> None:
        super().__init__()
        assert bits in (1, 2), "bits: 1 = ikili (±1), 2 = üçlü (−1/0/+1)"
        self.in_features, self.out_features, self.bits = in_features, out_features, bits
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, dtype=torch.bfloat16))
        self.weight.bit = bits                               # optimizer ayrımı için işaret
        self.register_buffer("base", torch.full((out_features,), 0.02))
        self.scale = nn.Parameter(torch.ones(out_features))
        self.rank = rank
        if rank:  # yoğun düşük-rank düzeltici: bit ağırlıkların tutamadığı yönler. U sıfırdan başlar (LoRA gibi)
            self.lr_u = nn.Parameter(torch.zeros(out_features, rank))
            self.lr_v = nn.Parameter(torch.randn(rank, in_features) * in_features ** -0.5)
        self.kernel = "off"                                  # "int8": Triton int8 GEMM (bitkernel.py)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 0.02) -> None:
        w = torch.randn(self.out_features, self.in_features) * std
        if self.bits == 1:
            self.weight.copy_(torch.where(w >= 0, 1.0, -1.0))
            self.base.fill_(std)                              # satır std'si diğer katmanlarla aynı
        else:
            gamma = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)   # satır başına absmean (BitNet b1.58)
            self.weight.copy_((w / gamma).round().clamp(-1, 1))
            self.base.copy_(gamma.squeeze(1))
        self.scale.fill_(1.0)

    @torch.no_grad()
    def zero_(self) -> None:
        """Artık dal çıkışı sıfırdan başlasın. Üçlüde ağırlıklar 0; ikilide 0 olamayacağı için ölçek 0."""
        if self.bits == 2:
            self.weight.zero_()
        else:
            self.scale.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel == "int8" and x.is_cuda:
            from .bitkernel import bit_int8_linear
            y = bit_int8_linear(x, self.weight)
            y = y * (self.base * self.scale).to(y.dtype)
        else:
            # Satır ölçeği ağırlığa katılır: x·(W ⊙ s)ᵀ = (x·Wᵀ) ⊙ s. Fark geri geçişte: ölçeğin gradyanı ağırlık boyutunda
            # (satır içi toplam) hesaplanır, çıktı boyutunda (16K token × 4096 kanal, sütun toplamı) değil; çıktı üzerinde
            # ayrı bir çarpma geçişi de kalmaz. W'nin gradyanı aynı: dW = d(W ⊙ s) ⊙ s.
            s = (self.base * self.scale)[:, None]
            dt = torch.get_autocast_dtype(x.device.type) if _autocast(x.device.type) else x.dtype
            y = F.linear(x, (self.weight.to(dt) * s.to(dt)))
        if self.rank:
            y = y + F.linear(F.linear(x, self.lr_v.to(x.dtype)), self.lr_u.to(x.dtype))
        return y

    def extra_repr(self) -> str:
        return f"{self.in_features}→{self.out_features}, {'ikili' if self.bits == 1 else 'üçlü'}"


class BitFlip(torch.optim.Optimizer):
    """Gizli ağırlıksız 1-bit / 1.58-bit optimizer. lr = adım başına çevrilen ağırlık oranı (ör. 2e-3)."""

    def __init__(self, params, lr: float = 2e-3, momentum: float = 0.9, sample: int = 16384) -> None:
        super().__init__(params, dict(lr=lr, momentum=momentum, sample=sample))
        self.last_flip_frac = 0.0
        self._idx = {}   # kuantil örnek indeksleri: optimizer durumuna KOYMA (load_state_dict tamsayı tensörü bf16'ya çevirir)

    @torch.no_grad()
    def step(self, closure=None):
        flipped = total = 0
        for group in self.param_groups:
            beta, rate = group["momentum"], group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p)
                if id(p) not in self._idx:
                    self._idx[id(p)] = torch.randint(0, p.numel(), (min(group["sample"], p.numel()),), device=p.device)
                idx = self._idx[id(p)]
                # Tek fp32 geçici (mf) + bool maskeler; eski sürüm parametre başına ~7 fp32 geçici kuruyordu
                # (r, pf, skor, yeni, where sonucu, ...). Karar aynı: skor = itme / satır RMS'i, skor > eşik ise çevir.
                mf = st["m"].float()
                mf.mul_(beta).add_(p.grad, alpha=1 - beta)
                if rate > 0:
                    rms = torch.linalg.vector_norm(mf, dim=1, keepdim=True).mul_(p.size(1) ** -0.5).add_(1e-12)
                    ms, ps = mf.flatten()[idx], p.flatten()[idx].float()          # eşik için örnek (16K eleman)
                    rs = ms / rms.flatten()[idx // p.size(1)]
                    if p.bit == 1:
                        ss = (ps * rs).clamp_(min=0)
                    else:
                        ss = torch.where(((ps >= 1) & (rs < 0)) | ((ps <= -1) & (rs > 0)), 0.0, rs.abs())
                    thr = torch.quantile(ss, 1.0 - rate).clamp(min=1e-12)
                    lim = rms * thr                                                # satır başına eşik [satır, 1]
                    if p.bit == 1:
                        mf.mul_(p)                              # itme = m·w (w = ±1): yerinde, sonra geri alınır
                        mask = mf > lim
                        mf.mul_(p)                              # ±1 ile çarpım kayıpsız: mf eski haline döner
                        p[mask] = -p[mask]                      # yalnız çevrilenler (≈ %0.5) okunur/yazılır
                    else:
                        up, down = mf < -lim, mf > lim          # r < 0: yukarı, r > 0: aşağı bir basamak
                        up &= p < 1                             # +1 daha yukarı, −1 daha aşağı gidemez
                        down &= p > -1
                        p[up] += 1
                        p[down] -= 1
                        mask = up.logical_or_(down)
                    mf.masked_fill_(mask, 0.0)                  # çevrilenin momentumu sıfırlanır
                    flipped += int(mask.sum())
                total += p.numel()
                st["m"].copy_(mf)
        self.last_flip_frac = flipped / max(total, 1)


def set_bit_kernel(model: nn.Module, kernel: str) -> None:
    for m in model.modules():
        if isinstance(m, BitLinear):
            m.kernel = kernel


def bit_stats(model: nn.Module) -> dict:
    n = zeros = 0
    for m in model.modules():
        if isinstance(m, BitLinear):
            n += m.weight.numel()
            zeros += int((m.weight == 0).sum())
    return {"bit_param": n, "sifir_orani": zeros / max(n, 1)}
