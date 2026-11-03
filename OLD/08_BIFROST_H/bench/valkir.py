"""
valkir.py — Eğitilmiş Valkir (saf CSL, valerois_exponential_csl yapısı) yükleyicisi
====================================================================================
Ağırlıklar `Valerois1MNativeModel` ile birebir aynı anahtarlara sahip. Fark yalnız
dilatasyonlu konvolüsyonun hesaplanışı: nn.Conv1d, dilatasyon 32768'de ~1M tokenlik
sıfır dolgu ayırıp çöküyor. Burada yalnız dizinin içine düşen tap'ler toplanır
(dolguya düşen tap'ler zaten 0 katkı verir), yani çıktı matematiksel olarak aynıdır.

Dilatasyon çizelgesi checkpoint'te kayıtlı değil; `schedule` ile verilir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

SCHEDULES = {
    "exp": [min(2 ** l, 32768) for l in range(16)],   # valerois_exponential_csl varsayılanı
    "const2": [2] * 16,
    "const1": [1] * 16,
    "mod4": [2 ** (l % 4) for l in range(16)],        # Hyperion / Aero-Titan çizelgesi
}


def causal_dw_conv(x: torch.Tensor, weight: torch.Tensor, dilation: int) -> torch.Tensor:
    """x: [B, T, D], weight: [D, 1, K]. y_t = Σ_j w[K-1-j] · x_{t - j·d} (PyTorch conv hizası)."""
    b, t, d = x.shape
    k = weight.size(-1)
    w = weight.squeeze(1)  # [D, K]; w[:, K-1] en yeni token
    y = x * w[:, k - 1]
    for j in range(1, k):
        shift = j * dilation
        if shift >= t:
            break
        y[:, shift:] += x[:, :-shift] * w[:, k - 1 - j]
    return y


def rms(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight


class Valkir(nn.Module):
    def __init__(self, state_dict: dict, schedule: str = "exp") -> None:
        super().__init__()
        self.sd = {k: v.float() for k, v in state_dict.items()}
        self.n_layers = 1 + max(int(k.split(".")[1]) for k in self.sd if k.startswith("layers."))
        self.dilations = SCHEDULES[schedule][: self.n_layers]

    def receptive_field(self) -> int:
        k_short = self.sd["layers.0.mixer.conv_short.weight"].size(-1)
        k_long = self.sd["layers.0.mixer.conv_dilated.weight"].size(-1)
        return sum(max(k_short - 1, (k_long - 1) * d) for d in self.dilations)

    @torch.no_grad()
    def forward(self, ids: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        sd = self.sd
        x = F.embedding(ids, sd["embed_tokens.weight"])
        for l, dil in enumerate(self.dilations):
            p = f"layers.{l}."
            h = rms(x, sd[p + "input_layernorm.weight"])
            xs = causal_dw_conv(h, sd[p + "mixer.conv_short.weight"], 1)
            xd = causal_dw_conv(h, sd[p + "mixer.conv_dilated.weight"], dil)
            m = rms(F.linear(h, sd[p + "mixer.in_proj.weight"]) + 0.5 * (xs + xd), sd[p + "mixer.norm.weight"])
            g = F.silu(F.linear(h, sd[p + "mixer.gate_proj.weight"]))
            x = x + F.linear(m * g, sd[p + "mixer.out_proj.weight"])
            h = rms(x, sd[p + "post_mixer_layernorm.weight"])
            x = x + F.linear(F.silu(F.linear(h, sd[p + "mlp.w1.weight"])) * F.linear(h, sd[p + "mlp.w2.weight"]),
                             sd[p + "mlp.w3.weight"])
        x = rms(x, sd["norm.weight"])
        return F.linear(x, sd["lm_head.weight"])


def load(path: str, schedule: str = "exp") -> Valkir:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return Valkir(ckpt["model"] if "model" in ckpt else ckpt, schedule)
