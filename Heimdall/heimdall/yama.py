"""
Bayt düzeyinde yamalı model (deney): tokenizer yok, büyük gövde her P baytta bir kez çalışır.

    bayt → yerel kodlayıcı (dar, bayt hızında) → P baytı tek vektöre indir → GÖVDE (geniş, yama hızında)
         → P bayta aç, P-1 kaydır → yerel çözücü (dar, bayt hızında) → sonraki bayt

Nedensellik: gövdenin k. yama çıktısı (k+1)·P'den önceki baytları bilir ve yalnız (k+1)·P ve sonrasındaki baytların
tahmininde kullanılır (açılmış dizi P-1 sağa kaydırılır: bir yamanın ilk baytı gövdeyi gecikmesiz görür).
Maliyet: gövde parametreleri bayt başına 1/P kez çalışır; yerel katmanlar dar olduğu için ucuzdur (d_yerel² / d_gövde²).
Fikir MegaByte / Byte Latent Transformer çizgisinden; gövde ve yerel katmanlar Heimdall blokları (DeltaNet + dikkat).
Ölçüm: bench/yama_duello.py (eşit GPU süresi, bit/bayt).
"""

from dataclasses import dataclass, field, replace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import HeimdallConfig
from .model import Block, GatedDeltaNet, RMSNorm


@dataclass
class PatchConfig:
    trunk: HeimdallConfig = field(default_factory=HeimdallConfig)
    patch: int = 8                 # P: gövde adımı başına bayt
    d_local: int = 256             # yerel katman genişliği
    local_enc: int = 1             # yerel kodlayıcı katmanı (DeltaNet)
    local_dec: int = 3             # yerel çözücü katmanı (DeltaNet)
    vocab_size: int = 256

    def local_cfg(self) -> HeimdallConfig:
        h = max(2, self.d_local // 128)
        return replace(self.trunk, d_model=self.d_local, n_heads=h, head_dim=64, archival_heads=0, layout="D",
                       vocab_size=self.vocab_size, moe_experts=0, bits=0, mtp_depth=0)


class PatchLM(nn.Module):
    def __init__(self, cfg: PatchConfig) -> None:
        super().__init__()
        self.cfg, P, dl, D = cfg, cfg.patch, cfg.d_local, cfg.trunk.d_model
        lc = cfg.local_cfg()
        self.embed = nn.Embedding(cfg.vocab_size, dl)
        self.enc = nn.ModuleList([Block(lc, "D") for _ in range(cfg.local_enc)])
        self.down = nn.Linear(P * dl, D, bias=False)
        self.trunk = nn.ModuleList([Block(cfg.trunk, k) for k in cfg.trunk.kinds])
        self.trunk_norm = RMSNorm(D, cfg.trunk.norm_eps)
        self.up = nn.Linear(D, P * dl, bias=False)
        self.start = nn.Parameter(torch.zeros(P - 1, dl))          # ilk P-1 konum: henüz gövde çıktısı yok
        self.dec = nn.ModuleList([Block(lc, "D") for _ in range(cfg.local_dec)])
        self.norm = RMSNorm(dl, cfg.trunk.norm_eps)
        self.lm_head = nn.Linear(dl, cfg.vocab_size, bias=False)
        self.grad_ckpt = False
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, GatedDeltaNet):
                m.reset_parameters()
        for name, m in self.named_modules():                       # artık dal çıkışları sıfırdan başlar
            if name.endswith(("o_proj", "w3")) and isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
        self.lm_head.weight = self.embed.weight

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def cost_params(self) -> float:
        """Bayt başına çalışan parametre eşdeğeri: yerel katmanlar tam, gövde 1/P."""
        trunk = sum(p.numel() for p in self.trunk.parameters()) + self.down.weight.numel() + self.up.weight.numel()
        return self.num_params() - trunk + trunk / self.cfg.patch

    def _run(self, blocks, x):
        for blk in blocks:
            x = checkpoint(blk, x, use_reentrant=False) if (self.grad_ckpt and self.training) else blk(x)
        return x

    def forward(self, ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """ids [B, T] (T, P'nin katı). Döndürür: targets verilirse kayıp (nat/bayt), yoksa logitler [B, T, 256]."""
        b, t = ids.shape
        P, dl = self.cfg.patch, self.cfg.d_local
        assert t % P == 0, f"dizi boyu ({t}) yama boyunun ({P}) katı olmalı"
        e = self._run(self.enc, self.embed(ids))                                   # [B, T, dl]
        h = self.down(e.reshape(b, t // P, P * dl))                                # [B, T/P, D]
        h = self.trunk_norm(self._run(self.trunk, h))
        u = self.up(h).reshape(b, t, dl)                                           # yama k'nin P dilimi
        u = torch.cat((self.start.to(u.dtype).expand(b, -1, -1), u[:, :t - (P - 1)]), 1)   # P-1 sağa kaydır
        x = self._run(self.dec, e + u)
        logits = self.lm_head(self.norm(x)).float()
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
