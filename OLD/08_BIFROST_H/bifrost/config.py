"""
Model konfigürasyonu.

`layout` her katmanın token karıştırıcısını belirler (bir harf = bir katman):
    K : Kuzgun — pencere dikkati (Huginn) + delta hafıza (Muninn), paralel
    W : Kuzgun yalnız pencere kolu (ablation)
    R : Kuzgun yalnız hafıza kolu (ablation)
    M : Mímir (kapılı delta hafıza, O(1))
    N : yok (katman yalnız Bifrost CSL + FFN -> saf CSL modeli)
    A : tam nedensel dikkat (YALNIZ Transformer++ baseline'ı için; Bifrost modellerinde kullanılmaz)

`csl=True` ise her bloğun başına Bifrost CSL v2 karıştırıcısı eklenir.
"""

from dataclasses import asdict, dataclass, replace
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    dim: int = 256
    layout: str = "MMMM"
    csl: bool = True
    csl_kernel: int = 4
    csl_dilation: int = 1
    ffn_mult: float = 8 / 3
    # Mímir
    mimir_heads: int = 4
    mimir_dk: int = 64
    mimir_dv: int = 64
    mimir_conv: int = 4
    mimir_chunk: int = 64
    negative_eigen: bool = False
    # Kuzgun ('K', 'W', 'R' katmanları)
    kuzgun_heads: int = 4
    kuzgun_head_dim: int = 64
    window: int = 64
    kuzgun_conv: int = 4
    coupled_decay: bool = False
    archival_heads: int = 0
    # Eğitim kararlılığı
    logit_softcap: Optional[float] = None
    zero_init_out: bool = False
    # Baseline dikkati ('A' katmanları)
    attn_heads: int = 4
    attn_kv_heads: Optional[int] = None
    attn_window: Optional[int] = None
    tie_embeddings: bool = True

    @property
    def n_layers(self) -> int:
        return len(self.layout)

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **kwargs) -> "ModelConfig":
        return replace(self, **kwargs)


PRESETS = {
    # Kuzgun ailesi (global dikkat yok; bellek O(1))
    "kuzgun-cpu": ModelConfig(dim=256, layout="K" * 6, csl=False, kuzgun_heads=4, window=64,
                              logit_softcap=30.0, zero_init_out=True),
    # Bifrost ailesi
    "bifrost-nano": ModelConfig(dim=128, layout="MMMM", mimir_heads=2),
    "bifrost-mini": ModelConfig(dim=320, layout="M" * 12, mimir_heads=5),
    "bifrost-small": ModelConfig(dim=512, layout="M" * 16, mimir_heads=8),
    "bifrost-base": ModelConfig(dim=768, layout="M" * 12, mimir_heads=6, mimir_dk=128, mimir_dv=128),
    # Karşılaştırma modelleri
    "csl-nano": ModelConfig(dim=128, layout="NNNN", csl_kernel=7),
    "transformer-nano": ModelConfig(dim=128, layout="AAAA", csl=False, attn_heads=2),
}
