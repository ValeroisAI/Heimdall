"""
Kuzgun model ve eğitim ön ayarları.

Parametre sayısı kabaca:  L · 13·d²  +  V·d   (tied embedding, SwiGLU 8/3·d)
16 GB'lık RX 9070 XT için ön ayarlar; mikro-batch ve gradient checkpointing değerleri
belleğe sığacak şekilde seçildi (gerekirse --micro-batch ile düşür).
"""

from dataclasses import asdict, dataclass, field, replace


@dataclass
class KuzgunConfig:
    vocab_size: int = 8192
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    head_dim: int = 64
    archival_heads: int = 4        # unutmayan (α ≡ 1) hafıza kafası sayısı — 1M+ token hafıza
    window: int = 256              # Huginn pencere boyu (token)
    conv_kernel: int = 4           # q/k/v üzerindeki nedensel kısa conv (Canon)
    chunk_size: int = 64           # Muninn chunk-paralel eğitim bloğu
    ffn_mult: float = 8 / 3
    ffn_multiple_of: int = 128     # GPU verimi için FFN genişliği yuvarlama
    rope_base: float = 10_000.0
    logit_softcap: float = 30.0
    negative_eigen: bool = False   # β ∈ (0, 2): durum takibi (parity vb.) için
    mtp: bool = False              # çok-token tahmini: ek olarak t+2 hedefi
    mtp_weight: float = 0.3
    tie_embeddings: bool = True
    norm_eps: float = 1e-6
    attn_backend: str = "sdpa"     # "sdpa" (her yerde çalışır) | "flex" (torch flex_attention, varsa daha hızlı)
    ternary: bool = False          # gizli ağırlıksız üçlü (−1/0/+1) doğrusal katmanlar, bkz. uclu.py
    ternary_int8: bool = False     # üçlü katmanlarda int8 ileri geçiş (torch._int_mm)
    memory_layers: tuple = ()      # product-key hafıza katmanı eklenecek blok indeksleri, bkz. hafiza.py
    memory_n_sub: int = 512        # alt-anahtar sayısı n → n² yuva
    memory_heads: int = 4
    memory_topk: int = 32
    memory_dq: int = 256

    @property
    def inner(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def ffn_hidden(self) -> int:
        h = int(self.ffn_mult * self.d_model)
        m = self.ffn_multiple_of
        return ((h + m - 1) // m) * m

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **kw) -> "KuzgunConfig":
        return replace(self, **kw)


@dataclass
class TrainPreset:
    model: KuzgunConfig
    seq_len: int = 2048
    micro_batch: int = 8            # GPU'ya tek seferde giren dizi sayısı
    batch_tokens: int = 131_072     # optimizer adımı başına token (gradient biriktirme ile)
    grad_ckpt: bool = False
    lr_muon: float = 0.02
    lr_adam: float = 3e-3
    seq_curriculum: tuple = field(default_factory=lambda: (0.15, 0.35))  # %15'e kadar T/4, %35'e kadar T/2


PRESETS = {
    # ~6M — evde hızlı duman testi (CPU'da da çalışır)
    "deneme": TrainPreset(KuzgunConfig(d_model=256, n_layers=6, n_heads=4, archival_heads=2, window=64),
                          seq_len=512, micro_batch=8, batch_tokens=16_384),
    # ~19M
    "kucuk": TrainPreset(KuzgunConfig(d_model=384, n_layers=8, n_heads=6, archival_heads=2, window=128),
                         seq_len=1024, micro_batch=16, batch_tokens=65_536),
    # ~45M
    "orta": TrainPreset(KuzgunConfig(d_model=512, n_layers=12, n_heads=8, archival_heads=3, window=256),
                        seq_len=2048, micro_batch=8, batch_tokens=131_072),
    # ~100M
    "temel": TrainPreset(KuzgunConfig(d_model=768, n_layers=12, n_heads=12, archival_heads=4, window=256),
                         seq_len=2048, micro_batch=4, batch_tokens=262_144, lr_adam=2e-3),
    # ~280M — gradient checkpointing ile 16 GB'a sığar
    "buyuk": TrainPreset(KuzgunConfig(d_model=1024, n_layers=20, n_heads=16, archival_heads=6, window=512),
                         seq_len=2048, micro_batch=8, batch_tokens=262_144, grad_ckpt=True,
                         lr_muon=0.015, lr_adam=1.5e-3),
}
