"""
Heimdall yapılandırması, mimari türleri ve 16 GB için eğitim ön ayarları.

Katman düzeni `layout` ile verilir ve derinlik boyunca tekrarlanır:
    D = Gated DeltaNet (doğrusal zamanlı, sabit boyutlu durum)
    A = dikkat (global ya da `attn_window` > 0 ise kayan pencere)
"DDDA" = her 4 katmandan 3'ü DeltaNet, 1'i dikkat. "A" = saf Transformer (karşılaştırma tabanı).
"""

from dataclasses import asdict, dataclass, replace


@dataclass
class HeimdallConfig:
    vocab_size: int = 8192
    d_model: int = 768
    n_layers: int = 12
    layout: str = "DDDA"
    # Gated DeltaNet
    n_heads: int = 6
    head_dim: int = 128
    archival_heads: int = 2        # α ≡ 1 (hiç unutmayan) kafa sayısı: çok uzun bağlamda hatırlama
    conv_kernel: int = 4           # q/k/v üzerinde nedensel kısa conv
    chunk_size: int = 64
    # dikkat
    attn_heads: int = 12
    attn_kv_heads: int = 4         # GQA: KV cache attn_heads / attn_kv_heads kat küçülür
    attn_head_dim: int = 64
    attn_window: int = 0           # 0 = global; > 0 = kayan pencere (çıkarım belleği O(1))
    attn_rope: str = "auto"        # auto: hibritte NoPE (konumu DeltaNet taşır), pencere/Transformer'da RoPE
    attn_gate: bool = True         # dikkat çıkışında sigmoid kapı
    rope_base: float = 10_000.0
    # FFN ve çıkış
    ffn_mult: float = 8 / 3
    ffn_multiple_of: int = 128
    logit_softcap: float = 30.0
    tie_embeddings: bool = True
    norm_eps: float = 1e-6
    kernel: str = "auto"           # auto: flash-linear-attention (Triton) varsa onu kullan | torch
    # MoE (uzman karışımı): 0 = yoğun SwiGLU. Toplam parametre büyür, token başına hesap küçük kalır.
    moe_experts: int = 0
    moe_topk: int = 4
    moe_hidden: int = 0            # uzman genişliği (0 → ffn_hidden / moe_topk)
    moe_shared: bool = True        # her tokenin geçtiği paylaşılan uzman
    # Gizli ağırlıksız düşük bit (bit.py): 0 = yoğun | 1 = ikili (±1) | 2 = üçlü (−1/0/+1). Embedding, LM başı,
    # kapılar ve normlar yoğun kalır.
    bits: int = 0
    bit_scope: str = "all"         # hangi matrisler bit: "all" | "ffn" (yalnız SwiGLU) | "mix" (yalnız karıştırıcı)
    bit_rank: int = 0              # >0: her BitLinear'a yoğun düşük-rank düzeltici (U·V, bit ağırlıkların yanında)
    # MTP (çok-token tahmini, DeepSeek-V3 / Qwen3.5 tarzı): k. modül x_{i+k+1}'i tahmin eder. Ek kayıp yalnız eğitimde;
    # doğrulama kaybı ana tahmindir (karşılaştırılabilir). Modüller DeltaNet bloğudur (sabit bellek, taslak üretimi için).
    mtp_depth: int = 0
    mtp_weight: float = 0.3

    def __post_init__(self) -> None:
        assert self.layout and set(self.layout) <= {"D", "A"}, "layout yalnız D ve A içerebilir"
        assert self.attn_heads % self.attn_kv_heads == 0, "attn_heads, attn_kv_heads'e bölünmeli"
        assert 0 <= self.archival_heads <= self.n_heads
        assert self.conv_kernel >= 2 and self.head_dim % 2 == 0 and self.attn_head_dim % 2 == 0
        assert self.bits in (0, 1, 2), "bits: 0, 1 ya da 2"
        assert self.bit_scope in ("all", "ffn", "mix") and self.bit_rank >= 0
        assert not (self.bits and self.moe_experts), "bits ile MoE henüz birlikte desteklenmiyor"
        assert self.attn_rope in ("auto", "rope", "none") and self.kernel in ("auto", "torch")
        assert self.mtp_depth >= 0 and self.mtp_weight >= 0

    @property
    def kinds(self) -> list:
        return [self.layout[i % len(self.layout)] for i in range(self.n_layers)]

    @property
    def use_rope(self) -> bool:
        if self.attn_rope != "auto":
            return self.attn_rope == "rope"
        return "D" not in self.layout or self.attn_window > 0

    @property
    def ffn_hidden(self) -> int:
        h, m = int(self.ffn_mult * self.d_model), self.ffn_multiple_of
        return ((h + m - 1) // m) * m

    def to_dict(self) -> dict:
        return asdict(self)


# Mimari türleri: aynı ön ayarın üç biçimi (adil karşılaştırma için aynı genişlik/derinlik)
ARCHS = ("hibrit", "transformer", "sabit", "bit")


def apply_arch(cfg: HeimdallConfig, arch: str) -> HeimdallConfig:
    if arch == "hibrit":        # varsayılan: 3 DeltaNet + 1 global dikkat (NoPE)
        return replace(cfg, layout="DDDA", attn_window=0)
    if arch == "bit":           # Heimdall-Bit: hibrit + FFN gizli ağırlıksız üçlü + yoğun düşük-rank düzeltici (GPU ölçümü: bench/)
        return replace(cfg, layout="DDDA", attn_window=0, bits=2, bit_scope="ffn", bit_rank=16)
    if arch == "transformer":   # taban: her katman tam dikkat, RoPE, MHA
        return replace(cfg, layout="A", attn_window=0, attn_kv_heads=cfg.attn_heads)
    if arch == "sabit":         # global dikkat yok: çıkarım belleği bağlamdan bağımsız
        return replace(cfg, layout="DDDA", attn_window=cfg.attn_window or 1024)
    raise ValueError(f"bilinmeyen mimari: {arch}")


@dataclass
class TrainPreset:
    model: HeimdallConfig
    seq_len: int = 2048
    micro_batch: int = 8
    batch_tokens: int = 262_144
    lr_muon: float = 0.02
    lr_adam: float = 3e-3
    grad_ckpt: bool = False


PRESETS = {
    # ~8M: duman testi, CPU'da da çalışır
    "mini": TrainPreset(HeimdallConfig(d_model=256, n_layers=8, n_heads=2, archival_heads=1,
                                       attn_heads=4, attn_kv_heads=2), 512, 16, 32_768),
    # ~45M
    "kucuk": TrainPreset(HeimdallConfig(d_model=512, n_layers=12, n_heads=4, archival_heads=1,
                                        attn_heads=8, attn_kv_heads=2), 1024, 16, 131_072),
    # ~100M
    "temel": TrainPreset(HeimdallConfig(d_model=768, n_layers=12, n_heads=6, archival_heads=2,
                                        attn_heads=12, attn_kv_heads=4), 2048, 8, 262_144, lr_adam=2e-3),
    # ~330M: gradient checkpointing ile 16 GB
    # ~540M toplam / ~150M aktif MoE: token başına "temel" kadar hesap, 5 kat kapasite
    "moe": TrainPreset(HeimdallConfig(d_model=768, n_layers=16, n_heads=6, archival_heads=2, attn_heads=12,
                                      attn_kv_heads=4, moe_experts=32, moe_topk=4, moe_hidden=384), 2048, 8, 262_144,
                       lr_adam=2e-3),
    "buyuk": TrainPreset(HeimdallConfig(d_model=1024, n_layers=24, n_heads=8, archival_heads=2,
                                        attn_heads=16, attn_kv_heads=4), 2048, 8, 262_144,
                         lr_muon=0.015, lr_adam=1.5e-3, grad_ckpt=True),
}
