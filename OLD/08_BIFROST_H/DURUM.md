# DURUM — Bifrost-H çalışma notu (kısa, güncel)

> Bu dosya oturumlar arası devir notudur. Uzun plan: `PLAN_BIFROST_H.md` (arka plan için).
> Son güncelleme: 24 Eylül 2026.

## Kararlar (kullanıcıyla)
- Modele **global (tam) dikkat konmaz**. Bellek O(1) veya hafif O(N) olmalı (CLAUDE.md kural 3).
- Transformer++ yalnız **ölçüm çizgisi** olarak benchmark'ta durur, modele girmez.
- Tokenizer: 8K BPE (`04_TOKENIZERLAR/valerois_tokenizer_8k.json`). Byte düzeyi reddedildi: diziler ~1.8× uzar, yavaşlatır.
- Kullanıcı kısa özet ister: her adımda birkaç satır + sonuç tablosu.

## Veri (repoda değil, yeniden üretilir)
- `python 08_BIFROST_H/tools/build_dev_corpus.py` → `08_BIFROST_H/data/dev_code_8k_{train,val}.bin`
  (40.9M / 1.55M token, Python kaynakları, dosya düzeyinde val ayrımı, 1.8 bayt/token).
- `stream_coder_100k.bin` (25.6M token) ve `Valkir.pt` (117M saf CSL) → `origin/main`'de (Valkir.pt Git LFS).
  `main` bu branch ile ortak geçmişe sahip değil; dosyalar ayrı klonla alınıp `data/` altına symlink'lendi.

## Ölçülen bulgular
| Konu | Sonuç | Kaynak |
|---|---|---|
| Eski katman denetimi | Radar ve turnuva B3 geleceği görüyor; QV5 step() `groups` hatası | `tools/audit_causality.py` |
| Mímir chunk-paralel | Referansla eşit (5e-7), CPU'da ileri+geri ~40-50× hızlı | `prototypes/mimir_proto.py` |
| Bellek/hız (nano, CPU) | Bifrost durum sabit 0.15 MB, ~19K tok/s (1K→1M). Transformer KV 4 MB/1K token, hız her 2×'de yarıya | `bench/memory_test.py` |
| Valkir.pt | Sabit dilatasyon 2 ile eğitilmiş (loss 0.72; üstel 1.73). Fiziksel görüş 992 token | `bench/valkir.py` |
| Valkir bağlam kullanımı | PROJE_ID iğnesi: 8-32 tokende +1.3…+2 nats, 128'de ~0.15, ≥512'de 0; hiçbir mesafede birebir kopyalayamıyor | `bench/valkir_context.py` |
| Valkir + Mímir ince ayar (9 dk CPU) | Başarısız: güçlü ek kayıp (×2) modeli bağlamdan bağımsız "ortalama sayı" tahminine çöktürdü (kazanç 0) | `bench/valkir_mimir.py` |
| Sentetik iğne, sıfırdan (CPU 11-25 dk) | Hiçbir model eşiği geçemedi (Transformer %27'de plato). CPU'da bu tür beceri binlerce adım istiyor | `bench/context_test.py` |

## Nihai mimari: Kuzgun (bkz. `KUZGUN.md`)
Pencere dikkati (Huginn, W token, RoPE) + delta hafıza (Muninn, O(1)), ortak q/k/v, kafa başına kapı.
Global dikkat yok. Arşiv kafaları (α ≡ 1) uzun mesafe unutmasını çözer.

| Deney (MQAR, d=64, 2 katman, 8 dk CPU) | 16K | 65K | 262K | 1M |
|---|---|---|---|---|
| Yalnız hafıza (R) | %100 | %100 | %100 | %100 |
| Kuzgun (K), unutma tüm kafalarda | %100 | %100 | %3 | %3 |
| Kuzgun, bağlı unutma | %98 | %59 | %2 | %0 |
| Tam dikkat (A) | %0 | — | — | — (bu bütçede öğrenemedi: 16 boşlukta %16) |
| Yalnız pencere (W) | şans | — | — | — |

Tanı (`bench/mqar_diagnose.py`, K_coupled, 65K): normal %58.6 → unutma kapalı (α=1) **%96.1**;
pencere kolu kapalı %58.6 (etkisiz). Öğrenilen α = 0.99997 → α^65536 = 0.12. Sebep: sızıntılı unutma.

Arşiv kafası düzeltmesi (yalnız hafıza, 2 kafa × 32, 1 arşiv): 256 → 262K boşlukta %94-97 (`results/mqar/R_arch.json`).
Kuzgun + arşiv kafası koşusu tamamlanmadı.

## Gerçek kod karşılaştırması (CPU, 1 thread, eşit 20 dk, T=512, Muon)
| Model | Param | Token | Hız | Final val (dev_code / stream_coder, 32 pencere) |
|---|---|---|---|---|
| Kuzgun (6×K, 4 kafa, 2 arşiv) | 7.21M | 1.04M | 860 tok/s | 2.881 / 3.106 |
| Transformer (6×A) | 6.77M | 1.54M | 1.283 tok/s | 2.632 / 2.913 |

- Eşit sürede Transformer önde: CPU'da T=512'de token başına 1.5× hızlı.
- Eşit tokende (hızlı val, 8 pencere) Kuzgun biraz önde görünüyor (0.91M'de 2.762 vs 0.95M'de 2.855). Ama LR çizelgesi süreye bağlı olduğu için bu kıyas karışık.
- Kuzgun'un avantajı uzun bağlamda (T ≫ W) ve çıkarım belleğinde. Kısa dizide ve CPU'da hız avantajı yok.
- Gradyan normları sağlıklı (Kuzgun blokları 0.10-0.49).

## ROCm eğitim sistemi
`GercekMimari/` (PR: ValeroisAI/Bifrost#1). CPU'da doğrulandı, GPU'da henüz çalıştırılmadı.

## Deneysel: gizli ağırlıksız üçlü eğitim + product-key hafıza (`GercekMimari`, `--uclu`, `--hafiza-katmani`)
CPU, d=192 × 4, T=256, 1M token, val loss: yoğun 2.811 | yoğun+hafıza 2.795 | üçlü (flip %0.5) 3.074 |
üçlü+hafıza 3.024 | donuk üçlü (alt sınır) 4.751. Üçlü eğitim öğreniyor (farkın %86'sı), eşit boyutta 0.26 nat
geride. Açık soru: aynı VRAM'de 2.5-3× büyük üçlü model yoğunu geçer mi? GPU'da ölçülecek.
