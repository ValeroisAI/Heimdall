# GercekMimari — Kuzgun

Global dikkati olmayan, çıkarım belleği bağlam uzunluğundan bağımsız (O(1)) bir dil modeli ve ROCm için hazırlanmış eğitim sistemi.

**Kuzgun katmanı** (Odin'in iki kuzgunu):
- **Huginn:** Son W tokene tam softmax dikkat. Yakın geçmiş token hassasiyetiyle görülür: sözdizimi, kopyalama.
- **Muninn:** Kapılı delta kuralı hafızası `S = αS + β k̂ (v − α Sᵀk̂)ᵀ`. Pencerenin ötesindeki her şey sabit boyutlu bir matriste tutulur. Aynı anahtara yeni değer gelince eskisi silinir, yani "proje id 1 → 2 → 3" sorusunda cevap 3 olur.
- **Arşiv kafaları (α ≡ 1):** Hafıza kafalarının bir kısmı hiç unutmaz. CPU deneylerinde, unutan hafızanın 65K+ tokende koptuğu, α=1'in bunu düzelttiği ölçüldü.
- Ortak q/k/v, kısa nedensel conv (Canon), kafa başına kapı, SwiGLU, logit softcap.
- Matematik ve ölçüm geçmişi: `../08_BIFROST_H/KUZGUN.md`.

## Kurulum (ROCm, RX 9070 XT)

```bash
python -m venv .venv && source .venv/bin/activate
# ROCm sürümüne uygun PyTorch (pytorch.org → Get Started → ROCm). Örnek:
pip install torch --index-url https://download.pytorch.org/whl/rocm6.4
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.version.hip, torch.cuda.get_device_name(0))"
```

ROCm'da PyTorch cihazı yine `cuda` adıyla görür, bu normal. Bellek ve dikkat çekirdeği ortam değişkenlerini `egit.py` kendisi ayarlar.

**Windows + DirectML:** `pip install torch-directml`, sonra komutlara `--device dml` ekle. DirectML'de `torch.compile` ve bf16 kullanılamıyor; eğitim fp32 ve derlemesiz çalışır. Bu yüzden Linux ROCm'a göre birkaç kat yavaştır; ciddi uzun eğitim için ROCm önerilir.

## Hızlı başlangıç

`GercekMimari/` klasörünün içinden çalıştır. `stream_coder_100k.bin` repo kökünde (`main` dalında) duruyor.

```bash
# 1) Hafıza ve akıl yürütme öğreten sentetik veri (~1 dk)
python araclar/sentetik_veri.py --dolgu ../stream_coder_100k.bin --cikti veri/sentetik_8k.bin --token 30e6

# 2) Duman testi: ~6M model, birkaç dakika
python egit.py --preset deneme --data ../stream_coder_100k.bin --tokens 20e6 --out kosular/deneme

# 3) Asıl eğitim: ~100M model
python egit.py --preset temel \
    --data ../stream_coder_100k.bin veri/sentetik_8k.bin:0.05 \
    --tokens 2e9 --out kosular/temel

# 4) Kaldığı yerden devam (Ctrl+C güvenli: çıkarken kaydeder)
python egit.py --resume kosular/temel/son.pt

# 5) Üretim: istem ne kadar uzun olursa olsun cache sabit
python uret.py --ckpt kosular/temel/son.pt --prompt "def quicksort(arr):"

# 6) Ölçüm: val loss, pozisyona göre loss, 1M tokene kadar PROJE_ID iğnesi + son değer testi
python degerlendir.py --ckpt kosular/temel/son.pt --data ../stream_coder_100k.bin
```

Kendi verin için: `python araclar/veri_hazirla.py --girdi ~/kodlarim --cikti veri/kendi_8k.bin`

Veri karışımı: `yol.bin:ağırlık` yazımı kullanılır. Ağırlık verilmeyen dosya 1.0 sayılır; hiçbirine ağırlık verilmezse pay dosya boyutuyla orantılı olur.
Örnek: `--data ../stream_coder_100k.bin:1 fineweb_edu_8k.bin:1 sentetik_8k.bin:0.1`

## Ön ayarlar (16 GB)

| Ön ayar | Param | d × L | Kafa (arşiv) | Pencere | T | Mikro-batch | Adım başına token | Grad ckpt |
|---|---|---|---|---|---|---|---|---|
| `deneme` | ~6M | 256 × 6 | 4 (2) | 64 | 512 | 8 | 16K | – |
| `kucuk` | ~19M | 384 × 8 | 6 (2) | 128 | 1024 | 16 | 64K | – |
| `orta` | ~45M | 512 × 12 | 8 (3) | 256 | 2048 | 8 | 128K | – |
| `temel` | ~100M | 768 × 12 | 12 (4) | 256 | 2048 | 4 | 256K | – |
| `buyuk` | ~280M | 1024 × 20 | 16 (6) | 512 | 2048 | 8 | 256K | ✓ |

Her ayar komut satırından ezilebilir: `--seq-len`, `--micro-batch`, `--batch-tokens`, `--window`, `--archival`, `--lr-muon`, `--lr-adam`.

## Hız ve verim

| Özellik | Açıklama |
|---|---|
| bf16 + `torch.compile` | Varsayılan olarak açık. İlk adımlar derleme yüzünden 1-3 dk yavaş, sonra hızlanır. `--compile-mode max-autotune-no-cudagraphs` daha uzun derler, daha hızlı koşabilir |
| Muon + fused AdamW | Gizli matrisler Muon'da (Newton-Schulz bf16'da), embedding/norm/kapılar AdamW'de. Karşılaştırma için `--adamw-only` |
| Dizi uzunluğu müfredatı | Eğitimin ilk %15'i T/4, %35'e kadar T/2, sonra T. Kısa aşamada batch otomatik büyür, adım başına token sabit kalır. Kapatmak için `--no-curriculum` |
| WSD çizelgesi | %1 ısınma, son %25'te doğrusal düşüş. Eğitim uzatılabilir |
| Arka plan veri yükleyici | Pinned bellek + asenkron GPU'ya taşıma |
| Delta hafıza | Chunk-paralel (UT dönüşümü), FP32. Token-token referansla birebir aynı sonuç |
| Pencere dikkati | Blok-yerel SDPA (her yerde çalışır). `--attn flex` PyTorch flex_attention ile blok-seyrek çekirdek dener |
| Bellek | OOM'da `--micro-batch` düşür ya da `--grad-ckpt` aç (~%25 yavaşlar) |
| MTP | `--mtp`: ek olarak t+2 tahmini. Daha büyük modellerde örnek verimini artırır |

Log her 10 adımda tok/s, tahmini TFLOPS, VRAM, grad normu yazar. `--peak-tflops` verilirse MFU da hesaplanır. `kosular/<ad>/log.jsonl` dosyasına da kaydeder.

## Deneysel: 16 GB'da daha büyük model

İki bağımsız bayrak var. İkisi de varsayılan olarak kapalı.

### `--uclu`: gizli ağırlıksız üçlü eğitim (`kuzgun/uclu.py`)
BitNet b1.58 çıkarımda üçlü ağırlık kullanır, ama eğitimde her ağırlığın fp32 gizli kopyasını ve AdamW durumunu tutar (~16 bayt/param). Burada gizli kopya yok:
- Ağırlık bf16 bir tensördür ama değerleri daima −1, 0 veya +1'dir ("sanal bf16": kart sıradan bf16 matmul görür). Satır başına öğrenilen bir ölçek vardır.
- `TernaryFlip` optimizer'ı gradyan momentumunu biriktirir. Her adımda momentumu en güçlü olan ağırlıkların `--flip-orani` kadarını bir basamak çevirir (+1 → 0 → −1 ya da tersi). Öğrenme hızının karşılığı bu orandır ve WSD çizelgesini izler.
- Bellek: ağırlık 2 + gradyan 2 + momentum 2 = **6 bayt/param**. AdamW + fp32 kopyada bu 16-18 bayttır, yani aynı 16 GB'a kabaca **2.5-3 kat büyük model** sığar. Aktivasyon belleği değişmez; gerekirse `--grad-ckpt` açılır.
- `--uclu-int8`: ileri geçişte aktivasyonlar int8'e nicemlenir, `torch._int_mm` ile tamsayı çarpımı yapılır. RDNA4'te INT8 hızı bf16'nın ~2 katıdır. `torch._int_mm` ROCm'da yoksa ya da yanlış sonuç veriyorsa uyarı basılır ve bf16'ya dönülür.
- Embedding, LM başı, kapılar ve normlar yoğun kalır (BitNet'te olduğu gibi).

**Risk:** Dil modellerinde gizli ağırlıksız eğitim denenmemiş bir yöntemdir. Aynı token bütçesinde yoğun modelden daha yüksek loss beklenir. Bu yöntem kazanç sağlıyorsa, bunun sebebi aynı bellekte çok daha büyük bir model eğitebilmek olur.

### `--hafiza-katmani 4 8`: product-key hafıza (`kuzgun/hafiza.py`)
Seçilen bloklara, karıştırıcı ile FFN arasına bir hafıza katmanı eklenir (Lample ve ark. 2019; Meta "memory layers" 2024).
- n² yuva vardır (`--hafiza-yuva 512` → 262K yuva). Token başına yalnız `kafa × top-k` (varsayılan 4 × 32 = 128) satır okunur.
- Parametre sayısı yüz milyonlara çıkar ama token başına hesap küçük kalır. Bu, bilgi kapasitesini FLOP'tan ayırmanın bir yoludur.
- Değer tabloları seyrek gradyanla ve satır başına tek ölçek tutan `SparseRowRMS` ile güncellenir (Adam'ın iki tam kopyası yok). Öğrenme hızı `--lr-hafiza` ile verilir, varsayılanı AdamW lr'sidir.
- Hafıza katmanı token başınadır, cache gerektirmez. Çıkarım belleği O(1) kalır.
- Yer: `hafiza-yuva 512`, d=768 için katman başına 805 MB fp32 tutar (ağırlık, seyrek gradyan ve optimizer durumu dahil ~1 GB).

```bash
# ~100M yoğun + 2 hafıza katmanı (2 × 201M hafıza parametresi), gizli matrisler üçlü
python egit.py --preset temel --uclu --uclu-int8 --hafiza-katmani 4 8 \
    --data ../stream_coder_100k.bin --tokens 2e9 --out kosular/uclu_hafiza
```

Genişlik ve derinlik ön ayardan bağımsız verilebilir: `--d-model`, `--n-layers`, `--n-heads`.

### İlk karşılaştırma (CPU, eşit token)
d=192, 4 katman, T=256, 1M token, aynı veri sırası. Val loss `stream_coder_100k` üzerinde ölçüldü.

| Koşu | Val loss | Fark |
|---|---|---|
| Yoğun (Muon + AdamW) | 2.811 | — |
| Yoğun + hafıza katmanı (16K yuva) | **2.795** | −0.016 |
| Üçlü, flip %0.5 | 3.074 | +0.263 |
| Üçlü, flip %2 | 3.112 | +0.301 |
| Üçlü + hafıza katmanı | 3.024 | +0.213 |
| Üçlü, flip 0: gizli katmanlar hiç öğrenmez (alt sınır) | 4.751 | +1.940 |

- **Gizli ağırlıksız üçlü eğitim öğreniyor.** Donuk alt sınır ile yoğun model arasındaki farkın %86'sını kapatıyor.
- Eşit boyutta yoğun modelden 0.26 nat geride. Asıl soru henüz ölçülmedi: aynı VRAM'e sığan 2.5-3 kat büyük üçlü model, yoğun modeli geçer mi? Bu ölçüm GPU'da yapılmalı.
- Hafıza katmanı 1M tokende az katkı veriyor, çünkü her yuva birkaç kez görülüyor. Üçlü modelde katkısı daha büyük (−0.05). Etkisi uzun eğitimde ölçülmeli.
- Bu küçük ölçekte bellek aktivasyonlarla dolu; üçlünün optimizer belleği kazancı ancak büyük modelde görünür.

## Veri miktarı notu
`stream_coder_100k.bin` 25.6M token. ~100M'lik bir model için azdır: birkaç epoch'a kadar tekrar sorun değil, ama daha fazlası ezberletir. `fineweb_edu_8k.bin`, `master_code_8k.bin` gibi dosyaları ekle ve val loss'u izle (`[val]` satırları).

## Durum
Kod bu oturumda CPU'da çalıştırılarak kontrol edildi:
- İleri/geri geçiş ve cache ile üretimin paralel hesapla eşdeğerliği.
- Uçtan uca zincir: sentetik veri → eğitim → checkpoint → devam → üretim → değerlendirme.

ROCm ve DirectML GPU'da henüz denenmedi. İlk koşuda sorun çıkarsa hata çıktısını ilet.
