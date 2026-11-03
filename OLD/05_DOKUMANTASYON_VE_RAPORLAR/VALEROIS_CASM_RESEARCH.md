# Valerois CASM: Counterfactual Adaptive Semantic Memory

## Dürüst hedef

Bu belge “8B genel model, bütün 14B coder modellerini geçer” iddiası değildir.
Test edilebilir hipotez şudur:

> Kodda, blok başına **pahalı global retrieval** ancak bu retrieval'ın gerçek
> next-token kaybını düşürdüğü yerlerde etkinleştirilirse; eşit aktif-FLOP ve
> eşit VRAM altında, sabit local-CSL veya sabit sparse-attention modelinden daha
> iyi bir code-loss / test-pass Pareto eğrisi elde edilir.

Bu, 7--8B bir coder'ın genel bilgi kapasitesiyle değil, test edilen repository
patch görevlerinde `test-pass / aktif FLOP / context token` oranıyla yarışmasını
hedefler.

## Neden "sıfırdan 8B" bu donanımda doğru ilk deney değil?

Yaklaşık eğitim hesabı `6 * active_parameters * training_tokens` FLOP'tur.
Yaygın compute-optimal başlangıç noktası olan 20 token/parametre ile 8B dense
model için:

```
tokens  ~= 160B
FLOPs   ~= 6 * 8B * 160B = 7.68e21
```

Bu, 100 TFLOP/s *sürdürülebilir* eğitim verimi varsayımında bile yaklaşık 2.4
GPU-yılıdır; model/gradient/optimizer state'i de tek 16 GB GPU'ya sığmaz. Bu
hesap bir ürün kararını belirler: önce 30--150M ölçekli, aynı algoritmayı
denetleyen bir deney yapılmalıdır. Bir iddia ancak bu deneyde eşit bütçeli
baseline'ı geçerse 7--8B ölçeğine taşınmaya değer.

## Mimari: üç hesap seviyesi

```
tokenlar
  └─ Fast path: causal local mixer (depth-wise conv / CSL)       [her blok]
       └─ semantic probe: lexer/AST state + blok özeti           [her blok]
            ├─ cheap: yalnız local sonucu kullan                 [çoğu blok]
            └─ expensive: top-K eski bloktan exact attention
                    + code expert                                [sınırlı blok]
```

* Bloklar 16 tokenlik mikrobloklardır; karar token bazında değil blok bazında
  verilir. Bu, GPU'da statik şekil ve dengeli batch sağlar.
* Global read, seçilen geçmiş blokların **ham K/V**'sine exact softmax uygular;
  yalnız ortalama state okumaz. Böylece identifier/sayı/call-site gibi keskin
  bilgi linear-memory süperpozisyonunda bulanıklaşmaz.
* `K=2--4` uzak blok ve bir local pencere sabittir. Dolayısıyla aktif global
  hesap için üst sınır önceden bellidir.
* Bir lexer/parser yalnız geçmiş tokenlardan üretilen özellikleri sağlar:
  indent derinliği, parantez/quote durumu, import/def/class/call adayları ve
  identifier hash sınıfı. Parser geleceği görmez; bu nedenle causal sızıntı
  olmaz.

## Farklı olan bölüm: counterfactual compute etiketi

Çoğu sparse router, `q·k` benzerliğini öğrenir. Bu "alakalı blok" demektir ama
"hesap harcamaya değer blok" demek değildir. CASM router'ı doğrudan ikinci
soruyu öğrenir.

Her eğitim mikroblokunda iki teacher-forcing forward sonucu hesaplanır:

```
L_fast = yalnız local CSL next-token CE
L_slow = aynı blok + selected global memory / code expert CE
gain   = stopgrad(L_fast - L_slow)
target = gain > compute_price
```

Router loss'u `BCE(router_probability, target)`; ana loss ise budget altında
seçilmiş slow-path ile hesaplanır. `compute_price`, seçili blok oranını örneğin
%20'de tutacak Lagrange çarpanıdır. Böylece router, önemlilik tahmini değil,
**ölçülmüş marjinal loss kazancı** tahmini yapar.

Bu fikir tek başına "dünyada ilk" diye iddia edilemez: adaptive compute,
sparse attention, compiler features ve counterfactual supervision'ın yakın
akrabaları vardır. Araştırma iddiası ancak aşağıdaki ablation'lar ve açık kod
ile savunulabilir: compiler-state özellikleri + block-budget + counterfactual
loss hedefinin birlikte Pareto üstünlüğü.

## Zorunlu ablation tablosu

Tüm modeller aynı tokenizer, parametre sayısı, training tokenı, seed sayısı
ve active-FLOP bütçesi ile çalışır:

| Model | Router hedefi | Compiler state | Exact remote KV |
|---|---|---:|---:|
| Dense local CSL | yok | hayır | hayır |
| Sabit sparse | sabit K blok | hayır | evet |
| Similarity router | q·k top-K | hayır | evet |
| CASM-no-lexer | counterfactual gain | hayır | evet |
| CASM | counterfactual gain | evet | evet |

Ölçümler:

1. Held-out code next-token loss ve syntax validity.
2. Random-renamed long-range symbol binding: exact identifier accuracy.
3. Repo patch görevi: patch apply + test-pass oranı.
4. `tokens/s`, peak allocated VRAM, seçilen blok oranı, active-FLOP.
5. Router calibration: seçilen blokların gerçek `L_fast-L_slow` kazancı.

Bir router'ın yalnız "top-K doğru chunk" seçmesi başarı değildir. Çıktı
doğruluğu ve gerçek wall-clock ölçülmelidir.

## Araştırma sırası

### Aşama 0 — 2 günde falsification

* 30--50M parametre, 8K context, 16-token mikroblok.
* Sentetik fakat zor code seti: rastgele identifier rename, nested scope,
  import alias, function signature ve çapraz-dosya call hedefleri.
* Yukarıdaki beş ablation'dan en az local CSL, similarity router ve CASM.

**Öldürme koşulu:** CASM, aynı active-FLOP'ta symbol accuracy veya validasyon
lossunda belirgin üstün değilse mimariyi büyütme.

### Aşama 1 — gerçek code

* Lisansı temiz Python corpus + commit-before/after patch çiftleri.
* 100--150M model, 16K context; önce code-loss ve long-context testleri.
* Kısa bir held-out repo listesinde context retrieval recall ölç.

### Aşama 2 — agent sistemi

* Bir 7--8B **mevcut** coder modeli 4-bit inference ile kullanılır; CASM'nin
  küçük modelini repo-context seçici olarak kullanır.
* Agent: issue -> context -> patch -> test -> yalnız hata varsa yeni context.
* Başarı ölçüsü sadece solve-rate değil: aynı solve-rate için toplam context
  tokenı, inference süresi ve VRAM.

### Aşama 3 — 7--8B ağırlık eğitimi (yalnız Aşama 0/1 kazanırsa)

Sıfırdan değil; açık ağırlıklı code base üzerinde QLoRA / continued pretraining
ile başla. 16 GB tek GPU'da yapılabilir olan budur. Sıfırdan 8B için önce çoklu
GPU/hibe/para gerekir; quantized frozen base + LoRA "from scratch pretraining"
değildir ve öyle sunulmamalıdır.

## Mevcut kod tabanından alınan ders

* `radar_attention.py` iyi araştırma sezgisine sahip (centroid ile route,
  sonra exact KV), fakat Python `for` döngüsü ve batch/head ortalama seçimiyle
  8B eğitim performansına taşınamaz.
* `valerois_moe_hierarchical_8b.py`deki soft-MoE tüm expert'ları hesaplıyor;
  gerçek top-k dispatch olmadığı için "active compute" iddiası doğru değil.
* QV4, inference/sonradan quantization katmanıdır. Sıfırdan eğitimde frozen
  4-bit base üzerine LoRA eklemek, tam ağırlık öğrenmesi değildir.

## Kaynaklar

* Hoffmann et al., Chinchilla compute/data scaling:
  https://arxiv.org/abs/2203.15556
* Dao & Gu, Mamba-2 / structured state space duality:
  https://arxiv.org/abs/2405.21060
* MoBA, block-level sparse attention:
  https://arxiv.org/abs/2502.13189

