# Kuzgun — nihai mimari (v1)

> Global dikkat yok. Her token için iş ve bellek bağlam uzunluğundan bağımsız (O(1)).
> Adı: Odin'in iki kuzgunu. **Huginn** (düşünce) yakın geçmişe tam dikkatle bakar,
> **Muninn** (hafıza) geri kalan her şeyi sabit boyutlu bir delta hafızada tutar.
> Kod: `bifrost/layers/kuzgun.py` · Eğitim: `train.py` · Testler: `tests/test_model.py`

## 1. Blok

$$x \leftarrow x + \mathrm{Kuzgun}(\mathrm{RMSNorm}(x)), \qquad x \leftarrow x + \mathrm{SwiGLU}(\mathrm{RMSNorm}(x))$$

Çıkış: $\text{logits} = 30\tanh\!\big(E^\top \mathrm{RMSNorm}(x)/30\big)$ (tied embedding, logit softcap).

## 2. Kuzgun katmanı (H kafa, kafa boyutu $d_h$, pencere W)

**Ortak anahtarlar (Canon conv):**
$$[q;k;v]_t = \mathrm{SiLU}\big(\mathrm{DWConv}_4(W_{qkv}x)\big)_t,\qquad \hat q=\tfrac{q}{\|q\|},\ \hat k=\tfrac{k}{\|k\|}$$

**Huginn — yakın geçmiş, tam token hassasiyeti:**
$$o^{H}_t=\sum_{s=t-W+1}^{t}\mathrm{softmax}_s\big(\tau_h\langle R_t\hat q_t,\,R_s\hat k_s\rangle\big)\,v_s$$
($R$: RoPE, $\tau_h$: kafa başına öğrenilen sıcaklık.) Maliyet O(W) / token, cache son W−1 token.

**Muninn — tüm geçmiş, sabit boyutlu delta hafıza:**
$$\alpha_t=\begin{cases}1 & \text{arşiv kafası}\\ \exp\!\big(-e^{A_h}\,\mathrm{softplus}(w_a^\top x_t+b_h)\big) & \text{unutan kafa}\end{cases}\qquad \beta_t=\sigma(w_b^\top x_t)$$
$$S_t=\alpha_t S_{t-1}+\beta_t\,\hat k_t\big(v_t-\alpha_t S_{t-1}^\top\hat k_t\big)^\top,\qquad o^{M}_t=S_t^\top\hat q_t/\sqrt{d_h}$$
Tahmin et → sürprizi bul → yalnız sürprizi yaz (V-PDM'in doğru hali). Aynı anahtara yeni değer eskisini siler.
Durum $S\in\mathbb{R}^{d_h\times d_h}$ kafa başına; bağlam uzunluğundan bağımsız.

**Birleşim ve çıkış:**
$$o_t=\sigma(m^H(x_t))\,\mathrm{RMSNorm}(o^H_t)+\sigma(m^M(x_t))\,\mathrm{RMSNorm}(o^M_t),\qquad y_t=W_o\big(o_t\odot\mathrm{SiLU}(W_g x_t)\big)$$

## 3. Neden bu tasarım (ölçülen kanıt)

| Karar | Kanıt (bu oturumda ölçüldü) |
|---|---|
| Uzun mesafe için delta hafıza | MQAR (8 çift, üzerine yazma dahil): yalnız hafıza **1M token boşlukta %100**. Aynı bütçede tam dikkat öğrenemedi (%15). Yalnız pencere, pencere dışında şans seviyesinde |
| Delta kuralı (toplamsal değil) | "Proje ID 1→2→3": delta kuralı 3'ü bulur (cos 1.00), toplamsal hafıza üçünü karıştırır (0.58/0.58/0.58) |
| **Arşiv kafaları (α ≡ 1)** | Tanı: öğrenilen α=0.99997; 65K tokende 0.12'ye iner, doğruluk %59. α=1 yapılınca %96. Pencere kolunun etkisi yok. Önemsiz token, delta kuralında yalnız kendi anahtar yönünü değiştirir |
| Pencere kolu (Huginn) | Yakın geçmiş için tam token kopyalama ve sözdizimi. Katkısı gerçek kod karşılaştırmasıyla ölçülüyor |
| Chunk-paralel eğitim | Token döngüsüyle aynı sonuç (5e-7); ileri+geri CPU'da ~40-50× hızlı |
| Muon + WSD | Gizli matrisler Muon'da, gerisi AdamW'de; süre bütçeli WSD |

## 4. Karmaşıklık

| | Eğitim / token | Çıkarım belleği | 1M token bağlamda |
|---|---|---|---|
| Transformer | O(T·d) (karesel toplam) | O(T) KV-cache | KV büyür (nano: ~4 GB) |
| **Kuzgun** | O(W·d + C·d_h + d_h²) | **O(1)**: pencere (W−1 token) + S + conv | Sabit (d=256, 6 katman: ~1.2 MB) |

## 5. Önerilen ayarlar

| Ölçek | d | L | H (arşiv) | W | T (eğitim) | Donanım |
|---|---|---|---|---|---|---|
| CPU deney | 256 | 6 | 4 (2) | 64 | 512 | bu oturum |
| ~125M | 768 | 12 | 12 (4) | 256 | 2K → 8K | GPU (öneri, ölçülmedi) |

Eğitim: Muon lr 0.02 (Nesterov, NS5, birleşik matrisler parça parça), AdamW 3e-3 (β = 0.9/0.95), WSD (%2 ısınma, son %30 düşüş), çıkış projeksiyonları sıfır başlangıç, grad clip 1.0.

## 6. Sınırlar (dürüstçe)
- Delta hafızanın kapasitesi kafa başına ~$d_h$ bağımsız anahtar. Daha fazlasında karışma olur; büyük ölçekte ölçülmeli.
- Pencere dışındaki uzun metinleri **birebir** geri getirme kayıplı. Gerekirse seyrek blok bulucu eklenebilir; o ancak ölçüm gerektirirse eklenecek.
- Genel dil kalitesinde Transformer'ı geçtiğine dair kanıt için GPU ölçeğinde eğitim gerekiyor.
