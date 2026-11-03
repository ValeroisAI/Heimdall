# 🤖 AGENT BRIEFING: VALEROIS NEURAL INTELLIGENCE
> **DİKKAT YENİ AJAN (AI AGENT):** Bu dosyayı baştan sona oku. Bu belge, projeyi sıfır saniyede anlaman ve kullanıcının bugünkü hedefini eksiksiz tamamlaman için hazırlanmış **resmi bağlam brifingidir.**

---

## 🎯 1. KULLANICININ BUGÜNKÜ GÖREVİ (TODAY'S MISSION)
Kullanıcı okul laboratuvarında çalışmaktadır. Bugünkü ana hedef:
1. **Bifrost CSL (Continuous State Learning) mimarisine:**
   - **Cache (Önbellek):** Rolling FIFO buffer / ring-buffer mekanizması eklemek (çıkarımda $O(1)$ sabit zaman/bellek).
   - **Sliding Window (Kayan Pencere):** Yerel yüksek çözünürlüklü token penceresi bağlamak.
   - **Attention-tarzı katman:** Saf konvolüsyonun çözemediği "ilişkisel anahtar-değer arama ve uzun vadeli değişken hatırlama" sorununu çözmek.
2. **Sanal Receptive Field Büyütme:**
   - Modelin tek bir ileri geçişte (forward pass) görebileceği sanal token sayısını (örneğin Landmark/Chunk özetleme veya üstel dilatasyon ile) binlerce/on binlerce tokene çıkarmak.

---

## 🏛️ 2. MİMARİ VE GEÇMİŞİN ÖZETİ (NE OLDU, NE BİTTİ?)
* **Valerois Nedir?** Standart Transformer'ın $O(N^2)$ işlem ve terabaytlarca KV-Cache hamallığını aşmak için geliştirilen $O(1)$ hafızalı, doğrusal akışlı model ailesidir.
* **Bifrost CSL:** Causal Depthwise Dilated Convolution + SwiGLU FFN bloğudur. GPU üzerinde ışık hızında akar (`groups=hidden`), yerel sözdizimini (syntax, girintiler) kusursuz öğrenir.
* **GCAM (Gated Content-Addressable Memory):** 64/512'lik chunk'lar içinde Softmax, chunk'lar arasında ise $H \times D_h \times D_h$ ($128 \times 128$) sabit matris durumu ($S$) kullanan mimari. 1 Milyon tokenlik bağlamı 25 MB bellek ile akıtır.
* **Son Turnuva (9-Model Grand Tournament):**
  - **Şampiyon (2.51 loss):** `Differential Attention + CSL Conv + Sparse gating` (C3).
  - **İkinci (2.52 loss):** `Differential Attention + CSL Conv` (B1).
  - **Keşif:** CSL Conv yerel sözdizimini çözerken, Differential Attention gürültüyü filtreliyor. Bu ikili harikalar yaratıyor!
* **Yeni Teori: V-PDM (Valerois Predictive Delta Memory):**
  - Formül: Her token için hafızadan tahmin et ($\hat{v}$), sürprizi bul ($\delta = v - \hat{v}$), ve **sadece sürprizi hafızaya kazı** ($S_t = \Lambda S_{t-1} + k \otimes \delta$).
  - Detaylar ve çalışan şablon `06_YENI_FIKIR_V_PDM/` içinde hazırdır.

---

## 📂 3. KLASÖR VE DOSYA REHBERİ
* `stream_coder_100k.bin`: 25.6M tokenlik gerçek kod veri akışı (kök dizinde ve `02_TURNUVALAR_VE_TESTLER/` altında kopyalandı, testler doğrudan çalışır!).
* `01_KATMANLAR_VE_MIMARILER/`: GCAM-v2, GCAM-v3, Delta-CSL, Pure-CSL, GLA, Matrix-SSD katmanları.
* `02_TURNUVALAR_VE_TESTLER/`: 9 modelli turnuva (`grand_architecture_tournament_9.py`), 1M streaming testleri (`test_1m_context.py`), İğne-Samanlık testleri (`test_valerois_gcam_needle.py`).
* `03_CEKIRDEK_MODELLER/`: `valerois_core_model.py`, `valerois_coder_7b_core.py`, `pulse_engine.py`, `valerois_custom_quant_engine.py`.
* `04_TOKENIZERLAR/`: `valerois_tokenizer_8k.json` (Vocab size = 8192).
* `05_DOKUMANTASYON_VE_RAPORLAR/`: Tüm teknik kitapçıklar, benchmark analizleri ve 44 deneylik maraton raporları.
* `06_YENI_FIKIR_V_PDM/`: Sıfırdan katman fikri V-PDM spesifikasyonu ve çalışan PyTorch şablonu.
* `07_BIFROST_CSL/`: Bifrost CSL, üstel dilatasyonlu CSL (`valerois_exponential_csl.py`), CSL-QV5 üç kanallı katman (`valerois_csl_qv5.py`).
* `08_BIFROST_H/`: **Güncel ana plan** (`PLAN_BIFROST_H.md`), mimari tanımı (`KUZGUN.md`), durum notu (`DURUM.md`) ve ölçüm betikleri. Yeni işe başlamadan önce `DURUM.md`'yi oku.
* `GercekMimari/`: ROCm için hazır Kuzgun eğitim sistemi (`egit.py`, `uret.py`, `degerlendir.py`, `araclar/`).

---

## ⚡ 4. KODLAMA VE ÇALIŞTIRMA KURALLARI
1. **Tokenizer:** Her zaman `valerois_tokenizer_8k.json` kullan (Sözlük boyutu = 8192).
2. **Device Fallback:** Kodlarda `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` kullan ki öğrenci bilgisayarlarında GPU olmasa bile CPU'da tıkır tıkır çalışsın.
3. **Bellek / OOM Hassasiyeti:** Batch size ve sequence length değerlerini makul tut (Örn: BS=4-8, Seq=256-512). Asla kontrolsüz VRAM tüketme.
4. **Veri Yükleme:**
   ```python
   raw_data = np.memmap("stream_coder_100k.bin", dtype=np.uint16, mode="r")
   ```
5. **Kullanıcıya Saygı:** Kullanıcı donanım sınırlarını bilen, deneysel sonuçlara ve ölçümlere önem veren kıdemli bir araştırmacıdır. Varsayımlarla değil, ölçülebilir loss, VRAM ve tok/s metrikleriyle konuş.
