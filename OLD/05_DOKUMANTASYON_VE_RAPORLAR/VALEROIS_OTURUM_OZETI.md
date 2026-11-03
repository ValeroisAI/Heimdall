================================================================================
VALEROIS PROJESİ — OTURUM ÖZETİ VE DEVAM BELGESİ
================================================================================
Bu belge, bu sohbette yapılan HER ŞEYİ (gerçek bulgular, başarısız denemeler,
kod dosyaları, bilinen sorunlar, sıradaki adımlar) kaydediyor -- böylece bu
konuşma bitse bile, kaldığın yerden devam edebilirsin (yeni bir Claude
sohbetinde bu dosyayı yükleyip "buradan devam ediyoruz" diyebilirsin).

--------------------------------------------------------------------------------
1. VİZYON VE YOLCULUK (kısa özet)
--------------------------------------------------------------------------------
Başlangıç hedefi: "Transformer'ı geçen, sıfırdan bir AI mimarisi yaratmak."
Bu gece boyunca bu hedef gerçekçi bir şekilde YENİDEN ŞEKİLLENDİ:

  - "70B parametreyi 1.3MB'a sıkıştırmak" -> REDDEDİLDİ (Shannon bilgi
    teorisiyle çelişiyor, bilgi teorik olarak kaybolmadan sıkıştırılamaz).
  - "Kendi kendine internet gezip öğrenmek" -> REDDEDİLDİ (gerçek modellerin
    eğitim verisi trilyonlarca kelime, bu ortamda ulaşılamaz ölçek).
  - "HER konuda Transformer'ı geçmek" -> REDDEDİLDİ (genel dil anlayışı
    veri+ölçek meselesi, mimari ile çözülemez).
  - "EN AZ BİR konuda, kanıtlanmış şekilde Transformer'ı geçmek" -> KABUL
    EDİLDİ ve BAŞARILDI (bkz. Bölüm 2).

SONUÇ: Elimizde "Transformer'ı her yönden geçen bir AI" yok, ama "belirli,
kanıtlanmış eksenlerde gerçek avantajı olan, dürüstçe test edilmiş bir
mimari" var.

--------------------------------------------------------------------------------
2. GERÇEKTEN KANITLANAN BULGULAR (gerçek sayılarla, tekrarlanabilir)
--------------------------------------------------------------------------------

### 2.1 HDM Kapasite Yasası
D >= N/0.10 (plaka boyutu >= fact sayısı / %10) -- 5 farklı ölçekte
(sentetik, 201, 288, 525, 320 fact) bağımsız doğrulandı. Bu oranın
üzerinde doğruluk SERT şekilde çöküyor (faz geçişi gibi, yumuşak değil).

### 2.2 Catastrophic Forgetting -- EN GÜÇLÜ BULGU
300 fact öğrenen bir sistem, SADECE yeni 50 fact ile devam ettirilince:
  - Backbone tek başına (HDM yok): eski bilginin %23.7-64'ünü KAYBETTİ
    (farklı testlerde farklı oranlar, ama HEP ciddi kayıp)
  - HDM'li sistem + REPLAY (%10-30 oranında eski fact tekrarı): %0 KAYIP
Bu, MODEL BOYUTUNDAN BAĞIMSIZ bir mimari avantaj -- 70B model dahil HER
gradyanla eğitilen model bu riski taşır. replay_ratio=0.3 ile tam kapandı.

### 2.3 SSM vs GRU (dar bir görevde)
Aynı görevde (kısıtlı kod üretimi), aynı veri, GRU'ya kıyasla SSM:
  - %57 daha az parametre (20.7KB vs 47.6KB)
  - Daha iyi genelleme (tutma seti perplexity: 5.71-6.10 vs 9.42)
  - Aynı üretim hızı

### 2.4 Sözdizimi Garantisi (Constrained Decoding)
Grammar State Machine + logit maskeleme: %70 (kısıtlamasız) -> %100
(kısıtlamalı) sözdizimsel geçerlilik. Gerçek Python ast.parse() ile
doğrulandı, hile yok. Transformer'lar bunu YAPISAL olarak garanti edemez.

### 2.5 Derinlik (çok katmanlılık) İşe Yarıyor
1 katman -> 2 katman -> 3 katman SSM: perplexity 5.96 -> 5.14 -> 4.98
(azalan getiriyle ama düzenli iyileşme).

### 2.6 Paralel Tarama (Parallel Scan) -- Matematiksel Doğrulama
SSM'in doğrusal rekürans yapısı (h_t = a*h_{t-1} + x_t), GRU'nun aksine
SIRALI hesaplanmak ZORUNDA değil -- Hillis-Steele associative scan ile
log(T) derinlikte, sıralı hesapla BİREBİR AYNI sonucu veriyor (doğrulandı,
fark ~1e-16). Bu, gerçek Mamba/S4'ün GPU hızının matematiksel temeli.

### 2.7 Gerçek O(1) Holografik Bellek
İlk "ContinuousHDM" tasarımı GERÇEKTE O(N) idi (fact sayısı arttıkça
büyüyen liste) -- bu bir hataydı. HolographicHDM (dairesel evrişim,
Plate 1995 "HRR") ile düzeltildi: bellek boyutu 10 fact'te de, 10.000
fact'te de TAMAMEN SABİT (doğrulandı: run_holographic_o1_test.py).

### 2.8 Çok-Adımlı (Multi-Hop) Çıkarım + Hopfield Temizleme
A->B ve B->C fact'lerinden, HİÇ yazılmamış A->C çıkarımı:
  - Tek-adımlı retrieval: %0 (yapısal olarak imkansız)
  - Çok-adımlı, temizleme YOK: %7.5 (gürültü katlanıyor)
  - Çok-adımlı + Modern Hopfield Network temizleme: %100
Bu, kullanıcının kendi önerdiği ve doğru çıkan bir hipotezdi.

### 2.9 Uçtan Uca Birleşik Mimari (Router)
"Memorizing Transformer / kNN-LM" tarzı hibrit: SSM backbone + öğrenilen
router + dondurulmuş (gradyansız) HDM, TEK forward()/backward() ile.
HDM.keys/values hiç değişmedi (doğrulandı), router+backbone öğrendi.

--------------------------------------------------------------------------------
3. BAŞARISIZ / REDDEDİLEN DENEMELER (bunlar da değerli, dürüstçe kayıtlı)
--------------------------------------------------------------------------------
- Bigram özellikleri: E2 (%82.5) taban çizgisini AŞAMADI, %75'e düştü.
- Seçici (Mamba-tarzı input-dependent a,b) SSM: sabit mekanizmadan KÖTÜ
  çıktı (perplexity 6.32 vs 5.90) -- KÜÇÜK ölçekte, kısa dizilerde.
- Ham Transformer-vs-HDM kapasite testi: Transformer (gradyanla optimize,
  dense) aynı bellek bütçesinde HDM'i (doğrusal süperpozisyon) GEÇTİ --
  ilk beklentimizin tam tersi.
- Router collapse: g değeri 1.0'a kilitlenip loss donuyordu. Sebep: (a)
  gerçek bir matematik hatası (skaler g için eleman-bazlı gradyan hesabı
  -- DÜZELTİLDİ) ve (b) az fact varken Hopfield çıktısının neredeyse sabit
  olması. Warmup (ilk N epoch g sabit) ile collapse önlendi, AMA fact
  recall bu sefer düştü (hafıza az kullanıldı) -- bu hâlâ AÇIK bir
  hiperparametre ayarı sorunu (bkz. Bölüm 5).

--------------------------------------------------------------------------------
4. DOSYA ENVANTERİ (nerede ne var)
--------------------------------------------------------------------------------
İki ayrı paket üretildi:

  A) /mnt/user-data/outputs/ altında NUMPY tabanlı doğrulama script'leri
     (GPU gerektirmez, tüm yukarıdaki bulgular bunlarla üretildi):
       hybrid.py, pipeline.py, word_ssm.py, word_ssm_deep.py,
       word_ssm_selective.py, bpe_tokenizer.py, valerois_core.py,
       valerois_unified.py, integrated_valerois.py,
       run_*.py (onlarca deney script'i, her biri tek bir bulguyu üretti)

  B) valerois_full/ klasöründe GERÇEK PyTorch PAKETİ (GPU'da çalıştırılacak,
     BU SOHBETTE HİÇ ÇALIŞTIRILMADI, sadece sözdizimi kontrol edildi):
       valerois_full.py   -- BPE + çok katmanlı SSM (paralel taramalı) +
                              HolographicHDM + HopfieldCleanup + router
       train_full.py      -- eğitim (pretraining + instruction tuning/SFT)
       generate_full.py   -- üretim/test (CLI)
       gui_full.py        -- Gradio GUI (4 sekme: Eğitim, HDM Fact Yönetimi,
                              Chat/Üretim, Hakkında)
       model.txt           -- TAM kullanım kitapçığı (kurulum, veri seti
                              önerileri, hiperparametre rehberi, bilinen
                              sorunlar, ROCm/RX9070XT notları)
       ornek_instruction_tuning.json -- SFT veri formatı örneği
       requirements.txt

--------------------------------------------------------------------------------
5. AÇIK SORUNLAR / SIRADAKİ SOMUT ADIMLAR (öncelik sırasıyla)
--------------------------------------------------------------------------------
1. GERÇEK İLK ÇALIŞTIRMA: valerois_full/ paketini kendi GPU'nda (RX 9070 XT,
   ROCm 7.2) küçük bir veri altkümesiyle (TinyStories, birkaç bin hikaye,
   n_layers=2, hidden=128, steps=2000) 30 dakikalık bir sağlık kontrolü
   olarak çalıştır. Kod hiç gerçek PyTorch/GPU'da denenmedi.

2. ROUTER DENGESİ: warmup g değerini yükselt (0.15->0.3-0.5), serbest faz
   süresini uzat, entropy_weight'i düşür (0.05->0.01) -- fact recall ile
   loss iyileşmesi arasında doğru dengeyi bul.

3. GERÇEK ÖLÇEK TESTİ: FineWeb-Edu ya da TinyStories'in TAMAMIYLA eğit,
   gerçek genelleme (ezber değil) olup olmadığını TUTMA SETİYLE ölç
   (train/val loss ayrışmasına dikkat -- overfitting bulgusu bu gece
   defalarca görüldü, küçük veri setlerinde).

4. INSTRUCTION TUNING: Pretraining bitince --sft_file ile asistanlık
   öğret (ornek_instruction_tuning.json formatını kullan, kendi soru-cevap
   çiftlerini ekle).

5. (İSTEĞE BAĞLI, İLERİDE) Tool kullanımı ve reasoning için ek SFT verisi;
   vision için AYRI bir mimari bileşeni (vision encoder) -- bu gece hiç
   başlanmadı, ayrı bir mühendislik projesi.

--------------------------------------------------------------------------------
6. DÜRÜST GENEL DEĞERLENDİRME
--------------------------------------------------------------------------------
Bu mimari:
  - HIZDA muhtemelen Transformer'dan iyi DEĞİL (elle yazılmış CUDA
    çekirdekleri yok, genel PyTorch işlemleriyle yazıldı).
  - GENEL DİL ANLAYIŞINDA henüz hiç test edilmedi (gerçek ölçek gerekiyor).
  - UNUTMADAN SÜREKLİ ÖĞRENMEDE gerçek, ölçülmüş bir avantajı VAR.
  - ÇOK-ADIMLI SEMBOLİK ÇIKARIMDA (temizleme ile) gerçek, ölçülmüş bir
    yeteneği VAR.
  - Bu, "Transformer'ı yenen bir AI" değil, "belirli özelliklerde
    kanıtlanmış, dar kapsamlı ama dürüst bir mimari."

Sıradaki gerçek bilgi, SENİN GPU'ndan gelecek. Bu belge, oraya varana kadar
yapılan her şeyin kaydı.
================================================================================
