# Kuyu — "yapay zeka ama yapay zeka değil"

> Mímir'in kuyusu: Odin bilgelik için gözünü verdi. Bu kuyu, bilgiyi içine yazdığın anda öğrenir.

Bir model değil, **yaşayan bir hafıza**. Gradyan eğitimi yok, metin üretimi yok. VRAM'de çalışır.

| "Yapay zeka" deyince akla gelen | Kuyu'da |
|---|---|
| Bir kez devasa eğitilir, bilgisi donar | **Öğrenmek = yazmak.** Bilgi çalışırken tek adımda hafızaya yazılır (delta kuralı). Yeniden eğitim yok, unutma yok |
| Metin üretir, uydurur | **Metin değil, karar.** Tipli değer + kalibre güven. Emin değilse "bilmiyorum" |
| Her girdi için aynı dev hesap | **Sürpriz kadar hesap.** Zaten bilinen bilgi hafızayı değiştirmez; yazma atlanır |
| Kara kutu | **Açık hafıza.** Tek satırla güncelle (`learn`) ya da sil (`forget`); sözlüğü okunabilir |

## Nasıl çalışır
1. **Anahtar (öğrenmesiz):** Metindeki kelimelerin **sırasız çiftleri** sabit rastgele vektörlerle bağlanır (HRR tarzı ⊙).
   - Hafızanın hiç görmediği kelimeler ("için", "nedir", "what is the") atılır.
   - Tanımadığı kelime kendi sözlüğündeki en yakın kelimeye düzeltilir ("prt" → "port").
   - Bu sayede "proje_X port", "proje_X için port değeri nedir" ve "what is the port of proje_X" aynı anahtara iner.
2. **Raf (LSH):** Anahtar benzer anahtarların rafına gider. VRAM'de `raf × d_k × d_v` hafıza.
3. **Yazma:** `S ← S + β k (v − Sᵀk)ᵀ`. Yalnız sürpriz yazılır. Aynı anahtar gelince eski değer silinir, yani "son değer" semantiği kendiliğinden oluşur. Raflar paralel, raf içi chunk-paralel yazılır (GPU dostu).
4. **Okuma:** `v̂ = Sᵀk`, en yakın değer kodu seçilir. Güven üç sinyalden hesaplanır: benzerlik, iki aday arasındaki fark ve **sorunun ayırt edici bir parçası var mı?**. Güven düşükse cevap "bilmiyorum" olur.

## Sonuçlar (CPU, 4 çekirdek, 100 bin bilgi, 1024 raf, 512 MB)

`python deney.py --varlik 20000 --raf 1024` → `sonuc_cpu_100k_1024raf.json`

| | Kuyu | Klasik ağ, tek geçiş | Klasik ağ, 10 tur | Toplamsal (Hebbian) hafıza |
|---|---|---|---|---|
| Öğrenme süresi | **3.9 s** (tek geçiş) | 6 s | 60 s | — |
| Tüm bilgileri hatırlama | **%96.2** | %29.7 | %48.1 | %72.2 |
| En eski %10 | **%94.8** | %20.1 | %48.0 | — |
| Güncellenen → son değer | **%97.4** | — | — | %44.0 |
| Türkçe / İngilizce soru, yazım hatası | **%93.4** | — | — | — |
| Silinen → "bilmiyorum" | **%94.4** | — | — | — |
| Hiç görülmemiş → "bilmiyorum" | **%100** | — | — | — |
| Kalibrasyon hatası (ECE) | 0.021 | — | — | — |

- Klasik ağ aynı anahtarları girdi alan bir sınıflandırıcı, Adam ile eğitildi.
- "Bilmiyorum" ve kalibrasyon oranları, güven hesabının uydurulmasında kullanılmayan ayrı yarıda ölçüldü.
- **Kapasite ≈ raf × d_k.** 256 rafta (128 MB) 100 bin bilgi hatırlaması %84'e düşüyor; 20 bin bilgide ise %98.5.

## Çalıştırma
```bash
python deney.py                        # CPU (varsayılan 100 bin bilgi, 256 raf)
python deney.py --raf 1024             # daha fazla kapasite (512 MB)
python deney.py --device dml           # Windows + AMD GPU:  pip install torch-directml
python deney.py --device cuda          # Linux + ROCm (PyTorch ROCm'da 'cuda' adını kullanır)
```

Kendi kodunda kullanım:
```python
from kuyu import Kuyu
k = Kuyu(n_shelves=1024)
k.learn(["proje_17 port", "proje_17 sahip"], ["8080", "ayse"])   # öğrenmek = yazmak
k.learn(["proje_17 port"], ["9090"])                                 # güncelle: son değer kazanır
k.forget(["proje_17 sahip"])                                         # sil
idx, guven_ozellikleri = k.recall(["what is the port of proje_17"])  # → "9090"
```

## Sınırlar
- **Eş anlamlıları bilmez.** "port" ile "bağlantı noktası" aynı sayılmaz. Çözüm yine yazmak: takma adları hafızaya öğretmek. Ya da v1'de öğrenilmiş bir algı katmanı (Kuzgun kodlayıcısı).
- **Değerler bir şemadan gelir.** Cevap, bilinen tipli değerlerden biridir; serbest metin üretmez (bilinçli tercih).
- **Kapasite hafıza ile orantılı.** Daha çok bilgi için daha çok raf (VRAM) gerekir.

## Sıradaki adımlar
1. Takma ad ve eş anlamlıları yazarak öğretmek.
2. "Doğrula, sonra hatırla" döngüsü: yavaş yol (hesap, test, arama) doğrulanmış sonucu yazar; bir dahaki sefere tek bakışta cevap.
3. Öğrenilmiş algı: Kuzgun'un kodlayıcısı anahtar üretsin, bilgi yine yazmayla öğrenilsin.
