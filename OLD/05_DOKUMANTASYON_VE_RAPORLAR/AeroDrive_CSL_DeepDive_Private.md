# AeroDrive CSL: Derinlemesine Mimari ve Matematiksel Analizi (Gizli Doküman)

Bu doküman, AeroDrive CSL (Continuous State Learning) mimarisinin arka planda çalışan matematiksel hilelerini, PyTorch / DirectML optimizasyon sırlarını ve Receptive-Field (Görüş Alanı) Bound Training teorisini derinlemesine açıklar.

## 1. Mimarinin Kalbi: `AeroDriveLayer`

Transformer'larda sequence boyutundaki her token, diğer tüm tokenlarla çarpılır ($O(N^2)$ karmaşıklık).
AeroDrive ise zamanı sadece bir yöne (ileriye) akan bir **Shift (Kaydırma)** ve **Gating (Filtreleme)** problemi olarak görür.

### Neden `groups=hidden` (Depthwise Convolution)?
DirectML ve PyTorch üzerinde standart konvolüsyonlar (örneğin 1536 kanaldan 1536 kanala çapraz çarpım) $O(C^2)$ donanım yükü yaratır. Bu yüzden eski denemelerde hızımız 4,000 byte/s seviyesine düşmüştü.
Biz kanalları kendi içine hapsettik: `groups=hidden` parametresiyle, 512 kanalın her biri *sadece kendi geçmişiyle* çarpılır. Bu işlem DirectML üzerinde **0.13 saniyeye** düşerek tam **40 Kat** hızlanma sağladı.

### FFN ve Gating (SwiGLU-benzeri)
Konvolüsyon zaman boyutunda (zamanda geriye) bilgi toplar. Ancak farklı kanallardaki bilgilerin birbirine karışması gerekir. Bunun için `up_proj` (Genişletme) kullanılır:
1. $H_{conv}$ (Zaman bilgisi) ile mevcut state toplanır (Residual).
2. `up_proj` ile boyut 1536'ya çıkarılır ve ikiye bölünür ($U$ ve $Gate$).
3. $Out = U * \text{SiLU}(Gate)$ formülüyle element-bazlı kapılama (gating) yapılır.
4. `down_proj` ile tekrar 512'ye sıkıştırılır. 

Bu yapı, Mamba (SSM) mimarisinin kalbindeki donanımsal darboğazları atlayarak sadece Matrix Çarpımı (MatMul) kullanan çok daha hızlı bir tasarımdır.

## 2. Receptive-Field Bound Training Teorisi

Mimarimizin en can alıcı sırrı eğitim döngüsüdür.

### Neden 512 Context Yerine 128 Context?
AeroDriveLayer'daki konvolüsyonun `kernel_size = 16`'dır. Modelin katmanları boyunca Dilation (Yayılım) oranları $1, 2, 4, 8$ olarak artar. 
Bir tokenin görebileceği maksimum "geçmiş" token sayısı (Receptive Field) formülü şudur:
$$ R = \sum (\text{kernel\_size} - 1) \times \text{dilation} $$
Bizim 6 katmanlı modelimizde bu toplam tam olarak **100-128** token aralığına denk gelir!
Yani model, matematiksel olarak zaten 128 tokenden daha öncesini **göremez**.

**Hile Burada Başlar:**
Madem model 128 tokenden öncesini göremiyor, neden PyTorch'a 512 tokenlik veriler verip Geriye Yayılım (Backpropagation) grafiğini 4 kat uzatıyoruz?
İşte bu yüzden `train_byte.py` içinde `seq_len = 128` yaptık. GPU'nun üzerindeki yük %75 oranında silindi. Hız 30,000'den **60,000 Token/Saniye** seviyesine fırladı.

## 3. FP16 NaN (Patlama) Koruması: Depth Scaling

`fp16` eğitiminde residual bağlantıları toplanarak büyür. Örneğin 6 katman sonunda varyans (sapma) çok yükselir ve değerler `65504` sınırını aşarak `NaN` (Not a Number) hatası verir.
Bunu çözmek için `_apply_depth_scaling` metodunu yazdık:
$$ \text{std} = \frac{0.02}{\sqrt{2 \times \text{katman\_sayisi}}} $$
Bu ufak matematiksel dokunuş, `down_proj` ağırlıklarını modelin derinliğine göre küçültür. Böylece varyans her katmanda stabil kalır ve `fp16` eğitiminde sıfır hata ile muazzam hıza ulaşılır.

## 4. BPE Streaming Decode Çözümü

Türkçe gibi dillerde UTF-8 karakterleri (örneğin 'ğ') birden fazla byte/token ile ifade edilir. `generate_byte.py` içinde her bir tokeni tek tek ekrana basmaya çalıştığımızda karakter tam oluşmadığı için `` işareti çıkıyordu.
**Çözüm:** Tüm üretilen metni bir `buffer` (havuz) içinde tutup sadece kusursuzca çözülmüş (decode edilmiş) olan *yeni* string parçasını ekrana basacak şekilde `printed_len` izleme algoritmasını entegre ettik.

---
Bu mimariyle artık 5 dakikada ajan (SFT) eğitebilir veya milyarlarca tokenlık veriyi saatler içinde işleyebilirsiniz. Kuralları biz koyduk!
