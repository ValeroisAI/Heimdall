# VALEROIS AI — KAPSAMLI PROJE, SİTE VE YÖNETİM PANELİ KILAVUZU

> **Tarih:** 15 Eylül 2026  
> **Geliştirici & Kurucu:** Valerois AI Research  
> **Masaüstü & Flash Bellek Yedek Sürümü:** v4.5 (Tam Entegre Paket)

---

## 1. PROJE ÖZETİ VE MİMARİ

Valerois AI, quadratik dikkat (quadratic self-attention) mekanizmasının yarattığı hafıza ve işlem darboğazlarını aşarak **sürekli nedensel durum konvolüsyonları (Continuous Causal State Convolutions - Bifrost CSL)** mimarisiyle çalışan, $O(1)$ sabit durum belleğine sahip ve sıfır KV-cache maliyetli yeni nesil yapay zeka temel modelleri geliştiren öncü bir araştırma girişimidir.

### Temel Bileşenler:
1. **Bifrost CSL Mimarisi:** Dizi uzunluğundan bağımsız sabit 8.8 MB hafıza tamponu ile çalışan lineer temel model ailesi.
2. **Valkir Model Ailesi:** Valkir 16L, Valkir 140M ve Valkir 1.5B modelleri.
3. **Pulse Desktop Studio:** Yerel ROCm / DirectML çıkarım ve benchmark arayüzü.
4. **Valerois Web Platformu:** Next.js 16, React 19, Tailwind CSS ve Three.js destekli ultra modern vitrin platformu (`https://valerois.com`).
5. **Masaüstü Yönetim Paneli (Admin Suite):** Python CustomTkinter ile uzaktaki cPanel REST API'yi yöneten, modelleri, uygulamaları ve kıyaslama matrislerini anlık güncelleyen bağımsız yönetim arayüzü.

---

## 2. PAKET KLASÖR YAPISI

Flash belleğe atılan bu tam paketin iç yapısı şu şekildedir:

```
VALEROIS_TAM_PROJE_YEDEGI/
│
├── 1_YONETIM_PANELI_ADMIN/
│   ├── Valerois_Admin.bat           # Çift tıklandığında paneli açan Windows başlatıcı
│   ├── valerois_admin_desktop.py    # CustomTkinter GUI yönetim panelinin Python kaynak kodu
│   └── requirements.txt             # Gerekli kütüphaneler (customtkinter, requests)
│
├── 2_CPANEL_CANLI_SITE_YAYINLAMA/
│   ├── valerois_cpanel_public_html.zip # cPanel Dosya Yöneticisine doğrudan yüklenecek zip
│   └── public_html/                 # Zip'in açık hali (tüm html, css, js ve php dosyaları)
│       ├── index.html               # Ana sayfa (3D Hero, mimari, telemetri)
│       ├── models.html              # Valkir modelleri listesi ve kartları
│       ├── apps.html                # Pulse Studio ve ekosistem araçları
│       ├── benchmarks.html          # İnteraktif kıyaslama matrisi (HumanEval, ARC, vb.)
│       ├── research.html            # Teknik araştırma makalesi ve matematiksel detaylar
│       ├── about.html               # Misyon, vizyon ve kurucu ekip
│       ├── contact.html             # İletişim ve akademik iş birliği formu
│       ├── valerois.css             # Kök düzey failsafe ana stil dosyası
│       ├── .htaccess                # Apache / LiteSpeed URL yönlendirme ve güvenlik
│       ├── sitemap.xml              # Google arama motoru site haritası
│       ├── robots.txt               # Googlebot tarama yönergeleri
│       ├── favicon.png              # 192x192 Google SERP ikonu
│       ├── apple-touch-icon.png     # iOS / Safari ana ekran ikonu
│       ├── og-banner.png            # 1200x630 Sosyal medya önizleme bannerı
│       ├── assets/                  # Yüksek çözünürlüklü logolar (logo.png, logobg.png)
│       └── api/                     # PHP REST API Backend
│           ├── config.php           # Veritabanı ve güvenlik anahtarları ayarları
│           ├── data.php             # Kamuya açık salt-okunur veri API'si
│           ├── admin.php            # Yetkilendirilmiş ekleme/silme/düzenleme API'si
│           ├── upload.php           # Dosya ve model ağırlığı yükleme API'si
│           ├── contact.php          # İletişim formu rölesi (sabit gönderen adı + Reply-To + inbox)
│           ├── .htaccess            # .json/.sql dosyalarına doğrudan web erişimini kapatır
│           ├── messages.json        # Gelen kutusu (her mesaj ayrı kayıt, panelden okunur)
│           ├── valerois_database.sql # phpMyAdmin MySQL şeması
│           └── valerois_data.default.json # İlk kurulum varsayılan veri şablonu
│
├── 3_SITE_KAYNAK_KODLARI_NEXTJS/
│   └── valerois-site/               # Next.js 16 kaynak kodları
│       ├── src/                     # React bileşenleri, layout ve sayfalar
│       ├── public/                  # Statik varlıklar ve görseller
│       ├── package.json             # NPM bağımlılıkları
│       └── next.config.ts           # Next.js yapılandırması
│
├── README_PROJE_KULLANIM_KILAVUZU.md # Bu kullanım kılavuzu
├── CHAT_LOG_VE_GELISIM_GUNCELLEMELERI.md # Tüm geliştirme süreci ve çözülen hatalar
└── CHAT_TRANSCRIPT_OZETI.txt        # Detaylı konuşma günlüğü
```

---

## 3. MASAÜSTÜ YÖNETİM PANELİ KULLANIMI

### Nasıl Başlatılır?
1. `1_YONETIM_PANELI_ADMIN` klasörüne girin.
2. `Valerois_Admin.bat` dosyasına çift tıklayın.
3. Batch dosyası sistemdeki Python ortamını kontrol eder; eğer `customtkinter` veya `requests` eksikse otomatik olarak kurar ve yönetim penceresini açar.

### Başka Bir Bilgisayarda / Stajda Çalıştırma:
Eğer Python yüklü olmayan bir bilgisayarda açacaksanız:
1. `python.org` üzerinden Python 3.10, 3.11 veya 3.12 kurun ("Add Python to PATH" seçeneğini işaretlemeyi unutmayın).
2. `Valerois_Admin.bat` dosyasına tıklayın; paneli anında başlatacaktır.

### Yönetim Paneli Özellikleri:
* **Canlı Sunucu Bağlantısı:** Sunucu adresi varsayılan olarak `https://valerois.com` olarak gelir.
* **Gizli Anahtar (Secret Key):** `<ADMIN_SECRET_KEY>` (Ayrıca `<ADMIN_PASSWORD>` parolasını da kabul eder).
* **Uygulama Yönetimi (Apps Tab):** Yeni uygulama ekleme, silme, indirme bağlantısı ve durum (Canlı, Beta, Geliştirme) güncelleme.
* **Model Yönetimi (Models Tab):** Parametre sayıları (M, B, T kısaltma butonlarıyla), mimari katmanları, durum ve indirme linkleri.
* **Kıyaslama ve Matris Yönetimi (Benchmarks Tab):** Yeni benchmark kategorisi/testi oluşturma, herhangi bir modele ait skorları tablodan çift tıklayarak anında düzenleme. Boş modellerde otomatik `~` işareti gösterilir.
* **Canlı Senkronizasyon:** Yapılan tüm değişiklikler anında `valerois.com/api/admin.php` üzerinden sunucuya iletilir ve web sitesinde canlıya geçer.

---

## 4. CPANEL WEB SİTESİNİ YAYINLAMA / GÜNCELLEME REHBERİ

Web sitesini sunucuda güncellemek istediğinizde izlenecek en hızlı yol:

1. **cPanel'e Giriş Yapın:** Hosting firmanızın cPanel adresine gidin.
2. **Dosya Yöneticisini Açın:** "Dosya Yöneticisi" (File Manager) simgesine tıklayın ve `public_html` klasörüne girin.
3. **Zip'i Yükleyin:** `2_CPANEL_CANLI_SITE_YAYINLAMA` klasöründeki `valerois_cpanel_public_html.zip` dosyasını `public_html` içine yükleyin.
4. **Çıkartın (Extract):** Yüklenen zip dosyasına sağ tıklayıp **Extract** (Çıkart) deyin.
5. **Tebrikler:** Siteniz en güncel kodlar, 3D Hero, SEO ayarları ve API ile yayına girmiştir.

### Verilerimin Silinmeme Garantisi:
Paket içindeki API mimarisi `valerois_data.default.json` kullanacak şekilde tasarlanmıştır. Canlı sunucudaki silinen/eklenen modellerinizi tutan `api/valerois_data.json` dosyası zip paketine bilerek dahil edilmemiştir. Böylece **yeni bir zip yükleyip açtığınızda panelden sildiğiniz hiçbir şey geri gelmez, verileriniz asla ezilmez**.

---

## 5. SSL VE WWW YÖNLENDİRME AYARI

### Neden `www` Sorun Yaratıyordu?
Sunucudaki Let's Encrypt SSL sertifikası sadece `valerois.com` için oluşturulmuş, `www.valerois.com` sertifikaya dahil edilmemişti. 

### Çözüm:
1. `.htaccess` dosyamıza otomatik 301 yönlendirmesi eklendi. Biri `www.valerois.com` yazsa bile sunucu onu doğrudan güvenli `https://valerois.com` adresine yönlendirir.
2. Doğrudan `www` için de yeşil kilit almak isterseniz:
   * cPanel -> **"SSL/TLS Durumu" (SSL/TLS Status)** bölümüne gidin.
   * `valerois.com` ve `www.valerois.com` kutularını işaretleyin.
   * Üstteki mavi **"AutoSSL'i Çalıştır" (Run AutoSSL)** butonuna basın. 1-2 dakikada her iki adres için de kilit yeşile döner.

---

## 6. GOOGLE SEO VE BİLGİ PANELİ (KNOWLEDGE PANEL)

Google arama motorunda `Valerois` arandığında sitenin dünya ikonu yerine kendi logosuyla çıkması ve sağ tarafta kafe yerine "Valerois Neural Intelligence" bilgi paneli oluşması için pakete şu standartlar eklendi:

1. **Favicon & Mobil Simgeler:** 192x192 `favicon.png`, 180x180 `apple-touch-icon.png` ve çoklu çözünürlüklü `favicon.ico`.
2. **Schema.org JSON-LD:** Sitenin kaynak koduna resmi `Organization` şeması gömüldü:
   * **İsim:** Valerois Neural Intelligence
   * **Logo:** `https://www.valerois.com/logo.png`
   * **Açıklama:** Pulse v1 ve Bifrost CSL modellerini geliştiren yapay zeka araştırma organizasyonu.
   * **Çapraz Bağlantılar:** GitHub, X ve LinkedIn profilleri.
3. **Sitemap & Robots:** `https://www.valerois.com/sitemap.xml` ve `robots.txt` tüm robotlara açık şekilde hazırlandı.

### Google İndeksini Hızlandırmak İçin:
* [Google Search Console](https://search.google.com/search-console) paneline girin.
* Üstteki arama kutusuna `https://www.valerois.com` yazıp **"URL Denetimi"** yapın.
* **"Dizine Eklenmesini İste" (Request Indexing)** butonuna basın. Botlar 24-48 saat içinde yeni logoyu ve şemayı işleyecektir.

---

## 7. YEREL GELİŞTİRME (LOCAL DEV) REHBERİ

Sitede yeni bir sayfa veya kod değişikliği yapmak isterseniz:
1. `3_SITE_KAYNAK_KODLARI_NEXTJS/valerois-site` klasörünü açın.
2. Terminalde komutları çalıştırın:
   ```bash
   npm install        # Bağımlılıkları kurar
   npm run dev        # http://localhost:3000 üzerinde geliştirme sunucusunu açar
   ```
3. Yeni cPanel paketi derlemek istediğinizde:
   ```bash
   python build_and_package.py
   ```
   Bu komut projeyi otomatik olarak statik HTML'e derler, failsafe CSS'i köke kopyalar, `_next` ve `next` klasörlerini eşitler ve masaüstüne yüklemeye hazır `valerois_cpanel_public_html.zip` oluşturur.

---

## 8. GÜVENLİK VE ŞİFRE BİLGİLERİ

| Parametre | Değer |
| :--- | :--- |
| **Admin Secret Key (Header)** | `<ADMIN_SECRET_KEY>` |
| **Yedek Admin Parolası** | `<ADMIN_PASSWORD>` |
| **Canlı API Uç Noktası** | `https://valerois.com/api/data.php` |
| **Canlı Admin Uç Noktası** | `https://valerois.com/api/admin.php` |
| **Varsayılan MySQL DB Adı** | `valerois_db` *(Özelleştirilebilir: `api/config.php`)* |
| **Varsayılan MySQL Kullanıcı** | `valerois_user` *(Özelleştirilebilir: `api/config.php`)* |

---
*Bu paket, Valerois AI platformunun tüm üretim, geliştirme ve yönetim varlıklarını bağımsız bir flash bellekten çalışabilecek şekilde eksiksiz barındırmaktadır.*
