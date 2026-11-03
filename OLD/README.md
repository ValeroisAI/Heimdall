> **Legacy archive (English note).** This folder holds earlier, superseded research (Bifrost CSL, V-PDM, Kuzgun, Valkir, Kuyu) written in Turkish. The active project is [`Heimdall/`](../Heimdall/README.md); see the [root README](../README.md). Nothing here is referenced by the results reported there.

---

# 🏛️ VALEROIS MİMARİ VE KOD PAKETİ (USB LAB KİTİ)

Bu klasör, Valerois projesinin tüm kritik kodlarını, mimari katmanlarını, tokenizer'larını, **Bifrost CSL (Continuous Causal State Convolutions)** motorunu ve dün gece tartıştığımız yepyeni **V-PDM (Predictive Delta Memory)** mimarisini içerir.

> **NOT:** Checkpoint'ler (.pt, .vlrs, .safetensors) ve gigabaytlarca büyük binary veri setleri özellikle pakete **dahil edilmemiştir**. Paket saf, hafif ve taşınabilir kodlardan oluşur.

---

## 📁 Dizin Yapısı ve İçerik

### `01_KATMANLAR_VE_MIMARILER/`
* `valerois_gcam_v2.py`: 1M context O(1) hafıza katmanı (Chunked GEMM + Softmax + Linear State).
* `valerois_gcam_v3.py`: Seçici yazma kapılı (Selective GCAM) v3 katmanı.
* `delta_csl_layer.py`: Delta kapılı Convolüsyonel Sıralama Katmanı.
* `matrix_ssd_layer.py`: Durum Uzayı (State Space Duality) katmanı.
* `pure_csl_layer.py`: Saf Dilated Conv + SwiGLU bloğu.
* `gated_linear_attention.py`: Kapılı Doğrusal Dikkat katmanı.
* `resonant_fourier_csl.py`: Fourier rezonans hafıza katmanı.
* `radar_attention.py` & `vectorized_radar_attention.py`: Radar dikkat katmanları.

### `02_TURNUVALAR_VE_TESTLER/`
* `grand_architecture_tournament_9.py`: 9 farklı model ve hibrit katmanın kafa kafaya yarıştığı ana turnuva betiği.
* `grand_tournament_results.json`: C3 (Full Hybrid) ve B1'in kazandığı turnuva sonuç tablosu.
* `test_valerois_vloop_tournament.py` & `test_valerois_vloop_stabilized.py`: V-Loop (Yinelemeli Derinlik) testleri.
* `test_vloop_crossover_1000.py`: 1.000 adımlık derinlik yarışması.
* `test_valerois_gcam_needle.py`: Sentetik "Ceren" İğne-Samanlık (Needle-in-a-Haystack) testi.
* `test_valerois_multicode_needle.py`: 4 farklı kod tabanı (Python, C++, Rust, Go) içinde uzun bağlam testi.
* `test_1m_context.py` & `test_1m_selective_gcam.py`: 1.000.000 tokenlik akış ve VRAM testleri.

### `03_CEKIRDEK_MODELLER/`
* `valerois_core_model.py`: Valerois ana model çekirdeği.
* `valerois_coder_7b_core.py`: 7B parametreli Coder omurgası.
* `graft_valerois_production.py`: Üretim seviyesi katman nakli (Grafting).
* `pulse_engine.py`: Hafif çıkarım motoru.
* `valerois_byte_core.py`: Byte seviyesi dil modeli çekirdeği.
* `valerois_triad_27b.py`: 27B Triad MoE (Lingua, Coder, Thinker) mimarisi.
* `valerois_custom_quant_engine.py`: VQ4 ve BitPack-2 kuantizasyon motoru.
* `valerois_bitnet_triton_kernel.py`: 1.58-bit BitNet Triton çekirdeği.
* `valerois_muon_hybrid.py`: Muon optimizasyon motoru.

### `04_TOKENIZERLAR/`
* `valerois_tokenizer_8k.json` (8.192 kelimelik ana tokenizer)
* `valerois_tokenizer_32k.json` (32.768 kelimelik süper tokenizer)
* `valerois_unified_tokenizer_12k.json`
* `valerois_code_tokenizer_4k.json`
* `valerois_appended_code_tokenizer.json`

### `05_DOKUMANTASYON_VE_RAPORLAR/`
* `VALEROIS_ARCHITECTURE.md`: Transformer'a karşı bileşen bileşen mimari analiz.
* `PROJE_TAM_HAFIZA_VE_KOD_REHBERI.md`: Projenin geçmişi ve tam hafıza kapsülü.
* `VALEROIS_VS_TRANSFORMER_BOOKLET.md`: Kitapçık ve teknik karşılaştırmalar.
* `SON_TURNUVA_RAPORU_VLOOP.md`: V-Loop turnuvasının analiz raporu.
* `FINAL_REPORT_10AM.md`: 44 gece deneyinin resmi şampiyonluk raporu.

### `06_YENI_FIKIR_V_PDM/` ⭐ (LAB DERSİ İÇİN YENİ BÖLÜM)
* `V_PDM_ARCHITECTURE_SPEC.md`: **Valerois Predictive Delta Memory** mimari spesifikasyonu. Attention yerine sıfırdan çalışan, sadece sürprizleri ($\delta$) hafızaya kazıyan devrimsel katman.
* `v_pdm_starter_template.py`: Okuldaki laboratuvarda hemen açıp çalıştırabileceğin, hazır, temiz ve test edilebilir PyTorch şablonu (`python v_pdm_starter_template.py`).

### `07_BIFROST_CSL/` ⚡ (BİFROST & EXPONENTIAL CSL ÇEKİRDEĞİ)
* `valerois_exponential_csl.py`: 1M+ Native Receptive Field üstel dilatasyonlu CSL katmanı.
* `valerois_csl_qv5.py`: CSL-QV5 üç kanallı nedensel dizi katmanı (Depthwise Conv + Chunk Summary Cross Attention + SwiGLU).
* `test_csl_qv5.py` & `bench_csl_qv5_train.py`: CSL-QV5 test ve eğitim benchmarkları.
* `train_csl_mi300x.py`: MI300X ve ağır siklet hızlandırıcı CSL eğitim betiği.
* `triton_csl_fused.py`: ROCm Triton Fused RMSNorm-Swish hızlandırma çekirdeği.
* `run_autonomous_csl_lab.py`: Otonom CSL laboratuvar motoru.
* `AeroDrive_CSL_DeepDive_Private.md` & `AeroDrive_CSL_Public.md`: CSL matematiksel ve Receptive-Field derin analiz dokümanları.
* `BIFROST_CSL_RESMI_KULLANIM_KILAVUZU.md`: Bifrost CSL mimarisinin resmi kullanım kılavuzu ve teknik detayları.
