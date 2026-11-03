#!/usr/bin/env bash
# Qwen3.8-27B'nin yalnız 16 tam dikkat katmanını sabit bellekli katmana çevirir (DeltaNet/FFN aynen kalır).
# RX 9070 XT 16 GB için ayarlı. Adımlar ADIM ile seçilir; sırayla:
#
#   ADIM=prova bash bench/qwen38_donustur.sh   # ÖNCE BU: Qwen3.5-0.8B (aynı mimari) ile ~10-20 dk uçtan uca prova
#   ADIM=plan    bash bench/qwen38_donustur.sh   # bellek özeti (indirmeden)
#   ADIM=veri    bash bench/qwen38_donustur.sh   # model (~56 GB bf16) + wikitext-103 tokenize
#   ADIM=a       bash bench/qwen38_donustur.sh   # aşama A: katman katman (bf16 öğretmen, ~6-8 GB GPU)
#   ADIM=test    bash bench/qwen38_donustur.sh   # perplexity + parola testi (öğretmen ↔ dönüşüm, bellek)
#   ADIM=b       bash bench/qwen38_donustur.sh   # aşama B (isteğe bağlı): 4-bit model, uçtan uca KL
#   ADIM=sohbet  bash bench/qwen38_donustur.sh   # terminalde sohbet (MTP ile)
#   ADIM=gguf    bash bench/qwen38_donustur.sh   # tek dosya .gguf (q8_0, ~29 GB; şimdilik yalnız bizim çalışma zamanımızda açılır)
#   ADIM=gece    bash bench/qwen38_donustur.sh   # gece boyu: a → test → b → test (log: $OUT/gece.log)
#
#   llama.cpp (Bifrost yamalı, GPU'da hızlı çıkarım):
#   ADIM=llama        bash bench/qwen38_donustur.sh   # kurulum + derleme (~10-20 dk)
#   ADIM=llamaprova   bash bench/qwen38_donustur.sh   # 0.8B prova: llama.cpp GPU ↔ PyTorch birebir mi
#   ADIM=llamagguf    bash bench/qwen38_donustur.sh   # 27B: q8_0 GGUF (~29 GB) → $QUANT (~14 GB)
#   ADIM=llamasohbet  bash bench/qwen38_donustur.sh   # llama-cli ile sohbet (bağlam $CTX)
#   ADIM=llamabench   bash bench/qwen38_donustur.sh   # hız: boş bağlam ↔ 32K dolu bağlam (sabit kalmalı)
#   ADIM=llamaprofil  bash bench/qwen38_donustur.sh   # Vulkan op süreleri (ORIJ=orijinal.gguf ile karşılaştırma)
#   ADIM=llamakeep    bash bench/qwen38_donustur.sh   # 3 katmanı daha tam dikkate döndür (KEEP4=15,31,47), hız + PPL karşılaştır
#   ADIM=llamaparola  bash bench/qwen38_donustur.sh   # parola testi 8K/32K/64K: 15/1, 12/4 (hafıza açık/kapalı), orijinal
#
#   Hafıza v2 (yarı ömür pencere..1M, okuma ölçeği sqrt(dh), pencere dışı kayıp, 4×pencere dizi + %25 sentetik hatırlama):
#   ADIM=provahafiza  bash bench/qwen38_donustur.sh   # 0.8B: v1 ↔ v2 ↔ v2 hafızasız ↔ orijinal (parola + PPL), ~30-60 dk
#   ADIM=a2           bash bench/qwen38_donustur.sh   # 27B aşama A v2 → $OUT2 (varsayılan runs/q38_h2), KEEP2 global kalır
#   ADIM=a2test       bash bench/qwen38_donustur.sh   # a2 sonrası: GGUF + parola + PPL + hız
#   ADIM=a2parola     bash bench/qwen38_donustur.sh   # parola: v2 ↔ v2 hafızasız ↔ orijinal (ORIJ=...), sınırsız cevap
#
#   Eğitim motoru (heimdall/motor.py: uzun dizi, öğretmen bir kez, parça parça tam gradyan):
#   ADIM=motorprova   bash bench/qwen38_donustur.sh   # 0.8B, 16K dizi, hepsi dönüşür: parola (hafıza açık/kapalı) + PPL
#   ADIM=motor        bash bench/qwen38_donustur.sh   # 27B → $OUT3 (runs/q38_m1), 64K dizi, 12/4; MOTOR_ARGS ile ayar
#   ADIM=motortest    bash bench/qwen38_donustur.sh   # motor sonrası: GGUF + parola (açık/kapalı) + PPL + hız
#   ADIM=erisim       bash bench/qwen38_donustur.sh   # erişimli dikkat (BIFROST_RET) kapalı ↔ açık: parola + hız (ERISIM_M=model)
#   ADIM=sunucu       bash bench/qwen38_donustur.sh   # Valerois Runtime: http://127.0.0.1:1234/v1 (Nova → Ayarlar → Modeller → Local)
#
# Ortam: MODEL, OUT, VERI, PY, PENCERE (1024), KALIB (64), SEQ (4096), ADIMA (400), ADIMB (200)
#        DEVAM=1: aşama A yarıda kaldıysa eğitilmiş katmanları atla (yalnız ileri geçiş)
#        KEEP="layers.63": o tam dikkat katmanı dönüştürülmez (birebir uzun-menzil hatırlama korunur;
#        cache token başına 4 KB büyür = orijinalin 1/16'sı, 262K'da 1 GB)
# İpucu: monitörü anakarttaki (iGPU) çıkışa takarsan masaüstü 9070 XT'den ~0.5-1 GB yemez.
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/.."
# sanal ortam açık değilse Heimdall/.venv101'i kendisi açar
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f .venv101/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv101/bin/activate
fi
ADIM=${ADIM:-plan}
PY=${PY:-python}
MODEL=${MODEL:-Qwen/Qwen3.8-27B}
OUT=${OUT:-runs/q38}
VERI=${VERI:-veri/qwen_wiki}
PENCERE=${PENCERE:-1024}
KALIB=${KALIB:-64}
SEQ=${SEQ:-4096}
ADIMA=${ADIMA:-400}
ADIMB=${ADIMB:-200}
KEEP=${KEEP:-}
LLAMA=${LLAMA:-$HOME/llama.cpp-bifrost}
QUANT=${QUANT:-IQ4_XS}
CTX=${CTX:-131072}
DERIN=${DERIN:-0,32768}      # llamabench bağlam dolulukları (ör. 0,32768,65536,131072)
GGUF_TIP=${GGUF_TIP:-q8_0}
OUT2=${OUT2:-runs/q38_h2}
OUT3=${OUT3:-runs/q38_m1}
KEEP2=${KEEP2:-layers.15 layers.31 layers.47 layers.63}   # 12/4: 4 tam dikkat katmanı kalır
pk() {   # parola testi: model, bağlamlar
  [ -x $LLAMA/build/bin/bifrost_passkey ] && [ llamacpp/bifrost_passkey.cpp -ot $LLAMA/build/bin/bifrost_passkey ] || \
    g++ -O2 -std=c++17 llamacpp/bifrost_passkey.cpp -I$LLAMA/include -I$LLAMA/ggml/include -L$LLAMA/build/bin \
      -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o $LLAMA/build/bin/bifrost_passkey
  echo "== $1"
  NGL=99 CTK=${CTK:-q8_0} CTV=${CTV:-q8_0} $LLAMA/build/bin/bifrost_passkey $1 $2 ${PAROLA_D:-0.1,0.5,0.9} ${PAROLA_N:-2} \
    2>/dev/null | grep -E "yanlış|DOĞRU|doğruluk" || true
}   # bf16 en iyisi ama 27B'de ~110 GB boş disk ister; q8_0 ~60 GB
hf_dir() { $PY -c "from heimdall.donustur import fetch_model; print(fetch_model('$1'))"; }
H="$PY -m heimdall.donustur_hibrit"
command -v "$PY" >/dev/null || { echo "python bulunamadı: önce 'source .venv101/bin/activate' (ya da PY=python3)"; exit 1; }

if [[ $ADIM == prova ]]; then
  M=Qwen/Qwen3.5-0.8B; O=runs/q35_08b_prova; V=veri/qwen_wiki_prova
  $H hazirla --model $M --veri $V --tokens 5e6
  $H asama-a --model $M --train $V/wiki_train.bin --out $O --window 256 --kalib 32 --seq-len 2048 --steps 150
  $H degerlendir --model $M --out $O --test $V/wiki_test.bin --eval-len 4096 --n 4 --sifre 1024 4096 16384 --deneme 2
  $PY -m heimdall.gguf_cevir bifrost --model $M --conv $O/donusum_param.pt --out $O/q35-08b-bifrost-q8_0.gguf --tip q8_0
  $PY -m heimdall.gguf_cevir dogrula --gguf $O/q35-08b-bifrost-q8_0.gguf --model $M --conv $O/donusum_param.pt
  echo "prova tamam. 'dönüşüm' satırı 'yalnız pencere'den iyi, 'öğretmen'e yakınsa 27B'ye geç: ADIM=plan"
  exit 0
fi

case $ADIM in
  plan)   $H plan --model $MODEL --window $PENCERE ;;
  veri)   $H hazirla --model $MODEL --veri $VERI --tokens 20e6 ;;
  a)      $H asama-a --model $MODEL --train $VERI/wiki_train.bin --out $OUT --window $PENCERE --kalib $KALIB \
            --seq-len $SEQ --steps $ADIMA ${KEEP:+--keep-global $KEEP} ${DEVAM:+--devam} ;;
  test)   $H degerlendir --model $MODEL --out $OUT --test $VERI/wiki_test.bin --eval-len 8192 --n 4 \
            --sifre 4096 16384 65536 --deneme 2 ;;
  b)      $H asama-b --model $MODEL --out $OUT --seq-len 2048 --steps $ADIMB ;;
  sohbet) $H sohbet --model $MODEL --out $OUT --mtp ;;
  gguf)   $PY -m heimdall.gguf_cevir bifrost --model $MODEL --conv $OUT/donusum_param.pt --out $OUT/q38-27b-bifrost-q8_0.gguf \
            --tip q8_0 ;;
  llama)  DIR=$LLAMA bash llamacpp/kur.sh ;;
  llamaprova)
          M=$(hf_dir Qwen/Qwen3.5-0.8B); O=runs/q35_08b_prova
          $PY llamacpp/bifrost_gguf.py --model "$M" --conv $O/donusum_param.pt --llama $LLAMA --outtype f32 \
            --out $O/q35-08b-bifrost-f32.gguf
          NGL=99 $PY llamacpp/karsilastir.py --gguf $O/q35-08b-bifrost-f32.gguf --model "$M" --conv $O/donusum_param.pt \
            --bin $LLAMA/build/bin --n 400 --parcalar 200,1,1
          NGL=99 $LLAMA/build/bin/bifrost_multiseq $O/q35-08b-bifrost-f32.gguf 2>/dev/null ;;
  llamagguf)
          $PY llamacpp/bifrost_gguf.py --model "$(hf_dir $MODEL)" --conv $OUT/donusum_param.pt --llama $LLAMA \
            --outtype $GGUF_TIP --out $OUT/q38-27b-bifrost-$GGUF_TIP.gguf
          $LLAMA/build/bin/llama-quantize $([ $GGUF_TIP = q8_0 ] && echo --allow-requantize) \
            $OUT/q38-27b-bifrost-$GGUF_TIP.gguf $OUT/q38-27b-bifrost-$QUANT.gguf $QUANT
          echo "hazır: $OUT/q38-27b-bifrost-$QUANT.gguf ($GGUF_TIP ara dosyası silinebilir)" ;;
  llamasohbet)
          $LLAMA/build/bin/llama-cli -m $OUT/q38-27b-bifrost-$QUANT.gguf -ngl 99 -c $CTX ;;
  llamabench)
          $LLAMA/build/bin/llama-bench -m $OUT/q38-27b-bifrost-$QUANT.gguf -ngl 99 -p 512 -n 64 -d $DERIN ;;
  llamaprofil)   # Vulkan op süreleri (decode): en pahalı 40 op; ORIJ=<orijinal gguf> verilirse onu da ölçer
          for M in $OUT/q38-27b-bifrost-$QUANT.gguf ${ORIJ:-}; do
            echo "== $M"
            GGML_VK_PERF_LOGGER=1 $LLAMA/build/bin/llama-bench -m "$M" -ngl 99 -p 0 -n 8 -r 1 -d ${PROFIL_D:-0} 2> $OUT/vkperf.log >/dev/null
            tac $OUT/vkperf.log | sed '/Vulkan Timings/q' | tac | grep ' us = ' \
              | awk -F' us = ' '{split($2,a," "); printf "%10.1f us  %s\n", a[1], $1}' | sort -gr | head -40
            tac $OUT/vkperf.log | grep -m1 'Total time'
          done ;;
  llamakeep)   # yeniden eğitim yok: katmanlar bağımsız eğitildi, yalnız dönüşümleri kapatılır
          K=$OUT/q38-27b-bifrost-k4-$QUANT.gguf
          [ -f $K ] || $PY llamacpp/bifrost_keep.py --in $OUT/q38-27b-bifrost-$QUANT.gguf --out $K --keep ${KEEP4:-15,31,47} --llama $LLAMA
          for m in $OUT/q38-27b-bifrost-$QUANT.gguf $K ${ORIJ:-}; do
            echo "== $m"
            $LLAMA/build/bin/llama-perplexity -m $m -f veri/wiki_test.txt -ngl 99 -c 8192 --chunks 8 2>&1 | grep "Final estimate" || true
          done
          echo "== $K, KV q4_0"
          $LLAMA/build/bin/llama-perplexity -m $K -f veri/wiki_test.txt -ngl 99 -c 8192 --chunks 8 -fa on -ctk q4_0 -ctv q4_0 2>&1 | grep "Final estimate" || true
          for t in q8_0 q4_0; do   # orijinal 131K'da sığmazsa o satır hata verir, diğerleri yine ölçülür
            for m in $K ${ORIJ:-}; do
              $LLAMA/build/bin/llama-bench -m $m -ngl 99 -fa 1 -ctk $t -ctv $t -p 512 -n 64 -d ${DERIN:-0,32768,131072} 2>&1 | grep -E "^\| qwen" || true
            done
          done ;;
  llamaparola)   # teşhis: hafıza açık/kapalı × 15/1, 12/4, orijinal. Doğru cevap: ilk sayı = parola
          [ -x $LLAMA/build/bin/bifrost_passkey ] && [ llamacpp/bifrost_passkey.cpp -ot $LLAMA/build/bin/bifrost_passkey ] || \
            g++ -O2 -std=c++17 llamacpp/bifrost_passkey.cpp -I$LLAMA/include -I$LLAMA/ggml/include -L$LLAMA/build/bin \
              -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o $LLAMA/build/bin/bifrost_passkey
          $PY bench/bifrost_param_ozet.py $OUT/donusum_param.pt
          B=$OUT/q38-27b-bifrost-$QUANT.gguf; K=$OUT/q38-27b-bifrost-k4-$QUANT.gguf
          [ -f $K ] || $PY llamacpp/bifrost_keep.py --in $B --out $K --keep ${KEEP4:-15,31,47} --llama $LLAMA
          [ -f ${B%.gguf}-memoff.gguf ] || $PY llamacpp/bifrost_keep.py --in $B --out ${B%.gguf}-memoff.gguf --mem-off --llama $LLAMA
          [ -f ${K%.gguf}-memoff.gguf ] || $PY llamacpp/bifrost_keep.py --in $K --out ${K%.gguf}-memoff.gguf --mem-off --llama $LLAMA
          for m in $B ${B%.gguf}-memoff.gguf $K ${K%.gguf}-memoff.gguf ${ORIJ:-}; do
            echo "== $m"
            NGL=99 CTK=${CTK:-q8_0} CTV=${CTV:-q8_0} $LLAMA/build/bin/bifrost_passkey $m ${PAROLA_L:-8192,32768,65536} \
              ${PAROLA_D:-0.1,0.5,0.9} ${PAROLA_N:-1} 2>/dev/null | grep -E "yanlış|DOĞRU|doğruluk" || true
          done
          echo "== 131K neden düştü: 12/4, KV q4_0"
          $LLAMA/build/bin/llama-bench -m $K -ngl 99 -fa 1 -ctk q4_0 -ctv q4_0 -p 0 -n 16 -d 131072 2>&1 | tail -6 || true ;;
  provahafiza)
          M=Qwen/Qwen3.5-0.8B; HFM=$(hf_dir $M); V=veri/qwen_wiki_prova; O1=runs/q35_08b_prova; O2=runs/q35_08b_h2
          [ -f $V/wiki_train.bin ] || $H hazirla --model $M --veri $V --tokens 5e6
          [ -f $O2/donusum_param.pt ] || $H asama-a --model $M --train $V/wiki_train.bin --out $O2 --window 256 \
            --kalib 32 --seq-len 2048 --steps 300 --sentetik 0.25 --mem-init long
          $PY bench/bifrost_param_ozet.py $O2/donusum_param.pt
          [ -f runs/q35_08b_orijinal-f16.gguf ] || $PY $LLAMA/convert_hf_to_gguf.py $HFM --outtype f16 \
            --outfile runs/q35_08b_orijinal-f16.gguf >/dev/null 2>&1
          for o in $O1 $O2; do
            [ -f $o/bifrost-f16.gguf ] || $PY llamacpp/bifrost_gguf.py --model $HFM --conv $o/donusum_param.pt --llama $LLAMA \
              --outtype f16 --out $o/bifrost-f16.gguf >/dev/null 2>&1
          done
          [ -f $O2/bifrost-f16-memoff.gguf ] || $PY llamacpp/bifrost_keep.py --in $O2/bifrost-f16.gguf \
            --out $O2/bifrost-f16-memoff.gguf --mem-off --llama $LLAMA >/dev/null
          for m in $O1/bifrost-f16.gguf $O2/bifrost-f16.gguf $O2/bifrost-f16-memoff.gguf runs/q35_08b_orijinal-f16.gguf; do
            pk $m ${PAROLA_L:-1024,2048,4096,8192}
            $LLAMA/build/bin/llama-perplexity -m $m -f veri/wiki_test.txt -ngl 99 -c 4096 --chunks 8 2>&1 | grep "Final estimate" || true
          done ;;
  a2)     $H asama-a --model $MODEL --train $VERI/wiki_train.bin --out $OUT2 --window $PENCERE --kalib ${KALIB2:-32} \
            --seq-len ${SEQ2:-8192} --steps ${ADIMA2:-300} --sentetik ${SENT:-0.25} --mem-init long \
            --keep-global $KEEP2 --topk ${TOPK2:-0} ${DEVAM:+--devam} ;;
  a2test)
          # taban: elde zaten olan nicemlenmiş Bifrost GGUF'u (yalnız Bifrost tensörleri değişir, ~13 GB, birkaç dakika)
          BASE=${BASE:-$OUT/q38-27b-bifrost-$QUANT.gguf}
          if [ ! -f $OUT2/q38-27b-h2-$QUANT.gguf ] && [ -f $BASE ]; then
            $PY llamacpp/bifrost_gguf.py --base $BASE --conv $OUT2/donusum_param.pt --llama $LLAMA \
              --out $OUT2/q38-27b-h2-$QUANT.gguf
          fi
          [ -f $OUT2/q38-27b-h2-$GGUF_TIP.gguf ] || [ -f $OUT2/q38-27b-h2-$QUANT.gguf ] || \
            $PY llamacpp/bifrost_gguf.py --model "$(hf_dir $MODEL)" --conv $OUT2/donusum_param.pt --llama $LLAMA \
              --outtype $GGUF_TIP --out $OUT2/q38-27b-h2-$GGUF_TIP.gguf
          [ -f $OUT2/q38-27b-h2-$QUANT.gguf ] || { $LLAMA/build/bin/llama-quantize $([ $GGUF_TIP = q8_0 ] && echo --allow-requantize) \
            $OUT2/q38-27b-h2-$GGUF_TIP.gguf $OUT2/q38-27b-h2-$QUANT.gguf $QUANT && rm -f $OUT2/q38-27b-h2-$GGUF_TIP.gguf; }
          $PY bench/bifrost_param_ozet.py $OUT2/donusum_param.pt
          pk $OUT2/q38-27b-h2-$QUANT.gguf ${PAROLA_L:-8192,32768,65536}
          $LLAMA/build/bin/llama-perplexity -m $OUT2/q38-27b-h2-$QUANT.gguf -f veri/wiki_test.txt -ngl 99 -c 8192 --chunks 8 \
            -fa on -ctk q8_0 -ctv q8_0 2>&1 | grep "Final estimate" || true
          $LLAMA/build/bin/llama-bench -m $OUT2/q38-27b-h2-$QUANT.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 64 \
            -d ${DERIN:-0,32768,131072} 2>&1 | grep -E "^\| qwen" || true ;;
  a2parola)
          G=$OUT2/q38-27b-h2-$QUANT.gguf; L=${PAROLA_L:-8192,32768,65536}
          rm -f ${G%.gguf}-memoff.gguf                      # eski yol (kopya): artık gerekmiyor
          pk $G $L
          echo "-- hafıza kapalı (BIFROST_MEM_OFF=1, aynı dosya)"; BIFROST_MEM_OFF=1 pk $G $L
          [ -n "${ORIJ:-}" ] && pk $ORIJ $L ;;
  erisim)   # tutulan tam dikkat katmanları: tüm KV ↔ çapa + pencere + en iyi 16 blok (BIFROST_RET=k,pencere,blok,çapa)
          G=${ERISIM_M:-$OUT2/q38-27b-h2-$QUANT.gguf}; L=${PAROLA_L:-32768,65536}; R=${RET:-16,2048,64,4}
          pk $G $L
          echo "-- BIFROST_RET=$R"; BIFROST_RET=$R pk $G $L
          for r in "" $R; do echo "-- hız, BIFROST_RET=${r:-kapalı}"
            BIFROST_RET=$r $LLAMA/build/bin/llama-bench -m $G -ngl 99 -fa 1 -ctk ${CTK:-q8_0} -ctv ${CTV:-q8_0} \
              -p 0 -n 64 -d ${DERIN:-32768,65536} 2>/dev/null | grep -E "tg64|error"; done ;;
  motorprova)
          M=Qwen/Qwen3.5-0.8B; HFM=$(hf_dir $M); V=veri/qwen_wiki_prova; O=runs/q35_08b_m1
          [ -f $V/wiki_train.bin ] || $H hazirla --model $M --veri $V --tokens 5e6
          [ -f $O/donusum_param.pt ] && [ ! -f $O/motor_durum.json ] || $PY -m heimdall.motor egit --model $M \
            --train $V/wiki_train.bin --out $O --window 256 --seq ${SEQM:-16384} --kalib 16 --keep-every 0 \
            --adim 300 ${DEVAM:+--devam}
          $PY -m heimdall.motor onar $O/donusum_param.pt      # düzeltmeden önce kaydedilmiş dosya için (zararsız)
          $PY bench/bifrost_param_ozet.py $O/donusum_param.pt
          [ -f $O/bifrost-f16.gguf ] || $PY llamacpp/bifrost_gguf.py --model $HFM --conv $O/donusum_param.pt \
            --llama $LLAMA --outtype f16 --out $O/bifrost-f16.gguf >/dev/null 2>&1
          L=${PAROLA_L:-1024,2048,4096,8192,16384}
          pk $O/bifrost-f16.gguf $L
          echo "-- hafıza kapalı"; BIFROST_MEM_OFF=1 pk $O/bifrost-f16.gguf $L
          $LLAMA/build/bin/llama-perplexity -m $O/bifrost-f16.gguf -f veri/wiki_test.txt -ngl 99 -c 4096 --chunks 8 \
            2>&1 | grep "Final estimate" || true ;;
  motor)  $PY -m heimdall.motor egit --model $MODEL --train $VERI/wiki_train.bin --out $OUT3 --window $PENCERE \
            ${MOTOR_ARGS:-} ${DEVAM:+--devam} && $PY -m heimdall.motor onar $OUT3/donusum_param.pt ;;
  motortest)
          OUT2=$OUT3 ADIM=a2test bash "$SELF" && OUT2=$OUT3 ADIM=a2parola bash "$SELF" ;;
  sunucu) runtime/valerois-runtime $OUT/q38-27b-bifrost-$QUANT.gguf --bin $LLAMA/build/bin --ctx $CTX ${PORT:+--port $PORT} ;;
  gece)   mkdir -p $OUT
          { ADIM=a bash "$SELF" && ADIM=test bash "$SELF" && cp $OUT/degerlendirme.json $OUT/degerlendirme_A.json \
            && ADIM=b bash "$SELF" && ADIM=test bash "$SELF"; } 2>&1 | tee -a $OUT/gece.log ;;
  *)      echo "bilinmeyen ADIM=$ADIM (prova | plan | veri | a | test | b | sohbet | gguf | gece | llama | llamaprova | llamagguf | llamasohbet | llamabench | llamaprofil | llamakeep | llamaparola | provahafiza | a2 | a2test | a2parola | motorprova | motor | motortest | sunucu)"; exit 1 ;;
esac
