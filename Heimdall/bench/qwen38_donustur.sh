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
GGUF_TIP=${GGUF_TIP:-q8_0}   # bf16 en iyisi ama 27B'de ~110 GB boş disk ister; q8_0 ~60 GB
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
  gece)   mkdir -p $OUT
          { ADIM=a bash "$SELF" && ADIM=test bash "$SELF" && cp $OUT/degerlendirme.json $OUT/degerlendirme_A.json \
            && ADIM=b bash "$SELF" && ADIM=test bash "$SELF"; } 2>&1 | tee -a $OUT/gece.log ;;
  *)      echo "bilinmeyen ADIM=$ADIM (prova | plan | veri | a | test | b | sohbet | gguf | gece | llama | llamaprova | llamagguf | llamasohbet | llamabench)"; exit 1 ;;
esac
