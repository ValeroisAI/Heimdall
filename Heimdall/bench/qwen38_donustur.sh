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
#
# Ortam: MODEL, OUT, VERI, PY, PENCERE (1024), KALIB (64), SEQ (4096), ADIMA (400), ADIMB (200)
#        KEEP="layers.63": o tam dikkat katmanı dönüştürülmez (birebir uzun-menzil hatırlama korunur;
#        cache token başına 4 KB büyür = orijinalin 1/16'sı, 262K'da 1 GB)
# İpucu: monitörü anakarttaki (iGPU) çıkışa takarsan masaüstü 9070 XT'den ~0.5-1 GB yemez.
set -euo pipefail
cd "$(dirname "$0")/.."
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
H="$PY -m heimdall.donustur_hibrit"

if [[ $ADIM == prova ]]; then
  M=Qwen/Qwen3.5-0.8B; O=runs/q35_08b_prova; V=veri/qwen_wiki
  $H hazirla --model $M --veri $V --tokens 5e6
  $H asama-a --model $M --train $V/wiki_train.bin --out $O --window 256 --kalib 32 --seq-len 2048 --steps 150
  $H degerlendir --model $M --out $O --test $V/wiki_test.bin --eval-len 4096 --n 4 --sifre 1024 4096 16384 --deneme 2
  echo "prova tamam. 'dönüşüm' satırı 'yalnız pencere'den iyi, 'öğretmen'e yakınsa 27B'ye geç: ADIM=plan"
  exit 0
fi

case $ADIM in
  plan)   $H plan --model $MODEL --window $PENCERE ;;
  veri)   $H hazirla --model $MODEL --veri $VERI --tokens 20e6 ;;
  a)      $H asama-a --model $MODEL --train $VERI/wiki_train.bin --out $OUT --window $PENCERE --kalib $KALIB \
            --seq-len $SEQ --steps $ADIMA ${KEEP:+--keep-global $KEEP} ;;
  test)   $H degerlendir --model $MODEL --out $OUT --test $VERI/wiki_test.bin --eval-len 8192 --n 4 \
            --sifre 4096 16384 65536 --deneme 2 ;;
  b)      $H asama-b --model $MODEL --out $OUT --seq-len 2048 --steps $ADIMB ;;
  sohbet) $H sohbet --model $MODEL --out $OUT --mtp ;;
  *)      echo "bilinmeyen ADIM=$ADIM (prova | plan | veri | a | test | b | sohbet)"; exit 1 ;;
esac
