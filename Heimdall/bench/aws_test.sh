#!/bin/bash
# AWS (NVIDIA) ilk test: ~1-2 saat, ~$5-15. Üç soruya cevap verir:
#   1) CUDA'da testler geçiyor mu?   2) 337M eğitim: tok/s, GPU belleği, KAYIP evdeki koşuyla uyuşuyor mu?
#   3) Qwen3-8B Bifrost dönüşümü sığıyor mu, saniyede kaç token?  (tepe GPU belleği ve hız yazılır)
#
# Kurulum (Deep Learning AMI, PyTorch'lu ortam etkin):
#   git clone https://github.com/valeroisai/bifrost.git && cd bifrost/Heimdall
#   pip install numpy tokenizers safetensors huggingface_hub pyarrow transformers accelerate   # (torch AMI'de var)
#   bash bench/aws_test.sh                   # tümü
#   ADIM=hiz bash bench/aws_test.sh          # yalnız 337M hız/kayıp   (ADIM = test | hiz | dnsm)
# Ortam: DATA="veri/fineweb.bin ..." (varsayılan: mevcut veri/*.bin, yoksa ../stream_coder_100k.bin: hız ölçülür, kayıp kıyası ANLAMSIZ)
#        MB=8 (micro-batch; H100 80 GB için)   T=600 (hız testi süresi, sn)   Q4=1 (48 GB kartta 8B için 4-bit öğretmen)
cd "$(dirname "$0")/.."
ADIM=${ADIM:-hepsi}; OUT=runs/aws_test; mkdir -p $OUT veri
LOG=$OUT/ozet.txt; : > $LOG
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
say "GPU: $(python -c 'import torch;print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/2**30), "GB, torch", torch.__version__)' 2>&1 | tail -1)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | tee -a $LOG

if [ "$ADIM" = test ] || [ "$ADIM" = hepsi ]; then
  say "1/3 birim testleri"
  for t in test_heimdall test_bifrost test_bit test_bitkernel test_planla; do
    python tests/$t.py > $OUT/$t.out 2>&1 && say "  OK   $t" || { say "  FAIL $t (bkz $OUT/$t.out)"; tail -3 $OUT/$t.out | tee -a $LOG; }
  done
fi

if [ "$ADIM" = hiz ] || [ "$ADIM" = hepsi ]; then
  D=${DATA:-$(ls veri/fineweb.bin veri/fw_b.bin veri/kod.bin veri/kod_b.bin veri/cosmo.bin 2>/dev/null | tr '\n' ' ')}
  REF=1
  if [ -z "$D" ]; then D=../stream_coder_100k.bin; REF=0; say "UYARI: veri/*.bin yok → stream_coder_100k.bin: hız geçerli, KAYIP kıyası geçersiz"; fi
  say "2/3 337M ('buyuk', 1-bit) hız ve kayıp testi: veri=$D, micro-batch ${MB:-8}, ${T:-600} sn"
  # toplam token evdeki koşuyla AYNI (3.696B): öğrenme hızı çizelgesi aynı olsun; süre dolunca kes
  timeout ${T:-600} python -m heimdall.train --arch bit --bits 1 --preset buyuk --data $D --tokens 3.696e9 --micro-batch ${MB:-8} \
      --no-grad-ckpt --eval-every 100000 --save-every 0 --log-every 10 --out $OUT/buyuk 2>&1 | grep --line-buffered -v amdgpu.ids > $OUT/buyuk.log
  grep -E "param|adım" $OUT/buyuk.log | awk 'NR==1{print} /adım/{last=$0; n++; if(n%5==0)print} END{print last}' | tee -a $LOG
  if [ "$REF" = 1 ]; then
    say "KAYIP KIYASI (evdeki koşu, AYNI veri ve ayar: adım 50: 5.0338 | 100: 4.1365 | 150: 3.8356 | 200: 3.3804). Beklenti: aynı büyüklükte (±0.1)."
  fi
fi

if [ "$ADIM" = dnsm ] || [ "$ADIM" = hepsi ]; then
  say "3/3 Qwen3-8B Bifrost dönüşümü mini koşu (20+20 adım, seq 1024, batch ${B:-1}); tepe GPU belleği ve hız yazılır"
  python -m heimdall.planla --model Qwen/Qwen3-8B --gpu $(python -c 'import torch;print(round(torch.cuda.get_device_properties(0).total_memory/2**30))') \
      --seq-len 1024 --batch ${B:-1} --grad-ckpt ${Q4:+--q4} 2>&1 | grep -v Warn | tee -a $LOG
  python -m heimdall.donustur --model Qwen/Qwen3-8B --wikitext veri/wikitext_q3_8b --wikitext-tokens 3e6 --sep 151643 --out $OUT/q3_8b \
      --window 64 --sinks 4 --seq-len 1024 --batch ${B:-1} --steps-a 20 --steps-b 20 --eval-n 4 --eval-len 1024 --grad-ckpt ${Q4:+--q4} \
      2>&1 | grep --line-buffered -v amdgpu.ids | tee $OUT/q3_8b.log | grep -E "Öğretmen|ogretmen|donusum|aşama|tepe GPU|Error|out of memory" | tee -a $LOG
fi
say "BİTTİ. Özet: $LOG   |   örneği SİL (terminate) etmeyi unutma."
