#!/usr/bin/env bash
# Bifrost yamalı llama.cpp'yi kurar (sabitlenmiş sürüm + bifrost.patch) ve derler.
#   bash llamacpp/kur.sh                 # ROCm/HIP, RX 9070 XT (gfx1201)
#   GPU=vulkan bash llamacpp/kur.sh      # Vulkan
#   GPU=cpu bash llamacpp/kur.sh         # yalnız CPU
# Ortam: DIR (varsayılan ~/llama.cpp-bifrost), GPU_TARGET (varsayılan gfx1201)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR=${DIR:-$HOME/llama.cpp-bifrost}
GPU=${GPU:-hip}
GPU_TARGET=${GPU_TARGET:-gfx1201}
COMMIT=$(cat "$HERE/LLAMA_CPP_COMMIT")

# araç kontrolü (eksikse ne kurulacağını söyle)
need() { command -v "$1" >/dev/null || { echo "EKSİK: $1 -> $2"; exit 1; }; }
need git   "sudo apt install git"
need cmake "sudo apt install cmake build-essential"
need g++   "sudo apt install build-essential"
if [ "$GPU" = hip ]; then
  for d in /opt/rocm/bin /opt/rocm-*/bin; do [ -x "$d/hipconfig" ] && export PATH="$d:$PATH" && break; done
  if ! command -v hipconfig >/dev/null; then
    echo "ROCm geliştirme araçları (hipconfig/hipcc) yok. İki seçenek:"
    echo "  1) Vulkan ile derle (en kolayı): sudo apt install libvulkan-dev glslc spirv-headers glslang-tools && GPU=vulkan bash llamacpp/kur.sh"
    echo "  2) ROCm HIP SDK kur (amdgpu-install deposu ekliyse): sudo apt install rocm-hip-sdk"
    exit 1
  fi
fi
if [ "$GPU" = vulkan ]; then
  need glslc "sudo apt install glslc libvulkan-dev spirv-headers glslang-tools"
  [ -d /usr/share/cmake/SPIRV-Headers ] || [ -d /usr/lib/cmake/SPIRV-Headers ] || ls /usr/share/*/cmake/SPIRV-Headers* >/dev/null 2>&1 || { echo "EKSİK: SPIRV-Headers -> sudo apt install spirv-headers"; exit 1; }
fi

if [ ! -d "$DIR/.git" ]; then
  git clone --filter=blob:none https://github.com/ggml-org/llama.cpp "$DIR"
fi
cd "$DIR"
git fetch --depth 1 origin "$COMMIT" || git fetch origin
git checkout -f -q "$COMMIT"
git reset -q --hard "$COMMIT"
git apply "$HERE/bifrost.patch"
echo "llama.cpp $COMMIT + bifrost.patch uygulandı"

case $GPU in
  hip)    HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
            cmake -S . -B build -DGGML_HIP=ON -DGPU_TARGETS="$GPU_TARGET" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF ;;
  vulkan) cmake -S . -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF ;;
  cpu)    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF ;;
  *)      echo "GPU=hip|vulkan|cpu"; exit 1 ;;
esac
cmake --build build -j"$(nproc)" --target llama-cli llama-server llama-quantize llama-bench llama-perplexity
g++ -O2 -std=c++17 "$HERE/bifrost_logits.cpp" -Iinclude -Iggml/include -Lbuild/bin -lllama -lggml -lggml-base \
    -Wl,-rpath,"$PWD/build/bin" -o build/bin/bifrost_logits
echo "hazır: $DIR/build/bin (llama-cli, llama-server, llama-quantize, llama-bench, bifrost_logits)"
