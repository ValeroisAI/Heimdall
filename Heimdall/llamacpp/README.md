# llama.cpp port of Bifrost layers (Qwen3.5 / 3.6 / 3.8 dense)

The converted full-attention layers run inside llama.cpp's own `qwen35` architecture as **recurrent layers with a constant
state**. That state is a ring of the last W+1 tokens' [k_rope | v | k_mem | g | beta], S sink tokens, and one delta-rule state
per KV head (the query heads of a group share it). The DeltaNet and FFN layers use llama.cpp's stock code. The memory update uses llama.cpp's fused
`ggml_gated_delta_net` op.

* `bifrost.patch`: diff against the llama.cpp commit in `LLAMA_CPP_COMMIT` (upstream code is MIT, see its LICENSE). It touches
  14 files (+554 lines) and applies cleanly on a fresh checkout. `kur.sh` clones, patches and builds (HIP / Vulkan / CPU).
* `bifrost_gguf.py`: llama.cpp's `convert_hf_to_gguf.py` output + our conversion tensors → Bifrost GGUF. Conversion tensors
  have no `.weight` suffix, so `llama-quantize` keeps them f32 while it quantizes everything else.
* `bifrost_logits.cpp` + `karsilastir.py`: dump llama.cpp logits and compare them with `heimdall/hibrit.py`.
* `bifrost_multiseq.cpp`: two sequences decoded together must match decoding them one at a time.

## Verified (CPU, Qwen3.5-0.8B, window 64, 4 sinks; memory branch active after token 64)

| GGUF | max logit diff (scale 26) | argmax agreement | NLL PyTorch → llama.cpp |
|---|---|---|---|
| stock llama.cpp, unconverted, f32 | 3.3e-2 | 1.000 | 1.5691 → 1.5692 |
| **Bifrost f32**, 160 tok (37,1,50 + singles) | 4.8e-5 | 1.000 | 2.0838 → 2.0838 |
| **Bifrost f32**, 400 tok in one chunk / mixed chunks / 300 single tokens | ≤ 3.9e-5 | 1.000 | equal |
| Bifrost Q8_0 (llama-quantize) | 0.66 | 0.990 | 1.9572 → 1.9570 |
| Bifrost Q4_K_M | 4.6 | 0.865 | 1.9572 → 1.9770 |
| unconverted Q4_K_M (same test) | 4.0 | 0.973 | 0.6343 → 0.6356 |

The Q4 rows differ mostly because the test text repeats and the tiny 0.8B conversion predicts it less confidently. The largest
4-bit logit error is the same with and without Bifrost.

Memory at `n_ctx = 262144` (0.8B Q4_K_M):
* original: KV cache 3072 MiB + 19 MiB recurrent;
* Bifrost: KV 0 MiB + 26 MiB recurrent + 14 MiB Bifrost state, the same at every length.

## Speed work (architecture-specific)

* **GPU memory op for head dim 256.** Qwen's own DeltaNet uses 128, and our memory runs at 256. Before this, Vulkan had no
  256 kernel, so the op ran on CPU every token and the 27B decoded at half speed (17 vs 37 t/s). The patch adds 256 to the
  Vulkan and CUDA/HIP `GATED_DELTA_NET` kernels. The whole graph now runs on the GPU (`splits = 1`).
* **State read in place.** Decoding no longer copies the whole per-layer state (~26 MB at 27B) every token. The ring has W+1
  slots, so the row leaving the window and the new row never share a slot, and no ordering ops are needed.
* **One delta state per KV head** instead of per query head: 6× less state and memory-op work at 27B (24 Q / 4 KV heads).
* **Window attention in decode** uses one query column per head with the KV head broadcast. A grouped variant (the 6 query
  heads of a group as 6 columns of one matmul) was 5× slower on RADV: 270 µs vs ~25 µs per layer for p·V. The region length
  is padded to a multiple of 32.
* **No layout copies in decode.** 132 → 12 `cont` ops (0.8B).

Measured on CPU (4 threads, Qwen3.5-0.8B Q4_K_M, window 64). The GPU numbers have to come from a real GPU.

| | pp512 | tg64 | pp512 @ d4096 | tg64 @ d4096 |
|---|---|---|---|---|
| original | 581 | 37.5 | 401 | 37.6 |
| Bifrost, before | 488 | 35.8 | 493 | 36.4 |
| **Bifrost, now** | 485 | **39.0** | **544** | **38.8** |

Decode ops per token: original 867, Bifrost before 1293, now 1125.

RX 9070 XT, Vulkan (RADV), Qwen3.8-27B Q3_K_M (Bifrost: window 2048, 4 sinks, `layers.63` kept global), llama-bench t/s:

| | tg d0 | tg d32K | tg d131K | pp512 d32K |
|---|---|---|---|---|
| original | 37.1 | 32.9 | — | 682 |
| Bifrost, GDN-256 on CPU | 17.2 | 17.1 | 16.8 | ~320–500 |
| Bifrost, grouped decode attention | 31.4 | 31.3 | 30.6 | 835 |
| Bifrost, per-head decode attention | 33.1 | 33.0 | 32.3 | 834 |
| **Bifrost, K/V planes** | **33.7** | **33.7** | **32.9** | **832** |

At 32K Bifrost decodes as fast as the original and prefills 22% faster. At 131K the speed is unchanged. The remaining
gap at short context (~3 ms/token) was the window attention: f32 keys read through a strided view (QK^T ≈ 70 µs per
layer) plus the V transpose copy.

* **K / V planes for decode.** The state row now also holds a K plane [D, L, G] and a transposed V plane [L, D, G]. Every
  write goes to both the region and the planes. The decode attention reads the planes directly: QK^T and p·V are plain
  contiguous matrix-vector products, with no strided view and no transpose copy. The planes are f32 inside the same state
  row, so sequence copy / clear / save keep working unchanged. Cost: +17 MB state per converted layer at 27B (~0.26 GB in
  total). f16 planes would need a separate cache tensor and save maybe ~0.2 ms/token more.
* With the planes, QK^T dropped from 70 to 44 µs per layer. The remaining gap at short context (~2.7 ms/token) is mostly
  the ~40 extra small ops per converted layer (3–7 µs each on RADV). Closing it needs a fused Bifrost decode kernel.
* `bifrost_multiseq` also checks a `seq_cp` copy that continues differently and a reused cell after `seq_rm`. Both match a
  fresh single sequence (≤ 1e-5).

Correctness after the speed work:
* CPU and Vulkan (lavapipe) match PyTorch at ≤ 4.1e-5 with argmax 1.000. Tested with one 400-token chunk, mixed chunks
  (3,61,1,1,200,2,100), all single tokens, and chunks that straddle the window.
* Two sequences decoded together (B=2) match decoding them one at a time (9.5e-6).

## Retrieval attention for the kept full-attention layers (`BIFROST_RET`)

The layers left as full attention (Qwen's own, or the kept ones of a 12/4 conversion) can read only part of their
KV cache at decode: the sinks, the recent window, and the `k` key blocks whose upper bound on q·k is highest
(per KV head; per-block key max/min kept next to the KV cache, Quest-style). The softmax over those tokens is exact.
No training; it works on the original model too.

    BIFROST_RET=16,2048,64,4 FA=1 bifrost_passkey model.gguf 32768,65536     # k, window, block, sinks

* Only single-token decode uses it; the prompt is read with full attention. It turns on once a position has at
  least `k` blocks before its window.
* Each attention layer then reads about `sinks + window + block + k*block` tokens per step (here ~3.1K) plus the
  block summary (f32 max and min per 64-token block: 1/16 of an f16 K cache), instead of the whole context.
* It needs flash attention (V not transposed) and one sequence whose cells are its positions (a fresh or cleared
  cache, no context shift). Saved session state does not include the summaries.
* The KV cache still sits on the GPU. Keeping it in host RAM and copying only the picked blocks is the next step.
* Verified (CPU, Qwen3.5-0.8B f32, `BIFROST_RET=4,256,64,4`, 500-token prefill then 352 single-token steps):
  logits match the PyTorch reference within 9e-4 relative (max), the same top token at every step; retrieval
  itself moves the logits by 5e-2 against full attention, so both picked the same blocks.
* Reference: `heimdall/hibrit.py` (`HybridAttention.set_retrieval`); `bench/erisim_parola.py` compares pass-key
  recall with full attention, window-only and retrieval.

## Not done / not verified

* **Vulkan is verified only on lavapipe (software).** Speed on a real GPU is not measured yet. The CUDA/HIP 256 kernel case is
  not compiled here (no toolkit). Run `ADIM=llamaprova` first on a new machine.
* **MTP draft decoding** with Bifrost layers needs recurrent rollback (`n_rs_seq > 0`), which is not implemented and asserts.
* Bifrost layers still get the unused per-layer DeltaNet r/s rows (~3 MB/layer at 27B).
* The MTP layer is left unconverted (full KV, ~4 KB/token) and is only used by the draft path.
