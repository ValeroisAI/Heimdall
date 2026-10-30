# llama.cpp port of Bifrost layers (Qwen3.5 / 3.6 / 3.8 dense)

The converted full-attention layers run inside llama.cpp's own `qwen35` architecture as **recurrent layers with a constant
state**. That state is a ring of the last W tokens' [k_rope | v | k_mem | g | beta], S sink tokens, and one delta-rule state
per query head. The DeltaNet and FFN layers use llama.cpp's stock code. The memory update uses llama.cpp's fused
`ggml_gated_delta_net` op.

* `bifrost.patch`: diff against the llama.cpp commit in `LLAMA_CPP_COMMIT` (upstream code is MIT, see its LICENSE). It touches
  10 files (+393 lines) and applies cleanly on a fresh checkout. `kur.sh` clones, patches and builds (HIP / Vulkan / CPU).
* `bifrost_gguf.py`: llama.cpp's `convert_hf_to_gguf.py` output + our conversion tensors → Bifrost GGUF. Conversion tensors
  have no `.weight` suffix, so `llama-quantize` keeps them f32 while it quantizes everything else.
* `bifrost_logits.cpp` + `karsilastir.py`: dump llama.cpp logits and compare them with `heimdall/hibrit.py`.

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

## Not done / not verified

* **GPU backends untested.** The memory op runs with head dim 256 (Qwen's own DeltaNet uses 128). If a GPU kernel lacks it, the
  scheduler may run it on CPU. Run `ADIM=llamaprova` first.
* **MTP draft decoding** with Bifrost layers needs recurrent rollback (`n_rs_seq > 0`), which is not implemented and asserts.
* Bifrost layers still get the unused per-layer DeltaNet r/s rows (~3 MB/layer at 27B).
* The MTP layer is left unconverted (full KV, ~4 KB/token) and is only used by the draft path.
