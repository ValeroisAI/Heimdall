# llama.cpp port (WORK IN PROGRESS, not verified yet)

Bifrost-converted Qwen3.5 / 3.6 / 3.8 layers inside llama.cpp's `qwen35` architecture. The converted
full-attention layers become recurrent (constant state): window ring + sinks + delta memory.

* `bifrost.patch` — diff against llama.cpp commit in `LLAMA_CPP_COMMIT` (MIT-licensed upstream code; see its LICENSE).
  It compiles on CPU; **logits have not been compared with the PyTorch reference yet**.
* `bifrost_gguf.py` — base GGUF (llama.cpp `convert_hf_to_gguf.py`) + our conversion tensors → Bifrost GGUF.
* `bifrost_logits.cpp` + `karsilastir.py` — dump llama.cpp logits and compare with `heimdall/hibrit.py`.
  Verified so far: stock llama.cpp Qwen3.5-0.8B equals our PyTorch teacher (argmax agreement 1.000, NLL 1.5691 vs 1.5692).

Not supported yet: MTP draft decoding with Bifrost layers (needs recurrent rollback), GPU kernels untested (head dim 256 in
the fused delta-net op).
