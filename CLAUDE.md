# Valerois Architecture Development Guide
Active project: `Heimdall/` (see `Heimdall/README.md`).

## Core Rules:
1. Architecture Focus: Heimdall hybrid (Gated DeltaNet + global attention), latent-free low-bit FFN (BitFlip), conversion of pretrained Transformers to constant memory.
2. Keep memory footprint O(1) or lightweight O(N), no full-sequence quadratic explosion without chunking/sliding window.
3. Tokenizer: `04_TOKENIZERLAR/valerois_tokenizer_8k.json` (vocab: 8192).
4. Dataset: `stream_coder_100k.bin` is available in root.
