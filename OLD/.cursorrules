# Valerois Architecture Development Guide
Read `AGENT_CONTEXT_BRIEFING.md` first. It contains the full architecture history and mission parameters.

## Core Rules:
1. Always check `AGENT_CONTEXT_BRIEFING.md` for background context.
2. Architecture Focus: Bifrost CSL + Sliding Window Cache + Attention/Retrieval Layer.
3. Keep memory footprint O(1) or lightweight O(N), no full-sequence quadratic explosion without chunking/sliding window.
4. Tokenizer: `04_TOKENIZERLAR/valerois_tokenizer_8k.json` (vocab: 8192).
5. Dataset: `stream_coder_100k.bin` is available in root and in `02_TURNUVALAR_VE_TESTLER/`.
