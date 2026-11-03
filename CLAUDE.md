# Valerois Architecture Development Guide
Active project: `Heimdall/` (see `Heimdall/README.md`). Old work (Bifrost CSL, Kuzgun, Valkir, ...) lives in `OLD/`; `OLD/AGENT_CONTEXT_BRIEFING.md` has the history.

## Core Rules:
1. Check `OLD/AGENT_CONTEXT_BRIEFING.md` only for historical background.
2. Architecture Focus: Heimdall hybrid (Gated DeltaNet + global attention), latent-free low-bit FFN (BitFlip), Bifrost conversion of pretrained Transformers.
3. Keep memory footprint O(1) or lightweight O(N), no full-sequence quadratic explosion without chunking/sliding window.
4. Tokenizer: `04_TOKENIZERLAR/valerois_tokenizer_8k.json` (vocab: 8192).
5. Dataset: `stream_coder_100k.bin` is available in root (copy also in `OLD/02_TURNUVALAR_VE_TESTLER/`).
