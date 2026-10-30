# Heimdall

A research codebase for efficient language-model training and inference on consumer GPUs (AMD ROCm and NVIDIA CUDA; CPU for tests). Summary of results and limitations: see the [repository README](../README.md). This document describes the design, the tools, and how to reproduce every number.

> Heimdall guards the Bifröst bridge: it sees far and forgets nothing.

**Status.** All components run end-to-end on an AMD RX 9070 XT (ROCm 10.1, PyTorch 2.14, Python 3.14) and on CPU. 16 unit tests pass on CPU. The largest model trained from scratch is 98 M parameters / 0.8 B tokens.

**CLI glossary** (flag and script names are Turkish): `hibrit` hybrid · `sabit` fixed(-memory) · `gece` overnight · `hafta` week · `hedef` target · `boyut` size · `uret` generate · `hazirla` prepare (data) · `donustur` convert · `karsilastir` compare · `veri` data · `kosu` run · `kucuk/orta/temel/buyuk` small/medium/base/large · `--flip-orani` flip rate · `--hassas-bit` sensitive-matrix bits.

---

## 1. Architecture

```
tokens → Embedding ─┬─ [D] Gated DeltaNet ─┐
                    ├─ [D] Gated DeltaNet   │  × (n_layers / 4), each layer followed by a SwiGLU FFN
                    ├─ [D] Gated DeltaNet   │  (pre-RMSNorm, residual connections)
                    └─ [A] Global attention ┘
                  → RMSNorm → tied LM head → logit soft-cap (30)
```
Three of every four layers are **Gated DeltaNet**, one is **global attention** (`layout="DDDA"`).

### D — Gated DeltaNet (linear-time, fixed-size memory)
```
[q;k;v] = SiLU(ShortConv4(W x))         q̂ = q/‖q‖·d^-½,  k̂ = k/‖k‖
α = exp(−e^A · softplus(W_a x + dt))     β = σ(W_b x)
S_t = α S_{t-1} + β k̂ (v − α S_{t-1}ᵀ k̂)ᵀ        o_t = S_tᵀ q̂
y = W_o( RMSNorm(o) ⊙ SiLU(W_g x) )
```
- The memory `S` is a fixed-size matrix (128×128 per head); it does not grow with context.
- **Delta rule:** writing a new value to an existing key erases the old one ("id: 1 → 2 → 3" answers 3), which purely additive linear attention cannot do.
- **Archival heads (α ≡ 1):** some heads never forget. In earlier experiments the learned forgetting gate converged to 0.99997, which erased ~88 % of stored information over 65 K tokens; archival heads fixed this.
- Training uses the chunk-parallel form (UT transform, O(T·C·d)); generation does constant work per token. A Triton kernel from `flash-linear-attention` is used when installed, after a start-up self-test against the PyTorch reference.

### A — Global attention
GQA (12 query / 4 KV heads), QK-norm, sigmoid output gate, **no positional encoding (NoPE)** in the hybrid (order information is carried by the DeltaNet layers and the short convolutions). Uses PyTorch SDPA (FlashAttention on CUDA, AOTriton on ROCm).

### Variants (`--arch`)
| `--arch` | Layout | Purpose |
|---|---|---|
| `hibrit` | DDDA, global attention, NoPE | Default hybrid; KV cache ≈ 1/12 of a same-size Transformer |
| `transformer` | A in every layer, RoPE, MHA | Baseline (same code, same training) |
| `sabit` | DDDA, 1024-token sliding-window attention, RoPE | No global attention: inference memory independent of context |
| `bit` | DDDA + 1-bit FFN (see §3) | The low-bit design (use with `--bits 1` for binary, default ternary) |

The layout is free: `--layout DDA`, `--layout DDDDDDDA`, … Optional MoE FFN (`--preset moe`: 32 fine-grained experts, top-4 + shared expert, sigmoid router, aux-loss-free balancing) is implemented and unit-tested but has not been trained at scale.

### Cache size (calculated for the 98 M "temel" preset, bf16, per sequence — not measured)
| Context | Transformer (MHA, 12 layers) | Hybrid | `sabit` |
|---|---|---|---|
| 4 K | 151 MB | 16 MB | 7 MB |
| 32 K | 1.21 GB | 104 MB | 7 MB |
| 128 K | 4.83 GB | 406 MB | 7 MB |

Measured: a trained 98 M hybrid reports a 5.1 MB cache after ~170 generated tokens.

---

## 2. Install
```bash
cd Heimdall
python3 -m venv .venv && source .venv/bin/activate
# AMD RX 9070 XT, ROCm 10.1 (configuration we measured):
pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1201]==2.14.0+rocm10.1.0"
# NVIDIA CUDA (code is device-agnostic; not benchmarked by us):
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy tokenizers safetensors fastapi "uvicorn[standard]" pydantic huggingface_hub pyarrow httpx
# optional: pip install flash-linear-attention lm-eval transformers
```
Tests (CPU, ~3 min total): `python tests/test_heimdall.py && python tests/test_bifrost.py && python tests/test_bit.py && python tests/test_bitkernel.py`.
Python 3.14 + PyTorch 2.14 supports `torch.compile` (on by default). On older stacks (PyTorch 2.9, Python 3.14) it does not; pass `--no-compile`.

---

## 3. Latent-free 1-bit / 1.58-bit training (`heimdall/bit.py`)

Standard BitNet training keeps an fp32 "latent" copy of every weight (and Adam moments); the savings exist only at inference. Here the low-bit weight **is** the parameter:

- `BitLinear` stores weights as a bf16 tensor whose values are only {−1,+1} (`bits=1`) or {−1,0,+1} (`bits=2`) — "virtual bf16": the hardware sees an ordinary matrix, and these values are exact in bf16. Output: `y = (x·Wᵀ) ⊙ (base·scale)` with a learned per-row `scale` (AdamW).
- `BitFlip` (our optimizer) accumulates gradient momentum, normalizes it per row, and each step flips the strongest `lr` fraction of weights (binary: sign flip; ternary: one step along −1→0→+1), resetting the momentum of flipped weights. The learning rate *is* the flip fraction (default 5·10⁻³ peak, following the WSD schedule to 0).
- Memory: 6 bytes/parameter during training (weight 2 + gradient 2 + momentum 2) versus 16–18 for AdamW with fp32 master weights.
- Options: `--bits {1,2}`, `--bit-scope {all,ffn,mix}` (which matrices), `--bit-rank R` (dense low-rank corrector `U·V` next to each bit matrix), `--flip-orani`, `--bit-kernel int8`.
- `--arch bit` = hybrid + FFN-only bits + rank-16 corrector. At `temel` size 58 % of parameters are 1-bit, 34 % are DeltaNet/attention mixers (Muon), 6 % embeddings.
- Embeddings, LM head, norms, gates and small projections stay dense. Not combined with MoE.

### Kernels (`heimdall/bitkernel.py`, Triton, CUDA & ROCm)
- **int8 GEMM:** per-token int8 activations × {−1,0,1} int8 weights → int32; straight-through backward. Training uses it only if it beats bf16 (it currently does not at training shapes).
- **Packed GEMM:** weights packed 8 per byte (1-bit) or 4 per byte (2-bit), unpacked inside the kernel, split-K for the memory-bound decode regime. `python -m heimdall.server --packed` replaces bit layers by packed ones.
- Kernel logic is tested in Triton's CPU interpreter against torch references (`tests/test_bitkernel.py`); speed is measured on the GPU with `bench/bitkernel.py`. If a kernel fails to compile on a GPU the code falls back to the dense path with a warning.

---

## 4. Training
```bash
# smoke test (~9 M params, minutes)
python -m heimdall.train --preset mini --data ../stream_coder_100k.bin --tokens 20e6 --out runs/mini
# 98 M binary-FFN hybrid
python -m heimdall.train --preset temel --arch bit --bits 1 --data ../stream_coder_100k.bin --tokens 800e6 --out runs/bit98
# resume (Ctrl+C is safe: it saves on exit)
python -m heimdall.train --resume runs/bit98/last.pt
```
| Preset | Hybrid params | d × L | T | Tokens / step |
|---|---|---|---|---|
| `mini` | 9.3 M | 256 × 8 | 512 | 32 K |
| `kucuk` | 44.8 M | 512 × 12 | 1024 | 128 K |
| `temel` | 96.1 M (`--arch bit`: 97.7 M) | 768 × 12 | 2048 | 256 K |
| `buyuk` | 333 M | 1024 × 24 | 2048 | 256 K |
| `moe` | 518 M total / 115 M active | 768 × 16 | 2048 | 256 K |

Stack: bf16, `torch.compile`, **Muon** (hidden dense matrices; fused qkv/w12 orthogonalized per part) + AdamW (embeddings, norms, gates) + `BitFlip`, WSD learning-rate schedule, gradient accumulation/clipping, background data loader, automatic non-finite-loss guard (stops without overwriting the last good checkpoint). Training sequence length is 2048; there is no length curriculum. On OOM lower `--micro-batch` or use `--grad-ckpt`.

**Multi-token prediction** (`--mtp-depth 1|2`, `--mtp-weight 0.3`) works in the DeepSeek-V3 / Qwen3.5 style. Module k takes the hidden state at i and the embedding of x_{i+k}, and predicts x_{i+k+1} with the shared head. Modules are DeltaNet blocks, so they stay constant-memory and can later serve as drafts. The extra loss applies only in training, so validation loss stays comparable with runs that do not use MTP. Literature caution: Gloeckle et al. (2024) found MTP *hurts* small models, while DeepSeek-V3 and Qwen use depth 1. Test 0/1/2 on `kucuk` before using it on a long run. Not measured here yet.

Data format: uint16 token files (`.bin`), mixtures as `path[:weight]`. Tokenizer: `04_TOKENIZERLAR/valerois_tokenizer_8k.json` (8192 tokens). `python -m heimdall.hazirla --repo <HF dataset> --dosya <parquet files> --kolon text --out veri/x.bin` downloads and tokenizes Hugging Face parquet data.

### Overnight run (98 M, `--arch bit --bits 1`, `bench/gece.sh`)
800 M tokens (60 % FineWeb-Edu, 32 % GitHub code, 8 % local code), 3 051 steps, 4.7 h at 47.4 K tok/s, 13.3 GB. Final validation loss: FineWeb-Edu 3.13 nats (ppl 22.9), GitHub code 0.90, local code 0.80. The code losses are optimistic (boilerplate and near-duplicates in a held-out *tail* of the training file); use a separately sourced held-out set for model comparisons (`bench/karsilastir_valkir.py`).
Sample (T=0.8, top-k 40): syntactically valid Python with a plausible class structure but semantically wrong logic; English text collapses into repetition. This is typical for 100 M parameters / 0.8 B tokens.

### Week-long run (337 M, `--preset buyuk --arch bit --bits 1`, `bench/hafta.sh`)
3.70 B tokens (14 100 steps × 262 144), mixture FineWeb-Edu 55 % / Cosmopedia 21 % / GitHub code 25 %, T=2048, Muon + AdamW + WSD (decay over the last 25 % of steps), torch.compile, ~11.2 K tok/s, 11.3 GB, paused and resumed from checkpoints several times (`bench/dur.sh`, `devam.sh`). Final validation loss: FineWeb-Edu 2.707, 2.668 (2nd split), GitHub code 0.743 / 0.606, Cosmopedia 1.773. Context-length behaviour (`bench/baglam.py`), the comparison with the legacy Valkir model and their caveats are in the root README, section 7.

---

## 5. Bifrost: converting a pretrained Transformer (`heimdall/donustur.py`)

Each attention layer of an open model (Llama, Qwen2/3, SmolLM) is replaced by: exact **sliding-window softmax attention** over the last W tokens plus 4 attention-sink tokens (identical to the teacher inside the window), plus a **gated delta-rule memory** that receives every token when it leaves the window, plus a **mass gate** that mixes them using the window's log-normalizer. The teacher's q/k/v/o weights are reused unchanged; only ~2 % new parameters are trained, in two stages: (A) layer-wise attention transfer, (B) end-to-end KL distillation.
```bash
python -m heimdall.donustur --model HuggingFaceTB/SmolLM2-135M --dogrula            # exactness check vs transformers
python -m heimdall.donustur --model HuggingFaceTB/SmolLM2-135M --wikitext veri/wikitext --wikitext-tokens 8e6 \
    --out runs/smol135 --window 64 --seq-len 1024 --batch 2 --steps-a 200 --steps-b 200 --eval-len 2048 --eval-n 16
python -m heimdall.bifrost_eval lmeval --model HuggingFaceTB/SmolLM2-135M --conv runs/smol135/donusum_param.pt --tasks arc_easy
python -m heimdall.bifrost_eval sifre  --model ... --conv ... --lengths 4000 16000 64000     # passkey retrieval
```
Result (SmolLM2-135M, window 64, GPU, 0.8 M training tokens): WikiText ppl 14.09 (original) → 25.76 (window only) → **17.61** (converted). Per bucket (tokens 1024–2048): 12.59 / 26.69 / 17.27. `--q4` keeps the frozen teacher in 4/8-bit (Qwen3-0.6B: ppl 13.88 bf16 → 14.90 mixed 4/8-bit at 318 MB → 16.26 all 4-bit); weights are streamed tensor by tensor. Verified in tests: teacher logits equal `transformers` exactly, token-by-token generation equals the parallel forward, and converted-layer cache size is independent of context length. **BitNet b1.58 teachers** (`microsoft/bitnet-b1.58-2B-4T-bf16`: ternary weights, 8-bit activations, relu², sub-norms) are supported: per-layer outputs match `transformers` to ~1e-5, and on real text the full-model NLL differs by 0.004 nats (3.8335 vs 3.8299), the same as the model's own sensitivity to 1e-6 input noise (activation rounding is discontinuous). Run: `bash bench/bitnet_donustur.sh`; conversion results are not yet measured. Long-context retrieval (`sifre`) of converted models is untested at sizes where it is meaningful (≥ 1.7 B).

### Already-hybrid teachers: Qwen3.5 / 3.6 / 3.8 dense (`heimdall/hibrit.py`, `heimdall/donustur_hibrit.py`)

Qwen3.8-27B is already built like Heimdall: 64 layers = 16 × [3 Gated DeltaNet + 1 gated full attention]. Its DeltaNet layers use the same equations as ours (L2-normalised q/k, `α = exp(−e^A·softplus(·))`, `β = σ(·)`, gated RMSNorm output). The differences are 16 QK heads shared by 48 V heads, separate depthwise convs, and no archival heads. Only the **16 full-attention layers** grow a KV cache (64 KB/token, 16 GB at 262 K), so only those are converted. DeltaNet and FFN, about 94 % of the compute, are untouched. Design choices:

* **Memory state per KV head.** The 6 query heads of a GQA group share one delta-rule state, because their writes (k, v) are identical anyway. That is 1 MB per layer. Window and sinks work as in Bifrost. **Inside the window the converted model is exactly the teacher** (tested), so chats shorter than W (default 1024) are unchanged. The window is cheap here because there are only 16 layers with 4 KV heads: whole-model cache ≈ 0.3 GB at any length.
* **Partial RoPE helps.** Only 64 of 256 dims are rotated. The other 192 carry position-free content, which is what a delta-rule memory can store.
* **Layer-streamed stage A, bf16 teacher.** One layer at a time comes from disk to the GPU. Hidden states of the calibration set (64 × 4096 tokens, 2.7 GB) stay on the GPU and are pushed through each layer, GPTQ-style. Only the full-attention layers train their new parameters (≈ 2 M per layer), matching the teacher's output after `W_o`. Peak memory is several GB, with no quantisation during conversion. The bf16 teacher's top-32 next-token probabilities are saved at the end.
* **Stage B (optional) on the 4-bit model** (g128, ≈ 12.9 GB) uses a sparse KL against the **bf16** teacher's top-k, so the new gates also absorb part of the 4-bit error. Activations are offloaded to CPU RAM and logits are computed in checkpointed chunks (vocab 248 K). By default only the gate/decay parameters train (`--params gates`).
* **MTP.** Qwen's own multi-token-prediction layer is loaded, its attention is converted the same way, and `sohbet --mtp` uses it as a draft. Each step verifies 2 tokens in one forward pass, and the output is identical to plain greedy decoding (tested). Which hidden state feeds the MTP layer was measured, not assumed: on Qwen3.5-0.8B the post-final-norm state gives 64 % draft top-1 against the teacher's next-next token, versus 34 % for the pre-norm state.

```bash
ADIM=prova bash bench/qwen38_donustur.sh    # Qwen3.5-0.8B (same architecture), end-to-end rehearsal
ADIM=plan  bash bench/qwen38_donustur.sh    # memory plan from config.json only
ADIM=veri  bash bench/qwen38_donustur.sh && ADIM=a bash bench/qwen38_donustur.sh && ADIM=test bash bench/qwen38_donustur.sh
ADIM=sohbet bash bench/qwen38_donustur.sh   # chat; cache stays constant across turns
```

Verified (CPU, `tests/test_hibrit.py`):
* The teacher matches `transformers`' `Qwen3_5ForCausalLM` on a random model (rel. 1e-4) and on real **Qwen3.5-0.8B** weights (max logit diff 4e-5 at scale 31).
* Chunked prefill plus token-by-token decoding equals the full forward for teacher, window-only and converted modes.
* Converted cache size is constant, and MTP-speculative output equals greedy output.
* A CPU smoke run of all four commands on 0.8B works (tiny data, numbers not meaningful).

**Not measured yet:** any conversion quality on Qwen3.8-27B, its speed on the RX 9070 XT, and stage B's real peak memory at 27B. The ≈ 12.9 GB weight figure is computed, not measured. Inference is plain PyTorch, dequantising 4-bit weights per matmul, so expect a few tokens/s; a llama.cpp port would be needed for fast daily use. MoE variants (e.g. 35B-A3B) are not supported.

---

### Will it fit? Planner and local UI (`planla.py`, `arayuz.py`)
```bash
python -m heimdall.planla --model Qwen/Qwen3-8B --gpu 16 --ram 32 --q4 --grad-ckpt   # reads only config.json, no weights
python -m heimdall.arayuz                                                           # http://127.0.0.1:8765 (local only, no auth)
```
The planner reports parameters, frozen-teacher memory (bf16 / `--q4` / BitNet ternary int8), new trainable parameters, KV cache of the original vs. the converted model (constant), and a green/amber/red training-memory verdict. **Parameter counts and KV/state bytes are exact** (unit-tested against the real `ConvLM`); **activation and logit memory are estimates (±40 %)** to be calibrated from the `tepe GPU` (peak GPU memory) lines that `donustur` now prints. It flags unsupported inputs (MoE, `rope_scaling`, multimodal checkpoints without a text-only loader). `--grad-ckpt` recomputes layers in the backward pass (identical loss and gradients, tested; ~30 % slower, far less activation memory). A layer-by-layer disk-streamed stage A is implemented for the hybrid Qwen3.5-family path (`donustur_hibrit.py`, above) but not yet for `donustur.py`. The UI plans, starts/stops a conversion and tails its log; it refuses to start while `heimdall.train` is running.

BitNet b1.58 teachers: `bash bench/bitnet_donustur.sh` (defaults: batch 1, grad-ckpt). One-command check after stopping a long run: `bash bench/ev_testi.sh`.

## 6. Serving (`heimdall/server.py`, `engine.py`)
```bash
python -m heimdall.export --ckpt runs/bit98/last.pt --out models/heimdall            # bf16 safetensors + config + tokenizer
HEIMDALL_API_KEYS="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" \
    python -m heimdall.server --model models/heimdall --port 8000 [--packed]
docker compose --profile rocm up -d --build          # or --profile cuda  (images provided, not built in CI)
```
OpenAI-compatible `POST /v1/completions` and `/v1/chat/completions` (SSE streaming), `GET /v1/models`, `/health`, `/metrics` (Prometheus). **Continuous batching** (position-tagged caches allow sequences of different lengths in one step; tested to give outputs identical to one-by-one generation), SHA-256-hashed API keys, per-key request-rate and concurrency limits, usage log (JSONL) for billing, client-disconnect cancellation, engine-error isolation, streaming stop sequences. Tested with unit tests (FastAPI test client); not load-tested.

---

## 7. Benchmarks and reproduction
| Script | What it measures |
|---|---|
| `bench/mqar.py H 8` | Multi-query associative recall (hybrid vs attention vs delta), CPU |
| `bench/bit_matrix.sh` | 7-variant design study (§3 of the root README), ~30 min GPU |
| `bench/bit_matrix2.sh gurultu\|olcek` | Seed noise and 46 M scale check |
| `bench/hedef.sh hiz\|kucuk\|orta\|temel` | Kernel table + torch.compile speed; bit model vs half-size dense at 6 M, 46 M, 98 M |
| `bench/bitkernel.py` | Kernel correctness/speed vs bf16 on the GPU |
| `bench/boyut.sh` | Model-size/micro-batch speed probe |
| `bench/gece.sh`, `hafta_veri.sh`, `hafta.sh` | Overnight and multi-day unattended runs (data prep, auto model choice, deadline-based token budget, auto-resume) |
| `bench/karsilastir_valkir.py` | Same-window comparison of a checkpoint with the legacy Valkir 16L model |
| `bench/uret.py` | Text generation from a checkpoint |
| `bench/dur.sh`, `devam.sh` | Cleanly pause (checkpoint at the current step) and resume the multi-day run |
| `bench/bitnet_donustur.sh`, `ev_testi.sh` | Bifrost conversion of Microsoft BitNet b1.58-2B-4T; one-command stop → tests → plan → GPU smoke test |
| `bench/lmeval_heimdall.py` | Zero-shot standard benchmarks (ARC, HellaSwag, PIQA, Winogrande, LAMBADA, WikiText word-perplexity) for a Heimdall checkpoint via lm-evaluation-harness; prints chance levels next to results. Adapter unit-tested (batched = one-by-one scoring); results and public-model comparison in the root README, section 8. `bench/ref_ozet.py` tabulates Heimdall and `lm_eval` reference runs |
| `bench/sft_veri.py` | Merges public SFT datasets into one OpenAI-chat JSONL for fine-tuning *other* models (not used by Heimdall's own training); filters broken tool traces, deduplicates, records provenance. Note on third-party model outputs in its docstring. |
| `python -m heimdall.compare runA runB` | Compare `log.jsonl` of runs |

## 8. Files
`heimdall/planla.py` memory planner · `arayuz.py` local web UI · `heimdall/config.py` configs & presets · `model.py` DeltaNet/attention/MoE/blocks/caches · `kernels.py` delta rule, attention, Triton self-test · `bit.py` low-bit layer + optimizer · `bitkernel.py` Triton kernels · `optim.py` Muon, AdamW groups, WSD · `data.py` mixtures/loader · `train.py` · `donustur.py` / `bifrost_eval.py` conversion & evaluation · `damit.py` offline top-k distillation · `engine.py` / `server.py` / `export.py` / `io.py` serving · `hazirla.py` data prep · `tests/` · `docker/`.

## 9. Known limitations
- Single GPU only; no DDP/FSDP.
- Bit-vs-dense advantage vanishes by ~100 M parameters in our tests; scale behaviour beyond that is unknown.
- Training kernels for low-bit matrix products are slower than bf16 on RDNA4 at training shapes; speed-ups apply to decoding only (layer-level, not end-to-end).
- 8 K-token tokenizer and ≤ 5 B tokens of data are not enough for a general-purpose chat model; the chat template in `server.py` is a placeholder.
- No paged KV cache; the global-attention layers' cache grows with context (≈ 1/12 of a Transformer's).
- `github-code-clean` contains mixed-license code; check licensing before publishing weights trained on it.
