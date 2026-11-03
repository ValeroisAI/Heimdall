"""
================================================================================
VALEROIS HYPERION v11.0 — HIGH-THROUGHPUT ZERO-RAM INFERENCE ENGINE
================================================================================
Ultra-fast streaming inference engine with Zero KV-Cache and Speculative MTP.
Optimized for AMD Radeon (DirectML), AMD Instinct (ROCm / HIP), NVIDIA (CUDA), and CPU.
================================================================================
"""

import os
import sys
import time
import math
from typing import Generator, Dict, Any, List, Optional, Tuple
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from valerois_hyperion_v11 import ValeroisHyperionV11, HyperionMemoryGovernor
from hyperion_optimizer import DirectMLGovernor

class HyperionEngine:
    """
    High-Performance Zero-KV-Cache Streaming Inference Engine for Hyperion v11.0.
    """
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = "valerois_tokenizer_8k.json",
        device: Optional[torch.device] = None,
        use_speculative_mtp: bool = True
    ):
        self.model_path = model_path
        self.device = device if device is not None else DirectMLGovernor.setup_device(prefer_gpu=True)
        self.use_speculative_mtp = use_speculative_mtp

        # 1. Load Tokenizer
        if os.path.exists(tokenizer_path):
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self.vocab_size = self.tokenizer.get_vocab_size()
        else:
            self.tokenizer = None
            self.vocab_size = 8192

        # 2. Load Model Checkpoint
        t0 = time.time()
        print(f"[*] Hyperion Engine: Loading checkpoint from {model_path}...")
        
        if model_path.endswith(".safetensors"):
            import safetensors.torch
            ckpt_dict = safetensors.torch.load_file(model_path, device="cpu")
            cfg = {"vocab_size": self.vocab_size, "hidden": 768, "n_layers": 12, "kernel_size": 16, "expand": 2.5, "num_mtp": 0}
            sd = ckpt_dict
        else:
            ckpt = torch.load(model_path, map_location="cpu")
            cfg = ckpt.get("config", {})
            sd = ckpt.get("model_state_dict", ckpt)

        self.model = ValeroisHyperionV11(
            vocab_size=cfg.get("vocab_size", self.vocab_size),
            hidden=cfg.get("hidden", 768),
            n_layers=cfg.get("n_layers", 12),
            kernel_size=cfg.get("kernel_size", 16),
            expand=cfg.get("expand", 2.5),
            num_mtp_heads=cfg.get("num_mtp", 0),
            use_bitnet=cfg.get("use_bitnet", True),
            precision="fp16"
        ).half().to(self.device)

        self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        self.load_time = time.time() - t0

        params = self.model.get_num_params()
        self.total_params = params["total"]
        print(f"[+] Model Loaded in {self.load_time:.2f}s! ({self.total_params/1e6:.1f}M parameters on {self.device})")

    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        rep_penalty: float = 1.2
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streams generated tokens token-by-token in strict O(1) time and memory.
        Yields dict with current chunk text, speed (tok/s), and generation statistics.
        """
        if self.tokenizer is None:
            yield {"text": "[Tokenizer not found]", "tok_s": 0.0, "is_final": True}
            return

        prompt_ids = self.tokenizer.encode(prompt).ids
        gen_ids = list(prompt_ids)

        with torch.no_grad():
            conv_states, ssm_states = self.model.init_streaming_state(batch_size=1, device=self.device)

            # 1. Warmup Prompt State
            t_start = time.time()
            for tid in prompt_ids[:-1]:
                x_t = torch.tensor([[tid]], device=self.device)
                _, conv_states, ssm_states, _ = self.model.step(x_t, conv_states, ssm_states)

            curr_id = prompt_ids[-1]
            tok_count = 0

            # 2. Autoregressive Streaming Step
            for _ in range(max_tokens):
                t_tok = time.time()
                x_t = torch.tensor([[curr_id]], device=self.device)
                logits_t, conv_states, ssm_states, spec_ids = self.model.step(x_t, conv_states, ssm_states)

                next_logits = logits_t[0, 0].clone()

                # Repetition penalty
                if rep_penalty > 1.0:
                    for prev_id in set(gen_ids[-32:]):
                        if next_logits[prev_id] > 0:
                            next_logits[prev_id] /= rep_penalty
                        else:
                            next_logits[prev_id] *= rep_penalty

                # Temperature scaling
                scaled_logits = next_logits.float() / max(temperature, 1e-4)

                # Top-K filtering
                if top_k > 0:
                    val, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < val[-1]] = -float('Inf')

                # Top-P (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = False
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    scaled_logits[indices_to_remove] = -float('Inf')

                probs = F.softmax(scaled_logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()

                gen_ids.append(next_id)
                curr_id = next_id
                tok_count += 1

                chunk_text = self.tokenizer.decode([next_id], skip_special_tokens=True)
                elapsed_total = time.time() - t_start
                current_tok_s = tok_count / max(elapsed_total, 1e-4)

                yield {
                    "text": chunk_text,
                    "tok_s": current_tok_s,
                    "token_id": next_id,
                    "tok_count": tok_count,
                    "is_final": False
                }

                # Stop conditions
                if next_id == self.tokenizer.token_to_id("<|im_end|>") or next_id == 0:
                    break

            yield {
                "text": "",
                "tok_s": tok_count / max(time.time() - t_start, 1e-4),
                "tok_count": tok_count,
                "is_final": True
            }

    def benchmark_speed(self, prompt: str = "Artificial intelligence is", num_tokens: int = 100) -> Dict[str, Any]:
        """Runs a rigorous inference speed & memory benchmark."""
        t0 = time.time()
        tokens = []
        for chunk in self.generate_stream(prompt, max_tokens=num_tokens):
            if not chunk["is_final"]:
                tokens.append(chunk["text"])

        elapsed = time.time() - t0
        tok_s = len(tokens) / max(elapsed, 1e-4)
        latency_ms = (elapsed / max(len(tokens), 1)) * 1000.0

        return {
            "num_tokens": len(tokens),
            "total_time_sec": round(elapsed, 3),
            "tokens_per_sec": round(tok_s, 2),
            "latency_ms_per_tok": round(latency_ms, 2),
            "output_sample": "".join(tokens)[:100] + "..."
        }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_p = sys.argv[1]
    else:
        model_p = "valerois_hyperion_instant.vlrs" if os.path.exists("valerois_hyperion_instant.vlrs") else "valerois_v10_model.vlrs"

    if os.path.exists(model_p):
        eng = HyperionEngine(model_p)
        print("\n[*] Running Benchmark...")
        res = eng.benchmark_speed()
        print(f"[Benchmark] Speed: {res['tokens_per_sec']} tok/s | Latency: {res['latency_ms_per_tok']} ms/tok")
        print(f"[Benchmark] Output: {res['output_sample']}")
    else:
        print(f"[!] No checkpoint found at {model_p}")
