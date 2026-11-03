"""
valerois_qwen_generate.py
=========================
DeepSeek-R1-Distill-Qwen-1.5B modelinden nakledilen ValeroisGCAM modelinin
sıfır KV-cache maliyetli bağımsız çıkarım (text generation) motoru.
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors.torch
from tokenizers import Tokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "/home/arashi/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
SAFETENSORS_PATH = os.path.join(MODEL_DIR, "model.safetensors")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")

# Mimari tanımları
from graft_deepseek_to_valerois import ValeroisQwen15BModel, perform_grafting

def generate(model, tokenizer, prompt, max_new_tokens=40, temperature=0.7, top_k=50, rep_penalty=1.15):
    input_ids = tokenizer.encode(prompt).ids
    tokens = list(input_ids)
    
    print(f"\n[Prompt]: {prompt}")
    print("[Üretilen]: ", end="", flush=True)
    
    for _ in range(max_new_tokens):
        x = torch.tensor([tokens], device=DEVICE, dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
            next_logits = logits[0, -1, :].clone().float()
            
            # Repetition Penalty (Kullanıcının istediği ceza mekanizması)
            for prev_tok in set(tokens[-30:]):
                if next_logits[prev_tok] > 0:
                    next_logits[prev_tok] /= rep_penalty
                else:
                    next_logits[prev_tok] *= rep_penalty
                    
            if temperature > 0:
                next_logits = next_logits / temperature
                if top_k > 0:
                    val, idx = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < val[-1]] = -float('Inf')
                probs = F.softmax(next_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).item()
            else:
                next_tok = torch.argmax(next_logits).item()
                
        # Stop tokens
        if next_tok in [151643, 151645]: # <|endoftext|>, <|im_end|>
            break
            
        tokens.append(next_tok)
        word = tokenizer.decode([next_tok])
        print(word, end="", flush=True)
        
    print("\n" + "-" * 60)
    return tokenizer.decode(tokens)

def main():
    print("=" * 70)
    print(" 🚀 VALEROIS 1.78B HYBRID GENERATION MOTORU (SIFIR KV-CACHE)")
    print("=" * 70)
    aligned_ckpt = "checkpoints_grafted/valerois_qwen_1.5b_aligned.pt"
    if os.path.exists(aligned_ckpt):
        print(f"[*] Hizalanmış Checkpoint Bulundu: {aligned_ckpt}")
        model = ValeroisQwen15BModel().to(DEVICE)
        ckpt = torch.load(aligned_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        print("[*] Hizalanmış model başarıyla yüklendi!")
    else:
        model, tokenizer = perform_grafting()
    
    prompts = [
        "def fibonacci(n):",
        "The capital of France is",
        "def calculate_total(price, tax):"
    ]
    
    for p in prompts:
        generate(model, tokenizer, p, max_new_tokens=30, temperature=0.7, rep_penalty=1.25)

if __name__ == "__main__":
    main()
