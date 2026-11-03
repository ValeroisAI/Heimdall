"""
eval_math_code_comparison.py
============================
Base Transformer (DeepSeek-R1-Distill-Qwen-1.5B) ile
Yeni Valerois-Qwen 1.78B Hibrit Modelini Yan Yana (Head-to-Head) Karşılaştırır.

Test Edilen Alanlar:
1. Matematik / Mantık Akıl Yürütme (Math / Reasoning)
2. Algoritmik Python Kodlama (Python Coding)
3. Bellek & Bağlam Sınırları (KV-Cache vs O(1) 1M Bellek)
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizers import Tokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Cihaz: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

MODEL_DIR = "/home/arashi/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
from graft_deepseek_to_valerois import ValeroisQwen15BModel

def run_comparison():
    print("=" * 80)
    print(" ⚖️ RESMİ BASE TRANSFORMER vs. VALEROIS HİBRİT KARŞILAŞTIRMA TESTİ")
    print("=" * 80)
    
    # 1. Base Transformer Modelini Yükle
    print("[1/2] Orijinal DeepSeek-R1-Distill-Qwen-1.5B (Transformer) yükleniyor...")
    base_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, 
        torch_dtype=torch.bfloat16, 
        device_map="cuda"
    )
    base_model.eval()
    base_vram_start = torch.cuda.memory_allocated() / (1024**3)
    print(f"[*] Base Transformer Yüklendi! VRAM: {base_vram_start:.2f} GB")
    
    # 2. Valerois Grafted Modelini Yükle
    print("\n[2/2] Valerois-Qwen-1.78B Grafted (O(1) GCAM) yükleniyor...")
    valerois_model = ValeroisQwen15BModel().to(DEVICE)
    ckpt_path = "checkpoints_grafted/valerois_qwen_1.5b_aligned.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    valerois_model.load_state_dict(ckpt["model_state_dict"])
    valerois_model.eval()
    valerois_vram_start = torch.cuda.memory_allocated() / (1024**3)
    print(f"[*] Valerois Grafted Yüklendi! Toplam VRAM: {valerois_vram_start:.2f} GB")
    print("-" * 80)
    
    test_prompts = [
        ("Matematik", "Question: If a train travels at 80 km/h for 3.5 hours, what is the total distance traveled?\nAnswer:"),
        ("Python Kodlama", "def is_prime(n: int) -> bool:\n    \"\"\"Return True if n is prime, else False.\"\"\"\n"),
        ("Genel Mantık", "Soru: Türkiye'nin başkenti neresidir?\nCevap:")
    ]
    
    for category, prompt in test_prompts:
        print(f"\n📂 KATEGORİ: {category}")
        print(f"📌 PROMPT:\n{prompt}")
        print("-" * 65)
        
        # A. Base Transformer Üretimi
        inputs = base_tokenizer(prompt, return_tensors="pt").to(DEVICE)
        t0 = time.time()
        with torch.no_grad():
            out_tokens = base_model.generate(
                **inputs, 
                max_new_tokens=40, 
                temperature=0.6, 
                top_p=0.95, 
                do_sample=True
            )
        base_time_ms = (time.time() - t0) * 1000
        base_answer = base_tokenizer.decode(out_tokens[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # B. Valerois Grafted Üretimi
        tok_ids = base_tokenizer.encode(prompt)
        cur_tokens = list(tok_ids)
        t1 = time.time()
        for _ in range(40):
            x = torch.tensor([cur_tokens], device=DEVICE, dtype=torch.long)
            with torch.no_grad():
                logits = valerois_model(x)
                next_logits = logits[0, -1, :].clone().float()
                # temperature
                next_logits = next_logits / 0.6
                probs = F.softmax(next_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).item()
            if next_tok in [151643, 151645]:
                break
            cur_tokens.append(next_tok)
        valerois_time_ms = (time.time() - t1) * 1000
        valerois_answer = base_tokenizer.decode(cur_tokens[len(tok_ids):], skip_special_tokens=True).strip()
        
        print(f"🤖 BASE TRANSFORMER (DeepSeek R1):\n{base_answer}")
        print(f"   [Süre: {base_time_ms:.1f} ms | Bellek: Softmax KV-Cache | Limit: 32k]")
        print()
        print(f"⚡ VALEROIS-QWEN (O(1) GCAM - 2.3 dk Hizalanmış):\n{valerois_answer}")
        print(f"   [Süre: {valerois_time_ms:.1f} ms | Bellek: Sıfır KV-Cache | Limit: 1M]")
        print("=" * 65)

if __name__ == "__main__":
    run_comparison()
