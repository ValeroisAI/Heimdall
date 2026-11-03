"""
graft_valerois_production.py
============================
DeepSeek-R1-Distill-Qwen-1.5B agirliklarini yeni ValeroisProductionGCAM
mimarisine aktarir ve 'valerois_qwen_1.5b_production_base.pt' olarak kaydeder.
"""

import os
import time
import torch
import safetensors.torch
from transformers import AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from valerois_core_model import ValeroisQwenProductionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "/home/arashi/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
SAFETENSORS_PATH = os.path.join(MODEL_DIR, "model.safetensors")
OUTPUT_CKPT = "checkpoints_grafted/valerois_qwen_1.5b_production_base.pt"

def perform_production_grafting(chunk_size=512):
    print("=" * 80)
    print(" 🏥 DEEPSEEK-R1 -> VALEROIS PRODUCTION HİBRİT MODEL NAKLİ (GRAFTING)")
    print("=" * 80)
    print(f"[*] Safetensors: {SAFETENSORS_PATH}")
    print(f"[*] Chunk Boyutu: {chunk_size} Token")

    t0 = time.time()
    model = ValeroisQwenProductionModel(chunk_size=chunk_size).to(DEVICE)
    
    print("[*] Safetensors agirliklari yukleniyor...")
    weights = safetensors.torch.load_file(SAFETENSORS_PATH, device="cpu")

    with torch.no_grad():
        # 1. Embedding & LM Head
        print("[1/3] Embedding ve LM Head aktariliyor...")
        model.embed_tokens.weight.copy_(weights["model.embed_tokens.weight"])
        model.lm_head.weight.copy_(weights["lm_head.weight"])
        model.norm.weight.copy_(weights["model.norm.weight"])

        # 2. 28 Katman
        print("[2/3] 28 Katmanin Attention ve MLP agirliklari aktariliyor...")
        for i in range(28):
            block = model.layers[i]
            prefix = f"model.layers.{i}."

            # Projeksiyonlar (Q, K, V, O)
            block.self_attn.q_proj.weight.copy_(weights[f"{prefix}self_attn.q_proj.weight"])
            block.self_attn.q_proj.bias.copy_(weights[f"{prefix}self_attn.q_proj.bias"])
            block.self_attn.k_proj.weight.copy_(weights[f"{prefix}self_attn.k_proj.weight"])
            block.self_attn.k_proj.bias.copy_(weights[f"{prefix}self_attn.k_proj.bias"])
            block.self_attn.v_proj.weight.copy_(weights[f"{prefix}self_attn.v_proj.weight"])
            block.self_attn.v_proj.bias.copy_(weights[f"{prefix}self_attn.v_proj.bias"])
            block.self_attn.o_proj.weight.copy_(weights[f"{prefix}self_attn.o_proj.weight"])

            # Layer Normlar
            block.input_layernorm.weight.copy_(weights[f"{prefix}input_layernorm.weight"])
            block.post_attention_layernorm.weight.copy_(weights[f"{prefix}post_attention_layernorm.weight"])

            # MLP Katmanlari (SwiGLU)
            block.mlp.gate_proj.weight.copy_(weights[f"{prefix}mlp.gate_proj.weight"])
            block.mlp.up_proj.weight.copy_(weights[f"{prefix}mlp.up_proj.weight"])
            block.mlp.down_proj.weight.copy_(weights[f"{prefix}mlp.down_proj.weight"])

    print(f"[*] Nakil Basariyla Tamamlandi! Sure: {time.time() - t0:.2f} s")
    
    # 3. Model Kontrol & Kaydetme
    os.makedirs("checkpoints_grafted", exist_ok=True)
    print(f"[3/3] Model diske kaydediliyor: {OUTPUT_CKPT}...")
    torch.save({"model_state_dict": model.state_dict(), "chunk_size": chunk_size}, OUTPUT_CKPT)
    print("[*] Model kaydedildi!")
    
    return model

def verify_production_model(model):
    print("\n" + "=" * 80)
    print(" 🔬 ÜRETİM MODELİ DOĞRULAMA TESTLERİ")
    print("=" * 80)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    cfg = Qwen2Config.from_pretrained(MODEL_DIR)
    rotary = Qwen2RotaryEmbedding(cfg).to(DEVICE)

    test_prompts = [
        ("GSM8K Matematik", "Question: If a car travels 150 km in 2 hours, what is its speed in km/h?\nAnswer:"),
        ("Python Algoritma", "def is_prime(n: int) -> bool:\n    \"\"\"Check if n is prime.\"\"\"\n"),
        ("Cok Adimli Mantik", "Question: John has 5 apples, gives 2 to Mary, then buys 3 more. How many apples does he have?\nAnswer:")
    ]

    for cat, p in test_prompts:
        input_ids = tokenizer.encode(p)
        cur = list(input_ids)
        for _ in range(35):
            x_in = torch.tensor([cur], device=DEVICE)
            with torch.no_grad():
                T = x_in.shape[1]
                pos_ids = torch.arange(T, device=DEVICE).unsqueeze(0)
                dummy_x = torch.zeros(1, T, 1536, dtype=torch.bfloat16, device=DEVICE)
                cos, sin = rotary(dummy_x, pos_ids)
                logits = model(x_in, cos=cos, sin=sin)
                tok = torch.argmax(logits[0, -1, :]).item()
            if tok in [151643, 151645]:
                break
            cur.append(tok)
        gen = tokenizer.decode(cur[len(input_ids):], skip_special_tokens=True).strip()
        print(f"\n📌 [{cat}]: {p.strip()}")
        print(f"🤖 [Model Yaniti]:\n{gen}")

if __name__ == "__main__":
    model = perform_production_grafting(chunk_size=512)
    verify_production_model(model)

