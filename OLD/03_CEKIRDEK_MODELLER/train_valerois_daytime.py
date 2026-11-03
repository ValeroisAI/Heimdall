"""
train_valerois_daytime.py
=========================
Valerois-Qwen 1.78B Gündüz Maratonu (Akşam 18:00'e Kadar).
- Model: ValeroisQwenProductionModel (28 Katman GCAM, Chunk Size: 512)
- Eğitilen Parametreler: 28 Katmanın Valerois Hafıza Kapıları (Gereklilik, Bozulma, Inter-gate) + Son 4 Katman Projeksiyonları
- Bağlam: 4,096 Token (8 Chunk)
- Gradient Checkpointing ile VRAM: ~6.0 GB (16 GB RX 9070 XT üzerinde son derece güvenli)
- Disk Koruması: Tek dosya üzerine döngüsel kayıt (Disk dolmaz)
"""

import os
import sys
import time
import math
import glob
import random
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from tokenizers import Tokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from datasets import load_dataset

from valerois_core_model import ValeroisQwenProductionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Cihaz: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

MODEL_DIR = "/home/arashi/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
BASE_CKPT_PATH = "checkpoints_grafted/valerois_qwen_1.5b_production_base.pt"
LATEST_CKPT_PATH = "checkpoints_grafted/valerois_qwen_1.5b_daytime_latest.pt"
FINAL_CKPT_PATH = "checkpoints_grafted/valerois_qwen_1.5b_daytime_final.pt"

def build_training_dataset(tokenizer):
    print("\n" + "=" * 80)
    print(" 📚 EĞİTİM VERİ SETİ HAZIRLANIYOR (GSM8K + CodeAlpaca + Python25k + Alpaca18k)...")
    print("=" * 80)
    samples = []

    # 1. GSM8K
    try:
        ds_gsm8k = load_dataset("openai/gsm8k", "main", split="train")
        for item in ds_gsm8k:
            text = f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n{item['answer']}<|im_end|>\n"
            samples.append(text)
        print(f"  [+] GSM8K Eklendi: {len(ds_gsm8k):,} örnek")
    except Exception as e:
        print("  [-] GSM8K hatası:", e)

    # 2. CodeAlpaca-20k
    try:
        ds_code = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        for item in ds_code:
            inst = item.get("instruction", "")
            inp = item.get("input", "")
            out = item.get("output", "")
            user_msg = f"{inst}\n{inp}".strip() if inp else inst
            text = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>\n"
            samples.append(text)
        print(f"  [+] CodeAlpaca Eklendi: {len(ds_code):,} örnek")
    except Exception as e:
        print("  [-] CodeAlpaca hatası:", e)

    # 3. Python-Codes-25k
    try:
        ds_py25k = load_dataset("Flytech/python-codes-25k", split="train")
        for item in ds_py25k:
            text = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>\n"
            samples.append(text)
        print(f"  [+] Python-Codes-25k Eklendi: {len(ds_py25k):,} örnek")
    except Exception as e:
        print("  [-] Python-Codes-25k hatası:", e)

    # 4. Yerel Proje Kodları
    py_files = glob.glob("/home/arashi/Desktop/v3/*.py")
    for fpath in py_files[:150]:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                c = f.read()
                if len(c) > 200:
                    samples.append(c)
        except Exception:
            pass
    print(f"  [+] Yerel Proje Kodları Eklendi: {len(py_files)} dosya")

    random.seed(42)
    random.shuffle(samples)
    print(f"[*] Toplam {len(samples):,} Zengin Örnek Toplandı! Tokenize ediliyor...")

    all_token_ids = []
    chunk_text = []
    chunk_char_count = 0

    for s in samples:
        chunk_text.append(s)
        chunk_char_count += len(s)
        if chunk_char_count > 500000:
            batch_text = "\n\n".join(chunk_text)
            all_token_ids.extend(tokenizer.encode(batch_text).ids)
            chunk_text = []
            chunk_char_count = 0

    if chunk_text:
        batch_text = "\n\n".join(chunk_text)
        all_token_ids.extend(tokenizer.encode(batch_text).ids)

    print(f"[*] 🏆 EĞİTİM VERİ HAVUZU HAZIR: {len(all_token_ids):,} Token!")
    return torch.tensor(all_token_ids, dtype=torch.long)

class CheckpointedProductionModel(nn.Module):
    """
    Tüm 28 katmanda gradient checkpointing uygulayarak VRAM'i ~6 GB seviyesinde tutar.
    """
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, cos, sin):
        B, T = input_ids.shape
        x = self.m.embed_tokens(input_ids)
        for layer in self.m.layers:
            x = checkpoint(layer, x, cos, sin, use_reentrant=False)
        x = self.m.norm(x)
        return self.m.lm_head(x)

def run_daytime_training():
    now = datetime.datetime.now()
    target_end = now.replace(hour=18, minute=0, second=0, microsecond=0)
    if target_end <= now:
        target_end += datetime.timedelta(days=1)

    total_seconds = (target_end - now).total_seconds()
    total_hours = total_seconds / 3600.0

    print("=" * 85)
    print(" ☀️ VALEROIS 1.78B: 4K CONTEXT & RECURRENT MEMORY GÜNDÜZ MARATONU")
    print("=" * 85)
    print(f"[*] Başlangıç Zamanı  : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[*] Bitiş Zamanı (Hedef): {target_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[*] Planlanan Süre    : {total_hours:.2f} Saat ({total_seconds:,.0f} saniye)")
    print("-" * 85)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    token_tensor = build_training_dataset(tokenizer)

    # Model Yükleme
    print(f"\n[*] Temel Üretim Modeli Yükleniyor: {BASE_CKPT_PATH}...")
    model = ValeroisQwenProductionModel(chunk_size=512).to(DEVICE)
    ckpt = torch.load(BASE_CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    # Dondurma ve Eğitim Yapılandırması
    for p in model.parameters():
        p.requires_grad = False

    trainable_params = []
    gcam_memory_params = 0
    top_layers_params = 0

    # 1. Tüm 28 Katmanın Valerois Hafıza Kapıları Eğitilir
    for i, block in enumerate(model.layers):
        for p in block.self_attn.w_necessity.parameters():
            p.requires_grad = True
            trainable_params.append(p)
            gcam_memory_params += p.numel()
        for p in block.self_attn.w_decay.parameters():
            p.requires_grad = True
            trainable_params.append(p)
            gcam_memory_params += p.numel()
        block.self_attn.inter_gate.requires_grad = True
        trainable_params.append(block.self_attn.inter_gate)
        gcam_memory_params += block.self_attn.inter_gate.numel()

        # 2. Son 4 Katmanın (24..27) Projeksiyonları ve Normları Açılır (Köprü)
        if i >= len(model.layers) - 4:
            for p in block.self_attn.q_proj.parameters():
                p.requires_grad = True
                trainable_params.append(p)
                top_layers_params += p.numel()
            for p in block.self_attn.k_proj.parameters():
                p.requires_grad = True
                trainable_params.append(p)
                top_layers_params += p.numel()
            for p in block.self_attn.v_proj.parameters():
                p.requires_grad = True
                trainable_params.append(p)
                top_layers_params += p.numel()
            for p in block.self_attn.o_proj.parameters():
                p.requires_grad = True
                trainable_params.append(p)
                top_layers_params += p.numel()
            for p in block.input_layernorm.parameters():
                p.requires_grad = True
                trainable_params.append(p)
                top_layers_params += p.numel()

    print("\n[*] Katman Eğitim Yapılandırması:")
    print(f"  [+] 28 Katmanın Valerois Hafıza Kapıları : {gcam_memory_params:,} parametre")
    print(f"  [+] Son 4 Katmanın Projeksiyon Köprüsü   : {top_layers_params:,} parametre")
    print(f"  [*] TOPLAM EĞİTİLEN PARAMETRE           : {sum(p.numel() for p in trainable_params):,} (~{sum(p.numel() for p in trainable_params)/1e6:.1f}M)")
    print(f"  [*] DONDURULMUŞ VE KORUNAN BİLGİ        : ~1.75 Milyar Parametre (%98.5)")

    train_model = CheckpointedProductionModel(model)

    # Rotary Embedding
    cfg = Qwen2Config.from_pretrained(MODEL_DIR)
    rotary = Qwen2RotaryEmbedding(cfg).to(DEVICE)

    # Hiperparametreler
    seq_len = 4096       # 4K Uzun Bağlam (8 Chunk)
    batch_size = 1
    grad_accum_steps = 4 # Efektif batch: 16,384 Token
    learning_rate = 2e-5

    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    step = 0
    total_tokens_processed = 0
    t0 = time.time()
    last_save_time = time.time()
    last_eval_time = time.time()

    print("\n[*] 🚀 4K CONTEXT GÜNDÜZ MARATONU BAŞLIYOR (Akşam 18:00'e Kadar)...")
    print(f"{'Adım':<8} | {'Loss':<8} | {'PPL':<8} | {'Hız':<12} | {'İşlenen Token':<16} | {'VRAM':<10} | {'Kalan':<10}")
    print("-" * 85)

    model.train()
    optimizer.zero_grad()

    # Precompute RoPE for 4K sequence length
    dummy_x = torch.zeros(1, seq_len, 1536, dtype=torch.bfloat16, device=DEVICE)
    pos_ids_4k = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
    cos_4k, sin_4k = rotary(dummy_x, pos_ids_4k)

    try:
        while True:
            current_time = time.time()
            if current_time >= target_end.timestamp():
                print("\n[!] ⏰ AKŞAM 18:00'E ULAŞILDI! Maraton Başarıyla Tamamlanıyor...")
                break

            step += 1
            t_s = time.time()

            max_idx = len(token_tensor) - seq_len - 2
            start_idx = random.randint(0, max_idx)
            bx = token_tensor[start_idx:start_idx + seq_len].unsqueeze(0).to(DEVICE)
            by = token_tensor[start_idx + 1:start_idx + seq_len + 1].unsqueeze(0).to(DEVICE)

            logits = train_model(bx, cos_4k, sin_4k)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), by.view(-1))
            loss = loss / grad_accum_steps

            loss.backward()

            if step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_tokens_processed += seq_len
            dt = time.time() - t_s
            speed = seq_len / max(dt, 0.001)

            if step % 20 == 0:
                rem_seconds = max(target_end.timestamp() - time.time(), 0)
                rem_h = rem_seconds / 3600.0
                vram_gb = torch.cuda.memory_allocated() / (1024**3)
                current_loss = loss.item() * grad_accum_steps
                ppl = math.exp(min(current_loss, 20))
                print(f"{step:<8d} | {current_loss:<8.4f} | {ppl:<8.2f} | {speed:>5.0f} tok/s | {total_tokens_processed:>10,} tok | {vram_gb:>6.2f} GB | {rem_h:>4.2f} saat", flush=True)

            # Her 30 dakikada bir tek dosya üzerine güvenli checkpoint
            if time.time() - last_save_time > 1800:
                print(f"[*] Periyodik Checkpoint Kaydediliyor ({LATEST_CKPT_PATH})...", flush=True)
                torch.save({"step": step, "model_state_dict": model.state_dict()}, LATEST_CKPT_PATH)
                last_save_time = time.time()

            # Her 60 dakikada bir canlı akıl yürütme testi
            if time.time() - last_eval_time > 3600:
                print("\n" + "-" * 70)
                print(f"[*] 🕒 SAATLİK CANLI DOĞRULAMA TESTİ (Adım {step})...")
                model.eval()
                test_prompt = "Question: If a car travels 150 km in 2 hours, what is its speed in km/h?\nAnswer:"
                test_ids = tokenizer.encode(test_prompt).ids
                cur_tokens = list(test_ids)
                for _ in range(25):
                    x_t = torch.tensor([cur_tokens], device=DEVICE)
                    T = x_t.shape[1]
                    pos_ids = torch.arange(T, device=DEVICE).unsqueeze(0)
                    dummy_t = torch.zeros(1, T, 1536, dtype=torch.bfloat16, device=DEVICE)
                    c_t, s_t = rotary(dummy_t, pos_ids)
                    with torch.no_grad():
                        l_t = model(x_t, cos=c_t, sin=s_t)
                        tok = torch.argmax(l_t[0, -1, :]).item()
                    if tok in [151643, 151645]:
                        break
                    cur_tokens.append(tok)
                live_text = tokenizer.decode(cur_tokens[len(test_ids):]).strip()
                print(f"  [Canlı GSM8K Çıktısı]: {live_text}")
                print("-" * 70 + "\n", flush=True)
                model.train()
                last_eval_time = time.time()

    except KeyboardInterrupt:
        print("\n[!] Manuel durduruldu.")

    # Final Model Kaydı
    print(f"\n[*] Final Model Kaydediliyor: {FINAL_CKPT_PATH}...", flush=True)
    torch.save({
        "step": step,
        "total_tokens": total_tokens_processed,
        "model_state_dict": model.state_dict(),
    }, FINAL_CKPT_PATH)
    print(f"[*] Model Başarıyla Kaydedildi! Toplam İşlenen: {total_tokens_processed:,} Token", flush=True)

    # Geçici latest checkpoint'i silerek disk alanını temizle
    if os.path.exists(LATEST_CKPT_PATH):
        try:
            os.remove(LATEST_CKPT_PATH)
            print(f"[*] Disk temizlendi: {LATEST_CKPT_PATH} kaldırıldı.")
        except Exception:
            pass

    # Akşam 18:00 Kapsamlı Benchmark Raporu
    print("\n" + "=" * 80)
    print(" 🔬 AKŞAM 18:00 RESMİ NİHAİ BENCHMARK RAPORU OLUŞTURULUYOR")
    print("=" * 80)
    model.eval()

    test_suite = [
        ("GSM8K Matematik", "Question: If a car travels 150 km in 2 hours, what is its speed in km/h?\nAnswer:"),
        ("Python Algoritma", "def is_prime(n: int) -> bool:\n    \"\"\"Check if n is prime.\"\"\"\n"),
        ("Çok Adımlı Mantık", "Question: John has 5 apples, gives 2 to Mary, then buys 3 more. How many apples does he have?\nAnswer:"),
        ("Kod Üretimi (ChatML)", "<|im_start|>user\nWrite a Python function to reverse a list in-place.<|im_end|>\n<|im_start|>assistant\n")
    ]

    report_lines = [
        "# Valerois-Qwen 1.78B Gündüz Maratonu Nihai Raporu (18:00)\n",
        f"- **Toplam Süre:** {(time.time() - t0)/3600.0:.2f} Saat",
        f"- **Toplam İşlenen Token:** {total_tokens_processed:,} Token",
        f"- **Bağlam Uzunluğu:** 4.096 Token (8x 512 Chunk)",
        f"- **Eğitilen Parametre:** {sum(p.numel() for p in trainable_params):,} (Valerois Kapıları + Son 4 Projeksiyon)",
        f"- **Final Checkpoint:** `{FINAL_CKPT_PATH}`\n",
        "## Akşam 18:00 Çıkarım Sonuçları:\n"
    ]

    for category, prompt in test_suite:
        input_ids = tokenizer.encode(prompt).ids
        cur_tokens = list(input_ids)
        for _ in range(50):
            x = torch.tensor([cur_tokens], device=DEVICE, dtype=torch.long)
            T = x.shape[1]
            pos_ids = torch.arange(T, device=DEVICE).unsqueeze(0)
            dummy_t = torch.zeros(1, T, 1536, dtype=torch.bfloat16, device=DEVICE)
            c_t, s_t = rotary(dummy_t, pos_ids)
            with torch.no_grad():
                l_t = model(x, cos=c_t, sin=s_t)
                tok = torch.argmax(l_t[0, -1, :]).item()
            if tok in [151643, 151645]:
                break
            cur_tokens.append(tok)

        gen_text = tokenizer.decode(cur_tokens[len(input_ids):]).strip()
        print(f"\n[{category}]: {prompt}")
        print(f"[Model Yanıtı]:\n{gen_text}")
        report_lines.append(f"### {category}\n**Prompt:** `{prompt.strip()}`\n**Yanıt:**\n```text\n{gen_text}\n```\n")

    with open("daytime_benchmark_report_1800.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("\n[*] Rapor diske yazıldı: daytime_benchmark_report_1800.md")
    print("=" * 80)

if __name__ == "__main__":
    run_daytime_training()

