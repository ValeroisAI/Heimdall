"""
================================================================================
 🚀 BEYOND TRANSFORMER: RADİKAL FİKİRLER TURNUVASI (DUAL-STREAM & JUMP-MTP)
================================================================================
 Kullanıcının radikal fikirlerini standart Transformer/LLM düzenine karşı
 bilimsel ve deneysel olarak test eden turnuva:
 
 1. Standart Klasik LLM (Baseline):
    - Düz soldan sağa ardışık eğitim (0 -> 1 -> 2 -> ... -> t+1).
 
 2. Dual-Stream Middle-Split (Kullanıcının Ortadan Bölme Fikri):
    - Metni tam ortasından (%0-50 ve %50-100) bölerek her iki yarıyı 
      paralel çift-akışlı (Dual-Stream) eğitir. Aynı adımda iki farklı 
      bağlamı birden görür.
 
 3. Dual-Horizon Jump-MTP (Kullanıcının Geleceğe Atlama Fikri):
    - Tek ileri yayılımda model hem sıradaki tokenı (t+1) hem de 
      metnin ilerisindeki ufuk taşını (t+16) tahmin eder.
================================================================================
"""

import os
os.environ["OMP_NUM_THREADS"] = "6"

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cpu")

print("=" * 80)
print(" ⚔️ RADİKAL FİKİRLER TURNUVASI: DUAL-STREAM & JUMP-MTP VS KLASİK LLM")
print(" Cihaz: CPU (Ryzen 5 5600 6-Core - Arka plandaki GPU eğitimi etkilenmez)")
print("=" * 80)

# 1. GERÇEK VERİ SETİ (stream_coder_100k.bin)
DATA_FILE = "stream_coder_100k.bin"
data = np.memmap(DATA_FILE, dtype=np.uint16, mode="r")
VOCAB_SIZE = 8192
TRAIN_DATA = data[:-100000]
VAL_DATA = data[-100000:]

# Hip-parametreler
DIM = 192
N_HEADS = 4
N_LAYERS = 3
SEQ_LEN = 128
BATCH_SIZE = 4
STEPS = 150
LR = 1.2e-3


# 2. HİBRİT MİMARİ (AeroDrive CSL + Causal Attention)
class AeroDriveBlock(nn.Module):
    def __init__(self, hidden=192, kernel_size=15):
        super().__init__()
        self.conv = nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size-1, groups=hidden, bias=False)
        self.norm = nn.RMSNorm(hidden)
        self.mlp_up = nn.Linear(hidden, hidden * 2, bias=False)
        self.mlp_down = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        res = x
        x_norm = self.norm(x)
        x_c = x_norm.transpose(1, 2)
        x_c = self.conv(x_c)[..., :x.size(1)].transpose(1, 2)
        fused = self.mlp_up(x_c + x_norm)
        g, u = fused.chunk(2, dim=-1)
        return res + self.mlp_down(F.silu(g) * u)


class CausalAttentionBlock(nn.Module):
    def __init__(self, hidden=192, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.norm = nn.RMSNorm(hidden)
        self.qkv = nn.Linear(hidden, hidden * 3, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        res = x
        B, T, C = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).chunk(3, dim=-1)
        q = qkv[0].view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = qkv[1].view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = qkv[2].view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return res + self.out(attn)


class SharedBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, DIM)
        self.layer1 = AeroDriveBlock(DIM)
        self.layer2 = CausalAttentionBlock(DIM, N_HEADS)
        self.layer3 = AeroDriveBlock(DIM)
        self.final_norm = nn.RMSNorm(DIM)

    def forward(self, x):
        h = self.embed(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        return self.final_norm(h)


def get_random_chunk(source, length, count):
    ix = np.random.randint(0, len(source) - length - 10, size=count)
    return np.stack([source[i : i + length] for i in ix])


# 3. YARIŞMACILARI EĞİTME VE DEĞERLENDİRME FONKSİYONU
def run_contestant(arm_name):
    print("\n" + "=" * 70)
    print(f" 🥊 YARIŞMACI BAŞLIYOR: {arm_name}")
    print("=" * 70)

    torch.manual_seed(1337)
    np.random.seed(1337)

    backbone = SharedBackbone().to(device)
    head_primary = nn.Linear(DIM, VOCAB_SIZE, bias=False).to(device)
    head_jump = nn.Linear(DIM, VOCAB_SIZE, bias=False).to(device) if "Jump-MTP" in arm_name else None

    params = list(backbone.parameters()) + list(head_primary.parameters())
    if head_jump is not None:
        params += list(head_jump.parameters())

    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    
    total_tokens_seen = 0
    t0 = time.time()

    backbone.train()
    head_primary.train()
    if head_jump: head_jump.train()

    for step in range(1, STEPS + 1):
        optimizer.zero_grad()

        if arm_name == "1. Standart Klasik LLM (Sequential Baseline)":
            # Standart: BATCH_SIZE adet SEQ_LEN uzunluğunda blok (tek akış)
            raw = get_random_chunk(TRAIN_DATA, SEQ_LEN + 1, BATCH_SIZE)
            bx = torch.tensor(raw[:, :-1], dtype=torch.long, device=device)
            by = torch.tensor(raw[:, 1:], dtype=torch.long, device=device)

            h = backbone(bx)
            logits = head_primary(h)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), by.reshape(-1))
            total_tokens_seen += BATCH_SIZE * SEQ_LEN

        elif arm_name == "2. Dual-Stream Middle-Split (Kullanıcının Ortadan Bölme Fikri)":
            # Kullanıcının Fikri: Metin blokları tam ortasından ikiye bölünür
            # Yarı 1: Başlangıç (%0 - %50)
            # Yarı 2: Orta nokta  (%50 - %100)
            # Model aynı hesaplama sınırında iki yarıyı da paralel tüketir!
            half_len = SEQ_LEN // 2  # 64
            full_blocks = get_random_chunk(TRAIN_DATA, SEQ_LEN + 2, BATCH_SIZE)
            
            # Stream A: %0-50
            stream_a = full_blocks[:, :half_len + 1]
            bx_a = torch.tensor(stream_a[:, :-1], dtype=torch.long, device=device)
            by_a = torch.tensor(stream_a[:, 1:], dtype=torch.long, device=device)

            # Stream B: %50-100
            stream_b = full_blocks[:, half_len:half_len * 2 + 1]
            bx_b = torch.tensor(stream_b[:, :-1], dtype=torch.long, device=device)
            by_b = torch.tensor(stream_b[:, 1:], dtype=torch.long, device=device)

            # İki akış tek batch'te birleştirilerek paralel ileri yayılım yapılır
            bx_dual = torch.cat([bx_a, bx_b], dim=0) # [2*B, half_len]
            by_dual = torch.cat([by_a, by_b], dim=0)

            h_dual = backbone(bx_dual)
            logits_dual = head_primary(h_dual)
            loss = F.cross_entropy(logits_dual.reshape(-1, VOCAB_SIZE), by_dual.reshape(-1))
            total_tokens_seen += (BATCH_SIZE * 2) * half_len # Çift akışla taranan tokenlar

        elif arm_name == "3. Dual-Horizon Jump-MTP (Yerel t+1 & Uzak t+16 Tahmini)":
            # Model girdi dizisini okurken hem t+1 hem de t+16'yı tahmin eder
            JUMP_OFFSET = 16
            raw = get_random_chunk(TRAIN_DATA, SEQ_LEN + JUMP_OFFSET + 1, BATCH_SIZE)
            bx = torch.tensor(raw[:, :SEQ_LEN], dtype=torch.long, device=device)
            by_next = torch.tensor(raw[:, 1:SEQ_LEN+1], dtype=torch.long, device=device)
            by_jump = torch.tensor(raw[:, JUMP_OFFSET:SEQ_LEN+JUMP_OFFSET], dtype=torch.long, device=device)

            h = backbone(bx)
            logits_next = head_primary(h)
            logits_jump = head_jump(h)

            loss_next = F.cross_entropy(logits_next.reshape(-1, VOCAB_SIZE), by_next.reshape(-1))
            loss_jump = F.cross_entropy(logits_jump.reshape(-1, VOCAB_SIZE), by_jump.reshape(-1))
            loss = loss_next + 0.4 * loss_jump
            total_tokens_seen += BATCH_SIZE * SEQ_LEN

        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 1:
            print(f"   Adım {step:03d}/{STEPS} | Eğitim Kaybı: {loss.item():.4f}")

    train_duration = time.time() - t0
    speed_tok_sec = total_tokens_seen / max(train_duration, 0.001)

    # 4. TAMAMEN GÖRÜLMEMİŞ DOĞRULAMA TESTİ (Holdout Val Evaluation)
    # TÜM MODELLER AYNI STANDART SIRADAKİ KELİME TAHMİNİ (t+1) İLE YARIŞIR!
    backbone.eval()
    head_primary.eval()
    val_losses = []

    with torch.no_grad():
        for _ in range(25):
            raw_v = get_random_chunk(VAL_DATA, SEQ_LEN + 1, 4)
            vx = torch.tensor(raw_v[:, :-1], dtype=torch.long, device=device)
            vy = torch.tensor(raw_v[:, 1:], dtype=torch.long, device=device)
            h_v = backbone(vx)
            l_v = head_primary(h_v)
            v_loss = F.cross_entropy(l_v.reshape(-1, VOCAB_SIZE), vy.reshape(-1))
            val_losses.append(v_loss.item())

    mean_val_loss = float(np.mean(val_losses))
    perplexity = float(np.exp(mean_val_loss))

    print(f"  🏁 Bitti! Holdout Loss: {mean_val_loss:.4f} | Perplexity (PPL): {perplexity:.2f} | Hız: {speed_tok_sec:,.0f} tok/s")

    return {
        "name": arm_name,
        "val_loss": mean_val_loss,
        "ppl": perplexity,
        "speed": speed_tok_sec,
        "duration": train_duration
    }


def main():
    contenders = [
        "1. Standart Klasik LLM (Sequential Baseline)",
        "2. Dual-Stream Middle-Split (Kullanıcının Ortadan Bölme Fikri)",
        "3. Dual-Horizon Jump-MTP (Yerel t+1 & Uzak t+16 Tahmini)"
    ]

    results = []
    for c in contenders:
        res = run_contestant(c)
        results.append(res)

    print("\n" + "=" * 85)
    print(" 🏆 RADİKAL FİKİRLER TURNUVASI FİNAL TABLOSU (BİLİMSEL SONUÇLAR)")
    print("=" * 85)
    print(f"{'Mimari / Eğitim Yöntemi':<42} | {'Doğrulama Kaybı':<15} | {'PPL (Şaşkınlık)':<15} | {'Durum'}")
    print("-" * 85)

    best_loss = min(r["val_loss"] for r in results)
    for r in results:
        is_best = (r["val_loss"] == best_loss)
        tag = "👑 KAZANAN (ŞAMPİYON!)" if is_best else "  Referans"
        print(f"{r['name']:<42} | {r['val_loss']:>12.4f}   | {r['ppl']:>12.2f}    | {tag}")

    print("=" * 85)


if __name__ == "__main__":
    main()
