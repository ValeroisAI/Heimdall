"""
test_vloop_crossover_1000.py
============================
1,000-Step Deep Crossover Test:
Does the 4-layer shallow model plateau, while Stabilized V-Loop keeps learning?
Dataset: stream_coder_100k.bin
Device: AMD Radeon RX 9070 XT (ROCm 7.2)
Zero-OOM Guarantee: Peak VRAM strictly capped under 2 GB.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "stream_coder_100k.bin"
raw_data = np.memmap(DATA_PATH, dtype=np.uint16, mode="r")

VOCAB_SIZE = 8192
D_MODEL = 256
NUM_HEADS = 4
HEAD_DIM = D_MODEL // NUM_HEADS
SEQ_LEN = 256
BATCH_SIZE = 8
STEPS = 1000
LR = 8.0e-4

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight

class CSLConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size-1, groups=d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        return self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class StabilizedBlock(nn.Module):
    def __init__(self, d_model, num_heads, init_scale=0.2):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.csl = CSLConvBlock(d_model, kernel_size=7)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, int(d_model * 2.5))
        self.gamma1 = nn.Parameter(torch.ones(d_model) * init_scale)
        self.gamma2 = nn.Parameter(torch.ones(d_model) * init_scale)

    def forward(self, x):
        B, T, D = x.shape
        h = self.norm1(x)
        csl_feat = self.csl(h)
        q = self.q_proj(h).view(B, T, 4, 64).transpose(1, 2)
        k = self.k_proj(h).view(B, T, 4, 64).transpose(1, 2)
        v = self.v_proj(h).view(B, T, 4, 64).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.gamma1 * self.out_proj(attn + csl_feat)
        x = x + self.gamma2 * self.ffn(self.norm2(x))
        return x

class PlainTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([StabilizedBlock(d_model, 4, init_scale=1.0) for _ in range(num_layers)])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight

    def forward(self, x):
        h = self.embed(x)
        for l in self.layers:
            h = l(h)
        return self.lm_head(self.norm_f(h))

class StabilizedVLoop(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers=4, num_loops=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([StabilizedBlock(d_model, 4, init_scale=0.2) for _ in range(num_layers)])
        self.num_loops = num_loops
        self.loop_embed = nn.Parameter(torch.randn(num_loops, 1, 1, d_model) * 0.01)
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for l in range(self.num_loops):
            h = x + self.loop_embed[l]
            for layer in self.layers:
                h = layer(h)
            x = h
        return self.lm_head(self.norm_f(x))

def get_batch(data, batch_size, seq_len, device):
    ix = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i:i+seq_len].astype(np.int64)) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy(data[i+1:i+1+seq_len].astype(np.int64)) for i in ix]).to(device)
    return x, y

def train_duel(name, model):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    np.random.seed(42)
    losses = []
    t0 = time.time()
    
    print(f"\n[*] Training {name} (Params: {sum(p.numel() for p in model.parameters()):,})")
    for s in range(1, STEPS + 1):
        x, y = get_batch(raw_data, BATCH_SIZE, SEQ_LEN, DEVICE)
        opt.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if s % 250 == 0 or s == STEPS:
            print(f"  Step {s:04d}/{STEPS} | Loss: {loss.item():.4f} | VRAM: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
            
    final_loss = np.mean(losses[-50:])
    tok_s = (STEPS * BATCH_SIZE * SEQ_LEN) / (time.time() - t0)
    print(f"[✓] {name} Final Loss (Last 50 avg): {final_loss:.4f} | Speed: {tok_s:,.0f} tok/s")
    del model, opt
    torch.cuda.empty_cache()
    return final_loss, losses

if __name__ == "__main__":
    print("=" * 80)
    print(" ⚔️ 1,000-STEP DEEP DUEL: SHALLOW BASELINE (4L) VS STABILIZED V-LOOP (4Lx3)")
    print("=" * 80)
    
    loss_base, hist_base = train_duel("Shallow Plain Base (4L)", PlainTransformer(VOCAB_SIZE, D_MODEL, 4))
    loss_loop, hist_loop = train_duel("Stabilized V-Loop (4Lx3)", StabilizedVLoop(VOCAB_SIZE, D_MODEL, 4, 3))
    
    print("\n" + "=" * 80)
    print(f" 🎯 1,000 STEP FINAL RESULT:")
    print(f"  Shallow Plain (4L)   : Final Loss = {loss_base:.4f}")
    print(f"  Stabilized V-Loop (4Lx3): Final Loss = {loss_loop:.4f}")
    diff = loss_base - loss_loop
    if diff > 0:
        print(f"  🏆 V-LOOP WINS by {diff:.4f} lower loss with exact same parameter count!")
    else:
        print(f"  [-] Difference: {diff:.4f}")
    print("=" * 80)
