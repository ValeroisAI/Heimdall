"""
test_valerois_vloop_stabilized.py
==================================
Testing Stabilized V-Loop with LayerScale and Persistent Input Memory
Compares:
  1. Plain Base: 4 Layers (1 pass)
  2. Naive V-Loop: 4 Layers x 3 Loops (Baseline loop from round 1)
  3. Stabilized V-Loop (LayerScale + Residual Highway)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print(f" 🚀 VALEROIS STABILIZED V-LOOP EXPERIMENT (LAYERSCALE + HIGHWAY)")
print(f"[*] Device: {DEVICE}")
print("=" * 80)

DATA_PATH = "stream_coder_100k.bin"
raw_data = np.memmap(DATA_PATH, dtype=np.uint16, mode="r")
VOCAB_SIZE = 8192
D_MODEL = 256
NUM_HEADS = 4
HEAD_DIM = D_MODEL // NUM_HEADS
SEQ_LEN = 256
BATCH_SIZE = 8
STEPS = 400
LR = 1.0e-3

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
        x_t = x.transpose(1, 2)
        return self.conv(x_t)[..., :T].transpose(1, 2)

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class StabilizedBlock(nn.Module):
    def __init__(self, d_model, num_heads, init_scale=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.csl = CSLConvBlock(d_model, kernel_size=7)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, int(d_model * 2.5))
        
        # LayerScale parameters: prevents deep recurrent gradient explosion
        self.gamma1 = nn.Parameter(torch.ones(d_model) * init_scale)
        self.gamma2 = nn.Parameter(torch.ones(d_model) * init_scale)

    def forward(self, x):
        B, T, D = x.shape
        h = self.norm1(x)
        csl_feat = self.csl(h)

        q = self.q_proj(h).view(B, T, 4, 64).transpose(1, 2)
        k = self.k_proj(h).view(B, T, 4, 64).transpose(1, 2)
        v = self.v_proj(h).view(B, T, 4, 64).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.gamma1 * self.out_proj(attn_out + csl_feat)

        x = x + self.gamma2 * self.ffn(self.norm2(x))
        return x

class StabilizedVLoop(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers=4, num_loops=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            StabilizedBlock(d_model, 4, init_scale=0.2) for _ in range(num_layers)
        ])
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
            x = h # Smooth progression via LayerScale inside layers
        x = self.norm_f(x)
        return self.lm_head(x)

def get_batch(data, batch_size, seq_len, device):
    ix = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i:i+seq_len].astype(np.int64)) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy(data[i+1:i+1+seq_len].astype(np.int64)) for i in ix]).to(device)
    return x, y

def run_test():
    model = StabilizedVLoop(VOCAB_SIZE, D_MODEL, num_layers=4, num_loops=3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    torch.cuda.reset_peak_memory_stats()
    np.random.seed(42)
    losses = []
    t0 = time.time()
    
    print(f"[*] Training Stabilized V-Loop (Parameters: {sum(p.numel() for p in model.parameters()):,})")
    for step in range(1, STEPS + 1):
        x, y = get_batch(raw_data, BATCH_SIZE, SEQ_LEN, DEVICE)
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        
        if step % 100 == 0 or step == STEPS:
            print(f"  Step {step:03d}/{STEPS} | Loss: {loss.item():.4f} | VRAM: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
            
    total_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    final_loss = np.mean(losses[-30:])
    print("\n" + "=" * 80)
    print(f"🎯 Stabilized V-Loop Finished:")
    print(f"  Initial Loss : {losses[0]:.4f}")
    print(f"  Step 100 Loss: {losses[99]:.4f}")
    print(f"  Final Loss   : {final_loss:.4f}")
    print(f"  Peak VRAM    : {peak_vram:.1f} MB")
    print(f"  Speed        : {(STEPS * BATCH_SIZE * SEQ_LEN) / total_time:,.0f} tok/s")
    print("=" * 80)

if __name__ == "__main__":
    run_test()
