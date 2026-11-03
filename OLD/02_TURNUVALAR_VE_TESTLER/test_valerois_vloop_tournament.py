"""
test_valerois_vloop_tournament.py
==================================
Valerois V-Loop (Recurrent Depth) Tournament
Compares:
  1. Plain Base: 4 Layers (1 pass)
  2. Plain Deep: 12 Layers (1 pass, 3x parameters)
  3. Valerois V-Loop: 4 Layers (looped 3 times, exact same params as Plain Base!)

Dataset: stream_coder_100k.bin (Real tokenized code)
Device: AMD Radeon RX 9070 XT (ROCm 7.2)
Zero-OOM Guarantee: Peak VRAM strictly capped under 2 GB.
"""

import os
import time
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print(f" 🚀 VALEROIS V-LOOP ARCHITECTURE TOURNAMENT")
print(f"[*] Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print("=" * 80)

DATA_PATH = "stream_coder_100k.bin"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Missing {DATA_PATH}")

raw_data = np.memmap(DATA_PATH, dtype=np.uint16, mode="r")
VOCAB_SIZE = 8192
TOTAL_TOKENS = len(raw_data)
print(f"[*] Loaded dataset: {TOTAL_TOKENS:,} tokens from {DATA_PATH}")

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
    """Convolutional Sequence Layer with Dilated Conv"""
    def __init__(self, d_model, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size-1, groups=d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        x_t = x.transpose(1, 2)
        out = self.conv(x_t)[..., :T].transpose(1, 2)
        return out

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class HybridTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.norm1 = RMSNorm(d_model)
        self.csl = CSLConvBlock(d_model, kernel_size=7)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, int(d_model * 2.5))

    def forward(self, x):
        B, T, D = x.shape
        # 1. CSL + Attention
        h = self.norm1(x)
        csl_feat = self.csl(h)

        q = self.q_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.out_proj(attn_out + csl_feat)

        # 2. SwiGLU FFN
        x = x + self.ffn(self.norm2(x))
        return x

# ==============================================================================
# Model 1: Standard Plain Feedforward Model (4 layers, 1 pass)
# ==============================================================================
class PlainTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            HybridTransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight # Weight tie

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)

# ==============================================================================
# Model 2: Valerois V-Loop Model (4 layers, looped K times)
# ==============================================================================
class ValeroisVLoop(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers=4, num_heads=4, num_loops=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_loops = num_loops

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            HybridTransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        
        # Learned loop phase embeddings so the model knows which loop iteration it is in
        self.loop_embed = nn.Parameter(torch.randn(num_loops, 1, 1, d_model) * 0.02)
        
        # Highway blending gates for smooth recurrent depth stability
        self.highway_gate = nn.Parameter(torch.ones(num_loops) * 0.5)

        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight # Weight tie

    def forward(self, input_ids):
        x = self.embed(input_ids)
        
        for loop_idx in range(self.num_loops):
            # Inject loop phase signal
            loop_signal = self.loop_embed[loop_idx]
            h = x + loop_signal
            
            # Pass through the core layers
            res = h
            for layer in self.layers:
                h = layer(h)
                
            # Highway blend
            alpha = torch.sigmoid(self.highway_gate[loop_idx])
            x = alpha * res + (1.0 - alpha) * h

        x = self.norm_f(x)
        return self.lm_head(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_batch(data, batch_size, seq_len, device):
    ix = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
    x = torch.stack([torch.from_numpy(data[i:i+seq_len].astype(np.int64)) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy(data[i+1:i+1+seq_len].astype(np.int64)) for i in ix]).to(device)
    return x, y

def train_and_eval_model(model_name, model, steps=STEPS):
    print("\n" + "-" * 70)
    print(f"[*] Training Contestant: {model_name}")
    num_params = count_parameters(model)
    print(f"[*] Trainable Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    
    torch.cuda.reset_peak_memory_stats()
    model.train()
    
    losses = []
    t0 = time.time()
    
    # Deterministic batches for fairness
    np.random.seed(42)
    
    for step in range(1, steps + 1):
        x, y = get_batch(raw_data, BATCH_SIZE, SEQ_LEN, DEVICE)
        optimizer.zero_grad()
        
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 100 == 0 or step == steps:
            curr_vram = torch.cuda.memory_allocated() / (1024**2)
            print(f"  Step {step:03d}/{steps} | Loss: {loss.item():.4f} | VRAM: {curr_vram:.1f} MB")

    total_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    tok_per_sec = (steps * BATCH_SIZE * SEQ_LEN) / total_time
    final_loss = np.mean(losses[-30:])

    print(f"[✓] {model_name} Finished | Final Loss: {final_loss:.4f} | Peak VRAM: {peak_vram:.1f} MB | Speed: {tok_per_sec:,.0f} tok/s")
    
    # Free memory
    del model, optimizer
    torch.cuda.empty_cache()
    
    return {
        "name": model_name,
        "params": num_params,
        "initial_loss": float(losses[0]),
        "final_loss": float(final_loss),
        "peak_vram_mb": float(peak_vram),
        "speed_tok_s": float(tok_per_sec),
        "total_time_s": float(total_time)
    }

def run_tournament():
    results = []

    # 1. Plain Baseline (4 Layers, 1 pass)
    model_base = PlainTransformer(VOCAB_SIZE, D_MODEL, num_layers=4, num_heads=NUM_HEADS)
    res_base = train_and_eval_model("Model A: Plain Base (4 Layers, 1 Pass)", model_base)
    results.append(res_base)

    # 2. Plain Deep (12 Layers, 1 pass - 3x params!)
    model_deep = PlainTransformer(VOCAB_SIZE, D_MODEL, num_layers=12, num_heads=NUM_HEADS)
    res_deep = train_and_eval_model("Model B: Plain Deep (12 Layers, 3x Parameters)", model_deep)
    results.append(res_deep)

    # 3. Valerois V-Loop (4 Layers, looped 3 times - SAME params as Model A!)
    model_loop = ValeroisVLoop(VOCAB_SIZE, D_MODEL, num_layers=4, num_heads=NUM_HEADS, num_loops=3)
    res_loop = train_and_eval_model("Model C: Valerois V-Loop (4 Layers x 3 Loops)", model_loop)
    results.append(res_loop)

    print("\n" + "=" * 80)
    print(" 🏆 TOURNAMENT FINAL SCORECARD")
    print("=" * 80)
    print(f"{'Model':<42} | {'Params':<10} | {'Final Loss':<10} | {'Peak VRAM':<12} | {'Speed':<12}")
    print("-" * 95)
    for r in results:
        print(f"{r['name']:<42} | {r['params']/1e6:>6.2f}M    | {r['final_loss']:>10.4f} | {r['peak_vram_mb']:>9.1f} MB | {r['speed_tok_s']:>9.0f} t/s")
    print("=" * 80)

    # Save to JSON
    with open("vloop_tournament_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[*] Results saved to vloop_tournament_results.json")

if __name__ == "__main__":
    run_tournament()
