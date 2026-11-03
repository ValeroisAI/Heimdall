"""
grand_architecture_tournament_9.py
====================================
9-Model Grand Architecture Tournament
Gerçek GPU (AMD Radeon RX 9070 XT, ROCm 7.2)
Gerçek kod verisi: stream_coder_100k.bin

GRUP A – Saf Mimariler:
  A1. Plain Softmax (baseline)
  A2. Differential Attention
  A3. GCAM-v2

GRUP B – İlk Hibrit Katman:
  B1. Differential Attention + CSL Conv
  B2. Differential Attention içinde GCAM-v2 (DiffGCAM)
  B3. Sparse Adaptive Attention

GRUP C – Tam Hibrit Kombinasyonlar:
  C1. Differential Attention + V-Loop
  C2. Sparse Adaptive + GCAM inter-chunk hafıza
  C3. Tam Hibrit (DiffAttn + CSL + Sparse)
"""

import os, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print(f" 🏆 9-MODEL GRAND ARCHITECTURE TOURNAMENT")
print(f"[*] Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print("=" * 80)

DATA_PATH = "stream_coder_100k.bin"
raw_data = np.memmap(DATA_PATH, dtype=np.uint16, mode="r")
VOCAB_SIZE = 8192
D = 256
H = 4
DH = D // H          # 64
SEQ = 256
BS  = 8
STEPS = 500
LR = 9e-4
CHUNK = 64           # GCAM chunk size
TOPK  = 32           # Sparse top-k

# ─────────────────────────────────────────────
# ORTAK BLOKLAR
# ─────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype) * self.w

class SwiGLU(nn.Module):
    def __init__(self, d, h=None):
        super().__init__()
        h = h or int(d * 2.5)
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(h, d, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class CSLConv(nn.Module):
    def __init__(self, d, k=7):
        super().__init__()
        self.conv = nn.Conv1d(d, d, k, padding=k-1, groups=d, bias=False)
    def forward(self, x):
        B, T, D = x.shape
        return self.conv(x.transpose(1,2))[..., :T].transpose(1,2)

# ─────────────────────────────────────────────
# A1: PLAIN SOFTMAX ATTENTION BLOCK
# ─────────────────────────────────────────────
class PlainAttnBlock(nn.Module):
    def __init__(self, d=D, nh=H, scale=1.0):
        super().__init__()
        self.nh, self.dh = nh, d//nh
        self.norm1 = RMSNorm(d)
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.op = nn.Linear(d, d, bias=False)
        self.norm2 = RMSNorm(d)
        self.ffn = SwiGLU(d)
        self.g1 = nn.Parameter(torch.ones(d) * scale)
        self.g2 = nn.Parameter(torch.ones(d) * scale)
    def forward(self, x):
        B, T, Dx = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B,T,self.nh,self.dh).transpose(1,2)
        k = k.view(B,T,self.nh,self.dh).transpose(1,2)
        v = v.view(B,T,self.nh,self.dh).transpose(1,2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1,2).contiguous().view(B,T,Dx)
        x = x + self.g1 * self.op(a)
        x = x + self.g2 * self.ffn(self.norm2(x))
        return x

# ─────────────────────────────────────────────
# A2: DIFFERENTIAL ATTENTION BLOCK
# Q/K her biri iki parçaya bölünür, iki softmax çıkarılır
# ─────────────────────────────────────────────
class DiffAttnBlock(nn.Module):
    def __init__(self, d=D, nh=H, scale=0.3):
        super().__init__()
        self.nh, self.dh = nh, d//nh
        self.norm1 = RMSNorm(d)
        # Q1,Q2,K1,K2,V: V tam boyut, Q ve K ikiye bölünmüş (dh/2 her biri)
        self.q_proj = nn.Linear(d, d, bias=False)    # [B,T,D] → 2 x [B,T,H,dh/2]
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.op = nn.Linear(d, d, bias=False)
        # λ: iki softmax arasındaki çıkarma katsayısı (öğrenilebilir)
        self.lam = nn.Parameter(torch.ones(nh) * 0.05)
        self.norm2 = RMSNorm(d)
        self.ffn = SwiGLU(d)
        self.g1 = nn.Parameter(torch.ones(d) * scale)
        self.g2 = nn.Parameter(torch.ones(d) * scale)
    def forward(self, x):
        B, T, Dx = x.shape
        h = self.norm1(x)
        half = self.dh // 2
        # Q → [B,H,T,dh] → q1:[B,H,T,half], q2:[B,H,T,half]
        q = self.q_proj(h).view(B, T, self.nh, self.dh).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.nh, self.dh).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.nh, self.dh).transpose(1, 2)
        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]
        scale = half ** -0.5
        # İki ayrı softmax (causal)
        s1 = torch.full((T,T), float('-inf'), device=x.device)
        s1 = torch.tril(torch.zeros(T,T, device=x.device))
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        sc1 = (q1 @ k1.transpose(-1,-2)) * scale
        sc1 = sc1.masked_fill(mask, float('-inf'))
        a1 = F.softmax(sc1, dim=-1)
        sc2 = (q2 @ k2.transpose(-1,-2)) * scale
        sc2 = sc2.masked_fill(mask, float('-inf'))
        a2 = F.softmax(sc2, dim=-1)
        lam = torch.sigmoid(self.lam).view(1, self.nh, 1, 1)
        # Differential: A = A1 - λ·A2
        a_diff = a1 - lam * a2
        out = (a_diff @ v).transpose(1,2).contiguous().view(B,T,Dx)
        x = x + self.g1 * self.op(out)
        x = x + self.g2 * self.ffn(self.norm2(x))
        return x

# ─────────────────────────────────────────────
# A3: GCAM-v2 BLOCK
# Intra-chunk: softmax attention | Inter-chunk: O(1) matrix state
# ─────────────────────────────────────────────
class GCAMv2Block(nn.Module):
    def __init__(self, d=D, nh=H, chunk=CHUNK, scale=0.3):
        super().__init__()
        self.nh, self.dh, self.chunk = nh, d//nh, chunk
        self.norm1 = RMSNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.op = nn.Linear(d, d, bias=False)
        self.w_nec  = nn.Linear(d, nh, bias=True)
        self.w_dec  = nn.Linear(d, nh, bias=True)
        nn.init.constant_(self.w_nec.bias, 2.0)
        nn.init.constant_(self.w_dec.bias, 2.0)
        self.inter_gate = nn.Parameter(torch.ones(nh) * 0.5)
        self.norm2 = RMSNorm(d)
        self.ffn = SwiGLU(d)
        self.g1 = nn.Parameter(torch.ones(d) * scale)
        self.g2 = nn.Parameter(torch.ones(d) * scale)
        self.register_buffer('cmask', torch.tril(torch.ones(chunk, chunk, dtype=torch.bool)), persistent=False)

    def forward(self, x):
        B, T, Dx = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        k = self.k_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        v = self.v_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        nec = torch.sigmoid(self.w_nec(h)).transpose(1,2).unsqueeze(-1)
        dec = torch.sigmoid(self.w_dec(h)).transpose(1,2).unsqueeze(-1)
        C = self.chunk
        nc = (T + C - 1) // C
        pad = nc * C - T
        if pad:
            q   = F.pad(q,   (0,0,0,pad))
            k   = F.pad(k,   (0,0,0,pad))
            v   = F.pad(v,   (0,0,0,pad))
            nec = F.pad(nec, (0,0,0,pad))
            dec = F.pad(dec, (0,0,0,pad), value=1.0)
        state = torch.zeros(B, self.nh, self.dh, self.dh, device=x.device, dtype=x.dtype)
        gate = torch.tanh(self.inter_gate).view(1,self.nh,1,1)
        chunks = []
        for i in range(nc):
            s, e = i*C, (i+1)*C
            qc, kc, vc = q[:,:,s:e], k[:,:,s:e], v[:,:,s:e]
            nc_, dc_ = nec[:,:,s:e], dec[:,:,s:e]
            sc = (qc @ kc.transpose(-1,-2)) / (self.dh**0.5)
            sc = sc.masked_fill(~self.cmask, -1e4)
            intra = F.softmax(sc, dim=-1) @ (vc * nc_)
            if i > 0:
                inter = (qc @ state) / (self.dh * self.chunk**0.5)
                out_c = intra + gate * inter
            else:
                out_c = intra
            chunks.append(out_c)
            kv = (kc * nc_).transpose(-1,-2) @ vc
            cd = dc_.mean(dim=2, keepdim=True)  # [B,H,1,1]
            state = state * cd + kv
        out = torch.cat(chunks, dim=2)[:,:,:T,:].transpose(1,2).contiguous().view(B,T,Dx)
        x = x + self.g1 * self.op(out)
        x = x + self.g2 * self.ffn(self.norm2(x))
        return x

# ─────────────────────────────────────────────
# B1: DiffAttn + CSL BLOCK (paralel iki akış)
# ─────────────────────────────────────────────
class DiffAttnCSLBlock(nn.Module):
    def __init__(self, d=D, nh=H, scale=0.3):
        super().__init__()
        self.nh, self.dh = nh, d//nh
        self.norm1 = RMSNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.op = nn.Linear(d, d, bias=False)
        self.lam = nn.Parameter(torch.ones(nh) * 0.05)
        self.csl = CSLConv(d, k=7)
        self.norm2 = RMSNorm(d)
        self.ffn = SwiGLU(d)
        self.g1 = nn.Parameter(torch.ones(d) * scale)
        self.g2 = nn.Parameter(torch.ones(d) * scale)

    def forward(self, x):
        B, T, Dx = x.shape
        h = self.norm1(x)
        half = self.dh // 2
        q = self.q_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        k = self.k_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        v = self.v_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        q1,q2 = q[...,:half], q[...,half:]
        k1,k2 = k[...,:half], k[...,half:]
        sc = half**-0.5
        mask = torch.triu(torch.ones(T,T, device=x.device, dtype=torch.bool), 1)
        s1 = (q1@k1.transpose(-1,-2))*sc; s1=s1.masked_fill(mask,float('-inf'))
        s2 = (q2@k2.transpose(-1,-2))*sc; s2=s2.masked_fill(mask,float('-inf'))
        lam = torch.sigmoid(self.lam).view(1,self.nh,1,1)
        a = F.softmax(s1,dim=-1) - lam * F.softmax(s2,dim=-1)
        attn_out = (a @ v).transpose(1,2).contiguous().view(B,T,Dx)
        csl_out = self.csl(h)
        x = x + self.g1 * self.op(attn_out + csl_out)
        x = x + self.g2 * self.ffn(self.norm2(x))
        return x

# ─────────────────────────────────────────────
# B2: DiffGCAM – GCAM intra-chunk'ında Differential Attention
# ─────────────────────────────────────────────
class DiffGCAMBlock(nn.Module):
    def __init__(self, d=D, nh=H, chunk=CHUNK, scale=0.3):
        super().__init__()
        self.nh, self.dh, self.chunk = nh, d//nh, chunk
        self.norm1 = RMSNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.op = nn.Linear(d, d, bias=False)
        self.w_nec = nn.Linear(d, nh, bias=True)
        self.w_dec = nn.Linear(d, nh, bias=True)
        nn.init.constant_(self.w_nec.bias, 2.0)
        nn.init.constant_(self.w_dec.bias, 2.0)
        self.lam = nn.Parameter(torch.ones(nh) * 0.05)
        self.inter_gate = nn.Parameter(torch.ones(nh) * 0.5)
        self.norm2 = RMSNorm(d)
        self.ffn = SwiGLU(d)
        self.g1 = nn.Parameter(torch.ones(d) * scale)
        self.g2 = nn.Parameter(torch.ones(d) * scale)
        self.register_buffer('cmask', torch.tril(torch.ones(chunk, chunk, dtype=torch.bool)), persistent=False)

    def forward(self, x):
        B, T, Dx = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        k = self.k_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        v = self.v_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        nec = torch.sigmoid(self.w_nec(h)).transpose(1,2).unsqueeze(-1)
        dec = torch.sigmoid(self.w_dec(h)).transpose(1,2).unsqueeze(-1)
        half = self.dh // 2
        C = self.chunk
        nc = (T + C - 1) // C
        pad = nc*C - T
        if pad:
            q   = F.pad(q,   (0,0,0,pad))
            k   = F.pad(k,   (0,0,0,pad))
            v   = F.pad(v,   (0,0,0,pad))
            nec = F.pad(nec, (0,0,0,pad))
            dec = F.pad(dec, (0,0,0,pad), value=1.0)
        state = torch.zeros(B, self.nh, self.dh, self.dh, device=x.device, dtype=x.dtype)
        gate = torch.tanh(self.inter_gate).view(1,self.nh,1,1)
        lam = torch.sigmoid(self.lam).view(1,self.nh,1,1)
        chunks = []
        mask_c = torch.triu(torch.ones(C, C, device=x.device, dtype=torch.bool), 1)
        for i in range(nc):
            s, e = i*C, (i+1)*C
            qc, kc, vc = q[:,:,s:e], k[:,:,s:e], v[:,:,s:e]
            nc_, dc_ = nec[:,:,s:e], dec[:,:,s:e]
            q1,q2 = qc[...,:half], qc[...,half:]
            k1,k2 = kc[...,:half], kc[...,half:]
            sc = half**-0.5
            s1 = (q1@k1.transpose(-1,-2))*sc; s1=s1.masked_fill(mask_c, float('-inf'))
            s2 = (q2@k2.transpose(-1,-2))*sc; s2=s2.masked_fill(mask_c, float('-inf'))
            a_diff = F.softmax(s1,dim=-1) - lam * F.softmax(s2,dim=-1)
            intra = a_diff @ (vc * nc_)
            if i > 0:
                inter = (qc @ state) / (self.dh * C**0.5)
                out_c = intra + gate * inter
            else:
                out_c = intra
            chunks.append(out_c)
            kv = (kc * nc_).transpose(-1,-2) @ vc
            cd = dc_.mean(dim=2, keepdim=True)  # [B,H,1,1]
            state = state * cd + kv
        out = torch.cat(chunks, dim=2)[:,:,:T,:].transpose(1,2).contiguous().view(B,T,Dx)
        x = x + self.g1 * self.op(out)
        x = x + self.g2 * self.ffn(self.norm2(x))
        return x

# ─────────────────────────────────────────────
# B3: SPARSE ADAPTIVE ATTENTION
# Her token için önem skoru → Top-K tam attention + geri kalanlar state'e
# ─────────────────────────────────────────────
class SparseAdaptiveBlock(nn.Module):
    def __init__(self, d=D, nh=H, topk=TOPK, scale=0.3):
        super().__init__()
        self.nh, self.dh, self.topk = nh, d//nh, topk
        self.norm1 = RMSNorm(d)
        self.imp = nn.Linear(d, 1, bias=True)   # Önem skoru
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.op = nn.Linear(d, d, bias=False)
        self.w_nec = nn.Linear(d, nh, bias=True)
        self.w_dec = nn.Linear(d, nh, bias=True)
        nn.init.constant_(self.w_dec.bias, 2.0)
        self.norm2 = RMSNorm(d)
        self.ffn = SwiGLU(d)
        self.g1 = nn.Parameter(torch.ones(d) * scale)
        self.g2 = nn.Parameter(torch.ones(d) * scale)

    def forward(self, x):
        B, T, Dx = x.shape
        h = self.norm1(x)
        # Önem skoru → hangi tokenlar "full attention" hak ediyor?
        imp = self.imp(h).squeeze(-1)                 # [B, T]
        topk = min(self.topk, T)
        _, top_idx = imp.topk(topk, dim=-1)           # [B, topk]
        top_idx, _ = top_idx.sort(dim=-1)

        q = self.q_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        k = self.k_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        v = self.v_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        nec = torch.sigmoid(self.w_nec(h)).transpose(1,2).unsqueeze(-1)  # [B,H,T,1]
        dec = torch.sigmoid(self.w_dec(h)).transpose(1,2).unsqueeze(-1)

        # Sparse: sadece top-K tokenlardan gelen K-V matrisini state'e yaz
        # Geri kalanlar: O(1) matrix accumulator
        idx_e = top_idx.unsqueeze(1).unsqueeze(-1).expand(B, self.nh, topk, self.dh)
        k_sel = torch.gather(k, 2, idx_e)   # [B, H, topk, dh]
        v_sel = torch.gather(v, 2, idx_e)
        nec_sel = torch.gather(nec, 2, top_idx.unsqueeze(1).unsqueeze(-1).expand(B,self.nh,topk,1))

        # Full softmax sadece seçili tokenlar için
        sc = (q @ k_sel.transpose(-1,-2)) / (self.dh**0.5)  # [B,H,T,topk]
        a  = F.softmax(sc, dim=-1)
        sparse_out = a @ (v_sel * nec_sel)             # [B,H,T,dh]

        # Non-selected: matrix state (toplam katkı)
        mask_all = torch.ones(B, T, dtype=torch.bool, device=x.device)
        kv_full = (k * nec).transpose(-1,-2) @ v       # [B,H,dh,dh]
        dec_mean = dec.mean(dim=2).squeeze(2)           # [B,H,1]
        # state olarak kv_full kullan (tek batch)
        state_out = (q @ kv_full) / (self.dh * T**0.5)

        # Birleştir
        alpha = torch.sigmoid(imp).unsqueeze(1).unsqueeze(-1)  # [B,1,T,1]
        out = sparse_out + (1.0 - alpha) * state_out
        out = out.transpose(1,2).contiguous().view(B,T,Dx)

        x = x + self.g1 * self.op(out)
        x = x + self.g2 * self.ffn(self.norm2(x))
        return x

# ─────────────────────────────────────────────
# C1: DiffAttn + V-Loop (4 katman × 3 döngü)
# ─────────────────────────────────────────────
class DiffAttnVLoopCore(nn.Module):
    """V-Loop içindeki tek blok (Differential Attention)"""
    def __init__(self, d=D, nh=H):
        super().__init__()
        self.nh, self.dh = nh, d//nh
        self.norm1 = RMSNorm(d)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.op = nn.Linear(d, d, bias=False)
        self.lam = nn.Parameter(torch.ones(nh) * 0.05)
        self.norm2 = RMSNorm(d)
        self.ffn = SwiGLU(d)
        self.g1 = nn.Parameter(torch.ones(d) * 0.2)
        self.g2 = nn.Parameter(torch.ones(d) * 0.2)
    def forward(self, x):
        B,T,Dx = x.shape
        h = self.norm1(x)
        half = self.dh//2
        q = self.q_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        k = self.k_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        v = self.v_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        q1,q2=q[...,:half],q[...,half:]
        k1,k2=k[...,:half],k[...,half:]
        sc=half**-0.5
        mask=torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool),1)
        s1=(q1@k1.transpose(-1,-2))*sc; s1=s1.masked_fill(mask,float('-inf'))
        s2=(q2@k2.transpose(-1,-2))*sc; s2=s2.masked_fill(mask,float('-inf'))
        lam=torch.sigmoid(self.lam).view(1,self.nh,1,1)
        a=F.softmax(s1,dim=-1)-lam*F.softmax(s2,dim=-1)
        out=(a@v).transpose(1,2).contiguous().view(B,T,Dx)
        x=x+self.g1*self.op(out)
        x=x+self.g2*self.ffn(self.norm2(x))
        return x

# ─────────────────────────────────────────────
# C2: Sparse Adaptive + GCAM inter-chunk state
# ─────────────────────────────────────────────
class SparseGCAMBlock(nn.Module):
    def __init__(self, d=D, nh=H, chunk=CHUNK, topk=TOPK, scale=0.3):
        super().__init__()
        self.nh, self.dh, self.chunk, self.topk = nh, d//nh, chunk, topk
        self.norm1 = RMSNorm(d)
        self.imp = nn.Linear(d, 1, bias=True)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.op = nn.Linear(d, d, bias=False)
        self.w_nec = nn.Linear(d, nh, bias=True)
        self.w_dec = nn.Linear(d, nh, bias=True)
        nn.init.constant_(self.w_nec.bias, 2.0)
        nn.init.constant_(self.w_dec.bias, 2.0)
        self.inter_gate = nn.Parameter(torch.ones(nh)*0.3)
        self.norm2 = RMSNorm(d)
        self.ffn = SwiGLU(d)
        self.g1 = nn.Parameter(torch.ones(d) * scale)
        self.g2 = nn.Parameter(torch.ones(d) * scale)
        self.register_buffer('cmask', torch.tril(torch.ones(chunk,chunk,dtype=torch.bool)), persistent=False)

    def forward(self, x):
        B, T, Dx = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        k = self.k_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        v = self.v_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        nec = torch.sigmoid(self.w_nec(h)).transpose(1,2).unsqueeze(-1)
        dec = torch.sigmoid(self.w_dec(h)).transpose(1,2).unsqueeze(-1)
        # Önem skoru her chunk için top-K belirliyor
        imp = torch.sigmoid(self.imp(h).squeeze(-1))  # [B,T]
        C = self.chunk
        nc = (T+C-1)//C
        pad = nc*C-T
        if pad:
            q   = F.pad(q,   (0,0,0,pad))
            k   = F.pad(k,   (0,0,0,pad))
            v   = F.pad(v,   (0,0,0,pad))
            nec = F.pad(nec, (0,0,0,pad))
            dec = F.pad(dec, (0,0,0,pad), value=1.0)
        state = torch.zeros(B, self.nh, self.dh, self.dh, device=x.device, dtype=x.dtype)
        gate = torch.tanh(self.inter_gate).view(1,self.nh,1,1)
        chunks = []
        mask_c = torch.triu(torch.ones(C,C,device=x.device,dtype=torch.bool),1)
        for i in range(nc):
            s, e = i*C, (i+1)*C
            qc,kc,vc = q[:,:,s:e], k[:,:,s:e], v[:,:,s:e]
            nc_,dc_ = nec[:,:,s:e], dec[:,:,s:e]
            # Intra-chunk: normal softmax
            sc2 = (qc @ kc.transpose(-1,-2))/(self.dh**0.5)
            sc2 = sc2.masked_fill(mask_c, float('-inf'))
            intra = F.softmax(sc2, dim=-1) @ (vc*nc_)
            # Inter-chunk: state
            if i > 0:
                inter = (qc @ state)/(self.dh*C**0.5)
                out_c = intra + gate*inter
            else:
                out_c = intra
            chunks.append(out_c)
            # Sadece önemli tokenlar (top-K) state günceller
            imp_c = imp[:,s:min(e,T)]   # [B, C_real]
            top_imp = imp_c.unsqueeze(1).unsqueeze(-1)[:,:,:vc.shape[2],:]  # broadcast guard
            kv = ((kc * nc_) * top_imp.clamp(0,1)).transpose(-1,-2) @ vc
            cd = dc_.mean(dim=2, keepdim=True)  # [B,H,1,1]
            state = state * cd + kv
        out = torch.cat(chunks,dim=2)[:,:,:T,:].transpose(1,2).contiguous().view(B,T,Dx)
        x = x + self.g1 * self.op(out)
        x = x + self.g2 * self.ffn(self.norm2(x))
        return x

# ─────────────────────────────────────────────
# C3: TAM HİBRİT — DiffAttn + CSL + Sparse gating
# ─────────────────────────────────────────────
class FullHybridBlock(nn.Module):
    def __init__(self, d=D, nh=H, topk=TOPK, scale=0.3):
        super().__init__()
        self.nh, self.dh, self.topk = nh, d//nh, topk
        self.norm1 = RMSNorm(d)
        self.imp = nn.Linear(d, 1, bias=True)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.op = nn.Linear(d, d, bias=False)
        self.lam = nn.Parameter(torch.ones(nh)*0.05)
        self.csl = CSLConv(d, k=7)
        self.norm2 = RMSNorm(d)
        self.ffn = SwiGLU(d)
        self.g1 = nn.Parameter(torch.ones(d)*scale)
        self.g2 = nn.Parameter(torch.ones(d)*scale)

    def forward(self, x):
        B,T,Dx = x.shape
        h = self.norm1(x)
        half = self.dh//2
        imp = torch.sigmoid(self.imp(h).squeeze(-1))   # [B,T]
        q = self.q_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        k = self.k_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        v = self.v_proj(h).view(B,T,self.nh,self.dh).transpose(1,2)
        q1,q2=q[...,:half],q[...,half:]
        k1,k2=k[...,:half],k[...,half:]
        sc=half**-0.5
        mask=torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool),1)
        s1=(q1@k1.transpose(-1,-2))*sc; s1=s1.masked_fill(mask,float('-inf'))
        s2=(q2@k2.transpose(-1,-2))*sc; s2=s2.masked_fill(mask,float('-inf'))
        lam=torch.sigmoid(self.lam).view(1,self.nh,1,1)
        a=F.softmax(s1,dim=-1)-lam*F.softmax(s2,dim=-1)
        # Sparse gating: attention ağırlığını önem skoru ile modüle et
        imp_w = imp.unsqueeze(1).unsqueeze(2)   # [B,1,1,T] → key dimension
        a = a * imp_w
        attn_out=(a@v).transpose(1,2).contiguous().view(B,T,Dx)
        csl_out=self.csl(h)
        x = x + self.g1 * self.op(attn_out + csl_out)
        x = x + self.g2 * self.ffn(self.norm2(x))
        return x

# ─────────────────────────────────────────────
# GENEL MODEL SARMALAYICI
# ─────────────────────────────────────────────
class LMModel(nn.Module):
    """Herhangi bir blok türünü alarak LM oluşturur."""
    def __init__(self, vocab_size, d, block_cls, n_layers=4, n_loops=1, **block_kwargs):
        super().__init__()
        self.n_loops = n_loops
        self.embed = nn.Embedding(vocab_size, d)
        self.layers = nn.ModuleList([block_cls(d=d, **block_kwargs) for _ in range(n_layers)])
        if n_loops > 1:
            self.loop_embed = nn.Parameter(torch.randn(n_loops,1,1,d)*0.01)
        self.norm_f = RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight

    def forward(self, ids):
        x = self.embed(ids)
        if self.n_loops == 1:
            for l in self.layers:
                x = l(x)
        else:
            for li in range(self.n_loops):
                h = x + self.loop_embed[li]
                for l in self.layers:
                    h = l(h)
                x = h
        return self.lm_head(self.norm_f(x))

# ─────────────────────────────────────────────
# EĞİTİM DÖNGÜSÜ
# ─────────────────────────────────────────────
def get_batch(data, bs, seq, dev):
    ix = np.random.randint(0, len(data)-seq-1, size=bs)
    x = torch.stack([torch.from_numpy(data[i:i+seq].astype(np.int64)) for i in ix]).to(dev)
    y = torch.stack([torch.from_numpy(data[i+1:i+1+seq].astype(np.int64)) for i in ix]).to(dev)
    return x, y

def run(name, model):
    n = sum(p.numel() for p in model.parameters())
    print(f"\n{'─'*70}")
    print(f"[*] {name}  |  Params: {n/1e6:.2f}M")
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    torch.cuda.reset_peak_memory_stats()
    np.random.seed(42)
    losses = []
    t0 = time.time()
    for s in range(1, STEPS+1):
        x, y = get_batch(raw_data, BS, SEQ, DEVICE)
        opt.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if s % 100 == 0 or s == STEPS:
            vram = torch.cuda.memory_allocated()/1024**2
            print(f"  Step {s:03d}/{STEPS} | Loss: {loss.item():.4f} | VRAM: {vram:.1f} MB")
    elapsed = time.time()-t0
    pk = torch.cuda.max_memory_allocated()/1024**2
    fl = np.mean(losses[-30:])
    spd = (STEPS*BS*SEQ)/elapsed
    print(f"[✓] Final Loss: {fl:.4f} | Peak VRAM: {pk:.1f} MB | {spd:,.0f} tok/s")
    del model, opt
    torch.cuda.empty_cache()
    return {"name":name,"params":n,"init_loss":losses[0],"final_loss":fl,"peak_vram_mb":pk,"tok_s":spd}

def safe_run(name, model):
    try:
        return run(name, model)
    except Exception as e:
        print(f"  [ERROR] {name} failed: {e}")
        try: del model
        except: pass
        torch.cuda.empty_cache()
        return {"name":name,"params":0,"init_loss":99.0,"final_loss":99.0,"peak_vram_mb":0,"tok_s":0,"error":str(e)}

# ─────────────────────────────────────────────
# TURNUVA
# ─────────────────────────────────────────────
if __name__ == "__main__":
    results = [
        {"name":"A1. Plain Softmax",         "params":5121280,"init_loss":4.1234,"final_loss":3.3256,"peak_vram_mb":586.8,"tok_s":76966},
        {"name":"A2. Differential Attention","params":5121280,"init_loss":3.4160,"final_loss":2.9446,"peak_vram_mb":685.9,"tok_s":70163},
    ]
    print("[*] A1 + A2 loaded from prior run. Resuming from A3...")

    # ── GRUP A: (A1 & A2 önceki çalıştırmadan yüklendi) ───────────────
    results.append(safe_run("A3. GCAM-v2",
        LMModel(VOCAB_SIZE, D, GCAMv2Block, n_layers=4, nh=H, chunk=CHUNK, scale=0.3)))

    # ── GRUP B: İLK HİBRİT KATMANLAR ──────────
    results.append(safe_run("B1. DiffAttn + CSL Conv",
        LMModel(VOCAB_SIZE, D, DiffAttnCSLBlock, n_layers=4, nh=H, scale=0.3)))

    results.append(safe_run("B2. DiffGCAM (Diff intra + GCAM inter)",
        LMModel(VOCAB_SIZE, D, DiffGCAMBlock, n_layers=4, nh=H, chunk=CHUNK, scale=0.3)))

    results.append(safe_run("B3. Sparse Adaptive Attention",
        LMModel(VOCAB_SIZE, D, SparseAdaptiveBlock, n_layers=4, nh=H, topk=TOPK, scale=0.3)))

    # ── GRUP C: TAM HİBRİT KOMBİNASYONLAR ─────
    results.append(safe_run("C1. DiffAttn + V-Loop (x3)",
        LMModel(VOCAB_SIZE, D, DiffAttnVLoopCore, n_layers=4, n_loops=3)))

    results.append(safe_run("C2. Sparse + GCAM inter-chunk",
        LMModel(VOCAB_SIZE, D, SparseGCAMBlock, n_layers=4, nh=H, chunk=CHUNK, topk=TOPK, scale=0.3)))

    results.append(safe_run("C3. Full Hybrid (DiffAttn + CSL + Sparse)",
        LMModel(VOCAB_SIZE, D, FullHybridBlock, n_layers=4, nh=H, topk=TOPK, scale=0.3)))

    # ─ FİNAL SKOR TABLOSU ─────────────────────
    print("\n" + "="*80)
    print(" 🏆 9-MODEL GRAND TOURNAMENT – FINAL SCOREBOARD")
    print("="*80)
    results_sorted = sorted(results, key=lambda r: r["final_loss"])
    print(f"{'#':<3} {'Model':<44} {'Params':>8} {'Init':>7} {'Final':>7} {'VRAM':>10} {'Tok/s':>10}")
    print("─"*95)
    for rank, r in enumerate(results_sorted, 1):
        medal = "🥇" if rank==1 else "🥈" if rank==2 else "🥉" if rank==3 else f" {rank}."
        err_mark = " ❌" if r.get("error") else ""
        print(f"{medal:<4} {r['name']:<44} {r['params']/1e6:>6.2f}M {r['init_loss']:>7.4f} {r['final_loss']:>7.4f} {r['peak_vram_mb']:>9.1f}MB {r['tok_s']:>9,.0f}{err_mark}")
    print("="*80)

    with open("grand_tournament_results.json", "w") as f:
        json.dump(results_sorted, f, indent=2)
    print("[*] Sonuçlar: grand_tournament_results.json")
