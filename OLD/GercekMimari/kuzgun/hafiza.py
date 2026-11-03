"""
hafiza.py — Product-key hafıza katmanı: milyarlık kapasite, küçük modelin hesabı
===============================================================================
N = n² yuva; her yuvada d boyutlu bir değer vektörü. Sorgu iki yarıya bölünür, her yarı n alt-anahtarla
karşılaştırılır; iki top-k listesinin kartezyen çarpımından en iyi k yuva seçilir (Lample ve ark., 2019).
Token başına hesap ~ n·d_q + k·d, yani N'den (neredeyse) bağımsız: parametre milyar, işlem küçük.

    y = W_o( Σ_i softmax(s)_i · V[idx_i]  ⊙  SiLU(W_g x) )

Gradyan tablosu yoğundur (N×d, gradyan biriktirmede sabit bellek; seyrek gradyan her okuma için bir satır
üretir ve biriktirme adımlarında büyür). SparseRowRMS yalnız bu adımda okunan satırları günceller ve satır
başına tek ölçek tutar (Adam'ın 2 tam kopyası yerine N sayı).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductKeyMemory(nn.Module):
    def __init__(self, d_model: int, n_sub: int = 512, heads: int = 4, topk: int = 32, d_query: int = 256) -> None:
        super().__init__()
        assert d_query % 2 == 0
        self.n, self.h, self.k, self.dq = n_sub, heads, topk, d_query
        self.q_proj = nn.Linear(d_model, heads * d_query, bias=False)
        self.q_norm = nn.LayerNorm(d_query, elementwise_affine=False)
        self.keys = nn.Parameter(torch.randn(heads, 2, n_sub, d_query // 2) / math.sqrt(d_query // 2))
        self.values = nn.EmbeddingBag(n_sub * n_sub, d_model, mode="sum")
        nn.init.normal_(self.values.weight, std=d_model ** -0.5)
        self.values.weight.sparse_rows = True   # optimizer seçimi için işaret (SparseRowRMS)
        self.g_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    @property
    def slots(self) -> int:
        return self.n * self.n

    def lookup(self, x: torch.Tensor):
        """x: [M, d] → (yuva indeksleri [M, h·k], ağırlıklar [M, h·k])."""
        m = x.size(0)
        q = self.q_norm(self.q_proj(x).view(m, self.h, self.dq).float())
        q1, q2 = q.chunk(2, dim=-1)                                                   # [M, h, dq/2]
        s1 = torch.einsum("mhd,hnd->mhn", q1, self.keys[:, 0].float())
        s2 = torch.einsum("mhd,hnd->mhn", q2, self.keys[:, 1].float())
        v1, i1 = s1.topk(self.k, dim=-1)
        v2, i2 = s2.topk(self.k, dim=-1)
        grid = (v1[..., :, None] + v2[..., None, :]).flatten(-2)                     # [M, h, k·k]
        best, pos = grid.topk(self.k, dim=-1)
        idx = i1.gather(-1, pos // self.k) * self.n + i2.gather(-1, pos % self.k)    # [M, h, k]
        w = torch.softmax(best, dim=-1)
        return idx.reshape(m, -1), (w / self.h).reshape(m, -1)

    @torch.compiler.disable  # EmbeddingBag derleyicinin dışında kalır
    def _read(self, idx, w):
        return self.values(idx, per_sample_weights=w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        idx, w = self.lookup(flat)
        mem = self._read(idx, w.to(self.values.weight.dtype)).to(x.dtype)
        y = self.o_proj(mem * F.silu(self.g_proj(flat)))
        return y.reshape(shape)


class SparseRowRMS(torch.optim.Optimizer):
    """Yalnız gradyanı sıfır olmayan satırları güncelleyen, satır başına RMS ölçekli SGD. Durum: satır başına 1 sayı."""

    def __init__(self, params, lr: float = 1e-2, beta: float = 0.99, eps: float = 1e-8) -> None:
        super().__init__(params, dict(lr=lr, beta=beta, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    g = p.grad.coalesce()
                    rows, vals = g.indices()[0], g.values()
                else:
                    rows = p.grad.ne(0).any(dim=1).nonzero().squeeze(1)
                    vals = p.grad[rows]
                st = self.state[p]
                v = st.get("v")
                if v is None:
                    v = st["v"] = torch.zeros(p.size(0), device=p.device)
                    st["t"] = 0
                st["t"] += 1
                v_rows = v[rows] * group["beta"] + (1 - group["beta"]) * vals.float().square().mean(dim=1)
                v[rows] = v_rows
                bias = 1 - group["beta"] ** st["t"]
                upd = vals.float() / ((v_rows / bias).sqrt()[:, None] + group["eps"])
                p.index_add_(0, rows, (-group["lr"] * upd).to(p.dtype))
