"""
Mímir — V-PDM v2: kapılı delta kuralı hafızası (softmax yok, O(1) durum).

Her kafa için sabit boyutlu bir durum matrisi S ∈ R^{Dk×Dv} tutulur:

    v_hat = α · S^T k           1. tahmin et
    δ     = v − v_hat           2. sürprizi bul
    S     = α · S + β · k δ^T   3. sadece sürprizi yaz
    o     = S^T q               4. oku

k L2-normalize, β ∈ (0, 1) (ya da negatif özdeğer seçeneğiyle (0, 2)). Aynı anahtara
yeni bir değer yazılınca eski değer silinir: "proje id 1 → 2 → 3" sorgusu 3 döner.

Eğitim: chunk-paralel (UT dönüşümü). Chunk içindeki bağımlılık birim alt-üçgen bir
sistemin çözümüne indirgenir (torch.linalg.solve_triangular); chunk'lar arasında
yalnız T/C adım kalır. Çıkarım: token başına O(Dk·Dv).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .csl import ShortConv
from .norms import RMSNorm


def gated_delta_recurrent(q, k, v, g, beta, initial_state: Optional[torch.Tensor] = None):
    """Token-token referans. q, k: [B,H,T,Dk], v: [B,H,T,Dv], g (log α ≤ 0), beta: [B,H,T]."""
    b, h, t, dk = k.shape
    state = initial_state if initial_state is not None else q.new_zeros(b, h, dk, v.shape[-1])
    out = []
    for i in range(t):
        state = state * g[:, :, i].exp()[..., None, None]
        pred = torch.einsum("bhk,bhkv->bhv", k[:, :, i], state)
        delta = (v[:, :, i] - pred) * beta[:, :, i, None]
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, :, i], delta)
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, :, i], state))
    return torch.stack(out, dim=2), state


def _unit_lower_inverse(a_strict: torch.Tensor) -> torch.Tensor:
    """(I + A)^{-1}, A kesin alt üçgen. solve_triangular yoksa (ör. DirectML) ileri yerine koyma."""
    c = a_strict.size(-1)
    eye = torch.eye(c, dtype=a_strict.dtype, device=a_strict.device)
    try:
        return torch.linalg.solve_triangular(eye + a_strict, eye.expand_as(a_strict), upper=False, unitriangular=True)
    except (RuntimeError, NotImplementedError):
        inv = -a_strict.clone()
        for i in range(1, c):
            inv[..., i, :i] = inv[..., i, :i] + (inv[..., i, :, None] * inv[..., :, :i]).sum(-2)
        return inv + eye


def gated_delta_chunk(q, k, v, g, beta, chunk_size: int = 64, initial_state: Optional[torch.Tensor] = None):
    """gated_delta_recurrent ile aynı sonuç; T herhangi bir uzunluk olabilir."""
    b, h, t, dk = k.shape
    dv = v.shape[-1]
    c = chunk_size
    pad = (-t) % c
    if pad:
        # β = 0 ve g = 0 dolgu durumu değiştirmez; çıktılar sonra kırpılır.
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
    n = (t + pad) // c

    def split(x):
        return x.reshape(b, h, n, c, *x.shape[3:])

    q, k, v, g, beta = split(q), split(k), split(v), split(g), split(beta)
    g_cum = g.cumsum(-1)
    decay = (g_cum[..., :, None] - g_cum[..., None, :]).tril().exp().tril()  # exp(g_i − g_j), i ≥ j
    k_beta = k * beta[..., None]
    w_inv = _unit_lower_inverse((k_beta @ k.transpose(-1, -2) * decay).tril(-1))
    u = w_inv @ (v * beta[..., None])
    w = w_inv @ (k_beta * g_cum.exp()[..., None])

    state = initial_state if initial_state is not None else q.new_zeros(b, h, dk, dv)
    causal = torch.ones(c, c, dtype=torch.bool, device=q.device).tril()
    out = []
    for i in range(n):
        q_i, k_i, g_i = q[:, :, i], k[:, :, i], g_cum[:, :, i]
        v_new = u[:, :, i] - w[:, :, i] @ state
        attn = (q_i @ k_i.transpose(-1, -2) * decay[:, :, i]).masked_fill(~causal, 0)
        out.append((q_i * g_i.exp()[..., None]) @ state + attn @ v_new)
        state = state * g_i[..., -1, None, None].exp() + (
            k_i * (g_i[..., -1:] - g_i).exp()[..., None]
        ).transpose(-1, -2) @ v_new
    o = torch.cat(out, dim=2)
    return o[:, :, :t], state


class Mimir(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim_k: int = 64, head_dim_v: int = 64,
                 conv_kernel: int = 4, chunk_size: int = 64, negative_eigen: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dk, self.dv = head_dim_k, head_dim_v
        self.chunk_size = chunk_size
        self.beta_scale = 2.0 if negative_eigen else 1.0
        self.q_proj = nn.Linear(dim, num_heads * head_dim_k, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * head_dim_k, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * head_dim_v, bias=False)
        self.q_conv = ShortConv(num_heads * head_dim_k, conv_kernel, activation="silu")
        self.k_conv = ShortConv(num_heads * head_dim_k, conv_kernel, activation="silu")
        self.v_conv = ShortConv(num_heads * head_dim_v, conv_kernel, activation="silu")
        self.a_proj = nn.Linear(dim, num_heads, bias=False)   # unutma (α) girdisi
        self.b_proj = nn.Linear(dim, num_heads, bias=True)    # yazma gücü (β)
        self.g_proj = nn.Linear(dim, num_heads * head_dim_v, bias=False)  # çıkış kapısı
        self.o_norm = RMSNorm(head_dim_v)
        self.o_proj = nn.Linear(num_heads * head_dim_v, dim, bias=False)

        # Gated DeltaNet başlatması: A ∈ [1, 16], dt ∈ [1e-3, 1e-1] -> başta yavaş unutma.
        self.A_log = nn.Parameter(torch.empty(num_heads).uniform_(1.0, 16.0).log())
        dt = torch.exp(torch.empty(num_heads).uniform_(math.log(1e-3), math.log(1e-1)))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # softplus^{-1}(dt)
        nn.init.zeros_(self.b_proj.bias)

    def _gates(self, x: torch.Tensor):
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias.float())
        beta = self.beta_scale * torch.sigmoid(self.b_proj(x).float())
        return g, beta  # [..., H]

    def _heads(self, x: torch.Tensor, d: int) -> torch.Tensor:
        return x.view(*x.shape[:-1], self.num_heads, d)

    def _output(self, o: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # o: [..., H, Dv]
        o = self.o_norm(o) * F.silu(self._heads(self.g_proj(x), self.dv))
        return self.o_proj(o.flatten(-2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_stateful(x, None)[0]

    def forward_stateful(self, x: torch.Tensor, state: Optional[dict]) -> Tuple[torch.Tensor, dict]:
        """state=None: sıfırdan. Aksi halde önceki parçanın durumu devam ettirilir (prefill)."""
        if state is None:
            state = self.init_state(x.size(0), x.device, x.dtype)
        q, sq = self.q_conv.forward_stateful(self.q_proj(x), state["q"])
        k, sk = self.k_conv.forward_stateful(self.k_proj(x), state["k"])
        v, sv = self.v_conv.forward_stateful(self.v_proj(x), state["v"])
        q = F.normalize(self._heads(q, self.dk).float(), dim=-1) / math.sqrt(self.dk)
        k = F.normalize(self._heads(k, self.dk).float(), dim=-1)
        v = self._heads(v, self.dv).float()
        g, beta = self._gates(x)
        tr = lambda z: z.transpose(1, 2)  # [B,T,H,...] -> [B,H,T,...]
        o, S = gated_delta_chunk(tr(q), tr(k), tr(v), tr(g), tr(beta), self.chunk_size, initial_state=state["S"])
        return self._output(tr(o).to(x.dtype), x), {"q": sq, "k": sk, "v": sv, "S": S}

    def init_state(self, batch: int, device, dtype) -> dict:
        return {
            "q": self.q_conv.init_state(batch, device, dtype),
            "k": self.k_conv.init_state(batch, device, dtype),
            "v": self.v_conv.init_state(batch, device, dtype),
            "S": torch.zeros(batch, self.num_heads, self.dk, self.dv, device=device, dtype=torch.float32),
        }

    def step(self, x: torch.Tensor, state: dict) -> Tuple[torch.Tensor, dict]:
        q, sq = self.q_conv.step(self.q_proj(x), state["q"])
        k, sk = self.k_conv.step(self.k_proj(x), state["k"])
        v, sv = self.v_conv.step(self.v_proj(x), state["v"])
        q = F.normalize(self._heads(q, self.dk).float(), dim=-1) / math.sqrt(self.dk)
        k = F.normalize(self._heads(k, self.dk).float(), dim=-1)
        v = self._heads(v, self.dv).float()
        g, beta = self._gates(x)
        S = state["S"] * g.exp()[..., None, None]
        delta = (v - torch.einsum("bhk,bhkv->bhv", k, S)) * beta[..., None]
        S = S + torch.einsum("bhk,bhv->bhkv", k, delta)
        o = torch.einsum("bhk,bhkv->bhv", q, S)
        return self._output(o.to(x.dtype), x), {"q": sq, "k": sk, "v": sv, "S": S}
