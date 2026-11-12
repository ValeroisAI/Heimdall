"""
Sayısal çekirdekler: CUDA, ROCm (HIP) ve CPU'da aynı kod.

- Kapılı delta kuralı: PyTorch referansı (chunk-paralel, UT dönüşümü) her yerde çalışır.
  `flash-linear-attention` (Triton, CUDA ve ROCm) kuruluysa `prepare()` onu referansa karşı sınar ve
  sonuç doğruysa eğitimde/prefill'de onu kullanır.
- Dikkat: PyTorch SDPA (CUDA'da FlashAttention, ROCm'da AOTriton), pencere için blok-yerel SDPA.
"""

import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)


def causal_conv(x: torch.Tensor, weight: torch.Tensor, prefix: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Kanal başına nedensel conv. x: [B,T,C], weight: [C,K] (son sütun = mevcut token), prefix: [B,K-1,C]."""
    k, t = weight.size(-1), x.size(1)
    full = F.pad(x, (0, 0, k - 1, 0)) if prefix is None else torch.cat((prefix.to(x.dtype), x), dim=1)
    w = weight.to(x.dtype)
    y = full[:, k - 1:k - 1 + t] * w[:, k - 1]
    for j in range(1, k):
        y = y + full[:, k - 1 - j:k - 1 - j + t] * w[:, k - 1 - j]
    return y


def rope(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    """x: [B,H,T,D], positions: [T] ya da [B,T] mutlak konum. Yarım-döndürme düzeni."""
    d = x.size(-1)
    inv = 1.0 / (base ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = positions.to(torch.float32)[..., None] * inv
    if ang.dim() == 3:
        ang = ang[:, None]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).to(x.dtype)


def _local_attention(q, k, v, window: int) -> torch.Tensor:
    """Her W'lik blok kendisine ve bir önceki bloğa bakar: bellek ve hesap O(T·W)."""
    b, h, t, d = q.shape
    w = window
    pad = (-t) % w
    if pad:
        q, k, v = (F.pad(z, (0, 0, 0, pad)) for z in (q, k, v))
    n = (t + pad) // w
    qb, kb, vb = (z.view(b, h, n, w, d) for z in (q, k, v))
    kk = torch.cat((F.pad(kb, (0, 0, 0, 0, 1, 0))[:, :, :-1], kb), dim=3)
    vv = torch.cat((F.pad(vb, (0, 0, 0, 0, 1, 0))[:, :, :-1], vb), dim=3)
    i = torch.arange(w, device=q.device)[:, None]
    j = torch.arange(2 * w, device=q.device)[None, :]
    band = (j > i) & (j <= i + w)
    first = band & (j >= w)
    mask = torch.cat((first[None], band[None].expand(n - 1, w, 2 * w)), 0) if n > 1 else first[None]
    out = F.scaled_dot_product_attention(qb.reshape(b, h * n, w, d), kk.reshape(b, h * n, 2 * w, d),
                                         vv.reshape(b, h * n, 2 * w, d), attn_mask=mask.repeat(h, 1, 1))
    return out.reshape(b, h, n * w, d)[:, :, :t]


def causal_attention(q, k, v, window: int = 0) -> torch.Tensor:
    """[B,H,T,D] nedensel dikkat; window > 0 ise her sorgu kendisi dahil son `window` anahtara bakar."""
    if window and q.size(2) > window:
        return _local_attention(q, k, v, window)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


# --------------------------------------------------------------------------- kapılı delta kuralı
def _unit_lower_inverse(a: torch.Tensor) -> torch.Tensor:
    """(I + A)⁻¹, A kesin alt üçgen."""
    c = a.size(-1)
    eye = torch.eye(c, dtype=a.dtype, device=a.device)
    try:
        return torch.linalg.solve_triangular(eye + a, eye.expand_as(a).contiguous(), upper=False, unitriangular=True)
    except (RuntimeError, NotImplementedError):
        inv = -a.clone()
        for i in range(1, c):
            inv[..., i, :i] = inv[..., i, :i] + (inv[..., i, :, None] * inv[..., :, :i]).sum(-2)
        return inv + eye


def delta_rule_reference(q, k, v, g, beta, chunk_size: int = 64,
                         initial_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """S_t = α_t S_{t-1} + β_t k_t (v_t − α_t S_{t-1}ᵀ k_t)ᵀ,  o_t = S_tᵀ q_t    (α_t = exp g_t)

    q, k: [B,T,H,Dk] (k L2-normalize, q ölçekli), v: [B,T,H,Dv], g: [B,T,H] (≤ 0), beta: [B,T,H].
    Döndürür o: [B,T,H,Dv] ve son durum S: [B,H,Dk,Dv]; ikisi de FP32.
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v = (z.float().transpose(1, 2) for z in (q, k, v))
        g, beta = g.float().transpose(1, 2), beta.float().transpose(1, 2)
        b, h, t, dk = k.shape
        dv, c = v.size(-1), chunk_size
        pad = (-t) % c
        if pad:  # β = 0, g = 0 dolgusu durumu değiştirmez
            q, k, v = (F.pad(z, (0, 0, 0, pad)) for z in (q, k, v))
            g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
        n = (t + pad) // c
        q, k, v = (z.view(b, h, n, c, -1) for z in (q, k, v))
        g, beta = g.view(b, h, n, c).cumsum(-1), beta.view(b, h, n, c)
        decay = (g[..., :, None] - g[..., None, :]).tril().exp().tril()
        k_beta = k * beta[..., None]
        w_inv = _unit_lower_inverse((k_beta @ k.transpose(-1, -2) * decay).tril(-1))
        u = w_inv @ (v * beta[..., None])
        w = w_inv @ (k_beta * g.exp()[..., None])
        attn = q @ k.transpose(-1, -2) * decay
        q_decay = q * g.exp()[..., None]
        k_tail = k * (g[..., -1:] - g).exp()[..., None]
        chunk_decay = g[..., -1].exp()[..., None, None]
        state = initial_state.float() if initial_state is not None else q.new_zeros(b, h, dk, dv)
        outs = []
        for i in range(n):
            v_new = u[:, :, i] - w[:, :, i] @ state
            outs.append(q_decay[:, :, i] @ state + attn[:, :, i] @ v_new)
            state = state * chunk_decay[:, :, i] + k_tail[:, :, i].transpose(-1, -2) @ v_new
        o = torch.stack(outs, dim=2).view(b, h, n * c, dv)[:, :, :t]
    return o.transpose(1, 2), state


def delta_rule_step(q, k, v, g, beta, state):
    """Tek token. q, k: [B,H,Dk], v: [B,H,Dv], g, beta: [B,H], state: [B,H,Dk,Dv] FP32."""
    q, k, v, g, beta = (z.float() for z in (q, k, v, g, beta))
    state = state * g.exp()[..., None, None]
    delta = (v - torch.einsum("bhk,bhkv->bhv", k, state)) * beta[..., None]
    state = state + k[..., :, None] * delta[..., None, :]
    return torch.einsum("bhk,bhkv->bhv", q, state), state


_FLA = None  # doğrulanmış flash-linear-attention fonksiyonu (prepare() ayarlar)

# fla'nın gated delta rule durum çekirdekleri (ileri h, geri dh) NVIDIA dışı ve Intel dışı kartlarda yalnız 2-4 warp ve
# BV=32 dener. RX 9070 XT'de geri çekirdek ileriden ~12 kat yavaş ölçüldü (bench/fla_hiz.py). Bu kartlarda arama alanını
# genişletiriz (8/16 warp, BV 16/32/64): sonuç aynı kalır (yalnız iş dağılımı değişir), en hızlıyı Triton seçer.
_FLA_TUNE_KERNELS = ("chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64")


def fla_tune_wide(force: bool = False) -> str:
    """ROCm'da (ya da force) fla durum çekirdeklerinin autotune alanını genişlet. Ne yapıldığını döndürür."""
    if not (force or torch.version.hip):
        return "gerek yok (ROCm değil)"
    try:
        import triton
        from fla.ops.common import chunk_delta_h as mod
    except Exception as e:
        return f"atlandı ({type(e).__name__})"
    done = []
    for name in _FLA_TUNE_KERNELS:
        obj = getattr(mod, name, None)
        while obj is not None and not hasattr(obj, "configs"):     # @triton.heuristics → autotuner → jit
            obj = getattr(obj, "fn", None)
        if obj is None:
            continue
        stages = sorted({c.num_stages for c in obj.configs}) or [1]
        obj.configs = [triton.Config({"BV": bv}, num_warps=w, num_stages=st)
                       for w in (2, 4, 8, 16) for st in stages for bv in (16, 32, 64)]
        if hasattr(obj, "cache"):
            obj.cache = {}                                         # önceki seçimler eski alandan: yeniden ölç
        done.append(f"{name.split('_kernel_')[1]}: {len(obj.configs)} ayar")
    return "genişletildi: " + ", ".join(done) if done else "çekirdek bulunamadı (fla sürümü farklı)"


@torch.compiler.disable
def _fla_apply(q, k, v, g, beta, initial_state):
    dt = torch.bfloat16
    o, s = _FLA(q.to(dt), k.to(dt), v.to(dt), g.float(), beta.to(dt), scale=1.0,
                initial_state=None if initial_state is None else initial_state.float(), output_final_state=True)
    return o, s


def delta_rule(q, k, v, g, beta, chunk_size: int = 64, initial_state=None, backend: str = "auto"):
    if _FLA is not None and backend == "auto" and q.is_cuda:
        return _fla_apply(q, k, v, g, beta, initial_state)
    return delta_rule_reference(q, k, v, g, beta, chunk_size, initial_state)


def prepare(device: torch.device, backend: str = "auto") -> str:
    """Hızlı çekirdeği bir kez sına (torch.compile'dan ÖNCE çağır). Kullanılan arka ucu döndürür."""
    global _FLA
    _FLA = None
    if backend != "auto" or device.type != "cuda":
        return "torch"
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as cand
    except Exception:
        return "torch (flash-linear-attention kurulu değil)"
    try:
        gen = torch.Generator().manual_seed(0)
        b, t, h, d = 2, 150, 2, 64
        q = F.normalize(torch.randn(b, t, h, d, generator=gen), dim=-1) * d ** -0.5
        k = F.normalize(torch.randn(b, t, h, d, generator=gen), dim=-1)
        v = torch.randn(b, t, h, d, generator=gen)
        g = -torch.rand(b, t, h, generator=gen) * 0.2
        beta = torch.rand(b, t, h, generator=gen)
        s0 = torch.randn(b, h, d, d, generator=gen) * 0.1
        ref_o, ref_s = delta_rule_reference(q, k, v, g, beta, 64, s0)
        _FLA = cand
        tune = fla_tune_wide()
        o, s = _fla_apply(*(z.to(device) for z in (q, k, v, g, beta, s0)))
        err = max(((o.float().cpu() - ref_o).norm() / ref_o.norm()).item(),
                  ((s.float().cpu() - ref_s).norm() / ref_s.norm()).item())
        if err < 0.03:
            return f"flash-linear-attention (göreli hata {err:.4f}; autotune {tune})"
        warnings.warn(f"flash-linear-attention referansla uyuşmuyor (göreli hata {err:.3f}); PyTorch kullanılacak")
    except Exception as e:  # sürüm/arka uç uyumsuzluğu: güvenli yola dön
        warnings.warn(f"flash-linear-attention çalışmadı ({type(e).__name__}: {e}); PyTorch kullanılacak")
    _FLA = None
    return "torch"


def enable_tunableop(path) -> str:
    """ROCm/CUDA GEMM'leri için PyTorch TunableOp: her matris boyutunda hipBLASLt/rocBLAS algoritmalarını dener, en
    hızlısını seçer ve dosyaya yazar (sonraki koşular dosyadan okur, yeniden denemez). İlk adımlar yavaşlar."""
    if not torch.cuda.is_available():
        return "kapalı (GPU yok)"
    try:
        tun = torch.cuda.tunable
        tun.enable(True)
        tun.tuning_enable(True)
        tun.set_filename(str(path))
        return f"açık ({path})"
    except Exception as e:  # eski PyTorch
        return f"kullanılamadı ({type(e).__name__})"
