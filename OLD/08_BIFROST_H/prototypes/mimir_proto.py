"""
mimir_proto.py — Mímir (V-PDM v2) kapılı delta kuralı prototipi
===============================================================
V-PDM'in düzeltilmiş hali: tahmin k ile yapılır (S^T k), anahtar L2-normalize,
yazma yalnızca sürprizdir. Literatürdeki adı Gated DeltaNet.

    v_hat = alpha * S^T k          1. tahmin et
    delta = v - v_hat              2. sürprizi bul
    S     = alpha * S + beta * k delta^T   3. sadece sürprizi yaz
    o     = S^T q                  4. oku

İki uygulama:
  * recurrent(): token-token referans (doğruluk ölçütü)
  * chunked():   chunk-paralel eğitim yolu. Chunk içi bağımlılık UT dönüşümü ile
                 tek bir birim-alt-üçgen çözüme indirgenir (solve_triangular);
                 chunk'lar arası yalnız T/C adım kalır.

Kullanım (CPU yeterli):
    python 08_BIFROST_H/prototypes/mimir_proto.py
"""

import time

import torch
import torch.nn.functional as F


def recurrent(q, k, v, g, beta):
    """q, k: [B,H,T,Dk] (L2-normalize), v: [B,H,T,Dv], g: [B,H,T] log-alpha (<=0), beta: [B,H,T]."""
    B, H, T, Dk = k.shape
    S = q.new_zeros(B, H, Dk, v.shape[-1])
    out = []
    for t in range(T):
        S = S * g[:, :, t].exp()[..., None, None]
        pred = torch.einsum("bhk,bhkv->bhv", k[:, :, t], S)
        delta = (v[:, :, t] - pred) * beta[:, :, t, None]
        S = S + torch.einsum("bhk,bhv->bhkv", k[:, :, t], delta)
        out.append(torch.einsum("bhk,bhkv->bhv", q[:, :, t], S))
    return torch.stack(out, dim=2), S


def chunked(q, k, v, g, beta, chunk_size=64):
    """recurrent() ile aynı sonuç; T, chunk_size'ın katı olmalı (prototip)."""
    B, H, T, Dk = k.shape
    C = chunk_size
    N = T // C

    def split(x):
        return x.reshape(B, H, N, C, *x.shape[3:])

    q, k, v, g, beta = split(q), split(k), split(v), split(g), split(beta)
    g_cum = g.cumsum(-1)
    decay = (g_cum[..., :, None] - g_cum[..., None, :]).tril().exp().tril()  # exp(g_i - g_j), i >= j
    k_beta = k * beta[..., None]

    # (I + A) W = I, A kesin alt üçgen: chunk içi delta etkileşimleri
    A = (k_beta @ k.transpose(-1, -2) * decay).tril(-1)
    eye = torch.eye(C, dtype=q.dtype, device=q.device)
    W = torch.linalg.solve_triangular(eye + A, eye.expand_as(A), upper=False, unitriangular=True)
    u = W @ (v * beta[..., None])
    w = W @ (k_beta * g_cum.exp()[..., None])

    S = q.new_zeros(B, H, Dk, v.shape[-1])
    causal = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    out = []
    for n in range(N):
        q_n, k_n, g_n = q[:, :, n], k[:, :, n], g_cum[:, :, n]
        v_new = u[:, :, n] - w[:, :, n] @ S
        attn = (q_n @ k_n.transpose(-1, -2) * decay[:, :, n]).masked_fill(~causal, 0)
        out.append((q_n * g_n.exp()[..., None]) @ S + attn @ v_new)
        S = S * g_n[..., -1, None, None].exp() + (
            k_n * (g_n[..., -1:] - g_n).exp()[..., None]
        ).transpose(-1, -2) @ v_new
    return torch.cat(out, dim=2), S


def random_inputs(B, H, T, D, requires_grad=False):
    q = F.normalize(torch.randn(B, H, T, D), dim=-1)
    k = F.normalize(torch.randn(B, H, T, D), dim=-1)
    v = torch.randn(B, H, T, D)
    if requires_grad:
        q, k, v = (x.requires_grad_() for x in (q, k, v))
    g = F.logsigmoid(torch.randn(B, H, T) + 3.0)
    beta = torch.sigmoid(torch.randn(B, H, T))
    return q, k, v, g, beta


def main() -> None:
    torch.manual_seed(0)

    print("1) Doğruluk: chunk-paralel == token-token referans")
    q, k, v, g, beta = random_inputs(2, 4, 512, 64)
    o_ref, s_ref = recurrent(q, k, v, g, beta)
    o_chk, s_chk = chunked(q, k, v, g, beta)
    print(f"   ileri : max|Δo| = {(o_ref - o_chk).abs().max():.2e}, max|ΔS| = {(s_ref - s_chk).abs().max():.2e}")
    grads = []
    for fn in (recurrent, chunked):
        torch.manual_seed(0)  # iki uygulamaya da aynı girdiler
        q, k, v, g, beta = random_inputs(2, 4, 256, 64, requires_grad=True)
        fn(q, k, v, g, beta)[0].square().sum().backward()
        grads.append([x.grad for x in (q, k, v)])
    diffs = ", ".join(f"{(a - b).abs().max():.1e}" for a, b in zip(*grads))
    print(f"   geri  : max|Δgrad| (q, k, v) = {diffs}")

    print("2) Hız: ileri + geri (CPU)")
    for T in (512, 2048):
        timings = {}
        for name, fn in (("token-döngü", recurrent), ("chunk-paralel", chunked)):
            q, k, v, g, beta = random_inputs(4, 4, T, 64, requires_grad=True)
            start = time.perf_counter()
            fn(q, k, v, g, beta)[0].sum().backward()
            timings[name] = time.perf_counter() - start
        ratio = timings["token-döngü"] / timings["chunk-paralel"]
        print(f"   T={T:5d}: token-döngü {timings['token-döngü'] * 1e3:6.0f} ms | "
              f"chunk-paralel {timings['chunk-paralel'] * 1e3:5.0f} ms | {ratio:.1f}x")

    print("3) 'Proje ID' testi: aynı anahtara 1, 2, 3 yazılıp anahtarla sorgulanır")
    key = F.normalize(torch.randn(64), dim=0)
    values = torch.eye(3, 64) * 5.0
    s_delta = torch.zeros(64, 64)
    s_add = torch.zeros(64, 64)
    for value in values:
        s_delta = s_delta + torch.outer(key, value - key @ s_delta)  # beta = 1 delta kuralı
        s_add = s_add + torch.outer(key, value)                      # birikimli (GCAM/GLA tarzı)
    for name, S in (("delta kuralı (Mímir)", s_delta), ("toplamsal (GCAM tarzı)", s_add)):
        read = key @ S
        sims = ", ".join(f"{F.cosine_similarity(read, value, dim=0).item():.2f}" for value in values)
        print(f"   {name:24s} -> cos(v1, v2, v3) = {sims}")


if __name__ == "__main__":
    main()
