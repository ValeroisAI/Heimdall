"""
fla delta kuralı: ileri ve geri geçiş ayrı ayrı, Heimdall boyutlarında (temel: 6 kafa × 128, 8 × 2048 token).
Profilde geri çekirdeği (chunk_gated_delta_rule_bwd_kernel_dhu) ileriden ~9 kat yavaş görünüyordu; NVIDIA'da bu oran
~2-3. Bu betik oranı tek başına ölçer, ayrıca fla'nın sürümünü ve Triton'u yazar.

    python bench/fla_hiz.py                 # bf16
    python bench/fla_hiz.py --dtype float32
"""
import argparse
import time

import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--b", type=int, default=8)
    p.add_argument("--t", type=int, default=2048)
    p.add_argument("--h", type=int, default=6)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--n", type=int, default=20)
    a = p.parse_args()
    import fla
    import triton
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    dt, dev = getattr(torch, a.dtype), torch.device("cuda")
    print(f"fla {getattr(fla, '__version__', '?')} | triton {triton.__version__} | {torch.cuda.get_device_name(0)} | "
          f"{a.b}×{a.t} token, {a.h} kafa × {a.d}, {a.dtype}")
    q = (F.normalize(torch.randn(a.b, a.t, a.h, a.d, device=dev), dim=-1) * a.d ** -0.5).to(dt).requires_grad_()
    k = F.normalize(torch.randn(a.b, a.t, a.h, a.d, device=dev), dim=-1).to(dt).requires_grad_()
    v = torch.randn(a.b, a.t, a.h, a.d, device=dev, dtype=dt, requires_grad=True)
    g = (-torch.rand(a.b, a.t, a.h, device=dev) * 0.1).requires_grad_()
    beta = torch.rand(a.b, a.t, a.h, device=dev, dtype=dt, requires_grad=True)
    do = torch.randn(a.b, a.t, a.h, a.d, device=dev, dtype=dt)

    def fwd():
        return chunk_gated_delta_rule(q, k, v, g, beta, scale=1.0)[0]

    def timed(fn, n):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t) / n * 1e3

    def measure(tag):
        t_f = timed(fwd, a.n)
        o = fwd()
        grads = torch.autograd.grad(o, (q, k, v, g, beta), do, retain_graph=True)
        t_b = timed(lambda: torch.autograd.grad(o, (q, k, v, g, beta), do, retain_graph=True), a.n)
        print(f"{tag:22} ileri {t_f:6.2f} ms | geri {t_b:6.2f} ms | geri/ileri {t_b / t_f:4.1f}×", flush=True)
        return o.detach().float(), [x.float() for x in grads], t_f + t_b

    o0, g0, t0 = measure("fla varsayılan ayarlar")
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    from heimdall.kernels import fla_tune_wide
    print(f"autotune: {fla_tune_wide(force=True)} (ilk çağrıda bütün ayarlar denenir, biraz sürer)", flush=True)
    o1, g1, t1 = measure("geniş arama (8/16 warp)")
    rel = lambda x, y: float((x - y).norm() / y.norm().clamp_min(1e-12))
    err = max([rel(o1, o0)] + [rel(x, y) for x, y in zip(g1, g0)])
    print(f"sonuç farkı (çıkış ve tüm gradyanlar, en büyük göreli): {err:.1e} | ileri+geri {t0:.2f} → {t1:.2f} ms "
          f"({t0 / t1:.2f}× hız)")


if __name__ == "__main__":
    main()
