"""
Eğitim adımı profili: GPU süresi hangi çekirdeklere gidiyor? (MFU neden ~%15?)

    python bench/profil_egitim.py                        # temel, bit, 2048 token, 8 mikro batch
    python bench/profil_egitim.py --arch transformer     # karşılaştırma
    python bench/profil_egitim.py --no-compile           # derlemesiz (çekirdek adları daha okunur)

Isınmadan sonra birkaç adımı torch.profiler ile ölçer ve GPU süresini gruplar: matris çarpımı, dikkat, delta kuralı
(fla/Triton), eleman bazlı, kopya/tür çevirme, optimizer. En pahalı 25 çekirdeği de listeler.
"""

import argparse
import collections
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heimdall import kernels  # noqa: E402
from heimdall.bit import BitFlip  # noqa: E402
from heimdall.config import ARCHS, PRESETS, apply_arch  # noqa: E402
from heimdall.model import HeimdallLM  # noqa: E402
from heimdall.optim import build_optimizers  # noqa: E402

GROUPS = [   # (grup, ad parçaları) — ilk eşleşen kazanır
    ("dikkat (flash)", ("attn", "flash", "fmha", "sdpa", "aotriton")),
    ("delta kuralı (fla)", ("chunk_", "gated_delta", "fwd_kernel", "bwd_kernel", "recompute", "solve_tril", "wy_", "kkt")),
    ("matris çarpımı", ("gemm", "cijk", "matmul", "mm_", "_mm", "bmm", "wmma", "hipblaslt", "tensile", "aten::mm",
                        "aten::addmm", "aten::linear")),
    ("kopya / tür çevirme", ("copy", "cast", "convert", "to_copy", "cat", "transpose", "contiguous", "elementwise_kernel_manual")),
    ("indirgeme (norm, toplam)", ("reduce", "norm", "softmax", "sum", "mean", "max")),
    ("optimizer", ("adam", "foreach", "multi_tensor", "lerp", "quantile", "sort", "topk")),
    ("eleman bazlı / triton füzyon", ("triton_", "elementwise", "vectorized", "unrolled", "fused", "aten::mul",
                                      "aten::add", "aten::div", "aten::sub", "aten::where", "aten::exp")),
]


def group_of(name: str) -> str:
    n = name.lower()
    for g, keys in GROUPS:
        if any(k in n for k in keys):
            return g
    return "diğer"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="temel", choices=list(PRESETS))
    p.add_argument("--arch", default="bit", choices=ARCHS)
    p.add_argument("--bits", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--micro", type=int, default=8)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    p.add_argument("--out", default="runs/profil_egitim.json", help="Chrome/Perfetto izi (chrome://tracing)")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = apply_arch(PRESETS[a.preset].model, a.arch)
    if a.arch == "bit" and a.bits:
        from dataclasses import replace
        cfg = replace(cfg, bits=a.bits)
    print(f"delta çekirdeği: {kernels.prepare(dev, cfg.kernel)}")
    model = HeimdallLM(cfg).to(dev)
    opts = build_optimizers(model, 0.02, 2e-3)
    fwd = torch.compile(model) if (a.compile and dev.type == "cuda") else model
    amp = torch.autocast(dev.type, dtype=torch.bfloat16) if dev.type == "cuda" else nullcontext()
    x = torch.randint(0, cfg.vocab_size, (a.micro, a.seq_len + 1), device=dev)

    def step():
        with amp:
            loss = fwd(x[:, :-1], x[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if not getattr(q, "bit", 0)], 1.0)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)

    for _ in range(3):                                            # derleme + ısınma
        step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    acts = [torch.profiler.ProfilerActivity.CPU] + ([torch.profiler.ProfilerActivity.CUDA] if dev.type == "cuda" else [])
    with torch.profiler.profile(activities=acts) as prof:
        for _ in range(a.steps):
            step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(a.out)

    gpu = dev.type == "cuda"
    # Yalnız GPU çekirdekleri: bir PyTorch işleminin (aten::mm, autograd düğümü, Optimizer.step) süresi altındaki
    # çekirdeklerin toplamıdır, ikisini birden saymak süreyi iki katına çıkarır. ROCm bazı çekirdekleri ".kd" ekiyle
    # ikinci kez bildirir: ek atılıp tek sayılır.
    from torch.autograd import DeviceType
    agg = {}
    for e in prof.key_averages():
        if gpu:
            if getattr(e, "device_type", None) != DeviceType.CUDA:
                continue
            t = e.self_device_time_total / 1e3                                          # ms
        else:
            if not e.key.startswith("aten::"):
                continue
            t = e.self_cpu_time_total / 1e3
        k = e.key[:-3] if e.key.endswith(".kd") else e.key
        if t > 0 and t >= agg.get(k, (0, 0))[0]:
            agg[k] = (t, e.count)
    rows = [(k, t, c) for k, (t, c) in agg.items()]
    total = sum(r[1] for r in rows)
    wall = prof.profiler.self_cpu_time_total / 1e3 if not gpu else None
    by = collections.defaultdict(float)
    for k, t, _ in rows:
        by[group_of(k)] += t
    tok = a.steps * a.micro * a.seq_len
    print(f"\n{a.arch} | {sum(q.numel() for q in model.parameters()) / 1e6:.0f}M param | {a.steps} adım × {a.micro}×{a.seq_len} token"
          f" | {'GPU' if gpu else 'CPU'} süresi toplam {total:.0f} ms → {tok / (total / 1e3):,.0f} token/s (yalnız çekirdekler)")
    print("\nGRUP                              süre (ms)    pay")
    for g, t in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"{g:32} {t:10.1f}  {100 * t / total:5.1f}%")
    print(f"\nEN PAHALI 25 ÇEKİRDEK (ad kısaltıldı)")
    for k, t, c in sorted(rows, key=lambda r: -r[1])[:25]:
        print(f"{100 * t / total:5.1f}%  {t:8.1f} ms  ×{c:<5d} [{group_of(k)[:12]}]  {k[:90]}")
    print(f"\nİz dosyası: {a.out}  (https://ui.perfetto.dev ile açılır)")


if __name__ == "__main__":
    main()
