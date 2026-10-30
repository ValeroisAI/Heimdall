"""
Heimdall eğitimi (CUDA / ROCm / CPU).

    python -m heimdall.train --preset temel --data ../stream_coder_100k.bin --tokens 2e9 --out runs/temel
    python -m heimdall.train --preset temel --arch transformer ... --out runs/temel_tf     # karşılaştırma tabanı
    python -m heimdall.train --resume runs/temel/last.pt                                 # devam
"""

import os

os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")  # RDNA'da flash/mem-efficient SDPA

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch

from . import kernels
from .config import ARCHS, PRESETS, HeimdallConfig, apply_arch
from .bit import BitFlip, bit_stats, set_bit_kernel
from .data import Loader, Mixture
from .model import HeimdallLM
from .optim import build_optimizers, set_lr, wsd


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Heimdall eğitimi")
    p.add_argument("--preset", default="temel", choices=list(PRESETS))
    p.add_argument("--arch", default="hibrit", choices=ARCHS, help="hibrit (varsayılan) | transformer | sabit")
    p.add_argument("--data", nargs="+", help="uint16 .bin dosyaları (yol[:ağırlık])")
    p.add_argument("--val-frac", type=float, default=0.005)
    p.add_argument("--tokens", type=float, default=1e9)
    p.add_argument("--seq-len", type=int)
    p.add_argument("--micro-batch", type=int)
    p.add_argument("--batch-tokens", type=int)
    p.add_argument("--d-model", type=int)
    p.add_argument("--n-layers", type=int)
    p.add_argument("--layout", help="katman düzeni, ör. DDDA, DDA, A")
    p.add_argument("--attn-window", type=int)
    p.add_argument("--bits", type=int, choices=[0, 1, 2], help="gizli ağırlıksız: 1 = ikili (±1), 2 = üçlü (−1/0/+1)")
    p.add_argument("--bit-scope", choices=["all", "ffn", "mix"], help="--bits: hangi matrisler bit (ffn = yalnız SwiGLU)")
    p.add_argument("--bit-rank", type=int, help="--bits: bit matrislere yoğun düşük-rank düzeltici (ör. 16)")
    p.add_argument("--bit-kernel", choices=["off", "int8"], default="off",
                   help="--bits: Triton int8 GEMM (eğitim/prefill). Önce bench/bitkernel.py ile hızı ölç")
    p.add_argument("--flip-orani", type=float, default=5e-3, help="--bits: adım başına çevrilen ağırlık oranı (tepe)")
    p.add_argument("--mtp-depth", type=int, help="MTP modül sayısı (0 kapalı; 1-2 önerilir, küçük modelde önce kısa deney)")
    p.add_argument("--mtp-weight", type=float, help="MTP ek kaybının ağırlığı (varsayılan 0.3)")
    p.add_argument("--lr-muon", type=float)
    p.add_argument("--lr-adam", type=float)
    p.add_argument("--adamw-only", action="store_true")
    p.add_argument("--warmup", type=float, default=0.01)
    p.add_argument("--decay-start", type=float, default=0.75)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-ckpt", action="store_true", default=None)
    p.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    p.add_argument("--compile", dest="compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--kernel", choices=["auto", "torch"], default="auto")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="runs/heimdall")
    p.add_argument("--resume")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def build_config(args) -> HeimdallConfig:
    cfg = apply_arch(PRESETS[args.preset].model, args.arch)
    over = {k: v for k, v in dict(d_model=args.d_model, n_layers=args.n_layers, layout=args.layout,
                                  attn_window=args.attn_window, kernel=args.kernel, bits=args.bits,
                                  bit_scope=args.bit_scope, bit_rank=args.bit_rank, mtp_depth=args.mtp_depth,
                                  mtp_weight=args.mtp_weight).items() if v is not None}
    return replace(cfg, **over)


def flops_per_token(cfg: HeimdallConfig, n_matmul: int, t: int) -> float:
    """6·N + karıştırıcı (ileri+geri ≈ 3× ileri)."""
    f = 6 * n_matmul
    for kind in cfg.kinds:
        if kind == "A":
            span = min(cfg.attn_window or t, t)
            f += 3 * 2 * 2 * cfg.attn_heads * cfg.attn_head_dim * span / (1 if cfg.attn_window else 2)
        else:
            f += 3 * 2 * 3 * cfg.n_heads * cfg.head_dim * (cfg.chunk_size + cfg.head_dim)
    return f


@torch.no_grad()
def evaluate(model, val, micro, device, amp) -> dict:
    model.eval()
    out = {}
    for name, ids in val:
        tot = 0.0
        for i in range(0, ids.size(0), micro):
            chunk = ids[i:i + micro].to(device)
            with amp:
                tot += model(chunk[:, :-1], chunk[:, 1:]).item() * chunk.size(0)
        out[name] = tot / ids.size(0)
    model.train()
    return out


def main(argv=None) -> None:
    args = parse_args(argv)
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        given = {a.split("=")[0] for a in (argv or sys.argv[1:]) if a.startswith("--")}
        for k, v in ckpt["args"].items():
            if k not in ("resume", "device") and "--" + k.replace("_", "-") not in given:
                setattr(args, k, v)
    if not args.data:
        sys.exit("--data gerekli")

    preset = PRESETS[args.preset]
    cfg = HeimdallConfig(**ckpt["config"]) if ckpt else build_config(args)
    seq_len = args.seq_len or preset.seq_len
    micro = args.micro_batch or preset.micro_batch
    batch_tokens = args.batch_tokens or preset.batch_tokens
    accum = max(1, batch_tokens // (micro * seq_len))
    lr_muon, lr_adam = args.lr_muon or preset.lr_muon, args.lr_adam or preset.lr_adam
    grad_ckpt = preset.grad_ckpt if args.grad_ckpt is None else args.grad_ckpt

    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp = torch.autocast(device.type, dtype=torch.bfloat16) if bf16 else nullcontext()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    backend = kernels.prepare(device, cfg.kernel)

    model = HeimdallLM(cfg).to(device)
    model.grad_ckpt = grad_ckpt
    set_bit_kernel(model, args.bit_kernel)
    optimizers = build_optimizers(model, lr_muon, lr_adam, use_muon=not args.adamw_only, flip_rate=args.flip_orani)
    flipper = next((o for o in optimizers if isinstance(o, BitFlip)), None)
    clip_params = [p for p in model.parameters() if not getattr(p, "bit", 0)]   # bit ağırlıkları kendi optimizer'ında
    step, tokens = 0, 0
    if ckpt:
        model.load_state_dict(ckpt["model"])
        for opt, st in zip(optimizers, ckpt["optim"]):
            opt.load_state_dict(st)
        step, tokens = ckpt["step"], ckpt["tokens"]

    mix = Mixture(args.data, args.val_frac, seed=args.seed + step)
    val = mix.val_batches(seq_len, args.eval_batches)
    total = max(1, int(args.tokens // (accum * micro * seq_len)))
    n_matmul = model.active_params() + model.lm_head.weight.numel()
    train_model = torch.compile(model) if args.compile and device.type != "cpu" else model

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"model": cfg.to_dict(), "args": vars(args)}, indent=2))
    logf = open(out / "log.jsonl", "a")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    hip = f" ROCm {torch.version.hip}" if torch.version.hip else ""
    print(f"Heimdall [{args.arch}, düzen {cfg.layout}] {model.num_params() / 1e6:.1f}M param ({model.active_params() / 1e6:.0f}M aktif) | d={cfg.d_model} "
          f"L={cfg.n_layers} | {gpu}{hip} | {'bf16' if bf16 else 'fp32'} | delta çekirdeği: {backend}")
    print(f"T={seq_len} mikro={micro} birikim={accum} → adım başına {accum * micro * seq_len:,} token, {total:,} adım "
          f"| Muon {lr_muon} AdamW {lr_adam} | veri: {mix.describe()}", flush=True)

    def save() -> None:
        state = {"config": cfg.to_dict(), "model": model.state_dict(), "optim": [o.state_dict() for o in optimizers],
                 "step": step, "tokens": tokens, "args": vars(args)}
        torch.save(state, out / ".last.tmp")
        (out / ".last.tmp").replace(out / "last.pt")

    loader = Loader(mix, micro, seq_len, device)
    model.train()
    t0, tok0 = time.time(), 0
    bad = False
    try:
        while step < total:
            progress = step / total
            set_lr(optimizers, wsd(progress, args.warmup, args.decay_start))
            loss_acc = 0.0
            for _ in range(accum):
                x, y = loader.next()
                with amp:
                    loss = train_model(x, y)
                (loss / accum).backward()
                loss_acc += loss.detach().float() / accum
            if not math.isfinite(float(loss_acc)):  # eğitim bozuldu: bozuk ağırlığı kaydetme, süreci düşür (gece betiği son iyi kayıttan sürdürür)
                bad = True
                raise FloatingPointError(f"adım {step}: kayıp sonlu değil ({float(loss_acc)})")
            gnorm = torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip)
            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            tokens += accum * micro * seq_len
            tok0 += accum * micro * seq_len
            if step % args.log_every == 0 or step == total:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.time() - t0
                tps = tok0 / dt
                rec = {"step": step, "tokens": tokens, "loss": float(loss_acc), "grad_norm": float(gnorm), "tok_s": tps,
                       "tflops": tps * flops_per_token(cfg, n_matmul, seq_len) / 1e12,
                       "mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0}
                extra = ""
                if flipper is not None:
                    rec["flip"] = flipper.last_flip_frac
                    rec["sifir_orani"] = bit_stats(model)["sifir_orani"]
                    extra = f" | flip {100 * rec['flip']:.3f}% sıfır {100 * rec['sifir_orani']:.0f}%"
                print(f"adım {step:6d}/{total} | loss {rec['loss']:.4f} | {tps:,.0f} tok/s | ~{rec['tflops']:.1f} TFLOPS"
                      f" | grad {rec['grad_norm']:.2f} | {rec['mem_gb']:.1f} GB | {tokens / 1e6:,.0f}M tok" + extra, flush=True)
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                t0, tok0 = time.time(), 0
            if args.eval_every and (step % args.eval_every == 0 or step == total):
                v = evaluate(model, val, micro, device, amp)
                print("  [val] " + " | ".join(f"{k}: {x:.4f} (ppl {math.exp(x):.1f})" for k, x in v.items()), flush=True)
                logf.write(json.dumps({"step": step, "tokens": tokens, "val": v}) + "\n")
                logf.flush()
                t0 = time.time()
            if args.save_every and step % args.save_every == 0:
                save()
    except KeyboardInterrupt:
        print("\nDurduruldu, kaydediliyor...")
    finally:
        loader.stop = True
        if not bad:
            save()
        print(f"{'KAYDEDİLMEDİ (bozuk durum)' if bad else 'Kaydedildi'}: {out / 'last.pt'} (adım {step}, {tokens / 1e6:,.0f}M token)", flush=True)


if __name__ == "__main__":
    main()
