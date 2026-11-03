"""
egit.py — Kuzgun eğitimi (ROCm / CUDA / CPU)
============================================
Örnekler:
    # hızlı duman testi (~6M)
    python egit.py --preset deneme --data ../stream_coder_100k.bin --tokens 20e6

    # asıl eğitim (~100M, RX 9070 XT)
    python egit.py --preset temel --data ../stream_coder_100k.bin veri/fineweb_edu_8k.bin:0.5 \
                   sentetik_8k.bin:0.05 --tokens 2e9 --out kosular/temel

    # kaldığı yerden devam
    python egit.py --resume kosular/temel/son.pt

    # deneysel: gizli ağırlıksız üçlü eğitim + 2 product-key hafıza katmanı
    python egit.py --preset temel --uclu --hafiza-katmani 4 8 --data ../stream_coder_100k.bin --tokens 2e9

Tüm ayarlar için:  python egit.py -h
"""

import os

# Bellek parçalanmasını azalt + tüketici RDNA kartlarında AOTriton flash/mem-efficient dikkati aç.
# (torch içe aktarılmadan ÖNCE ayarlanmalı.)
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import torch

from kuzgun import PRESETS, KuzgunConfig, KuzgunLM
from kuzgun.cihaz import get_device, is_dml
from kuzgun.data import DataMixture, Prefetcher
from kuzgun.optim import build_optimizers, set_lr, wsd
from kuzgun.uclu import TernaryFlip, int8_available, set_int8, ternary_stats

HERE = Path(__file__).resolve().parent
DEFAULT_TOKENIZER = HERE.parent / "04_TOKENIZERLAR" / "valerois_tokenizer_8k.json"


def parse_args():
    p = argparse.ArgumentParser(description="Kuzgun eğitimi")
    p.add_argument("--preset", default="orta", choices=list(PRESETS))
    p.add_argument("--data", nargs="+", help="uint16 .bin dosyaları (yol[:ağırlık])")
    p.add_argument("--val-frac", type=float, default=0.005, help="her dosyanın sonundan ayrılan validation oranı")
    p.add_argument("--tokens", type=float, default=1e9, help="toplam eğitim tokeni")
    p.add_argument("--seq-len", type=int, help="hedef dizi uzunluğu (ön ayarı ezer)")
    p.add_argument("--micro-batch", type=int, help="tek ileri geçişteki dizi sayısı")
    p.add_argument("--batch-tokens", type=int, help="optimizer adımı başına token")
    p.add_argument("--no-curriculum", action="store_true", help="baştan tam uzunlukta eğit")
    p.add_argument("--lr-muon", type=float)
    p.add_argument("--lr-adam", type=float)
    p.add_argument("--adamw-only", action="store_true", help="Muon yerine yalnız AdamW")
    p.add_argument("--warmup", type=float, default=0.01)
    p.add_argument("--decay-start", type=float, default=0.75)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-ckpt", action="store_true", default=None, help="gradient checkpointing (bellek ↓, hız ~%%25 ↓)")
    p.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    p.add_argument("--mtp", action="store_true", help="çok-token tahmini (t+2) ek kaybı")
    p.add_argument("--attn", choices=["sdpa", "flex"], help="pencere dikkati arka ucu")
    p.add_argument("--window", type=int)
    p.add_argument("--archival", type=int, help="unutmayan hafıza kafası sayısı")
    p.add_argument("--d-model", type=int, help="model genişliği (ön ayarı ezer)")
    p.add_argument("--n-layers", type=int)
    p.add_argument("--n-heads", type=int)
    p.add_argument("--uclu", action="store_true", help="deneysel: gizli ağırlıksız üçlü (−1/0/+1) gizli matrisler")
    p.add_argument("--uclu-int8", action="store_true", help="üçlü katmanlarda int8 ileri geçiş (destek yoksa bf16)")
    p.add_argument("--flip-orani", type=float, default=2e-3, help="üçlü: adım başına çevrilen ağırlık oranı (tepe)")
    p.add_argument("--hafiza-katmani", type=int, nargs="*", help="product-key hafıza eklenecek blok indeksleri")
    p.add_argument("--hafiza-yuva", type=int, help="alt-anahtar sayısı n (yuva = n²), varsayılan 512")
    p.add_argument("--lr-hafiza", type=float, help="hafıza değer tablosu öğrenme hızı (varsayılan: AdamW lr)")
    p.add_argument("--compile", dest="compile", action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--compile-mode", default="default", help="default | max-autotune-no-cudagraphs")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default=None)
    p.add_argument("--device", default=None, help="cuda | dml (Windows DirectML) | cpu; boşsa otomatik")
    p.add_argument("--out", default="kosular/kuzgun")
    p.add_argument("--resume", help="checkpoint yolu (model + optimizer + adım)")
    p.add_argument("--init-from", help="yalnız ağırlıkları yükle (ince ayar)")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--sample-every", type=int, default=1000, help="örnek metin üretimi (0 = kapalı)")
    p.add_argument("--sample-prompt", default="def fibonacci(n):\n")
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--peak-tflops", type=float, default=None, help="MFU hesabı için kartın bf16 tepe TFLOPS'u")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def flops_per_token(cfg: KuzgunConfig, n_params_dense: int, seq_len: int) -> float:
    """Kaba tahmin: 6·N (matrisler) + pencere dikkati + delta hafıza (ileri+geri)."""
    window = min(cfg.window, seq_len)
    mixing = cfg.n_layers * 12 * cfg.inner * (window + cfg.chunk_size + cfg.head_dim)
    return 6 * n_params_dense + mixing


@torch.no_grad()
def evaluate(model, batches, micro_batch, device, amp):
    model.eval()
    out = {}
    for path, ids in batches:
        total, count = 0.0, 0
        for i in range(0, ids.size(0), micro_batch):
            chunk = ids[i:i + micro_batch].to(device)
            with amp:
                loss, _ = model(chunk[:, :-1], chunk[:, 1:])
            total += loss.item() * chunk.size(0)
            count += chunk.size(0)
        out[Path(path).stem] = total / max(count, 1)
    model.train()
    return out


@torch.no_grad()
def sample_text(model, tokenizer, prompt, device, amp, n_new=96, temperature=0.8, top_k=40):
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    with amp:
        cache = model.new_cache(1)
        logits, cache = model.prefill(ids, cache)
        out = []
        for _ in range(n_new):
            probs = torch.softmax(logits.float() / temperature, dim=-1)
            if top_k:
                v, i = probs.topk(top_k)
                nxt = i.gather(-1, torch.multinomial(v / v.sum(-1, keepdim=True), 1))
            else:
                nxt = torch.multinomial(probs, 1)
            out.append(nxt.item())
            logits, cache = model.step(nxt.view(1), cache)
    model.train()
    return prompt + tokenizer.decode(out)


def main() -> None:
    args = parse_args()
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        given = {a.split("=")[0] for a in sys.argv[1:] if a.startswith("--")}
        for k, v in ckpt["args"].items():  # komut satırında AÇIKÇA verilmeyen her ayar kayıttan gelir
            flag = "--" + k.replace("_", "-")
            if k not in ("resume", "device") and flag not in given and flag.replace("--", "--no-", 1) not in given:
                setattr(args, k, v)

    if not args.data:
        sys.exit("--data gerekli (ör. --data ../stream_coder_100k.bin)")

    preset = PRESETS[args.preset]
    cfg: KuzgunConfig = KuzgunConfig(**ckpt["config"]) if ckpt else preset.model
    overrides = {k: v for k, v in dict(window=args.window, archival_heads=args.archival, attn_backend=args.attn,
                                       d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                                       memory_n_sub=args.hafiza_yuva).items() if v is not None}
    if args.hafiza_katmani:
        overrides["memory_layers"] = tuple(args.hafiza_katmani)
    for flag, key in ((args.mtp, "mtp"), (args.uclu, "ternary"), (args.uclu_int8, "ternary_int8")):
        if flag:
            overrides[key] = True
    if overrides and not ckpt:
        cfg = replace(cfg, **overrides)
    seq_len = args.seq_len or preset.seq_len
    micro_batch = args.micro_batch or preset.micro_batch
    batch_tokens = args.batch_tokens or preset.batch_tokens
    grad_ckpt = preset.grad_ckpt if args.grad_ckpt is None else args.grad_ckpt
    lr_muon = args.lr_muon or preset.lr_muon
    lr_adam = args.lr_adam or preset.lr_adam

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    is_rocm = torch.version.hip is not None
    if is_dml(device):  # DirectML: torch.compile ve bf16 autocast yok → fp32, derlemesiz
        args.compile, args.dtype = False, "fp32"
    if args.dtype is None:
        args.dtype = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.dtype]
    amp = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else nullcontext()
    scaler = torch.amp.GradScaler(enabled=args.dtype == "fp16")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = KuzgunLM(cfg).to(device)
    if is_dml(device):
        model.float()  # üçlü ağırlıklar dahil her şey fp32
    model.grad_ckpt = grad_ckpt
    if cfg.ternary_int8 and not (device.type == "cuda" and int8_available(device)):
        print("Uyarı: torch._int_mm bu cihazda yok ya da hatalı; üçlü katmanlar bf16 matmul kullanacak.")
        set_int8(model, False)
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location="cpu", weights_only=False)["model"])
    optimizers = build_optimizers(model, lr_muon, lr_adam, use_muon=not args.adamw_only,
                                  flip_rate=args.flip_orani, lr_memory=args.lr_hafiza)
    flipper = next((o for o in optimizers if isinstance(o, TernaryFlip)), None)
    clip_params = [p for p in model.parameters()  # üçlü ve seyrek gradyanlar kendi optimizer'larında normalize
                   if not getattr(p, "ternary", False) and not getattr(p, "sparse_rows", False)]
    step, tokens_seen = 0, 0
    if ckpt:
        model.load_state_dict(ckpt["model"])
        for opt, state in zip(optimizers, ckpt["optim"]):
            opt.load_state_dict(state)
        step, tokens_seen = ckpt["step"], ckpt["tokens"]

    data = DataMixture(args.data, val_frac=args.val_frac, seed=args.seed + step)
    val_batches = data.val_batches(seq_len, args.eval_batches)
    total_steps = max(1, int(args.tokens // batch_tokens))
    n_dense = model.num_params(non_embedding=True) - model.memory_params() + model.lm_head.weight.numel()

    tokenizer = None
    if args.sample_every and Path(args.tokenizer).exists():
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(args.tokenizer)

    train_model = model
    if args.compile:
        torch._dynamo.config.cache_size_limit = 64
        train_model = torch.compile(model, mode=args.compile_mode, dynamic=False)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"model": cfg.to_dict(), "args": vars(args)}, indent=2))
    log_file = open(out_dir / "log.jsonl", "a")

    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else ("DirectML" if is_dml(device) else "CPU")
    print(f"Kuzgun | {model.num_params() / 1e6:.1f}M param ({n_dense / 1e6:.1f}M yoğun) | {cfg.n_layers} katman, "
          f"d={cfg.d_model}, {cfg.n_heads} kafa ({cfg.archival_heads} arşiv), pencere {cfg.window}")
    print(f"Cihaz: {dev_name}{' [ROCm ' + torch.version.hip + ']' if is_rocm else ''} | {args.dtype} | "
          f"compile={args.compile} | grad_ckpt={grad_ckpt} | MTP={cfg.mtp} | dikkat={cfg.attn_backend}")
    if cfg.ternary:
        n3 = ternary_stats(model)["uclu_param"]
        print(f"Üçlü (gizli ağırlıksız): {n3 / 1e6:.1f}M ağırlık −1/0/+1, flip oranı {args.flip_orani} | "
              f"int8={any(getattr(m, 'use_int8', False) for m in model.modules())}")
    if cfg.memory_layers:
        print(f"Hafıza katmanları {list(cfg.memory_layers)}: {cfg.memory_n_sub ** 2:,} yuva/katman, top-{cfg.memory_topk}"
              f" × {cfg.memory_heads} kafa | {model.memory_params() / 1e6:.1f}M param (token başına yalnız "
              f"{cfg.memory_heads * cfg.memory_topk} satır okunur)")
    print(f"Hedef T={seq_len}, mikro-batch {micro_batch}, adım başına {batch_tokens:,} token, "
          f"{total_steps:,} adım ({args.tokens / 1e9:.2f}B token) | Muon {lr_muon} / AdamW {lr_adam}")
    print(f"Veri: {data.describe()}", flush=True)

    def stage_seq_len(progress: float) -> int:
        if args.no_curriculum:
            return seq_len
        a, b = preset.seq_curriculum
        if progress < a:
            return max(64, seq_len // 4)
        if progress < b:
            return max(64, seq_len // 2)
        return seq_len

    cur_t = stage_seq_len(step / total_steps)
    loader = Prefetcher(data, micro_batch * (seq_len // cur_t), device).start(cur_t)

    def save(tag: str) -> None:
        state = {"config": cfg.to_dict(), "model": model.state_dict(), "optim": [o.state_dict() for o in optimizers],
                 "step": step, "tokens": tokens_seen, "args": vars(args)}
        tmp = out_dir / f".{tag}.tmp"
        torch.save(state, tmp)
        tmp.replace(out_dir / f"{tag}.pt")

    model.train()
    t_log, tok_log = time.time(), 0
    try:
        while step < total_steps:
            progress = step / total_steps
            t = stage_seq_len(progress)
            if t != cur_t:  # müfredat aşaması: aynı token/mikro-adım için batch büyür
                cur_t = t
                loader.batch = micro_batch * (seq_len // cur_t)
                loader.set_seq_len(cur_t)
                print(f"  [müfredat] dizi uzunluğu → {cur_t}", flush=True)
            mb = micro_batch * (seq_len // cur_t)
            accum = max(1, batch_tokens // (mb * cur_t))
            set_lr(optimizers, wsd(progress, args.warmup, args.decay_start))

            logs_acc = {}
            for _ in range(accum):
                x, y = loader.next()
                with amp:
                    loss, logs = train_model(x, y)
                scaler.scale(loss / accum).backward()
                for k, v in logs.items():
                    logs_acc[k] = logs_acc.get(k, 0.0) + v.float().item() / accum
            for opt in optimizers:
                scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(clip_params, args.grad_clip)
            for opt in optimizers:
                scaler.step(opt)
            scaler.update()
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            step += 1
            step_tokens = accum * mb * cur_t
            tokens_seen += step_tokens
            tok_log += step_tokens

            if step % args.log_every == 0 or step == total_steps:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.time() - t_log
                tps = tok_log / dt
                tflops = tps * flops_per_token(cfg, n_dense, cur_t) / 1e12
                mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                rec = {"step": step, "tokens": tokens_seen, "seq": cur_t, "lr_factor": wsd(progress, args.warmup, args.decay_start),
                       "grad_norm": float(gnorm), "tok_s": tps, "tflops": tflops, "mem_gb": mem, **logs_acc}
                extra = ""
                if flipper is not None:
                    rec["flip"] = flipper.last_flip_frac
                    rec["sifir_orani"] = ternary_stats(model)["sifir_orani"]
                    extra = f" | flip {100 * rec['flip']:.3f}% sıfır {100 * rec['sifir_orani']:.0f}%"
                mfu = f" | MFU {100 * tflops / args.peak_tflops:4.1f}%" if args.peak_tflops else ""
                print(f"adım {step:6d}/{total_steps} | loss {logs_acc.get('loss_lm', float('nan')):.4f}"
                      + (f" (mtp {logs_acc['loss_mtp']:.3f})" if "loss_mtp" in logs_acc else "")
                      + f" | T={cur_t} | {tps:,.0f} tok/s | ~{tflops:.1f} TFLOPS{mfu} | grad {float(gnorm):.2f}"
                      + f" | {mem:.1f} GB | {tokens_seen / 1e6:,.0f}M tok" + extra, flush=True)
                log_file.write(json.dumps(rec) + "\n")
                log_file.flush()
                t_log, tok_log = time.time(), 0

            if args.eval_every and (step % args.eval_every == 0 or step == total_steps):
                val = evaluate(model, val_batches, micro_batch, device, amp)
                print("  [val] " + " | ".join(f"{k}: {v:.4f} (ppl {math.exp(v):.1f})" for k, v in val.items()), flush=True)
                log_file.write(json.dumps({"step": step, "tokens": tokens_seen, "val": val}) + "\n")
                log_file.flush()
                t_log = time.time()
            if tokenizer is not None and args.sample_every and step % args.sample_every == 0:
                print("  [örnek]\n" + sample_text(model, tokenizer, args.sample_prompt, device, amp) + "\n", flush=True)
                t_log = time.time()
            if args.save_every and step % args.save_every == 0:
                save("son")
    except KeyboardInterrupt:
        print("\nDurduruldu — checkpoint kaydediliyor...", flush=True)
    finally:
        loader.stop = True
        save("son")
        print(f"Kaydedildi: {out_dir / 'son.pt'} (adım {step}, {tokens_seen / 1e6:,.0f}M token)", flush=True)


if __name__ == "__main__":
    main()
