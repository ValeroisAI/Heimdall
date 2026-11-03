"""
train.py — Bifrost/Kuzgun dil modeli eğitimi (tek dosya, CPU/GPU)
=================================================================
Süre bütçeli (--minutes) eğitim: LR çizelgesi (WSD) geçen süreye göre ilerler, böylece
farklı mimariler AYNI DUVAR SAATİ bütçesinde adil karşılaştırılır (eğitim hızı dahil).

Örnek (CPU, 1 thread):
    python train.py --name kuzgun --layout KKKKKK --minutes 20 --threads 1
    python train.py --name transformer --layout AAAAAA --minutes 20 --threads 1

Çıktılar: results/runs/<name>.json (özet + eğriler), results/runs/<name>.pt (ağırlıklar, repoda değil)
"""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from bifrost import BifrostLM, ModelConfig
from bifrost.data import TokenData
from bifrost.optim import build_optimizers, set_lr, wsd_factor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


@torch.no_grad()
def evaluate(model, windows):
    model.eval()
    out = {}
    for name, ids in windows:
        losses = []
        for i in range(0, ids.size(0), 8):
            batch = ids[i:i + 8]
            logits = model(batch[:, :-1])
            losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1),
                                          reduction="sum").item())
        out[Path(name.split("@")[0]).stem] = sum(losses) / (ids.size(0) * (ids.size(1) - 1))
    model.train()
    return out


def grad_norms_by_block(model) -> list:
    norms = []
    for block in model.blocks:
        sq = sum(p.grad.float().pow(2).sum().item() for p in block.parameters() if p.grad is not None)
        norms.append(math.sqrt(sq))
    return norms


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--layout", default="KKKKKK")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--ffn-mult", type=float, default=8 / 3)
    p.add_argument("--csl", action="store_true", help="her bloğa Bifrost CSL v2 ekle")
    p.add_argument("--archival", type=int, default=0, help="Kuzgun'da unutmayan (α≡1) hafıza kafası sayısı")
    p.add_argument("--softcap", type=float, default=30.0)
    p.add_argument("--minutes", type=float, default=20.0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    p.add_argument("--lr-muon", type=float, default=0.02)
    p.add_argument("--lr-adam", type=float, default=3e-3)
    p.add_argument("--eval-every", type=float, default=150.0, help="saniye")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true", help="torch.compile (CPU'da ~1.7× hız, derleme birkaç dakika)")
    p.add_argument("--train", nargs="+", default=[str(DATA / "dev_code_8k_train.bin"),
                                                   str(DATA / "stream_coder_100k.bin") + "@0:0.99"])
    p.add_argument("--val", nargs="+", default=[str(DATA / "dev_code_8k_val.bin"),
                                                 str(DATA / "stream_coder_100k.bin") + "@0.99:1"])
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    cfg = ModelConfig(vocab_size=8192, dim=args.dim, layout=args.layout, csl=args.csl, ffn_mult=args.ffn_mult,
                      kuzgun_heads=args.heads, kuzgun_head_dim=args.head_dim, window=args.window,
                      mimir_heads=args.heads, attn_heads=args.heads, logit_softcap=args.softcap,
                      zero_init_out=True, archival_heads=args.archival)
    model = BifrostLM(cfg)
    optimizers = build_optimizers(model, args.optimizer, args.lr_muon, args.lr_adam)
    train_model = torch.compile(model) if args.compile else model  # değerlendirme derlenmemiş modelle
    data = TokenData(args.train, args.val, seed=args.seed)
    quick_val = data.val_windows(args.seq, 8)
    n_params = model.num_params()
    print(f"[{args.name}] {n_params / 1e6:.2f}M param | düzen {args.layout} | {args.optimizer} | "
          f"T={args.seq} B={args.batch} | {args.minutes} dk | {args.threads} thread | "
          f"eğitim verisi {data.train_tokens / 1e6:.1f}M token", flush=True)

    out_dir = ROOT / "results" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.compile:  # derlemeyi süre bütçesinin dışında yap
        tc = time.time()
        x, y = data.batch(args.batch, args.seq)
        F.cross_entropy(train_model(x).reshape(-1, cfg.vocab_size), y.reshape(-1)).backward()
        model.zero_grad(set_to_none=True)
        print(f"  derleme {time.time() - tc:.0f}s (bütçeye dahil değil)", flush=True)
    budget = args.minutes * 60
    t0 = time.time()
    next_eval = args.eval_every
    step = tokens = 0
    curve, grad_log, train_losses = [], [], []
    while True:
        elapsed = time.time() - t0
        if elapsed >= budget:
            break
        set_lr(optimizers, wsd_factor(elapsed / budget))
        x, y = data.batch(args.batch, args.seq)
        logits = train_model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        if step % 25 == 0:
            grad_log.append(grad_norms_by_block(model))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()
        step += 1
        tokens += x.numel()
        train_losses.append(loss.item())
        if time.time() - t0 >= next_eval:
            val = evaluate(model, quick_val)
            curve.append({"sec": round(time.time() - t0, 1), "step": step, "tokens": tokens,
                          "train": sum(train_losses[-50:]) / len(train_losses[-50:]), **{f"val_{k}": v for k, v in val.items()}})
            print(f"  {curve[-1]['sec']:6.0f}s | adım {step:5d} | {tokens / 1e6:5.2f}M tok | "
                  f"train {curve[-1]['train']:.3f} | " + " | ".join(f"val {k} {v:.3f}" for k, v in val.items()), flush=True)
            next_eval += args.eval_every

    train_time = time.time() - t0
    final = evaluate(model, data.val_windows(args.seq, 32))
    tail = grad_log[len(grad_log) // 2:] or grad_log
    grad_mean = [sum(col) / len(col) for col in zip(*tail)] if tail else []
    summary = {
        "name": args.name, "commit": git_commit(), "config": cfg.to_dict(), "args": vars(args),
        "params": n_params, "steps": step, "tokens": tokens, "train_seconds": round(train_time, 1),
        "tokens_per_s": tokens / train_time, "final_val": final, "curve": curve,
        "grad_norm_by_block": grad_mean,
    }
    (out_dir / f"{args.name}.json").write_text(json.dumps(summary, indent=2))
    torch.save({"config": cfg.to_dict(), "model": model.state_dict()}, out_dir / f"{args.name}.pt")
    print(f"[{args.name}] BİTTİ | {step} adım | {tokens / 1e6:.2f}M token | {tokens / train_time:,.0f} tok/s | "
          + " | ".join(f"val {k} {v:.3f}" for k, v in final.items())
          + " | blok grad normları " + " ".join(f"{g:.2f}" for g in grad_mean), flush=True)


if __name__ == "__main__":
    main()
