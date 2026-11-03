"""
degerlendir.py — Eğitilmiş Kuzgun modelini ölç
=============================================
    python degerlendir.py --ckpt kosular/temel/son.pt --data ../stream_coder_100k.bin

1) Validation loss / perplexity (dosyanın son --val-frac kısmı, eğitimde görülmeyen bölge)
2) Pozisyona göre loss: model uzak bağlamı kullanıyor mu? (loss pozisyon arttıkça düşmeli)
3) PROJE_ID iğnesi:   PROJE_ID = 4721  …[d token gerçek kod]…  assert PROJE_ID == 4721
   ve güncelleme testi:  PROJE_ID = 1111 … PROJE_ID = 4721 … assert PROJE_ID == 4721 (son değer)
   Mesafe 1M tokene kadar çıkabilir: istem parça parça işlenir, bellek sabit kalır.
"""

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from kuzgun.cihaz import get_device
from uret import DEFAULT_TOKENIZER, load


@torch.no_grad()
def val_loss(model, data, seq_len, n, device, amp):
    starts = np.linspace(0, len(data) - seq_len - 2, n).astype(np.int64)
    total = 0.0
    for s in starts:
        ids = torch.from_numpy(data[s:s + seq_len + 1].astype(np.int64))[None].to(device)
        with amp:
            loss, _ = model(ids[:, :-1], ids[:, 1:])
        total += loss.item()
    return total / n


@torch.no_grad()
def loss_by_position(model, data, length, n, device, amp):
    starts = np.linspace(0, len(data) - length - 2, n).astype(np.int64)
    acc = torch.zeros(length)
    for s in starts:
        ids = torch.from_numpy(data[s:s + length + 1].astype(np.int64))[None].to(device)
        with amp:
            logits = model(ids[:, :-1])
        acc += F.cross_entropy(logits[0].float(), ids[0, 1:], reduction="none").cpu()
    acc /= n
    edges = [0, 16, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768]
    return [{"from": a, "to": b, "loss": acc[a:b].mean().item()} for a, b in zip(edges, edges[1:]) if b <= length]


@torch.no_grad()
def value_logprob(model, prefix_ids, value_ids, device, amp):
    """prefix'i sabit bellekle işle, sonra değer tokenlerini zorla ve log-olasılık + tam isabet döndür."""
    with amp:
        cache = model.new_cache(1)
        logits, cache = model.prefill(torch.tensor([prefix_ids], device=device), cache)
        total, exact = 0.0, True
        for tok in value_ids:
            logp = torch.log_softmax(logits.float(), dim=-1)[0]
            total += logp[tok].item()
            exact &= int(logp.argmax()) == tok
            logits, cache = model.step(torch.tensor([tok], device=device), cache)
    return total, exact


def needle_test(model, tok, data, distance, n, device, amp, rng):
    q = tok.encode("\nassert PROJE_ID == ").ids
    res = {"distance": distance, "gain_nats": 0.0, "exact": 0, "exact_latest": 0, "n": n}
    for _ in range(n):
        v_old, v = str(rng.integers(1000, 9999)), str(rng.integers(1000, 9999))
        start = int(rng.integers(0, len(data) - distance - 1))
        filler = data[start:start + distance].astype(np.int64).tolist()
        needle = tok.encode(f"PROJE_ID = {v}\n").ids
        value = tok.encode(v).ids
        lp_with, hit = value_logprob(model, needle + filler + q, value, device, amp)
        lp_without, _ = value_logprob(model, filler + q, value, device, amp)
        half = len(filler) // 2
        latest = tok.encode(f"PROJE_ID = {v_old}\n").ids + filler[:half] + tok.encode(f"\nPROJE_ID = {v}\n").ids \
            + filler[half:] + q
        _, hit_latest = value_logprob(model, latest, value, device, amp)
        res["gain_nats"] += (lp_with - lp_without) / n
        res["exact"] += hit / n
        res["exact_latest"] += hit_latest / n
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True, help="değerlendirme .bin (son --val-frac kısmı kullanılır)")
    p.add_argument("--val-frac", type=float, default=0.005)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--position-len", type=int, default=8192)
    p.add_argument("--distances", type=int, nargs="+", default=[16, 128, 1024, 8192, 65536, 262144, 1048576])
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    amp = torch.autocast(device.type, dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    model = load(args.ckpt, device)
    tok = Tokenizer.from_file(args.tokenizer)
    full = np.memmap(args.data, dtype=np.uint16, mode="r")
    val = full[int(len(full) * (1 - args.val_frac)):]
    filler_src = val if len(val) > max(args.distances) + 10 else full  # uzun iğneler için tüm dosya
    report = {"ckpt": args.ckpt}

    vl = val_loss(model, val, args.seq_len, 32, device, amp)
    report["val_loss"] = vl
    print(f"1) Validation loss: {vl:.4f}  (perplexity {math.exp(vl):.2f})", flush=True)

    if len(val) > args.position_len + 2:
        lbp = loss_by_position(model, val, args.position_len, 8, device, amp)
        report["loss_by_position"] = lbp
        print("2) Pozisyona göre loss: " + " | ".join(f"{r['from']}-{r['to']}: {r['loss']:.3f}" for r in lbp), flush=True)

    print("3) PROJE_ID iğnesi (kazanç = iğneli − iğnesiz log-olasılık; tam isabet = değerin tamamı doğru)")
    rng = np.random.default_rng(0)
    report["needle"] = []
    for d in args.distances:
        n = args.samples if d <= 65536 else max(2, args.samples // 4)
        t0 = time.time()
        r = needle_test(model, tok, filler_src, d, n, device, amp, rng)
        report["needle"].append(r)
        print(f"   mesafe {d:>9,}: kazanç {r['gain_nats']:+6.2f} nats | tam isabet {r['exact']:5.0%} | "
              f"son değer {r['exact_latest']:5.0%} | n={n} | {time.time() - t0:5.1f}s", flush=True)

    out = Path(args.out or Path(args.ckpt).with_suffix(".degerlendirme.json"))
    out.write_text(json.dumps(report, indent=2))
    print(f"Kaydedildi: {out}")


if __name__ == "__main__":
    main()
