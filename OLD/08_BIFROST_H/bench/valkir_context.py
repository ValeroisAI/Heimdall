"""
valkir_context.py — Eğitilmiş Valkir'in gerçek bağlam kullanımı
===============================================================
1) Pozisyona göre loss: uzun (eğitimde görülmemiş) Python dosyalarında her pozisyon
   aralığının ortalama loss'u. Loss'un düşmeyi bıraktığı yer = kullanılan bağlam.
2) Doğal kod iğnesi:   PROJE_ID = 4721  ...[gerçek kod, d token]...  assert PROJE_ID == 4721
   Değer tokenlerinin log-olasılığı, iğne VARKEN ve YOKKEN karşılaştırılır.
   Fark > 0 ise model uzaktaki tanımı kullanıyor demektir.

Kullanım:
    python bench/valkir_context.py --ckpt data/Valkir.pt --schedule const2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.valkir import load  # noqa: E402

TOKENIZER = ROOT.parent / "04_TOKENIZERLAR" / "valerois_tokenizer_8k.json"


def loss_by_position(model, data, length, n_seq, rng, buckets):
    starts = rng.integers(0, len(data) - length - 1, n_seq)
    total = torch.zeros(length)
    for s in starts:
        ids = torch.tensor(data[s:s + length + 1].astype(np.int64))[None]
        logits = model(ids[:, :-1])
        total += F.cross_entropy(logits[0], ids[0, 1:], reduction="none")
    per_pos = total / n_seq
    return [{"from": a, "to": b, "loss": per_pos[a:b].mean().item()} for a, b in buckets if b <= length]


def needle(model, tok, data, distance, n, rng):
    """İğne var/yok iken değer tokenlerinin ortalama log-olasılığı ve tam isabet oranı."""
    out = {"with": [], "without": [], "hit_with": 0, "hit_without": 0}
    for _ in range(n):
        value = str(rng.integers(1000, 9999))
        needle_ids = tok.encode(f"PROJE_ID = {value}\n").ids
        query_ids = tok.encode("\nassert PROJE_ID == ").ids
        value_ids = tok.encode(value).ids
        s = rng.integers(0, len(data) - distance - 1)
        filler = data[s:s + distance].astype(np.int64).tolist()
        for mode, prefix in (("with", needle_ids), ("without", [])):
            seq = prefix + filler + query_ids + value_ids
            ids = torch.tensor(seq)[None]
            logp = F.log_softmax(model(ids[:, :-1])[0], -1)
            pos = torch.arange(len(seq) - len(value_ids) - 1, len(seq) - 1)
            tgt = ids[0, pos + 1]
            out[mode].append(logp[pos, tgt].sum().item())
            out["hit_" + mode] += int((logp[pos].argmax(-1) == tgt).all())
    return {
        "distance": distance,
        "logp_with": float(np.mean(out["with"])), "logp_without": float(np.mean(out["without"])),
        "gain_nats": float(np.mean(out["with"]) - np.mean(out["without"])),
        "exact_with": out["hit_with"] / n, "exact_without": out["hit_without"] / n,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(ROOT / "data" / "Valkir.pt"))
    p.add_argument("--schedule", default="const2")
    p.add_argument("--val", default=str(ROOT / "data" / "dev_code_8k_val.bin"))
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(0)
    model = load(args.ckpt, args.schedule)
    tok = Tokenizer.from_file(str(TOKENIZER))
    data = np.memmap(args.val, dtype=np.uint16, mode="r")
    print(f"Valkir | çizelge {args.schedule} | teorik RF {model.receptive_field():,} token", flush=True)

    buckets = [(0, 16), (16, 64), (64, 256), (256, 512), (512, 1024), (1024, 1536), (1536, 2048),
               (2048, 3072), (3072, 4096)]
    lbp = loss_by_position(model, data, 4096, 12, rng, buckets)
    print("\n1) Pozisyona göre loss (hiç görülmemiş Python dosyaları)")
    for row in lbp:
        print(f"   {row['from']:5d}-{row['to']:5d}: {row['loss']:.3f}", flush=True)

    print("\n2) PROJE_ID iğnesi (değer tokenlerinin log-olasılığı; kazanç = iğneli − iğnesiz)")
    rows = []
    for d in (8, 32, 128, 512, 900, 1100, 2048, 4000):
        r = needle(model, tok, data, d, 24, rng)
        rows.append(r)
        print(f"   mesafe {d:5d}: kazanç {r['gain_nats']:+6.2f} nats | tam isabet iğneli {r['exact_with']:5.0%}"
              f" / iğnesiz {r['exact_without']:4.0%}", flush=True)

    out = ROOT / "results" / "context" / f"valkir_{args.schedule}.json"
    out.write_text(json.dumps({"schedule": args.schedule, "receptive_field": model.receptive_field(),
                               "loss_by_position": lbp, "needle": rows}, indent=2))


if __name__ == "__main__":
    main()
