"""
mqar.py — Çok anahtarlı ilişkisel hatırlama (MQAR) + "son değer" + uzunluk genellemesi
======================================================================================
Dizi:  k1 v1 k2 v2 … kn vn  [G dolgu tokeni]  q1 ? q2 ? … (her anahtar bir kez sorulur)
- Çiftlerin bir kısmı İKİ kez atanır (k v … k v'): doğru cevap SON değerdir (Proje-ID).
- Kayıp yalnız sorulan değerlerde. Eğitimde G küçüktür; testte G 1M tokene kadar büyütülür.
- Pencere modellerinde W=16: çiftler ile sorgu arası ≥ 16 ise bilgi yalnız hafızadan gelebilir.

Kullanım:
    python bench/mqar.py --model K --minutes 6
Modeller: K (Kuzgun), W (yalnız pencere), R (yalnız delta hafıza), A (tam dikkat, referans)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bifrost import BifrostLM, ModelConfig  # noqa: E402
from bifrost.optim import build_optimizers, set_lr, wsd_factor  # noqa: E402

FILL = 0
N_KEYS, N_VALUES = 64, 64
KEY0, VAL0 = 1, 1 + N_KEYS
VOCAB = 1 + N_KEYS + N_VALUES


def make_batch(rng, batch, n_pairs, gap, overwrite_p=0.3):
    """Döndürür: ids [B, L], targets [B, L] (-100 = kayıp yok). L tüm örneklerde aynıdır."""
    rows, tgts = [], []
    for _ in range(batch):
        keys = rng.choice(N_KEYS, size=n_pairs, replace=False)
        final = {}
        writes = []
        for k in keys:
            v = rng.integers(N_VALUES)
            writes.append((k, v))
            final[k] = v
        for k in keys[rng.random(n_pairs) < overwrite_p]:  # bazı anahtarlar ikinci kez atanır
            v = rng.integers(N_VALUES)
            writes.append((k, v))
            final[k] = v
        # yazma sırası: ilk atamalar önce, ikinci atamalar sonra (sonraki = güncel değer)
        kv = [t for k, v in writes for t in (KEY0 + k, VAL0 + v)]
        seq = kv + [FILL] * gap
        tgt = [-100] * len(seq)
        for k in rng.permutation(keys):
            seq += [KEY0 + k, FILL]
            tgt += [VAL0 + final[k], -100]
        rows.append(seq)
        tgts.append(tgt)
    length = max(len(r) for r in rows)
    ids = torch.full((batch, length), FILL, dtype=torch.long)
    targets = torch.full((batch, length), -100, dtype=torch.long)
    for i, (r, t) in enumerate(zip(rows, tgts)):
        ids[i, :len(r)] = torch.tensor(r)
        targets[i, :len(t)] = torch.tensor(t)
    return ids, targets


def model_for(kind: str, coupled: bool = False, heads: int = 1, head_dim: int = 64, archival: int = 0) -> BifrostLM:
    cfg = ModelConfig(vocab_size=VOCAB, dim=64, layout=kind * 2, csl=False, kuzgun_heads=heads, kuzgun_head_dim=head_dim,
                      window=16, attn_heads=1, mimir_chunk=64, ffn_mult=2.0, zero_init_out=True,
                      coupled_decay=coupled, archival_heads=archival)
    return BifrostLM(cfg)


@torch.no_grad()
def accuracy_at_gap(model, gap, n_pairs, samples, rng, chunk=8192):
    ids, targets = make_batch(rng, samples, n_pairs, gap)
    mask = targets != -100
    if gap <= chunk:
        logits = model(ids)
        return (logits.argmax(-1)[mask] == targets[mask]).float().mean().item()
    # Uzun boşluk: dizi parça parça işlenir (durum taşınır); yalnız sorgu bölümünün logit'leri tutulur.
    state, preds = None, []
    query_start = ids.size(1) - 2 * n_pairs
    for s in range(0, ids.size(1), chunk):
        piece = ids[:, s:s + chunk]
        if s + piece.size(1) <= query_start:
            _, state = model.forward_stateful(piece, state, last_only=True)
        else:
            logits, state = model.forward_stateful(piece, state)
            preds.append((s, logits.argmax(-1)))
    pred = torch.full_like(ids, -1)
    for s, p in preds:
        pred[:, s:s + p.size(1)] = p
    return (pred[mask] == targets[mask]).float().mean().item()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list("KWRA"), required=True)
    p.add_argument("--minutes", type=float, default=6.0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--pairs", type=int, default=8)
    p.add_argument("--max-train-gap", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-gap", type=int, default=1 << 20)
    p.add_argument("--coupled", action="store_true", help="bağlı unutma (log α ← β·log α)")
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--archival", type=int, default=0, help="unutmayan (α≡1) kafa sayısı")
    p.add_argument("--long-gap-p", type=float, default=0.0, help="uzun boşluklu (1K-8K) batch olasılığı")
    p.add_argument("--tag", default="")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = model_for(args.model, coupled=args.coupled, heads=args.heads, head_dim=args.head_dim, archival=args.archival)
    name = args.model + (f"_{args.tag}" if args.tag else "")
    opts = build_optimizers(model, "muon")
    print(f"[{args.model}] {model.num_params() / 1e3:.0f}K param | {args.pairs} çift | eğitim boşluğu ≤ {args.max_train_gap}",
          flush=True)

    budget, t0, step, hist = args.minutes * 60, time.time(), 0, []
    while time.time() - t0 < budget:
        set_lr(opts, wsd_factor((time.time() - t0) / budget))
        gap = int(rng.integers(0, args.max_train_gap + 1))
        batch = args.batch
        if rng.random() < args.long_gap_p:
            gap, batch = int(rng.integers(1024, 8193)), max(4, args.batch // 8)
        ids, targets = make_batch(rng, batch, args.pairs, gap)
        logits = model(ids)
        mask = targets != -100
        loss = F.cross_entropy(logits[mask], targets[mask])
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
        step += 1
        hist.append((loss.item(), (logits[mask].argmax(-1) == targets[mask]).float().mean().item()))
        if step % 200 == 0:
            l, a = np.mean(hist[-200:], axis=0)
            print(f"  adım {step:5d} | {time.time() - t0:5.0f}s | loss {l:.3f} | doğruluk {a:6.1%}", flush=True)

    model.eval()
    out = ROOT / "results" / "mqar"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.cfg.to_dict(), "model": model.state_dict()}, out / f"{name}.pt")
    limit = {"A": 16384, "W": 16384}.get(args.model, args.max_gap)
    gaps = [g for g in [0, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576] if g <= limit]
    rows = []
    for g in gaps:
        n = 256 if g <= 1024 else (64 if g <= 16384 else (16 if g <= 65536 else 8))
        acc = accuracy_at_gap(model, g, args.pairs, n, np.random.default_rng(10_000 + g))
        rows.append({"gap": g, "accuracy": acc, "samples": n})
        print(f"  boşluk {g:8d} | doğruluk {acc:6.1%} (n={n})", flush=True)
    (out / f"{name}.json").write_text(json.dumps({
        "model": args.model, "tag": args.tag, "coupled_decay": args.coupled, "heads": args.heads, "archival": args.archival, "long_gap_p": args.long_gap_p, "params": model.num_params(), "steps": step, "minutes": args.minutes,
        "train_tail": dict(zip(("loss", "acc"), map(float, np.mean(hist[-200:], axis=0)))),
        "chance": 1 / N_VALUES, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
