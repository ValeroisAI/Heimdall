"""
context_test.py — "Model fiziksel olarak ne kadar uzağı görüyor?" testi
======================================================================
Görev (anahtar-değer iğnesi):  ... MARK k v ... [dolgu + çeldirici çiftler] ... QUERY k -> v
Model kısa dizilerde (T=256) eğitilir, sonra iğne ile soru arasındaki mesafe
64 tokenden 1M tokene kadar büyütülerek doğruluk ölçülür. Uzun diziler
forward_stateful ile parça parça işlenir (bellek sabit kalır mı, o da ölçülür).

Kullanım:
    python bench/context_test.py --model bifrost --minutes 7 --threads 2
    python bench/context_test.py --model csl --minutes 7 --threads 1
    python bench/context_test.py --model transformer --minutes 7 --threads 1
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bifrost import BifrostLM, ModelConfig  # noqa: E402

# Sözlük: 0 PAD, 1 MARK, 2 QUERY, anahtarlar, değerler, dolgu
PAD, MARK, QUERY = 0, 1, 2
N_KEYS, N_VALUES = 64, 64
KEY0 = 16
VAL0 = KEY0 + N_KEYS
FILL0 = VAL0 + N_VALUES
VOCAB = 256
PAIR_EVERY = 64  # ortalama her 64 tokende bir çeldirici çift

MODELS = {
    "bifrost": ModelConfig(vocab_size=VOCAB, dim=128, layout="MMMM", csl=True, mimir_heads=2,
                           mimir_dk=64, mimir_dv=64, mimir_chunk=64),
    "csl": ModelConfig(vocab_size=VOCAB, dim=160, layout="NNNN", csl=True, csl_kernel=7),
    "transformer": ModelConfig(vocab_size=VOCAB, dim=128, layout="AAAA", csl=False, attn_heads=2),
}


def fill_with_distractors(rng, length, exclude_key):
    """Dolgu tokenleri + ortalama PAIR_EVERY'de bir çeldirici 'MARK k v' (hedef anahtar hariç)."""
    seq = rng.integers(FILL0, VOCAB, size=length)
    n_pairs = length // PAIR_EVERY
    if n_pairs:
        starts = np.sort(rng.choice(max(1, length // 3), size=n_pairs, replace=False)) * 3
        starts = starts[starts + 3 <= length]
        keys = rng.integers(0, N_KEYS - 1, size=starts.size)
        keys = keys + (keys >= exclude_key)  # hedef anahtarı atla
        seq[starts] = MARK
        seq[starts + 1] = KEY0 + keys
        seq[starts + 2] = VAL0 + rng.integers(0, N_VALUES, size=starts.size)
    return seq


def train_batch(rng, batch, length, n_needles=8, n_queries=8):
    """Eğitim dizisi: n_needles farklı anahtar gömülür, sonda n_queries soru sorulur."""
    xs, ys, ms = [], [], []
    body = length - 3 * n_queries
    for _ in range(batch):
        keys = rng.choice(N_KEYS, size=n_needles, replace=False)
        vals = rng.integers(0, N_VALUES, size=n_needles)
        seq = rng.integers(FILL0, VOCAB, size=body)
        slots = rng.choice(body // 3, size=n_needles, replace=False) * 3
        for s, k, v in zip(slots, keys, vals):
            seq[s:s + 3] = (MARK, KEY0 + k, VAL0 + v)
        ask = rng.permutation(n_needles)[:n_queries]
        tail = np.array([[QUERY, KEY0 + keys[a], VAL0 + vals[a]] for a in ask]).ravel()
        full = np.concatenate((seq, tail, [PAD]))
        mask = np.zeros(length, dtype=bool)
        mask[body + 1 + 3 * np.arange(n_queries)] = True  # 'k' pozisyonunda hedef v
        xs.append(full[:-1]); ys.append(full[1:]); ms.append(mask)
    return (torch.tensor(np.array(xs)), torch.tensor(np.array(ys)), torch.tensor(np.array(ms)))


def eval_sequences(rng, batch, distance):
    """Başta iğne, arada `distance` token dolgu + çeldirici, sonda soru."""
    seqs, answers = [], []
    for _ in range(batch):
        k, v = rng.integers(N_KEYS), rng.integers(N_VALUES)
        middle = fill_with_distractors(rng, distance, k)
        seqs.append(np.concatenate(([MARK, KEY0 + k, VAL0 + v], middle, [QUERY, KEY0 + k])))
        answers.append(VAL0 + v)
    return torch.tensor(np.array(seqs)), torch.tensor(answers)


@torch.no_grad()
def evaluate(model, distance, n_samples, chunk=4096, seed=0):
    rng = np.random.default_rng(seed + distance)
    ids, answers = eval_sequences(rng, n_samples, distance)
    state, logits = None, None
    t0 = time.perf_counter()
    for start in range(0, ids.size(1), chunk):
        piece = ids[:, start:start + chunk]
        logits, state = model.forward_stateful(piece, state, last_only=True)
    elapsed = time.perf_counter() - t0
    acc = (logits[:, -1].argmax(-1) == answers).float().mean().item()
    return acc, ids.numel() / elapsed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--minutes", type=float, default=7.0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--max-distance", type=int, default=1 << 20)
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "context"))
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cfg = MODELS[args.model]
    model = BifrostLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    print(f"[{args.model}] {model.num_params() / 1e6:.2f}M param, eğitim T={args.length}, {args.minutes} dk", flush=True)

    deadline = time.time() + args.minutes * 60
    step, hist = 0, []
    while time.time() < deadline:
        x, y, m = train_batch(rng, args.batch, args.length)
        logits = model(x)
        loss = F.cross_entropy(logits[m], y[m])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        acc = (logits[m].argmax(-1) == y[m]).float().mean().item()
        hist.append((loss.item(), acc))
        if step % 100 == 0:
            l, a = np.mean(hist[-100:], axis=0)
            print(f"  adım {step:5d} | loss {l:.3f} | eğitim doğruluğu {a:.3f}", flush=True)

    model.eval()
    limits = {"transformer": 8192, "csl": 16384}
    max_d = min(args.max_distance, limits.get(args.model, args.max_distance))
    distances = [d for d in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                             65536, 131072, 262144, 524288, 1048576] if d <= max_d]
    rows = []
    for d in distances:
        n = 64 if d <= 4096 else (16 if d <= 65536 else 4)
        acc, tps = evaluate(model, d, n)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        rows.append({"distance": d, "accuracy": acc, "samples": n, "tokens_per_s": tps, "peak_rss_mb": rss})
        print(f"  mesafe {d:8d} | doğruluk {acc:6.1%} (n={n}) | {tps:9,.0f} tok/s | tepe RAM {rss:7.0f} MB", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = args.model + (f"_{args.tag}" if args.tag else "")
    torch.save(model.state_dict(), out / f"{name}.pt")
    (out / f"{name}.json").write_text(json.dumps({
        "model": args.model, "config": cfg.to_dict(), "params": model.num_params(), "train_steps": step,
        "train_length": args.length, "final_train": dict(zip(("loss", "acc"), map(float, np.mean(hist[-100:], axis=0)))),
        "chance": 1 / N_VALUES, "results": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
