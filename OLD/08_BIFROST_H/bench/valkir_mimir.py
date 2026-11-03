"""
valkir_mimir.py — Dondurulmuş Valkir (saf CSL) + araya eklenen Mímir katmanları
==============================================================================
Valkir'in 16 CSL katmanı dondurulur; her `every` katmandan sonra bir Mímir katmanı
eklenir (çıkış projeksiyonu sıfır -> başlangıçta model Valkir ile birebir aynı).
Yalnız Mímir eğitilir. Veri: gerçek Python kodu + araya serpiştirilmiş
"isim = değer" atamaları ve sonda "assert isim == değer" sorguları (bazı isimler
iki kez atanır; doğru cevap SON değerdir).

Kullanım:
    python bench/valkir_mimir.py --minutes 9 --length 256
"""

import argparse
import json
import string
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.valkir import SCHEDULES, causal_dw_conv, rms  # noqa: E402
from bifrost.layers import Mimir, RMSNorm  # noqa: E402

TOKENIZER = ROOT.parent / "04_TOKENIZERLAR" / "valerois_tokenizer_8k.json"


class ValkirMimir(nn.Module):
    def __init__(self, state_dict: dict, schedule: str = "const2", every: int = 4, heads: int = 6) -> None:
        super().__init__()
        self.frozen = nn.ParameterDict({k.replace(".", "__"): nn.Parameter(v.float(), requires_grad=False)
                                        for k, v in state_dict.items()})
        self.n_layers = 1 + max(int(k.split(".")[1]) for k in state_dict if k.startswith("layers."))
        self.dilations = SCHEDULES[schedule][: self.n_layers]
        dim = state_dict["embed_tokens.weight"].size(1)
        self.slots = [l for l in range(self.n_layers) if (l + 1) % every == 0]
        self.mimir = nn.ModuleDict({str(l): Mimir(dim, heads, 64, 64, chunk_size=64) for l in self.slots})
        self.mimir_norm = nn.ModuleDict({str(l): RMSNorm(dim) for l in self.slots})
        for m in self.mimir.values():
            nn.init.zeros_(m.o_proj.weight)

    def w(self, name: str) -> torch.Tensor:
        return self.frozen[name.replace(".", "__")]

    def forward(self, ids: torch.Tensor, use_mimir: bool = True) -> torch.Tensor:
        x = F.embedding(ids, self.w("embed_tokens.weight"))
        for l, dil in enumerate(self.dilations):
            p = f"layers.{l}."
            h = rms(x, self.w(p + "input_layernorm.weight"))
            xs = causal_dw_conv(h, self.w(p + "mixer.conv_short.weight"), 1)
            xd = causal_dw_conv(h, self.w(p + "mixer.conv_dilated.weight"), dil)
            m = rms(F.linear(h, self.w(p + "mixer.in_proj.weight")) + 0.5 * (xs + xd), self.w(p + "mixer.norm.weight"))
            x = x + F.linear(m * F.silu(F.linear(h, self.w(p + "mixer.gate_proj.weight"))), self.w(p + "mixer.out_proj.weight"))
            h = rms(x, self.w(p + "post_mixer_layernorm.weight"))
            x = x + F.linear(F.silu(F.linear(h, self.w(p + "mlp.w1.weight"))) * F.linear(h, self.w(p + "mlp.w2.weight")),
                             self.w(p + "mlp.w3.weight"))
            if use_mimir and str(l) in self.mimir:
                x = x + self.mimir[str(l)](self.mimir_norm[str(l)](x))
        return F.linear(rms(x, self.w("norm.weight")), self.w("lm_head.weight"))


def random_name(rng) -> str:
    letters = string.ascii_lowercase
    parts = ["".join(rng.choice(list(letters), size=rng.integers(3, 7))) for _ in range(rng.integers(1, 3))]
    name = "_".join(parts)
    return name.upper() if rng.random() < 0.3 else name


def make_sample(rng, tok, data, length, n_vars=3, overwrite_p=0.3):
    """Gerçek kod içine atamalar serpiştirir; sonda her isim sorgulanır (cevap: son değer)."""
    names = [random_name(rng) for _ in range(n_vars)]
    values = {n: [str(rng.integers(100, 99999))] for n in names}
    for n in names:
        if rng.random() < overwrite_p:
            values[n].append(str(rng.integers(100, 99999)))
    assigns = [(n, v) for n in names for v in values[n]]
    order = rng.permutation(len(assigns))
    assigns = [assigns[i] for i in order]
    # aynı isim iki kez atandıysa sıralarını koru (ilk değer önce)
    for n in names:
        idx = [i for i, (m, _) in enumerate(assigns) if m == n]
        for i, v in zip(idx, values[n]):
            assigns[i] = (n, v)
    queries, targets = [], []
    for n in rng.permutation(names):
        q = tok.encode(f"\nassert {n} == ").ids
        v = tok.encode(values[n][-1]).ids
        queries.append((q, v))
    assign_ids = [tok.encode(f"\n{n} = {v}\n").ids for n, v in assigns]
    tail_len = sum(len(q) + len(v) for q, v in queries)
    body = length + 1 - tail_len - sum(len(a) for a in assign_ids)
    s = rng.integers(0, len(data) - body - 1)
    filler = data[s:s + body].astype(np.int64).tolist()
    cuts = np.sort(rng.integers(0, max(1, body // 2), size=len(assign_ids)))  # atamalar ilk yarıda
    seq, prev = [], 0
    for c, a in zip(cuts, assign_ids):
        seq += filler[prev:c] + a
        prev = c
    seq += filler[prev:]
    mask = [0] * len(seq)
    for q, v in queries:
        seq += q + v
        mask += [0] * len(q) + [1] * len(v)
    seq, mask = seq[: length + 1], mask[: length + 1]
    return seq, mask


def batch(rng, tok, data, b, length):
    xs, ms = zip(*(make_sample(rng, tok, data, length) for _ in range(b)))
    ids = torch.tensor(xs)
    mask = torch.tensor(ms)[:, 1:].bool()
    return ids[:, :-1], ids[:, 1:], mask


@torch.no_grad()
def needle_eval(model, tok, data, distances, n, rng, use_mimir=True):
    rows = []
    for d in distances:
        gains, hits, hits_over = [], 0, 0
        for _ in range(n):
            v1, v2 = str(rng.integers(1000, 9999)), str(rng.integers(1000, 9999))
            s = rng.integers(0, len(data) - d - 1)
            filler = data[s:s + d].astype(np.int64).tolist()
            q, val = tok.encode("\nassert PROJE_ID == ").ids, tok.encode(v1).ids
            res = {}
            for mode, prefix in (("with", tok.encode(f"PROJE_ID = {v1}\n").ids), ("without", [])):
                seq = prefix + filler + q + val
                ids = torch.tensor(seq)[None]
                logp = F.log_softmax(model(ids[:, :-1], use_mimir)[0], -1)
                pos = torch.arange(len(seq) - len(val) - 1, len(seq) - 1)
                res[mode] = (logp[pos, ids[0, pos + 1]].sum().item(), bool((logp[pos].argmax(-1) == ids[0, pos + 1]).all()))
            gains.append(res["with"][0] - res["without"][0])
            hits += res["with"][1]
            # güncelleme testi: PROJE_ID = v2 ... PROJE_ID = v1 ... assert -> v1 (son değer)
            seq = tok.encode(f"PROJE_ID = {v2}\n").ids + filler[: d // 2] + tok.encode(f"\nPROJE_ID = {v1}\n").ids \
                + filler[d // 2:] + q + val
            ids = torch.tensor(seq)[None]
            logp = F.log_softmax(model(ids[:, :-1], use_mimir)[0], -1)
            pos = torch.arange(len(seq) - len(val) - 1, len(seq) - 1)
            hits_over += bool((logp[pos].argmax(-1) == ids[0, pos + 1]).all())
        rows.append({"distance": d, "gain_nats": float(np.mean(gains)), "exact": hits / n, "exact_latest": hits_over / n})
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(ROOT / "data" / "Valkir.pt"))
    p.add_argument("--train", default=str(ROOT / "data" / "dev_code_8k_train.bin"))
    p.add_argument("--val", default=str(ROOT / "data" / "dev_code_8k_val.bin"))
    p.add_argument("--minutes", type=float, default=9)
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    tok = Tokenizer.from_file(str(TOKENIZER))
    train = np.memmap(args.train, dtype=np.uint16, mode="r")
    val = np.memmap(args.val, dtype=np.uint16, mode="r")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = ValkirMimir(ckpt["model"])
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Mímir katmanları {model.slots} | eğitilen {sum(p.numel() for p in trainable) / 1e6:.1f}M param", flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    deadline, step, t0 = time.time() + args.minutes * 60, 0, time.time()
    hist = []
    while time.time() < deadline:
        x, y, m = batch(rng, tok, train, args.batch, args.length)
        logits = model(x)
        lm = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        recall = F.cross_entropy(logits[m], y[m])
        loss = lm + 2.0 * recall
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        step += 1
        hist.append((lm.item(), recall.item(), (logits[m].argmax(-1) == y[m]).float().mean().item()))
        if step % 10 == 0:
            a = np.mean(hist[-10:], axis=0)
            print(f"  adım {step:4d} | LM {a[0]:.3f} | değer loss {a[1]:.3f} | değer tokeni doğruluğu {a[2]:.0%} "
                  f"| {(time.time() - t0) / step:.1f} s/adım", flush=True)

    model.eval()
    erng = np.random.default_rng(1)
    distances = [32, 128, 512, 900, 1100, 2048]
    print("\nİğne testi (hiç görülmemiş kod; n=16)")
    results = {}
    for label, use in (("Valkir (Mímir kapalı)", False), ("Valkir + Mímir", True)):
        rows = needle_eval(model, tok, val, distances, 16, np.random.default_rng(1), use)
        results[label] = rows
        print(f"  {label}")
        for r in rows:
            print(f"    mesafe {r['distance']:5d}: kazanç {r['gain_nats']:+6.2f} nats | tam isabet {r['exact']:4.0%} "
                  f"| güncelleme (son değer) {r['exact_latest']:4.0%}", flush=True)
    out = ROOT / "results" / "context" / "valkir_mimir.json"
    out.write_text(json.dumps({"steps": step, "length": args.length, "train_tail": np.mean(hist[-20:], axis=0).tolist(),
                               "needle": results}, indent=2))


if __name__ == "__main__":
    main()
