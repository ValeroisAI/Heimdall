"""
MQAR (çoklu anahtar-değer hatırlama): dizinin başında 8 anahtar→değer çifti (bazıları güncellenir), araya boşluk,
sonra anahtarlar sorulur; doğru cevap anahtarın SON değeridir. Eğitimde boşluk ≤ 64, testte 16K'ya kadar.

    python bench/mqar.py H 8      # H = hibrit (delta + dikkat), C = kısa evrişim + dikkat, A = saf dikkat, R = saf delta; 8 = dakika
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(1)
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



from heimdall import HeimdallConfig, HeimdallLM
from heimdall.optim import build_optimizers, set_lr, wsd
kind, minutes = sys.argv[1], float(sys.argv[2])
base = HeimdallConfig(vocab_size=VOCAB, d_model=64, n_layers=2, n_heads=1, head_dim=64, archival_heads=1, attn_heads=1,
                      attn_kv_heads=1, attn_head_dim=64, ffn_mult=2.0, ffn_multiple_of=64, chunk_size=64, kernel="torch")
cfg = {"R": base.__class__(**{**base.to_dict(), "layout": "D"}),
       "A": base.__class__(**{**base.to_dict(), "layout": "A", "attn_rope": "rope"}),
       "H": base.__class__(**{**base.to_dict(), "layout": "DA"}),
       "C": base.__class__(**{**base.to_dict(), "layout": "CA"})}[kind]
torch.manual_seed(0); m = HeimdallLM(cfg)
opts = build_optimizers(m, 0.02, 3e-3)
rng = np.random.default_rng(0); t0 = time.time(); step = 0
while time.time() - t0 < minutes * 60:
    set_lr(opts, wsd((time.time() - t0) / (minutes * 60), 0.02, 0.7))
    ids, tg = make_batch(rng, 64, 8, int(rng.integers(0, 65)))
    loss = F.cross_entropy(m(ids).reshape(-1, VOCAB), tg.reshape(-1), ignore_index=-100)
    loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    for o in opts: o.step(); o.zero_grad(set_to_none=True)
    step += 1
res = {"kind": kind, "params": m.num_params(), "steps": step}
m.eval(); erng = np.random.default_rng(123)
with torch.no_grad():
    for gap, n in ((64, 128), (1024, 64), (16384, 16), (262144, 4)):
        if kind != "R" and gap > 16384: continue
        ids, tg = make_batch(erng, n, 8, gap)
        p = m(ids).argmax(-1); mk = tg != -100
        res[f"gap_{gap}"] = round((p[mk] == tg[mk]).float().mean().item(), 3)
print(json.dumps(res), flush=True)
