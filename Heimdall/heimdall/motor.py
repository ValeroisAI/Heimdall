"""Bifrost eğitim motoru: önceden eğitilmiş bir modelin tam dikkat katmanlarını, uzun dizilerle (64K+) ve tek bir
16 GB GPU'da sabit belleğe çevirir. Eski aşama A (donustur_hibrit.py) ile aynı hedef, aynı parametreler, aynı GGUF.

    python -m heimdall.motor plan --model Qwen/Qwen3.8-27B
    python -m heimdall.motor egit --model Qwen/Qwen3.8-27B --train veri/qwen_wiki/wiki_train.bin --out runs/q38_m1

Neden hızlı (eski aşama A her adımda öğretmeni ve pencereyi baştan hesaplıyordu):
 1. Öğretmenin tam dikkati dizi başına BİR KEZ, blok blok (çevrimiçi softmax) hesaplanır; bellek O(T · blok).
 2. Pencere + çapa dikkatinin eğitilen parametresi yok: o da bir kez hesaplanır, yalnız sonucu saklanır.
 3. Yalnız hafıza eğitilir. Zaman parçalara bölünür; parça sınırında yalnız durum (KV başı başına D×D) tutulur ve
    geri yayılımda parça yeniden hesaplanır: gradyan TAM (kesilmiş değil), bellek dizi boyuna değil parça boyuna bağlı.
 4. GQA'ya özel delta kuralı: durum KV başı başına, okuma R sorgu başıyla (Qwen3.8: 24/4 → 6× daha az iş).
 5. Kayıp seçilmiş konumlarda: rastgele bir alt küme + sentetik soruların tamamı (ağırlıklı); disk ~8× küçük.
 6. Gizli durumlar diskte (memmap): RAM'den bağımsız. Bir sonraki dizi arka planda yüklenir.
 7. Kayıt her katmandan sonra: yarıda kalırsa --devam kaldığı katmandan sürer (gizli durumlar diskte).
Doğruluk: tests/test_motor.py motorun kaybını ve gradyanını HybridAttention.distill_pair ile birebir karşılaştırır.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .donustur import fetch_model
from .donustur_hibrit import eta, gb, load_tokenizer, recall_sequence
from .hibrit import ATTN, HLayer, HybridConfig, Weights, materialize, n_archival


# --------------------------------------------------------------------------- çekirdekler (yalnız temel işlemler)
def blockwise_attention(qr, kr, vt, qblk: int = 2048, kblk: int = 8192):
    """Nedensel tam dikkat, GQA, çevrimiçi softmax (flash mantığı, saf PyTorch).
    qr [B,H,T,D] (RoPE'lu), kr / vt [B,G,T,D]. Döndürür o [B,T,H,D] fp32. Bellek O(qblk · kblk), T² değil."""
    b, h, t, d = qr.shape
    g = kr.size(1)
    r, scale = h // g, d ** -0.5
    out = torch.empty(b, t, h, d, dtype=torch.float32, device=qr.device)
    with torch.autocast(qr.device.type, enabled=False):
        for q0 in range(0, t, qblk):
            q1 = min(t, q0 + qblk)
            q = qr[:, :, q0:q1].float().reshape(b, g, r, q1 - q0, d) * scale
            m = q.new_full((b, g, r, q1 - q0, 1), float("-inf"))
            l, acc = torch.zeros_like(m), torch.zeros_like(q)
            qi = torch.arange(q0, q1, device=qr.device)[:, None]
            for k0 in range(0, q1, kblk):
                k1 = min(q1, k0 + kblk)
                s = q @ kr[:, :, None, k0:k1].float().transpose(-1, -2)              # [b,g,r,qb,kb]
                if k1 > q0 + 1:                                                     # blok köşegeni kesiyor
                    s = s.masked_fill(torch.arange(k0, k1, device=qr.device)[None] > qi, float("-inf"))
                bm = s.amax(-1, keepdim=True)
                mn = torch.where(bm > m, bm, m)
                p, corr = (s - mn).exp(), (m - mn).exp()
                l = l * corr + p.sum(-1, keepdim=True)
                acc = acc * corr + p @ vt[:, :, None, k0:k1].float()
                m = mn
            out[:, q0:q1] = (acc / l).reshape(b, h, q1 - q0, d).transpose(1, 2)
    return out


def unit_lower_inverse(a: torch.Tensor) -> torch.Tensor:
    """(I + A)^-1 for strictly lower-triangular A [..., c, c], in ceil(log2 c) batched steps: A is nilpotent, so
    (I + A)^-1 = (I - A)(I + A^2)(I + A^4)... exactly. Same result as forward substitution (to fp32 rounding), but a
    handful of matmuls instead of a c-step Python loop, and it does not depend on solve_triangular, which some ROCm
    builds lack (kernels._unit_lower_inverse then falls back to that loop)."""
    n = -a
    p = torch.eye(a.size(-1), dtype=a.dtype, device=a.device) + n
    for _ in range(max(0, math.ceil(math.log2(a.size(-1))) - 1)):
        n = n @ n
        p = p + p @ n
    return p


def delta_rule_gqa(q, k, v, g, beta, chunk: int = 64, state=None):
    """Kapılı delta kuralı, durum KV başı başına, okuma R sorgu başıyla (delta_rule_reference ile aynı denklem).
    q [B,T,G,R,Dk], k [B,T,G,Dk] (L2-normalize), v [B,T,G,Dv], g / beta [B,T,G], state [B,G,Dk,Dv].
    Döndürür o [B,T,G,R,Dv] ve son durum [B,G,Dk,Dv], ikisi de fp32."""
    with torch.autocast(q.device.type, enabled=False):
        q = q.float().permute(0, 2, 3, 1, 4)                                        # [b,G,R,t,dk]
        k, v = k.float().transpose(1, 2), v.float().transpose(1, 2)                 # [b,G,t,d]
        g, beta = g.float().transpose(1, 2), beta.float().transpose(1, 2)           # [b,G,t]
        b, G, R, t, dk = q.shape
        dv, c = v.size(-1), chunk
        pad = (-t) % c
        if pad:                                                                     # β = 0, g = 0: durum değişmez
            q, k, v = F.pad(q, (0, 0, 0, pad)), F.pad(k, (0, 0, 0, pad)), F.pad(v, (0, 0, 0, pad))
            g, beta = F.pad(g, (0, pad)), F.pad(beta, (0, pad))
        n = (t + pad) // c
        q, k, v = q.reshape(b, G, R, n, c, dk), k.reshape(b, G, n, c, dk), v.reshape(b, G, n, c, dv)
        g, beta = g.reshape(b, G, n, c).cumsum(-1), beta.reshape(b, G, n, c)
        decay = (g[..., :, None] - g[..., None, :]).tril().exp().tril()           # [b,G,n,c,c]
        k_beta = k * beta[..., None]
        w_inv = unit_lower_inverse((k_beta @ k.transpose(-1, -2) * decay).tril(-1))
        u, w = w_inv @ (v * beta[..., None]), w_inv @ (k_beta * g.exp()[..., None])
        attn = (q @ k[:, :, None].transpose(-1, -2)) * decay[:, :, None]           # [b,G,R,n,c,c]
        q_decay = q * g.exp()[:, :, None, :, :, None]
        k_tail = k * (g[..., -1:] - g).exp()[..., None]
        chunk_decay = g[..., -1].exp()[..., None, None]
        S = state.float() if state is not None else q.new_zeros(b, G, dk, dv)
        outs = []
        for i in range(n):
            v_new = u[:, :, i] - w[:, :, i] @ S                                     # [b,G,c,dv]
            outs.append(q_decay[:, :, :, i] @ S[:, :, None] + attn[:, :, :, i] @ v_new[:, :, None])
            S = S * chunk_decay[:, :, i] + k_tail[:, :, i].transpose(-1, -2) @ v_new
        o = torch.stack(outs, 3).reshape(b, G, R, n * c, dv)[:, :, :, :t]
    return o.permute(0, 3, 1, 2, 4), S


def window_attention(a, qr, kr, v):
    """Pencere + çapa dikkati, boş halkayla dizinin başından (HybridAttention._student'ın cache'siz yolu)."""
    b, h, t, d = qr.shape
    w, s, dev = a.window, a.sinks, qr.device
    vt = v.transpose(1, 2)
    pos = torch.arange(t, device=dev)
    ext_pos = torch.cat((torch.full((w,), -1, dtype=torch.long, device=dev), pos))
    ext_k = torch.cat((kr.new_zeros(b, kr.size(1), w, d), kr), 2)
    ext_v = torch.cat((vt.new_zeros(b, vt.size(1), w, d), vt), 2)
    n = min(s, t)
    return a._window(qr, ext_k, ext_v, ext_pos, kr[:, :, :n] if n else None, vt[:, :, :n] if n else None, pos)


def attn_teacher(a, x):
    """Öğretmen dikkat katmanının çıkışı (W_o sonrası), blok blok: x [1,T,d] → [1,T,d]."""
    q, k, v, qr, kr, gate = a._proj(x, 0)
    o = blockwise_attention(qr, kr, v.transpose(1, 2))
    return a._out(o.to(x.dtype), gate, x)


# --------------------------------------------------------------------------- bir katmanın verisi ve kaybı
@torch.no_grad()
def precompute(layer, h, P, wq):
    """Bir dizinin, eğitim boyunca değişmeyen her şeyi. h [T,d] (katman girdisi), P: kayıp konumları (≥ W, sıralı),
    wq: konum ağırlıkları. Döndürür (kayıt sözlüğü, öğretmenin dikkat çıkışı yt [T,d])."""
    a = layer.self_attn
    x = layer.input_layernorm(h[None])
    q, k, v, qr, kr, gate = a._proj(x, 0)
    o_t = blockwise_attention(qr, kr, v.transpose(1, 2))
    yt = a._out(o_t.to(x.dtype), gate, x)[0]
    del o_t
    o_w, lse = window_attention(a, qr, kr, v)                                       # [1,T,H,D], [1,H,T]
    owP = o_w[0, P].float()
    del o_w
    sg = torch.sigmoid(gate[0, P].float())                                          # [p, H·D]
    base = F.linear((owP.reshape(len(P), -1) * sg).to(x.dtype), a.o_proj.weight, a.o_proj.bias).float()
    yP = yt[P].float()
    sd = x.dtype
    d = dict(kw=k[0].to(sd), vw=v[0].to(sd), xw=x[0].to(sd), P=P, wq=wq.float(), qP=q[0, P].to(sd),
             xP=x[0, P].to(sd), owP=owP.to(sd), sgP=sg.to(sd), lseP=lse[0][:, P].T.float().contiguous(),
             rP=(yP - base), den=(wq.float() * yP.square().sum(-1)).sum())
    return {kk: vv.cpu() for kk, vv in d.items()}, yt


def seq_loss(a, d, seg: int, W_o, chunk: int = 64):
    """Bir dizinin göreli hatası (eski rel_mse ile aynı, P konumlarında, wq ağırlıklı). Zaman `seg`'lik parçalarda;
    parça sınırında yalnız hafıza durumu tutulur, parça geri yayılımda yeniden hesaplanır (gradyan tam)."""
    c = a.c
    G, R, D, W, S0 = c.hkv, c.h // c.hkv, c.dh, a.window, a.sinks
    T = d["kw"].size(0)
    P = d["P"]
    dev = W_o.device
    state = torch.zeros(1, G, D, D, device=dev)
    total = torch.zeros((), device=dev)

    def part(state, kw, vw, xw, n_zero, j0, sel, Pl):
        km = F.normalize(torch.einsum("tgd,gde->tge", kw.float(), a.fk), dim=-1)
        aa, bb = a.ab(xw.float()).chunk(2, dim=-1)
        gg = -a.A_log.exp() * F.softplus(aa + a.dt_bias) * a.decay_mask
        tok = torch.arange(j0, j0 + kw.size(0), device=dev)
        be = torch.sigmoid(bb) * (tok >= S0).float()[:, None]                       # çapa tokenler yazılmaz
        z = lambda u: F.pad(u, (0, 0) * (u.dim() - 1) + (n_zero, 0))                # pencere dolana kadar yazma yok
        wk, wv, wg, wb = z(km), z(vw.float()), z(gg), z(be)
        L = wk.size(0)
        qm = F.normalize(torch.einsum("phd,hde->phe", d_["qP"][sel].float(), a.fq), dim=-1) * D ** -0.5
        qf = torch.zeros(L, G, R, D, device=dev).index_copy(0, Pl, qm.reshape(-1, G, R, D))
        o, state = delta_rule_gqa(qf[None], wk[None], wv[None], wg[None], wb[None], chunk, state)
        om = o[0, Pl].reshape(-1, c.h, D)
        gam = torch.sigmoid(a.mix(d_["xP"][sel].float()) + a.mix_lse * (d_["lseP"][sel] - math.log(W)))
        delta = gam[..., None] * (a.mem_scale[:, None] * om - d_["owP"][sel].float())
        y = (delta.reshape(len(Pl), -1) * d_["sgP"][sel].float()) @ W_o.T
        return state, (d_["wq"][sel] * (y - d_["rP"][sel]).square().sum(-1)).sum()

    d_ = d
    for t0 in range(0, T, seg):
        t1 = min(T, t0 + seg)
        if t1 <= W:                                                                 # yazma da okuma da yok
            continue
        sel = (P >= t0) & (P < t1)
        j0, j1 = max(0, t0 - W), t1 - W
        n_zero = (t1 - t0) - (j1 - j0)
        Pl = P[sel] - t0
        args = (state, d["kw"][j0:j1], d["vw"][j0:j1], d["xw"][j0:j1], n_zero, j0, sel, Pl)
        if torch.is_grad_enabled():
            state, lp = checkpoint(part, *args, use_reentrant=False)
        else:
            state, lp = part(*args)
        total = total + lp
    return total / d["den"]


def window_only_loss(d) -> float:
    """Hafızasız (yalnız pencere + çapa) aynı kayıp: karşılaştırma tabanı."""
    return float((d["wq"] * d["rP"].float().square().sum(-1)).sum() / d["den"])


# --------------------------------------------------------------------------- veri
def calib_set(path, vocab: int, n: int, t: int, seed: int, tok, synth: float, W: int, frac: float, w_q: float):
    """n dizi × t token (wiki + sentetik hatırlama) ve her dizi için kayıp konumları P (≥ W) ile ağırlıkları."""
    data = np.memmap(path, dtype=np.uint32 if vocab >= 65536 else np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    n_syn = int(round(synth * n)) if tok is not None else 0
    syn_at = {(j * n) // max(n_syn, 1) for j in range(n_syn)}
    ids, pos = [], []
    for j in range(n):
        spans = []
        if j in syn_at:
            seq, spans = recall_sequence(tok, data, t, rng, n_facts=max(12, t // 4096), spans=True)
        else:
            s = int(rng.integers(0, len(data) - t - 1))
            seq = np.array(data[s:s + t], dtype=np.int64)
        m = np.zeros(t, dtype=np.float32)
        cand = np.arange(W, t)
        if len(cand):
            m[rng.choice(cand, size=max(1, int(frac * len(cand))), replace=False)] = 1.0
        for st, ln in spans:
            m[st:st + ln] = w_q                                                     # soru + cevap rakamları
        m[:W] = 0.0
        P = np.nonzero(m)[0]
        ids.append(torch.from_numpy(seq.astype(np.int64)))
        pos.append((torch.from_numpy(P), torch.from_numpy(m[P])))
    return torch.stack(ids), pos


class HiddenStore:
    """Gizli durumlar [n, T, d] diskte (memmap). bf16 numpy'da yok: uint16 olarak saklanır."""

    def __init__(self, path: Path, n: int, t: int, d: int, dtype, create: bool):
        self.dtype, self.bf = dtype, dtype == torch.bfloat16
        self.arr = np.memmap(path, dtype=np.uint16 if self.bf else np.float32, mode="w+" if create else "r+",
                             shape=(n, t, d))

    def get(self, j: int) -> torch.Tensor:
        x = torch.from_numpy(np.array(self.arr[j]))
        return x.view(torch.bfloat16) if self.bf else x

    def put(self, j: int, x: torch.Tensor) -> None:
        x = x.detach().cpu()
        self.arr[j] = (x.view(torch.int16).numpy().view(np.uint16) if self.bf else x.float().numpy())

    def flush(self) -> None:
        self.arr.flush()


# --------------------------------------------------------------------------- plan
def token_bytes(c: HybridConfig) -> int:
    """Estimated GPU bytes per token at the peak of precompute (one layer, one sequence): the q/gate projection,
    normed and roped q, teacher and window outputs (fp32) and the bf16 activations. An estimate, not a measurement:
    27B (h*dh = 6144, d = 5120) gives ~185 KB/token, i.e. ~12 GB at 64K, too much next to the weights on 16 GB."""
    return 24 * c.h * c.dh + 8 * c.d


def layer_bytes(c: HybridConfig) -> int:
    """bf16 weights of one dense layer (attention + MLP)."""
    return 2 * (c.d * (2 * c.h * c.dh + 2 * c.hkv * c.dh) + c.h * c.dh * c.d + 3 * c.d * c.inter)


def auto_plan(c: HybridConfig, vram_gb: float, seq: int | None, kalib: int | None, keep_every: int):
    attn = [i for i, k in enumerate(c.layer_types) if k == ATTN]
    keep = [f"layers.{i}" for n_, i in enumerate(attn) if keep_every and (n_ + 1) % keep_every == 0]
    if seq is None:      # the longest sequence whose estimated per-layer peak fits in ~75% of the GPU
        budget = 0.75 * vram_gb * 2**30 - layer_bytes(c) - 2**30
        seq = next((t for t in (65536, 32768, 16384) if t * token_bytes(c) <= budget), 8192)
    if kalib is None:
        kalib = max(8, min(32, int(2 ** 20 / seq * 16)))                            # ~1M token
    return dict(seq=seq, kalib=kalib, keep=keep)


def plan_report(c, p, W, frac, seg) -> str:
    T, n, d = p["seq"], p["kalib"], c.d
    gd = c.hkv * c.dh
    per = T * (d + 2 * gd) * 2 + int(frac * T) * (c.h * c.dh * 2 * 3 + d * 2 + d * 4)
    conv = sum(1 for i, k in enumerate(c.layer_types) if k == ATTN and f"layers.{i}" not in p["keep"])
    return (f"model: {c.n_layers} katman, {c.layer_types.count(ATTN)} tam dikkat → {conv} dönüştürülür, "
            f"{len(p['keep'])} kalır ({' '.join(p['keep']) or '-'})\n"
            f"kalibrasyon: {n} dizi × {T} token, pencere {W}, parça {seg}, kayıp oranı {frac}\n"
            f"disk: gizli durumlar {gb(n * T * d * 2)} + katman önbelleği {gb(n * per)} (katman bitince silinir)\n"
            f"GPU (tahmin): katman ağırlığı {gb(layer_bytes(c))} + dizi başına tepe {gb(T * token_bytes(c))}")


# --------------------------------------------------------------------------- eğitim
def cmd_egit(a) -> None:
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    gpu = dev.type == "cuda"
    dtype = torch.bfloat16 if gpu else torch.float32
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "motor.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    path = fetch_model(a.model)
    w = Weights(path)
    c = HybridConfig(w.config)
    vram = torch.cuda.get_device_properties(dev).total_memory / 2**30 if gpu else 0
    p = auto_plan(c, vram, a.seq, a.kalib, a.keep_every)
    keep = set(a.keep_global) if a.keep_global is not None else set(p["keep"])
    p["keep"] = sorted(keep, key=lambda s_: int(s_.split(".")[1]))
    T, n = p["seq"], p["kalib"]
    log(plan_report(c, p, a.window, a.kayip_orani, a.parca))

    st_file, h_file = out / "motor_durum.json", out / "gizli.bin"
    prog = json.loads(st_file.read_text()) if (a.devam and st_file.exists()) else {"next": 0}
    tok = load_tokenizer(a.model) if a.sentetik > 0 else None
    ids, pos = calib_set(a.train, c.vocab, n, T, a.seed, tok, a.sentetik, a.window, a.kayip_orani, a.soru_agirlik)
    H = HiddenStore(h_file, n, T, c.d, dtype, create=prog["next"] == 0)
    if prog["next"] == 0:
        emb = w.get("embed_tokens.weight").to(dtype)
        for j in range(n):
            H.put(j, emb[ids[j]])
        del emb
    conv, params, report = [], {}, {}
    if prog["next"] > 0 and (out / "donusum_param.pt").exists():
        stp = torch.load(out / "donusum_param.pt", map_location="cpu", weights_only=False)
        conv, params, report = stp["conv"], stp["params"], stp.get("meta", {}).get("rapor", {})
        log(f"devam: katman {prog['next']}'den sürülüyor ({len(conv)} katman hazır)")
    cache_dir = out / "katman_onbellek"
    pool = ThreadPoolExecutor(1)
    t_all, done_attn = time.time(), 0
    n_conv = sum(1 for i, k in enumerate(c.layer_types) if k == ATTN and f"layers.{i}" not in keep)

    for i, kind in enumerate(c.layer_types):
        if i < prog["next"]:
            continue
        t0 = time.time()
        name = f"layers.{i}"
        with torch.device("meta"):
            layer = HLayer(c, kind)
        materialize(layer, f"{name}.", w, dev, dtype)
        convert = kind == ATTN and name not in keep
        if convert:
            layer.self_attn.convert(a.window, a.sinks, n_archival(a.archival_frac, c.hkv, "long"), "long")
            cache_dir.mkdir(exist_ok=True)
        for j in range(n):                                     # ileri geçiş (öğretmen) + değişmeyen verinin kaydı
            h = H.get(j).to(dev)
            with torch.no_grad():
                if kind == ATTN:
                    if convert:
                        rec, yt = precompute(layer, h, pos[j][0].to(dev), pos[j][1].to(dev))
                        torch.save(rec, cache_dir / f"{j}.pt")
                    else:
                        yt = attn_teacher(layer.self_attn, layer.input_layernorm(h[None]))[0]
                    hm = h + yt
                    del yt
                    for r0 in range(0, T, a.parca):
                        hm[r0:r0 + a.parca] += layer.mlp(layer.post_attention_layernorm(hm[r0:r0 + a.parca]))
                    h = hm
                else:
                    cache, outs = {}, []
                    for r0 in range(0, T, a.parca):
                        outs.append(layer(h[None, r0:r0 + a.parca], cache, r0)[0])
                    h = torch.cat(outs)
            H.put(j, h)
            del h
        H.flush()
        t_fwd = time.time() - t0
        if convert:
            rep = train_layer(layer.self_attn, cache_dir, n, a, dev, pool, log, name)
            report[name] = rep
            at = layer.self_attn
            conv.append(dict(name=name, window=at.window, sinks=at.sinks, archival=int((at.decay_mask == 0).sum())))
            params.update({f"{name}.self_attn.{k}": v.detach().float().cpu() for k, v in at.named_parameters()
                           if k.split(".")[0] in at.NEW})
            for f in cache_dir.glob("*.pt"):
                f.unlink()
            done_attn += 1
        torch.save({"conv": conv, "params": params, "mtp_hidden": "normed",
                    "meta": {"model": a.model, "rapor": report, "son_katman": i, "motor": settings(a)}},
                   out / "donusum_param.pt")
        st_file.write_text(json.dumps({"next": i + 1}))
        del layer
        if gpu:
            torch.cuda.empty_cache()
        el = time.time() - t_all
        left = (el / max(done_attn, 1)) * (n_conv - done_attn) if done_attn else 0
        peak = f" | tepe GPU {gb(torch.cuda.max_memory_allocated())}" if gpu else ""
        sw = swap_gb()
        parts = f" (ileri {t_fwd:.0f} + eğitim {time.time() - t0 - t_fwd:.0f})" if convert else ""
        log(f"katman {i:2d}/{c.n_layers} ({'dönüşüm' if convert else 'dikkat' if kind == ATTN else 'DeltaNet'}) "
            f"{time.time() - t0:5.0f} sn{parts} | {eta(left) if done_attn else 'tahmin ilk dönüşümden sonra'}{peak}"
            f"{f' | takas {sw:.1f} GB' if sw > 0.5 else ''}")
    log(f"bitti: {out / 'donusum_param.pt'} ({len(conv)} katman, toplam {(time.time() - t_all) / 3600:.2f} sa)")
    h_file.unlink(missing_ok=True)
    st_file.unlink(missing_ok=True)


def train_layer(at, cache_dir: Path, n: int, a, dev, pool, log, name):
    W_o = at.o_proj.weight.float()
    params = at.new_params("all")
    for p_ in params:
        p_.requires_grad_(True)
    n_val = max(1, n // 16)
    tr = np.arange(n - n_val)
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)

    def load(j):
        d = torch.load(cache_dir / f"{j}.pt", map_location="cpu", weights_only=False, mmap=True)
        return {k: v.to(dev, non_blocking=True) for k, v in d.items()}

    def val():
        tot = base = 0.0
        with torch.no_grad():
            for j in range(n - n_val, n):
                d = load(j)
                tot += float(seq_loss(at, d, a.parca, W_o, a.delta_parca))
                base += window_only_loss(d)
        return tot / n_val, base / n_val

    v0, vw = val()
    order = rng.choice(tr, a.adim, replace=True)
    nxt = pool.submit(load, int(order[0]))
    t0 = time.time()
    for step in range(1, a.adim + 1):
        for g_ in opt.param_groups:
            g_["lr"] = a.lr * 0.5 * (1 + math.cos(math.pi * step / a.adim))
        d = nxt.result()
        if step < a.adim:
            nxt = pool.submit(load, int(order[step]))                                # bir sonraki dizi arka planda
        loss = seq_loss(at, d, a.parca, W_o, a.delta_parca)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 50 == 0 or step == a.adim:
            per = (time.time() - t0) / step
            log(f"    {name} adım {step}/{a.adim} | kayıp {loss.item():.4f} | {per:.2f} sn/adım | "
                f"bu katman {eta(per * (a.adim - step))}")
    v1, _ = val()
    log(f"  {name}: göreli hata (doğrulama) yalnız pencere {vw:.4f} | eğitimsiz {v0:.4f} → eğitilmiş {v1:.4f}")
    for p_ in params:
        p_.requires_grad_(False)
    return {"pencere": vw, "egitimsiz": v0, "egitilmis": v1}


def settings(a) -> dict:
    """Run settings as plain values (the argparse namespace also holds the command function, which must not be pickled:
    run as `python -m`, it pickles as __main__.cmd_egit and no other program could load the file)."""
    return {k: v for k, v in vars(a).items() if k != "fn"}


def swap_gb() -> float:
    """Swap in use (GB, Linux): the run slows down a lot once the hidden states no longer fit in RAM."""
    try:
        m = dict(l.split(":", 1) for l in Path("/proc/meminfo").read_text().splitlines())
        return (int(m["SwapTotal"].split()[0]) - int(m["SwapFree"].split()[0])) / 2**20
    except (OSError, KeyError, ValueError):
        return 0.0


def cmd_onar(a) -> None:
    """Repair a donusum_param.pt saved before the fix (it held __main__.cmd_egit). Runs as __main__, so it loads."""
    for f in a.files:
        st = torch.load(f, map_location="cpu", weights_only=False)
        mo = st.get("meta", {}).get("motor")
        if isinstance(mo, dict) and "fn" in mo:
            mo.pop("fn")
            torch.save(st, f)
            print(f"onarıldı: {f}")
        else:
            print(f"sorun yok: {f}")


def cmd_plan(a) -> None:
    c = HybridConfig(Weights(fetch_model(a.model)).config)
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 16
    p = auto_plan(c, vram, a.seq, a.kalib, a.keep_every)
    print(plan_report(c, p, a.window, a.kayip_orani, a.parca))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("plan", "egit"):
        s = sub.add_parser(nm)
        s.add_argument("--model", required=True)
        s.add_argument("--seq", type=int, default=None, help="dizi uzunluğu (varsayılan: GPU belleğine göre, 16 GB → 65536)")
        s.add_argument("--kalib", type=int, default=None, help="dizi sayısı (varsayılan: ~1M token)")
        s.add_argument("--keep-every", type=int, default=4, help="her k. tam dikkat katmanı dönüştürülmez (0: hepsi dönüşür)")
        s.add_argument("--window", type=int, default=2048)
        s.add_argument("--kayip-orani", type=float, default=0.125, help="kaybın hesaplandığı rastgele konum oranı")
        s.add_argument("--parca", type=int, default=8192, help="zaman parçası (bellek ↔ hız)")
        s.add_argument("--delta-parca", type=int, default=128,
                       help="delta kuralı blok boyu: büyük blok = daha az sıralı adım (sonuç aynı, yalnız yuvarlama)")
        if nm == "egit":
            s.add_argument("--train", required=True)
            s.add_argument("--out", required=True)
            s.add_argument("--keep-global", nargs="*", default=None, help="elle: dönüştürülmeyecek katmanlar")
            s.add_argument("--sinks", type=int, default=4)
            s.add_argument("--archival-frac", type=float, default=0.25)
            s.add_argument("--adim", type=int, default=300, help="dönüştürülen katman başına adım")
            s.add_argument("--lr", type=float, default=3e-3)
            s.add_argument("--sentetik", type=float, default=0.5, help="sentetik hatırlama dizisi oranı")
            s.add_argument("--soru-agirlik", type=float, default=4.0, help="soru + cevap konumlarının kayıp ağırlığı")
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--device", default=None)
            s.add_argument("--devam", action="store_true", help="yarıda kalan koşuyu kaldığı katmandan sür")
        s.set_defaults(fn=cmd_plan if nm == "plan" else cmd_egit)
    s = sub.add_parser("onar", help="eski donusum_param.pt dosyasını başka programların açabileceği hale getir")
    s.add_argument("files", nargs="+")
    s.set_defaults(fn=cmd_onar)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
