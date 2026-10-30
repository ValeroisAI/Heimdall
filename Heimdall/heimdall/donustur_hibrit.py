"""
donustur_hibrit.py — Qwen3.5 / 3.6 / 3.8 (dense) modellerinin yalnız TAM DİKKAT katmanlarını sabit bellekli katmana çevirir.
Model koduna bakınız: hibrit.py. DeltaNet ve FFN katmanlarına (ağır hesabın ~%94'ü) dokunulmaz.

16 GB kartta 27B için tasarım:
  * Aşama A KATMAN KATMAN akar: her seferinde diskte tek katman bf16 GPU'ya gelir (~0.6 GB). Kalibrasyon dizilerinin gizli
    durumları GPU'da tutulur ve katmandan katmana ilerletilir (GPTQ'daki gibi). Tam dikkat katmanına gelince yeni
    parametreler öğretmenin çıkışını taklit edecek şekilde eğitilir. Tepe bellek ~6-8 GB, nicemleme YOK (öğretmen tam bf16).
    Sonda bf16 öğretmenin token olasılıklarının ilk k'sı (top-k) diske yazılır; aşama B bunları kullanır.
  * Aşama B (isteğe bağlı) uçtan uca: 4-bit model (g128, ~13 GB) + yeni parametreler, bf16 öğretmenin top-k'sına KL.
    Yeni parametreler böylece 4-bit hatasının bir kısmını da telafi eder. Ara aktivasyonlar CPU RAM'e taşınır.
  * Çıkarım: W'den kısa bağlamda dönüştürülmüş model öğretmenle BİREBİR aynıdır (yalnız pencere çalışır). Bağlam ne kadar
    uzarsa uzasın cache sabittir (27B, W=1024: ~0.3 GB; orijinal 262K'da 16 GB).

    python -m heimdall.donustur_hibrit hazirla  --model Qwen/Qwen3.8-27B --veri veri/qwen_wiki
    python -m heimdall.donustur_hibrit asama-a  --model Qwen/Qwen3.8-27B --train veri/qwen_wiki/wiki_train.bin --out runs/q38
    python -m heimdall.donustur_hibrit asama-b  --model Qwen/Qwen3.8-27B --out runs/q38
    python -m heimdall.donustur_hibrit degerlendir --model Qwen/Qwen3.8-27B --out runs/q38 --test veri/qwen_wiki/wiki_test.bin
    python -m heimdall.donustur_hibrit sohbet   --model Qwen/Qwen3.8-27B --out runs/q38 --mtp
"""

import os

os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import argparse
import contextlib
import json
import math
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .donustur import fetch_model, prepare_wikitext
from .hibrit import ATTN, MTP, HLayer, HybridConfig, HybridLM, Weights, ZNorm, materialize

FILLER = ("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
          "The river runs to the sea. A small bird sat on the old wall. ")


def setup(args):
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    gpu = dev.type == "cuda"
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if gpu else contextlib.nullcontext()
    return dev, gpu, amp, (torch.bfloat16 if gpu else torch.float32)


def gb(n: float) -> str:
    return f"{n / 2**30:.1f} GB"


def peak(gpu: bool) -> str:
    return f" | tepe GPU {gb(torch.cuda.max_memory_allocated())}" if gpu else ""


def plan_text(c: HybridConfig, window: int, q4_group: int = 128) -> str:
    """Bellek özeti: ağırlıklar (4-bit g128, embedding CPU'da) ve cache (orijinal ↔ dönüşüm)."""
    d, n_attn, n_lin = c.d, c.layer_types.count(ATTN), c.layer_types.count("linear_attention")
    kd, vd = c.lin_hk * c.lin_dk, c.lin_hv * c.lin_dv
    lin = d * (2 * kd + vd) + d * vd + vd * d
    att = d * c.h * c.dh * (2 if c.gate else 1) + 2 * d * c.hkv * c.dh + c.h * c.dh * d
    ffn = 3 * d * c.inter
    mats = n_lin * lin + n_attn * att + c.n_layers * ffn + (0 if c.tie else c.vocab * d)
    mtp = (att + ffn + 2 * d * d) if c.mtp_layers else 0
    bits = 4 + 32 / q4_group                                        # 4-bit + grup başına fp16 ölçek ve sıfır
    w4 = (mats + mtp) * bits / 8
    gdn_state = n_lin * (c.lin_hv * c.lin_dk * c.lin_dv * 4 + (c.conv_k - 1) * (2 * kd + vd) * 2)
    conv_state = (n_attn + (1 if c.mtp_layers else 0)) * (c.hkv * window * c.dh * (2 * 2 + 4) + c.hkv * c.dh ** 2 * 4)
    kv = c.kv_bytes_per_token()
    return (f"model: {c.n_layers} katman ({n_lin} DeltaNet + {n_attn} tam dikkat), d={d}, {(mats + mtp + c.vocab * d) / 1e9:.1f}B param\n"
            f"4-bit (g{q4_group}) ağırlık GPU'da ≈ {gb(w4)} (+ embedding CPU'da {gb(c.vocab * d * 2)})\n"
            f"orijinal KV cache: token başına {kv // 1024} KB → 32K: {gb(kv * 32768)}, 128K: {gb(kv * 131072)}, "
            f"262K: {gb(kv * 262144)}\n"
            f"dönüşüm cache (her uzunlukta sabit): {gb(gdn_state + conv_state)} (pencere {window})")


def calib_ids(path, vocab: int, n: int, t: int, seed: int) -> torch.Tensor:
    data = np.memmap(path, dtype=np.uint32 if vocab >= 65536 else np.uint16, mode="r")
    starts = np.random.default_rng(seed).integers(0, len(data) - t - 1, n)
    return torch.from_numpy(np.stack([data[s:s + t] for s in starts]).astype(np.int64))


def rel_mse(ys, yt):
    return (ys.float() - yt.float()).square().mean() / yt.float().square().mean().clamp(min=1e-8)


# --------------------------------------------------------------------------- aşama A: katman katman akış
def train_attention(name, layer, H, args, dev, amp, gpu, log):
    """Bir tam dikkat katmanının yeni parametrelerini öğretmen çıkışına (W_o sonrası) eğitir."""
    a = layer.self_attn
    a.convert(args.window, args.sinks, int(args.archival_frac * a.c.hkv))
    params = a.new_params("all")
    for p in params:
        p.requires_grad_(True)
    n = H.size(0)
    n_val = max(1, n // 16)
    tr = np.arange(n - n_val)
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    def val(mode: str) -> float:
        a.mode = mode
        tot = 0.0
        with torch.no_grad(), amp:
            for j in range(n - n_val, n):
                yt, ys = a.distill_pair(layer.input_layernorm(H[j:j + 1].to(dev)))
                tot += rel_mse(ys, yt).item()
        a.mode = "student"
        return tot / n_val

    v_win, v0 = val("window"), val("student")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        j = torch.from_numpy(rng.choice(tr, args.batch, replace=False))
        with amp:
            yt, ys = a.distill_pair(layer.input_layernorm(H[j].to(dev)))
            loss = rel_mse(ys, yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 50 == 0 or step == args.steps:
            log(f"    {name} adım {step}/{args.steps} | kayıp {loss.item():.4f} | {time.time() - t0:.0f} sn{peak(gpu)}")
    v1 = val("student")
    log(f"  {name}: göreli hata (doğrulama) yalnız pencere {v_win:.4f} | eğitimsiz {v0:.4f} → eğitilmiş {v1:.4f}")
    for p in params:
        p.requires_grad_(False)
    a.mode = "teacher"
    return {"pencere": v_win, "egitimsiz": v0, "egitilmis": v1}


@torch.no_grad()
def propagate(layer, H, batch: int, dev, amp) -> None:
    for j in range(0, H.size(0), batch):
        with amp:
            H[j:j + batch] = layer(H[j:j + batch].to(dev)).to(H.dtype)


def cmd_asama_a(args) -> None:
    dev, gpu, amp, dtype = setup(args)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "asama_a.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    path = fetch_model(args.model)
    w = Weights(path)
    c = HybridConfig(w.config)
    log(plan_text(c, args.window))
    ids = calib_ids(args.train, c.vocab, args.kalib, args.seq_len, args.seed)
    emb = w.get("embed_tokens.weight").to(dtype)                           # CPU
    h_dev = dev if (gpu and not args.h_cpu) else torch.device("cpu")
    H = emb[ids].to(h_dev)
    log(f"kalibrasyon: {ids.size(0)} dizi × {ids.size(1)} token, gizli durumlar {gb(H.numel() * H.element_size())} "
        f"({h_dev.type}) | pencere {args.window} + {args.sinks} çapa")
    keep = set(args.keep_global)
    report, conv, params = {}, [], {}
    t_all = time.time()
    for i, kind in enumerate(c.layer_types):
        t0 = time.time()
        with torch.device("meta"):
            layer = HLayer(c, kind)
        materialize(layer, f"layers.{i}.", w, dev, dtype)
        name = f"layers.{i}"
        if kind == ATTN and name not in keep:
            report[name] = train_attention(name, layer, H, args, dev, amp, gpu, log)
            a = layer.self_attn
            conv.append(dict(name=name, window=a.window, sinks=a.sinks, archival=int((a.decay_mask == 0).sum())))
            params.update({f"{name}.self_attn.{n}": p.detach().float().cpu() for n, p in a.named_parameters()
                           if n.split(".")[0] in a.NEW})
        propagate(layer, H, args.batch, dev, amp)
        del layer
        if gpu:
            torch.cuda.empty_cache()
        log(f"katman {i:2d} ({'dikkat ' if kind == ATTN else 'DeltaNet'}) {time.time() - t0:5.0f} sn{peak(gpu)}")
        torch.save({"conv": conv, "params": params, "mtp_hidden": "normed",
                    "meta": {"model": args.model, "rapor": report, "son_katman": i}}, out / "donusum_param.pt")

    # öğretmen (bf16) top-k olasılıkları: aşama B'nin hedefi
    with torch.device("meta"):
        head = nn.ModuleDict({"norm": ZNorm(c.d, c.eps), "lm_head": nn.Linear(c.d, c.vocab, bias=False)})
    materialize(head["norm"], "norm.", w, dev, dtype)
    if w.has("lm_head.weight"):
        materialize(head["lm_head"], "lm_head.", w, dev, dtype)
    else:
        head["lm_head"].weight = nn.Parameter(emb.to(dev), requires_grad=False)
    k = args.topk
    top_i = torch.zeros(ids.size(0), ids.size(1), k, dtype=torch.int32)
    top_lp = torch.zeros(ids.size(0), ids.size(1), k, dtype=torch.float16)
    with torch.no_grad(), amp:
        for j in range(ids.size(0)):
            hn = head["norm"](H[j:j + 1].to(dev))
            for s in range(0, ids.size(1), 512):
                lp = F.log_softmax(head["lm_head"](hn[:, s:s + 512]).float(), -1)
                v, ix = lp.topk(k, -1)
                top_i[j, s:s + 512], top_lp[j, s:s + 512] = ix[0].int().cpu(), v[0].half().cpu()
    torch.save({"ids": ids, "top_i": top_i, "top_lp": top_lp}, out / "ogretmen_topk.pt")
    log(f"öğretmen top-{k} kaydedildi: {out / 'ogretmen_topk.pt'}")

    mtp_hidden = "normed"
    if w.has("mtp.fc.weight") and not args.no_mtp:
        with torch.device("meta"):
            mtp = MTP(c)
        materialize(mtp, "mtp.", w, dev, dtype)
        with torch.no_grad(), amp:   # MTP'ye giden gizli durum: ölçerek seç (taslak isabeti)
            acc = {}
            j = ids.size(0) - 1
            hr = H[j:j + 1].to(dev)
            hn = head["norm"](hr)
            e = emb[ids[j:j + 1, 1:]].to(dev)
            for mode, hh in (("normed", hn), ("raw", hr)):
                lg = head["lm_head"](mtp.norm(mtp.layers[0](mtp.inputs(e, hh[:, :-1]))))
                acc[mode] = (lg[0, :-1].argmax(-1).cpu() == ids[j, 2:]).float().mean().item()
            mtp_hidden = max(acc, key=acc.get)
            log(f"MTP taslak isabeti (x_(i+2), öğretmen): {acc} → giriş = {mtp_hidden}")
            for jj in range(ids.size(0)):                                    # H ← MTP katmanının girdisi
                hr = H[jj:jj + 1].to(dev)
                hh = head["norm"](hr) if mtp_hidden == "normed" else hr
                H[jj:jj + 1, :-1] = mtp.inputs(emb[ids[jj:jj + 1, 1:]].to(dev), hh[:, :-1]).to(H.dtype)
        name = "mtp.layers.0"
        report[name] = train_attention(name, mtp.layers[0], H[:, :-1], args, dev, amp, gpu, log)
        a = mtp.layers[0].self_attn
        conv.append(dict(name=name, window=a.window, sinks=a.sinks, archival=int((a.decay_mask == 0).sum())))
        params.update({f"{name}.self_attn.{n}": p.detach().float().cpu() for n, p in a.named_parameters()
                       if n.split(".")[0] in a.NEW})
    torch.save({"conv": conv, "params": params, "mtp_hidden": mtp_hidden,
                "meta": {"model": args.model, "rapor": report, "args": {k: v for k, v in vars(args).items() if k != "fn"}}}, out / "donusum_param.pt")
    log(f"aşama A bitti: {(time.time() - t_all) / 60:.1f} dk → {out / 'donusum_param.pt'}")


# --------------------------------------------------------------------------- aşama B: uçtan uca (4-bit model)
def load_model(args, dev, gpu, dtype, mtp: bool = True) -> HybridLM:
    path = fetch_model(args.model)
    q4 = args.q4 and gpu or args.q4_cpu
    out = Path(args.out)
    q4c = (out / f"q4_g{args.q4_group}.pt") if q4 and not args.no_q4_cache else None
    t0 = time.time()
    m = HybridLM.load(path, device=dev, dtype=dtype, q4_group=args.q4_group if q4 else 0,
                      embed_device="cpu" if gpu else None, mtp=mtp, q4_cache=q4c)
    print(f"model yüklendi ({'4-bit g' + str(args.q4_group) if q4 else str(dtype)}) {time.time() - t0:.0f} sn{peak(gpu)}",
          flush=True)
    return m


def sparse_kl(h, lm_head, t_i, t_lp, chunk: int = 256):
    """Öğretmenin top-k olasılıklarına KL (+ kalan kütle tek terim). Logitler parça parça, geri yayılımda yeniden hesaplanır."""
    def part(hc, ix, lp):
        ls = F.log_softmax(lm_head(hc).float(), -1).gather(-1, ix)
        pt = lp.float().exp()
        tail_t = (1 - pt.sum(-1)).clamp(min=1e-6)
        tail_s = (1 - ls.exp().sum(-1)).clamp(min=1e-6)
        return ((pt * (lp.float() - ls)).sum(-1) + tail_t * (tail_t.log() - tail_s.log())).sum()
    tot = 0.0
    for s in range(0, h.size(1), chunk):
        tot = tot + checkpoint(part, h[:, s:s + chunk], t_i[:, s:s + chunk], t_lp[:, s:s + chunk], use_reentrant=False)
    return tot / (h.size(0) * h.size(1))


def cmd_asama_b(args) -> None:
    dev, gpu, amp, dtype = setup(args)
    out = Path(args.out)
    st = torch.load(out / "donusum_param.pt", map_location="cpu", weights_only=False)
    tk = torch.load(out / "ogretmen_topk.pt", map_location="cpu", weights_only=False)
    m = load_model(args, dev, gpu, dtype, mtp=False)
    m.apply_conversion(st)
    m.grad_ckpt = True
    params = m.new_params(args.params)
    for p in params:
        p.requires_grad_(True)
    print(f"aşama B: {sum(p.numel() for p in params) / 1e6:.1f}M eğitilen parametre ({args.params}) | "
          f"dizi {args.seq_len} | {args.steps} adım", flush=True)
    t = min(args.seq_len, tk["ids"].size(1))
    n = tk["ids"].size(0)
    n_val = max(1, n // 16)
    rng = np.random.default_rng(args.seed)
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    offload = torch.autograd.graph.save_on_cpu(pin_memory=True) if gpu else contextlib.nullcontext()

    def loss_on(j):
        ids = tk["ids"][j:j + 1, :t].to(dev)
        with amp:
            h, _ = m.hidden(ids)
            return sparse_kl(h, m.lm_head, tk["top_i"][j:j + 1, :t].long().to(dev), tk["top_lp"][j:j + 1, :t].to(dev))

    @torch.no_grad()
    def val():
        m.grad_ckpt = False
        v = sum(loss_on(j).item() for j in range(n - n_val, n)) / n_val
        m.grad_ckpt = True
        return v

    v0 = val()
    print(f"  doğrulama KL (bf16 öğretmene) başta {v0:.4f}", flush=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        with offload:
            loss = loss_on(int(rng.integers(0, n - n_val)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == args.steps:
            print(f"  adım {step}/{args.steps} | KL {loss.item():.4f} | {(time.time() - t0) / step:.1f} sn/adım{peak(gpu)}",
                  flush=True)
    v1 = val()
    print(f"  doğrulama KL: {v0:.4f} → {v1:.4f}", flush=True)
    if not (out / "donusum_param_A.pt").exists():
        (out / "donusum_param.pt").rename(out / "donusum_param_A.pt")
    new = m.conversion_state(dict(st["meta"], asama_b={"kl_once": v0, "kl_sonra": v1, "adim": args.steps}))
    for k_, v_ in st["params"].items():         # MTP katmanı aşama B'de yüklenmedi: aşama A değerleri korunur
        new["params"].setdefault(k_, v_)
    new["conv"] += [cv for cv in st["conv"] if cv["name"].startswith("mtp.")]
    new["mtp_hidden"] = st.get("mtp_hidden", "normed")
    torch.save(new, out / "donusum_param.pt")
    print(f"kaydedildi: {out / 'donusum_param.pt'} (aşama A sürümü: donusum_param_A.pt)", flush=True)


# --------------------------------------------------------------------------- değerlendirme
@torch.no_grad()
def nll_curve(m, wins, chunk: int, amp) -> torch.Tensor:
    tot = torch.zeros(wins.size(1) - 1)
    for w in wins:
        cache = m.new_cache()
        ids = w[None, :-1].to(m.device)
        tgt = w[1:]
        nll = []
        for a in range(0, ids.size(1), chunk):
            with amp:
                hn, _ = m.hidden(ids[:, a:a + chunk], cache)
                for s in range(0, hn.size(1), 512):
                    lg = m.logits(hn[:, s:s + 512])[0]
                    nll.append(F.cross_entropy(lg, tgt[a + s:a + s + lg.size(0)].to(lg.device), reduction="none").cpu())
        tot += torch.cat(nll)
    return tot / wins.size(0)


def buckets(length: int, window: int):
    edges = sorted({0, window, 2 * window, 4096, 8192, 16384, 32768, length} & set(range(0, length + 1)))
    return [(a, b) for a, b in zip(edges[:-1], edges[1:])]


@torch.no_grad()
def passkey(m, tok, length: int, depth: float, key: str, chunk: int, amp) -> bool:
    needle = tok.encode(f" The pass key is {key}. Remember it. {key} is the pass key. ").ids
    question = tok.encode(" What is the pass key? The pass key is").ids
    filler = tok.encode(FILLER).ids
    body = max(length - len(needle) - len(question), 8)
    ids = (filler * (body // len(filler) + 1))[:body]
    cut = int(depth * len(ids))
    ids = torch.tensor([ids[:cut] + needle + ids[cut:] + question])
    with amp:
        out, _ = m.generate(ids, max_new=8, chunk=chunk)
    return key in tok.decode(out).replace(" ", "")


def cmd_degerlendir(args) -> None:
    from tokenizers import Tokenizer
    dev, gpu, amp, dtype = setup(args)
    out = Path(args.out)
    m = load_model(args, dev, gpu, dtype, mtp=args.mtp)
    st = torch.load(out / "donusum_param.pt", map_location="cpu", weights_only=False)
    m.apply_conversion(st)
    w = st["conv"][0]["window"]
    print(plan_text(m.c, w, args.q4_group), flush=True)
    res = {"args": {k: v for k, v in vars(args).items() if k != "fn"}}
    if args.test:
        data = np.memmap(args.test, dtype=np.uint32 if m.c.vocab >= 65536 else np.uint16, mode="r")
        rng = np.random.default_rng(0)
        starts = rng.integers(0, len(data) - args.eval_len - 2, args.n)
        wins = torch.from_numpy(np.stack([data[s:s + args.eval_len + 1] for s in starts]).astype(np.int64))
        bk = buckets(args.eval_len, w)
        print(f"\nkonuma göre perplexity ({args.n} pencere × {args.eval_len} token):", flush=True)
        print(f"{'':16s}" + "".join(f"{f'{a}-{b}':>13s}" for a, b in bk) + f"{'tümü':>9s}")
        for mode, label in (("teacher", "öğretmen"), ("window", "yalnız pencere"), ("student", "dönüşüm")):
            m.set_mode(mode)
            try:
                nll = nll_curve(m, wins, args.chunk, amp)
            except torch.OutOfMemoryError:
                print(f"{label:16s} BELLEK YETMEDİ (KV cache)")
                torch.cuda.empty_cache()
                continue
            row = {f"{a}-{b}": math.exp(nll[a:b].mean()) for a, b in bk}
            row["tumu"] = math.exp(nll.mean())
            res[mode] = row
            print(f"{label:16s}" + "".join(f"{v:13.2f}" for v in row.values()), flush=True)
        m.set_mode("student")
    if args.sifre:
        tok = Tokenizer.from_file(str(Path(fetch_model(args.model)) / "tokenizer.json"))
        rng = np.random.default_rng(1)
        print("\nparola (passkey) testi: derinlik 10/50/90%, uzunluk başına "
              f"{args.deneme} deneme", flush=True)
        for mode, label in (("student", "dönüşüm"), ("teacher", "öğretmen")):
            for length in args.sifre:
                ok, tried = 0, 0
                if gpu:
                    torch.cuda.reset_peak_memory_stats()
                try:
                    m.set_mode(mode)
                    for _ in range(args.deneme):
                        for d in (0.1, 0.5, 0.9):
                            ok += passkey(m, tok, length, d, str(int(rng.integers(10000, 99999))), args.chunk, amp)
                            tried += 1
                    print(f"  {label:9s} {length:7d} token: {ok}/{tried}{peak(gpu)}", flush=True)
                except torch.OutOfMemoryError:
                    print(f"  {label:9s} {length:7d} token: BELLEK YETMEDİ", flush=True)
                    torch.cuda.empty_cache()
                res.setdefault("sifre", {}).setdefault(mode, {})[length] = [ok, tried]
        m.set_mode("student")
    (out / "degerlendirme.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\nyazıldı: {out / 'degerlendirme.json'}")


# --------------------------------------------------------------------------- sohbet
def cmd_sohbet(args) -> None:
    from tokenizers import Tokenizer
    dev, gpu, amp, dtype = setup(args)
    out = Path(args.out)
    m = load_model(args, dev, gpu, dtype, mtp=args.mtp)
    if not args.ogretmen:
        m.apply_conversion(torch.load(out / "donusum_param.pt", map_location="cpu", weights_only=False))
    tok = Tokenizer.from_file(str(Path(fetch_model(args.model)) / "tokenizer.json"))
    im_end = tok.token_to_id("<|im_end|>")
    stop = {t for t in (im_end, tok.token_to_id("<|endoftext|>"), m.c.eos) if t is not None}
    cache, first = m.new_cache(), True
    print(f"{'öğretmen (orijinal)' if args.ogretmen else 'dönüştürülmüş model'} hazır. Çıkmak için boş satır.\n", flush=True)
    while True:
        try:
            msg = input("sen> ").strip()
        except EOFError:
            break
        if not msg:
            break
        text = (f"<|im_start|>system\n{args.sistem}<|im_end|>\n" if first and args.sistem else "")
        text += f"<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n"
        if not args.dusun:
            text += "<think>\n\n</think>\n\n"
        first = False
        ids = torch.tensor([tok.encode(text).ids])
        stats = {}
        t0 = time.time()
        with amp:
            new, cache = m.generate(ids, max_new=args.max_new, stop=stop, chunk=args.chunk, cache=cache,
                                    mtp=args.mtp, stats=stats)
        dt = time.time() - t0
        if new and new[-1] in stop:
            new = new[:-1]
        print(f"model> {tok.decode(new)}\n")
        acc = f" | MTP kabul {stats['kabul']}/{stats['taslak']}" if args.mtp and stats.get("taslak") else ""
        print(f"[{len(new)} token, {len(new) / dt:.1f} tok/s | bağlam {cache['pos']} token | "
              f"cache {m.cache_bytes(cache) / 2**20:.0f} MB{acc}{peak(gpu)}]\n", flush=True)
        if im_end is not None:               # asistan turunu kapat: sonraki tur aynı cache'in üstüne
            with amp:
                m.prefill(torch.tensor([[im_end] + tok.encode("\n").ids]), cache, args.chunk,
                          cache.get("mtp_state") if args.mtp else None)


# --------------------------------------------------------------------------- komut satırı
def cmd_hazirla(args) -> None:
    path = fetch_model(args.model)
    c = HybridConfig(json.loads((Path(path) / "config.json").read_text()))
    tr, te = prepare_wikitext(path, args.veri, int(args.tokens), c.eos)
    print(f"hazır: {tr}, {te} (ayırıcı {c.eos})")


def cmd_plan(args) -> None:
    path = fetch_model(args.model) if Path(args.model).exists() else None
    if path:
        hf = json.loads((Path(path) / "config.json").read_text())
    else:
        from huggingface_hub import hf_hub_download
        hf = json.loads(Path(hf_hub_download(args.model, "config.json")).read_text())
    print(plan_text(HybridConfig(hf), args.window, args.q4_group))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s, out=True):
        s.add_argument("--model", required=True, help="HF adı ya da klasör (ör. Qwen/Qwen3.8-27B)")
        if out:
            s.add_argument("--out", required=True)
        s.add_argument("--device")
        s.add_argument("--seed", type=int, default=0)

    def model_load(s):
        s.add_argument("--q4", action=argparse.BooleanOptionalAction, default=True, help="GPU'da 4-bit (varsayılan açık)")
        s.add_argument("--q4-cpu", action="store_true", help="CPU'da da 4-bit (test)")
        s.add_argument("--q4-group", type=int, default=128)
        s.add_argument("--no-q4-cache", action="store_true", help="nicemlenmiş ağırlıkları --out'a kaydetme")
        s.add_argument("--chunk", type=int, default=2048, help="prefill parça boyutu")

    s = sub.add_parser("plan", help="bellek özeti (indirmeden)")
    s.add_argument("--model", required=True)
    s.add_argument("--window", type=int, default=1024)
    s.add_argument("--q4-group", type=int, default=128)
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser("hazirla", help="wikitext-103'ü modelin tokenizer'ıyla .bin'e çevir")
    common(s, out=False)
    s.add_argument("--veri", required=True)
    s.add_argument("--tokens", type=float, default=20e6)
    s.set_defaults(fn=cmd_hazirla)

    s = sub.add_parser("asama-a", help="katman katman dönüşüm (bf16 öğretmen, düşük bellek)")
    common(s)
    s.add_argument("--train", required=True, help="eğitim .bin (öğretmen tokenizer'ı)")
    s.add_argument("--window", type=int, default=1024)
    s.add_argument("--sinks", type=int, default=4)
    s.add_argument("--archival-frac", type=float, default=0.25, help="hiç unutmayan hafıza kafası oranı")
    s.add_argument("--keep-global", nargs="*", default=[], help="dönüştürülmeyecek katmanlar, ör. layers.63")
    s.add_argument("--kalib", type=int, default=64, help="kalibrasyon dizisi sayısı")
    s.add_argument("--seq-len", type=int, default=4096)
    s.add_argument("--steps", type=int, default=400, help="dikkat katmanı başına adım")
    s.add_argument("--batch", type=int, default=1)
    s.add_argument("--lr", type=float, default=3e-3)
    s.add_argument("--topk", type=int, default=32)
    s.add_argument("--h-cpu", action="store_true", help="gizli durumları CPU RAM'de tut (GPU belleği azsa)")
    s.add_argument("--no-mtp", action="store_true")
    s.set_defaults(fn=cmd_asama_a)

    s = sub.add_parser("asama-b", help="uçtan uca KL (4-bit model, bf16 öğretmen top-k)")
    common(s)
    model_load(s)
    s.add_argument("--params", choices=["gates", "all"], default="gates",
                   help="gates: yalnız kapılar/bozunma (az bellek), all: + fq/fk")
    s.add_argument("--seq-len", type=int, default=2048)
    s.add_argument("--steps", type=int, default=200)
    s.add_argument("--lr", type=float, default=1e-3)
    s.set_defaults(fn=cmd_asama_b)

    s = sub.add_parser("degerlendir", help="perplexity (konuma göre) + parola testi + bellek")
    common(s)
    model_load(s)
    s.add_argument("--test", help="test .bin")
    s.add_argument("--eval-len", type=int, default=8192)
    s.add_argument("--n", type=int, default=4)
    s.add_argument("--sifre", type=int, nargs="*", default=[], help="parola testi uzunlukları, ör. 4096 32768 131072")
    s.add_argument("--deneme", type=int, default=2)
    s.add_argument("--mtp", action="store_true")
    s.set_defaults(fn=cmd_degerlendir)

    s = sub.add_parser("sohbet", help="terminalde sohbet (cache turlar boyunca sabit kalır)")
    common(s)
    model_load(s)
    s.add_argument("--mtp", action="store_true", help="MTP taslak + doğrulama (aynı çıktı, daha hızlı)")
    s.add_argument("--ogretmen", action="store_true", help="karşılaştırma: dönüşümsüz orijinal model")
    s.add_argument("--dusun", action="store_true", help="düşünme (thinking) modunu aç")
    s.add_argument("--sistem", default="")
    s.add_argument("--max-new", type=int, default=512)
    s.set_defaults(fn=cmd_sohbet)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
