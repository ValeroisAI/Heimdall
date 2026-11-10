"""
Düello: eşit GPU süresinde hangi model metni daha iyi öğrenir? Ölçü bit/bayt (tokenizer'dan bağımsız, düşük iyi).

    python bench/yama_duello.py --dakika 10                      # 4 kol × 10 dk
    python bench/yama_duello.py --dakika 3 --kollar tf yama8     # hızlı bakış
    python bench/yama_duello.py --cpu-duman                      # kod çalışıyor mu (CPU, ~1 dk)

Kollar (hepsi aynı metinde, aynı süre, aynı optimizer ayarı):
    tf      düz Transformer, 8K token            hibrit   DeltaNet + dikkat, 8K token
    bayt    hibrit, bayt düzeyi, yamasız (kontrol: baytın kendisi ne kaybettiriyor)
    yama8   bayt + yama 8  (gövde ~2.5× büyük)   yama16   bayt + yama 16 (gövde ~5× büyük)
Bayt verisi token dosyasının çözülmesiyle üretilir (aynı metin); doğrulama aynı metnin son %0.5'i.
Süre saati derleme/ısınma bittikten sonra başlar; öğrenme oranı çizelgesi süreye göre ilerler.
"""

import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from heimdall import kernels  # noqa: E402
from heimdall.config import PRESETS, HeimdallConfig, apply_arch  # noqa: E402
from heimdall.data import Loader, Mixture  # noqa: E402
from heimdall.io import DEFAULT_TOKENIZER  # noqa: E402
from heimdall.model import HeimdallLM  # noqa: E402
from heimdall.optim import build_optimizers, set_lr, wsd  # noqa: E402
from heimdall.yama import PatchConfig, PatchLM  # noqa: E402

VAL_FRAC = 0.005


def byte_data(tok_bin: Path, tok_json: Path, out: Path) -> float:
    """Token dosyasını bayta çözer (uint16, değerler < 256). Eğitim ve doğrulama bölümleri ayrı çözülür ki sınır
    aynı metne denk gelsin. Döndürür: doğrulama bölümünde token başına bayt."""
    meta = out.with_suffix(".json")
    if out.exists() and meta.exists():
        return json.loads(meta.read_text())["val_bayt_per_token"]
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tok_json))
    data = np.memmap(tok_bin, dtype=np.uint16, mode="r")
    split = int(len(data) * (1 - VAL_FRAC))

    def decode(a, b):
        parts = []
        for s in range(a, b, 200_000):
            parts.append(tok.decode(data[s:min(b, s + 200_000)].tolist(), skip_special_tokens=False).encode("utf-8"))
        return np.frombuffer(b"".join(parts), dtype=np.uint8)
    print(f"bayt verisi üretiliyor: {tok_bin.name} ({len(data) / 1e6:.1f}M token) ...", flush=True)
    tr, va = decode(0, split), decode(split, len(data))
    # Mixture doğrulamayı dosyanın son VAL_FRAC'ından alır: doğrulama baytlarını o orana denk gelecek kadar sona koy
    n_val = int(math.ceil(len(va)))
    n_tr = int(n_val * (1 - VAL_FRAC) / VAL_FRAC)
    tr = tr[:n_tr] if len(tr) >= n_tr else tr
    n_val = int(len(tr) * VAL_FRAC / (1 - VAL_FRAC))
    full = np.concatenate((tr, va[:n_val])).astype(np.uint16)
    full.tofile(out)
    ratio = len(va) / (len(data) - split)
    meta.write_text(json.dumps({"val_bayt_per_token": ratio, "bayt": int(len(full))}))
    print(f"  {out.name}: {len(full) / 1e6:.1f}M bayt, token başına {ratio:.2f} bayt", flush=True)
    return ratio


def arms(base: HeimdallConfig, small: bool):
    """ad → (model kurucu, veri türü, dizi boyu, mikro batch)."""
    t_tok, t_byte = (128, 512) if small else (1024, 4096)
    mb_tok, mb_byte = (4, 4) if small else (16, 12)

    def trunk(d, heads, ah):
        return replace(apply_arch(base, "hibrit"), d_model=d, n_heads=heads, attn_heads=ah, vocab_size=256)
    dl = 64 if small else 256
    wide = (base.d_model, base.n_heads, base.attn_heads) if small else None
    return {
        "tf": (lambda: HeimdallLM(apply_arch(base, "transformer")), "token", t_tok, mb_tok),
        "hibrit": (lambda: HeimdallLM(apply_arch(base, "hibrit")), "token", t_tok, mb_tok),
        "bayt": (lambda: HeimdallLM(replace(apply_arch(base, "hibrit"), vocab_size=256)), "bayt", t_byte, mb_byte // 3 or 1),
        "yama8": (lambda: PatchLM(PatchConfig(trunk(*(wide or (768, 6, 12))), patch=8, d_local=dl)), "bayt", t_byte, mb_byte),
        "yama16": (lambda: PatchLM(PatchConfig(trunk(*(wide or (1024, 8, 16))), patch=16, d_local=dl)), "bayt", t_byte, mb_byte),
    }


@torch.no_grad()
def evaluate(model, val, micro, device, amp) -> float:
    model.eval()
    tot, n = 0.0, 0
    for _, ids in val:
        for i in range(0, ids.size(0), micro):
            c = ids[i:i + micro].to(device)
            with amp:
                tot += model(c[:, :-1], c[:, 1:]).item() * c.size(0)
            n += c.size(0)
    model.train()
    return tot / n                                                                  # nat / birim


def run(name, spec, data, budget_s, device, args, bytes_per_token):
    build, kind, T, micro = spec
    torch.manual_seed(args.seed)
    model = build().to(device)
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp = torch.autocast(device.type, dtype=torch.bfloat16) if bf16 else nullcontext()
    opts = build_optimizers(model, args.lr_muon, args.lr_adam, use_muon=True)
    mix = Mixture([str(data[kind])], VAL_FRAC, seed=args.seed)
    val = mix.val_batches(T, args.eval_batches)
    loader = Loader(mix, micro, T, device)
    fwd = torch.compile(model) if (args.compile and device.type == "cuda") else model
    per_byte = 1.0 if kind == "bayt" else bytes_per_token

    def step(lr_factor):
        set_lr(opts, lr_factor)
        x, y = loader.next()
        with amp:
            loss = fwd(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
            o.zero_grad(set_to_none=True)
        return loss

    for _ in range(3):                                           # derleme + ısınma: süreye sayılmaz, lr ≈ 0
        step(1e-6)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0, n, seen, last = time.time(), 0, 0, 0.0
    while True:
        el = time.time() - t0
        if el >= budget_s:
            break
        loss = step(wsd(el / budget_s, 0.02, 0.6))
        n += 1
        seen += micro * T
        if el - last >= args.log_s:
            last = el
            print(f"  [{name}] {el / 60:4.1f} dk | kayıp {loss.item() / math.log(2) / per_byte:.3f} bit/bayt | "
                  f"{seen * per_byte / max(el, 1e-9) / 1e3:,.0f}K bayt/s", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    el = time.time() - t0
    loader.stop = True
    bpb = evaluate(model, val, micro, device, amp) / math.log(2) / per_byte
    n_par = sum(p.numel() for p in model.parameters())
    cost = model.cost_params() if hasattr(model, "cost_params") else n_par
    res = dict(kol=name, param_M=n_par / 1e6, bayt_basina_M=cost / per_byte / 1e6, adim=n,
               metin_MB=seen * per_byte / 1e6, bayt_s_K=seen * per_byte / el / 1e3, bit_bayt=bpb,
               bellek_GB=torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0)
    del model, fwd, opts
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT.parent / "stream_coder_100k.bin"))
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--kollar", nargs="+", default=["tf", "hibrit", "yama8", "yama16"])
    p.add_argument("--dakika", type=float, default=10.0, help="kol başına eğitim süresi")
    p.add_argument("--preset", default="kucuk", help="token kollarının boyutu (kucuk ≈ 45M, temel ≈ 100M)")
    p.add_argument("--lr-muon", type=float, default=0.02)
    p.add_argument("--lr-adam", type=float, default=3e-3)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--log-s", type=float, default=60.0)
    p.add_argument("--no-compile", dest="compile", action="store_false", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "runs" / "yama_duello"))
    p.add_argument("--cpu-duman", action="store_true", help="küçük modeller, 12 sn/kol, CPU: yalnız kodun çalıştığını gösterir")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if a.cpu_duman or not torch.cuda.is_available() else "cuda")
    small = a.cpu_duman
    base = PRESETS["mini" if small else a.preset].model
    if small:
        base = replace(base, d_model=128, n_layers=4)
        a.eval_batches, a.log_s = 4, 5.0
    budget = 12.0 if small else a.dakika * 60
    tok_bin = Path(a.data)
    byte_bin = out / (tok_bin.stem + "_bayt.bin")
    ratio = byte_data(tok_bin, Path(a.tokenizer), byte_bin)
    data = {"token": tok_bin, "bayt": byte_bin}
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"delta çekirdeği: {kernels.prepare(device)}")
    table = arms(base, small)
    print(f"cihaz {device} | kol başına {budget / 60:.1f} dk | token başına {ratio:.2f} bayt\n", flush=True)
    results = []
    for name in a.kollar:
        print(f"== {name}", flush=True)
        r = run(name, table[name], data, budget, device, a, ratio)
        results.append(r)
        print(f"  → {r['bit_bayt']:.4f} bit/bayt | {r['metin_MB']:.0f} MB metin gördü | {r['param_M']:.0f}M parametre\n", flush=True)
        (out / "sonuc.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"{'kol':8} {'param':>8} {'bayt başına':>12} {'metin':>9} {'hız':>12} {'bellek':>8} {'bit/bayt':>9}")
    for r in sorted(results, key=lambda r: r["bit_bayt"]):
        print(f"{r['kol']:8} {r['param_M']:7.0f}M {r['bayt_basina_M']:11.1f}M {r['metin_MB']:7.0f}MB "
              f"{r['bayt_s_K']:9,.0f}K/s {r['bellek_GB']:6.1f}GB {r['bit_bayt']:9.4f}")
    print("\nbayt başına: bir bayt metin için çalışan parametre (maliyet); bit/bayt: doğrulama, düşük iyi.")


if __name__ == "__main__":
    main()
