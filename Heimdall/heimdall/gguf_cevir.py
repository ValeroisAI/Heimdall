"""
gguf_cevir.py — Modellerimizi tek dosyalık GGUF'a çevirir ve geri yükler.

    # Heimdall (eğitim checkpoint'i ya da servis klasörü): 1-bit / üçlü FFN → TQ2_0 (kayıpsız), gerisi bf16
    python -m heimdall.gguf_cevir heimdall --ckpt runs/hafta/model/last.pt --out heimdall-337m.gguf
    # Bifrost ile dönüştürülmüş Qwen3.5 / 3.8: orijinal ağırlıklar + dönüşüm parametreleri tek dosyada
    python -m heimdall.gguf_cevir bifrost --model Qwen/Qwen3.8-27B --conv runs/q38/donusum_param.pt \\
        --out q38-27b-bifrost-q8_0.gguf --tip q8_0
    # geri yükleyip orijinalle karşılaştır (logit farkı)
    python -m heimdall.gguf_cevir dogrula --gguf heimdall-337m.gguf --ckpt runs/hafta/model/last.pt
    python -m heimdall.gguf_cevir dogrula --gguf q38.gguf --model Qwen/Qwen3.8-27B --conv runs/q38/donusum_param.pt

DÜRÜST NOT: llama.cpp / LM Studio / Ollama bu dosyaları ŞU AN çalıştıramaz. Mimariler ("heimdall", "bifrost_qwen35")
llama.cpp'de kayıtlı değil, katmanlarımızın (DeltaNet + NoPE dikkat + ikili FFN; pencere + çapa + gecikmeli delta hafıza)
llama.cpp'de yazılması gerekir. O zamana kadar bu dosyalar:
  * bizim çalışma zamanımızda açılır: `--model x.gguf` (donustur_hibrit sohbet/degerlendir, heimdall.server, io.load_model),
  * Hugging Face'te tek dosya dağıtım içindir (tokenizer ve config dosyanın içinde),
  * llama.cpp portuna hazırdır: tipler llama.cpp'nin kendi tipleri; port bitince `llama-quantize` ile Q4_K_M / IQ4_XS.

Tipler (gguf-py'nin yazabildikleri): f32, bf16, f16, q8_0, q5_0, q4_0. K-quant / IQ tipleri gguf-py'de yok.
27B için q4_0 = 14.6 GB (16 GB kartta sığmaz); evde bizim 4-bit önbelleğimiz (g128, 12.9 GB) daha küçük.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

TIPLER = ("f32", "bf16", "f16", "q8_0", "q5_0", "q4_0")


def _gg():
    try:
        import gguf
    except ImportError as e:
        raise SystemExit("gguf paketi gerekli: pip install gguf") from e
    return gguf


def _qtype(name: str):
    return getattr(_gg().GGMLQuantizationType, name.upper())


# --------------------------------------------------------------------------- yazıcı (akış: tensör tensör, RAM az)
class Writer:
    def __init__(self, path, arch: str) -> None:
        gguf = _gg()
        self.gguf = gguf
        self.w = gguf.GGUFWriter(str(path), arch)
        self.items = []

    def kv(self, key: str, value) -> None:
        if isinstance(value, bool):
            self.w.add_bool(key, value)
        elif isinstance(value, int):
            self.w.add_uint32(key, value)
        elif isinstance(value, float):
            self.w.add_float32(key, value)
        else:
            self.w.add_string(key, value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))

    def add(self, name: str, shape, qtype: str, get) -> None:
        """get() → torch tensörü (yazma anında çağrılır). qtype blok boyuna uymazsa bf16/f32'ye düşer."""
        shape = tuple(int(x) for x in shape)
        if qtype not in ("f32", "bf16", "f16"):
            block = self.gguf.GGML_QUANT_SIZES[_qtype(qtype)][0]
            if len(shape) != 2 or shape[-1] % block:
                qtype = "bf16" if len(shape) == 2 else "f32"
        if len(shape) < 2:
            qtype = "f32"
        self.items.append((name, shape, qtype, get))

    def _byte_shape(self, shape, qtype):
        block, size = self.gguf.GGML_QUANT_SIZES[_qtype(qtype)]
        return shape[:-1] + (shape[-1] // block * size,)

    def _encode(self, t: torch.Tensor, shape, qtype: str) -> np.ndarray:
        t = t.detach().reshape(shape)
        if qtype == "f32":
            return t.float().numpy()
        if qtype == "bf16":
            return t.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint8).reshape(self._byte_shape(shape, qtype))
        if qtype == "f16":
            return t.to(torch.float16).numpy().view(np.uint8).reshape(self._byte_shape(shape, qtype))
        out = np.empty(self._byte_shape(shape, qtype), dtype=np.uint8)
        q = _qtype(qtype)
        for r in range(0, shape[0], 8192):            # satır parçaları: büyük embedding'de RAM tepe noktası küçük
            out[r:r + 8192] = self.gguf.quants.quantize(t[r:r + 8192].float().numpy(), q)
        return out

    def write(self, log=print) -> None:
        for name, shape, qtype, _ in self.items:
            if qtype == "f32":
                self.w.add_tensor_info(name, shape, np.dtype(np.float32), int(np.prod(shape)) * 4)
            else:
                bs = self._byte_shape(shape, qtype)
                self.w.add_tensor_info(name, bs, np.dtype(np.uint8), int(np.prod(bs)), raw_dtype=_qtype(qtype))
        self.w.write_header_to_file()
        self.w.write_kv_data_to_file()
        self.w.write_ti_data_to_file()
        total = len(self.items)
        for i, (name, shape, qtype, get) in enumerate(self.items):
            self.w.write_tensor_data(self._encode(get(), shape, qtype))
            if (i + 1) % 50 == 0 or i + 1 == total:
                log(f"  {i + 1}/{total} tensör yazıldı")
        self.w.close()


# --------------------------------------------------------------------------- okuyucu
class Reader:
    def __init__(self, path) -> None:
        gguf = _gg()
        self.gguf = gguf
        self.r = gguf.GGUFReader(str(path))
        self.tensors = {t.name: t for t in self.r.tensors}

    def field(self, key: str, default=None):
        f = self.r.fields.get(key)
        if f is None:
            return default
        if hasattr(f, "contents"):
            return f.contents()
        return bytes(f.parts[f.data[0]]).decode()

    def json(self, key: str, default=None):
        v = self.field(key)
        return default if v is None else json.loads(v)

    def get(self, name: str) -> torch.Tensor:
        t = self.tensors[name]
        q = t.tensor_type
        if q == self.gguf.GGMLQuantizationType.F32:
            arr = np.array(t.data, dtype=np.float32)
        else:
            arr = self.gguf.quants.dequantize(np.array(t.data), q)
        shape = tuple(int(x) for x in reversed(t.shape))
        return torch.from_numpy(np.ascontiguousarray(arr).reshape(shape).astype(np.float32))

    def tokenizer(self):
        from tokenizers import Tokenizer
        s = self.field("tokenizer.huggingface.json")
        return None if s is None else Tokenizer.from_str(s)


def gguf_tokenizer_file(path) -> str:
    """GGUF içindeki tokenizer.json'u yanına çıkarır (tokenizer yolu bekleyen kod için)."""
    out = Path(str(path) + ".tokenizer.json")
    if not out.exists():
        out.write_text(Reader(path).field("tokenizer.huggingface.json"))
    return str(out)


# --------------------------------------------------------------------------- Heimdall
def _heimdall_bits(model):
    from .bit import BitLinear
    return {n: m for n, m in model.named_modules() if isinstance(m, BitLinear)}


def export_heimdall(ckpt, out, tip: str = "bf16", tokenizer=None, log=print) -> None:
    from .io import load_model, tokenizer_path
    model = load_model(ckpt).float()
    cfg = model.cfg
    bits = _heimdall_bits(model)
    w = Writer(out, "heimdall")
    w.kv("general.name", Path(str(ckpt)).stem)
    w.kv("heimdall.config", cfg.to_dict())
    w.kv("heimdall.bit_layers", {n: m.bits for n, m in bits.items()})
    tok = tokenizer or tokenizer_path(ckpt)
    if tok and Path(tok).exists():
        w.kv("tokenizer.huggingface.json", Path(tok).read_text())
    sd = model.state_dict()
    skip = {f"{n}.{k}" for n in bits for k in ("weight", "base", "scale")}
    if cfg.tie_embeddings:
        skip.add("lm_head.weight")
    for n, m in bits.items():   # etkin ağırlık W·(taban·ölçek): satır başına tek değer → TQ2_0 blok ölçeği, kayıpsız
        get = (lambda m=m: m.weight.float() * (m.base * m.scale).float()[:, None])
        w.add(f"{n}.weight", m.weight.shape, "tq2_0", get)
    for k, t in sd.items():
        if k in skip:
            continue
        big = t.dim() == 2 and min(t.shape) >= 64
        w.add(k, t.shape, tip if big else "f32", (lambda t=t: t))
    w.write(log)
    log(f"yazıldı: {out} ({Path(out).stat().st_size / 2**20:.1f} MB)")


def load_heimdall_gguf(path):
    from .config import HeimdallConfig
    from .model import HeimdallLM
    r = Reader(path)
    model = HeimdallLM(HeimdallConfig(**r.json("heimdall.config")))
    bits = r.json("heimdall.bit_layers", {})
    sd = {}
    for k in model.state_dict():
        mod, _, attr = k.rpartition(".")
        if mod in bits and attr in ("base", "scale"):
            continue
        if mod in bits and attr == "weight":
            wt = r.get(k)
            d = wt.abs().amax(1)
            sd[k] = torch.where(d[:, None] > 0, wt / d.clamp(min=1e-30)[:, None], torch.zeros_like(wt)).round()
            sd[f"{mod}.base"], sd[f"{mod}.scale"] = d, torch.ones_like(d)
        elif k in r.tensors:
            sd[k] = r.get(k)
    if model.cfg.tie_embeddings:
        sd["lm_head.weight"] = sd["embed.weight"]
    model.load_state_dict(sd)
    return model.eval()


# --------------------------------------------------------------------------- Bifrost (Qwen3.5 ailesi)
def export_bifrost(model_path, conv, out, tip: str = "q8_0", log=print) -> None:
    from .hibrit import Weights
    w = Weights(model_path)
    st = torch.load(conv, map_location="cpu", weights_only=False) if conv else None
    g = Writer(out, "bifrost_qwen35")
    g.kv("general.name", Path(str(model_path)).name + ("-bifrost" if st else ""))
    g.kv("bifrost.hf_config", w.config)
    if st:
        g.kv("bifrost.conversion", {"conv": st["conv"], "mtp_hidden": st.get("mtp_hidden", "normed"),
                                    "meta": {k: v for k, v in st.get("meta", {}).items() if k != "rapor"}})
    tok = Path(model_path) / "tokenizer.json"
    if tok.exists():
        g.kv("tokenizer.huggingface.json", tok.read_text())
    for name in sorted(w.map):
        key, f = w.map[name]
        with w._open(str(w.path / f), framework="pt") as fh:
            shape = fh.get_slice(key).get_shape()
        big = len(shape) == 2 and min(shape) >= 256
        g.add(name, shape, tip if big else "f32", (lambda name=name: w.get(name)))
    for name, t in (st["params"].items() if st else ()):
        g.add(name, t.shape, "f32", (lambda t=t: t))
    g.write(log)
    log(f"yazıldı: {out} ({Path(out).stat().st_size / 2**30:.2f} GB)")


class GGUFWeights:
    """hibrit.Weights ile aynı arayüz (config, map, has, get); bifrost GGUF'u okur."""

    def __init__(self, path) -> None:
        self.r = Reader(path)
        self.path = Path(path)
        self.config = self.r.json("bifrost.hf_config")
        self.map = {n: (n, None) for n in self.r.tensors}

    def has(self, name: str) -> bool:
        return name in self.r.tensors

    def get(self, name: str) -> torch.Tensor:
        return self.r.get(name)

    def conversion_state(self):
        cv = self.r.json("bifrost.conversion")
        if not cv:
            return None
        from .hibrit import HybridAttention
        prefixes = [f"{c['name']}.self_attn." for c in cv["conv"]]
        params = {n: self.r.get(n) for n in self.r.tensors for p in prefixes
                  if n.startswith(p) and n[len(p):].split(".")[0] in HybridAttention.NEW}
        return {"conv": cv["conv"], "params": params, "mtp_hidden": cv.get("mtp_hidden", "normed"), "meta": cv.get("meta", {})}


# --------------------------------------------------------------------------- doğrulama
SAMPLE = ("The history of the city goes back more than two thousand years. It was founded on a hill above the river, "
          "and for centuries it served as a market town where farmers from the surrounding valleys sold grain, wool and "
          "wine. In the nineteenth century the railway arrived, factories were built along the water, and the population "
          "grew from a few thousand to more than two hundred thousand people. Today the old town is known for its narrow "
          "streets, its cathedral and its university, which was founded in 1472 and is one of the oldest in the country.")


@torch.no_grad()
def verify(args) -> None:
    """Gerçek metinde: logit farkı, argmax uyuşması ve NLL (orijinal ↔ GGUF)."""
    import torch.nn.functional as F
    tok = Reader(args.gguf).tokenizer()
    text = Path(args.metin).read_text() if args.metin else SAMPLE
    if tok is not None:
        ids = torch.tensor([tok.encode(text).ids[:args.n]])
    else:
        ids = torch.randint(0, 1000, (1, args.n))
        print("uyarı: tokenizer yok, rastgele tokenler (uyuşma oranı yanıltıcı olur)")
    if args.ckpt:
        from .io import load_model
        a = load_model(args.ckpt).float()(ids)
        b = load_heimdall_gguf(args.gguf).float()(ids)
    else:
        from .hibrit import HybridLM
        ref = HybridLM.load(args.model)
        if args.conv:
            ref.apply_conversion(torch.load(args.conv, map_location="cpu", weights_only=False))
        a = ref(ids)
        g = HybridLM.load(args.gguf)
        if g.embedded_conversion:
            g.apply_conversion(g.embedded_conversion)
        b = g(ids)
    nll = lambda z: F.cross_entropy(z[0, :-1], ids[0, 1:]).item()
    agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    print(f"{ids.size(1)} token | en büyük logit farkı {(a - b).abs().max().item():.3e} (ölçek {a.abs().max().item():.1f}) | "
          f"argmax uyuşması {agree:.3f} | NLL orijinal {nll(a):.4f} → GGUF {nll(b):.4f}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("heimdall")
    s.add_argument("--ckpt", required=True, help="eğitim .pt ya da servis klasörü")
    s.add_argument("--out", required=True)
    s.add_argument("--tip", choices=TIPLER, default="bf16", help="yoğun matrisler (bit FFN her zaman TQ2_0)")
    s.add_argument("--tokenizer")
    s = sub.add_parser("bifrost")
    s.add_argument("--model", required=True, help="HF adı ya da klasör")
    s.add_argument("--conv", help="donusum_param.pt (yoksa dönüşümsüz orijinal yazılır)")
    s.add_argument("--out", required=True)
    s.add_argument("--tip", choices=TIPLER, default="q8_0")
    s = sub.add_parser("dogrula")
    s.add_argument("--gguf", required=True)
    s.add_argument("--ckpt", help="Heimdall: karşılaştırılacak checkpoint")
    s.add_argument("--model", help="Bifrost: HF modeli")
    s.add_argument("--conv")
    s.add_argument("--n", type=int, default=256)
    s.add_argument("--metin", help="doğrulama metni (.txt); boşsa yerleşik İngilizce paragraf")
    a = p.parse_args(argv)
    if a.cmd == "heimdall":
        export_heimdall(a.ckpt, a.out, a.tip, a.tokenizer)
    elif a.cmd == "bifrost":
        from .donustur import fetch_model
        export_bifrost(fetch_model(a.model), a.conv, a.out, a.tip)
    else:
        if a.model:
            from .donustur import fetch_model
            a.model = fetch_model(a.model)
        verify(a)


if __name__ == "__main__":
    main()
