"""
veri_hazirla.py — Metin/kod dosyalarını eğitim için uint16 .bin'e çevirir (8K tokenizer)
======================================================================================
    python araclar/veri_hazirla.py --girdi ~/kodlarim ~/notlar/*.txt --cikti veri/kendi_8k.bin
    python araclar/veri_hazirla.py --girdi veri.jsonl --jsonl-alani text --cikti veri/jsonl_8k.bin

- Klasörler özyinelemeli taranır (--uzantilar ile filtrelenir).
- Aynı içerikli dosyalar bir kez alınır; belgeler karıştırılır; aralarına <eos> konur.
"""

import argparse
import glob
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
DEFAULT_TOKENIZER = HERE.parent.parent / "04_TOKENIZERLAR" / "valerois_tokenizer_8k.json"
DEFAULT_EXT = ".py,.txt,.md,.js,.ts,.java,.c,.h,.cpp,.hpp,.go,.rs,.cs,.php,.rb,.sh,.sql,.html,.css,.json,.yaml,.yml,.toml"


def iter_documents(inputs, extensions, jsonl_field, max_bytes):
    seen = set()
    paths = []
    for pattern in inputs:
        for item in glob.glob(str(Path(pattern).expanduser()), recursive=True) or [pattern]:
            p = Path(item)
            if p.is_dir():
                paths += [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in extensions]
            elif p.is_file():
                paths.append(p)
    for path in sorted(set(paths)):
        if path.suffix.lower() == ".jsonl":
            with open(path, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    try:
                        text = json.loads(line).get(jsonl_field, "")
                    except json.JSONDecodeError:
                        continue
                    if text:
                        yield text
            continue
        if path.stat().st_size > max_bytes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        digest = hashlib.sha1(text.encode("utf-8")).digest()
        if text.strip() and digest not in seen:
            seen.add(digest)
            yield text


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--girdi", nargs="+", required=True, help="dosya, klasör veya glob; .jsonl desteklenir")
    p.add_argument("--cikti", required=True)
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--uzantilar", default=DEFAULT_EXT + ",.jsonl")
    p.add_argument("--jsonl-alani", default="text")
    p.add_argument("--max-dosya-mb", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    eos = tok.token_to_id("<eos>")
    exts = {e.strip().lower() for e in args.uzantilar.split(",")}
    docs = list(iter_documents(args.girdi, exts, args.jsonl_alani, int(args.max_dosya_mb * 2**20)))
    random.Random(args.seed).shuffle(docs)
    print(f"{len(docs):,} belge tokenize ediliyor...", flush=True)

    chunks, n_tokens, n_bytes = [], 0, 0
    for i in range(0, len(docs), 512):
        batch = docs[i:i + 512]
        for text, enc in zip(batch, tok.encode_batch(batch)):
            ids = np.asarray(enc.ids + [eos], dtype=np.uint16)
            chunks.append(ids)
            n_tokens += ids.size
            n_bytes += len(text.encode("utf-8"))
    out = Path(args.cikti)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.concatenate(chunks).tofile(out)
    meta = {"belge": len(docs), "token": n_tokens, "bayt": n_bytes, "bayt_per_token": n_bytes / max(n_tokens, 1),
            "tokenizer": Path(args.tokenizer).name}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"Yazıldı: {out}  ({n_tokens / 1e6:.1f}M token, {meta['bayt_per_token']:.2f} bayt/token)")


if __name__ == "__main__":
    main()
