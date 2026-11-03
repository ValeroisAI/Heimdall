"""
build_dev_corpus.py — Lisansı temiz Python kodundan geliştirme verisi üretir
===========================================================================
Makinedeki Python kaynaklarını (stdlib + kurulu paketler) 8K tokenizer ile
tokenize eder ve stream_coder_100k.bin ile aynı formatta (uint16) yazar.
Validation, eğitimden DOSYA düzeyinde ayrılır (aynı dosyanın parçaları iki
tarafa düşmez). Dosyalar arasına <eos> konur.

Çıktılar (varsayılan: 08_BIFROST_H/data/):
    dev_code_8k_train.bin, dev_code_8k_val.bin, dev_code_8k_meta.json

Kullanım:
    python 08_BIFROST_H/tools/build_dev_corpus.py --max-tokens 40000000
"""

import argparse
import hashlib
import json
import random
import sys
import sysconfig
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER = ROOT / "04_TOKENIZERLAR" / "valerois_tokenizer_8k.json"


def source_roots() -> list:
    roots = {sysconfig.get_paths()["stdlib"], sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}
    roots.update(p for p in sys.path if p.endswith(("site-packages", "dist-packages")))
    return sorted(Path(r) for r in roots if Path(r).is_dir())


def collect_files(min_bytes: int, max_bytes: int) -> list:
    seen_hashes, files = set(), []
    for root in source_roots():
        for path in sorted(root.rglob("*.py")):
            try:
                size = path.stat().st_size
                if not (min_bytes <= size <= max_bytes):
                    continue
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            files.append((str(path), text))
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--out-dir", default=str(ROOT / "08_BIFROST_H" / "data"))
    parser.add_argument("--name", default="dev_code_8k")
    parser.add_argument("--max-tokens", type=int, default=40_000_000)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--min-bytes", type=int, default=1_000)
    parser.add_argument("--max-bytes", type=int, default=400_000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<eos>")
    assert tokenizer.get_vocab_size() <= 65_535 and eos_id is not None

    files = collect_files(args.min_bytes, args.max_bytes)
    random.Random(args.seed).shuffle(files)
    n_val = max(1, int(len(files) * args.val_fraction))
    splits = {"val": files[:n_val], "train": files[n_val:]}
    print(f"[*] {len(files)} benzersiz dosya ({n_val} val)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"tokenizer": Path(args.tokenizer).name, "eos_id": eos_id, "seed": args.seed, "splits": {}}
    budget = {"val": int(args.max_tokens * args.val_fraction), "train": args.max_tokens}
    for split, items in splits.items():
        chunks, n_tokens, n_bytes, n_files = [], 0, 0, 0
        for start in range(0, len(items), 256):
            batch = items[start:start + 256]
            for (_, text), enc in zip(batch, tokenizer.encode_batch([t for _, t in batch])):
                ids = enc.ids + [eos_id]
                chunks.append(np.asarray(ids, dtype=np.uint16))
                n_tokens += len(ids)
                n_bytes += len(text.encode("utf-8"))
                n_files += 1
            if n_tokens >= budget[split]:
                break
        data = np.concatenate(chunks)
        path = out_dir / f"{args.name}_{split}.bin"
        data.tofile(path)
        meta["splits"][split] = {
            "file": path.name, "files": n_files, "tokens": int(data.size),
            "bytes": n_bytes, "bytes_per_token": n_bytes / data.size,
        }
        print(f"[*] {split}: {n_files} dosya, {data.size / 1e6:.2f}M token, {n_bytes / data.size:.2f} bayt/token -> {path}")
    (out_dir / f"{args.name}_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
