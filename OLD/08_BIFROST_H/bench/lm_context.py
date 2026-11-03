"""
lm_context.py — train.py ile eğitilmiş modellerin gerçek bağlam kullanımı
========================================================================
valkir_context.py ile aynı iki test (pozisyona göre loss + PROJE_ID iğnesi), ama
results/runs/<name>.pt checkpoint'leri için.

Kullanım:
    python bench/lm_context.py --runs kuzgun transformer --threads 4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.valkir_context import TOKENIZER, loss_by_position, needle  # noqa: E402
from bifrost import BifrostLM, ModelConfig  # noqa: E402


def load_run(name: str) -> BifrostLM:
    ckpt = torch.load(ROOT / "results" / "runs" / f"{name}.pt", map_location="cpu", weights_only=False)
    model = BifrostLM(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.eval()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--val", default=str(ROOT / "data" / "dev_code_8k_val.bin"))
    p.add_argument("--length", type=int, default=2048)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    tok = Tokenizer.from_file(str(TOKENIZER))
    data = np.memmap(args.val, dtype=np.uint16, mode="r")
    buckets = [(0, 16), (16, 64), (64, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 4096)]
    for name in args.runs:
        model = load_run(name)
        forward = lambda ids, _m=model: _m(ids)
        with torch.no_grad():
            lbp = loss_by_position(forward, data, args.length, 8, np.random.default_rng(0), buckets)
            rows = [needle(forward, tok, data, d, 16, np.random.default_rng(d))
                    for d in (8, 32, 128, 512, 1024, 2000)]
        print(f"\n[{name}] pozisyona göre loss: " + " | ".join(f"{r['from']}-{r['to']}: {r['loss']:.3f}" for r in lbp))
        for r in rows:
            print(f"  iğne mesafe {r['distance']:5d}: kazanç {r['gain_nats']:+6.2f} nats | tam isabet {r['exact_with']:4.0%}")
        out = ROOT / "results" / "runs" / f"{name}_context.json"
        out.write_text(json.dumps({"loss_by_position": lbp, "needle": rows}, indent=2))


if __name__ == "__main__":
    main()
