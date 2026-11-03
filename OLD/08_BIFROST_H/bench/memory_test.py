"""
memory_test.py — Bağlam uzadıkça durum (cache) belleği ve işleme hızı
=====================================================================
batch=1, parça parça (prefill) işleme. Her uzunlukta modelin taşıdığı durumun
bayt cinsinden boyutu ve saniyedeki token sayısı ölçülür. Eğitimden bağımsızdır.

Kullanım:
    python bench/memory_test.py --threads 2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bifrost import BifrostLM  # noqa: E402
from bench.context_test import MODELS  # noqa: E402


def state_bytes(state) -> int:
    if isinstance(state, torch.Tensor):
        return state.numel() * state.element_size()
    if isinstance(state, dict):
        return sum(state_bytes(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return sum(state_bytes(v) for v in state)
    return 0


@torch.no_grad()
def run(model_name: str, max_len: int, chunk: int) -> list:
    torch.manual_seed(0)
    model = BifrostLM(MODELS[model_name]).eval()
    checkpoints = {1 << p for p in range(10, 21)}
    state, done, rows = None, 0, []
    t0 = time.perf_counter()
    while done < max_len:
        ids = torch.randint(0, model.cfg.vocab_size, (1, chunk))
        _, state = model.forward_stateful(ids, state, last_only=True)
        done += chunk
        if done in checkpoints:
            elapsed = time.perf_counter() - t0
            rows.append({"tokens": done, "state_mb": state_bytes(state) / 2**20, "tokens_per_s": done / elapsed})
            print(f"  {model_name:12s} {done:9,d} token | durum {rows[-1]['state_mb']:9.2f} MB | "
                  f"{rows[-1]['tokens_per_s']:8,.0f} tok/s", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--chunk", type=int, default=1024)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    results = {
        "bifrost": run("bifrost", 1 << 20, args.chunk),
        "transformer": run("transformer", 1 << 16, args.chunk),
    }
    out = Path(__file__).resolve().parents[1] / "results" / "context" / "memory.json"
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
