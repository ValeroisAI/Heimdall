"""
uret.py — Kuzgun ile metin/kod üretimi (sabit boyutlu cache)
============================================================
    python uret.py --ckpt kosular/temel/son.pt --prompt "def quicksort(arr):"
    python uret.py --ckpt kosular/temel/son.pt --prompt-file uzun_dosya.py --max-new 300

İstem ne kadar uzun olursa olsun cache boyutu sabittir (pencere + delta hafıza); ekrana
istem uzunluğu, cache boyutu ve üretim hızı yazılır.
"""

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from tokenizers import Tokenizer

from kuzgun import KuzgunConfig, KuzgunLM
from kuzgun.cihaz import get_device

HERE = Path(__file__).resolve().parent
DEFAULT_TOKENIZER = HERE.parent / "04_TOKENIZERLAR" / "valerois_tokenizer_8k.json"


def load(ckpt_path: str, device: torch.device) -> KuzgunLM:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = KuzgunLM(KuzgunConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits.float() / temperature
    if top_k:
        kth = logits.topk(min(top_k, logits.size(-1))).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_p, idx = probs.sort(descending=True)
        drop = sorted_p.cumsum(-1) - sorted_p > top_p
        sorted_p = sorted_p.masked_fill(drop, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, idx, sorted_p)
    return torch.multinomial(probs / probs.sum(-1, keepdim=True), 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="def fibonacci(n):\n")
    p.add_argument("--prompt-file")
    p.add_argument("--max-new", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    amp = torch.autocast(device.type, dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    tok = Tokenizer.from_file(args.tokenizer)
    eos = tok.token_to_id("<eos>")
    model = load(args.ckpt, device)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else args.prompt
    ids = torch.tensor([tok.encode(prompt).ids], device=device)

    with torch.no_grad(), amp:
        cache = model.new_cache(1)
        t0 = time.time()
        logits, cache = model.prefill(ids, cache)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_prefill = time.time() - t0
        sys.stdout.write(prompt)
        out, printed = [], ""
        t1 = time.time()
        for _ in range(args.max_new):
            nxt = sample(logits, args.temperature, args.top_k, args.top_p)
            if nxt.item() == eos:
                break
            out.append(nxt.item())
            text = tok.decode(out)
            if not text.endswith("�"):  # yarım UTF-8 karakteri basma
                sys.stdout.write(text[len(printed):])
                sys.stdout.flush()
                printed = text
            logits, cache = model.step(nxt.view(1), cache)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_gen = time.time() - t1
    print(f"\n\n[istem {ids.size(1):,} token, {ids.size(1) / max(t_prefill, 1e-9):,.0f} tok/s prefill | "
          f"üretim {len(out)} token, {len(out) / max(t_gen, 1e-9):,.1f} tok/s | "
          f"cache {KuzgunLM.cache_bytes(cache) / 2**20:.2f} MB (bağlam uzunluğundan bağımsız)]")


if __name__ == "__main__":
    main()
