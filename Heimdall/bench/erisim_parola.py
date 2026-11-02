"""Retrieval attention on the original (unconverted) model: pass-key recall with full attention vs. sinks + window
+ top-k key blocks, at decode. Shows how much of the exact recall survives when each full-attention layer reads only
a few thousand tokens per step.

  python bench/erisim_parola.py --model PATH --lengths 8192 16384 --depths 0.1 0.5 0.9 --n 2 \
      --configs full w512 w512k16
Config names: full | w<W> (window only) | w<W>k<K>[b<L>] (window + K blocks of L tokens, default 64) [p: also prefill]
"""

import argparse
import random
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heimdall.hibrit import HybridLM  # noqa: E402

FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


def prompt(tok, length, depth, key):
    head = ("There is an important piece of information hidden inside a lot of irrelevant text. Find it and "
            "memorize it. I will quiz you about it at the end.\n")
    needle = f"The pass key is {key}. Remember it. {key} is the pass key.\n"
    tail = "\nWhat is the pass key? The pass key is"
    unit = tok.encode(FILLER).ids
    n_fill = (length - len(tok.encode(head + needle + tail).ids)) // len(unit)
    at = int(n_fill * depth)
    ids = tok.encode(head).ids
    for i in range(n_fill):
        if i == at:
            ids += tok.encode(needle).ids
        ids += unit
    return ids + tok.encode(tail).ids


def parse(cfg):
    if cfg == "full":
        return None
    m = re.fullmatch(r"w(\d+)(?:k(\d+))?(?:b(\d+))?(p?)", cfg)
    return dict(window=int(m[1]), k=int(m[2] or 0), block=int(m[3] or 64), prefill=bool(m[4]))


def main():
    from tokenizers import Tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[8192])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--configs", nargs="+", default=["full", "w512", "w512k16"])
    ap.add_argument("--max-new", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=2048)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    model = HybridLM.load(a.model, mtp=False).eval()
    tok = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    rng = random.Random(1234)
    cases = [(L, d, rng.randint(10000, 99999)) for L in a.lengths for d in a.depths for _ in range(a.n)]
    score = {c: 0 for c in a.configs}
    for L, d, key in cases:
        ids = torch.tensor([prompt(tok, L, d, key)])
        line = f"bağlam {ids.size(1):6d}  derinlik {d:.2f}  parola {key}"
        for c in a.configs:
            r = parse(c)
            model.set_retrieval(k=None) if r is None else model.set_retrieval(sinks=4, **r)
            t0 = time.time()
            out, _ = model.generate(ids, max_new=a.max_new, chunk=a.chunk)
            text = tok.decode(out).strip().replace("\n", " ")
            ok = re.search(r"\d+", text)
            ok = bool(ok) and ok[0] == str(key)
            score[c] += ok
            line += f" | {c}: {text[:16]!r} {'✓' if ok else '✗'} {time.time() - t0:.0f}s"
        print(line, flush=True)
    print("doğruluk: " + "  ".join(f"{c} {score[c]}/{len(cases)}" for c in a.configs))


if __name__ == "__main__":
    main()
