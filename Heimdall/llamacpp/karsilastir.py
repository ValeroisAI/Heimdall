"""llama.cpp (bifrost_logits) ile PyTorch referansımızı (hibrit.py) aynı tokenlerde karşılaştırır.

    python llamacpp/karsilastir.py --gguf x.gguf --model <HF klasörü> [--conv donusum_param.pt] --bin <llama.cpp/build/bin>
Parçalar: önce prefill parçaları, kalanı tek tek (cache yolunun da sınanması için).
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heimdall.hibrit import HybridLM  # noqa: E402

METIN = ("The history of the city goes back more than two thousand years. It was founded on a hill above the river, "
         "and for centuries it served as a market town where farmers from the surrounding valleys sold grain, wool and "
         "wine. In the nineteenth century the railway arrived, factories were built along the water, and the population "
         "grew from a few thousand to more than two hundred thousand people. Today the old town is known for its narrow "
         "streets, its cathedral and its university, which was founded in 1472 and is one of the oldest in the country. ")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--conv")
    p.add_argument("--bin", required=True)
    p.add_argument("--n", type=int, default=160)
    p.add_argument("--parcalar", default="37,1,50,1,1")
    a = p.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    ids = (tok.encode(METIN * 4).ids)[:a.n]
    with tempfile.TemporaryDirectory() as d:
        tf, of = Path(d) / "t.bin", Path(d) / "o.bin"
        np.array(ids, dtype=np.int32).tofile(tf)
        r = subprocess.run([str(Path(a.bin) / "bifrost_logits"), a.gguf, str(tf), str(of), a.parcalar],
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-3000:])
            raise SystemExit("llama.cpp çalışmadı")
        lc = torch.from_numpy(np.fromfile(of, dtype=np.float32)).view(len(ids), -1)
    m = HybridLM.load(a.model).eval()
    if a.conv:
        m.apply_conversion(torch.load(a.conv, map_location="cpu", weights_only=False))
    with torch.no_grad():
        ref = m(torch.tensor([ids]))[0]
    lc = lc[:, :ref.size(1)]
    err = (ref - lc).abs().max().item()
    agree = (ref.argmax(-1) == lc.argmax(-1)).float().mean().item()
    nll = lambda z: F.cross_entropy(z[:-1], torch.tensor(ids[1:])).item()
    print(f"{len(ids)} token | en büyük logit farkı {err:.3e} (ölçek {ref.abs().max().item():.1f}) | argmax uyuşması "
          f"{agree:.3f} | NLL PyTorch {nll(ref):.4f} ↔ llama.cpp {nll(lc):.4f}")
    w = 64
    for lo, hi in ((0, w), (w, len(ids))):
        e = (ref[lo:hi] - lc[lo:hi]).abs().max().item()
        print(f"  konum {lo}-{hi}: en büyük fark {e:.3e}")


if __name__ == "__main__":
    main()
