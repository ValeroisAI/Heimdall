"""
mqar_diagnose.py — Kuzgun uzun boşlukta neden bozuluyor? Kolları tek tek kapatarak ölç.

    python bench/mqar_diagnose.py --run K_coupled --gap 65536
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.mqar import accuracy_at_gap  # noqa: E402
from bifrost import BifrostLM, ModelConfig  # noqa: E402
from bifrost.layers import Kuzgun  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--gap", type=int, default=65536)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    ckpt = torch.load(ROOT / "results" / "mqar" / f"{args.run}.pt", map_location="cpu", weights_only=False)
    model = BifrostLM(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    layers = [b.mixer for b in model.blocks if isinstance(b.mixer, Kuzgun)]
    orig_gates = Kuzgun._memory_gates
    orig_combine = Kuzgun._combine

    def no_decay(self, x):
        g, beta = orig_gates(self, x)
        return torch.zeros_like(g), beta

    def memory_only(self, x, o_w, o_m):
        return orig_combine(self, x, torch.zeros_like(o_w) if o_w is not None else None, o_m)

    variants = {
        "normal": {},
        "unutma kapalı (α=1)": {"_memory_gates": no_decay},
        "pencere kolu sıfır": {"_combine": memory_only},
    }
    # Dolgu tokeni için öğrenilen kapılar (katman başına ortalama α ve β)
    with torch.no_grad():
        x = model.embed(torch.zeros(1, 4, dtype=torch.long))
        for i, block in enumerate(model.blocks):
            if isinstance(block.mixer, Kuzgun) and block.mixer.use_memory:
                g, beta = orig_gates(block.mixer, block.mixer_norm(x))
                print(f"  katman {i}: dolguda α = {g.exp().mean():.6f} → α^{args.gap} = {g.exp().mean() ** args.gap:.3e} | β = {beta.mean():.3f}")
            x = block(x)
    for name, patch in variants.items():
        for attr, fn in patch.items():
            setattr(Kuzgun, attr, fn)
        acc = accuracy_at_gap(model, args.gap, 8, args.samples, np.random.default_rng(10_000 + args.gap))
        print(f"  {name:22s} boşluk {args.gap}: doğruluk {acc:6.1%}", flush=True)
        Kuzgun._memory_gates, Kuzgun._combine = orig_gates, orig_combine


if __name__ == "__main__":
    main()
