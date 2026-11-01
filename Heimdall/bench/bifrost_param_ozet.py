"""Eğitilmiş Bifrost parametrelerinin özeti: hafıza gerçekten kullanılıyor mu?
    python bench/bifrost_param_ozet.py runs/q38/donusum_param.pt"""
import math
import sys

import torch
import torch.nn.functional as F

st = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("katman     arşiv | mem_scale      | gamma (bias) | mix_lse      | hafıza yarı ömrü, token (KV başı başına, a=0)")
for cv in st["conv"]:
    n = cv["name"]
    if not n.startswith("layers."):
        continue
    p = lambda k: st["params"][f"{n}.self_attn.{k}"].float()
    ms, mb, ml = p("mem_scale"), p("mix.bias"), p("mix_lse")
    decay = p("A_log").exp() * F.softplus(p("dt_bias"))          # |g| per token when the input gate a = 0
    decay[: cv["archival"]] = 0
    life = ["∞" if d == 0 else f"{math.log(2) / d:.0f}" for d in decay.tolist()]
    print(f"{n:10s} {cv['archival']:5d} | {ms.min():5.2f}..{ms.max():5.2f} | {torch.sigmoid(mb).mean():.3f}"
          f"        | {ml.min():5.2f}..{ml.max():5.2f} | {' '.join(life)}")
