"""
Eski Valkir 16L ile yeni Heimdall checkpoint'ini AYNI pencerelerde karşılaştırır (aynı tokenizer, token başına kayıp).

    # 1) İkisinin de eğitimde görmediği kod: GitHub kodunun uzak bir dosyası (eğitimde 0 ve 1 kullanıldı)
    python -m heimdall.hazirla --repo codeparrot/github-code-clean --dosya data/train-00500-of-00880.parquet \
        --kolon code --out veri/kod_heldout.bin --max-token 3e6
    # 2) Karşılaştır
    python bench/karsilastir_valkir.py --ckpt runs/gece/last.pt --veri veri/kod_heldout.bin

Raporlanan: tüm pozisyonlar, 256+ pozisyonlar (Valkir'in erken pozisyon dezavantajını gösterir) ve yeni modelin 2048 bağlamı.
Uyarı: stream_coder_100k.bin üzerinde Valkir'e haksız avantaj olur (185 tur görmüş, kuyruğunu da görmüş olabilir).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Heimdall"))
sys.path.insert(0, str(ROOT / "OLD" / "08_BIFROST_H" / "bench"))

from heimdall.io import load_model  # noqa: E402


def windows(path, t, n, seed=0):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(data) - t - 2, n)
    return torch.from_numpy(np.stack([np.asarray(data[s:s + t + 1], dtype=np.int64) for s in starts]))


@torch.no_grad()
def per_pos_loss(model, wins, dev, bs=8, bf16=False):
    out = torch.zeros(wins.size(1) - 1)
    for i in range(0, wins.size(0), bs):
        x = wins[i:i + bs].to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
            logits = model(x[:, :-1])
        logits = logits.float()
        out += F.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction="none").sum(0).cpu()
    return out / wins.size(0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Heimdall last.pt")
    p.add_argument("--valkir", default=str(ROOT / "OLD" / "Valkir.pt"))
    p.add_argument("--veri", required=True, help="uint16 .bin (ikisinin de görmediği veri)")
    p.add_argument("--n", type=int, default=200, help="pencere sayısı")
    p.add_argument("--schedule", default="const2", choices=["exp", "const2", "const1", "mod4"],
                   help="Valkir dilatasyon çizelgesi. Valkir.pt SABİT dilatasyon 2 ile eğitildi (OLD/08_BIFROST_H/DURUM.md); "
                        "yanlış çizelge kaybı şişirir (üstel: 1.73, doğru: 0.72)")
    p.add_argument("--device", default=None)
    args = p.parse_args()
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bf16 = dev.type == "cuda"

    import valkir
    old = valkir.load(args.valkir, args.schedule)
    old.sd = {k: v.to(dev) for k, v in old.sd.items()}    # Valkir ağırlıkları düz sözlükte: .to(dev) onları taşımaz
    new = load_model(args.ckpt, dev)
    print(f"Valkir: {old.n_layers} katman, alıcı alan ~{old.receptive_field()} token | "
          f"Heimdall: {new.num_params() / 1e6:.1f}M param, düzen {new.cfg.layout}, bits={new.cfg.bits}")

    w512 = windows(args.veri, 512, args.n)
    lo = per_pos_loss(old, w512, dev)
    ln = per_pos_loss(new, w512, dev, bf16=bf16)
    w2k = windows(args.veri, 2048, max(8, args.n // 8))
    ln2k = per_pos_loss(new, w2k, dev, bs=2, bf16=bf16)

    def row(name, l, a=0, b=None):
        v = l[a:b]
        return f"{name:34s} {v.mean():7.4f}  (ppl {v.mean().exp():8.2f})"

    print(f"\nVeri: {args.veri} | {args.n} pencere × 512 token\n")
    print(row("Valkir 16L, tüm pozisyonlar", lo))
    print(row("Heimdall, tüm pozisyonlar", ln))
    print(row("Valkir 16L, pozisyon 256-512", lo, 256))
    print(row("Heimdall, pozisyon 256-512", ln, 256))
    print(row("Heimdall, 2048 bağlam, 1024-2048", ln2k, 1024))
    d = float(ln[256:].mean() - lo[256:].mean())
    print(f"\nFark (Heimdall − Valkir, pozisyon 256+): {d:+.4f} nat → {'Heimdall DAHA İYİ' if d < 0 else 'Valkir daha iyi'}")


if __name__ == "__main__":
    main()
