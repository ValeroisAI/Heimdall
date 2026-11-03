"""
audit_causality.py — Mevcut katmanlar için gelecek-sızıntısı ve cache denetimi
==============================================================================
Her katmana aynı girdiyi iki kez veririz; ikinci kopyada t0 ve sonrasındaki
tokenler rastgele değiştirilir. Nedensel bir katmanda t0 öncesindeki çıktılar
birebir aynı kalmalıdır (max|Δ| = 0). Sıfırdan büyük fark = gelecek sızıntısı.

Ayrıca CSL-QV5'in step() (cache) yolunu dener: orijinal hali hata veriyor mu,
`groups` düzeltmesiyle step() paralel forward ile aynı sonucu veriyor mu?

Kullanım (repo kökünden, CPU yeterli):
    python 08_BIFROST_H/tools/audit_causality.py
"""

import contextlib
import io
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
torch.manual_seed(0)


def load_module(rel_path: str, name: str, patch=None) -> types.ModuleType:
    """Repo dosyasını (isteğe bağlı kaynak yamasıyla) modül olarak yükler."""
    src = (ROOT / rel_path).read_text(encoding="utf-8")
    if patch is not None:
        src = patch(src)
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, str(ROOT / rel_path), "exec"), mod.__dict__)
    return mod


def past_delta(fn, batch=2, time=256, dim=64, t0=150) -> float:
    """t0 öncesi çıktılardaki en büyük mutlak fark (0 olmalı)."""
    x = torch.randn(batch, time, dim)
    x_changed = x.clone()
    x_changed[:, t0:] = torch.randn(batch, time - t0, dim)
    with torch.no_grad():
        return (fn(x)[:, :t0] - fn(x_changed)[:, :t0]).abs().max().item()


def report(name: str, delta: float) -> None:
    verdict = "NEDENSEL" if delta < 1e-5 else "GELECEK SIZINTISI"
    print(f"  {name:38s} max|Δ| = {delta:9.3e}  -> {verdict}")


def main() -> None:
    print("=" * 78)
    print(" Nedensellik denetimi (t0 sonrası değiştirilir, t0 öncesi aynı kalmalı)")
    print("=" * 78)

    radar = load_module("01_KATMANLAR_VE_MIMARILER/vectorized_radar_attention.py", "audit_radar")
    layer = radar.VectorizedRadarAttention(
        dim=64, n_heads=4, chunk_size=32, top_k_chunks=2, local_chunks=2, dtype=torch.float32
    ).eval()
    report("VectorizedRadarAttention", past_delta(layer))

    def tournament_patch(src: str) -> str:
        # Veri dosyası gerektirmesin; yalnız blok tanımlarını al.
        src = src.replace('raw_data = np.memmap(DATA_PATH, dtype=np.uint16, mode="r")', "raw_data = None")
        return src.split("# GENEL MODEL SARMALAYICI")[0]

    tour = load_module(
        "02_TURNUVALAR_VE_TESTLER/grand_architecture_tournament_9.py", "audit_tournament", tournament_patch
    )
    for name, cls in [
        ("Turnuva A3 GCAMv2Block", tour.GCAMv2Block),
        ("Turnuva B1 DiffAttnCSLBlock", tour.DiffAttnCSLBlock),
        ("Turnuva B3 SparseAdaptiveBlock", tour.SparseAdaptiveBlock),
        ("Turnuva C2 SparseGCAMBlock", tour.SparseGCAMBlock),
        ("Turnuva C3 FullHybridBlock", tour.FullHybridBlock),
    ]:
        report(name, past_delta(cls(d=64, nh=4).eval()))

    print()
    print("=" * 78)
    print(" CSL-QV5 cache yolu: step() == forward ?")
    print("=" * 78)
    qv5_path = "07_BIFROST_CSL/valerois_csl_qv5.py"
    ids = torch.randint(0, 257, (2, 40))
    kwargs = dict(vocab_size=257, dim=64, num_layers=2, chunk_size=8, num_memory_heads=2, memory_head_dim=32)

    def stepped_logits(module) -> tuple:
        torch.manual_seed(1)
        lm = module.ValeroisCSLQv5LM(**kwargs).eval()
        with torch.no_grad():
            full = lm(ids)
            state = lm.init_state(ids.size(0), torch.device("cpu"))
            outs = []
            for token in ids.unbind(dim=1):
                logits, state = lm.step(token, state)
                outs.append(logits)
        return full, torch.stack(outs, dim=1)

    original = load_module(qv5_path, "audit_qv5_orig")
    try:
        stepped_logits(original)
        print("  Orijinal step(): çalıştı")
    except RuntimeError as err:
        print(f"  Orijinal step(): HATA -> {str(err).splitlines()[0][:90]}")

    fixed = load_module(
        qv5_path,
        "audit_qv5_fixed",
        lambda s: s.replace(
            "F.conv1d(full, self.local_conv.weight)",
            "F.conv1d(full, self.local_conv.weight, groups=self.dim)",
        ),
    )
    full, stepped = stepped_logits(fixed)
    print(f"  groups=self.dim düzeltmesiyle: max|forward - step| = {(full - stepped).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
