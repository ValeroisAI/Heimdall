"""Bifrost GGUF'unda seçilen katmanların dönüşümünü kapatır: o katmanlar yeniden tam dikkat (KV cache) olur.

    python llamacpp/bifrost_keep.py --in q38-27b-bifrost-Q3_K_M.gguf --out q38-27b-bifrost-k4-Q3_K_M.gguf \
        --keep 15,31,47 --llama ~/llama.cpp-bifrost

Yeniden eğitim gerekmez: aşama A her katmanı öğretmenin girdileriyle ayrı ayrı eğitir, katmanlar birbirinden bağımsızdır.
Ağırlıklar yeniden nicemlenmez, olduğu gibi kopyalanır (Q3_K_M kalır). Numara = blok numarası (layers.N).
--mem-off: teşhis için hafıza dalını kapatır (karışım kapısı ~0): yalnız pencere + çapa kalır.
"""
import argparse
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--keep", default="", help="tam dikkate dönecek katmanlar, ör. 15,31,47")
    p.add_argument("--mem-off", action="store_true", help="hafıza dalını kapat (gamma ≈ 0), teşhis için")
    p.add_argument("--llama", required=True, help="llama.cpp klasörü (gguf-py için)")
    a = p.parse_args()
    sys.path.insert(0, str(Path(a.llama).expanduser() / "gguf-py"))
    import gguf

    keep = {int(x) for x in a.keep.split(",") if x.strip()}
    r = gguf.GGUFReader(a.src)
    arch = r.fields[gguf.Keys.General.ARCHITECTURE].contents()
    key = f"{arch}.bifrost.layers"
    flags = list(r.fields[key].contents())
    if not keep and not a.mem_off:
        raise SystemExit("--keep ya da --mem-off gerekli")
    bad = sorted(i for i in keep if i >= len(flags) or not flags[i])
    if bad:
        raise SystemExit(f"bu katmanlar zaten dönüştürülmemiş (ya da yok): {bad}")
    for i in keep:
        flags[i] = 0
    print(f"dönüştürülmüş katmanlar: {sum(flags)} | tam dikkat: {sorted(keep)} + önceden tutulanlar")

    w = gguf.GGUFWriter(a.out, arch=arch, endianess=r.endianess)
    for f in r.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        t = f.types[0]
        sub = f.types[-1] if t == gguf.GGUFValueType.ARRAY else None
        val = flags if f.name == key else f.contents()
        w.add_key_value(f.name, val, t, sub_type=sub)
    drop = tuple(f"blk.{i}.bifrost_" for i in keep)
    tensors = [t for t in r.tensors if not t.name.startswith(drop)]
    for t in tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    import numpy as np
    for t in tensors:
        data = t.data
        if a.mem_off and t.name.endswith(".bifrost_mix_b"):
            data = np.full_like(np.asarray(data), -30.0)      # sigmoid(-30) ≈ 1e-13
        w.write_tensor_data(data, tensor_endianess=r.endianess)
    w.close()
    print(f"yazıldı: {a.out} ({len(r.tensors) - len(tensors)} Bifrost tensörü çıkarıldı"
          f"{', hafıza kapalı' if a.mem_off else ''})")


if __name__ == "__main__":
    main()
