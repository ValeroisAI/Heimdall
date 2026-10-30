"""
bifrost_gguf.py — Dönüştürülmüş Qwen3.5 / 3.6 / 3.8 için llama.cpp GGUF'u (Bifrost yamalı llama.cpp ile çalışır).

    # 1) llama.cpp'nin kendi çeviricisiyle taban GGUF (bf16 / q8_0 / f32), 2) dönüşüm tensörleri + ayarları eklenir
    python llamacpp/bifrost_gguf.py --model <HF klasörü> --conv runs/q38/donusum_param.pt \\
        --llama ~/llama.cpp --outtype bf16 --out runs/q38/q38-27b-bifrost-bf16.gguf
    # sonra llama.cpp ile 4-bit:  llama-quantize q38-27b-bifrost-bf16.gguf q38-27b-bifrost-IQ4_XS.gguf IQ4_XS
    # (dönüşüm tensörleri adlarında ".weight" taşımadığı için f32 kalır)

Eklenenler: <arch>.bifrost.window, <arch>.bifrost.sinks, <arch>.bifrost.layers (katman başına 0/1) ve katman başına
blk.N.bifrost_{fq,fk,ab,ab_b,mix,mix_b,mix_lse,mem_scale,a,dt}. MTP katmanının dönüşümü şimdilik eklenmez
(llama.cpp'de MTP taslağı geri-alma ister; Bifrost katmanları henüz desteklemiyor).
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def tensors_from_conv(st: dict):
    """donusum_param.pt → {katman: {ad: np.float32}}, pencere, çapa."""
    out, windows, sinks = {}, set(), set()
    for cv in st["conv"]:
        name = cv["name"]
        if not name.startswith("layers."):
            continue
        i = int(name.split(".")[1])
        p = lambda k: st["params"][f"{name}.self_attn.{k}"].float()
        mask = torch.ones_like(p("A_log"))
        mask[:cv["archival"]] = 0.0
        out[i] = {
            "bifrost_fq": p("fq").transpose(1, 2).contiguous(),
            "bifrost_fk": p("fk").transpose(1, 2).contiguous(),
            "bifrost_ab": p("ab.weight"), "bifrost_ab_b": p("ab.bias"),
            "bifrost_mix": p("mix.weight"), "bifrost_mix_b": p("mix.bias"),
            "bifrost_mix_lse": p("mix_lse"), "bifrost_mem_scale": p("mem_scale"),
            "bifrost_a": -p("A_log").exp() * mask, "bifrost_dt": p("dt_bias"),
        }
        out[i] = {k: v.numpy().astype(np.float32) for k, v in out[i].items()}
        windows.add(cv["window"])
        sinks.add(cv["sinks"])
    assert len(windows) == 1 and len(sinks) == 1, "tüm katmanlarda aynı pencere/çapa bekleniyor"
    return out, windows.pop(), sinks.pop()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="HF klasörü (taban GGUF yoksa llama.cpp çeviricisi çalıştırılır)")
    p.add_argument("--base", help="hazır taban GGUF (convert_hf_to_gguf.py çıktısı)")
    p.add_argument("--conv", required=True, help="donusum_param.pt")
    p.add_argument("--llama", required=True, help="llama.cpp klasörü (convert_hf_to_gguf.py ve gguf-py için)")
    p.add_argument("--outtype", default="bf16", choices=["f32", "f16", "bf16", "q8_0"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    sys.path.insert(0, str(Path(a.llama) / "gguf-py"))
    import gguf

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=Path(a.out).parent) as tmp:   # /tmp küçük olabilir: taban GGUF çıktının yanında
        base = a.base
        if base is None:
            assert a.model, "--model ya da --base gerekli"
            base = str(Path(tmp) / "base.gguf")
            subprocess.run([sys.executable, str(Path(a.llama) / "convert_hf_to_gguf.py"), a.model, "--outtype",
                            a.outtype, "--outfile", base], check=True)
        r = gguf.GGUFReader(base)
        arch = r.fields[gguf.Keys.General.ARCHITECTURE].contents()
        assert arch == "qwen35", f"yalnız qwen35 (Qwen3.5/3.6/3.8 dense) destekleniyor, gelen: {arch}"
        n_block = r.fields[f"{arch}.block_count"].contents()
        st = torch.load(a.conv, map_location="cpu", weights_only=False)
        new, window, sinks = tensors_from_conv(st)

        w = gguf.GGUFWriter(a.out, arch=arch)
        for field in r.fields.values():
            if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
                continue
            vt = field.types[0]
            sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
            w.add_key_value(field.name, field.contents(), vt, sub_type=sub)
        w.add_uint32(f"{arch}.bifrost.window", window)
        w.add_uint32(f"{arch}.bifrost.sinks", sinks)
        w.add_array(f"{arch}.bifrost.layers", [1 if i in new else 0 for i in range(n_block)])
        extra = [(f"blk.{i}.{k}", v) for i in sorted(new) for k, v in new[i].items()]
        for t in r.tensors:
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        for name, v in extra:
            w.add_tensor_info(name, v.shape, v.dtype, v.nbytes)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_ti_data_to_file()
        for t in r.tensors:
            w.write_tensor_data(t.data, tensor_endianess=r.endianess)
        for _, v in extra:
            w.write_tensor_data(v)
        w.close()
    print(f"yazıldı: {a.out} | {len(new)} Bifrost katmanı, pencere {window}, çapa {sinks}")


if __name__ == "__main__":
    main()
