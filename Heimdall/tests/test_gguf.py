"""gguf_cevir.py: yaz → oku → aynı logitler (Heimdall ikili FFN TQ2_0 ile, Bifrost dönüşümü dosyanın içinde)."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

pytest.importorskip("gguf")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from heimdall.config import HeimdallConfig  # noqa: E402
from heimdall.gguf_cevir import export_bifrost, export_heimdall, load_heimdall_gguf  # noqa: E402
from heimdall.hibrit import HybridLM  # noqa: E402
from heimdall.model import HeimdallLM  # noqa: E402
from test_hibrit import _converted, tiny_dir  # noqa: E402,F401


def _heimdall_ckpt(tmp_path):
    torch.manual_seed(0)
    cfg = HeimdallConfig(vocab_size=512, d_model=256, n_layers=4, n_heads=2, head_dim=64, archival_heads=1,
                         attn_heads=4, attn_kv_heads=2, attn_head_dim=64, bits=1, bit_scope="ffn", bit_rank=4,
                         mtp_depth=1, kernel="torch")
    m = HeimdallLM(cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if not getattr(p, "bit", 0):
                p.add_(torch.randn_like(p) * 0.02)
    path = tmp_path / "h.pt"
    torch.save({"config": cfg.to_dict(), "model": m.state_dict()}, path)
    return m.eval(), path


@pytest.mark.parametrize("tip", ["f32", "bf16"])
def test_heimdall_roundtrip(tmp_path, tip):
    m, ckpt = _heimdall_ckpt(tmp_path)
    out = tmp_path / f"h_{tip}.gguf"
    export_heimdall(ckpt, out, tip, log=lambda *_: None)
    m2 = load_heimdall_gguf(out)
    ffn = m.blocks[0].ffn.w12
    assert torch.equal(m2.blocks[0].ffn.w12.weight, ffn.weight)           # ikili ağırlıklar birebir
    ids = torch.randint(0, 512, (1, 40))
    with torch.no_grad():
        a, b = m(ids), m2(ids)
    tol = 2e-3 if tip == "f32" else 5e-2
    assert (a - b).abs().max() < tol * a.abs().max()
    assert out.stat().st_size < ckpt.stat().st_size


@pytest.mark.parametrize("tip", ["f32", "q8_0"])
def test_bifrost_roundtrip(tiny_dir, tmp_path, tip):  # noqa: F811
    m = _converted(tiny_dir)
    conv = tmp_path / "conv.pt"
    torch.save(m.conversion_state(), conv)
    out = tmp_path / f"b_{tip}.gguf"
    export_bifrost(tiny_dir, conv, out, tip, log=lambda *_: None)
    g = HybridLM.load(out).eval()
    g.apply_conversion(g.embedded_conversion)
    ids = torch.randint(0, 300, (1, 40))
    with torch.no_grad():
        a, b = m(ids), g(ids)
    if tip == "f32":
        assert (a - b).abs().max() < 1e-5
    else:
        assert (a.argmax(-1) == b.argmax(-1)).float().mean() > 0.9
