"""Triton çekirdekleri, Triton yorumlayıcısında (CPU) bağımsız torch referansına karşı.
GPU'da derlenmiş çekirdek ayrıca bench/bitkernel.py ile doğrulanır."""
import os
import sys
from pathlib import Path

os.environ["TRITON_INTERPRET"] = "1"
import torch  # noqa: E402
import pytest  # noqa: E402

import importlib.util  # noqa: E402

needs_triton = pytest.mark.skipif(importlib.util.find_spec("triton") is None,
                                  reason="Triton is not installed (pip install triton; on CPU set TRITON_INTERPRET=1)")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heimdall.bitkernel import int8_matmul, pack_weights, packed_matmul, unpack_weights  # noqa: E402


def test_pack_roundtrip():
    for bits, vals in ((1, [-1.0, 1.0]), (2, [-1.0, 0.0, 1.0])):
        for k in (8, 13, 64, 77):
            w = torch.tensor(vals)[torch.randint(0, len(vals), (11, k))]
            assert torch.equal(unpack_weights(pack_weights(w, bits), bits, k), w), (bits, k)


@needs_triton
def test_int8_matmul_exact():
    torch.manual_seed(0)
    for m, k, n in ((40, 200, 70), (3, 65, 17), (130, 64, 130)):
        xq = torch.randint(-127, 128, (m, k), dtype=torch.int8)
        w = torch.randint(-1, 2, (n, k), dtype=torch.int8)
        sx, sw = torch.rand(m) + 0.1, torch.rand(n) + 0.1
        ref = (xq.long() @ w.long().t()).float() * sx[:, None]
        assert torch.allclose(int8_matmul(xq, w, sx), ref, rtol=1e-6, atol=1e-4), (m, k, n)
        assert torch.allclose(int8_matmul(xq, w, sx, sw), ref * sw[None], rtol=1e-6, atol=1e-4)


@needs_triton
def test_packed_matmul():
    torch.manual_seed(1)
    for bits, vals in ((1, [-1.0, 1.0]), (2, [-1.0, 0.0, 1.0])):
        for m, k, n in ((1, 200, 70), (5, 131, 33), (40, 64, 100), (1, 1024, 40), (3, 777, 50)):
            w = torch.tensor(vals)[torch.randint(0, len(vals), (n, k))]
            x, s = torch.randn(m, k), torch.rand(n) + 0.5
            ref = (x @ w.t()) * s
            got = packed_matmul(x, pack_weights(w, bits), s, bits, k)
            assert torch.allclose(got, ref, rtol=1e-4, atol=1e-4), (bits, m, k, n, (got - ref).abs().max())


def test_module_paths_match_dense():
    from dataclasses import replace

    from heimdall import HeimdallConfig, HeimdallLM
    from heimdall.bitkernel import PackedBitLinear, pack_bit_model

    cfg = HeimdallConfig(vocab_size=300, d_model=64, n_layers=2, n_heads=2, head_dim=32, archival_heads=1, attn_heads=4,
                         attn_kv_heads=2, attn_head_dim=16, chunk_size=16, kernel="torch")
    for bits, rank in ((1, 0), (2, 0), (2, 4)):
        torch.manual_seed(2)
        m = HeimdallLM(replace(cfg, bits=bits, bit_rank=rank)).eval()
        with torch.no_grad():
            for n, p in m.named_parameters():
                if n.endswith(("o_proj.scale", "w3.scale")):
                    p.fill_(1.0)
                if n.endswith("lr_u"):
                    p.normal_(0, 0.1)
            ids = torch.randint(0, 300, (2, 24))
            dense = m(ids)
            saved = pack_bit_model(m)                  # CPU'da torch açma yolu (çekirdek yok); format doğrulanır
            assert saved > 0 and any(isinstance(x, PackedBitLinear) for x in m.modules())
            assert (m(ids) - dense).abs().max().item() < 1e-4, bits


if __name__ == "__main__":
    for fn in (test_pack_roundtrip, test_int8_matmul_exact, test_packed_matmul, test_module_paths_match_dense):
        fn()
        print("OK ", fn.__name__)
