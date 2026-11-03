"""motor.py: blok blok dikkat = SDPA, GQA delta kuralı = referans, motorun kaybı ve gradyanı = eski aşama A'nınki
(HybridAttention.distill_pair + rel_mse), uçtan uca küçük bir koşu."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from heimdall.donustur_hibrit import rel_mse  # noqa: E402
from heimdall.hibrit import HybridLM  # noqa: E402
from heimdall.kernels import delta_rule_reference  # noqa: E402
from heimdall.motor import blockwise_attention, delta_rule_gqa, precompute, seq_loss, window_only_loss  # noqa: E402
from test_hibrit import _converted, tiny_dir  # noqa: E402,F401


def test_blockwise_attention_equals_sdpa():
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 6, 61, 16), torch.randn(1, 2, 61, 16), torch.randn(1, 2, 61, 16)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True).transpose(1, 2)
    for qb, kb in ((16, 8), (7, 13), (64, 64)):
        assert (blockwise_attention(q, k, v, qb, kb) - ref).abs().max() < 1e-5


def test_delta_rule_gqa_equals_reference():
    torch.manual_seed(0)
    B, T, G, R, D = 1, 77, 2, 3, 8
    q = torch.randn(B, T, G, R, D)
    k = F.normalize(torch.randn(B, T, G, D), dim=-1)
    v, g, beta = torch.randn(B, T, G, D), -torch.rand(B, T, G), torch.rand(B, T, G)
    s0 = torch.randn(B, G, D, D) * 0.1
    rep = lambda z: z.repeat_interleave(R, 2)
    o_ref, s_ref = delta_rule_reference(q.reshape(B, T, G * R, D), rep(k), rep(v), rep(g), rep(beta), 16,
                                        s0.repeat_interleave(R, 1))
    o, s = delta_rule_gqa(q, k, v, g, beta, 16, s0)
    assert (o.reshape(B, T, G * R, D) - o_ref).abs().max() < 1e-5 and (s - s_ref[:, ::R]).abs().max() < 1e-5


@pytest.mark.parametrize("seg", [16, 24, 128])
def test_motor_loss_and_grad_equal_stage_a(tiny_dir, seg):
    """Bütün konumlar (≥ W) ağırlık 1 iken motorun kaybı = rel_mse(öğrenci, öğretmen, skip=W); gradyanlar da aynı."""
    m = _converted(tiny_dir, window=8, sinks=2)
    layer = m.layers[3]
    a = layer.self_attn
    torch.manual_seed(1)
    h = torch.randn(61, m.c.d)
    P = torch.arange(a.window, 61)
    rec, _ = precompute(layer, h, P, torch.ones(len(P)))
    params = [a.fq, a.fk, a.ab.weight, a.ab.bias, a.mix.weight, a.mix_lse, a.mem_scale, a.A_log, a.dt_bias]
    for p in params:
        p.requires_grad_(True)
    yt, ys = a.distill_pair(layer.input_layernorm(h[None]))
    ref = rel_mse(ys, yt, a.window)
    g_ref = torch.autograd.grad(ref, params)
    got = seq_loss(a, rec, seg, a.o_proj.weight.float())
    g_got = torch.autograd.grad(got, params)
    assert abs(got.item() - ref.item()) < 1e-5 * max(1.0, ref.item())
    for x, y in zip(g_got, g_ref):
        assert (x - y).abs().max() <= 1e-4 * max(1.0, y.abs().max().item())
    a.mode = "window"
    yt, yw = a.distill_pair(layer.input_layernorm(h[None]))
    assert abs(window_only_loss(rec) - rel_mse(yw, yt, a.window).item()) < 1e-5


def test_motor_end_to_end(tiny_dir, tmp_path):
    """Küçük modelde tam koşu: dönüşüm dosyası yüklenir, cache yolu tam diziyle aynı, hafıza pencereyi geçer."""
    from heimdall import motor
    data = np.random.default_rng(0).integers(1, 300, 20000).astype(np.uint16)
    data.tofile(tmp_path / "train.bin")
    sys.argv = ["motor", "egit", "--model", str(tiny_dir), "--train", str(tmp_path / "train.bin"),
                "--out", str(tmp_path / "run"), "--seq", "96", "--kalib", "4", "--window", "16", "--sinks", "2",
                "--parca", "32", "--adim", "20", "--sentetik", "0", "--kayip-orani", "0.5", "--keep-every", "2",
                "--device", "cpu"]
    motor.main()
    st = torch.load(tmp_path / "run" / "donusum_param.pt", weights_only=False)
    assert [cv["name"] for cv in st["conv"]] == ["layers.3"]           # 2 dikkat katmanından 2.si (7) global kalır
    m = HybridLM.load(tiny_dir).eval()
    m.apply_conversion(st)
    ids = torch.randint(0, 300, (1, 70))
    with torch.no_grad():
        full = m(ids)
        cache, outs = m.new_cache(), []
        for a_, b_ in ((0, 33), (33, 34), (34, 70)):
            outs.append(m(ids[:, a_:b_], cache))
    assert (torch.cat(outs, 1) - full).abs().max() < 1e-4
    rep = st["meta"]["rapor"]["layers.3"]
    assert rep["egitilmis"] <= rep["egitimsiz"] + 1e-6
