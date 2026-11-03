"""Nedensellik, cache (step/prefill) eşdeğerliği ve Mímir chunk ≡ recurrent testleri.

Çalıştırma (08_BIFROST_H içinden):  python -m pytest -q tests
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bifrost import BifrostLM, ModelConfig  # noqa: E402
from bifrost.layers.kuzgun import local_window_attention  # noqa: E402
from bifrost.layers.mimir import _unit_lower_inverse, gated_delta_chunk, gated_delta_recurrent  # noqa: E402

LAYOUTS = [("MMMM", True), ("NNNN", True), ("AAAA", False), ("MAMN", True), ("KKWR", False)]


def tiny(layout, csl):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=97, dim=64, layout=layout, csl=csl, mimir_heads=2, mimir_dk=32,
                      mimir_dv=32, mimir_chunk=16, attn_heads=2, kuzgun_heads=2, kuzgun_head_dim=32,
                      window=8, logit_softcap=30.0, archival_heads=1)
    return BifrostLM(cfg).eval()


@pytest.mark.parametrize("layout,csl", LAYOUTS)
def test_causality(layout, csl):
    model = tiny(layout, csl)
    ids = torch.randint(0, 97, (2, 60))
    changed = ids.clone()
    changed[:, 35:] = torch.randint(0, 97, (2, 25))
    with torch.no_grad():
        assert (model(ids)[:, :35] - model(changed)[:, :35]).abs().max() == 0


@pytest.mark.parametrize("layout,csl", LAYOUTS)
def test_step_and_prefill_match_forward(layout, csl):
    model = tiny(layout, csl)
    ids = torch.randint(0, 97, (2, 45))
    with torch.no_grad():
        full = model(ids)
        a, state = model.forward_stateful(ids[:, :17])
        b, state = model.forward_stateful(ids[:, 17:], state)
        torch.testing.assert_close(torch.cat((a, b), 1), full, rtol=1e-4, atol=1e-5)
        state = model.init_state(2)
        stepped = []
        for token in ids.unbind(1):
            logits, state = model.step(token, state)
            stepped.append(logits)
        torch.testing.assert_close(torch.stack(stepped, 1), full, rtol=1e-4, atol=1e-5)


def test_mimir_chunk_matches_recurrent_forward_and_grad():
    torch.manual_seed(0)
    b, h, t, d = 2, 2, 50, 16  # t chunk'ın katı değil: dolgu yolu da test edilir
    base = [F.normalize(torch.randn(b, h, t, d), dim=-1), F.normalize(torch.randn(b, h, t, d), dim=-1),
            torch.randn(b, h, t, d)]
    g = F.logsigmoid(torch.randn(b, h, t) + 2)
    beta = torch.sigmoid(torch.randn(b, h, t))
    s0 = torch.randn(b, h, d, d) * 0.1
    outs, grads = [], []
    for fn in (gated_delta_recurrent, lambda *a, **k: gated_delta_chunk(*a, chunk_size=16, **k)):
        q, k, v = (x.clone().requires_grad_() for x in base)
        o, s = fn(q, k, v, g, beta, initial_state=s0)
        (o.square().sum() + s.sum()).backward()
        outs.append((o.detach(), s.detach()))
        grads.append([x.grad for x in (q, k, v)])
    torch.testing.assert_close(outs[0][0], outs[1][0], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(outs[0][1], outs[1][1], rtol=1e-4, atol=1e-5)
    for ga, gb in zip(*grads):
        torch.testing.assert_close(ga, gb, rtol=1e-3, atol=1e-4)


def test_unit_lower_inverse():
    a = torch.randn(3, 8, 8).tril(-1) * 0.3
    ref = torch.linalg.inv(torch.eye(8) + a)
    torch.testing.assert_close(_unit_lower_inverse(a), ref, rtol=1e-5, atol=1e-5)


def test_delta_rule_overwrites_same_key():
    """'Proje ID 1 → 2 → 3': aynı anahtara üç yazımdan sonra okuma son değeri verir."""
    key = F.normalize(torch.randn(1, 1, 3, 32), dim=-1)[:, :, :1].expand(1, 1, 3, 32)
    values = torch.eye(3, 32).view(1, 1, 3, 32) * 4
    g = torch.zeros(1, 1, 3)
    beta = torch.ones(1, 1, 3)
    out, _ = gated_delta_recurrent(key, key, values, g, beta)
    torch.testing.assert_close(out[0, 0, -1], values[0, 0, -1], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("t,w", [(37, 8), (64, 16), (5, 8), (40, 1 + 39)])
def test_local_window_attention_matches_masked_full(t, w):
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, t, 16) for _ in range(3))
    i = torch.arange(t)
    dist = i[:, None] - i[None, :]
    mask = (dist >= 0) & (dist < w)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
    torch.testing.assert_close(local_window_attention(q, k, v, w), ref, rtol=1e-5, atol=1e-5)
