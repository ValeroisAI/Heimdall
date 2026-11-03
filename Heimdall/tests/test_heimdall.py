"""
Doğruluk testleri (CPU'da ~1 dk):  python -m pytest tests -q   ya da   python tests/test_heimdall.py
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heimdall import HeimdallConfig, HeimdallLM, apply_arch  # noqa: E402
from heimdall.engine import Engine, Request  # noqa: E402
from heimdall.kernels import delta_rule_reference, delta_rule_step  # noqa: E402

BASE = HeimdallConfig(vocab_size=300, d_model=64, n_layers=4, n_heads=2, head_dim=32, archival_heads=1,
                      attn_heads=4, attn_kv_heads=2, attn_head_dim=16, chunk_size=16, kernel="torch")
VARIANTS = {"hibrit": apply_arch(BASE, "hibrit"), "transformer": apply_arch(BASE, "transformer"),
            "sabit": replace(apply_arch(BASE, "sabit"), attn_window=8),
            "moe": replace(apply_arch(BASE, "hibrit"), moe_experts=8, moe_topk=2, moe_hidden=32)}


def _model(cfg, seed=0):
    torch.manual_seed(seed)
    m = HeimdallLM(cfg)
    with torch.no_grad():  # sıfır başlatılan çıkışları aç: testler tüm yolları görsün
        for name, p in m.named_parameters():
            if name.endswith(("o_proj.weight", "w3.weight", "ffn.w3")):
                p.normal_(0, 0.05)
    return m.eval()


def test_delta_rule_chunk_equals_recurrence():
    torch.manual_seed(0)
    b, t, h, d = 2, 45, 2, 16
    q = F.normalize(torch.randn(b, t, h, d), dim=-1) * d ** -0.5
    k = F.normalize(torch.randn(b, t, h, d), dim=-1)
    v, g, beta = torch.randn(b, t, h, d), -torch.rand(b, t, h) * 0.3, torch.rand(b, t, h)
    s0 = torch.randn(b, h, d, d) * 0.1
    o, s = delta_rule_reference(q, k, v, g, beta, 16, s0)
    st, outs = s0, []
    for i in range(t):
        oi, st = delta_rule_step(q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i], st)
        outs.append(oi)
    assert torch.allclose(o, torch.stack(outs, 1), atol=1e-4) and torch.allclose(s, st, atol=1e-4)


def test_causal():
    for name, cfg in VARIANTS.items():
        m = _model(cfg)
        ids = torch.randint(0, 300, (2, 40))
        ids2 = ids.clone()
        ids2[:, 25:] = torch.randint(0, 300, (2, 15))
        with torch.no_grad():
            a, b = m(ids), m(ids2)
        assert torch.allclose(a[:, :25], b[:, :25], atol=1e-5), name
        assert not torch.allclose(a[:, 25:], b[:, 25:]), name


def test_cache_matches_full_forward():
    for name, cfg in VARIANTS.items():
        m = _model(cfg)
        ids = torch.randint(0, 300, (2, 50))
        with torch.no_grad():
            full = m(ids)
            c = m.new_cache(2)
            parts = [m(ids[:, s:e], cache=c) for s, e in ((0, 13), (13, 30))]   # parçalı prefill
            parts += [m(ids[:, t:t + 1], cache=c) for t in range(30, 50)]       # token token
        err = (torch.cat(parts, 1) - full).abs().max().item()
        assert err < 1e-4, (name, err)


def test_continuous_batching_equals_single():
    for name, cfg in VARIANTS.items():
        m = _model(cfg)
        prompts = [torch.randint(3, 300, (n,)).tolist() for n in (5, 17, 3, 11)]
        ref = []
        with torch.no_grad():
            for p in prompts:  # tek tek açgözlü üretim
                c = m.new_cache(1)
                logits = m(torch.tensor([p]), cache=c)[:, -1]
                out = []
                for _ in range(12):
                    t = int(logits.argmax())
                    out.append(t)
                    logits = m(torch.tensor([[t]]), cache=c)[:, -1]
                ref.append(out)
        eng = Engine(m, max_batch=3, max_context=64)  # 4 istek, 3 yuva: biri kuyrukta bekler, sonra katılır
        reqs = [eng.submit(Request(p, max_tokens=12, temperature=0.0)) for p in prompts]
        eng.start()
        import time
        t0 = time.time()
        while not all(r.done for r in reqs) and time.time() - t0 < 60:
            time.sleep(0.01)
        eng.shutdown()
        assert [r.out for r in reqs] == ref, name


def test_server_api():
    from fastapi.testclient import TestClient
    from tokenizers import Tokenizer

    from heimdall.io import DEFAULT_TOKENIZER
    from heimdall.server import Settings, create_app, hash_key

    tok = Tokenizer.from_file(str(DEFAULT_TOKENIZER))
    m = _model(replace(VARIANTS["hibrit"], vocab_size=tok.get_vocab_size()))
    eng = Engine(m, max_batch=4, max_context=256).start()
    app = create_app(eng, tok, Settings(api_key_hashes={hash_key("gizli"): "test"}, rpm=5))
    cl = TestClient(app)
    auth = {"Authorization": "Bearer gizli"}
    assert cl.post("/v1/completions", json={"prompt": "def f():"}).status_code == 401
    r = cl.post("/v1/completions", headers=auth, json={"prompt": "def f():", "max_tokens": 8, "temperature": 0})
    assert r.status_code == 200 and r.json()["usage"]["completion_tokens"] <= 8, r.text
    with cl.stream("POST", "/v1/chat/completions", headers=auth,
                   json={"messages": [{"role": "user", "content": "selam"}], "max_tokens": 6, "stream": True}) as s:
        lines = [x for x in s.iter_lines() if x.startswith("data: ")]
    assert lines[-1] == "data: [DONE]" and json.loads(lines[0][6:])["object"] == "chat.completion.chunk"
    codes = [cl.post("/v1/completions", headers=auth, json={"prompt": "x", "max_tokens": 1}).status_code
             for _ in range(5)]
    assert 429 in codes, codes                                    # dakikada 5 istek sınırı
    assert "heimdall_requests_total" in cl.get("/metrics").text and cl.get("/health").json()["status"] == "ok"
    eng.shutdown()


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn()
        print("OK ", fn.__name__)


def test_mtp_loss_train_only():
    """MTP: eğitimde ek kayıp, değerlendirmede ana kayıp aynı; gradyan MTP modüllerine akar."""
    from heimdall.config import HeimdallConfig
    from heimdall.model import HeimdallLM
    torch.manual_seed(0)
    base = HeimdallConfig(vocab_size=128, d_model=64, n_layers=4, n_heads=2, head_dim=32, archival_heads=1,
                          attn_heads=4, attn_kv_heads=2, attn_head_dim=16, kernel="torch")
    m0 = HeimdallLM(base)
    from dataclasses import replace
    m2 = HeimdallLM(replace(base, mtp_depth=2))
    m2.load_state_dict(m0.state_dict(), strict=False)
    x = torch.randint(0, 128, (2, 33))
    m0.eval(), m2.eval()
    with torch.no_grad():
        assert abs(m0(x[:, :-1], x[:, 1:]).item() - m2(x[:, :-1], x[:, 1:]).item()) < 1e-5
    m2.train()
    loss = m2(x[:, :-1], x[:, 1:])
    loss.backward()
    assert all(p.grad is not None for p in m2.mtp.parameters() if p.requires_grad)
    assert len(m2.mtp) == 2
