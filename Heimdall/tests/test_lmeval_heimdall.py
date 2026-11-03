"""lm-eval adaptörü: toplu (sağ dolgulu) puanlama, tek tek hesapla AYNI mı? Rolling pencereler ve bağlam/devam ayrımı doğru mu?"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("lm_eval", reason="lm-evaluation-harness is not installed (pip install lm-eval)")
import lmeval_heimdall as E  # noqa: E402
from heimdall import HeimdallConfig, HeimdallLM, apply_arch  # noqa: E402
from heimdall.io import DEFAULT_TOKENIZER  # noqa: E402


def _setup():
    cfg = apply_arch(HeimdallConfig(vocab_size=8192, d_model=64, n_layers=4, n_heads=2, head_dim=32, archival_heads=1,
                                    attn_heads=4, attn_kv_heads=2, attn_head_dim=16, chunk_size=16, kernel="torch"), "hibrit")
    torch.manual_seed(0)
    m = HeimdallLM(cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith(("o_proj.weight", "w3.weight", "ffn.w3")):
                p.normal_(0, 0.05)
    return m.eval(), Tokenizer.from_file(str(DEFAULT_TOKENIZER))


def _req(*args):
    from lm_eval.api.instance import Instance
    return Instance(request_type="loglikelihood", doc={}, arguments=args, idx=0)


def test_batched_equals_naive():
    m, tok = _setup()
    lm = E.make_lm(m, tok, torch.device("cpu"), max_len=64, batch_size=3)
    reqs = [("The capital of France is", " Paris"), ("Question: what is 2+2?\nAnswer:", " 4"),
            ("def f(x):\n    return", " x + 1"), ("Once upon a time there was a", " small dragon who liked tea"),
            ("Hello ", "world"), ("a", " b c d e f g h")]
    got = lm.loglikelihood([_req(c, k) for c, k in reqs])
    eos = tok.token_to_id("<eos>")
    for (c, k), (lp, greedy) in zip(reqs, got):
        ctx, cont = None, None
        n = len(c) - len(c.rstrip())
        c2, k2 = (c[:-n], c[-n:] + k) if n else (c, k)
        whole, ctx = tok.encode(c2 + k2).ids, tok.encode(c2).ids
        cont = whole[len(ctx):] if whole[:len(ctx)] == ctx and len(whole) > len(ctx) else tok.encode(k2).ids
        ids = torch.tensor([(ctx or [eos]) + cont])
        with torch.no_grad():
            lg = F.log_softmax(m(ids[:, :-1]).float(), -1)[0]
        ref = float(lg[-len(cont):].gather(-1, ids[0, -len(cont):, None]).sum())
        assert abs(ref - lp) < 1e-3, (c, k, ref, lp)                   # sağ dolgu sonucu değiştirmemeli
        assert isinstance(greedy, bool)
    assert all(lp < 0 for lp, _ in got)


def test_trailing_space_moves_to_continuation():
    m, tok = _setup()
    lm = E.make_lm(m, tok, torch.device("cpu"), max_len=64)
    a = lm.loglikelihood([_req("The answer is ", "yes")])[0][0]
    b = lm.loglikelihood([_req("The answer is", " yes")])[0][0]
    assert abs(a - b) < 1e-4, (a, b)


def test_rolling_and_generate():
    from lm_eval.api.instance import Instance
    m, tok = _setup()
    lm = E.make_lm(m, tok, torch.device("cpu"), max_len=32)
    text = "The quick brown fox jumps over the lazy dog. " * 12                  # >32 token: birden çok pencere
    r = lm.loglikelihood_rolling([Instance(request_type="loglikelihood_rolling", doc={}, arguments=(text,), idx=0)])[0]
    n_tok = len(tok.encode(text).ids)
    assert r < 0 and n_tok > 32 and r / n_tok > -12                              # her token puanlandı, makul büyüklük
    g = lm.generate_until([Instance(request_type="generate_until", doc={}, arguments=("Hello", {"until": ["\n"], "max_gen_toks": 5}), idx=0)])
    assert isinstance(g[0], str)


if __name__ == "__main__":
    test_batched_equals_naive()
    test_trailing_space_moves_to_continuation()
    test_rolling_and_generate()
    print("OK lm-eval adaptörü: toplu = tek tek, boşluk kuralı, rolling, üretim")
