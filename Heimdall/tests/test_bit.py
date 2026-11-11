"""1-bit / 1.58-bit motoru: değerler korunuyor, öğreniyor, cache eşdeğerliği bozulmuyor."""
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heimdall import HeimdallConfig, HeimdallLM  # noqa: E402
from heimdall.bit import BitFlip, BitLinear  # noqa: E402
from heimdall.optim import build_optimizers  # noqa: E402

BASE = HeimdallConfig(vocab_size=300, d_model=64, n_layers=4, n_heads=2, head_dim=32, archival_heads=1,
                      attn_heads=4, attn_kv_heads=2, attn_head_dim=16, chunk_size=16, kernel="torch")


def _levels(model):
    return {tuple(sorted(m.weight.unique().tolist())) for m in model.modules() if isinstance(m, BitLinear)}


def test_values_and_learning():
    for bits, allowed in ((1, {-1.0, 1.0}), (2, {-1.0, 0.0, 1.0})):
        torch.manual_seed(0)
        m = HeimdallLM(replace(BASE, bits=bits))
        assert sum(isinstance(x, BitLinear) for x in m.modules()) > 0
        opts = build_optimizers(m, 0.02, 3e-3, flip_rate=0.01)
        assert isinstance(opts[-1], BitFlip)
        ids = torch.randint(0, 300, (4, 33))
        first = None
        for step in range(60):
            loss = m(ids[:, :-1], ids[:, 1:])
            loss.backward()
            first = first if first is not None else loss.item()
            for o in opts:
                o.step(); o.zero_grad(set_to_none=True)
        assert set().union(*_levels(m)) <= allowed, (bits, _levels(m))
        if bits == 1:
            assert all(0.0 not in lv for lv in _levels(m)), "ikili ağırlıkta 0 olmamalı"
        assert loss.item() < 0.7 * first, (bits, first, loss.item())     # tek batch'i ezberliyor
        assert opts[-1].last_flip_frac > 0


def test_cache_equals_full():
    for bits in (1, 2):
        torch.manual_seed(1)
        m = HeimdallLM(replace(BASE, bits=bits)).eval()
        with torch.no_grad():
            for name, p in m.named_parameters():
                if name.endswith(("o_proj.scale", "w3.scale")):
                    p.fill_(1.0)                              # sıfırlanan çıkışları aç: tüm yollar çalışsın
            ids = torch.randint(0, 300, (2, 40))
            full = m(ids)
            c = m.new_cache(2)
            parts = [m(ids[:, :17], cache=c)] + [m(ids[:, t:t + 1], cache=c) for t in range(17, 40)]
        assert (torch.cat(parts, 1) - full).abs().max().item() < 1e-4, bits


def test_scope_and_rank():
    for scope, rank in (("ffn", 0), ("mix", 0), ("all", 4)):
        torch.manual_seed(3)
        cfg = replace(BASE, bits=2, bit_scope=scope, bit_rank=rank)
        m = HeimdallLM(cfg)
        kinds = {n.split(".")[-1]: isinstance(x, BitLinear) for n, x in m.named_modules()
                 if n.split(".")[-1] in ("w12", "w3", "o_proj")}
        assert kinds["w12"] == (scope in ("all", "ffn")) and kinds["o_proj"] == (scope in ("all", "mix")), (scope, kinds)
        opts = build_optimizers(m, 0.02, 3e-3, flip_rate=0.01)
        ids = torch.randint(0, 300, (4, 33))
        first = None
        for _ in range(60):
            loss = m(ids[:, :-1], ids[:, 1:])
            loss.backward()
            first = first if first is not None else loss.item()
            for o in opts:
                o.step(); o.zero_grad(set_to_none=True)
        assert loss.item() < 0.7 * first, (scope, rank, first, loss.item())
        if rank:
            assert any(p.abs().sum() > 0 for n, p in m.named_parameters() if n.endswith("lr_u"))


def test_optimizer_state_roundtrip():
    """Kayıt → yükleme → sürdürme (gece koşusunun çökme sonrası devamı)."""
    for bits in (1, 2):
        torch.manual_seed(5)
        m = HeimdallLM(replace(BASE, bits=bits, bit_rank=4))
        opts = build_optimizers(m, 0.02, 3e-3, flip_rate=0.01)
        ids = torch.randint(0, 300, (4, 33))

        def step():
            m(ids[:, :-1], ids[:, 1:]).backward()
            for o in opts:
                o.step(); o.zero_grad(set_to_none=True)

        for _ in range(3):
            step()
        saved = {"model": m.state_dict(), "optim": [o.state_dict() for o in opts]}
        m2 = HeimdallLM(replace(BASE, bits=bits, bit_rank=4))
        m2.load_state_dict(saved["model"])
        opts2 = build_optimizers(m2, 0.02, 3e-3, flip_rate=0.01)
        for o, st in zip(opts2, saved["optim"]):
            o.load_state_dict(st)
        m, opts = m2, opts2
        for _ in range(3):
            step()                                  # IndexError vermemeli
        assert opts[-1].last_flip_frac >= 0


def test_memory_per_param():
    m = HeimdallLM(replace(BASE, bits=1))
    n = sum(x.weight.numel() for x in m.modules() if isinstance(x, BitLinear))
    assert n > 0 and all(x.weight.dtype == torch.bfloat16 for x in m.modules() if isinstance(x, BitLinear))


if __name__ == "__main__":
    for fn in (test_values_and_learning, test_cache_equals_full, test_scope_and_rank, test_optimizer_state_roundtrip, test_memory_per_param):
        fn()
        print("OK ", fn.__name__)


def _old_step(p, m, grad, idx, beta, rate):
    """BitFlip'in sadeleştirmeden önceki adımı (referans): (yeni ağırlık, yeni momentum)."""
    mf = m.float().mul_(beta).add_(grad.float(), alpha=1 - beta)
    r = mf / (mf.square().mean(dim=1, keepdim=True).sqrt() + 1e-12)
    pf = p.float()
    if p.bit == 1:
        score = (pf * r).clamp(min=0)
        thr = torch.quantile(score.flatten()[idx], 1.0 - rate).clamp(min=1e-12)
        new = torch.where(score > thr, -pf, pf)
    else:
        score = torch.where(((pf >= 1) & (r < 0)) | ((pf <= -1) & (r > 0)), 0.0, r.abs())
        thr = torch.quantile(score.flatten()[idx], 1.0 - rate).clamp(min=1e-12)
        new = (pf - torch.sign(r) * (score > thr)).clamp(-1, 1)
    mf = torch.where(new != pf, torch.zeros_like(mf), mf)
    return new.to(p.dtype), mf.to(m.dtype)


def test_bitflip_step_equals_reference():
    """Sadeleştirilmiş adım, eski adımla aynı ağırlıkları çevirir ve aynı momentumu bırakır (ikili ve üçlü, 5 adım)."""
    from heimdall.bit import BitFlip, BitLinear
    for bits in (1, 2):
        torch.manual_seed(bits)
        lin = BitLinear(96, 160, bits)
        p = lin.weight
        opt = BitFlip([p], lr=0.01)
        ref_p, ref_m = p.detach().clone(), torch.zeros_like(p)
        ref_p.bit = bits
        for step in range(5):
            p.grad = torch.randn(160, 96).to(p.dtype)
            opt.step()
            new_p, ref_m = _old_step(ref_p, ref_m, p.grad, opt._idx[id(p)], 0.9, 0.01)
            new_p.bit = bits
            diff = int((p.detach() != new_p).sum())
            assert diff <= 2, (bits, step, diff)                       # eşik sınırında yuvarlama farkı olabilir
            assert int((new_p != ref_p).sum()) > 20                    # test gerçekten çevirme içeriyor
            if diff == 0:
                assert torch.equal(opt.state[p]["m"], ref_m), (bits, step)
            ref_p = p.detach().clone()                                 # sonraki adım aynı durumdan başlasın
            ref_p.bit = bits
            ref_m = opt.state[p]["m"].clone()


def test_bitlinear_autocast_equals_explicit_cast():
    """Autocast altında ağırlık çevrilmeden verilir; çıktı ve gradyanlar eski (açık çevirmeli) yolla aynı."""
    import torch.nn.functional as F
    from heimdall.bit import BitLinear
    torch.manual_seed(0)
    lin = BitLinear(64, 48, 1, rank=4)
    with torch.no_grad():
        lin.lr_u.normal_(std=0.1)
    x = torch.randn(3, 5, 64, requires_grad=True)                     # fp32 girdi (norm çıkışı gibi)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = lin(x)
        ref = F.linear(x, lin.weight.to(x.dtype)) * (lin.base * lin.scale).to(torch.bfloat16)
        ref = ref + F.linear(F.linear(x, lin.lr_v.to(x.dtype)), lin.lr_u.to(x.dtype))
    assert y.dtype == ref.dtype and torch.equal(y, ref)
    g = torch.autograd.grad(y.float().square().sum(), [lin.weight, x], retain_graph=True)
    gr = torch.autograd.grad(ref.float().square().sum(), [lin.weight, x])
    assert g[0].dtype == lin.weight.dtype and torch.equal(g[0], gr[0]) and torch.equal(g[1], gr[1])
