"""hibrit.py: Qwen3.5 öğretmeni transformers ile birebir mi, dönüştürülmüş katmanın cache yolu tam diziyle aynı mı,
bellek sabit mi, MTP'li üretim MTP'siz üretimle aynı mı."""

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heimdall.hibrit import HybridConfig, HybridLM  # noqa: E402

TEXT = dict(model_type="qwen3_5_text", vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
            num_attention_heads=4, num_key_value_heads=2, head_dim=32, linear_num_key_heads=2,
            linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
            full_attention_interval=4, attn_output_gate=True, rms_norm_eps=1e-6, hidden_act="silu",
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
            tie_word_embeddings=False, mtp_num_hidden_layers=1, eos_token_id=0)


def _noise(model, scale=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=g) * scale)


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    """Rastgele küçük Qwen3.5 (MTP dahil), Qwen3_5ForConditionalGeneration ad düzeniyle kaydedilir."""
    from safetensors.torch import save_file
    torch.manual_seed(0)
    m = HybridLM(HybridConfig(TEXT))
    for layer in m.layers:  # gerçekçi başlangıç: conv'un son sütunu baskın, A_log küçük
        if layer.kind == "linear_attention":
            with torch.no_grad():
                layer.linear_attn.conv1d.weight[:, 0, -1] = 1.0
    _noise(m)
    sd = {}
    for k, v in m.state_dict().items():
        key = k if k.startswith(("lm_head", "mtp.")) else "model.language_model." + k
        sd[key] = v.contiguous()
    d = tmp_path_factory.mktemp("tiny_qwen35")
    save_file(sd, str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": TEXT}))
    return d


def _converted(tiny_dir, window=8, sinks=2):
    m = HybridLM.load(tiny_dir).eval()
    m.convert(window=window, sinks=sinks, archival_frac=0.5)
    for _, a in m.attn_modules():  # eğitilmiş gibi: yeni parametreler sıfırdan uzak
        _noise(a.ab, 0.3, 1), _noise(a.mix, 0.3, 2)
        with torch.no_grad():
            a.mix.bias.fill_(0.5)
            a.fq.add_(torch.randn_like(a.fq) * 0.1)
            a.fk.add_(torch.randn_like(a.fk) * 0.1)
    return m


def test_teacher_matches_transformers(tmp_path):
    tr = pytest.importorskip("transformers")
    if not hasattr(tr, "Qwen3_5ForCausalLM"):
        pytest.skip("transformers sürümünde qwen3_5 yok")
    cfg = tr.Qwen3_5TextConfig(**{k: v for k, v in TEXT.items() if k not in ("model_type", "mtp_num_hidden_layers")})
    torch.manual_seed(0)
    ref = tr.Qwen3_5ForCausalLM(cfg).eval()
    _noise(ref)
    ref.save_pretrained(tmp_path)
    ours = HybridLM.load(tmp_path).eval()
    ids = torch.randint(0, 300, (2, 50))
    with torch.no_grad():
        a, b = ref(ids).logits, ours(ids)
    assert (a - b).abs().max() < 1e-4 * a.abs().max()


@pytest.mark.parametrize("mode", ["teacher", "student", "window"])
def test_chunked_cache_equals_full(tiny_dir, mode):
    m = _converted(tiny_dir)
    m.set_mode(mode)
    ids = torch.randint(0, 300, (1, 61))
    with torch.no_grad():
        full = m(ids)
        cache, outs, a = m.new_cache(), [], 0
        for n in (5, 1, 13, 1, 1, 9, 30, 1):          # prefill parçaları + tek tokenler (halka sınırları dahil)
            outs.append(m(ids[:, a:a + n], cache))
            a += n
        assert a == ids.size(1)
    assert (torch.cat(outs, 1) - full).abs().max() < 1e-4


def test_memory_constant(tiny_dir):
    m = _converted(tiny_dir)
    sizes = []
    with torch.no_grad():
        cache = m.new_cache()
        m(torch.randint(0, 300, (1, 20)), cache)
        for n in range(120):
            m(torch.randint(0, 300, (1, 1)), cache)
            if n in (20, 119):
                sizes.append(m.cache_bytes(cache))
    assert sizes[0] == sizes[1]
    m.set_mode("teacher")                               # öğretmen: KV büyür
    with torch.no_grad():
        c1, c2 = m.new_cache(), m.new_cache()
        m(torch.randint(0, 300, (1, 20)), c1)
        m(torch.randint(0, 300, (1, 80)), c2)
    assert m.cache_bytes(c2) > m.cache_bytes(c1)


def test_memory_branch_matters(tiny_dir):
    m = _converted(tiny_dir)
    ids = torch.randint(0, 300, (1, 40))
    with torch.no_grad():
        s = m(ids)
        m.set_mode("window")
        w = m(ids)
    assert (s[:, :8] - w[:, :8]).abs().max() < 1e-5    # pencere dolmadan hafıza devrede değil
    assert (s[:, 8:] - w[:, 8:]).abs().max() > 1e-3


@pytest.mark.parametrize("mode", ["teacher", "student"])
def test_mtp_speculative_equals_greedy(tiny_dir, mode):
    m = _converted(tiny_dir)
    m.set_mode(mode)
    ids = torch.randint(0, 300, (1, 23))
    plain, _ = m.generate(ids, max_new=30, chunk=7)
    stats = {}
    spec, _ = m.generate(ids, max_new=30, chunk=7, mtp=True, stats=stats)
    assert spec == plain and stats["taslak"] > 0


def test_conversion_state_roundtrip(tiny_dir):
    m = _converted(tiny_dir)
    st = m.conversion_state({"x": 1})
    m2 = HybridLM.load(tiny_dir).eval()
    m2.apply_conversion(st)
    ids = torch.randint(0, 300, (1, 30))
    with torch.no_grad():
        assert (m(ids) - m2(ids)).abs().max() < 1e-6


def test_q4_cache_roundtrip(tiny_dir, tmp_path):
    q = HybridLM.load(tiny_dir, q4_group=32, q4_cache=tmp_path / "q4.pt").eval()
    q2 = HybridLM.load(tiny_dir, q4_group=32, q4_cache=tmp_path / "q4.pt").eval()
    full = HybridLM.load(tiny_dir).eval()
    ids = torch.randint(0, 300, (1, 30))
    with torch.no_grad():
        a, b, c = q(ids), q2(ids), full(ids)
    assert (a - b).abs().max() < 1e-6
    assert (a - c).abs().max() < 0.5 * c.abs().max()     # 4-bit: yakın ama aynı değil


def test_exact_within_window(tiny_dir):
    """Bağlam pencereden kısayken dönüştürülmüş model öğretmenle birebir aynıdır."""
    m = _converted(tiny_dir, window=16, sinks=2)
    ids = torch.randint(0, 300, (1, 40))
    with torch.no_grad():
        s = m(ids)
        m.set_mode("teacher")
        t = m(ids)
    assert (s[:, :16] - t[:, :16]).abs().max() < 1e-4
    assert (s[:, 16:] - t[:, 16:]).abs().max() > 1e-3


def test_tied_head_q4_cache(tmp_path):
    """Paylaşılan embedding (Qwen3.5-0.8B gibi) + 4-bit + önbellek: lm_head ayrı 4-bit kopya, önbellekten aynı sonuç."""
    from safetensors.torch import save_file
    text = dict(TEXT, tie_word_embeddings=True)
    torch.manual_seed(1)
    m = HybridLM(HybridConfig(text))
    _noise(m)
    sd = {("mtp." + k[4:] if k.startswith("mtp.") else "model.language_model." + k): v.contiguous()
          for k, v in m.state_dict().items() if k != "lm_head.weight"}
    sd = {k.replace("model.language_model.mtp.", "mtp."): v for k, v in sd.items()}
    save_file(sd, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": text}))
    full = HybridLM.load(tmp_path).eval()
    a = HybridLM.load(tmp_path, q4_group=32, q4_cache=tmp_path / "q4.pt").eval()
    b = HybridLM.load(tmp_path, q4_group=32, q4_cache=tmp_path / "q4.pt").eval()
    ids = torch.randint(0, 300, (1, 20))
    with torch.no_grad():
        x, y, z = full(ids), a(ids), b(ids)
    assert (y - z).abs().max() < 1e-6
    assert (x.argmax(-1) == y.argmax(-1)).float().mean() > 0.7


def test_vocab_linear_chunks_equal_full():
    """LM başı parça parça = tek parça (ileri ve geri), düz ve 4-bit."""
    from heimdall.donustur import Q4Linear
    from heimdall.hibrit import vocab_linear
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 300, bias=False)
    q4 = Q4Linear(lin.weight.detach(), 32, bits=4)
    for mod in (lin, q4):
        h = torch.randn(2, 5, 64, requires_grad=True)
        a = mod(h)
        a.sum().backward()
        g_full = h.grad.clone()
        h.grad = None
        b = vocab_linear(mod, h, chunk=64)
        b.sum().backward()
        assert (a - b).abs().max() < 1e-5 and (g_full - h.grad).abs().max() < 1e-5
