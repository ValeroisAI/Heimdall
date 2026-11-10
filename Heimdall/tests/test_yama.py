"""Yamalı bayt modeli: nedensellik (gelecek baytlar geçmiş tahminleri etkilemez) ve eğitim adımı."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heimdall.config import HeimdallConfig  # noqa: E402
from heimdall.yama import PatchConfig, PatchLM  # noqa: E402


def _model(patch):
    torch.manual_seed(0)
    trunk = HeimdallConfig(vocab_size=256, d_model=64, n_layers=4, n_heads=2, head_dim=32, archival_heads=1,
                           attn_heads=4, attn_kv_heads=2, attn_head_dim=16, kernel="torch")
    m = PatchLM(PatchConfig(trunk, patch=patch, d_local=64, local_enc=1, local_dec=2)).eval()
    with torch.no_grad():                               # sıfır başlatılan çıkışlar: nedensellik testi anlamlı olsun
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return m


def test_causal():
    for patch in (4, 8):
        m = _model(patch)
        ids = torch.randint(0, 256, (2, 64))
        with torch.no_grad():
            a = m(ids)
            for cut in (1, patch - 1, patch, patch + 1, 3 * patch + 2, 63):
                ids2 = ids.clone()
                ids2[:, cut:] = torch.randint(0, 256, (2, 64 - cut))
                b = m(ids2)
                assert torch.allclose(a[:, :cut], b[:, :cut], atol=1e-5), (patch, cut)
                assert not torch.allclose(a[:, cut:], b[:, cut:], atol=1e-5)


def test_trunk_reaches_next_byte():
    """Bir yamanın son baytı, sonraki yamanın İLK baytının tahminini gövde üzerinden etkiler (gecikme yok)."""
    m = _model(8)
    for blk in list(m.enc) + list(m.dec):               # yerel yol kapalı: etki yalnız gövdeden gelebilir
        for p in blk.parameters():
            p.data.zero_()
    ids = torch.randint(0, 256, (1, 32))
    ids2 = ids.clone()
    ids2[0, 6] = (ids[0, 6] + 1) % 256                  # 0. yamanın içi (son bayt 7 kendi gömmesiyle etkilerdi)
    with torch.no_grad():
        a, b = m(ids), m(ids2)
    assert not torch.allclose(a[0, 7], b[0, 7], atol=1e-6)      # konum 7, bayt 8'i (1. yamanın ilki) tahmin eder


def test_train_step_reduces_loss():
    m = _model(4).train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    ids = torch.randint(0, 256, (4, 33))
    first = None
    for _ in range(30):
        loss = m(ids[:, :-1], ids[:, 1:])
        first = first or loss.item()
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert loss.item() < first - 0.5
    assert m.cost_params() < m.num_params()
