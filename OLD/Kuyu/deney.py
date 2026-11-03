"""
deney.py — Kuyu v0 deneyi: "öğrenmek = yazmak" gerçekten işe yarıyor mu?
=======================================================================
    python deney.py                      # CPU
    python deney.py --device dml         # Windows + AMD/Intel GPU (pip install torch-directml)
    python deney.py --device cuda        # ROCm / CUDA

Akış: N bilgi ("proje_1234 port" → "8080") TEK GEÇİŞTE yazılır; sonra %20'si güncellenir,
%5'i silinir, %30'u tekrar gelir (bilinenler atlanmalı). Ölçülenler:
  hatırlama, en eski bilgiler, güncelleme (son değer), silme ve hiç görülmemiş soruda "bilmiyorum",
  farklı yazım / yazım hatası, kalibrasyon, hız, bellek.
Karşılaştırma: aynı anahtarlarla gradyanla öğrenen klasik ağ (tek geçiş ve 10 tur), toplamsal (Hebbian) hafıza.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kuyu import Calibrator, Kuyu

ETYPES = ["proje", "sunucu", "musteri", "ekip", "depo"]
CITIES = ["istanbul", "ankara", "izmir", "bursa", "antalya", "adana", "konya", "gaziantep", "kayseri", "eskisehir",
          "trabzon", "samsun", "mersin", "diyarbakir", "erzurum", "van", "malatya", "sakarya", "kocaeli", "denizli"]


def make_domains(rng):
    syl = ["ka", "ri", "mo", "ta", "le", "su", "na", "po", "zi", "de", "ve", "ro", "mi", "tu", "sa", "ko", "ne", "ba"]
    names = sorted({"".join(rng.choice(syl, size=3)) for _ in range(600)})[:300]
    return {
        "port": [str(p) for p in rng.choice(np.arange(1000, 10000), size=800, replace=False)],
        "sahip": names,
        "surum": sorted({f"v{a}.{b}.{c}" for a, b, c in rng.integers(0, 10, size=(400, 3))})[:200],
        "sehir": CITIES,
        "durum": ["aktif", "pasif", "arsiv", "beklemede"],
        "oncelik": [str(i) for i in range(1, 6)],
        "renk": ["kirmizi", "mavi", "yesil", "sari", "mor", "turuncu", "siyah", "beyaz", "gri", "pembe", "lacivert", "bordo"],
        "boyut": [str(i) for i in range(1, 1001)],
    }


def make_facts(rng, n_entities, attrs_per_entity, domains):
    attrs = list(domains)
    ents = [f"{ETYPES[i % len(ETYPES)]}_{rng.integers(10**5, 10**6)}{i}" for i in range(n_entities)]
    facts = []
    for e in ents:
        for a in rng.choice(attrs, size=attrs_per_entity, replace=False):
            facts.append((e, a, str(rng.choice(domains[a]))))
    order = rng.permutation(len(facts))
    return [facts[i] for i in order], ents


def text(e, a):
    return f"{e} {a}"


def paraphrase(e, a, kind):
    if kind == "turkce":
        return f"{e} için {a} değeri nedir"
    if kind == "ingilizce":
        return f"what is the {a} of {e}"
    if kind == "yazim_hatasi":  # özelliğin bir harfi eksik
        i = len(a) // 2
        return f"{e} {a[:i] + a[i + 1:]}"
    raise ValueError(kind)


def evaluate_kuyu(kuyu, cal, queries, truth, probes=1, threshold=0.5):
    idx, feats = kuyu.recall(queries, probes)
    p = cal.prob(feats)
    answered = p >= threshold
    pred = [kuyu.values[i] if i >= 0 else None for i in idx.tolist()]
    correct = torch.tensor([pv == t for pv, t in zip(pred, truth)])
    return {
        "dogruluk_hepsi": correct.float().mean().item(),
        "cevaplanan_oran": answered.float().mean().item(),
        "cevaplananlarda_dogruluk": correct[answered].float().mean().item() if answered.any() else float("nan"),
    }, p, correct


def ece(p, correct, bins=10):
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            total += m.float().mean().item() * abs(p[m].mean().item() - correct[m].float().mean().item())
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", help="cpu | cuda | dml")
    ap.add_argument("--varlik", type=int, default=20_000, help="varlık sayısı (bilgi = varlık × özellik)")
    ap.add_argument("--ozellik", type=int, default=5)
    ap.add_argument("--raf", type=int, default=256)
    ap.add_argument("--dk", type=int, default=512)
    ap.add_argument("--dv", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kodlayici", choices=["cift", "ngram"], default="cift", help="anahtar kodlayıcı")
    ap.add_argument("--cikti", default="sonuc.json")
    args = ap.parse_args()

    if args.device == "dml":
        import torch_directml
        device = torch_directml.device()
    else:
        device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    domains = make_domains(rng)
    facts, ents = make_facts(rng, args.varlik, args.ozellik, domains)
    N = len(facts)
    kuyu = Kuyu(args.raf, args.dk, args.dv, device=device, seed=args.seed, encoder=args.kodlayici)
    for d in domains.values():  # değer kod kitabı (şema)
        for v in d:
            kuyu.code(v)
    report = {"bilgi": N, "cihaz": str(device), "hafiza_mb": kuyu.memory_mb()}
    print(f"Kuyu v0 | {N:,} bilgi | {args.raf} raf × {args.dk}×{args.dv} | hafıza {kuyu.memory_mb():.0f} MB | {device}")

    # 1) tek geçişte öğrenme
    texts = [text(e, a) for e, a, _ in facts]
    vals = [v for _, _, v in facts]
    t0 = time.time()
    keys = kuyu.enc.encode(texts, mark=True)
    t_enc = time.time() - t0
    t0 = time.time()
    kuyu.write(keys, torch.tensor([kuyu.code_of[v] for v in vals]))
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_write = time.time() - t0
    report["yazma_bilgi_per_s"] = N / t_write
    print(f"1) Öğrenme: {N:,} bilgi tek geçişte {t_write:.2f} s ({N / t_write:,.0f} bilgi/s; kodlama {t_enc:.2f} s)")

    # 2) güncelleme, silme, tekrar
    truth = {(e, a): v for e, a, v in facts}
    upd = rng.choice(N, size=N // 5, replace=False)
    upd_facts = []
    for i in upd:
        e, a, old = facts[i]
        new = str(rng.choice([x for x in domains[a] if x != old]))
        truth[(e, a)] = new
        upd_facts.append((e, a, new))
    kuyu.learn([text(e, a) for e, a, _ in upd_facts], [v for _, _, v in upd_facts])
    alive = list(truth)
    dele = set(rng.choice(len(alive), size=N // 20, replace=False).tolist())
    deleted = [alive[i] for i in dele]
    kuyu.forget([text(e, a) for e, a in deleted])
    for k in deleted:
        truth.pop(k)
    rep = rng.choice(len(facts), size=int(N * 0.3), replace=False)
    rep_keys = [(facts[i][0], facts[i][1]) for i in rep if (facts[i][0], facts[i][1]) in truth]
    before = dict(kuyu.stats)
    kuyu.learn([text(e, a) for e, a in rep_keys], [truth[k] for k in rep_keys], surprise_eps=0.3)
    skipped = kuyu.stats["atlanan"] - before["atlanan"]
    report["tekrar_atlanan_oran"] = skipped / max(len(rep_keys), 1)
    print(f"2) {len(upd_facts):,} güncelleme, {len(deleted):,} silme; {len(rep_keys):,} tekrarın "
          f"%{100 * skipped / max(len(rep_keys), 1):.1f}'i zaten biliniyordu → yazılmadı")

    # 3) kalibrasyon: bilinen + hiç görülmemiş + silinmiş sorular; yarısıyla uydur, diğer yarısıyla ölç
    known = list(truth)
    unseen = [(f"{ETYPES[i % 5]}_{rng.integers(10**7, 10**8)}", str(rng.choice(list(domains)))) for i in range(10000)]
    uns_fit, uns_eval = unseen[:5000], unseen[5000:]
    del_fit, del_eval = deleted[: len(deleted) // 2], deleted[len(deleted) // 2:]
    cal_known = [known[i] for i in rng.choice(len(known), 5000, replace=False)]
    pool = cal_known + uns_fit + del_fit
    idx, feats = kuyu.recall([text(e, a) for e, a in pool])
    pred = [kuyu.values[i] for i in idx.tolist()]
    correct = torch.tensor([p == truth.get(k) for p, k in zip(pred, pool)])
    cal = Calibrator().fit(feats, correct)
    held = [known[i] for i in rng.choice(len(known), 5000, replace=False)] + uns_eval + del_eval
    idx2, feats2 = kuyu.recall([text(e, a) for e, a in held])
    corr2 = torch.tensor([kuyu.values[i] == truth.get(k) for i, k in zip(idx2.tolist(), held)])
    report["ece"] = ece(cal.prob(feats2), corr2)

    # 4) değerlendirme
    rows = {}
    t0 = time.time()
    rows["tum_bilgiler"], _, corr_all = evaluate_kuyu(kuyu, cal, [text(e, a) for e, a in known], [truth[k] for k in known])
    t_read = time.time() - t0
    report["okuma_soru_per_s"] = len(known) / t_read
    oldest = [(e, a) for e, a, _ in facts[: N // 10] if (e, a) in truth]
    rows["en_eski_%10"], _, _ = evaluate_kuyu(kuyu, cal, [text(e, a) for e, a in oldest], [truth[k] for k in oldest])
    upd_alive = [(e, a) for e, a, _ in upd_facts if (e, a) in truth]
    rows["guncellenen(son_deger)"], _, _ = evaluate_kuyu(kuyu, cal, [text(e, a) for e, a in upd_alive], [truth[k] for k in upd_alive])
    _, p_del, _ = evaluate_kuyu(kuyu, cal, [text(e, a) for e, a in del_eval], [None] * len(del_eval))
    _, p_uns, _ = evaluate_kuyu(kuyu, cal, [text(e, a) for e, a in uns_eval], [None] * len(uns_eval))
    rows["silinen→bilmiyorum"] = {"bilmiyorum_orani": (p_del < 0.5).float().mean().item()}
    rows["hic_gorulmemis→bilmiyorum"] = {"bilmiyorum_orani": (p_uns < 0.5).float().mean().item()}
    sample = [known[i] for i in rng.choice(len(known), 5000, replace=False)]
    for kind in ("turkce", "ingilizce", "yazim_hatasi"):
        rows[f"farkli_yazim:{kind}"], _, _ = evaluate_kuyu(kuyu, cal, [paraphrase(e, a, kind) for e, a in sample],
                                                           [truth[k] for k in sample], probes=4)
    report["kuyu"] = rows

    # 5) karşılaştırmalar
    # 5a) toplamsal (Hebbian) hafıza: aynı raflar, delta yok
    heb = Kuyu(args.raf, args.dk, args.dv, device=device, seed=args.seed, encoder=args.kodlayici)
    heb.codes, heb.code_of, heb.values, heb.enc = kuyu.codes, kuyu.code_of, kuyu.values, kuyu.enc
    all_writes = [(text(e, a), v) for e, a, v in facts] + [(text(e, a), v) for e, a, v in upd_facts]
    hk = heb.enc.encode([t for t, _ in all_writes], known_only=True)
    hv = heb.codes[torch.tensor([heb.code_of[v] for _, v in all_writes])]
    uniq, gidx = heb._group(heb.shelf(hk)[:, 0].numpy())      # raf başına S = Σ k vᵀ (toplamsal)
    gmask = torch.from_numpy(gidx >= 0)[..., None]
    gsafe = torch.from_numpy(np.maximum(gidx, 0))
    heb.S[torch.from_numpy(uniq).to(device)] = ((hk[gsafe] * gmask).transpose(1, 2) @ (hv[gsafe] * gmask)).to(device)
    idx_h, _ = heb.recall([text(e, a) for e, a in known])
    heb_acc = np.mean([heb.values[i] == truth[k] for i, k in zip(idx_h.tolist(), known)])
    idx_hu, _ = heb.recall([text(e, a) for e, a in upd_alive])
    heb_upd = np.mean([heb.values[i] == truth[k] for i, k in zip(idx_hu.tolist(), upd_alive)])

    # 5b) klasik ağ: aynı anahtarlar → doğrusal sınıflandırıcı, gradyanla (Adam)
    n_vals = len(kuyu.values)
    X_stream = torch.cat((keys, kuyu.enc.encode([text(e, a) for e, a, _ in upd_facts], known_only=True)))
    y_stream = torch.tensor([kuyu.code_of[v] for v in vals] + [kuyu.code_of[v] for _, _, v in upd_facts])
    X_eval = kuyu.enc.encode([text(e, a) for e, a in known], known_only=True)
    y_eval = torch.tensor([kuyu.code_of[truth[k]] for k in known])
    oldest_set = set(oldest)
    oldest_mask = torch.tensor([k in oldest_set for k in known])

    def classic(epochs, shuffle):
        torch.manual_seed(0)
        net = torch.nn.Linear(args.dk, n_vals)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        t0 = time.time()
        for _ in range(epochs):
            order = torch.randperm(len(X_stream)) if shuffle else torch.arange(len(X_stream))
            for b in range(0, len(order), 64):
                i = order[b:b + 64]
                loss = F.cross_entropy(net(X_stream[i] * 16), y_stream[i])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            ok = net(X_eval * 16).argmax(-1) == y_eval
        return ok.float().mean().item(), ok[oldest_mask].float().mean().item(), time.time() - t0

    c1 = classic(1, False)
    c10 = classic(10, True)
    report["karsilastirma"] = {
        "hebbian": {"dogruluk": heb_acc, "guncellenen": heb_upd},
        "klasik_tek_gecis": {"dogruluk": c1[0], "en_eski_%10": c1[1], "sure_s": c1[2]},
        "klasik_10_tur": {"dogruluk": c10[0], "en_eski_%10": c10[1], "sure_s": c10[2]},
    }

    # özet
    k = rows["tum_bilgiler"]
    print(f"3) Hatırlama (tüm {len(known):,} bilgi): %{100 * k['dogruluk_hepsi']:.2f} | "
          f"en eski %10: %{100 * rows['en_eski_%10']['dogruluk_hepsi']:.2f} | "
          f"güncellenenler (son değer): %{100 * rows['guncellenen(son_deger)']['dogruluk_hepsi']:.2f}")
    print(f"4) 'Bilmiyorum': silinenlerde %{100 * rows['silinen→bilmiyorum']['bilmiyorum_orani']:.1f} | "
          f"hiç görülmemişlerde %{100 * rows['hic_gorulmemis→bilmiyorum']['bilmiyorum_orani']:.1f} | "
          f"kalibrasyon hatası (ECE) {report['ece']:.3f}")
    for kind in ("turkce", "ingilizce", "yazim_hatasi"):
        r = rows[f"farkli_yazim:{kind}"]
        print(f"5) Farklı yazım [{kind}]: doğru %{100 * r['dogruluk_hepsi']:.1f} | cevapladığı %{100 * r['cevaplanan_oran']:.1f}"
              f" | cevapladıklarında doğru %{100 * r['cevaplananlarda_dogruluk']:.1f}")
    print(f"6) Hız: yazma {report['yazma_bilgi_per_s']:,.0f} bilgi/s | okuma {report['okuma_soru_per_s']:,.0f} soru/s")
    print(f"7) Karşılaştırma (aynı anahtarlar):")
    print(f"   Hebbian (toplamsal) hafıza : %{100 * heb_acc:.1f} | güncellenenlerde %{100 * heb_upd:.1f}")
    print(f"   Klasik ağ, tek geçiş       : %{100 * c1[0]:.1f} | en eski %10: %{100 * c1[1]:.1f} | {c1[2]:.1f} s")
    print(f"   Klasik ağ, 10 tur          : %{100 * c10[0]:.1f} | en eski %10: %{100 * c10[1]:.1f} | {c10[2]:.1f} s")
    Path(args.cikti).write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
