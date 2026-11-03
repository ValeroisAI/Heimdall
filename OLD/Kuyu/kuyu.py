"""
kuyu.py — Mímir'in Kuyusu: yaşayan hafıza motoru (v0)
=====================================================
Bir "model" değil, bir hafıza. Dört ilke:

1) Öğrenmek = yazmak.   Bilgi gradyanla ağırlıklara gömülmez; çalışırken tek adımda
                         hafızaya yazılır (delta kuralı). Yeniden eğitim yok, unutma yok.
2) Metin değil, karar.   Cevap tipli bir değer + güven skorudur; emin değilse "bilmiyorum".
3) Sürpriz kadar hesap.  Zaten bilinen bilgi hafızayı değiştirmez (δ ≈ 0 → yazma atlanır).
4) Açık hafıza.          Bir bilgi tek satırla güncellenir ya da silinir.

Yapı:
  anahtar  k = HashEncoder(metin)          öğrenmesiz: kelime, kelime çifti, harf üçlüsü → sabit rastgele vektörler
  raf      r = LSH(k)                       benzer anahtarlar aynı rafa düşer (VRAM'de n_raf adet d×d_v matris)
  yazma    S_r ← S_r + β k (v − S_rᵀ k)ᵀ   delta kuralı: yalnız sürprizi yaz (aynı anahtar → eski değer silinir)
  okuma    v̂ = S_rᵀ k  →  en yakın değer kodu + güven (kalibre edilir)
Yazmalar raflar arasında paralel, raf içinde chunk-paralel (UT dönüşümü) yapılır: GPU/DirectML dostu.
"""

import hashlib
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

TOKEN_RE = re.compile(r"[a-z0-9çğıöşü]+")


def stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


class HashEncoder:
    """Metin → birim anahtar. Öğrenme yok; yazım farklarına harf üçlüleriyle dayanıklı."""

    def __init__(self, dim: int = 512, buckets: int = 1 << 15, seed: int = 0,
                 w_word: float = 1.0, w_pair: float = 2.0, w_char: float = 0.35) -> None:
        g = torch.Generator().manual_seed(seed)
        self.table = torch.randn(buckets, dim, generator=g) / math.sqrt(dim)
        self.buckets = buckets
        self.w = (w_word, w_pair, w_char)
        self._cache: Dict[str, int] = {}
        self.seen = torch.zeros(buckets, dtype=torch.bool)  # hafızaya yazılmış işaretler (açık sözlük)

    def _h(self, f: str) -> int:
        h = self._cache.get(f)
        if h is None:
            h = self._cache[f] = stable_hash(f) % self.buckets
        return h

    def features(self, text: str) -> Tuple[List[int], List[float]]:
        words = TOKEN_RE.findall(text.lower())
        ids, wts = [], []
        for w in words:
            ids.append(self._h("w:" + w)); wts.append(self.w[0])
            p = f"^{w}$"
            for i in range(len(p) - 2):
                ids.append(self._h("c:" + p[i:i + 3])); wts.append(self.w[2])
        for a, b in zip(words, words[1:]):
            ids.append(self._h(f"p:{a}_{b}")); wts.append(self.w[1])
        return ids, wts

    def encode(self, texts: Sequence[str], mark: bool = False, known_only: bool = False) -> torch.Tensor:
        """mark: yazılan işaretleri kaydet. known_only: sorguda hafızanın hiç görmediği işaretleri yok say
        (ör. "için", "nedir", "what is the" gibi soru kalıpları anahtarı bozmaz)."""
        rows, ids, wts = [], [], []
        for r, t in enumerate(texts):
            i, w = self.features(t)
            rows += [r] * len(i); ids += i; wts += w
        ids_t, wts_t, rows_t = torch.tensor(ids), torch.tensor(wts), torch.tensor(rows)
        if mark:
            self.seen[ids_t] = True
        if known_only:
            keep = self.seen[ids_t]
            ids_t, wts_t, rows_t = ids_t[keep], wts_t[keep], rows_t[keep]
        out = torch.zeros(len(texts), self.table.size(1))
        out.index_add_(0, rows_t, self.table[ids_t] * wts_t[:, None])
        return F.normalize(out, dim=-1)


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class PairEncoder:
    """Metin → anahtar = hafızanın tanıdığı kelimelerin SIRASIZ çiftlerinin bağlanması (HRR tarzı ⊙).

    - Kelime sırası ve soru kalıpları ("için", "nedir", "what is the") anahtarı değiştirmez:
      hafızanın hiç görmediği kelimeler atılır, kalan kelimelerin çift kümesi aynıdır.
    - Tanınmayan kelime, hafızanın kendi sözlüğündeki en yakın kelimeye düzeltilir (yazım hatası).
    - idf=True: nadir çiftlere ağırlık (sayım ile). Varsayılan kapalı: ağırlıklar akış boyunca
      değiştiği için yazılan ve sorulan anahtar arasında uyumsuzluk yaratıyor (ölçüldü: %97 → %78).
    Öğrenme yok: rastgele sabit vektörler, sayım ve bağlama.
    """

    def __init__(self, dim: int = 512, buckets: int = 1 << 16, pair_buckets: int = 1 << 21, seed: int = 0,
                 idf: bool = False) -> None:
        self.idf = idf
        g = torch.Generator().manual_seed(seed)
        self.table = torch.randn(buckets, dim, generator=g)
        self.buckets, self.pair_buckets = buckets, pair_buckets
        self.pair_count = torch.zeros(pair_buckets)
        self.vocab: Dict[str, int] = {}
        self.bigram_index: Dict[str, set] = {}
        self._cache: Dict[str, int] = {}

    def _h(self, f: str, mod: int) -> int:
        key = f"{mod}:{f}"
        h = self._cache.get(key)
        if h is None:
            h = self._cache[key] = stable_hash(f) % mod
        return h

    @staticmethod
    def _bigrams(w: str):
        p = f"^{w}$"
        return {p[i:i + 2] for i in range(len(p) - 1)}

    def _learn_word(self, w: str) -> None:
        if w not in self.vocab:
            self.vocab[w] = self._h("w:" + w, self.buckets)
            if not w.isdigit():
                for bg in self._bigrams(w):
                    self.bigram_index.setdefault(bg, set()).add(w)

    def _correct(self, w: str) -> Optional[str]:
        if w in self.vocab:
            return w
        if w.isdigit() or len(w) < 3:
            return None
        votes: Dict[str, int] = {}
        for bg in self._bigrams(w):
            for cand in self.bigram_index.get(bg, ()):
                votes[cand] = votes.get(cand, 0) + 1
        best, best_d = None, max(1, len(w) // 4) + 1
        for cand, _ in sorted(votes.items(), key=lambda kv: -kv[1])[:30]:
            d = levenshtein(w, cand)
            if d < best_d:
                best, best_d = cand, d
        return best

    def words(self, text: str, learn: bool) -> List[str]:
        out = []
        for w in TOKEN_RE.findall(text.lower()):
            if learn:
                self._learn_word(w)
                out.append(w)
            else:
                c = self._correct(w)
                if c is not None:
                    out.append(c)
        return list(dict.fromkeys(out))  # tekrarları at, sırayı koru

    def specificity(self, texts: Sequence[str]) -> torch.Tensor:
        """Sorgudaki EN NADİR çiftin hafızada görülme sayısı (log). Ayırt edici parça yoksa (ör. bilinmeyen
        bir varlık adı atıldıysa, geriye yalnız 'proje|port' gibi yaygın çiftler kalır) değer büyüktür."""
        out = torch.zeros(len(texts))
        for r, t in enumerate(texts):
            ws = self.words(t, learn=False)
            pids = [self._h("|".join(sorted((ws[i], ws[j]))), self.pair_buckets)
                    for i in range(len(ws)) for j in range(i + 1, len(ws))]
            out[r] = math.log1p(float(self.pair_count[torch.tensor(pids)].min())) if pids else 20.0
        return out

    def encode(self, texts: Sequence[str], mark: bool = False, known_only: bool = True) -> torch.Tensor:
        out = torch.zeros(len(texts), self.table.size(1))
        for r, t in enumerate(texts):
            ws = self.words(t, learn=mark)
            if len(ws) == 1:
                out[r] = self.table[self.vocab[ws[0]]]
                continue
            vecs, weights, pids = [], [], []
            for i in range(len(ws)):
                for j in range(i + 1, len(ws)):
                    a, b = sorted((ws[i], ws[j]))
                    pid = self._h(f"{a}|{b}", self.pair_buckets)
                    if mark:
                        self.pair_count[pid] += 1
                    pids.append(pid)
                    vecs.append(self.table[self.vocab[a]] * self.table[self.vocab[b]])
            if vecs:
                if self.idf:
                    w = self.pair_count[torch.tensor(pids)].clamp(min=1).rsqrt()
                    out[r] = (torch.stack(vecs) * w[:, None]).sum(0)
                else:
                    out[r] = torch.stack(vecs).sum(0)
        return F.normalize(out, dim=-1)


def unit_lower_inverse(a: torch.Tensor) -> torch.Tensor:
    """(I + A)^{-1}, A kesin alt üçgen. solve_triangular yoksa (DirectML) ileri yerine koyma."""
    c = a.size(-1)
    eye = torch.eye(c, dtype=a.dtype, device=a.device)
    try:
        return torch.linalg.solve_triangular(eye + a, eye.expand_as(a).contiguous(), upper=False, unitriangular=True)
    except (RuntimeError, NotImplementedError):
        inv = -a.clone()
        for i in range(1, c):
            inv[..., i, :i] = inv[..., i, :i] + (inv[..., i, :, None] * inv[..., :, :i]).sum(-2)
        return inv + eye


def delta_write_chunk(S: torch.Tensor, K: torch.Tensor, V: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Sıralı delta yazmalarının (S ← S + β k (v − Sᵀk)ᵀ) chunk-paralel eşdeğeri.
    S: [n, dk, dv], K: [n, C, dk], V: [n, C, dv], beta: [n, C] (0 = dolgu / atla)."""
    kb = K * beta[..., None]
    w_inv = unit_lower_inverse((kb @ K.transpose(1, 2)).tril(-1))
    u = w_inv @ (V * beta[..., None])
    w = w_inv @ kb
    return S + K.transpose(1, 2) @ (u - w @ S)


class Kuyu:
    def __init__(self, n_shelves: int = 256, dim_key: int = 512, dim_val: int = 256, device="cpu",
                 chunk: int = 64, seed: int = 0, encoder: str = "cift") -> None:
        assert n_shelves & (n_shelves - 1) == 0, "raf sayısı 2'nin kuvveti olmalı"
        self.device = device
        self.n, self.dk, self.dv, self.chunk = n_shelves, dim_key, dim_val, chunk
        self.bits = int(math.log2(n_shelves))
        g = torch.Generator().manual_seed(seed + 1)
        self.planes = torch.randn(self.bits, dim_key, generator=g)            # LSH hiper-düzlemleri (CPU)
        self.S = torch.zeros(n_shelves, dim_key, dim_val, device=device)    # hafıza (VRAM)
        self.enc = PairEncoder(dim_key, seed=seed) if encoder == "cift" else HashEncoder(dim_key, seed=seed)
        self.codes = torch.zeros(0, dim_val)                                  # değer kod kitabı
        self.code_of: Dict[str, int] = {}
        self.values: List[str] = []
        self._g = torch.Generator().manual_seed(seed + 2)
        self.stats = {"yazilan": 0, "atlanan": 0}

    # -------------------------------------------------------------- değer kodları
    def code(self, value: str) -> int:
        i = self.code_of.get(value)
        if i is None:
            i = self.code_of[value] = len(self.values)
            self.values.append(value)
            new = F.normalize(torch.randn(1, self.dv, generator=self._g), dim=-1)
            self.codes = torch.cat((self.codes, new))
        return i

    def shelf(self, keys: torch.Tensor, probes: int = 1) -> torch.Tensor:
        """LSH raf numarası; probes > 1 ise en belirsiz bitler çevrilerek ek raflar (çoklu yoklama)."""
        proj = keys @ self.planes.T                                           # [N, bits]
        bits = (proj > 0).long()
        weights = 2 ** torch.arange(self.bits)
        base = (bits * weights).sum(-1)
        if probes == 1:
            return base[:, None]
        order = proj.abs().argsort(dim=-1)[:, :probes - 1]                    # en düşük güvenli bitler
        flips = base[:, None] ^ (2 ** order)
        return torch.cat((base[:, None], flips), dim=1)

    # -------------------------------------------------------------- raf-gruplu yerleşim
    def _group(self, shelves: np.ndarray):
        """Öğeleri raflara göre grupla (akış sırası korunur). Döndürür: raf listesi, [n_raf, L] indeks matrisi (−1 = boş)."""
        order = np.argsort(shelves, kind="stable")
        s_sorted = shelves[order]
        uniq, start, counts = np.unique(s_sorted, return_index=True, return_counts=True)
        L = int(counts.max())
        idx = np.full((len(uniq), L), -1, dtype=np.int64)
        for j, (a, c) in enumerate(zip(start, counts)):
            idx[j, :c] = order[a:a + c]
        return uniq, idx

    def _read_shelves(self, keys: torch.Tensor, shelves: np.ndarray) -> torch.Tensor:
        uniq, idx = self._group(shelves)
        mask = torch.from_numpy(idx >= 0)
        K = keys[torch.from_numpy(np.maximum(idx, 0))] * mask[..., None]                 # [u, L, dk]
        pred = (K.to(self.device) @ self.S[torch.from_numpy(uniq).to(self.device)]).cpu()  # [u, L, dv]
        out = torch.zeros(keys.size(0), self.dv)
        out[torch.from_numpy(idx[idx >= 0])] = pred[mask]
        return out

    # -------------------------------------------------------------- yazma
    @torch.no_grad()
    def write(self, keys: torch.Tensor, value_ids: Optional[torch.Tensor], surprise_eps: float = 0.0) -> None:
        """Akış sırasıyla yaz. value_ids=None → silme (v = 0). surprise_eps > 0 → bilinenleri atla."""
        N = keys.size(0)
        V = torch.zeros(N, self.dv) if value_ids is None else self.codes[value_ids]
        beta = torch.ones(N)
        shelves = self.shelf(keys)[:, 0].numpy()
        if surprise_eps > 0:
            surprise = (V - self._read_shelves(keys, shelves)).norm(dim=-1)
            beta = (surprise > surprise_eps).float()
        self.stats["yazilan"] += int(beta.sum()); self.stats["atlanan"] += int(N - beta.sum())
        uniq, idx = self._group(shelves)
        shelf_t = torch.from_numpy(uniq).to(self.device)
        S = self.S[shelf_t]
        for c0 in range(0, idx.shape[1], self.chunk):                          # raf içinde sıralı chunk'lar
            part = idx[:, c0:c0 + self.chunk]
            valid = torch.from_numpy(part >= 0)
            safe = torch.from_numpy(np.maximum(part, 0))
            Kc = (keys[safe] * valid[..., None]).to(self.device)
            Vc = (V[safe] * valid[..., None]).to(self.device)
            bc = (beta[safe] * valid).to(self.device)
            S = delta_write_chunk(S, Kc, Vc, bc)
        self.S[shelf_t] = S

    def learn(self, texts: Sequence[str], values: Sequence[str], surprise_eps: float = 0.0) -> None:
        ids = torch.tensor([self.code(v) for v in values])
        self.write(self.enc.encode(texts, mark=True), ids, surprise_eps)

    def forget(self, texts: Sequence[str]) -> None:
        self.write(self.enc.encode(texts), None)

    # -------------------------------------------------------------- okuma
    @torch.no_grad()
    def recall(self, texts: Sequence[str], probes: int = 1):
        """Döndürür: tahmin edilen değer indeksleri, özellikler (en iyi benzerlik, fark) — güven için."""
        keys = self.enc.encode(texts, known_only=True)
        cand = self.shelf(keys, probes)
        best_idx = torch.full((len(texts),), -1, dtype=torch.long)
        best_top = torch.full((len(texts),), -1e9)
        best_margin = torch.zeros(len(texts))
        for p in range(cand.size(1)):
            v = self._read_shelves(keys, cand[:, p].numpy())
            sims = v @ self.codes.T                                            # [N, n_değer]
            top2 = sims.topk(min(2, sims.size(1)), dim=-1)
            top = top2.values[:, 0]
            margin = top - (top2.values[:, 1] if sims.size(1) > 1 else 0)
            better = top > best_top
            best_top = torch.where(better, top, best_top)
            best_margin = torch.where(better, margin, best_margin)
            best_idx = torch.where(better, top2.indices[:, 0], best_idx)
        feats = [best_top, best_margin]
        if hasattr(self.enc, "specificity"):
            feats.append(self.enc.specificity(texts))
        return best_idx, torch.stack(feats, dim=-1)

    def memory_mb(self) -> float:
        return self.S.numel() * self.S.element_size() / 2**20


class Calibrator:
    """(benzerlik, fark, özgüllük) → P(doğru). Küçük lojistik regresyon (Platt ölçekleme)."""

    def __init__(self) -> None:
        self.w = None

    def fit(self, feats: torch.Tensor, correct: torch.Tensor, steps: int = 500) -> "Calibrator":
        x = torch.cat((feats, torch.ones(len(feats), 1)), dim=1)
        w = torch.zeros(x.size(1), requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=steps)

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(x @ w, correct.float())
            loss.backward()
            return loss
        opt.step(closure)
        self.w = w.detach()
        return self

    def prob(self, feats: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(torch.cat((feats, torch.ones(len(feats), 1)), dim=1) @ self.w)
