"""
sentetik_veri.py — Hafıza ve akıl yürütmeyi öğreten sentetik kod verisi (.bin)
=============================================================================
Gerçek kodun (dolgu) arasına şu kalıpları serpiştirir; cevaplar hep sorgudan SONRA gelir,
böylece model sonraki tokeni tahmin ederken bağlamdan bulmak zorunda kalır:

  atama + uzak sorgu      proje_id = 4721 … [kod] … assert proje_id == 4721
  güncelleme (son değer)  x = 5 … x = 9 … assert x == 9
  zincir (çok adım)       a = 17 … b = a … c = b … assert c == 17
  sözlük                  ayar = {"port": 8080, …} … assert ayar["port"] == 8080
  fonksiyon               def f(): return 42 … assert f() == 42
  küçük aritmetik         t = 12 … t += 7 … assert t == 19

Atama–sorgu mesafeleri log-düzgün seçilir (0 … --max-mesafe); bir kısmı Huginn penceresini
aşar, bu yüzden bilgi Muninn hafızasından bulunmalıdır. İsimler rastgele üretilir:
model ezber değil bağlam içi bağlama öğrenir.

    python araclar/sentetik_veri.py --dolgu ../stream_coder_100k.bin --cikti sentetik_8k.bin --token 30e6
Eğitimde küçük payla karıştır:  --data ../stream_coder_100k.bin sentetik_8k.bin:0.05
"""

import argparse
import math
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
DEFAULT_TOKENIZER = HERE.parent.parent / "04_TOKENIZERLAR" / "valerois_tokenizer_8k.json"
SYLLABLES = ["ka", "ri", "mo", "ta", "le", "su", "na", "po", "zi", "de", "ve", "ro", "mi", "tu", "sa", "ko",
             "user", "data", "id", "max", "min", "count", "port", "path", "size", "rate", "key", "val", "tmp",
             "proje", "kayit", "sayi", "deger", "hedef", "kaynak", "ayar", "sonuc", "liste", "toplam"]


class Generator:
    def __init__(self, tok: Tokenizer, filler: np.ndarray, max_gap: int, rng: np.random.Generator) -> None:
        self.tok, self.filler, self.max_gap, self.rng = tok, filler, max_gap, rng
        self.eos = tok.token_to_id("<eos>")

    def enc(self, text: str):
        return self.tok.encode(text).ids

    def name(self) -> str:
        parts = self.rng.choice(SYLLABLES, size=self.rng.integers(1, 4))
        n = "_".join(parts)
        if self.rng.random() < 0.2:
            n = n.upper()
        return n + (str(self.rng.integers(0, 10)) if self.rng.random() < 0.2 else "")

    def value(self) -> str:
        r = self.rng.random()
        if r < 0.6:
            return str(int(self.rng.integers(0, 100_000)))
        if r < 0.85:
            return '"' + "".join(self.rng.choice(list("abcdefghijklmnoprstuvyz"), size=self.rng.integers(3, 9))) + '"'
        return f"{self.rng.random() * 100:.2f}"

    def gap(self):
        """Log-düzgün mesafe; dolgu gerçek koddan alınır."""
        n = int(math.exp(self.rng.uniform(0, math.log(self.max_gap + 1)))) - 1
        if n <= 0:
            return self.enc("\n")
        s = int(self.rng.integers(0, len(self.filler) - n - 1))
        return self.enc("\n") + self.filler[s:s + n].astype(np.int64).tolist() + self.enc("\n")

    # --------------------------------------------------------------- kalıplar: (kurulum, sorgu, cevap)
    def assign(self):
        n, v = self.name(), self.value()
        return [f"{n} = {v}\n"], f"assert {n} == ", v

    def overwrite(self):
        n = self.name()
        vals = [self.value() for _ in range(int(self.rng.integers(2, 4)))]
        return [f"{n} = {v}\n" for v in vals], f"assert {n} == ", vals[-1]

    def chain(self):
        names = [self.name() for _ in range(int(self.rng.integers(2, 5)))]
        v = self.value()
        setup = [f"{names[0]} = {v}\n"] + [f"{b} = {a}\n" for a, b in zip(names, names[1:])]
        return setup, f"assert {names[-1]} == ", v

    def dictionary(self):
        d, keys = self.name(), [self.name() for _ in range(int(self.rng.integers(2, 5)))]
        vals = [self.value() for _ in keys]
        body = ", ".join(f'"{k}": {v}' for k, v in zip(keys, vals))
        i = int(self.rng.integers(len(keys)))
        return [f"{d} = {{{body}}}\n"], f'assert {d}["{keys[i]}"] == ', vals[i]

    def function(self):
        f, v = self.name(), self.value()
        return [f"def {f}():\n    return {v}\n"], f"assert {f}() == ", v

    def arithmetic(self):
        n = self.name()
        x = int(self.rng.integers(0, 100))
        setup = [f"{n} = {x}\n"]
        for _ in range(int(self.rng.integers(1, 4))):
            d = int(self.rng.integers(1, 20))
            if self.rng.random() < 0.5:
                setup.append(f"{n} += {d}\n")
                x += d
            else:
                setup.append(f"{n} -= {d}\n")
                x -= d
        return setup, f"assert {n} == ", str(x)

    def document(self):
        kinds = [self.assign, self.overwrite, self.chain, self.dictionary, self.function, self.arithmetic]
        weights = np.array([0.25, 0.2, 0.15, 0.15, 0.1, 0.15])
        tasks = [kinds[i]() for i in self.rng.choice(len(kinds), size=int(self.rng.integers(1, 5)), p=weights)]
        ids = []
        for setup, _, _ in tasks:                     # kurulumlar, aralarında gerçek kod
            for line in setup:
                ids += self.gap() + self.enc(line)
        for i in self.rng.permutation(len(tasks)):    # sorgular, karışık sırayla
            _, query, answer = tasks[i]
            ids += self.gap() + self.enc(query) + self.enc(answer) + self.enc("\n")
        return ids + [self.eos]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dolgu", required=True, help="gerçek kod .bin (dolgu için; ör. stream_coder_100k.bin)")
    p.add_argument("--cikti", required=True)
    p.add_argument("--token", type=float, default=30e6)
    p.add_argument("--max-mesafe", type=int, default=4096)
    p.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    filler = np.memmap(args.dolgu, dtype=np.uint16, mode="r")
    gen = Generator(tok, filler, args.max_mesafe, np.random.default_rng(args.seed))
    target, chunks, total = int(args.token), [], 0
    while total < target:
        ids = np.asarray(gen.document(), dtype=np.uint16)
        chunks.append(ids)
        total += ids.size
        if len(chunks) % 20_000 == 0:
            print(f"  {total / 1e6:.1f}M / {target / 1e6:.0f}M token", flush=True)
    out = Path(args.cikti)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.concatenate(chunks).tofile(out)
    print(f"Yazıldı: {out} ({total / 1e6:.1f}M token, {len(chunks):,} belge)")


if __name__ == "__main__":
    main()
