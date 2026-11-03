"""
Token veri akışı: uint16 .bin dosyalarından rastgele pencereler.

Kaynak yazımı:  "yol"  ya da  "yol@başlangıç:bitiş"  (oran olarak, ör. stream.bin@0:0.99)
Böylece aynı dosyanın son %1'i validation olarak ayrılabilir.
"""

from typing import List, Sequence, Tuple

import numpy as np
import torch


class Source:
    def __init__(self, spec: str) -> None:
        path, _, frac = spec.partition("@")
        data = np.memmap(path, dtype=np.uint16, mode="r")
        lo, hi = (float(x) for x in frac.split(":")) if frac else (0.0, 1.0)
        self.data = data[int(lo * len(data)): int(hi * len(data))]
        self.name = spec

    def __len__(self) -> int:
        return len(self.data)


class TokenData:
    def __init__(self, train: Sequence[str], val: Sequence[str], seed: int = 0) -> None:
        self.train = [Source(s) for s in train]
        self.val = [Source(s) for s in val]
        sizes = np.array([len(s) for s in self.train], dtype=np.float64)
        self.weights = sizes / sizes.sum()
        self.rng = np.random.default_rng(seed)

    @property
    def train_tokens(self) -> int:
        return sum(len(s) for s in self.train)

    def batch(self, batch: int, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rows = []
        for src_idx in self.rng.choice(len(self.train), size=batch, p=self.weights):
            data = self.train[src_idx].data
            start = self.rng.integers(0, len(data) - length - 1)
            rows.append(data[start:start + length + 1].astype(np.int64))
        ids = torch.from_numpy(np.stack(rows))
        return ids[:, :-1], ids[:, 1:]

    def val_windows(self, length: int, per_source: int, seed: int = 1234) -> List[Tuple[str, torch.Tensor]]:
        """Sabit (deterministik) validation pencereleri: her kaynaktan eşit aralıklı `per_source` adet."""
        out = []
        for src in self.val:
            starts = np.linspace(0, len(src) - length - 2, per_source).astype(np.int64)
            ids = torch.from_numpy(np.stack([src.data[s:s + length + 1].astype(np.int64) for s in starts]))
            out.append((src.name, ids))
        return out
