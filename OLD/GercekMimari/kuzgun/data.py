"""
Veri akışı: uint16 .bin dosyaları (valerois_tokenizer_8k ile tokenize edilmiş, <eos> ile ayrılmış).

Kaynak yazımı:  yol.bin            (hiç ağırlık verilmezse pay = dosya boyutuyla orantılı)
                yol.bin:0.2        (göreli ağırlık; verilmeyenler 1.0 sayılır, hepsi normalize edilir)
Her dosyanın son `val_frac` kısmı validation'a ayrılır ve eğitimde hiç görülmez.
Arka plan thread'i batch'leri pinned belleğe hazırlar; GPU beklemez.
"""

import queue
import threading
from typing import List, Sequence, Tuple

import numpy as np
import torch


def parse_spec(spec: str):
    """'yol.bin:0.2' -> ('yol.bin', 0.2); 'C:\\veri\\x.bin' gibi Windows yolları da çalışır."""
    if ":" in spec:
        path, weight = spec.rsplit(":", 1)
        try:
            return path, float(weight)
        except ValueError:
            pass
    return spec, None


class Source:
    def __init__(self, spec: str, val_frac: float) -> None:
        self.path, weight = parse_spec(spec)
        data = np.memmap(self.path, dtype=np.uint16, mode="r")
        cut = int(len(data) * (1.0 - val_frac))
        self.train, self.val = data[:cut], data[cut:]
        self.explicit_weight = weight


class DataMixture:
    def __init__(self, specs: Sequence[str], val_frac: float = 0.005, seed: int = 0) -> None:
        self.sources = [Source(s, val_frac) for s in specs]
        if all(s.explicit_weight is None for s in self.sources):   # ağırlık yoksa: dosya boyutuyla orantılı
            w = np.array([len(s.train) for s in self.sources], dtype=np.float64)
        else:                                                      # ağırlık verilmeyen kaynak = 1.0
            w = np.array([1.0 if s.explicit_weight is None else s.explicit_weight for s in self.sources])
        self.probs = w / w.sum()
        self.rng = np.random.default_rng(seed)

    def describe(self) -> str:
        return ", ".join(f"{s.path} ({len(s.train) / 1e6:.1f}M tok, pay {p:.0%})" for s, p in zip(self.sources, self.probs))

    def sample(self, batch: int, seq_len: int) -> torch.Tensor:
        rows = np.empty((batch, seq_len + 1), dtype=np.int64)
        for i, src in enumerate(self.rng.choice(len(self.sources), size=batch, p=self.probs)):
            data = self.sources[src].train
            start = self.rng.integers(0, len(data) - seq_len - 1)
            rows[i] = data[start:start + seq_len + 1]
        return torch.from_numpy(rows)

    def val_batches(self, seq_len: int, n_per_source: int) -> List[Tuple[str, torch.Tensor]]:
        out = []
        for s in self.sources:
            if len(s.val) < seq_len + 2:
                continue
            starts = np.linspace(0, len(s.val) - seq_len - 2, n_per_source).astype(np.int64)
            out.append((s.path, torch.from_numpy(np.stack([s.val[a:a + seq_len + 1].astype(np.int64) for a in starts]))))
        return out


class Prefetcher:
    """Arka planda batch hazırlar (pinned) ve GPU'ya asenkron taşır."""

    def __init__(self, data: DataMixture, batch: int, device: torch.device, depth: int = 4) -> None:
        self.data, self.batch, self.device = data, batch, device
        self.seq_len = None
        self.q: "queue.Queue" = queue.Queue(maxsize=depth)
        self.lock = threading.Lock()
        self.stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self, seq_len: int) -> "Prefetcher":
        self.seq_len = seq_len
        self.thread.start()
        return self

    def set_seq_len(self, seq_len: int) -> None:
        with self.lock:
            if seq_len != self.seq_len:
                self.seq_len = seq_len
                while not self.q.empty():  # eski uzunluktaki batch'leri at
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        break

    def _run(self) -> None:
        pin = self.device.type == "cuda"
        while not self.stop:
            with self.lock:
                t = self.seq_len
            ids = self.data.sample(self.batch, t)
            if pin:
                ids = ids.pin_memory()
            self.q.put((t, ids))

    def next(self) -> Tuple[torch.Tensor, torch.Tensor]:
        while True:
            t, ids = self.q.get()
            if t == self.seq_len:
                break
        ids = ids.to(self.device, non_blocking=True)
        return ids[:, :-1], ids[:, 1:]
