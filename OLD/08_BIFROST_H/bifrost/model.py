"""
BifrostLM — düzen dizgesine (layout) göre kurulan dil modeli.

Blok:  x = x + CSL(norm(x))          (config.csl=True ise)
       x = x + Mixer(norm(x))        (M: Mímir, A: baseline dikkati, N: yok)
       x = x + SwiGLU(norm(x))

forward(ids)                      -> logits
forward_stateful(ids, state)      -> logits, state   (uzun diziyi parça parça işleme)
step(id, state)                   -> logits, state   (token token üretim)
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .config import ModelConfig
from .layers import BifrostCSL, CausalAttention, Kuzgun, Mimir, RMSNorm, SwiGLU


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, kind: str) -> None:
        super().__init__()
        self.kind = kind
        d = cfg.dim
        self.csl = BifrostCSL(d, cfg.csl_kernel, cfg.csl_dilation) if cfg.csl else None
        self.csl_norm = RMSNorm(d) if cfg.csl else None
        if kind == "M":
            self.mixer = Mimir(d, cfg.mimir_heads, cfg.mimir_dk, cfg.mimir_dv, cfg.mimir_conv,
                               cfg.mimir_chunk, cfg.negative_eigen)
        elif kind in ("K", "W", "R"):
            self.mixer = Kuzgun(d, cfg.kuzgun_heads, cfg.kuzgun_head_dim, cfg.window, cfg.kuzgun_conv,
                                cfg.mimir_chunk, {"K": "both", "W": "window", "R": "memory"}[kind],
                                cfg.negative_eigen, coupled_decay=cfg.coupled_decay,
                                archival_heads=cfg.archival_heads)
        elif kind == "A":
            self.mixer = CausalAttention(d, cfg.attn_heads, cfg.attn_kv_heads, window=cfg.attn_window)
        elif kind == "N":
            self.mixer = None
        else:
            raise ValueError(f"Bilinmeyen katman türü: {kind}")
        self.mixer_norm = RMSNorm(d) if self.mixer is not None else None
        hidden = int(round(cfg.ffn_mult * d / 32)) * 32
        self.ffn_norm = RMSNorm(d)
        self.ffn = SwiGLU(d, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.csl is not None:
            x = x + self.csl(self.csl_norm(x))
        if self.mixer is not None:
            x = x + self.mixer(self.mixer_norm(x))
        return x + self.ffn(self.ffn_norm(x))

    def init_state(self, batch: int, device, dtype) -> dict:
        return {
            "csl": self.csl.init_state(batch, device, dtype) if self.csl is not None else None,
            "mixer": self.mixer.init_state(batch, device, dtype) if self.mixer is not None else None,
        }

    def _run(self, x: torch.Tensor, state: dict, method: str) -> Tuple[torch.Tensor, dict]:
        new = {"csl": None, "mixer": None}
        if self.csl is not None:
            y, new["csl"] = getattr(self.csl, method)(self.csl_norm(x), state["csl"])
            x = x + y
        if self.mixer is not None:
            y, new["mixer"] = getattr(self.mixer, method)(self.mixer_norm(x), state["mixer"])
            x = x + y
        return x + self.ffn(self.ffn_norm(x)), new

    def forward_stateful(self, x, state):
        return self._run(x, state, "forward_stateful")

    def step(self, x, state):
        return self._run(x, state, "step")


class BifrostLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg, kind) for kind in cfg.layout])
        self.norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.reset_parameters()
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def reset_parameters(self) -> None:
        out_std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                is_out = name.endswith(("out_proj", "o_proj", "w3"))
                if is_out and self.cfg.zero_init_out:
                    nn.init.zeros_(module.weight)  # artık dal başta kimlik: derin ağda temiz gradyan akışı
                else:
                    nn.init.normal_(module.weight, std=out_std if is_out else 0.02)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _head(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(self.norm(x))
        cap = self.cfg.logit_softcap
        return cap * torch.tanh(logits / cap) if cap else logits

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self._head(x)

    def init_state(self, batch: int, device=None, dtype=None) -> List[dict]:
        device = device or self.embed.weight.device
        dtype = dtype or self.embed.weight.dtype
        return [b.init_state(batch, device, dtype) for b in self.blocks]

    def forward_stateful(self, ids: torch.Tensor, state: Optional[List[dict]] = None,
                         last_only: bool = False) -> Tuple[torch.Tensor, List[dict]]:
        state = state or self.init_state(ids.size(0))
        x = self.embed(ids)
        new_state = []
        for block, s in zip(self.blocks, state):
            x, s = block.forward_stateful(x, s)
            new_state.append(s)
        if last_only:
            x = x[:, -1:]
        return self._head(x), new_state

    def step(self, ids: torch.Tensor, state: List[dict]) -> Tuple[torch.Tensor, List[dict]]:
        x = self.embed(ids)
        new_state = []
        for block, s in zip(self.blocks, state):
            x, s = block.step(x, s)
            new_state.append(s)
        return self._head(x), new_state
