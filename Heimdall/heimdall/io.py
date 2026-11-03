"""
Model paketleri: eğitim checkpoint'i (.pt) → servis paketi (config.json + model.safetensors + tokenizer.json).
"""

import json
import shutil
from pathlib import Path
from typing import Optional

import torch

from .config import HeimdallConfig
from .model import HeimdallLM

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER = REPO_ROOT / "04_TOKENIZERLAR" / "valerois_tokenizer_8k.json"


def save_bundle(model: HeimdallLM, out_dir, tokenizer: Optional[str] = None, meta: Optional[dict] = None,
                dtype: torch.dtype = torch.bfloat16) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(model.cfg.to_dict(), indent=2))
    (out / "meta.json").write_text(json.dumps(meta or {}, indent=2, default=str))
    model = model.to(dtype)
    try:
        from safetensors.torch import save_model
        save_model(model, str(out / "model.safetensors"))
    except ImportError:
        torch.save(model.state_dict(), out / "model.pt")
    if tokenizer and Path(tokenizer).exists():
        shutil.copy(tokenizer, out / "tokenizer.json")
    return out


def load_model(path, device="cpu", dtype: Optional[torch.dtype] = None) -> HeimdallLM:
    """Servis paketi klasörü ya da eğitim checkpoint'i (.pt) yükler."""
    p = Path(path)
    if p.suffix == ".gguf":
        from .gguf_cevir import load_heimdall_gguf
        model = load_heimdall_gguf(p)
    elif p.is_dir():
        model = HeimdallLM(HeimdallConfig(**json.loads((p / "config.json").read_text())))
        if (p / "model.safetensors").exists():
            from safetensors.torch import load_model as st_load
            st_load(model, str(p / "model.safetensors"))
        else:
            model.load_state_dict(torch.load(p / "model.pt", map_location="cpu"))
    else:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        model = HeimdallLM(HeimdallConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model"])
    model = model.to(device)
    if dtype is not None:
        model = model.to(dtype)
    return model.eval()


def tokenizer_path(model_path, override: Optional[str] = None) -> str:
    if override:
        return override
    p = Path(model_path)
    if p.suffix == ".gguf":
        from .gguf_cevir import gguf_tokenizer_file
        return gguf_tokenizer_file(p)
    if p.is_dir() and (p / "tokenizer.json").exists():
        return str(p / "tokenizer.json")
    return str(DEFAULT_TOKENIZER)
