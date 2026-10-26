"""
Hugging Face yüklemesi için klasör hazırlar (YÜKLEME YAPMAZ): safetensors paketi + model kartı + ağırlık lisansı.

    python bench/hf_hazirla.py --ckpt runs/hafta/model/last.pt --lmeval runs/hafta/lmeval.json --out models/heimdall-bit-337m

Karar (kullanıcı): ağırlıklar açık kaynak (Apache-2.0), model KODU (mimari uygulaması) mimari doğrulanana kadar kapalı kalır
(depodaki LICENSE: yalnız inceleme). Bu yüzden kart, ağırlıkların kodsuz çalıştırılamayacağını açıkça söyler.

Dikkat: eğitim verisinin ~%25'i github-code-clean (karışık, copyleft dahil lisanslar). Ağırlıkları geniş lisansla yayınlamadan önce
bunun hukuki değerlendirmesini kendin yap; script kartta bunu açıkça belirtir.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heimdall.io import DEFAULT_TOKENIZER, load_model, save_bundle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REFS = {  # README bölüm 8 ile aynı: aynı harness, aynı makine, 0-shot
    "GPT-2 124M": {"arc_easy": 43.8, "arc_challenge": 22.7, "hellaswag": 31.1, "piqa": 62.9, "winogrande": 51.6, "wikitext": 37.4},
    "Pythia-160M": {"arc_easy": 43.2, "arc_challenge": 23.1, "hellaswag": 30.2, "piqa": 62.1, "winogrande": 52.6, "wikitext": 33.5},
    "Pythia-410M": {"arc_easy": 51.9, "arc_challenge": 24.2, "hellaswag": 40.5, "piqa": 66.7, "winogrande": 53.8, "wikitext": 20.1},
    "SmolLM2-135M": {"arc_easy": 64.3, "arc_challenge": 29.7, "hellaswag": 43.2, "piqa": 68.3, "winogrande": 53.4, "wikitext": 23.1},
}
ROWS = [("ARC-easy (acc)", "arc_easy", "acc"), ("ARC-challenge (acc_norm)", "arc_challenge", "acc_norm"),
        ("HellaSwag (acc_norm)", "hellaswag", "acc_norm"), ("PIQA (acc)", "piqa", "acc"), ("Winogrande (acc)", "winogrande", "acc"),
        ("WikiText word-perplexity ↓", "wikitext", "word_perplexity")]


def metric(res: dict, task: str, name: str):
    r = res.get(task, {})
    for k, v in r.items():
        if k.split(",")[0] == name:
            return v
    return None


def results_table(lmeval: dict | None) -> str:
    head = "| Benchmark (0-shot) | **Heimdall-Bit 337M** | " + " | ".join(REFS) + " |\n|---|---|" + "---|" * len(REFS) + "\n"
    out = head
    for label, task, m in ROWS:
        v = metric((lmeval or {}).get("results", {}), task, m)
        mine = "n/a" if v is None else (f"{v:.1f}" if task == "wikitext" else f"{100 * v:.1f}")
        out += f"| {label} | **{mine}** | " + " | ".join(f"{REFS[r][task]:.1f}" for r in REFS) + " |\n"
    out += "| Training tokens | **3.7 B** | WebText | ~300 B | ~300 B | ~2 T |\n"
    return out


def model_card(meta: dict, table: str, repo_hint: str) -> str:
    return f'''---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- base-model
- research
- hybrid
- linear-attention
- 1-bit
---

# Heimdall-Bit 337M (base model, research checkpoint)

A {meta.get("params", 0) / 1e6:.0f}M-parameter language model trained from scratch on **one 16 GB consumer GPU (AMD RX 9070 XT)** in ~4 days on
3.7 B tokens. It is a **hybrid** of constant-memory recurrent layers and a few global-attention layers, and its feed-forward weights are **1-bit**
(±1). It is a **base model**: no instruction tuning, no safety tuning. Expect fluent but often factually wrong text and repetition under greedy decoding.

## Important: these weights need the Heimdall model code
The weights are released under **Apache-2.0**. The *model implementation* (needed to run them) lives in
[{repo_hint}](https://github.com/{repo_hint}) under a **review-only license** until the architecture has been validated; it will be opened afterwards.
There is no `transformers` class and no GGUF export (the architecture is not supported by llama.cpp). Files: `model.safetensors` (bf16; the 1-bit
FFN weights are stored as ±1 in bf16, not packed), `config.json`, `tokenizer.json` (8 192-token BPE), `meta.json`.

```python
# with the Heimdall code checked out:
from heimdall.io import load_model
model = load_model("path/to/this/folder")
```

## Evaluation (zero-shot, lm-evaluation-harness, same machine, single run, standard error ~±1 point)
{table}
Heimdall saw ~80× fewer tokens than Pythia but has 2× the parameters. It is on par with GPT-2 / Pythia-160M on ARC and HellaSwag, behind them on PIQA and
WikiText, and well behind Pythia-410M and SmolLM2-135M. We did not train a dense baseline, so the effect of the 1-bit FFN and of the hybrid layout is **not isolated**.

## Training data
FineWeb-Edu (~55 %), Cosmopedia (~21 %), GitHub code from `codeparrot/github-code-clean` (~25 %). **The GitHub code has mixed licenses (including copyleft);
you are responsible for assessing whether using weights trained on it is acceptable for your purpose.**

## Limitations
Base model; hallucinates; loops under greedy decoding; trained at 2 048 tokens (retrieval works up to ~4K, degrades at 8K in our tests); English and code only;
single training run and seed.
'''


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lmeval", help="bench/lmeval_heimdall.py --out çıktısı (model kartındaki tablo için)")
    p.add_argument("--repo", default="valeroisai/bifrost", help="kod deposu (kart bağlantısı)")
    p.add_argument("--hf-repo", default="valeroisai/heimdall-bit-337m")
    a = p.parse_args()
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = load_model(a.ckpt)
    meta = {"step": ckpt.get("step"), "tokens": ckpt.get("tokens"), "arch": ckpt.get("args", {}).get("arch"),
            "params": model.num_params(), "license_weights": "apache-2.0"}
    out = save_bundle(model, a.out, str(DEFAULT_TOKENIZER), meta, torch.bfloat16)
    lm = json.loads(Path(a.lmeval).read_text()) if a.lmeval else None
    (out / "README.md").write_text(model_card(meta, results_table(lm), a.repo))
    shutil.copy(ROOT / "docs" / "hf" / "LICENSE-APACHE-2.0", out / "LICENSE")
    size = sum(f.stat().st_size for f in out.iterdir()) / 2 ** 20
    print(f"Hazır: {out} ({size:.0f} MB, {meta['params'] / 1e6:.0f}M param)")
    print("Kontrol: model kartını oku (README.md), sonra giriş yapıp yükle:")
    print(f"  hf auth login   (eski sürümlerde: huggingface-cli login)")
    print(f"  hf upload {a.hf_repo} {out} .")


if __name__ == "__main__":
    main()
