"""Generate the README figures (docs/img/*.svg) from the measured numbers below.

    python docs/figures.py

Every number here comes from a run recorded in the README or in Heimdall/llamacpp/README.md; edit the tables at the top
and re-run to refresh the figures. The SVGs follow the viewer's light/dark setting (prefers-color-scheme).
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent / "img"

# --------------------------------------------------------------------------- measured numbers
# Qwen3.8-27B Q3_K_M, RX 9070 XT, Vulkan (RADV), llama-bench, tokens/s
SPEED_DECODE = {                       # context -> (original, Bifrost 15/1), both with f16 KV (tg64)
    "empty": (37.1, 33.7),
    "32K": (32.9, 33.7),
    "131K": (None, 32.9),
}
# Pass-key retrieval, Bifrost 12/4 memory v2, 2 keys x 3 depths per length (greedy, generate until the key or EOS)
PASSKEY = {                            # length -> (memory on, memory off), hits out of 6
    "8K": (4, 3),
    "32K": (6, 3),
    "64K": (4, None),
}
PPL = [("Original", 5.976), ("Bifrost 15/1", 6.175), ("Bifrost 12/4", 6.154), ("12/4 + KV q4_0", 6.154),
       ("12/4 memory v2", 6.150)]
# Cache at 128K tokens (GiB): growing KV of full-attention layers, constant Bifrost state
CACHE_128K = [("Original, 16 full-attention layers", 8.0, 0.0), ("Bifrost 12/4, KV f16", 2.0, 0.51),
              ("Bifrost 12/4, KV q4_0", 0.56, 0.51)]
# Retrieval attention, Qwen3.5-0.8B f32, CPU 4 threads, decode tokens/s
RETRIEVAL_SPEED = {"8K": (13.8, 14.1), "32K": (9.3, 14.1)}

# --------------------------------------------------------------------------- style
CSS = """
  .bg{fill:#ffffff}.fg{fill:#172033}.mu{fill:#5b6577}.ln{stroke:#d5dae3}.st{stroke:#172033}
  .dn{fill:#c9ced8}.win{fill:#c47a12}.mem{fill:#0f8a80}.keep{fill:#6b4bd6}.sink{fill:#b4384f}.orig{fill:#8a93a3}
  .twin{fill:#fbefdc}.tmem{fill:#dcf3f0}.tkeep{fill:#ebe5fb}.box{fill:#f6f7f9}
  .swin{stroke:#c47a12}.smem{stroke:#0f8a80}.skeep{stroke:#6b4bd6}
  text{font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif}
  .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
  @media (prefers-color-scheme:dark){
    .bg{fill:#0d1117}.fg{fill:#e6e9ef}.mu{fill:#9aa4b5}.ln{stroke:#2c3442}.st{stroke:#e6e9ef}
    .dn{fill:#3a414d}.win{fill:#e6a03c}.mem{fill:#3cc4b7}.keep{fill:#a08bff}.sink{fill:#ef6f86}.orig{fill:#6b7484}
    .twin{fill:#3a2b14}.tmem{fill:#12332f}.tkeep{fill:#2a2346}.box{fill:#161b22}
    .swin{stroke:#e6a03c}.smem{stroke:#3cc4b7}.skeep{stroke:#a08bff}
  }
"""


def svg(w, h, body, label):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{label}"><style>{CSS}</style>'
            f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="fg"/></marker></defs>'
            f'<rect width="{w}" height="{h}" rx="10" class="bg"/>{body}</svg>')


def t(x, y, s, cls="fg", size=13, anchor="start", weight=None, mono=False):
    w = f' font-weight="{weight}"' if weight else ""
    c = cls + (" mono" if mono else "")
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" class="{c}"{w}>{s}</text>'


def box(x, y, w, h, fill="box", stroke="st", rx=8, sw=1.2):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="{fill} {stroke}" stroke-width="{sw}"/>'


def arrow(d, cls="st", dash=False, sw=1.3):
    da = ' stroke-dasharray="5 4"' if dash else ""
    return f'<path d="{d}" fill="none" class="{cls}" stroke-width="{sw}"{da} marker-end="url(#a)"/>'


# --------------------------------------------------------------------------- figures
def fig_layers():
    b = [t(24, 34, "Qwen3.8-27B: 64 layers", size=15, weight="600"),
         t(24, 54, "every 4th layer is full attention; Bifrost converts 12 of the 16 and keeps 4", "mu", 12.5)]
    for i in range(64):
        r, c = divmod(i, 16)
        x, y = 70 + c * 46, 74 + r * 34
        if i % 4 == 3:
            cls = "keep" if i in (15, 31, 47, 63) else "mem"
            b.append(f'<rect x="{x}" y="{y}" width="42" height="28" rx="5" class="{cls}"/>')
            b.append(t(x + 21, y + 19, i, "bg", 11.5, "middle", "600"))
        else:
            b.append(f'<rect x="{x}" y="{y}" width="42" height="28" rx="5" class="dn"/>')
    for r in range(4):
        b.append(t(60, 93 + r * 34, f"{r * 16}", "mu", 11, "end", mono=True))
    lx, ly = 70, 228
    for cls, s in (("dn", "Gated DeltaNet, unchanged (48)"), ("mem", "converted to Bifrost (12)"),
                   ("keep", "kept full attention (4)")):
        b.append(f'<rect x="{lx}" y="{ly - 11}" width="14" height="14" rx="3" class="{cls}"/>')
        b.append(t(lx + 22, ly, s, "mu", 12.5))
        lx += 250
    return svg(820, 248, "".join(b), "64 layers; layers 15, 31, 47 and 63 stay full attention, the other 12 attention layers become Bifrost layers")


def fig_layer():
    b = [t(24, 32, "Inside a Bifrost layer", size=15, weight="600"),
         t(24, 52, "the teacher's own q, k, v; only ~8 small tensors are new", "mu", 12.5),
         box(24, 160, 100, 50), t(74, 182, "x", size=13, anchor="middle"), t(74, 199, "layer input", "mu", 11, "middle"),
         box(160, 145, 140, 80), t(230, 176, "q, k, v", size=13, anchor="middle"),
         t(230, 194, "pretrained, frozen", "mu", 11, "middle"),
         arrow("M124,185 L156,185"),
         box(350, 74, 230, 96, "twin", "swin", sw=1.6),
         t(465, 102, "Window attention", "win", 14, "middle", "600"),
         t(465, 124, "4 sinks + last 2048 tokens", size=12, anchor="middle"),
         t(465, 142, "exact softmax", size=12, anchor="middle"),
         t(465, 160, "o_win, lse_win", "mu", 11, "middle", mono=True),
         box(350, 230, 230, 100, "tmem", "smem", sw=1.6),
         t(465, 258, "Delta-rule memory", "mem", 14, "middle", "600"),
         t(465, 280, "one 256×256 state per KV head", size=12, anchor="middle"),
         t(465, 298, "read: q_m · S", size=12, anchor="middle"),
         t(465, 318, "o_mem, constant size", "mu", 11, "middle", mono=True),
         arrow("M300,170 C326,170 326,122 346,122"), arrow("M300,200 C326,200 326,280 346,280"),
         arrow("M440,170 L440,226", "smem", dash=True, sw=1.6),
         t(448, 196, "token t − 2048 leaves the", "mem", 11.5), t(448, 211, "window and is written", "mem", 11.5),
         f'<rect x="630" y="145" width="120" height="80" rx="40" class="box st" stroke-width="1.2"/>',
         t(690, 181, "mix γ", size=14, anchor="middle", weight="600"), t(690, 199, "per token", "mu", 11, "middle"),
         arrow("M580,122 C606,122 606,170 626,170", "swin"), arrow("M580,280 C606,280 606,200 626,200", "smem"),
         box(780, 160, 76, 50), t(818, 182, "o_proj", size=12.5, anchor="middle"), t(818, 199, "frozen", "mu", 11, "middle"),
         arrow("M750,185 L776,185"),
         t(24, 368, "o = o_win + γ·(mem_scale·o_mem − o_win),   γ = σ(W_mix·x + mix_lse·(lse_win − log W)), 0 before the window fills",
           "fg", 12.5, mono=True)]
    return svg(880, 390, "".join(b), "Bifrost layer: window attention and delta memory share q, k, v; tokens enter memory when they leave the window; gamma mixes the two")


def fig_timeline():
    b = [t(24, 32, "Where a token lives", size=15, weight="600"),
         '<rect x="24" y="70" width="26" height="40" class="sink"/>',
         '<rect x="50" y="70" width="560" height="40" class="tmem smem" stroke-width="1.2"/>',
         '<rect x="610" y="70" width="246" height="40" class="twin swin" stroke-width="1.2"/>',
         t(37, 62, "4 sinks", "sink", 11.5, "middle"),
         t(330, 95, "delta memory: compressed, constant size", "mem", 13, "middle"),
         t(733, 95, "window: last 2048, exact", "win", 13, "middle"),
         '<line x1="24" y1="132" x2="852" y2="132" class="st" stroke-width="1" marker-end="url(#a)"/>',
         t(24, 152, "position 0", "mu", 11), t(610, 152, "t − 2048", "mu", 11, "middle"), t(856, 152, "t (now)", "mu", 11, "end"),
         arrow("M620,64 C590,40 545,40 520,66", "smem", dash=True, sw=1.4),
         t(630, 52, "one token per step moves into memory", "mem", 11.5),
         t(24, 182, "Memory v2: the 4 KV heads forget at different rates (half-lives log-spaced from 2048 to 1M tokens);",
           "mu", 12), t(24, 199, "one archival head never forgets.", "mu", 12)]
    return svg(880, 215, "".join(b), "Token timeline: sinks, compressed memory, exact window")


def fig_cache():
    x0, scale = 300, 62.5           # 8 GiB = 500 px
    b = [t(24, 32, "Cache at 128K tokens, Qwen3.8-27B", size=15, weight="600"),
         t(24, 52, "by calculation: 4 KB per token per full-attention layer (f16); DeltaNet state is the same in all rows",
           "mu", 12)]
    for i, (name, kv, st) in enumerate(CACHE_128K):
        y = 80 + i * 52
        b.append(t(24, y + 18, name, size=12.5))
        b.append(f'<rect x="{x0}" y="{y}" width="{kv * scale:.1f}" height="26" class="keep"/>')
        if st:
            b.append(f'<rect x="{x0 + kv * scale:.1f}" y="{y}" width="{st * scale:.1f}" height="26" class="mem"/>')
        lab = f"{kv:g} GB" if not st else f"{kv:g} + {st:g} GB"
        b.append(t(x0 + (kv + st) * scale + 8, y + 18, lab, size=12.5, mono=True))
    y = 80 + 3 * 52
    b.append(f'<line x1="{x0}" y1="{y}" x2="{x0 + 500}" y2="{y}" class="ln" stroke-width="1"/>')
    for g in range(0, 9, 2):
        b.append(t(x0 + g * scale, y + 16, f"{g}", "mu", 11, "middle", mono=True))
    lx = 24
    for cls, s in (("keep", "growing KV (full-attention layers)"), ("mem", "Bifrost state (12 layers, constant)")):
        b.append(f'<rect x="{lx}" y="{y + 30}" width="14" height="14" rx="3" class="{cls}"/>')
        b.append(t(lx + 22, y + 41, s, "mu", 12.5))
        lx += 290
    return svg(880, y + 58, "".join(b), "Cache at 128K: original 8 GB; Bifrost 12/4 2 GB KV plus 0.5 GB state; with q4_0 KV 0.56 plus 0.5 GB")


def bars(title, sub, groups, series, vmax, unit, ticks, w=880, label=""):
    """Grouped vertical bars. groups: [(name, [v or None per series])], series: [(name, css class)]."""
    h, top, bot, left = 330, 78, 270, 60
    ph = bot - top
    b = [t(24, 32, title, size=15, weight="600"), t(24, 52, sub, "mu", 12)]
    for v in ticks:
        y = bot - v / vmax * ph
        b.append(f'<line x1="{left}" y1="{y:.1f}" x2="{w - 24}" y2="{y:.1f}" class="ln" stroke-width="1"/>')
        b.append(t(left - 8, y + 4, f"{v:g}", "mu", 11, "end", mono=True))
    gw = (w - 24 - left) / len(groups)
    n = len(series)
    bw = min(56, (gw - 40) / n)
    for gi, (gname, vals) in enumerate(groups):
        gx = left + gi * gw + (gw - n * bw - (n - 1) * 6) / 2
        for si, v in enumerate(vals):
            x = gx + si * (bw + 6)
            if v is None:
                b.append(t(x + bw / 2, bot - 8, "n/m", "mu", 10.5, "middle"))
                continue
            y = bot - v / vmax * ph
            b.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bot - y:.1f}" rx="3" class="{series[si][1]}"/>')
            b.append(t(x + bw / 2, y - 6, f"{v:g}" if float(v).is_integer() and vmax <= 10 else f"{v:.1f}", "fg", 11.5,
                       "middle", mono=True))
        b.append(t(left + gi * gw + gw / 2, bot + 20, gname, "fg", 12.5, "middle"))
    lx = left
    for name, cls in series:
        b.append(f'<rect x="{lx}" y="{h - 26}" width="14" height="14" rx="3" class="{cls}"/>')
        b.append(t(lx + 22, h - 15, name, "mu", 12.5))
        lx += 26 + 8 * len(name)
    b.append(t(w - 24, h - 15, unit, "mu", 11, "end"))
    return svg(w, h, "".join(b), label or title)


def fig_speed():
    groups = [(f"{k} context", list(v)) for k, v in SPEED_DECODE.items()]
    return bars("Qwen3.8-27B decode speed, one RX 9070 XT (Vulkan, Q3_K_M, f16 KV)",
                "llama-bench tg64, tokens/s. The original was not run at 131K (n/m = not measured)",
                groups, [("original", "orig"), ("Bifrost 15/1", "mem")], 40, "tokens / s", [0, 10, 20, 30, 40],
                w=700, label="Decode speed: original 37.1 then 32.9 at 32K; Bifrost 33.7, 33.7 and 32.9 at 131K")


def fig_passkey():
    groups = [(f"{k} tokens", [v[0], v[1]]) for k, v in PASSKEY.items()]
    return bars("Pass-key recall, Qwen3.8-27B Bifrost 12/4 (memory v2)",
                "hits out of 6 (2 keys × 3 depths); memory off = same file with BIFROST_MEM_OFF=1. n/m = not measured",
                groups, [("memory on", "mem"), ("memory off", "orig")], 6, "hits / 6", [0, 2, 4, 6],
                label="Pass-key hits out of 6: memory on 4, 6, 4 at 8K, 32K, 64K; memory off 3 and 3 at 8K and 32K")


def fig_engine():
    b = [t(24, 32, "Training engine (heimdall/motor.py): one layer and one sequence on the GPU at a time", size=15, weight="600"),
         box(24, 70, 150, 84), t(99, 102, "Disk", size=13, anchor="middle"),
         t(99, 122, "hidden states h_i", "mu", 11, "middle"), t(99, 138, "32 seq × 32K tokens", "mu", 11, "middle"),
         box(214, 70, 150, 84), t(289, 102, "Layer i", size=13, anchor="middle"),
         t(289, 122, "weights to the GPU", "mu", 11, "middle"), t(289, 138, "~0.7 GB bf16", "mu", 11, "middle"),
         box(404, 70, 176, 84, "tkeep", "skeep", sw=1.4), t(492, 98, "Teacher, once", "keep", 13, "middle", "600"),
         t(492, 118, "full-attention output y_t,", size=11, anchor="middle"),
         t(492, 134, "window output, fixed inputs", size=11, anchor="middle"),
         box(620, 70, 236, 84, "tmem", "smem", sw=1.4), t(738, 98, "300 training steps", "mem", 13, "middle", "600"),
         t(738, 118, "new parameters only, exact gradient", size=11, anchor="middle"),
         t(738, 134, "through time segments", size=11, anchor="middle"),
         arrow("M174,112 L210,112"), arrow("M364,112 L400,112"), arrow("M580,112 L616,112"),
         arrow("M738,154 L738,206 L99,206 L99,158"),
         t(418, 226, "forward pass with the trained layer → h_(i+1) written to disk → next layer", "mu", 12, "middle"),
         t(24, 262, "Loss on 12.5% random positions past the window, ×4 on question/answer spans of synthetic recall sequences.",
           "mu", 12),
         t(24, 280, "Sequence length chosen from the estimated per-layer GPU peak (27B on 16 GB: 32K). DEVAM=1 resumes.", "mu", 12)]
    return svg(880, 298, "".join(b), "Training engine loop: disk hidden states, layer weights, teacher once, train new parameters, write next hidden states")


def fig_retrieval():
    b = [t(24, 32, "Retrieval attention for the 4 kept layers (BIFROST_RET)", size=15, weight="600"),
         t(24, 52, "per 64-token block: min and max of every key dimension; read only the blocks whose upper bound is highest",
           "mu", 12),
         box(390, 70, 170, 40, rx=20), t(475, 95, "new query q", size=13, anchor="middle"),
         '<rect x="24" y="160" width="14" height="38" class="sink"/>']
    picked = {2, 7, 8, 14, 18}
    for k in range(20):
        x = 40 + k * 28
        cls = "keep" if k in picked else "box ln"
        b.append(f'<rect x="{x}" y="160" width="26" height="38" class="{cls}" stroke-width="1"/>')
    b.append('<rect x="604" y="160" width="252" height="38" class="twin swin" stroke-width="1.2"/>')
    b.append(t(730, 184, "window: last 2048 tokens", "win", 12.5, "middle"))
    for k in sorted(picked):
        cx = 40 + k * 28 + 13
        b.append(arrow(f"M475,110 C475,135 {cx},130 {cx},156", "skeep", sw=1.3))
    b.append(arrow("M520,110 C600,130 700,130 720,156", "swin", sw=1.3))
    b.append(t(600, 86, "top-k blocks by upper bound", "keep", 11.5))
    b.append(t(24, 226, "Upper bound = Σ_d max(q_d·min_d, q_d·max_d): no key in the block can score higher, so a relevant block is not", "mu", 12))
    b.append(t(24, 243, "skipped. Softmax over the picked tokens is exact. Reads 4 + 2048 + 64 + 16×64 ≈ 3.1K tokens at any context length.", "mu", 12))
    return svg(880, 262, "".join(b), "Retrieval attention: block summaries give an upper bound; the top blocks plus window and sinks are read")


def fig_retrieval_speed():
    groups = [(f"{k} context", list(v)) for k, v in RETRIEVAL_SPEED.items()]
    return bars("Retrieval attention: decode speed (Qwen3.5-0.8B, llama.cpp, CPU)",
                "full attention vs BIFROST_RET=16,512,64,4; the step stops slowing down with context",
                groups, [("full attention", "orig"), ("retrieval", "keep")], 16, "tokens / s", [0, 4, 8, 12, 16], w=640)


def main():
    OUT.mkdir(exist_ok=True)
    figs = {"layers": fig_layers, "bifrost_layer": fig_layer, "timeline": fig_timeline, "cache": fig_cache,
            "speed": fig_speed, "passkey": fig_passkey, "engine": fig_engine, "retrieval": fig_retrieval,
            "retrieval_speed": fig_retrieval_speed}
    for name, f in figs.items():
        (OUT / f"{name}.svg").write_text(f())
        print(f"docs/img/{name}.svg")


if __name__ == "__main__":
    main()
