#!/usr/bin/env python3
"""
Valerois logosu üreticisi (yalnız standart kütüphane; font gerekmez).  python docs/brand/generate_logo.py

Fikir: SİMGE = V ve A tek bir zikzak çizgiyi paylaşır (V'nin sağ kolu = A'nın sol bacağı). YAZI = V ve çatı A ayrı harfler (zikzak
yazının içinde "VLEROIS" gibi okunuyordu); A ve I aynı koyu mor.
Renk: obsidyen (çok koyu mor-siyah) ve koyu mor; V tarafı koyu, A ve I mor. Koyu zeminde açık lavanta ve açık mor.
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent
LIGHT = dict(ink="#190f2c", pur="#5b2ab8", tile0="#f7f4fd", tile1="#e6def6")
DARK = dict(ink="#efe9fb", pur="#a07cff", tile0="#1d1233", tile1="#0c0614")
SW = 10.0          # çizgi kalınlığı (büyük harf yüksekliği 100)
GAP = 20.0         # harf arası

# VA zikzağı: (sol üst) V ucu -> V köşesi (alt) -> A tepesi (üst) -> A sağ ayağı (alt)
ZIG = [(5.5, 5.5), (32.0, 94.5), (58.5, 5.5), (85.0, 94.5)]
ZIG_W = 85.0 + 5.5 - 5.5 + 5.5        # sağ kenar ~ 90.5 (yuvarlak uç payı)
LETTERS = {  # VALEROIS: tek çizgili, yuvarlak uçlu, yükseklik 5.5..94.5 (A: çatı, çapraz çubuk yok)
    "V": ("M5.5 5.5 L32 94.5 L58.5 5.5", 64),
    "A": ("M5.5 94.5 L32 5.5 L58.5 94.5", 64),
    "L": ("M5.5 5.5 L5.5 94.5 L42 94.5", 48),
    "E": ("M44 5.5 L5.5 5.5 L5.5 94.5 L44 94.5 M5.5 50 L36 50", 50),
    "R": ("M5.5 94.5 L5.5 5.5 H31 A25.25 25.25 0 0 1 31 56 H5.5 M29 56 L51 94.5", 58),
    "O": ("M37 5.5 A31.5 44.5 0 1 1 36.99 5.5 Z", 74),
    "I": ("M5.5 5.5 L5.5 94.5", 11),
    "S": ("M48 24 C48 12 40 5.5 27 5.5 C14 5.5 6 13 6 25 C6 38 16 44 27 48 C38 52 48 58 48 72 C48 86 40 94.5 27 94.5 C13 94.5 6 87 6 76", 54),
}


def zig_path(pts) -> str:
    return "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)


def grad(gid: str, x0: float, x1: float, c: dict) -> str:
    """V yarısı koyu, A yarısı mor; geçiş ortadaki paylaşılan kolda."""
    return (f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" x1="{x0}" y1="0" x2="{x1}" y2="0">'
            f'<stop offset="0.36" stop-color="{c["ink"]}"/><stop offset="0.64" stop-color="{c["pur"]}"/></linearGradient>')


def stroke_attrs(sw=SW) -> str:
    return f'fill="none" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round"'


def wordmark(c: dict, gid="gw") -> tuple[str, float]:
    x = 0.0
    parts = []
    for ch in "VALEROIS":
        d, w = LETTERS[ch]
        parts.append(f'<path transform="translate({x:.1f} 0)" d="{d}" stroke="{c["pur"] if ch in "AI" else c["ink"]}"/>')
        x += w + GAP
    return f'<g {stroke_attrs()}>{"".join(parts)}</g>', x - GAP


def svg_wordmark(c: dict) -> str:
    body, total = wordmark(c)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-6 -6 {total + 12:.1f} 112" role="img" aria-label="Valerois">{body}</svg>\n'


def glyph_body(c: dict, gid: str, scale: float, ox: float, oy: float, sw: float) -> str:
    pts = [(x * scale + ox, y * scale + oy) for x, y in ZIG]
    return (f'<defs>{grad(gid, pts[0][0], pts[-1][0], c)}</defs>'
            f'<path d="{zig_path(pts)}" {stroke_attrs(sw)} stroke="url(#{gid})"/>')


def svg_glyph(c: dict) -> str:  # zeminsiz zikzak
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" role="img" aria-label="Valerois">'
            f'{glyph_body(c, "gg", 0.95, 2.5, 2.5, 10)}</svg>\n')


def tile_body(c: dict, gid: str) -> str:
    # 100x100 yuvarlak köşeli kare, içinde ortalanmış zikzak (zikzağın kutusu 79.5x89 → ölçek 0.70)
    sc = 0.70
    w = (ZIG[-1][0] - ZIG[0][0]) * sc; h = (ZIG[1][1] - ZIG[0][1]) * sc
    ox = (100 - w) / 2 - ZIG[0][0] * sc; oy = (100 - h) / 2 - ZIG[0][1] * sc
    return (f'<defs><linearGradient id="{gid}t" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{c["tile0"]}"/>'
            f'<stop offset="1" stop-color="{c["tile1"]}"/></linearGradient></defs>'
            f'<rect width="100" height="100" rx="24" fill="url(#{gid}t)"' + (' stroke="#a07cff" stroke-opacity=".30" stroke-width="1.2"' if c is DARK else '') + '/>' + glyph_body(c, gid, sc, ox, oy, 12))


def svg_mark(c: dict) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" role="img" aria-label="Valerois">{tile_body(c, "gm")}</svg>\n'


def svg_lockup(c: dict, tile_c: dict) -> str:
    body, total = wordmark(c, "gl")
    mk, gap = 100, 36
    inner = tile_body(tile_c, "gk")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {mk + gap + total + 6:.1f} 100" role="img" aria-label="Valerois">'
            f'{inner}<g transform="translate({mk + gap} 0)">{body}</g></svg>\n')


def main() -> None:
    files = {
        "valerois-mark.svg": svg_mark(LIGHT),                    # açık yüzeyli simge (açık zemine)
        "valerois-mark-dark.svg": svg_mark(DARK),                # koyu yüzeyli simge
        "valerois-glyph-on-light.svg": svg_glyph(LIGHT),
        "valerois-glyph-on-dark.svg": svg_glyph(DARK),
        "valerois-wordmark-on-light.svg": svg_wordmark(LIGHT),
        "valerois-wordmark-on-dark.svg": svg_wordmark(DARK),
        "valerois-lockup-on-light.svg": svg_lockup(LIGHT, DARK),  # koyu simge + koyu yazı, açık zemin
        "valerois-lockup-on-dark.svg": svg_lockup(DARK, DARK),
    }
    for name, s in files.items():
        (OUT / name).write_text(s)
    print("yazıldı:", ", ".join(files))


if __name__ == "__main__":
    main()
