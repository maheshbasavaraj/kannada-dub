"""Scan the Kannada translation for characters outside Kannada/Latin/punct."""
import json
import unicodedata
from pathlib import Path

d = Path("work/c__DMV_Clog")

def script_of(ch):
    o = ord(ch)
    if 0x0C80 <= o <= 0x0CFF:
        return "kannada"
    if o < 0x80:
        return "ascii"
    if 0x0900 <= o <= 0x097F:
        return "devanagari"
    if 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF:
        return "hangul"
    if 0x3040 <= o <= 0x30FF:
        return "kana"
    if 0x4E00 <= o <= 0x9FFF:
        return "han"
    if 0x0B80 <= o <= 0x0BFF:
        return "tamil"
    if 0x0C00 <= o <= 0x0C7F:
        return "telugu"
    if 0x0600 <= o <= 0x06FF:
        return "arabic"
    if 0x0400 <= o <= 0x04FF:
        return "cyrillic"
    return "other:" + unicodedata.name(ch, "?")

for fn in ("segments.json", "units.json"):
    p = d / fn
    if not p.exists():
        continue
    data = json.loads(p.read_text(encoding="utf-8"))
    segs = data["segments"] if isinstance(data, dict) else data
    print(f"=== {fn}: {len(segs)} units")
    for s in segs:
        for key in ("kn", "kn_fit"):
            txt = s.get(key)
            if not txt:
                continue
            bad = sorted({ch for ch in txt if script_of(ch) not in ("kannada", "ascii")})
            if bad:
                print(f"  unit {s['id']} [{key}]: {bad}")
                print(f"      {txt}")