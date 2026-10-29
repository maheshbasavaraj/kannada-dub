"""What is actually audible at 0-8s of the output, and which units are placed there."""
import json
import subprocess
from pathlib import Path

import torch  # noqa: F401
from faster_whisper import WhisperModel

d = Path("work/c__DMV_Clog")
mp4 = Path("out/c__DMV_Clog_kn.mp4")
tmp = d / "_win.wav"
subprocess.run(["ffmpeg", "-y", "-ss", "0", "-t", "8", "-i", str(mp4),
                "-ac", "1", "-ar", "16000", str(tmp)], check=True, capture_output=True)


def scripts(s):
    out = {}
    for ch in s:
        if ch.isspace() or ch.isdigit():
            continue
        o = ord(ch)
        if 0x0C80 <= o <= 0x0CFF:
            k = "kannada"
        elif o < 128:
            k = "latin" if ch.isalpha() else "punct"
        elif 0x0900 <= o <= 0x0C7F:
            k = "devanagari/telugu"
        elif 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF:
            k = "hangul"
        elif 0x3040 <= o <= 0x30FF:
            k = "kana"
        elif 0x4E00 <= o <= 0x9FFF:
            k = "han"
        else:
            k = f"U+{o:04X}"
        out[k] = out.get(k, 0) + 1
    return out


m = WhisperModel("large-v3", device="cuda", compute_type="float16")
segs, _ = m.transcribe(str(tmp), language="kn")
print("=== heard 0-8s ===")
for s in segs:
    if s.end > 8:
        break
    print(f"[{s.start:6.2f}-{s.end:6.2f}] {s.text.strip()}")
    print(f"          scripts: {scripts(s.text)}")

rep = json.loads((d / "fit_report.json").read_text(encoding="utf-8"))
segsrc = {x["id"]: x for x in json.loads((d / "segments.json").read_text(encoding="utf-8"))["segments"]}
print("\n=== placed before 8s ===")
if isinstance(rep, dict):
    for k, v in rep.items():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "start" in v[0]:
            for u in v:
                if float(u.get("start", 9e9)) < 8:
                    uid = u.get("id", u.get("unit"))
                    t = segsrc.get(uid, {}).get("kn_fit", "")
                    print(f"  unit {uid} start={u['start']:.2f} dur={u.get('final', u.get('dur'))} :: {t}")
