"""Is the head of each synthesized clip garbage / foreign reference leakage?"""
import json
from pathlib import Path

import torch  # noqa: F401
from faster_whisper import WhisperModel

d = Path("work/c__DMV_Clog")
print("ref-ish files:", [p.name for p in d.rglob("*ref*")] if list(d.rglob("*ref*")) else "none")
print("fit clips:", len(list((d / "fit").glob("*.wav"))))

m = WhisperModel("large-v3", device="cuda", compute_type="float16")
for uid in (0, 1, 2, 15):
    p = d / "fit" / f"unit_{uid:04d}.wav"
    if not p.exists():
        p = d / "tts" / f"unit_{uid:04d}.wav"
    if not p.exists():
        print(uid, "missing")
        continue
    # no language forced -> let whisper identify what is really in there
    segs, info = m.transcribe(str(p), word_timestamps=True)
    print(f"\n--- unit {uid}  lang={info.language} p={info.language_probability:.2f}")
    for s in segs:
        words = s.words or []
        for w in words[:6]:
            print(f"    {w.start:5.2f}-{w.end:5.2f}  {w.word.strip()!r}")
        if len(words) > 6:
            print(f"    ... +{len(words)-6} more words")
        break_info = s.end
