"""Read back a few fit clips and compare to the intended Kannada line.

Confirms (a) no doubled/reference speech leaking into a clip and (b) the words
are the intended ones, not a garbled condensation.
"""
import json
from pathlib import Path

import torch  # noqa: F401  -- loads the CUDA runtime DLLs ctranslate2 needs
from faster_whisper import WhisperModel

d = Path("work/c__DMV_Clog")
segs = {s["id"]: s for s in json.loads((d / "segments.json").read_text(encoding="utf-8"))["segments"]}

try:
    model = WhisperModel("small", device="cuda", compute_type="float16")
except Exception:
    model = WhisperModel("small", device="cpu", compute_type="int8")

for uid in (4, 5, 8, 20, 33, 51):
    p = d / "fit" / f"unit_{uid:04d}.wav"
    if not p.exists():
        print(f"unit {uid}: MISSING")
        continue
    out, _ = model.transcribe(str(p), language="kn")
    heard = " ".join(s.text for s in out).strip()
    print(f"--- unit {uid}")
    print(f"    want : {segs[uid].get('kn_fit') or segs[uid]['kn']}")
    print(f"    heard: {heard}")
