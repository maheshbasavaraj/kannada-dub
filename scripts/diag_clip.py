"""Check whether a single TTS/fit clip contains more speech than its own line."""
import json
from pathlib import Path

from faster_whisper import WhisperModel

d = Path("work/c__DMV_Clog")
segs = {s["id"]: s for s in json.loads((d / "segments.json").read_text(encoding="utf-8"))["segments"]}
model = WhisperModel("large-v3", device="cuda", compute_type="int8")

for uid in [4, 8, 22, 37]:
    for tag, p in [("tts", d / "tts" / f"unit_{uid:04d}.wav"),
                   ("fit", d / "fit" / f"unit_{uid:04d}.wav")]:
        if not p.exists():
            print(uid, tag, "MISSING")
            continue
        out, _ = model.transcribe(str(p), language="kn", vad_filter=False)
        words = " ".join(s.text.strip() for s in out)
        print(f"--- unit {uid} [{tag}] expected: {segs[uid]['kn'][:70]}")
        print(f"    heard: {words[:160]}")
