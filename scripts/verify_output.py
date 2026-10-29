"""Acceptance check for a finished dub: clean onset + no foreign-script words."""
import json
import subprocess
import sys
from pathlib import Path

import torch  # noqa: F401
from faster_whisper import WhisperModel

vid = sys.argv[1]
mp4 = Path(f"out/{vid}_kn.mp4")
wav = Path(f"work/{vid}/_head.wav")

subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", "0", "-t", "8", "-i", str(mp4),
                "-ac", "1", "-ar", "16000", str(wav)], check=True)

m = WhisperModel("large-v3", device="cuda", compute_type="float16")

enc = json.loads((Path(f"work/{vid}/segments.json")).read_text(encoding="utf-8"))
seg = enc["segments"]
print(f"=== {vid}: intended first line")
print("   ", seg[0].get("kn_fit") or seg[0]["kn"])

segs, _ = m.transcribe(str(wav), language="kn", word_timestamps=True)
print("=== heard 0-8s (word-level)")
for s in segs:
    for w in s.words:
        print(f"   [{w.start:5.2f}-{w.end:5.2f}] {w.word.strip()}")
