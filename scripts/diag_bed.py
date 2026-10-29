"""Transcribe the no-vocals bed to see whether original speech survived separation."""
import sys
from pathlib import Path

from faster_whisper import WhisperModel

src = Path("work/c__DMV_Clog/stems/no_vocals.wav")
model = WhisperModel("small.en", device="cuda", compute_type="int8")
segs, info = model.transcribe(str(src), language="en", vad_filter=True,
                              vad_parameters={"min_silence_duration_ms": 500})
hits = [(round(s.start, 1), round(s.end, 1), s.text.strip()) for s in segs]
print("bed speech segments:", len(hits))
for h in hits[:40]:
    print(h)
