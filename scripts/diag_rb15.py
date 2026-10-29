import sys
import torch  # noqa: F401
from faster_whisper import WhisperModel

m = WhisperModel("large-v3", device="cuda", compute_type="float16")
out, _ = m.transcribe("work/c__DMV_Clog/fit/unit_0015.wav", language="kn")
print("heard:", " ".join(s.text for s in out).strip())
