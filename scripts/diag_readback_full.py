"""Read the finished narration back with large-v3 and compare to intended lines."""
import json
from pathlib import Path

import torch  # noqa: F401
from faster_whisper import WhisperModel

d = Path("work/c__DMV_Clog")
segs = json.loads((d / "segments.json").read_text(encoding="utf-8"))["segments"]

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
out, info = model.transcribe(str(d / "narration_post.wav"), language="kn")
print(f"detected lang={info.language} prob={info.language_probability:.2f} dur={info.duration:.1f}s")
for s in out:
    print(f"{s.start:7.2f}-{s.end:7.2f}  {s.text.strip()}")
