"""Does the finished MP4 still contain original English speech (-> audible overlap)?"""
from pathlib import Path

from faster_whisper import WhisperModel

from pipeline.common import ffmpeg

mp4 = Path("out/c__DMV_Clog_kn.mp4")
raw = Path("work/c__DMV_Clog/final_audio.wav")
ffmpeg(["-y", "-i", str(mp4), "-vn", "-ac", "1", "-ar", "16000", str(raw)])

model = WhisperModel("small.en", device="cpu", compute_type="int8")
for start in (30, 90, 150, 210):
    chunk = Path(f"work/c__DMV_Clog/chunk_{start}.wav")
    ffmpeg(["-y", "-ss", str(start), "-t", "30", "-i", str(raw), str(chunk)])
    segs, _ = model.transcribe(str(chunk), language="en", vad_filter=True,
                               vad_parameters={"min_silence_duration_ms": 400})
    hits = [round(s.confidence, 2) if hasattr(s, "confidence") else 0 for s in []]
    texts = [s.text.strip() for s in model.transcribe(str(chunk), language="en", vad_filter=True)[0]]
    print(f"t={start}s  EN segments: {len(texts)}")
    for t in texts[:8]:
        print("   ", t)
