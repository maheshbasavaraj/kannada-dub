"""Smoke test: can IndicF5 clone the narrator's voice and speak Kannada?

  PYTHONPATH= .venv/Scripts/python.exe scripts/smoke_tts.py --id <video_id>

Cuts a reference clip out of the source audio, loads IndicF5, speaks a couple of
Kannada lines in the narrator's voice, then transcribes the result back with
faster-whisper so the output can be judged without listening to it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import s4_voice  # noqa: E402
from pipeline.common import SR, WORK, ffmpeg, log  # noqa: E402

LINES = [
    "ಒಂದಾನೊಂದು ಕಾಲದಲ್ಲಿ ಒಂದು ಸಣ್ಣ ಹಳ್ಳಿಯಲ್ಲಿ ವಾಸಿಸುತ್ತಿದ್ದ ವೃದ್ಧೆ ಇದ್ದಳು.",
    "ಅವಳು ಬಹಳ ದಯೆಯಿಂದ, ಬುದ್ಧಿವಂತಿಕೆಯಿಂದ ಮತ್ತು ಯಾವಾಗಲೂ ಸಂತೋಷದಿಂದ ಇದ್ದಳು.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--unit", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verify-model", default="small")
    a = ap.parse_args()

    d = WORK / a.id
    data = json.loads((d / "segments.json").read_text(encoding="utf-8"))
    u = data["segments"][a.unit]

    ref = d / "smoke_ref.wav"
    ffmpeg(["-ss", f"{u['start']:.3f}", "-t", f"{a.seconds:.3f}",
            "-i", str(d / "bed_44k.wav"), "-ac", "1", "-ar", str(SR),
            "-c:a", "pcm_s16le", str(ref)])
    log(f"[smoke] ref: unit {u['id']} ({a.seconds:.1f}s) '{u['text'][:70]}'")

    import soundfile as sf
    model = s4_voice._load_model(a.device)

    outdir = d / "smoke"
    outdir.mkdir(exist_ok=True)
    made = []
    for i, line in enumerate(LINES):
        try:
            audio = model(line, ref_audio_path=str(ref), ref_text=u["text"])
            arr = s4_voice._to_float(audio).flatten()
            p = outdir / f"line{i}.wav"
            sf.write(str(p), arr, SR)
            made.append((p, line))
            log(f"[smoke] line{i}: {arr.size / SR:.2f}s -> {p}")
        except Exception as e:  # noqa: BLE001
            log(f"[smoke] line{i} FAILED: {type(e).__name__}: {e}")
            return 1

    # hear it back: transcribe the generated Kannada
    try:
        from faster_whisper import WhisperModel
        m = WhisperModel(a.verify_model, device="cpu", compute_type="int8")
        for p, line in made:
            segs, _ = m.transcribe(str(p), language="kn", beam_size=5)
            heard = " ".join(s.text.strip() for s in segs)
            log(f"[smoke] {p.name} wanted : {line}")
            log(f"[smoke] {p.name} heard  : {heard}")
    except Exception as e:  # noqa: BLE001
        log(f"[smoke] playback check skipped ({e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
