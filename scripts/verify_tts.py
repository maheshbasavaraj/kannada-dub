"""End-to-end TTS verification: matched reference -> synthesis -> read-back.

The first clone attempt produced speech-shaped babble. Suspicion: the reference
clip was an 11s span while its ref_text was a long two-sentence line, so the
audio and its transcript disagree and F5 has nothing to align against.

This builds a tightly-trimmed ~4-6s reference from a single short unit (taken
from the music-free vocals stem), synthesises that unit's Kannada line, then
transcribes the result back so we can see whether it is real Kannada.

Run:  PYTHONPATH= TORCHDYNAMO_DISABLE=1 .venv/Scripts/python.exe \
        -u scripts/verify_tts.py --id 7nb3gdchiKA
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tts_src"))

import soundfile as sf  # noqa: E402

MIN_S, MAX_S = 3.5, 6.5


def pick_unit(segs: list) -> dict:
    """A short, clean, single-sentence unit makes the best clone reference."""
    best = None
    for u in segs:
        dur = u["dur"]
        if not (MIN_S <= dur <= MAX_S):
            continue
        if not u.get("kn_fit"):
            continue
        score = len(u["text"])
        if best is None or score < best[0]:
            best = (score, u)
    if best is None:
        raise SystemExit(f"no unit between {MIN_S}s and {MAX_S}s with a translation")
    return best[1]


def build_ref(stem: Path, u: dict, out: Path) -> Path:
    from pipeline.common import SR, load_mono, trim_silence
    y, sr = load_mono(stem, SR)
    a, b = int(u["start"] * sr), int((u["start"] + u["dur"]) * sr)
    clip = trim_silence(y[a:b], top_db=35, sr=sr)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), clip, sr, subtype="PCM_16")
    print(f"[verify] ref unit {u['id']} dur={u['dur']:.2f}s trimmed={len(clip)/sr:.2f}s -> {out.name}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()

    from pipeline import s4_voice
    wd = ROOT / "work" / a.id
    data = json.loads((wd / "segments.json").read_text(encoding="utf-8"))
    segs = data["segments"]
    stem = wd / "stems" / "vocals.wav"
    if not stem.exists():
        raise SystemExit(f"missing {stem} (run the ref stage first)")

    u = pick_unit(segs)
    ref = build_ref(stem, u, wd / "refs" / f"{a.id}_clone_ref.wav")
    print(f"[verify] ref text   : {u['text']}")
    print(f"[verify] target kn  : {u['kn_fit']}")

    model = s4_voice._load_model(a.device)
    dev = str(getattr(model, "device", a.device))

    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
    ref_path, ref_text = preprocess_ref_audio_text(str(ref), u["text"])

    t0 = time.time()
    audio, sr, _ = infer_process(ref_path, ref_text, u["kn_fit"],
                                 model.ema_model, model.vocoder,
                                 mel_spec_type="vocos", speed=a.speed,
                                 device=dev)
    print(f"[verify] generated {len(audio) / sr:.2f}s in {time.time() - t0:.0f}s")
    out = wd / "verify_clone.wav"
    sf.write(str(out), audio, sr, subtype="PCM_16")
    print(f"[verify] wrote {out}")

    from faster_whisper import WhisperModel
    m = WhisperModel("small", device="cpu", compute_type="int8")
    got, info = m.transcribe(str(out), language="kn", vad_filter=False, beam_size=5)
    heard = " ".join(s.text.strip() for s in got)
    print(f"[verify] TARGET : {u['kn_fit']}")
    print(f"[verify] HEARD  : {heard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
