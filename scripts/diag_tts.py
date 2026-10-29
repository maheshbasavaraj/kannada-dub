"""Diagnose IndicF5 generation: is the raw vocoder output speech, or is the
pydub post-processing inside the repo's model.py destroying it?

Runs the model's own inference path directly (infer_process) and reports the
statistics of the waveform *before* the silence-strip / loudness-normalise step
that model.forward() applies.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import s4_voice  # noqa: E402
from pipeline.common import SR, WORK, ffmpeg, log  # noqa: E402

LINE = "ಒಂದಾನೊಂದು ಕಾಲದಲ್ಲಿ ಒಂದು ಸಣ್ಣ ಹಳ್ಳಿಯಲ್ಲಿ ವಾಸಿಸುತ್ತಿದ್ದ ವೃದ್ಧೆ ಇದ್ದಳು."


def stats(tag: str, a) -> None:
    import numpy as np
    a = np.asarray(a)
    print(f"{tag}: dtype={a.dtype} shape={a.shape} min={a.min()} max={a.max()} "
          f"mean={a.mean():.5f} std={a.std():.5f} "
          f"uniq={len(np.unique(np.round(a.astype('float64'), 6)))}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--unit", type=int, default=0)
    ap.add_argument("--ref", default=None, help="reference wav (default: smoke_ref)")
    ap.add_argument("--ref-text", default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    a = ap.parse_args()

    import numpy as np
    import soundfile as sf

    d = WORK / a.id
    data = json.loads((d / "segments.json").read_text(encoding="utf-8"))
    u = data["segments"][a.unit]
    ref = Path(a.ref) if a.ref else d / "smoke_ref.wav"
    ref_text = a.ref_text or u["text"]

    model = s4_voice._load_model("cuda")
    log(f"[diag] ref={ref} ref_text={ref_text[:60]!r}")

    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
    ref_path, ref_text_pp = preprocess_ref_audio_text(str(ref), ref_text)
    log(f"[diag] ref preprocessed -> {ref_path}")
    log(f"[diag] ref text -> {ref_text_pp[:80]!r}")

    audio, sr, _ = infer_process(
        ref_path, ref_text_pp, LINE, model.ema_model, model.vocoder,
        mel_spec_type="vocos", speed=a.speed, device=model.device,
    )
    stats("[diag] raw infer_process", audio)

    out = d / "diag_raw.wav"
    arr = np.asarray(audio, dtype=np.float32).flatten()
    sf.write(str(out), arr, sr)
    log(f"[diag] raw written {out} ({arr.size / sr:.2f}s @ {sr})")

    # and what model.forward() returns after its pydub pass
    try:
        fwd = model(LINE, ref_audio_path=str(ref), ref_text=ref_text)
        stats("[diag] model.forward()", fwd)
        fo = d / "diag_forward.wav"
        farr = s4_voice._to_float(fwd).flatten()
        sf.write(str(fo), farr, SR)
        log(f"[diag] forward written {fo} ({farr.size / SR:.2f}s)")
    except Exception as e:  # noqa: BLE001
        log(f"[diag] forward failed: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
