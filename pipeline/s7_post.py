"""Stage 7 - post-process the narration so the Kannada voice is actually clear.

Nothing before this stage does any real audio work: each TTS unit is peak
normalised on its own and the assembled track is peak normalised again. That
leaves the narration quiet, uneven, and full of low-level hiss - exactly the
"clarity drops here and there" that comes through on playback.

The chain, and why each part is in it:

  highpass    - drop sub-85Hz rumble; it carries no speech and muddies it
  afftdn      - gentle FFT denoise. TTS leaves low-level noise between words
                and the reference audio bleeds some of its own. Kept mild
                (nr=10) because aggressive denoise makes speech sound watery
  equalizer   - presence lift around 3kHz. This is the single biggest
                intelligibility win: Kannada consonant distinctions live here,
                and TTS output is usually dull in this band
  deesser     - tame the harsh sibilants TTS tends to overshoot
  acompressor - even out level differences between units, so no line jumps out
                or disappears under the next
"""
from __future__ import annotations

from pathlib import Path

from .common import log

# Tuned for speech, not music. Kept deliberately conservative: this has to help
# every line, not just the loud ones.
NARRATION_CHAIN = ",".join([
    "highpass=f=85",
    "afftdn=nr=10:nf=-30",
    "equalizer=f=3000:t=q:w=1.2:g=2.5",
    "deesser=i=0.35:m=0.5:f=0.5",
    "acompressor=threshold=-20dB:ratio=2.5:attack=6:release=120",
])

# YouTube/streaming target. Measured output was -20.7 LUFS before this stage,
# which plays back ~7dB quiet and reads as muffled.
TARGET_LUFS = -14.0
TRUE_PEAK = -1.5
LRA = 11.0

LOUDNORM = f"loudnorm=I={TARGET_LUFS}:TP={TRUE_PEAK}:LRA={LRA}"


def _two_pass_loudnorm(src: Path, dst: Path, sr) -> None:
    """EBU R128 in two passes.

    Single-pass loudnorm is a guess and overshoots/undershoots. Pass 1 measures
    the real loudness, pass 2 applies it with those measured values, which
    lands on target instead of near it.
    """
    import json as _json
    import re
    import subprocess

    from .common import ffmpeg

    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(src), "-af",
         f"loudnorm=I={TARGET_LUFS}:TP={TRUE_PEAK}:LRA={LRA}:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    if not m:
        # measurement failed - fall back to single pass rather than skip it
        log("[s7] loudnorm measurement failed; single-pass")
        ffmpeg(["-i", str(src), "-filter:a", LOUDNORM,
                "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)])
        return
    d = _json.loads(m.group(0))
    ffmpeg(["-i", str(src), "-filter:a",
            f"loudnorm=I={TARGET_LUFS}:TP={TRUE_PEAK}:LRA={LRA}"
            f":measured_I={d['input_i']}:measured_TP={d['input_tp']}"
            f":measured_LRA={d['input_lra']}:measured_thresh={d['input_thresh']}"
            f":offset={d.get('target_offset', 0)}:linear=true",
            "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)])


def post(meta: dict, fitted: dict, denoise: bool = True) -> dict:
    """Clean up the assembled narration track. Returns the new path."""
    from .common import SR, ensure_dirs, ffmpeg, ffprobe_duration

    src = Path(fitted["narration"])
    if not src.exists():
        raise FileNotFoundError(f"narration not found: {src}")
    dst = src.with_name("narration_post.wav")
    tmp = src.with_name("narration_clean.wav")
    ensure_dirs(src.parent)

    chain = NARRATION_CHAIN if denoise else ",".join([
        "highpass=f=85",
        "equalizer=f=3000:t=q:w=1.2:g=2.5",
        "acompressor=threshold=-20dB:ratio=2.5:attack=6:release=120",
    ])
    log(f"[s7] narration: denoise={denoise}, then two-pass loudnorm "
        f"to {TARGET_LUFS} LUFS")
    ffmpeg(["-i", str(src), "-filter:a", chain,
            "-ar", str(SR), "-c:a", "pcm_s16le", str(tmp)])
    _two_pass_loudnorm(tmp, dst, SR)

    dur = ffprobe_duration(dst)
    log(f"[s7] narration_post.wav {dur:.1f}s -> {dst}")
    fitted = dict(fitted)
    fitted["narration_raw"] = str(src)
    fitted["narration"] = str(dst)
    return fitted


def loudness_report(path) -> dict:
    """Measure integrated loudness / true peak / LRA of any file."""
    import re
    import subprocess

    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af",
         "loudnorm=print_format=summary", "-f", "null", "-"],
        capture_output=True, text=True)
    txt = r.stderr
    def grab(label):
        m = re.search(rf"{label}:\s*(-?[\d.]+)", txt)
        return float(m.group(1)) if m else None
    out = {"lufs": grab("Input Integrated"),
           "true_peak": grab("Input True Peak"),
           "lra": grab("Input LRA")}
    log(f"[s7] {Path(path).name}: {out['lufs']} LUFS, "
        f"peak {out['true_peak']} dBTP, LRA {out['lra']}")
    return out
