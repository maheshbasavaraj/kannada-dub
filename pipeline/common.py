"""Shared helpers for the Kannada dubbing pipeline."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
OUT = ROOT / "out"
REFS = ROOT / "refs"

SR = 24000          # narration / TTS sample rate
MIX_SR = 44100      # bed + final mix sample rate
GAP = 0.12          # seconds of breathing room kept before the next line starts


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def log(*a) -> None:
    print(*a, flush=True)


def run(cmd, **kw):
    cmd = [str(c) for c in cmd]
    log("+", " ".join(cmd))
    return subprocess.run(cmd, check=True, **kw)


def ffmpeg(args) -> None:
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args])


def ffprobe_duration(path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def ytdlp_bin() -> str:
    for cand in (shutil.which("yt-dlp"),
                 str(Path.home() / ".local" / "bin" / "yt-dlp.exe"),
                 str(Path.home() / ".local" / "bin" / "yt-dlp")):
        if cand and Path(cand).exists():
            return cand
    raise RuntimeError("yt-dlp not found - install with: uv tool install yt-dlp")


def load_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(path, obj) -> None:
    p = Path(path)
    ensure_dirs(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def load_mono(path, sr: int = SR):
    """Read any audio file as mono float32 at `sr`.

    Deliberately avoids librosa: actual librosa submodule access (e.g.
    librosa.effects.trim) pulls librosa/decompose.py -> sklearn.decomposition,
    and Smart App Control on this box blocks sklearn's compiled extensions.
    soundfile + soxr do the same job with no sklearn anywhere.
    """
    import numpy as np
    import soundfile as sf

    y, native = sf.read(str(path), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if int(native) != int(sr):
        try:
            import soxr
            y = soxr.resample(y, int(native), int(sr))
        except ImportError:  # pragma: no cover
            from math import gcd
            import scipy.signal as sp
            g = gcd(int(native), int(sr))
            y = sp.resample_poly(y, int(sr) // g, int(native) // g)
    return np.ascontiguousarray(y, dtype="float32"), int(sr)


def trim_silence(y, top_db: float = 30.0, sr: int = SR):
    """numpy replacement for librosa.effects.trim (RMS frames vs peak)."""
    import numpy as np

    frame = max(1, int(0.02 * sr))
    n = len(y) // frame
    if n < 2:
        return y
    rms = np.sqrt((y[: n * frame].reshape(n, frame) ** 2).mean(axis=1) + 1e-12)
    db = 20.0 * np.log10(rms / max(float(rms.max()), 1e-12))
    keep = np.nonzero(db > -abs(top_db))[0]
    if keep.size == 0:
        return y
    return y[keep[0] * frame: (keep[-1] + 1) * frame]


def srt_ts(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        s, ms = s + 1, 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path, items) -> None:
    """items: list of (start, end, text)"""
    lines = []
    for i, (a, b, txt) in enumerate(items, 1):
        lines += [str(i), f"{srt_ts(a)} --> {srt_ts(b)}", txt.strip(), ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
