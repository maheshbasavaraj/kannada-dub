"""Stage 1 - fetch source video (YouTube or local file) and split its audio."""
from __future__ import annotations

import re
from pathlib import Path

from .common import (WORK, ensure_dirs, ffmpeg, ffprobe_duration, load_json,
                     log, run, save_json, ytdlp_bin)


def video_id(url_or_path: str) -> str:
    s = str(url_or_path)
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", s)
    if m:
        return m.group(1)
    return re.sub(r"[^A-Za-z0-9_-]+", "_", Path(s).stem)[:60] or "clip"


def fetch(url_or_path: str, max_height: int = 1080) -> dict:
    vid = video_id(url_or_path)
    d = WORK / vid
    ensure_dirs(d)

    meta_path = d / "meta.json"
    src = d / "source.mp4"
    local = Path(url_or_path)
    is_local = local.exists() and local.is_file()

    if is_local and local.suffix.lower() != ".mp4":
        if not src.exists():
            ffmpeg(["-i", str(local), "-c:v", "copy", "-c:a", "aac", str(src)])
    elif is_local:
        src = local

    if not src.exists():
        run([ytdlp_bin(),
             "-f", f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/b[height<={max_height}]",
             "--merge-output-format", "mp4",
             "-o", str(d / "source.%(ext)s"),
             "--write-info-json", "--no-warnings", str(url_or_path)])
        if not src.exists():
            raise RuntimeError(f"download failed: no {src}")

    meta = load_json(meta_path, {})
    if not meta:
        info = next(iter(d.glob("source.info.json")), None)
        j = load_json(info, {}) if info else {}
        meta = {
            "id": vid,
            "title": j.get("title") or vid,
            "uploader": j.get("uploader") or "",
            "url": j.get("webpage_url") or ("" if is_local else url_or_path),
            "duration": round(ffprobe_duration(src), 3),
            "source": str(src),
        }
        save_json(meta_path, meta)

    a16 = d / "asr_16k.wav"
    a44 = d / "bed_44k.wav"
    if not a16.exists():
        ffmpeg(["-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                "-c:a", "pcm_s16le", str(a16)])
    if not a44.exists():
        ffmpeg(["-i", str(src), "-vn", "-ac", "2", "-ar", "44100",
                "-c:a", "pcm_s16le", str(a44)])

    meta["asr_wav"] = str(a16)
    meta["bed_wav"] = str(a44)
    save_json(meta_path, meta)
    log(f"[s1] {meta['title']}  {meta['duration']:.1f}s -> {d}")
    return meta
