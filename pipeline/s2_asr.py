"""Stage 2 - transcribe English audio with word timestamps and cut it into
dubbing units (short, sentence-like spans that fit in one breath)."""
from __future__ import annotations

import re
from pathlib import Path

import torch  # noqa: F401  -- import BEFORE ctranslate2/faster-whisper. On Windows
               # torch registers the CUDA runtime DLL directory, so ctranslate2 can
               # find cublas64_12.dll. Without it: "cublas64_12.dll is not found".

from .common import WORK, log, save_json

MAX_UNIT = 12.0     # never make a dubbing unit longer than this
MIN_UNIT = 1.8      # merge very short fragments with neighbours. Whisper splits
                    # mid-sentence ("on the other."); isolated ~1s translations
                    # are often impossible to speak clearly within their slot.
PAUSE = 0.45        # a silence this long ends a unit
SENT_END = re.compile(r"[.!?]['\"]?$")


def _words(seg) -> list:
    return [w for w in (seg.words or []) if w.word and w.end is not None]


def transcribe(meta: dict, model_size: str = "large-v3",
               compute_type: str = "float16", device: str = "cuda") -> dict:
    from faster_whisper import WhisperModel

    d = Path(meta["source"]).parent
    out = d / "segments.json"
    if out.exists():
        data = __import__("json").loads(out.read_text(encoding="utf-8"))
        if (data.get("model") == model_size and data.get("compute") == compute_type
                and data.get("segments")):
            log(f"[s2] cached: {len(data['segments'])} units")
            return data

    log(f"[s2] loading faster-whisper {model_size} ({device}/{compute_type})")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segs, info = model.transcribe(
        meta["asr_wav"], language="en", beam_size=5,
        word_timestamps=True, vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=250),
        condition_on_previous_text=False,
    )

    raw = []
    for s in segs:
        ws = _words(s)
        if ws:
            raw.append((ws[0].start, ws[-1].end, ws))
    log(f"[s2] {len(raw)} whisper segments -> grouping")

    # group word runs into sentence-ish units
    units = []
    cur: list = []

    def flush():
        if not cur:
            return
        a, b = cur[0][1], cur[-1][2]
        units.append({"start": round(a, 3), "end": round(b, 3),
                      "text": " ".join(w[0] for w in cur).strip()})
        cur.clear()

    for a, b, ws in raw:
        for w in ws:
            text = w.word.strip()
            if cur:
                gap = w.start - cur[-1][2]
                dur = w.end - cur[0][1]
                if gap >= PAUSE or dur > MAX_UNIT:
                    flush()
            cur.append((text, w.start, w.end))
            if SENT_END.search(text) or (w.end - cur[0][1]) >= MAX_UNIT:
                flush()
    flush()

    # merge too-short units into the previous one
    merged: list = []
    for u in units:
        if merged and (u["end"] - u["start"] < MIN_UNIT
                       or merged[-1]["end"] - merged[-1]["start"] < MIN_UNIT):
            merged[-1]["end"] = u["end"]
            merged[-1]["text"] = (merged[-1]["text"] + " " + u["text"]).strip()
        else:
            merged.append(dict(u))

    for i, u in enumerate(merged):
        u["id"] = i
        u["dur"] = round(u["end"] - u["start"], 3)
        u["next_start"] = round(merged[i + 1]["start"], 3) if i + 1 < len(merged) else None

    data = {"model": model_size, "compute": compute_type, "language": "en",
            "duration": meta["duration"], "segments": merged}
    save_json(out, data)
    log(f"[s2] {len(merged)} dubbing units "
        f"(median {sorted(u['dur'] for u in merged)[len(merged)//2]:.2f}s)")
    return data
