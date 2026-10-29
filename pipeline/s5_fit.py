"""Stage 5 - time-fit each Kannada unit into the original speech slot and lay
them all onto one narration timeline that starts where the English did.

Two properties this stage must hold, both of which were violated before:

1. **A real pause between lines.** `GAP` is subtracted when computing each
   unit's tempo target, so placement must also advance by `GAP`. Placing with
   `cursor = start + final` butt-joins every clip and erases the pause, which
   is heard as one line running straight into the next.

2. **The narration must fit the video.** The per-unit tempo cap alone cannot
   guarantee this: when many units are over-cap the tail drifts past the end
   and `-shortest` cuts the last lines off entirely. After the per-unit pass we
   measure the real tail and, if it overshoots, apply one extra uniform tempo to
   every line so the whole track lands inside the runtime.
"""
from __future__ import annotations

from pathlib import Path

from .common import GAP, SR, ensure_dirs, ffmpeg, ffprobe_duration, log, save_json

MAX_TEMPO = 1.40        # fastest a single line will ever be squeezed
MIN_TEMPO = 0.90        # slowest we will stretch (beyond this it sounds drunk)
MAX_TOTAL_TEMPO = 1.85  # absolute ceiling when a whole-track squeeze is needed
FIT_TOLERANCE = 0.25    # overshoot tolerated before forcing a global squeeze


def _tempo(src: Path, factor: float, dst: Path) -> None:
    if abs(factor - 1.0) < 0.01:
        dst.write_bytes(src.read_bytes())
        return
    ffmpeg(["-i", str(src), "-filter:a", f"rubberband=tempo={factor:.4f}",
            "-ar", str(SR), "-c:a", "pcm_s16le", str(dst)])


def fit(meta: dict, data: dict, schedule: str = "flow") -> dict:
    import numpy as np
    import soundfile as sf

    d = Path(meta["source"]).parent
    tts = d / "tts"
    fitdir = d / "fit"
    ensure_dirs(fitdir)

    segs = data["segments"]
    total = float(meta["duration"])

    def render(factor_fn) -> list:
        out = []
        for u in segs:
            src = tts / f"unit_{u['id']:04d}.wav"
            if not src.exists():
                continue
            natural = ffprobe_duration(src)
            nxt = u.get("next_start")
            avail = ((nxt - u["start"]) if nxt else (total - u["start"])) - GAP
            avail = max(avail, 0.4)
            base = natural / avail
            used = factor_fn(natural, avail, base)
            dst = fitdir / f"unit_{u['id']:04d}.wav"
            _tempo(src, used, dst)
            real = ffprobe_duration(dst)
            out.append({"id": u["id"], "start": u["start"],
                        "natural": round(natural, 3), "avail": round(avail, 3),
                        "tempo": round(used, 3), "final": round(real, 3),
                        "overflow": round(max(0.0, real - avail), 3),
                        "tight": base > MAX_TEMPO + 0.02})
        return out

    def place(report: list) -> tuple:
        """Flow (or strict) placement, always leaving GAP after each line."""
        cursor, overlaps, shifted, max_shift = 0.0, 0, 0, 0.0
        for r in report:
            start = r["start"]
            if start < cursor:
                overlaps += 1
                if schedule == "flow":
                    start = cursor
            r["placed"] = round(start, 3)
            r["shift"] = round(start - r["start"], 3)
            if r["shift"] > 0.001:
                shifted += 1
                max_shift = max(max_shift, r["shift"])
            cursor = start + r["final"] + GAP
        return overlaps, shifted, round(max_shift, 3)

    def tail_of(report: list) -> float:
        return max([r["placed"] + r["final"] for r in report] or [0.0])

    # ---- pass 1: per-unit fit, capped so no single line is mangled
    report = render(lambda n, a, b: min(max(b, MIN_TEMPO), MAX_TEMPO))
    overlaps, shifted, max_shift = place(report)
    tail = tail_of(report)
    squeeze = 1.0

    # ---- pass 2: if the whole track still overshoots, spread one extra tempo
    # evenly over every line. Only the part of the overshoot that the per-unit
    # cap could not absorb is taken here, and MAX_TOTAL_TEMPO stops runaway.
    if tail > total + FIT_TOLERANCE:
        squeeze = tail / total
        log(f"[s5] track overshoots by {tail - total:.2f}s - "
            f"applying uniform {squeeze:.3f}x squeeze to fit the runtime")
        report = render(lambda n, a, b: min(max(b * squeeze, MIN_TEMPO),
                                            MAX_TOTAL_TEMPO))
        overlaps, shifted, max_shift = place(report)
        tail = tail_of(report)

    # ---- lay onto the timeline (sized from where the audio actually ends)
    n = int(max(total, tail) * SR) + SR
    track = np.zeros(n, dtype=np.float32)
    for r in report:
        p = fitdir / f"unit_{r['id']:04d}.wav"
        if not p.exists():
            continue
        a, _ = sf.read(str(p), dtype="float32")
        a = a.flatten()
        if a.size == 0:
            continue
        fade = min(int(0.008 * SR), a.size // 2)
        if fade > 0:
            a[:fade] *= np.linspace(0, 1, fade)
            a[-fade:] *= np.linspace(1, 0, fade)
        i = int(r["placed"] * SR)
        j = min(i + a.size, n)
        if j <= i:
            continue
        track[i:j] += a[: j - i]

    peak = float(np.max(np.abs(track))) or 1.0
    track = (track / peak) * 0.89
    narr = d / "narration.wav"
    sf.write(str(narr), track, SR)

    tight = [r for r in report if r["tight"]]
    save_json(d / "fit_report.json",
              {"units": report, "tight": len(tight), "schedule": schedule,
               "shifted": shifted, "max_shift": max_shift,
               "squeeze": round(squeeze, 4), "tail": round(tail, 3),
               "video": total, "gap": GAP})
    log(f"[s5] fitted {len(report)} units; {len(tight)} over-speed, "
        f"max overflow {max([r['overflow'] for r in report] or [0]):.2f}s; "
        f"schedule={schedule}, {overlaps} would have collided, "
        f"{shifted} shifted (max {max_shift:.2f}s); "
        f"tail {tail:.2f}s vs video {total:.2f}s -> {narr}")
    return {"narration": str(narr), "report": report}
