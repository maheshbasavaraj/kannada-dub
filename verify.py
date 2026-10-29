"""Objective check of a finished dub: did Kannada speech land where the English
was, and does it say what we asked it to say?

  python verify.py --id <video_id> [--model small] [--wav narration]

Reports, per dubbing unit:
  * hit      - Kannada speech detected inside the expected time window
  * offset   - detected speech onset minus expected start (seconds)
  * cer      - character error rate of what the ASR heard vs the script
No audio leaves the machine; the model is whatever faster-whisper can load.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline.common import WORK, log  # noqa: E402


def cer(ref: str, hyp: str) -> float:
    a = "".join(ref.split())
    b = "".join(hyp.split())
    if not a:
        return 0.0
    sm = difflib.SequenceMatcher(None, a, b)
    return 1.0 - sm.ratio()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--model", default="small")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute", default="int8")
    ap.add_argument("--wav", help="override the audio to check")
    a = ap.parse_args()

    d = WORK / a.id
    data = json.loads((d / "segments.json").read_text(encoding="utf-8"))
    wav = a.wav or str(d / "narration.wav")
    if not Path(wav).exists():
        log(f"no audio at {wav}")
        return 1

    from faster_whisper import WhisperModel
    log(f"[verify] {a.model} on {Path(wav).name}")
    m = WhisperModel(a.model, device=a.device, compute_type=a.compute)
    segs, _ = m.transcribe(wav, language="kn", beam_size=5, word_timestamps=True,
                           vad_filter=True, condition_on_previous_text=False)
    heard = [{"start": s.start, "end": s.end, "text": s.text.strip()}
             for s in segs if s.text.strip()]
    log(f"[verify] {len(heard)} Kannada spans detected")

    rows, hits, cers, offs = [], 0, [], []
    for u in data["segments"]:
        want = (u.get("kn_fit") or u.get("kn") or "").strip()
        exp_start = u["start"]
        exp_end = (u.get("next_start") or exp_start + u["dur"]) - 0.05
        inside = [h for h in heard
                  if h["end"] > exp_start - 0.35 and h["start"] < exp_end + 0.35]
        if inside:
            hits += 1
            onset = min(h["start"] for h in inside)
            off = round(onset - exp_start, 3)
            offs.append(off)
            hyp = " ".join(h["text"] for h in inside)
            c = round(cer(want, hyp), 3)
            cers.append(c)
        else:
            off, c, hyp = None, None, ""
        rows.append({"id": u["id"], "start": exp_start, "script": want,
                     "heard": hyp, "offset": off, "cer": c})

    n = len(data["segments"])
    report = {
        "units": n,
        "coverage": round(hits / n, 3) if n else 0,
        "mean_offset": round(sum(offs) / len(offs), 3) if offs else None,
        "max_abs_offset": round(max((abs(o) for o in offs), default=0), 3),
        "mean_cer": round(sum(cers) / len(cers), 3) if cers else None,
        "rows": rows,
    }
    (d / "verify_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[verify] coverage {report['coverage']:.0%}  "
        f"mean offset {report['mean_offset']}s  max |offset| {report['max_abs_offset']}s  "
        f"mean CER {report['mean_cer']}")
    for r in rows[:12]:
        log(f"  #{r['id']:>3} @{r['start']:>7.2f}s off={r['offset']} cer={r['cer']}\n"
            f"      want: {r['script'][:70]}\n      heard: {r['heard'][:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
