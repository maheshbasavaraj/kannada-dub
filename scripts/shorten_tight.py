"""Shorten only overlong dubbed units, preserving the English meaning.

Run after a measured fit; writes new translations and invalidates only changed
TTS wavs. Never replace a line if the candidate is not shorter and valid.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from pipeline import s3_mt
from pipeline.common import log, save_json

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "work" / "c__DMV_Clog"
segments_path = D / "segments.json"
segments = json.loads(segments_path.read_text(encoding="utf-8"))
report = json.loads((D / "fit_report.json").read_text(encoding="utf-8"))["units"]
by_id = {u["id"]: u for u in segments["segments"]}
meta = json.loads((D / "meta.json").read_text(encoding="utf-8"))
backup = D / "segments.before_shortening.json"
if not backup.exists():
    shutil.copy2(segments_path, backup)

# Target ≈1.25x natural/slot; retain at least 60% of the original line.
# The fitted narration starts drifting when ratio exceeds the 1.4x cap.
targets = {}
for r in report:
    ratio = r["natural"] / r["avail"]
    if ratio <= 1.34:
        continue
    u = by_id[r["id"]]
    target_chars = max(14, int(len(u["kn"]) * max(.60, min(.93, 1.24 / ratio))))
    if target_chars < len(u["kn"]) - 3:
        targets[u["id"]] = target_chars

if not s3_mt.ollama_up():
    raise RuntimeError("Local Ollama not reachable; original translations left intact")

changed = []
for uid, budget in targets.items():
    u = by_id[uid]
    old = u["kn"]
    context = s3_mt._context_line(meta)
    prompt = (
        f"{context}\nThis is spoken narration. Condense the EXISTING Kannada to no more "
        f"than {budget} Unicode characters. Preserve meaning, metaphor, and tone; "
        "do not introduce a different claim. Prefer direct words over extra clauses. "
        "Output only one Kannada line; no numbering or explanations.\n"
        f"English meaning: {u['text']}\nExisting Kannada: {old}"
    )
    for attempt in range(3):
        result = s3_mt._ollama(s3_mt.DEFAULT_LLM, [
            {"role": "system", "content": s3_mt.SYS},
            {"role": "user", "content": prompt},
        ], num_predict=256).strip().strip('"')
        if not s3_mt._bad(result) and len(result) <= min(len(old) - 3, int(budget * 1.15)):
            break
        prompt += (f"\nYour previous attempt was {len(result)} characters: "
                   f"{result}. Rewrite under {budget}, preserve meaning.")
    else:
        log(f"[shorten] unit {uid} unchanged: could not reach {budget} characters")
        continue
    u["kn"] = u["kn_fit"] = result
    u["kn_chars"] = len(result)
    save_json(segments_path, segments)
    wav = D / "tts" / f"unit_{uid:04d}.wav"
    wav.unlink(missing_ok=True)
    changed.append((uid, len(old), len(result), budget))
    log(f"[shorten] {uid}: {len(old)} to {len(result)} chars (budget {budget})")

log(f"[shorten] changed {len(changed)}/{len(targets)}: {changed}")
