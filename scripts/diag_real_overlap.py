"""Rebuild placement from the real fit wavs and count true sample overlaps."""
import json
from pathlib import Path
import soundfile as sf

d = Path("work/c__DMV_Clog")
rep = json.loads((d / "fit_report.json").read_text(encoding="utf-8"))
units = sorted(rep["units"], key=lambda r: r["placed"])

rows = []
for r in units:
    p = d / "fit" / f"unit_{r['id']:04d}.wav"
    info = sf.info(str(p))
    real = info.frames / info.samplerate
    rows.append((r["id"], r["placed"], real, r["final"], round(real - r["final"], 3)))

print(f"{'id':>3} {'placed':>8} {'actual':>8} {'reported':>9} {'delta':>7}")
for row in rows:
    print(f"{row[0]:>3} {row[1]:>8.3f} {row[2]:>8.3f} {row[3]:>9.3f} {row[4]:>7.3f}")

ov = []
for a, b in zip(rows, rows[1:]):
    end = a[1] + a[2]
    if end > b[1] + 0.001:
        ov.append((a[0], b[0], round(end - b[1], 3)))
print("\nREAL overlap pairs:", len(ov), ov[:20])
tail = max(r[1] + r[2] for r in rows)
print("real narration end:", round(tail, 2), "| video:", 291.828,
      "| overshoot:", round(tail - 291.828, 2))
