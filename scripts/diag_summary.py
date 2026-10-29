"""Summary of the new fit: tempo, gaps, drift, and whether it clears the video."""
import json
from pathlib import Path

d = Path("work/c__DMV_Clog")
rep = json.loads((d / "fit_report.json").read_text(encoding="utf-8"))
us = sorted(rep["units"], key=lambda r: r["placed"])
total = 291.83

print(f"units={len(us)}  total_video={total:.2f}s")
print(f"tight(capped)={rep.get('tight')}  shifted={rep.get('shifted')}  "
      f"max_shift={rep.get('max_shift')}  extra={rep.get('global_extra')}")

tails = [r["placed"] + r["final"] for r in us]
print(f"narration tail={max(tails):.2f}s  over_video={max(tails) - total:+.2f}s")

gaps = [us[i + 1]["placed"] - (us[i]["placed"] + us[i]["final"]) for i in range(len(us) - 1)]
print(f"gaps: min={min(gaps):.3f}  median={sorted(gaps)[len(gaps)//2]:.3f}  max={max(gaps):.3f}")
print(f"units with gap<0.05s: {sum(g <= 0.05 for g in gaps)}")

tempos = sorted(r["tempo"] for r in us)
print(f"tempo: min={tempos[0]:.2f} median={tempos[len(tempos)//2]:.2f} max={tempos[-1]:.2f}")

overlaps = [i for i in range(len(us) - 1)
            if us[i]["placed"] + us[i]["final"] > us[i + 1]["placed"] + 1e-6]
print(f"overlapping pairs: {len(overlaps)}")
