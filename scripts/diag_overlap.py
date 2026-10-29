import json, pathlib
d = pathlib.Path("work/c__DMV_Clog")
rep = json.loads((d/"fit_report.json").read_text(encoding="utf-8"))
u = sorted(rep["units"], key=lambda r: r["placed"])
ov = []
for a, b in zip(u, u[1:]):
    end = a["placed"] + a["final"]
    if end > b["placed"] + 0.001:
        ov.append((a["id"], b["id"], round(end-b["placed"], 3)))
print("schedule:", rep["schedule"], "shifted:", rep["shifted"], "max_shift:", rep["max_shift"])
print("units:", len(u), "OVERLAP PAIRS:", len(ov))
print("first 15 overlaps:", ov[:15])
tail = max(r["placed"]+r["final"] for r in u)
print("narration ends at", round(tail, 2), "video is", json.loads((d/"meta.json").read_text())["duration"])
print("units with overflow>0.01:", sum(1 for r in u if r["overflow"] > 0.01))
print("tight (>1.4x):", sum(1 for r in u if r["tight"]))
segs = json.loads((d/"segments.json").read_text(encoding="utf-8"))["segments"]
print("seg0 kn:", segs[0].get("kn"))
print("any kn_fit:", any("kn_fit" in s for s in segs))
print("tts wavs:", len(list((d/"tts").glob("*.wav"))), "fit wavs:", len(list((d/"fit").glob("*.wav"))))
