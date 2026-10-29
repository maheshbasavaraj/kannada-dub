import json, re
from pathlib import Path

d = Path("work/c__DMV_Clog")

def parse_srt(p):
    blocks = re.findall(r"(\d+)\n(\d\d:\d\d:\d\d,\d+) --> (\d\d:\d\d:\d\d,\d+)\n(.*?)\n\n", p.read_text(encoding="utf-8"), re.S)
    out = []
    for i, a, b, t in blocks:
        def sec(x):
            h, m, rest = x.split(":")
            s, ms = rest.split(",")
            return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000
        out.append((int(i), sec(a), sec(b), t.replace("\n", " ")))
    return out

for name in ["c__DMV_Clog_kn.srt", "c__DMV_Clog_en.srt"]:
    p = Path("out")/name
    if not p.exists():
        continue
    subs = parse_srt(p)
    ov = [(a[0], b[0], round(a[2]-b[1], 2)) for a, b in zip(subs, subs[1:]) if a[2] > b[1] + 0.001]
    print(name, "entries:", len(subs), "timestamp overlaps:", len(ov), ov[:10])

segs = json.loads((d/"segments.json").read_text(encoding="utf-8"))["segments"]
sov = [(a["id"], b["id"]) for a, b in zip(segs, segs[1:]) if a["end"] > b["start"] + 0.001]
print("segment speech overlaps in ASR:", len(sov), sov[:10])
print("meta:", json.loads((d/"meta.json").read_text(encoding="utf-8")))
