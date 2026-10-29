"""One-off repair for unit 15.

The LLM emitted the Korean word \uacf5\uac04 ("space") inside a Kannada line. The
script-guard retry then re-translated the WHOLE line and lost "need for
separateness", so instead restore the model's original (good) sentence with the
single foreign token replaced by the Kannada word \u0c9c\u0cbe\u0c97.
"""
from pathlib import Path

from pipeline.common import save_json

d = Path("work/c__DMV_Clog")
sp = d / "segments.json"
import json

data = json.loads(sp.read_text(encoding="utf-8"))

FIX = ("\u0cb5\u0cc8\u0caf\u0c95\u0ccd\u0ca4\u0cbf\u0c95 \u0c9c\u0cbe\u0c97\u0ca6 "
       "\u0c85\u0c97\u0ca4\u0ccd\u0caf\u0c95\u0ccd\u0c95\u0cc6 \u0c85\u0ca5\u0cb5\u0cbe "
       "\u0cb8\u0c82\u0c97\u0cbe\u0ca4\u0cbf\u0caf \u0c95\u0ccb\u0caa\u0c95\u0ccd\u0c95\u0cc6 "
       "\u0cb9\u0cc6\u0ca6\u0cb0\u0cbf, \u0c85\u0cb5\u0cb0\u0cc1 "
       "\u0cae\u0ccc\u0ca8\u0cb5\u0cbe\u0c97\u0cc1\u0cb5 \u0c85\u0ca5\u0cb5\u0cbe "
       "\u0ca4\u0caa\u0ccd\u0caa\u0cbf\u0cb8\u0cbf\u0c95\u0cca\u0cb3\u0ccd\u0cb3\u0cc1\u0cb5 "
       "\u0cb9\u0cbe\u0ca6\u0cbf \u0cb9\u0cbf\u0ca1\u0cbf\u0caf\u0cc1\u0cb5\u0cc1\u0ca6\u0cbf\u0cb2\u0ccd\u0cb2.")

done = False
for s in data["segments"]:
    if s["id"] == 15:
        s["kn"] = s["kn_fit"] = FIX
        s["kn_chars"] = len(FIX)
        done = True
print("unit 15 set:", done)
print(FIX)
save_json(sp, data)

w = d / "tts" / "unit_0015.wav"
if w.exists():
    w.unlink()
    print("deleted", w.name)
