"""Why does every translation come back empty?

s3_mt reads data["message"]["content"] from ollama's /api/chat. If the model
spends its whole num_predict budget on reasoning tokens, content is empty and
the thinking lands in a separate field. Try the same prompt with and without
thinking enabled and print both fields.
"""
from __future__ import annotations

import json
import time
import urllib.request

URL = "http://127.0.0.1:11434/api/chat"
PROMPT = "Translate to Kannada: The sun rises in the east."


def call(label: str, think=None, num_predict: int = 512) -> None:
    body = {
        "model": "gemma4:12b",
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": False,
        "options": {"num_predict": num_predict, "num_ctx": 4096, "temperature": 0.2},
    }
    if think is not None:
        body["think"] = think
    t0 = time.time()
    req = urllib.request.Request(URL, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read().decode("utf-8"))
    dt = time.time() - t0
    m = d.get("message", {}) or {}
    print(f"--- {label}  (think={think}, num_predict={num_predict})  {dt:.0f}s")
    print(f"    message keys : {list(m)}")
    print(f"    content      : {m.get('content','')!r}"[:400])
    print(f"    thinking     : {m.get('thinking','')!r}"[:300])
    print(f"    done_reason  : {d.get('done_reason')}  eval_count={d.get('eval_count')}")


if __name__ == "__main__":
    call("A: default")
    call("B: think=False", think=False)
    call("C: no-think burst", think=False, num_predict=64)
