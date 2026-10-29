"""Stage 3 - translate each dubbing unit English -> Kannada.

Primary engine is a local LLM served by Ollama (default gemma4:12b): it handles
Kannada register and, crucially, can be told how many seconds the line has to
fit in, so the translation is born the right length instead of being squeezed
afterwards. NLLB-200 is kept as an offline fallback.

Adds to every unit:
  kn       - the translation
  kn_fit   - duration-aware translation (what stage 5 synthesises)
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from .common import log, save_json

NLLB = "facebook/nllb-200-distilled-600M"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_LLM = "gemma4:12b"
BATCH = 6                      # units per LLM call
KANNADA = re.compile(r"[\u0C80-\u0CFF]")
LATIN_WORD = re.compile(r"[A-Za-z]{4,}")
# Any script that is neither Kannada nor ASCII. The LLM occasionally drops a
# token from another language into an otherwise-Kannada line - seen for real:
# Korean 공간 for "space", Devanagari पातलारे for "ಪಾತಳಾರೆ". Such a line still
# contains Kannada, so the two checks above pass it; it must be rejected here or
# it gets synthesised as gibberish.
FOREIGN = re.compile(
    r"[\u0900-\u0C7F"                 # Devanagari .. Telugu (Kannada is 0C80+)
    r"\u0D00-\u0DFF"                  # Malayalam, Sinhala
    r"\u0E00-\u0E7F"                  # Thai
    r"\u0400-\u04FF"                  # Cyrillic
    r"\u0590-\u06FF"                  # Hebrew, Arabic
    r"\u1100-\u11FF\u3040-\u30FF"     # Hangul jamo, kana
    r"\u3400-\u4DBF\u4E00-\u9FFF"     # CJK
    r"\uAC00-\uD7AF]"                 # Hangul syllables
)

# Kannada narration rate, used to give the model a CHARACTER budget.
#
# This is the fix for the real bottleneck. Measured on a full video, the
# translated audio came out 471.7s for 282.0s of slots - a median of 1.67x too
# long, p90 2.5x. Tempo-fitting cannot absorb that (1.67x sounds awful), so the
# translation has to be born the right length. "About 2 words per second" was
# too loose a guide; a character count is something the model can actually hit.
CHARS_PER_SEC = 13.0
BUDGET_SLACK = 1.15            # allow this much over budget before retrying

SYS = (
    "You are a dubbing translator for Kannada films and YouTube narration. "
    "You translate English narration into natural, everyday spoken Kannada.\n"
    "Rules:\n"
    "1. Output ONLY the translations, in the exact numbered format asked for. "
    "No notes, no explanations, no English, no romanisation.\n"
    "2. Write Kannada script (\u0c95\u0ca8\u0ccd\u0ca8\u0ca1 \u0cb2\u0cbf\u0caa\u0cbf) only. "
    "Never leave an English sentence untranslated. Do not emit characters from "
    "any other writing system (no Devanagari, no Hangul, no Chinese, no other script).\n"
    "3. Translate the MEANING and the intent, not the words. A Kannada viewer "
    "must feel the same thing the English viewer felt. If a literal rendering "
    "sounds stiff or foreign, rewrite it the way a Kannada narrator would "
    "actually say it.\n"
    "4. Carry the speaker's tone across. A wry line stays wry, a serious line "
    "stays serious, a warm line stays warm. Do not flatten everything into one "
    "neutral register.\n"
    "5. LENGTH IS A HARD REQUIREMENT. Each line comes with a character budget "
    "in braces, e.g. {max 36}. Your Kannada line must not exceed it. This is "
    "the time the actor has to speak the line. When you are over budget, cut "
    "and merge ideas - drop adjectives, use one shorter synonym, prefer the "
    "direct verb - rather than translating every English word. Go UNDER the "
    "budget rather than over it.\n"
    "6. Keep personal names as they are, written in Kannada script. Translate "
    "idioms to the Kannada idiom that means the same thing, never word for word.\n"
    "7. Use the polite/neutral narration register, not slang.\n"
    "8. Never repeat a word or phrase more than once."
)


def _context_line(meta: dict | None) -> str:
    """Give the model the subject so it picks the right sense of ambiguous words."""
    if not meta:
        return ""
    bits = []
    if meta.get("title"):
        bits.append(f"Video title: {meta['title']}")
    if meta.get("uploader"):
        bits.append(f"Channel: {meta['uploader']}")
    if bits:
        bits.append("Use this to choose the right meaning where a word is "
                    "ambiguous, and keep terminology consistent across lines.")
    return "\n".join(bits)


# --------------------------------------------------------------- ollama call
def _ollama(model: str, messages: list, timeout: int = 600,
            temperature: float = 0.1, num_ctx: int = 4096,
            num_predict: int = 512) -> str:
    body = json.dumps({
        "model": model, "messages": messages, "stream": False,
        # gemma4 is a reasoning model: left to itself it spends the whole
        # num_predict budget on hidden thinking tokens and returns empty content.
        "think": False,
        "options": {"temperature": temperature, "top_p": 0.9,
                    "repeat_penalty": 1.15, "num_ctx": num_ctx,
                    "num_predict": num_predict},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return (data.get("message") or {}).get("content", "").strip()


def ollama_up(model: str = DEFAULT_LLM, timeout: int = 90, attempts: int = 3) -> bool:
    """Probe ollama, retrying: a cold start can exceed a single short timeout,
    and falling through to NLLB silently is worse than waiting."""
    for i in range(attempts):
        try:
            _ollama(model, [{"role": "user", "content": "hi"}],
                    timeout=timeout, num_predict=8, num_ctx=512)
            return True
        except Exception as e:  # noqa: BLE001
            log(f"[s3] ollama probe {i + 1}/{attempts} failed: {e}")
            time.sleep(2)
    return False


# ------------------------------------------------------------------ validation
def _bad(text: str) -> str | None:
    """Return a reason string when the translation is unusable."""
    t = text.strip()
    if not t:
        return "empty"
    if not KANNADA.search(t):
        return "no kannada script"
    if LATIN_WORD.search(t):
        return "leftover english words"
    m = FOREIGN.search(t)
    if m:
        return f"foreign script {m.group(0)!r}"
    words = t.split()
    if len(words) >= 4:
        for i in range(len(words) - 3):
            if words[i:i + 3] == words[i + 1:i + 4] and words[i] != words[i + 1]:
                return "stutter"
        # any single token repeated 4+ times anywhere
        from collections import Counter
        if Counter(words).most_common(1)[0][1] >= max(4, len(words) // 3):
            return "repetition loop"
    return None


def _parse(raw: str, n: int) -> list:
    """Pull '1| text' lines out of a model reply."""
    out = {}
    for line in raw.splitlines():
        m = re.match(r"\s*(\d+)\s*[|.:)\-]\s*(.+)", line.strip())
        if m:
            i = int(m.group(1))
            if 1 <= i <= n:
                out[i] = m.group(2).strip()
    return [out.get(i, "") for i in range(1, n + 1)]


def _target_seconds(u: dict) -> float:
    nxt = u.get("next_start")
    return (nxt - u["start"]) if nxt else u["dur"]


def _char_budget(u: dict) -> int:
    """How many Kannada characters this line has time for."""
    return max(10, int(_target_seconds(u) * CHARS_PER_SEC))


def _over_budget(text: str, budget: int) -> bool:
    return len(text.strip()) > budget * BUDGET_SLACK


def _llm_batch(model: str, chunk: list, ctx: str = "") -> list:
    lines = []
    for i, u in enumerate(chunk, 1):
        secs = max(1.0, _target_seconds(u) - 0.15)
        lines.append(f"{i}| [{secs:.1f}s {{max {_char_budget(u)}}} ] {u['text']}")
    prompt = ("Translate these English narration lines into Kannada.\n"
              "Respect the character budget in braces for every line - it is the "
              "time available to speak it.\n"
              "Reply with exactly the same number of lines, same numbering, "
              "format '<n>| <kannada text>'.\n"
              + (f"\n{ctx}\n" if ctx else "")
              + "\n" + "\n".join(lines))
    raw = _ollama(model, [{"role": "system", "content": SYS},
                          {"role": "user", "content": prompt}])
    return _parse(raw, len(chunk))


def _llm_one(model: str, u: dict, extra: str = "", ctx: str = "") -> str:
    secs = max(1.0, _target_seconds(u) - 0.15)
    budget = _char_budget(u)
    prompt = (f"Translate this English narration line into Kannada.\n"
              f"Time available when spoken: about {secs:.1f} seconds. "
              f"Hard limit: {budget} Kannada characters - stay under it, even if "
              f"that means dropping detail.\n"
              f"Reply with the Kannada line only.\n"
              + (f"{ctx}\n" if ctx else "")
              + f"{extra}\n"
              f"English: {u['text']}")
    return _ollama(model, [{"role": "system", "content": SYS},
                           {"role": "user", "content": prompt}])


# ------------------------------------------------------------------- nllb fallback
def _nllb(units: list, model_name: str, device: str, batch: int = 16) -> None:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    log(f"[s3] nllb fallback: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
    tgt_id = tok.convert_tokens_to_ids("kan_Knda")
    for i in range(0, len(units), batch):
        chunk = units[i:i + batch]
        enc = tok([u["text"] for u in chunk], return_tensors="pt",
                  padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(**enc, forced_bos_token_id=tgt_id,
                                 num_beams=4, no_repeat_ngram_size=3,
                                 max_new_tokens=256)
        for u, t in zip(chunk, tok.batch_decode(out, skip_special_tokens=True)):
            u["kn"] = t.strip()


# ------------------------------------------------------------------------- main
def translate(meta: dict, model_name: str = NLLB, device: str = "cuda",
              llm: str | None = DEFAULT_LLM, force: bool = False) -> dict:
    d = Path(meta["source"]).parent
    sp = d / "segments.json"
    data = json.loads(sp.read_text(encoding="utf-8"))
    segs = data["segments"]

    if not force and all(s.get("kn") and not _bad(s["kn"]) for s in segs):
        log("[s3] cached translations (all clean)")
        return data

    todo = segs if force else [s for s in segs if not s.get("kn") or _bad(s["kn"])]
    ctx = _context_line(meta)
    if llm and ollama_up(llm):
        log(f"[s3] translating {len(todo)} units with ollama/{llm}")
        if ctx:
            log(f"[s3] context: {ctx.splitlines()[0]}")
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            try:
                got = _llm_batch(llm, chunk, ctx=ctx)
            except Exception as e:  # noqa: BLE001
                log(f"[s3] batch failed ({e}); per-unit retry")
                got = ["" for _ in chunk]
            for u, t in zip(chunk, got):
                budget = _char_budget(u)
                if _bad(t) or _over_budget(t, budget):
                    for attempt in range(2):
                        extra = ""
                        bad = _bad(t)
                        if bad and bad.startswith("foreign script"):
                            extra = ("Your last attempt used characters from "
                                     "another script. Write the line in Kannada "
                                     "script only - no Devanagari, no Hangul, no "
                                     "other writing system.")
                        elif _over_budget(t, budget):
                            extra = (f"Your last attempt was {len(t)} characters, "
                                     f"over the {budget} character limit. Say the "
                                     f"same thing in fewer words.")
                        try:
                            t2 = _llm_one(llm, u, extra=extra, ctx=ctx)
                        except Exception:  # noqa: BLE001
                            continue
                        if _bad(t2):
                            continue
                        t = t2
                        if not _over_budget(t2, budget):
                            break
                reason = _bad(t)
                if reason:
                    log(f"[s3] unit {u['id']} still bad ({reason}) - nllb later")
                    u["kn"] = ""
                else:
                    u["kn"] = t
            for s in segs:
                s["kn_fit"] = s.get("kn", "")
                s["kn_chars"] = len(s["kn_fit"])
            save_json(sp, data)  # partial: a late nllb failure must not cost the run
            log(f"[s3] {min(i + BATCH, len(todo))}/{len(todo)}")
    else:
        log(f"[s3] ollama/{llm} not reachable - using nllb only")

    left = [s for s in segs if not s.get("kn") or _bad(s["kn"])]
    if left:
        _nllb(left, model_name, device)
        for u in left:
            u["kn"] = u["kn"].strip()

    still = [s["id"] for s in segs if not s.get("kn") or _bad(s["kn"])]
    for s in segs:
        s["kn_fit"] = s.get("kn", "")
        s["kn_chars"] = len(s["kn_fit"])
    data["mt_model"] = f"ollama:{llm}" if llm else model_name
    data["mt_bad"] = still
    save_json(sp, data)
    log(f"[s3] translated {len(segs)} units; unusable: {still or 'none'}")
    return data
