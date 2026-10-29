"""Kannada dub pipeline - English YouTube video in, Kannada voice-matched dub out.

  python dub.py --url <youtube url>
  python dub.py --file <local.mp4> --burn-subs
  python dub.py --url ... --stages fetch,asr,mt,ref,tts,fit,mix
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import s1_fetch, s2_asr, s3_mt  # noqa: E402
from pipeline.common import log  # noqa: E402

ALL = ["fetch", "asr", "mt", "ref", "tts", "fit", "post", "mix"]


def _stage(name: str):
    """Import a stage only when it is actually needed.

    The torch-backed stages (s4/s5/s6) drag in transformers -> sklearn, which is
    both slow and, on machines with Smart App Control enabled, can fail outright.
    Fetch/ASR/MT should not be hostage to that.
    """
    import importlib
    return importlib.import_module(f"pipeline.{name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="YouTube URL")
    ap.add_argument("--file", help="local video file")
    ap.add_argument("--stages", default=",".join(ALL))
    ap.add_argument("--whisper", default="large-v3")
    ap.add_argument("--compute", default="float16",
                    help="ctranslate2 compute type: float16 (cuda) / int8 (cpu)")
    ap.add_argument("--mt", default=s3_mt.DEFAULT_LLM,
                    help="translation engine: an Ollama model tag, or 'nllb' for the offline fallback")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--voice", default="narrator",
                    help="'narrator' clones the original speaker; 'happy_f' uses "
                         "the clean studio voice bundled with IndicF5 (no cloning, "
                         "no source noise). Voice cloning is optional.")
    ap.add_argument("--limit", type=int, help="only synthesise first N units (smoke test)")
    ap.add_argument("--burn-subs", action="store_true")
    ap.add_argument("--no-denoise", action="store_true",
                    help="skip the FFT denoise pass in post (keeps EQ + loudness)")
    ap.add_argument("--keep-original-audio", action="store_true",
                    help="do not duck the bed, just overlay")
    a = ap.parse_args()

    if not (a.url or a.file):
        ap.error("need --url or --file")
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]

    meta = s1_fetch.fetch(a.url or a.file)
    data = stems = ref = fitted = None

    # Resume: a consuming stage (tts/fit/mix) reuses what earlier stages wrote
    # instead of demanding they be re-run in the same invocation.
    if any(s in stages for s in ("tts", "fit", "mix")):
        from pipeline.common import WORK as _WORK, load_json as _load
        _d = _WORK / meta["id"]
        if data is None:
            data = _load(_d / "units.json") or _load(_d / "segments.json")

    if "asr" in stages:
        data = s2_asr.transcribe(meta, model_size=a.whisper, device=a.device,
                                 compute_type=a.compute)
    if "mt" in stages:
        # a.mt is the ollama tag; model_name is the nllb fallback repo.
        data = s3_mt.translate(meta, model_name=s3_mt.NLLB, device=a.device, llm=a.mt)
    if any(s in stages for s in ("ref", "tts", "fit", "mix")):
        stems = _stage("s4_voice").separate(meta)
    if "ref" in stages or "tts" in stages:
        ref = _stage("s4_voice").build_ref(meta, data, stems, voice=a.voice)
    if "tts" in stages:
        data = _stage("s4_voice").synth(meta, data, ref, device=a.device, limit=a.limit)
    if "fit" in stages:
        fitted = _stage("s5_fit").fit(meta, data)
    if "post" in stages:
        # Clean-up + loudness. Must run before mix: mix ducks the bed against
        # this narration, so its level has to be final by then.
        fitted = _stage("s7_post").post(meta, fitted,
                                        denoise=not a.no_denoise)
    if "mix" in stages:
        if fitted is None:
            # mix without a fresh fit/post: fall back to whatever is on disk so
            # a mix-only rerun uses the cleaned narration, not the raw one.
            from pipeline.common import WORK as _W, load_json as _lj
            _d = _W / meta["id"]
            _post = _d / "narration_post.wav"
            _raw = _d / "narration.wav"
            if _post.exists() or _raw.exists():
                fitted = {"narration": str(_post if _post.exists() else _raw)}
        result = _stage("s6_mix").mix(meta, data, stems, fitted,
                            burn_subs=a.burn_subs,
                            keep_original_audio=a.keep_original_audio)
        print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
