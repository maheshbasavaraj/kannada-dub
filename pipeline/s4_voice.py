"""Stage 4 - separate the bed, capture the narrator's voice as a reference
clip, then synthesise every Kannada unit in that cloned voice with IndicF5."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .common import (MIX_SR, REFS, ROOT, SR, ensure_dirs, ffmpeg, load_json, log,
                     run, save_json)

# ai4bharat/IndicF5 is gated on the Hub; this mirror carries the same weights and
# the same model.py. Override with INDICF5_REPO if you have official access.
INDICF5 = os.environ.get("INDICF5_REPO", "6Morpheus6/IndicF5")
# F5 generates a total conditioned sequence, then slices off the reference
# frames before vocoding (utils_infer.py:473-485). The reference is NOT added
# to the returned WAV. Duration of generated speech depends on the ratio of
# reference AUDIO length to UTF-8 length of its MATCHING transcript. Never
# shorten a reference without updating its transcript to the same spoken words.
REF_SECONDS = (2.5, 4.5)
TTS_SPEED = 1.4          # modest speed-up; timing fit has a separate 1.4x cap

# Voice cloning is OPTIONAL. These are the clean studio prompts bundled with
# IndicF5 itself - using one means no narrator audio at all, so nothing from the
# source recording (noise, room tone, music bleed) can be cloned into the output.
BUILTIN_VOICES = {
    "happy_f": ("prompts/PAN_F_HAPPY_00001.wav",
                "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ "
                "ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"),
}
DEFAULT_VOICE = "narrator"   # "narrator" clones the original; else a builtin key


# ---------------------------------------------------------------- separation
def separate(meta: dict) -> dict:
    """Demucs two-stem split -> vocals.wav / no_vocals.wav (music + sfx)."""
    d = Path(meta["source"]).parent
    stems = d / "stems"
    voc, nov = stems / "vocals.wav", stems / "no_vocals.wav"
    if voc.exists() and nov.exists():
        return {"vocals": str(voc), "no_vocals": str(nov)}

    ensure_dirs(stems)
    log("[s4] demucs two-stem separation")
    try:
        run([__import__("sys").executable, "-m", "demucs", "--two-stems=vocals",
             "-n", "htdemucs", "-o", str(stems), meta["bed_wav"]])
        produced = sorted(stems.glob("htdemucs/*/vocals.wav"))
        if not produced:
            raise RuntimeError("demucs produced nothing")
        p = produced[0].parent
        shutil.copy2(p / "vocals.wav", voc)
        shutil.copy2(p / "no_vocals.wav", nov)
        shutil.rmtree(p.parent, ignore_errors=True)
    except Exception as e:  # noqa: BLE001
        log(f"[s4] demucs unavailable ({e}); falling back to speech-gated bed")
        ffmpeg(["-i", meta["bed_wav"], "-ac", "2", "-ar", str(MIX_SR), str(voc)])
        gate = _speech_gate_filter(meta)
        ffmpeg(["-i", meta["bed_wav"], "-af", gate, "-ac", "2",
                "-ar", str(MIX_SR), str(nov)])
    return {"vocals": str(voc), "no_vocals": str(nov)}


def _speech_gate_filter(meta: dict) -> str:
    """Mute the original English speech spans so it cannot bleed under Kannada."""
    d = Path(meta["source"]).parent
    data = load_json(d / "segments.json", {"segments": []})
    exprs = []
    for s in data["segments"]:
        a, b = max(0.0, s["start"] - 0.15), s["end"] + 0.15
        exprs.append(f"between(t,{a:.3f},{b:.3f})")
    if not exprs:
        return "anull"
    return "volume=volume=0:enable='" + "+".join(exprs) + "'"


# ------------------------------------------------------------- ref selection
def _unit_snr(y, sr) -> float:
    """Rough SNR of a speech span: frame RMS vs the 5th-percentile frame RMS.

    Used to pick a *clean* reference. Voice character is copied from the
    reference, so a noisy reference clones its noise into every line - which is
    what made the Kannada come out unclear.
    """
    import numpy as np

    fr = max(1, int(0.02 * sr))
    n = len(y) // fr
    if n < 4:
        return 0.0
    rms = np.sqrt((y[: n * fr].reshape(n, fr) ** 2).mean(axis=1) + 1e-12)
    floor = float(np.percentile(rms, 5))
    sig = float(np.percentile(rms, 70))
    return 20.0 * np.log10(max(sig, 1e-9) / max(floor, 1e-9))


def _best_ref_unit(data: dict, stems: dict | None = None) -> dict:
    """Pick the reference clip: right length AND cleanest audio.

    F5 removes reference frames before returning audio, but the duration of
    generated speech depends on reference audio length / transcript byte length.
    Cleanliness matters because reference noise is cloned with the voice.
    """
    lo, hi = REF_SECONDS
    cands = [u for u in data["segments"] if lo <= u["dur"] <= hi]
    if not cands:  # fall back to the nearest-length unit
        cands = sorted(data["segments"], key=lambda u: abs(u["dur"] - (lo + hi) / 2))[:5]
    # A naturally paced reference is critical: F5 estimates generation time
    # from reference audio seconds per UTF-8 byte of its transcript. A slow
    # three-second phrase with few words can make every Kannada line too long.
    # Exclude extreme pause-heavy or hurried prompts before SNR ranking.
    paced = [u for u in cands if 0.025 <= u["dur"] / len(u["text"].encode("utf-8")) <= 0.055]
    if paced:
        cands = paced

    if stems:
        try:
            from .common import SR, load_mono

            y, sr = load_mono(stems["vocals"], SR)
            best, best_score = None, -1e9
            for u in cands:
                a = int(u["start"] * sr)
                b = min(int(u["end"] * sr), len(y))
                if b - a < int(0.5 * sr):
                    continue
                snr = _unit_snr(y[a:b], sr)
                # prefer clean, then the length closest to the middle of the window
                pace = u["dur"] / len(u["text"].encode("utf-8"))
                score = snr - abs(u["dur"] - (lo + hi) / 2) * 2.0 - max(0, pace - 0.05) * 100
                if score > best_score:
                    best, best_score = u, score
            if best is not None:
                log(f"[s4] ref candidate unit {best['id']} "
                    f"({best['dur']:.1f}s, ~{best_score:.1f} score)")
                return best
        except Exception as e:  # noqa: BLE001
            log(f"[s4] ref quality scan skipped ({e})")

    return max(cands, key=lambda u: u["dur"])


def _builtin_ref(meta: dict, voice: str, force: bool = False) -> dict:
    """Reference from a clean prompt shipped with IndicF5 (cloning disabled)."""
    ref = REFS / f"{meta['id']}_{voice}.wav"
    refj = REFS / f"{meta['id']}_{voice}.json"
    if ref.exists() and refj.exists() and not force:
        return load_json(refj)

    from huggingface_hub import hf_hub_download

    fname, text = BUILTIN_VOICES[voice]
    src = Path(hf_hub_download(INDICF5, fname))
    ensure_dirs(REFS)
    shutil.copy2(src, ref)
    from .common import ffprobe_duration

    info = {"ref_audio": str(ref), "ref_text": text, "ref_unit": None,
            "ref_dur": round(ffprobe_duration(ref), 3),
            "ref_window": list(REF_SECONDS), "ref_source": "builtin",
            "voice": voice}
    save_json(refj, info)
    log(f"[s4] using builtin voice '{voice}' ({info['ref_dur']:.1f}s) - "
        "cloning disabled")
    return info


INDIC_REF_VOICE = "happy_f"   # bundled prompt whose transcript is published


def build_ref(meta: dict, data: dict, stems: dict, force: bool = False,
              voice: str = DEFAULT_VOICE) -> dict:
    """Build the F5 conditioning reference.

    Do NOT condition on the source recording. F5 takes (ref_audio, ref_text) as
    its prompt; feeding it an English clip plus its English transcript while
    asking for Kannada is out of distribution for an Indic model. Measured
    effect: every clip opens with ~0.6-3.0s of garbled, foreign-sounding
    syllables before the real line begins, and clips come out far too long
    (one test line: 15.7s from the English prompt vs 9.0s from the Indic one).
    The bundled Indic prompt is in-distribution: clean speech from 0.00s.
    """
    if not voice or voice == "narrator":
        voice = INDIC_REF_VOICE
    return _builtin_ref(meta, voice, force=force)



# ------------------------------------------------------------------ synthesis
def _repo_dir() -> str:
    """Make `f5_tts` importable for the remote model.py.

    Prefers the project-local patched copy (scripts/prepare_tts_src.py), falls
    back to the raw Hub snapshot.
    """
    import sys
    local = ROOT / "tts_src"
    if (local / "f5_tts").exists():
        p = str(local)
    else:
        from huggingface_hub import snapshot_download
        p = snapshot_download(INDICF5)
    if p not in sys.path:
        sys.path.insert(0, p)
    return p


def _repair_weights(model) -> int:
    """Load the DiT weights for real.

    model.safetensors was saved from a torch.compile'd EMA module, so its keys
    read 'ema_model._orig_mod.*' while the module tree wants '*'. model.py loads
    with strict=False, so that mismatch is silent and the DiT stays randomly
    initialised -- which is what produced NaN audio. Strip the wrappers, load
    again, and refuse to continue if it still does not line up.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    sd = load_file(hf_hub_download(INDICF5, "model.safetensors"))
    target = model.ema_model
    want = set(target.state_dict().keys())

    clean, unused = {}, 0
    for k, v in sd.items():
        short = k
        for pre in ("ema_model._orig_mod.", "ema_model.", "_orig_mod."):
            if short.startswith(pre):
                short = short[len(pre):]
                break
        if short in want:
            clean[short] = v
        else:
            unused += 1

    res = target.load_state_dict(clean, strict=False)
    log(f"[s4] weights {len(clean)}/{len(want)} matched, "
        f"{len(res.missing_keys)} missing, {unused} unused")
    if len(clean) < 0.9 * len(want):
        raise RuntimeError(
            f"IndicF5 weights did not load ({len(clean)}/{len(want)} tensors) - "
            "refusing to synthesise NaN audio")
    return len(clean)


def _load_model(device: str = "cuda"):
    import torch
    from transformers import AutoModel
    # model.py wraps the DiT and the vocoder in torch.compile, which needs
    # Triton; on Windows the passthrough is the only workable setting.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    _repo_dir()
    log(f"[s4] loading {INDICF5}")
    model = AutoModel.from_pretrained(INDICF5, trust_remote_code=True)
    _repair_weights(model)
    try:
        model = model.to(device)
    except Exception as e:  # noqa: BLE001
        log(f"[s4] staying on CPU ({e})")
    return model


def _to_float(audio) -> "object":
    """IndicF5 hands back int16 samples; normalise to float32 in [-1, 1]."""
    import numpy as np
    raw = np.asarray(audio)
    if np.issubdtype(raw.dtype, np.integer):
        return raw.astype(np.float32) / 32768.0
    return raw.astype(np.float32)


def synth(meta: dict, data: dict, ref: dict, device: str = "cuda",
          limit: int | None = None) -> dict:
    import numpy as np
    import soundfile as sf

    d = Path(meta["source"]).parent
    tts = d / "tts"
    ensure_dirs(tts)
    units = data["segments"][:limit] if limit else data["segments"]
    pending = [u for u in units if not (tts / f"unit_{u['id']:04d}.wav").exists()]
    if not pending:
        log("[s4] all units already synthesised")
        return data

    model = _load_model(device)
    import sys as _sys
    _src = Path(__file__).resolve().parent.parent / "tts_src"
    if str(_src) not in _sys.path:
        _sys.path.insert(0, str(_src))
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text
    rp, rt = preprocess_ref_audio_text(ref["ref_audio"], ref["ref_text"])
    for n, u in enumerate(pending, 1):
        text = u.get("kn_fit") or u["kn"]
        wav_path = tts / f"unit_{u['id']:04d}.wav"
        try:
            # Generated pace is set by the reference's audio-seconds per UTF-8
            # byte of its matching transcript, so a slow, pause-heavy reference
            # makes every Kannada line come out long. A short, naturally paced
            # reference plus a mild speed-up is the lever; s5_fit then closes
            # whatever gap remains, per unit, and guarantees the track fits.
            audio, _sr, _ = infer_process(rp, rt, text, model.ema_model,
                                          model.vocoder, mel_spec_type="vocos",
                                          speed=TTS_SPEED, device=device)
            a = _to_float(audio).flatten()
            if a.size == 0:
                raise RuntimeError("empty audio")
            peak = float(np.max(np.abs(a))) or 1.0
            a = (a / peak) * 0.85
            sf.write(str(wav_path), a, SR)
        except Exception as e:  # noqa: BLE001
            log(f"[s4] unit {u['id']} failed ({e}); silencing")
            sf.write(str(wav_path), np.zeros(int(SR * 0.2), dtype=np.float32), SR)
        if n % 10 == 0 or n == len(pending):
            log(f"[s4] tts {n}/{len(pending)}")
    return data


def synth_one(meta: dict, text: str, ref: dict, out_path: str,
              device: str = "cuda"):
    import numpy as np
    import soundfile as sf
    model = _load_model(device)
    a = _to_float(model(text, ref_audio_path=ref["ref_audio"],
                        ref_text=ref["ref_text"])).flatten()
    peak = float(np.max(np.abs(a))) or 1.0
    sf.write(out_path, (a / peak) * 0.85, SR)
    return out_path
