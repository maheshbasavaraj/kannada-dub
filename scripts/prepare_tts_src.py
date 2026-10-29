"""Vendor a patched copy of the model repo's `f5_tts` source into tts_src/.

The published IndicF5 `model.py` does not match the `f5_tts` source shipped
beside it, and the source assumes a training environment we never use. Four
patches, all applied to a project-local copy so the HF cache stays untouched:

  1. load_model(): ckpt_path is required by the shipped signature, but model.py
     builds the model without one (it loads the state dict itself afterwards).
     Make it optional and skip checkpoint loading when absent.
  2. torchaudio.load -> soundfile: torchaudio >= 2.9 routes load() through
     TorchCodec, which ships no Windows wheels.
  3. Drop the training-only `Trainer` import from model/__init__.py. It pulls in
     `datasets`, which we never use.

Run:  PYTHONPATH= .venv/Scripts/python.exe scripts/prepare_tts_src.py
Idempotent: safe to re-run; re-copies from the Hub snapshot first.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_NAME = "f5_tts"
DEST = ROOT / "tts_src"

SNIP_OLD = """def load_model(
    model_cls,
    model_cfg,
    ckpt_path,
"""
SNIP_NEW = """def load_model(
    model_cls,
    model_cfg,
    ckpt_path=None,
"""
CALL_OLD = """    model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)
"""
CALL_NEW = """    if ckpt_path:
        model = load_checkpoint(model, ckpt_path, device, dtype=dtype, use_ema=use_ema)
"""
# torchaudio >= 2.9 routes torchaudio.load through TorchCodec, which has no
# Windows wheels. soundfile reads the same files with no extra dependency.
AUDIO_OLD = """    audio, sr = torchaudio.load(ref_audio)
"""
AUDIO_NEW = """    import soundfile as _sf
    _data, sr = _sf.read(str(ref_audio), dtype="float32", always_2d=True)
    audio = torch.from_numpy(_data.T).contiguous()
"""
TRAINER_IMPORT = "from f5_tts.model.trainer import Trainer\n"


def patch_utils_infer(target: Path, notes: list) -> None:
    f = target / "infer" / "utils_infer.py"
    text = f.read_text(encoding="utf-8")
    for old, new, note, marker in (
        (SNIP_OLD, SNIP_NEW, "load_model() ckpt_path optional", "ckpt_path=None,"),
        (CALL_OLD, CALL_NEW, "skip load_checkpoint when no ckpt_path", "if ckpt_path:"),
        (AUDIO_OLD, AUDIO_NEW, "torchaudio.load -> soundfile", "soundfile as _sf"),
    ):
        if marker in text:
            continue
        if old not in text:
            raise SystemExit(f"anchor missing for: {note} - f5_tts layout changed")
        text = text.replace(old, new, 1)
        notes.append(note)
    f.write_text(text, encoding="utf-8")


def patch_model_init(target: Path, notes: list) -> None:
    f = target / "model" / "__init__.py"
    text = f.read_text(encoding="utf-8")
    if TRAINER_IMPORT in text:
        text = text.replace(TRAINER_IMPORT, "")
        text = text.replace(', "Trainer"]', "]").replace('"Trainer", ', "")
        notes.append("dropped training-only Trainer import (needs datasets)")
        f.write_text(text, encoding="utf-8")


def main() -> int:
    from huggingface_hub import snapshot_download

    repo = sys.argv[1] if len(sys.argv) > 1 else "6Morpheus6/IndicF5"
    snap = Path(snapshot_download(repo))
    src = snap / SRC_NAME
    if not src.exists():
        print(f"no {SRC_NAME}/ in {snap}")
        return 1

    target = DEST / SRC_NAME
    if target.exists():
        shutil.rmtree(target)
    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, target)
    print(f"copied {src} -> {target}")

    notes: list = []
    patch_utils_infer(target, notes)
    patch_model_init(target, notes)
    print("patched: " + ("; ".join(notes) if notes else "already up to date"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
