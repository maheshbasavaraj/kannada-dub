"""Does an English prompt cause the garbled onset? Same Kannada text, 3 prompts."""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch  # noqa: F401
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path("tts_src").resolve()))
from pipeline import s4_voice as s4  # noqa: E402
from pipeline.common import load_json  # noqa: E402
from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text  # noqa: E402

VID = "c__DMV_Clog"
D = Path("work") / VID
segs = {x["id"]: x for x in load_json(D / "segments.json")["segments"]}
TEXT = segs[0].get("kn_fit") or segs[0]["kn"]
print("target:", TEXT)

model = s4._load_model("cuda")
PAN = s4.BUILTIN_VOICES["happy_f"][1]
pan_wav = hf_hub_download(s4.INDICF5, "prompts/PAN_F_HAPPY_00001.wav")
KN_REF = "ನಮಸ್ಕಾರ, ಇದು ಒಂದು ಸ್ಪಷ್ಟವಾದ ಕನ್ನಡ ಧ್ವನಿ ಮಾದರಿಯಾಗಿದೆ."


def gen(ref_audio, ref_text, tag):
    rp, rt = preprocess_ref_audio_text(ref_audio, ref_text)
    a, sr, _ = infer_process(rp, rt, TEXT, model.ema_model, model.vocoder,
                             mel_spec_type="vocos", speed=s4.TTS_SPEED, device="cuda")
    a = s4._to_float(a).flatten()
    p = D / f"_onset_{tag}.wav"
    sf.write(str(p), a, sr)
    print(f"{tag}: {a.size/sr:.2f}s -> {p}")
    return str(p)


narr = Path("refs") / f"{VID}_narrator.json"
n = load_json(narr)
p_en = gen(n["ref_audio"], n["ref_text"], "en")
p_pa = gen(pan_wav, PAN, "pa")
kn_ref_wav = gen(pan_wav, PAN, "knref_src")   # Kannada audio from the Indic prompt
# reuse that same Kannada clip (whose transcript we know) as the prompt
p_kn = gen(kn_ref_wav, KN_REF, "kn")

from faster_whisper import WhisperModel  # noqa: E402
m = WhisperModel("large-v3", device="cuda", compute_type="float16")
for tag, p in (("en", p_en), ("pa", p_pa), ("kn", p_kn)):
    segs2, info = m.transcribe(p, language="kn", word_timestamps=True)
    print(f"\n=== prompt={tag}  lang={info.language} p={info.language_probability:.2f}")
    for s in segs2:
        for w in (s.words or [])[:7]:
            print(f"   {w.start:5.2f}-{w.end:5.2f} {w.word.strip()!r}")
        break
