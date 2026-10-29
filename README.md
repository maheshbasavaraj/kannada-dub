# Kannada dub pipeline

English YouTube video in -> Kannada dubbed video out, lined up with the original
timing. Voice cloning is **optional**: it can match the original narrator, or use
a clean studio voice bundled with the model.

## Flow

| stage | module | what it does |
|-------|--------|--------------|
| fetch | `pipeline/s1_fetch.py` | yt-dlp download, split audio (16k mono for ASR, 44.1k stereo for the mix) |
| asr   | `pipeline/s2_asr.py` | faster-whisper word timestamps, grouped into sentence-sized "dubbing units" (<= 12 s) |
| mt    | `pipeline/s3_mt.py` | translation EN -> KN per unit via local Ollama, duration-aware, meaning-first prompt |
| ref   | `pipeline/s4_voice.py` | demucs two-stem split; pick the **cleanest short** narrator clip as the clone reference |
| tts   | `pipeline/s4_voice.py` | IndicF5 zero-shot clone, one wav per unit |
| fit   | `pipeline/s5_fit.py` | rubberband pitch-preserving tempo fit into each original speech slot, laid on one timeline |
| post  | `pipeline/s7_post.py` | **denoise + presence EQ + de-ess + compressor, then two-pass EBU R128 loudness to -14 LUFS** |
| mix   | `pipeline/s6_mix.py` | sidechain-duck the original music/sfx under the Kannada voice, mux onto the untouched video, write `.srt` |

## Run

```bash
cd "F:/AI Projects/kannada-dub"
PYTHONPATH= .venv/Scripts/python.exe dub.py --url "https://youtu.be/XXXX"          # full run
PYTHONPATH= .venv/Scripts/python.exe dub.py --url "..." --stages fetch,asr,mt      # stop early
PYTHONPATH= .venv/Scripts/python.exe dub.py --file clip.mp4 --burn-subs            # local file, hardsubs
PYTHONPATH= .venv/Scripts/python.exe dub.py --url "..." --limit 3 --stages tts     # smoke test 3 units

# after changing the reference, TTS params or post-processing:
PYTHONPATH= .venv/Scripts/python.exe dub.py --url "..." --stages tts,fit,post,mix
```

Useful flags:

| flag | effect |
|------|--------|
| `--voice narrator` | clone the original speaker (default) |
| `--voice happy_f` | use the clean studio voice bundled with IndicF5 — no cloning, so no source noise is inherited |
| `--no-denoise` | skip the FFT denoise in `post` (keeps EQ + loudness) |
| `--keep-original-audio` | overlay instead of ducking the bed |
| `--burn-subs` | burn the Kannada subtitles into the picture |

Every stage caches into `work/<video_id>/`, so re-runs are cheap. Delete the
stage's file to redo it (`segments.json`, `tts/`, `fit/`, `narration.wav`).

Outputs land in `out/<video_id>_kn.mp4` plus `_kn.srt` / `_en.srt`.

## Tuning the voice

Two knobs dominate perceived quality, and both are set in `pipeline/s4_voice.py`:

- **`REF_SECONDS`** (default `(2.5, 4.5)`) — the reference clip window. F5's
  duration is `ref_len + ratio*gen_len/speed`, so **the reference length is a
  floor under every synthesised line**. A long reference (e.g. 11 s) makes short
  lines impossible to fit and forces everything to be rushed. Keep it short.
- **`TTS_SPEED`** (default `1.3`) — delivery rate. Raising it masks a too-long
  reference but makes speech sound hurried; with a short reference it can stay
  calm.

The reference is also chosen for **cleanliness** (SNR), not just length: voice
character is cloned from it, so a noisy reference clones its noise into every
line. Changing `REF_SECONDS` automatically invalidates the cached reference.

## Notes

- `PYTHONPATH=` in front is required on this machine: Hermes exports its own
  site-packages, which otherwise shadows the project venv.
- `TORCHDYNAMO_DISABLE=1` avoids the missing Triton compiler on Windows.
- `fit_report.json` flags units that needed more than 1.40x speed-up (`tight`).
  Those are the ones to rewrite shorter in the translation if they sound rushed.
- Post-processing targets **-14 LUFS** (streaming normalisation target). Output
  measured -20.7 LUFS before this stage existed, which plays back ~7 dB quiet
  and reads as muffled.
- Voice clone consent: IndicF5's licence requires explicit permission from the
  speaker whose voice is cloned. This is a reason to prefer `--voice happy_f`
  when the narrator has not consented.
