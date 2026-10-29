# Kannada Dubbing Pipeline — Implementation Guide

**Project:** `F:\AI Projects\kannada-dub`
**Purpose:** Turn an English YouTube video into a Kannada-dubbed video where the Kannada
narration is spoken **in the original narrator's cloned voice**, timed to the original
speech, mixed over the original background music.

**Status: working end-to-end.** Verified output:

```
out/c__DMV_Clog_kn.mp4      115.5 MB
  video : vp9   1920x1080
  audio : aac   44100 Hz stereo
  length: 291.71 s   (source 291.80 s — in sync)
```

Source: *"Why Every Couple Struggles With Closeness and Distance"* (291.8 s, 68 speech units).

---

## 1. How it works — data flow

```
YouTube URL
   │
   │  s1_fetch      yt-dlp download + ffmpeg audio extraction
   ▼
work/<id>/source.info.json, asr_16k.wav, bed_44k.wav
   │
   │  s2_asr        faster-whisper, word-level timestamps
   ▼
segments.json      68 timed English speech units
   │
   │  s3_mt         local Ollama LLM (gemma4:12b) translation
   ▼
units.json         each unit: {start, dur, en, kn, kn_fit}
   │
   │  ├── s4_voice  demucs splits vocals/bed → build reference clip
   │  │             IndicF5 zero-shot clones the narrator, speaks the Kannada
   ▼  ▼
tts/unit_XXXX.wav   one Kannada wav per unit, in the narrator's voice
   │
   │  s5_fit        rubberband tempo-stretch each clip into its time slot
   ▼
narration.wav      single continuous Kannada narration track
   │
   │  s6_mix        duck original bed under narration, mux onto video
   ▼
out/<id>_kn.mp4  +  _kn.srt (Kannada subs)  +  _en.srt (English subs)
```

Every stage is **resumable**: artifacts are JSON/wav files on disk, and a stage refuses to
redo work that already exists (e.g. `synth()` only processes units with no wav yet).

---

## 2. Layout

| File | Lines | Role |
|---|---|---|
| `dub.py` | 88 | CLI entry point; orchestrates stages, lazy-imports heavy ones |
| `pipeline/common.py` | 128 | Paths, ffmpeg/ffprobe wrappers, JSON I/O, **audio load/trim helpers** |
| `pipeline/s1_fetch.py` | 71 | Download, `meta.json`, speech/bed audio extraction |
| `pipeline/s2_asr.py` | 94 | Whisper transcription → timed units |
| `pipeline/s3_mt.py` | 225 | Translation via Ollama, duration budgets, retries |
| `pipeline/s4_voice.py` | 239 | Reference extraction + IndicF5 voice cloning |
| `pipeline/s5_fit.py` | 79 | rubberband time-fitting into slots |
| `pipeline/s6_mix.py` | 61 | Ducking, muxing, subtitle generation |
| `pipeline/s7_post.py` | 121 | Denoise, EQ, de-ess, compression, EBU R128 loudness |
| `verify.py` | 99 | Timing/alignment verification |
| `scripts/prepare_tts_src.py` | 106 | Vendors + patches the `f5_tts` source tree |
| `scripts/verify_tts.py` | 105 | Isolated clone test with Whisper read-back |
| `scripts/smoke_tts.py` | 78 | Quick TTS smoke test |
| `scripts/diag_*.py` | 44–81 | One-off diagnostics (Ollama, weights, audio) |
| `tts_src/f5_tts/` | — | Vendored (patched) F5-TTS source |
| `work/<video_id>/` | — | All intermediates for one video |
| `out/` | — | Final videos + subtitles |
| `refs/` | — | Narrator reference clips |

---

## 3. Stage detail

### s1_fetch — download and split audio
`yt-dlp` fetches ≤1080p video, then ffmpeg produces:
- `asr_16k.wav` — mono 16 kHz, for Whisper
- `bed_44k.wav` — the audio that becomes the background bed

Writes `meta.json`: `{id, title, uploader, url, duration, source}`. The `id` is the key that
names every later artifact.

### s2_asr — transcript with timings
`faster-whisper` with word-level timestamps. Model/device are CLI-selectable
(`--whisper small.en|large-v3`, `--compute float16|int8`). Segments are merged into
**speech units** — one dub-able line each — giving the `{start, dur, en}` skeleton.

For this video: 55 Whisper segments → **68 units**, median 2.74 s.

### s3_mt — translation
Local **Ollama `gemma4:12b`**, one request per unit, with the unit's duration attached so the
prompt asks for text that can be spoken in that time. Output goes to `units.json` as `kn`.
Validates the returned Kannada (rejects empties / English leakage), retries failures, and
**saves partial results every batch** so a long run can't lose everything.

### s4_voice — the voice cloning core
Four jobs:

1. **`separate(meta)`** — Demucs `htdemucs --two-stems=vocals` isolates the narrator's vocal
   from music/SFX. The vocal becomes the cloning reference; the rest is the bed.
2. **`build_ref(meta, data, stems)`** — picks a **short clean reference clip** (`REF_SECONDS =
   (2.5, 4.5)`) and pairs it with its matching transcript. Both halves matter: a clip whose
   `ref_text` doesn't match its audio produces babble, not speech.
3. **Weight repair** — the published `model.safetensors` stores keys prefixed
   `ema_model._orig_mod.*`; the model expects `transformer.*`. Loading with `strict=False`
   silently matched **0 of 364 tensors**, leaving the DiT randomly initialised. `_repair_weights`
   strips the prefix and *refuses to continue on a partial load*.
4. **`synth(meta, data, ref)`** — for each pending unit, calls `infer_process(...)` with the
   reference (zero-shot clone) and the Kannada text. Result → `tts/unit_XXXX.wav` at 24 kHz.

### s5_fit — make it fit the timeline
Each clip is placed at its original `start`; rubberband `tempo=` adjusts its speed to fit the
unit's slot (`MAX_TEMPO`-clamped, `GAP` breathing room). Everything is laid onto one
continuous `narration.wav`. Writes `fit.json`, reporting **`overflow` per unit** — how far each
line missed its slot — so quality is measurable, not vibes.

For this video: **41 units hit the speed cap, max overflow 2.18 s.**

### s7_post — making the voice clear
The stage that did not exist at first, and the one that answers "the voice is not
clear / there is noise". Everything before it does crude peak normalisation only.
The chain is `highpass` (drop sub-85 Hz rumble) → `afftdn` (gentle FFT denoise)
→ `equalizer` (presence lift at 3 kHz, where Kannada consonant distinctions
live) → `deesser` (tame harsh sibilants) → `acompressor` (even out level
differences between units) → **two-pass EBU R128 loudness to -14 LUFS**. Two
passes because single-pass loudnorm is a guess; pass 1 measures, pass 2 applies
the measured values. Measured effect: **-21.1 → -14.5 LUFS, LRA 5.1 → 2.9**.

### s6_mix — final assembly
Ducks the bed under the Kannada narration (sidechain/volume automation), mixes narration +
bed, and muxes back onto the untouched video stream. Emits `out/<id>_kn.mp4`, plus Kannada
and English `.srt`. `--burn-subs` burns them in; `--keep-original-audio` overlays instead of ducking.

---

## 4. The six bugs that mattered

These are the reason the pipeline took real work to get running. Each was found by measurement.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | 35 min of MT produced **zero** translations | `gemma4:12b` is a *reasoning* model — it spent the entire 512-token budget on hidden thinking, so `content` came back empty (`done_reason=length`) | pass `"think": False` → 37 tokens, 3 s, clean Kannada |
| 2 | MT fallback would crash **after** a long run | `dub.py` passed the Ollama tag as the NLLB repo name → `from_pretrained("gemma4:12b")` | pass `s3_mt.NLLB` separately; save partial batches |
| 3 | TTS returned **NaN** audio | checkpoint keys `ema_model._orig_mod.*` vs expected `transformer.*` → 0/364 tensors loaded, DiT random | strip prefix, enforce `364/364`, fail loudly on partial |
| 4 | Cloned speech was **babble** | reference clip was 11 s of audio paired with a 35-word transcript — audio and text disagreed | short matched reference (`REF_SECONDS = (2.5, 4.5)`) |
| 5 | Synthesized audio far too long (13 s for a 3.8 s line) | `fix_duration` **collapses** generation (1.22 s for a 137-char line); F5's real duration is `ref_len + ref_len/ref_text_len*gen_len/speed`, so **the reference length is a floor on every line** | drop `fix_duration`, use `speed=1.5`, shorten the reference |
| 6 | Stages couldn't run separately | `data` was only built by `asr`/`mt`, so `--stages tts` had nothing; `--stages` also takes **commas**, not spaces | resume-load `units.json` in `dub.py`; `--stages tts,fit,mix` |

**Bonus (environment):** `librosa.effects.trim` → `librosa/decompose.py` → `import sklearn.decomposition`,
and Windows **Smart App Control** blocks `sklearn/utils/murmurhash.pyd`. Rather than fight it,
librosa was removed from our path entirely — `common.load_mono()` (soundfile + soxr/scipy) and
`common.trim_silence()` (numpy RMS) replace it. Same story for `torchaudio.load` →
`soundfile.read` inside the vendored `f5_tts`.

---

## 5. Running it

**Always** prefix the venv python with an empty `PYTHONPATH=` — the Hermes shell exports a
`PYTHONPATH` that shadows the venv — and disable TorchDynamo on Windows (no Triton):

```bash
cd "/f/AI Projects/kannada-dub"

# full pipeline, one command
PYTHONPATH= TORCHDYNAMO_DISABLE=1 .venv/Scripts/python.exe -u dub.py \
  --url "https://www.youtube.com/watch?v=<ID>" --stages fetch,asr,mt,tts,fit,mix --device cuda

# individual stages (comma-separated; reuses existing artifacts)
... dub.py --url "<url>" --stages tts,fit,mix --device cuda
... dub.py --url "<url>" --stages mt --mt gemma4:12b

# smoke test: synthesize only the first N units
... dub.py --url "<url>" --stages tts --limit 3 --device cuda
```

`--url` or `--file` is always required (it resolves the video id); earlier artifacts are reused,
so nothing is re-downloaded unless `fetch` is in `--stages`.

Useful flags: `--whisper large-v3` (accuracy), `--compute int8` (CPU), `--burn-subs`,
`--keep-original-audio`.

### Environment
- Python 3.11 venv at `.venv/` — `torch 2.11.0+cu128`, `transformers 4.49.0`, `numpy 1.26.4`,
  `faster-whisper`, `vocos`, `x-transformers`, `soundfile`
- GPU: RTX 4070 Laptop (CUDA). `TORCHDYNAMO_DISABLE=1` is required on Windows.
- Ollama running locally with `gemma4:12b` pulled
- Model: `6Morpheus6/IndicF5` (public mirror — `ai4bharat/IndicF5` is gated)

---

## 6. Verifying a result (don't trust the exit code)

```bash
# real streams + duration of the output
ffprobe -v error -show_entries format=duration \
  -show_entries stream=codec_type,codec_name,channels,sample_rate,width,height \
  -of default=noprint_wrappers=1 out/<id>_kn.mp4

# fit quality: per-unit tempo + overflow
grep -E "\[s5\]" work/<id>/full_run.log
```

Synthesis progress is `[s4] tts N/68` in the log; `scripts/verify_tts.py` runs an isolated
clone + Whisper read-back to confirm the audio is intelligible Kannada.

---

## 7. Known limitations

1. **Long Kannada lines overrun their slots.** 41 of 68 units hit the tempo cap, worst case
   2.18 s late. Kannada for a given English sentence is often longer, and the reference length
   sets a floor on F5's output. Options: merge sub-1.5 s Whisper fragments into fewer, longer
   units; allow a higher `MAX_TEMPO`; or accept slight drift.
2. **Very short units** (e.g. `1.02 s`, "on the other.") are Whisper splitting mid-sentence —
   they get compressed hard and can sound clipped.
3. **Smart App Control** can still intermittently block `caffe2_nvrtc.dll` on long GPU runs.
   Turning it off (Settings → Privacy & security → Windows security → App & browser control →
   Smart App Control → Off) removes that class of failure; it is one-way.
4. **Translation** is only as good as the local LLM; a stronger model or a reviewed glossary of
   terms would improve it. Units are currently translated independently, so cross-sentence flow
   relies on the LLM.
