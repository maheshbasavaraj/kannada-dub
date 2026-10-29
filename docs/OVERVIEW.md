# Kannada Dubbing — What It Does & What It Runs On

Short version. Two parts: **features** implemented, and **models** used.

---

## 1. Features implemented

| # | Feature | What it does |
|---|---|---|
| 1 | **YouTube download** | Fetches any YouTube video up to 1080p (`yt-dlp`), or accepts a local file. |
| 2 | **Audio extraction** | Pulls two tracks from the source: mono **16 kHz** for speech recognition, and the full **44.1 kHz** audio that becomes the background bed. |
| 3 | **Vocal / music separation** | Splits the audio into narrator vocals and background (music + SFX), so the narrator's voice can be cloned and the music can be kept. |
| 4 | **Speech recognition with word timings** | Transcribes the English narration with word-level timestamps, then merges it into **speech units** — one dub-able line each, with a start time and duration. |
| 5 | **Kannada translation** | Translates every unit English → Kannada using a **local LLM**, with the unit's duration passed in so it asks for text that can be spoken in that time. Rejects empty/English output and retries. |
| 6 | **Zero-shot voice cloning** | Speaks the Kannada text in the **original narrator's voice** — no training needed, it learns the voice from a short reference clip taken from the source video. |
| 7 | **Automatic reference clip picking** | Chooses a short, clean stretch of the narrator's voice and pairs it with its matching transcript (both must agree or the output is gibberish). |
| 8 | **Time-fitting** | Each Kannada clip is placed at the original moment and speed-adjusted to fit its time slot, then laid onto one continuous narration track. |
| 9 | **Background music preserved** | The original music/SFX bed is **ducked** (lowered) under the Kannada narration rather than deleted, so the video keeps its atmosphere. |
| 10 | **Final mux + subtitles** | Audio is mixed back onto the video and written out as MP4, plus Kannada and English `.srt` subtitle files. Optionally burns subtitles into the picture. |
| 11 | **Resumable pipeline** | Every stage saves its output to disk, so a stage can be re-run alone and finished work is never redone. |
| 12 | **Smoke tests + verification** | Test scripts synthesize a few lines before a full run, and a verifier checks the cloned audio is intelligible (Whisper read-back) and the timing/overflow per line. |
| 13 | **CLI control** | One command for the whole pipeline, or run/finish individual stages; flags for model size, device (GPU/CPU), language and limits. |

**Pipeline order:** download → recognize → translate → clone voice → fit timing → mix → MP4.

---

## 2. Models used

| Purpose | Model | Notes |
|---|---|---|
| **Speech recognition** | **Whisper** via `faster-whisper` | Default `large-v3`; `small.en` used for fast passes. Runs on GPU (float16) or CPU (int8). |
| **Translation (EN → Kannada)** | **`gemma4:12b`** served locally by **Ollama** | Local, offline, no API keys. Must be called with thinking disabled, or the reasoning tokens eat the whole reply budget. An NLLB-200 path exists as an offline fallback. |
| **Vocal / background separation** | **Demucs** (`htdemucs`) | Two-stem mode: vocals vs. everything else. Provides both the cloning reference and the background bed. |
| **Voice cloning + speech synthesis** | **IndicF5** (F5-TTS architecture) | The core model. **Zero-shot** — clones the narrator from a few seconds of reference audio. Chosen because it handles **Kannada**; the official `ai4bharat/IndicF5` repo is gated, so the public `6Morpheus6/IndicF5` mirror is used. |
| **Vocoder (audio reconstruction)** | **Vocos** (`charactr/vocos-mel-24khz`) | Converts the model's mel-spectrogram into 24 kHz waveform audio. |
| **Speed/tempo fitting** | **FFmpeg + rubberband** | Per-line tempo change to land each clip inside its time slot without changing pitch. |
| **Audio I/O & resampling** | `soundfile`, `soxr`/`scipy` | Deliberately used instead of librosa, which drags in a library Windows security blocks on this machine. |

**Runtime:** local only — Python 3.11 venv, PyTorch 2.11 + CUDA on an **RTX 4070 Laptop** GPU. No cloud services, no API keys, nothing sent off the machine.

---

## 3. Verified output

| | |
|---|---|
| Source | *Why Every Couple Struggles With Closeness and Distance* — 291.8 s |
| Result | `out/c__DMV_Clog_kn.mp4` |
| Video | VP9, 1920×1080 |
| Audio | AAC, 44.1 kHz stereo (Kannada narration over ducked original bed) |
| Length | 291.71 s (in sync with the original) |
| Speech units | 68, all translated and synthesized in the cloned narrator voice |

---

## 4. Limits worth knowing

- **Long Kannada lines can overrun their slot.** 41 of 68 lines hit the speed cap, worst case 2.18 s late, because Kannada often needs more time than the English it replaces.
- **Very short units** (under ~1.5 s) are mid-sentence splits and can sound clipped.
- **Translation quality** depends on the local LLM; units are translated one at a time.
- **Voice cloning quality** depends on how clean the extracted reference clip is.
