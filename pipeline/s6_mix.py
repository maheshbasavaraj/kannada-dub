"""Stage 6 - duck the original music/sfx under the Kannada narration, mux it
back onto the untouched video, and write subtitles."""
from __future__ import annotations

from pathlib import Path

from .common import MIX_SR, OUT, ensure_dirs, ffmpeg, log, write_srt


def mix(meta: dict, data: dict, stems: dict, fitted: dict,
        burn_subs: bool = False, keep_original_audio: bool = False) -> dict:
    d = Path(meta["source"]).parent
    narr = fitted["narration"]
    ensure_dirs(OUT)
    out_mp4 = OUT / f"{meta['id']}_kn.mp4"

    # subtitles (Kannada + original English for reference)
    kn_items, en_items = [], []
    for u in data["segments"]:
        txt = (u.get("kn_fit") or u.get("kn") or "").strip()
        if txt:
            kn_items.append((u["start"], u["end"], txt))
        if u.get("text"):
            en_items.append((u["start"], u["end"], u["text"]))
    kn_srt = OUT / f"{meta['id']}_kn.srt"
    en_srt = OUT / f"{meta['id']}_en.srt"
    write_srt(kn_srt, kn_items)
    write_srt(en_srt, en_items)

    # music bed: ducked under the voice, then normalised to a streaming target
    # so the finished file is not ~7dB quiet (measured -20.7 LUFS before).
    from .s7_post import LOUDNORM

    bed = Path(stems["no_vocals"])
    mixwav = d / "mix.wav"
    if keep_original_audio:
        ffmpeg(["-i", str(bed), "-i", narr, "-filter_complex",
                f"[0:a]volume=0.9[bed];[1:a]volume=1.0[n];"
                f"[bed][n]amix=inputs=2:weights='1 1':normalize=0,"
                f"alimiter=limit=0.95,{LOUDNORM},aresample={MIX_SR}[m]",
                "-map", "[m]", "-ar", str(MIX_SR), str(mixwav)])
    else:
        ffmpeg(["-i", str(bed), "-i", narr, "-filter_complex",
                "[1:a]asplit=2[n][sc];"
                "[0:a]volume=0.55[bed];"
                "[bed][sc]sidechaincompress=threshold=0.03:ratio=12:attack=15:"
                "release=350:makeup=1[duck];"
                "[duck][n]amix=inputs=2:weights='1 1':normalize=0,"
                f"alimiter=limit=0.95,{LOUDNORM},aresample={MIX_SR}[m]",
                "-map", "[m]", "-ar", str(MIX_SR), str(mixwav)])

    if burn_subs:
        ffmpeg(["-i", str(meta["source"]), "-i", str(mixwav),
                "-vf", f"subtitles={kn_srt.name}:force_style="
                       "'FontName=Nirmala UI,FontSize=20,Outline=2'",
                "-c:v", "libx264", "-crf", "20", "-preset", "medium",
                "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_mp4)],
               )
    else:
        ffmpeg(["-i", str(meta["source"]), "-i", str(mixwav),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_mp4)])

    log(f"[s6] wrote {out_mp4}")
    return {"video": str(out_mp4), "kn_srt": str(kn_srt), "en_srt": str(en_srt)}
