#!/usr/bin/env python3
"""Generate the Swedish 'chippets kropp' sensor-walkthrough MP3 via OpenAI TTS."""
import re, subprocess
from pathlib import Path
from openai import OpenAI

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
SCRIPT = ROOT / "results/IDENTITY_H7_2026-06-09/h7_sensorer_narration_sv.md"
OUTDIR = ROOT / "results/IDENTITY_H7_2026-06-09"
CHUNKS = OUTDIR / "sensor_chunks"
FINAL = OUTDIR / "h7_sensorer_sv.mp3"
MODEL = "gpt-4o-mini-tts"      # newest steerable TTS; good Swedish
VOICE = "onyx"                 # calm explanatory
MAXC = 3800


def key():
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            if k.strip().lower() in ("openai_api_key", "openai_key"):
                return v.strip().strip('"').strip("'")
    raise RuntimeError("no openai key")


def chunk(text):
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    out, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 > MAXC:
            if cur: out.append(cur); cur = ""
        cur = (cur + "\n\n" + p) if cur else p
    if cur: out.append(cur)
    return out


def main():
    client = OpenAI(api_key=key())
    text = SCRIPT.read_text()
    parts = chunk(text)
    CHUNKS.mkdir(parents=True, exist_ok=True)
    files = []
    for i, p in enumerate(parts):
        fp = CHUNKS / f"part_{i:02d}.mp3"
        print(f"  TTS chunk {i+1}/{len(parts)} ({len(p)} chars)...", flush=True)
        with client.audio.speech.with_streaming_response.create(
            model=MODEL, voice=VOICE, input=p,
            instructions="Lugn, tydlig, pedagogisk svensk berättarröst. Naturligt tempo, som en kunnig forskare som förklarar för en nyfiken kollega."
        ) as resp:
            resp.stream_to_file(fp)
        files.append(fp)
    # concat
    lst = CHUNKS / "list.txt"
    lst.write_text("\n".join(f"file '{f}'" for f in files))
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(FINAL)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=noprint_wrappers=1:nokey=1", str(FINAL)],
                         capture_output=True, text=True).stdout.strip()
    print(f"\n>>> saved {FINAL}  ({float(dur):.0f}s, {len(parts)} chunks)")


if __name__ == "__main__":
    main()
