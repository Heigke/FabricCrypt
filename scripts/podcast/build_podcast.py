#!/usr/bin/env python3
"""Build Swedish NS-RAM podcast via OpenAI tts-1-hd."""
import os, re, sys, subprocess, time, pathlib
from openai import OpenAI

ROOT = pathlib.Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
SCRIPT = ROOT / "scripts/podcast/nsram_podcast_sv.txt"
OUT_DIR = ROOT / "tmp/nsram_podcast"
CHUNK_DIR = OUT_DIR / "chunks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# Load API key from .env (key=value, lowercase var)
env_path = ROOT / ".env"
for line in env_path.read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k.strip().upper()] = v.strip()
api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("openai_api_key")
assert api_key, "no key"
client = OpenAI(api_key=api_key)

VOICE = os.environ.get("PODCAST_VOICE", "nova")
MODEL = "tts-1-hd"
MAX_CHARS = 3800  # under 4096 hard limit

text = SCRIPT.read_text()

# Split by section headers (### ...) -> chunks
sections = re.split(r"\n(?=### )", text.strip())
print(f"[info] {len(sections)} sections, voice={VOICE}, model={MODEL}", flush=True)

def further_split(s, limit=MAX_CHARS):
    """If a section >limit chars, split at paragraph boundaries."""
    if len(s) <= limit:
        return [s]
    paras = s.split("\n\n")
    out, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 <= limit:
            cur = (cur + "\n\n" + p) if cur else p
        else:
            if cur:
                out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    return out

chunks = []
for s in sections:
    # Strip the "### HEADER" markers; they're for our reading, not TTS speech
    body = re.sub(r"^### .*\n", "", s).strip()
    if not body:
        continue
    chunks.extend(further_split(body))

print(f"[info] {len(chunks)} TTS chunks", flush=True)
for i, c in enumerate(chunks):
    print(f"  chunk {i:02d}: {len(c)} chars, {len(c.split())} words", flush=True)

# Generate each chunk
chunk_paths = []
t0 = time.time()
for i, c in enumerate(chunks):
    out_mp3 = CHUNK_DIR / f"chunk_{i:02d}.mp3"
    if out_mp3.exists() and out_mp3.stat().st_size > 1000:
        print(f"[skip] {out_mp3.name} exists", flush=True)
        chunk_paths.append(out_mp3)
        continue
    ts = time.time()
    with client.audio.speech.with_streaming_response.create(
        model=MODEL, voice=VOICE, input=c, response_format="mp3", speed=1.0
    ) as resp:
        resp.stream_to_file(str(out_mp3))
    dt = time.time() - ts
    print(f"[ok] chunk {i:02d} -> {out_mp3.name} ({out_mp3.stat().st_size} B, {dt:.1f}s)", flush=True)
    chunk_paths.append(out_mp3)

print(f"[info] all chunks done in {time.time()-t0:.1f}s", flush=True)

# Generate a short silence (800ms) for joining
silence = OUT_DIR / "silence_800ms.mp3"
if not silence.exists():
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
        "-t", "0.8", "-q:a", "9", "-acodec", "libmp3lame", str(silence)
    ], check=True, capture_output=True)

# Build concat list interleaved with silence between sections
concat_list = OUT_DIR / "concat.txt"
lines = []
for i, p in enumerate(chunk_paths):
    lines.append(f"file '{p}'")
    if i != len(chunk_paths) - 1:
        lines.append(f"file '{silence}'")
concat_list.write_text("\n".join(lines) + "\n")

# Re-encode (not -c copy) for clean joins / no clicks
final_mp3 = OUT_DIR / "podcast_ns-ram_sv.mp3"
subprocess.run([
    "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
    "-ar", "24000", "-ac", "1", "-b:a", "96k", "-acodec", "libmp3lame",
    str(final_mp3)
], check=True, capture_output=True)

# Probe duration
res = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "default=noprint_wrappers=1:nokey=1", str(final_mp3)],
                     capture_output=True, text=True)
dur = float(res.stdout.strip())
size = final_mp3.stat().st_size
print(f"\n=== FINAL ===")
print(f"path:  {final_mp3}")
print(f"dur:   {dur:.1f}s ({dur/60:.2f} min)")
print(f"size:  {size} B ({size/1024/1024:.2f} MB)")
print(f"voice: {VOICE}")
