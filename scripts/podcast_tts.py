#!/usr/bin/env python3
"""Generate Swedish NS-RAM podcast MP3 via OpenAI TTS."""
import os
import re
import sys
from pathlib import Path
from openai import OpenAI

OUT_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/podcast_2026-05-17")
CHUNKS_DIR = OUT_DIR / "chunks"
SCRIPT = OUT_DIR / "script.md"
FINAL = OUT_DIR / "nsram_podcast.mp3"
MAX_CHARS = 3500  # OpenAI TTS limit is 4096, leave headroom
VOICE = "nova"
MODEL = "tts-1-hd"

def load_api_key():
    env_path = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/.env")
    for line in env_path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            if k.strip().lower() in ("openai_api_key", "openai_key"):
                return v.strip().strip('"').strip("'")
    raise RuntimeError("No OpenAI API key found")

def clean_script(text):
    # Remove markdown headers, horizontal rules, but keep paragraph breaks
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            # Skip markdown headers (they become pauses via blank lines)
            continue
        if s == "---":
            continue
        lines.append(line)
    cleaned = "\n".join(lines)
    # Collapse 3+ newlines to 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

def chunk_text(text, max_chars=MAX_CHARS):
    """Split on paragraph boundaries, then sentence boundaries if needed."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= max_chars:
            current = (current + "\n\n" + p) if current else p
        else:
            if current:
                chunks.append(current)
            if len(p) <= max_chars:
                current = p
            else:
                # Split paragraph by sentences
                sentences = re.split(r"(?<=[.!?])\s+", p)
                current = ""
                for s in sentences:
                    if len(current) + len(s) + 1 <= max_chars:
                        current = (current + " " + s) if current else s
                    else:
                        if current:
                            chunks.append(current)
                        current = s
    if current:
        chunks.append(current)
    return chunks

def main():
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)
    raw = SCRIPT.read_text()
    cleaned = clean_script(raw)
    chunks = chunk_text(cleaned)
    print(f"Total chars: {len(cleaned)}")
    print(f"Chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"  chunk {i:02d}: {len(c)} chars")

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(chunks):
        out = CHUNKS_DIR / f"chunk_{i:02d}.mp3"
        if out.exists() and out.stat().st_size > 1000:
            print(f"skip {out.name} (already exists)")
            continue
        print(f"generating {out.name}...", flush=True)
        resp = client.audio.speech.create(
            model=MODEL,
            voice=VOICE,
            input=chunk,
        )
        with open(out, "wb") as f:
            f.write(resp.content)
        print(f"  wrote {out.stat().st_size} bytes")
    print("DONE")

if __name__ == "__main__":
    main()
