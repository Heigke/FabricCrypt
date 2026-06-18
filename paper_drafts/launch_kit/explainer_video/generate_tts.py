#!/usr/bin/env python3
"""Generate OpenAI TTS MP3 for each narration section."""
import sys
from pathlib import Path
from openai import OpenAI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from narration_sections import SECTIONS  # noqa: E402

OUT = HERE / "audio"
OUT.mkdir(exist_ok=True)

VOICE = "nova"
MODEL = "tts-1-hd"


def load_api_key():
    env = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/.env")
    for line in env.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            if k.strip().lower() in ("openai_api_key", "openai_key"):
                return v.strip().strip('"').strip("'")
    raise RuntimeError("No OpenAI API key found")


def main():
    client = OpenAI(api_key=load_api_key())
    for name, text in SECTIONS:
        path = OUT / f"{name}.mp3"
        if path.exists() and path.stat().st_size > 1000:
            print(f"skip {name} (already exists)")
            continue
        print(f"tts {name} ({len(text)} chars)...")
        with client.audio.speech.with_streaming_response.create(
            model=MODEL, voice=VOICE, input=text, response_format="mp3"
        ) as r:
            r.stream_to_file(str(path))
        print(f"  -> {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
