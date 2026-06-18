#!/usr/bin/env python3
"""Quick voice test for Swedish pronunciation."""
import os, sys, pathlib
from openai import OpenAI

ROOT = pathlib.Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
for line in (ROOT/".env").read_text().splitlines():
    if "=" in line and not line.strip().startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k.strip().upper()] = v.strip()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

text = ("Välkomna till podden. Idag pratar vi om NS-RAM, en liten kiselcell som "
        "beter sig nästan som en hjärncell. Vi börjar enkelt och bygger upp långsamt.")

voice = sys.argv[1] if len(sys.argv) > 1 else "nova"
out = ROOT / f"tmp/nsram_podcast/voice_test_{voice}.mp3"
out.parent.mkdir(parents=True, exist_ok=True)
with client.audio.speech.with_streaming_response.create(
    model="tts-1-hd", voice=voice, input=text, response_format="mp3"
) as r:
    r.stream_to_file(str(out))
print(f"ok: {out} ({out.stat().st_size} B)")
