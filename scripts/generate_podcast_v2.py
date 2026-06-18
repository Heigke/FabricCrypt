"""Generate the long-form podcast (v2) with female voice (nova)."""
from __future__ import annotations
import os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"
OUT = ROOT / "results/podcast"

for line in ENV.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    os.environ[k.strip().upper()] = v.strip().strip('"').strip("'")

SCRIPT = (OUT / "feel_nsram_long_script.txt").read_text()

from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

print(f"[v2] script length: {len(SCRIPT)} chars; voice=nova (female); model=tts-1-hd")

# Split at paragraph boundaries (double newlines), max 4000 chars
chunks = []
remaining = SCRIPT.strip()
while remaining:
    if len(remaining) <= 4000:
        chunks.append(remaining)
        break
    split = remaining.rfind("\n\n", 0, 4000)
    if split < 1500:
        split = remaining.rfind(". ", 0, 4000)
    if split < 1500:
        split = 4000
    chunks.append(remaining[:split].strip())
    remaining = remaining[split:].lstrip()

print(f"[v2] split into {len(chunks)} chunks")

audio_parts = []
for i, ch in enumerate(chunks):
    print(f"  chunk {i+1}/{len(chunks)}: {len(ch)} chars", flush=True)
    t0 = time.time()
    resp = client.audio.speech.create(
        model="tts-1-hd",
        voice="nova",
        input=ch,
        response_format="mp3",
    )
    audio_parts.append(resp.content)
    print(f"    {len(resp.content)/1024:.0f} KB in {time.time()-t0:.1f}s", flush=True)

out_path = OUT / "feel_nsram_long_2026_05_03_nova.mp3"
with open(out_path, "wb") as f:
    for p in audio_parts:
        f.write(p)
print(f"\n[v2] saved: {out_path}")
print(f"[v2] total size: {out_path.stat().st_size/1024:.0f} KB")
