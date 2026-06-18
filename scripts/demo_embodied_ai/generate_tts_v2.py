"""Generate TTS audio per narration section using OpenAI tts-1-hd.

Loads OPENAI_API_KEY from .env (key name: openai_api_key).
Outputs paper_drafts/demo_video_v2/audio/audio_S{0..7}.mp3.
"""
from __future__ import annotations
import os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT  = REPO / 'paper_drafts' / 'demo_video_v2' / 'audio'
SCRIPTS_DIR = REPO / 'paper_drafts' / 'demo_video_v2'
OUT.mkdir(parents=True, exist_ok=True)

# Load API key from .env (custom format with lowercase keys)
def load_env_key():
    env_path = REPO / '.env'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if '=' in line and not line.strip().startswith('#'):
                k, v = line.split('=', 1)
                if k.strip().lower() == 'openai_api_key':
                    return v.strip().strip('"').strip("'")
    return os.environ.get('OPENAI_API_KEY')

API_KEY = load_env_key()
if not API_KEY:
    print('[tts] no OpenAI API key — aborting (no fallback installed)', file=sys.stderr)
    sys.exit(2)

from openai import OpenAI
client = OpenAI(api_key=API_KEY)

VOICE = 'nova'   # warm, clear, female
MODEL = 'tts-1-hd'

SECTIONS = [f'narration_S{i}.txt' for i in range(8)]

total_chars = 0
for i, fname in enumerate(SECTIONS):
    text = (SCRIPTS_DIR / fname).read_text().strip()
    out_path = OUT / f'audio_S{i}.mp3'
    if out_path.exists() and out_path.stat().st_size > 1000:
        print(f'[tts] skip S{i} (exists, {out_path.stat().st_size} bytes)')
        continue
    total_chars += len(text)
    print(f'[tts] S{i}: {len(text)} chars -> {out_path.name}')
    resp = client.audio.speech.create(model=MODEL, voice=VOICE, input=text)
    # New API: stream_to_file is deprecated; use .write_to_file or iter_bytes
    with open(out_path, 'wb') as fp:
        fp.write(resp.content)
    print(f'   {out_path.stat().st_size} bytes')

print(f'[tts] total {total_chars} chars  est cost ${total_chars * 30 / 1e6:.4f}')
