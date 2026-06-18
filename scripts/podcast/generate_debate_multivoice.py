"""Multi-voice debate podcast generator.

Reads a script with lines like `[SPEAKER]: text`, maps each speaker to an
OpenAI TTS voice, synthesizes per-line audio, concatenates with ffmpeg.

Usage:
    python3 generate_debate_multivoice.py SCRIPT_PATH [OUT_MP3]
"""
from __future__ import annotations
import os, re, sys, time, hashlib, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ENV = ROOT / ".env"
for line in ENV.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line: continue
    k, v = line.split("=", 1)
    os.environ[k.strip().upper()] = v.strip().strip('"').strip("'")

from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Speaker -> (model, voice, speed). gpt-4o-mini-tts supports the newer voices.
VOICE_MAP = {
    "Claude":     ("gpt-4o-mini-tts", "sage",    1.0),
    "GPT-5":      ("gpt-4o-mini-tts", "onyx",    1.0),
    "Gemini":     ("gpt-4o-mini-tts", "nova",    1.05),
    "Grok":       ("gpt-4o-mini-tts", "ash",     1.05),
    "DeepSeek":   ("gpt-4o-mini-tts", "echo",    1.0),
    "Skeptikern": ("gpt-4o-mini-tts", "coral",   1.0),
}

# Per-speaker instructions for gpt-4o-mini-tts to shape the persona
PERSONA = {
    "Claude":     "Lugn, balanserad moderator. Tydlig artikulation, vänlig men bestämd. Svenska.",
    "GPT-5":      "Auktoritativ, formell, lite pompös. Tala med tyngd. Svenska.",
    "Gemini":     "Skarp, snabb, lite kall. Klippt artikulation. Svenska.",
    "Grok":       "Uppkäftig, ironisk, internet-kadens. Lite snabb. Svenska.",
    "DeepSeek":   "Eftertänksam ingenjör. Talar lugnt och konkret. Svenska.",
    "Skeptikern": "Hård recensent. Tydlig, kritisk, lite hes. Svenska.",
}

LINE_RE = re.compile(r"^\[([^\]]+)\]\s*:\s*(.+)$")

def parse_script(text: str) -> list[tuple[str, str]]:
    lines = []
    current_speaker = None
    current_text = []
    for raw in text.splitlines():
        raw = raw.rstrip()
        if not raw:
            continue
        m = LINE_RE.match(raw)
        if m:
            if current_speaker is not None and current_text:
                lines.append((current_speaker, " ".join(current_text).strip()))
            current_speaker = m.group(1).strip()
            current_text = [m.group(2).strip()]
        else:
            if current_speaker is not None:
                current_text.append(raw.strip())
    if current_speaker is not None and current_text:
        lines.append((current_speaker, " ".join(current_text).strip()))
    return [(s, t) for s, t in lines if t]

def synth_line(speaker: str, text: str, cache_dir: Path) -> Path:
    if speaker not in VOICE_MAP:
        speaker = "Claude"
    model, voice, speed = VOICE_MAP[speaker]
    instructions = PERSONA[speaker]
    key = hashlib.sha1(f"{model}|{voice}|{speed}|{instructions}|{text}".encode()).hexdigest()[:20]
    out = cache_dir / f"{speaker}_{voice}_{key}.mp3"
    if out.exists() and out.stat().st_size > 100:
        return out
    kwargs = dict(model=model, voice=voice, input=text, response_format="mp3", speed=speed)
    if model == "gpt-4o-mini-tts":
        kwargs["instructions"] = instructions
    resp = client.audio.speech.create(**kwargs)
    out.write_bytes(resp.content)
    return out

def main():
    if len(sys.argv) < 2:
        print("usage: generate_debate_multivoice.py SCRIPT_PATH [OUT_MP3]")
        sys.exit(1)
    script_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results/podcast" / f"debate_{int(time.time())}.mp3"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    text = script_path.read_text()
    lines = parse_script(text)
    print(f"[parse] {len(lines)} lines, speakers: {sorted(set(s for s,_ in lines))}")

    cache_dir = ROOT / "tmp" / "podcast_debate_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    part_paths = []
    t_start = time.time()
    for i, (sp, txt) in enumerate(lines):
        # Truncate any single line to avoid TTS limits
        if len(txt) > 3500:
            txt = txt[:3500]
        t0 = time.time()
        try:
            p = synth_line(sp, txt, cache_dir)
        except Exception as e:
            print(f"  [{i+1}/{len(lines)}] {sp}: FAILED {e}")
            continue
        part_paths.append(p)
        print(f"  [{i+1}/{len(lines)}] {sp} ({len(txt)} chars) -> {p.stat().st_size//1024} KB in {time.time()-t0:.1f}s")

    # Concatenate via ffmpeg concat demuxer
    list_file = cache_dir / "concat_list.txt"
    with open(list_file, "w") as f:
        for p in part_paths:
            f.write(f"file '{p.absolute()}'\n")

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
           "-acodec", "libmp3lame", "-ab", "96k", str(out_path)]
    print(f"[ffmpeg] concatenating {len(part_paths)} parts -> {out_path}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("FFMPEG STDERR:", r.stderr[-2000:])
        sys.exit(2)

    sz = out_path.stat().st_size / 1024
    dur_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", str(out_path)]
    dur = subprocess.run(dur_cmd, capture_output=True, text=True).stdout.strip()
    print(f"\n[done] {out_path}")
    print(f"       size={sz:.0f} KB, duration={dur}s, total_synth_time={time.time()-t_start:.0f}s")

if __name__ == "__main__":
    main()
