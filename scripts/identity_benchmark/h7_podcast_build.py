"""Build the H7 pedagogical podcast from h7_podcast_script.txt using piper (CPU, no GPU contention).

M: -> lisa (host), E: -> nst (expert). Uses the piper Python API to grab FLOAT audio so we can
apply ONE global peak-normalization (-> guaranteed no int16 clipping, which caused the earlier 'brus').
length-scale 0.95 for a brisk, non-draggy pace. Output: single clean mp3 via ffmpeg (no loudnorm).
"""
from __future__ import annotations
import subprocess, wave
from pathlib import Path
import numpy as np
from piper import PiperVoice
from piper.config import SynthesisConfig

ROOT = Path(__file__).resolve().parents[2]
VDIR = ROOT/"assets/piper_voices"
SCRIPT = ROOT/"scripts/identity_benchmark/h7_podcast_script.txt"
OUTDIR = ROOT/"results/IDENTITY_H7_2026-06-09"
MP3 = OUTDIR/"h7_embodiment_podcast_sv.mp3"
SR = 22050
GAP_TURN = 0.45
PEAK_TARGET = 0.84          # -1.5 dBFS headroom -> no clipping
LENGTH_SCALE = 0.95
NOISE_W = 0.7

SYN = SynthesisConfig(length_scale=LENGTH_SCALE, noise_w_scale=NOISE_W,
                      normalize_audio=False, volume=1.0)


def synth_float(voice: PiperVoice, text: str) -> np.ndarray:
    parts = [c.audio_float_array for c in voice.synthesize(text, SYN)]
    return np.concatenate(parts).astype(np.float32)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    voices = {"M": PiperVoice.load(str(VDIR/"sv_SE-lisa-medium.onnx")),
              "E": PiperVoice.load(str(VDIR/"sv_SE-nst-medium.onnx"))}
    lines = []
    for raw in SCRIPT.read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("#"): continue
        if s[:2] in ("M:", "E:"): lines.append((s[0], s[2:].strip()))
    print(f"{len(lines)} turns", flush=True)

    gap = np.zeros(int(SR*GAP_TURN), np.float32)
    segs = []
    for i, (spk, text) in enumerate(lines):
        audio = synth_float(voices[spk], text)
        # 5 ms fade in/out per turn to kill boundary clicks
        f = int(SR*0.005)
        if len(audio) > 2*f:
            audio[:f] *= np.linspace(0, 1, f, dtype=np.float32)
            audio[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
        segs.append(audio); segs.append(gap)
        if i % 10 == 0: print(f"  synth {i+1}/{len(lines)} ({spk}) peak={np.abs(audio).max():.3f}", flush=True)

    full = np.concatenate(segs)
    peak = float(np.abs(full).max())
    full *= (PEAK_TARGET/peak)                       # ONE global gain -> no clipping anywhere
    print(f"global peak {peak:.3f} -> scaled to {PEAK_TARGET}", flush=True)
    pcm = np.clip(np.round(full*32767.0), -32767, 32767).astype(np.int16)
    clipped = int(np.sum(np.abs(pcm) >= 32767))
    dur = len(pcm)/SR
    print(f"clipped samples after scale: {clipped}  dur={dur/60:.1f} min", flush=True)

    wav = OUTDIR/"_podcast_tmp.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    subprocess.run(["ffmpeg", "-y", "-i", str(wav), "-af", "highpass=f=45",
                    "-b:a", "192k", str(MP3)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wav.unlink()
    print(f">>> {MP3} ({MP3.stat().st_size//1024} KB)", flush=True)


if __name__ == "__main__":
    main()
