"""Local TTS engine wrapping Piper (Swedish voices).

Dual-voice: `host` uses sv_SE-nst-medium, `guest` uses sv_SE-lisa-medium when
available. Falls back to host voice if guest model is missing.

Synthesizes text -> WAV bytes (in-memory), transcodes to MP3 for streaming.
Includes SHA1 cache keyed on (text, voice).
"""
from __future__ import annotations

import hashlib
import io
import logging
import subprocess
import wave
from pathlib import Path
from typing import Optional

from piper import PiperVoice

LOG = logging.getLogger("radio.tts")

ROOT = Path(__file__).resolve().parent
VOICE_DIR = ROOT / "voices"
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

VOICES = {
    "host":  VOICE_DIR / "sv_SE-nst-medium.onnx",
    "guest": VOICE_DIR / "sv_SE-lisa-medium.onnx",
}


class TTSEngine:
    def __init__(self, mp3_bitrate: str = "64k"):
        self.mp3_bitrate = mp3_bitrate
        self.voices: dict[str, PiperVoice] = {}
        self.total_synth_seconds = 0.0
        for key, path in VOICES.items():
            if path.exists():
                LOG.info("loading piper voice %s -> %s", key, path.name)
                self.voices[key] = PiperVoice.load(str(path))
            else:
                LOG.warning("voice missing: %s (%s)", key, path)
        if not self.voices:
            raise FileNotFoundError("no piper voices found in voices/")
        if "host" not in self.voices:
            # promote any voice to host
            first = next(iter(self.voices))
            self.voices["host"] = self.voices[first]

    def _cache_path(self, text: str, voice: str) -> Path:
        h = hashlib.sha1((voice + "::" + text.strip()).encode("utf-8")).hexdigest()[:16]
        return CACHE_DIR / f"{voice}_{h}.mp3"

    def synthesize(self, text: str, voice: str = "host") -> tuple[bytes, float]:
        text = (text or "").strip()
        if not text:
            return b"", 0.0
        v = self.voices.get(voice) or self.voices["host"]
        cache = self._cache_path(text, voice)
        if cache.exists():
            data = cache.read_bytes()
            return data, self._mp3_duration_estimate(len(data))

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            v.synthesize_wav(text, wf)
        wav_bytes = wav_buf.getvalue()

        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as rf:
                frames = rf.getnframes()
                rate = rf.getframerate()
                duration = frames / float(rate)
        except Exception:
            duration = len(wav_bytes) / (22050 * 2.0)

        mp3 = self._wav_to_mp3(wav_bytes)
        cache.write_bytes(mp3)
        self.total_synth_seconds += duration
        return mp3, duration

    def _wav_to_mp3(self, wav_bytes: bytes) -> bytes:
        proc = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y",
             "-f", "wav", "-i", "pipe:0",
             "-c:a", "libmp3lame", "-b:a", self.mp3_bitrate,
             "-ac", "1", "-ar", "22050",
             "-f", "mp3", "pipe:1"],
            input=wav_bytes, capture_output=True, check=True,
        )
        return proc.stdout

    @staticmethod
    def _mp3_duration_estimate(num_bytes: int, bitrate_bps: int = 64000) -> float:
        return (num_bytes * 8) / bitrate_bps


_singleton: Optional[TTSEngine] = None


def get_engine() -> TTSEngine:
    global _singleton
    if _singleton is None:
        _singleton = TTSEngine()
    return _singleton


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    text = " ".join(sys.argv[1:]) or "Test av radio-systemet."
    eng = get_engine()
    for v in ["host", "guest"]:
        mp3, dur = eng.synthesize(text, voice=v)
        out = Path(f"/tmp/tts_{v}.mp3")
        out.write_bytes(mp3)
        print(f"{v}: {out} ({len(mp3)} bytes, ~{dur:.2f}s)")
