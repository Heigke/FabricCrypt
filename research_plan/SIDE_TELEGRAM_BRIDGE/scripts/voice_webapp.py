#!/usr/bin/env python3
"""
voice_webapp.py — Local server that mints ephemeral OpenAI Realtime API tokens
and serves a single-page WebRTC voice client.

Run:
    pip install fastapi uvicorn httpx
    export OPENAI_API_KEY=sk-...
    uvicorn voice_webapp:app --host 0.0.0.0 --port 8080

Open http://<this-host>:8080 on your phone (same Wi-Fi).
"""
from __future__ import annotations

import os
import pathlib
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

REALTIME_MODEL = os.environ.get("REALTIME_MODEL", "gpt-4o-realtime-preview-2024-12-17")
VOICE = os.environ.get("REALTIME_VOICE", "verse")  # verse, alloy, shimmer, echo, ash, ballad, coral, sage

HERE = pathlib.Path(__file__).resolve().parent
STATIC = HERE / "static"
LOG_PATH = HERE.parent / "01_LOG.md"  # research-context, optional

app = FastAPI(title="voice_webapp")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def _research_context(max_lines: int = 20) -> str:
    """Inject last N lines of 01_LOG.md (if present) into system prompt."""
    if not LOG_PATH.exists():
        return ""
    try:
        lines = LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
        tail = "\n".join(lines[-max_lines:])
        return f"\n\nCURRENT RESEARCH LOG (last {max_lines} lines):\n{tail}"
    except Exception:
        return ""


def _system_instructions() -> str:
    base = (
        "You are a research collaborator for the AMD gfx1151 / NS-RAM / FPGA "
        "energy-based computing project. Speak concisely. When the user asks for "
        "details, pull from the research log context. Reply in the user's language."
    )
    return base + _research_context()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC / "index.html"))


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": REALTIME_MODEL,
        "voice": VOICE,
        "has_key": bool(os.environ.get("OPENAI_API_KEY")),
        "log_present": LOG_PATH.exists(),
    }


@app.post("/session")
async def session() -> JSONResponse:
    """
    Mint an ephemeral Realtime session token.
    Returns the JSON from OpenAI directly (contains client_secret.value, valid ~60s).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server")

    payload = {
        "model": REALTIME_MODEL,
        "voice": VOICE,
        "modalities": ["audio", "text"],
        "instructions": _system_instructions(),
        "input_audio_transcription": {"model": "whisper-1"},
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500,
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "realtime=v1",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            "https://api.openai.com/v1/realtime/sessions",
            json=payload,
            headers=headers,
        )
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return JSONResponse(r.json())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("voice_webapp:app", host="0.0.0.0", port=8080, reload=False)
