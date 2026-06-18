"""
Voice server: bridges Vonage WebSocket audio <-> Gemini Live API.

Endpoints:
  GET  /answer  -> Vonage Answer URL webhook, returns NCCO JSON
  POST /event   -> Vonage event webhook, log only
  WS   /ws      -> Vonage audio websocket (PCM L16 16kHz mono)

The websocket bridges:
  Vonage(16kHz PCM16 in)  -> Gemini Live (16kHz PCM16)
  Gemini Live (24kHz PCM16 out) -> resample -> Vonage (16kHz PCM16)

Tool calling: exposes `log_decision(decision: str)` which appends to
research_plan/01_LOG.md so the autonomous research loop can read it.

NEVER hardcodes credentials. Loads from .env at repo root.
"""

from __future__ import annotations

import asyncio
import audioop
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

LOG_PATH = REPO_ROOT / "research_plan" / "01_LOG.md"
RESULTS_DIR = REPO_ROOT / "results"

GEMINI_API_KEY = os.environ.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")
GEMINI_VOICE = os.environ.get("GEMINI_VOICE", "Puck")

WEBHOOK_BASE_URL = os.environ.get("WEBHOOK_BASE_URL", "")
PORT = int(os.environ.get("VOICE_SERVER_PORT", "5050"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("voice_server")

app = FastAPI(title="claude_hive voice bridge")


# ---------- helpers ----------------------------------------------------------

def _tail_lines(path: Path, n: int = 30) -> str:
    if not path.exists():
        return "(no log file yet)"
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
        return "".join(lines[-n:])
    except Exception as exc:
        return f"(failed to read log: {exc})"


def _latest_summary() -> str:
    if not RESULTS_DIR.exists():
        return "(no results dir)"
    try:
        summaries = sorted(
            RESULTS_DIR.glob("z*/summary.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not summaries:
            return "(no summary.json found)"
        latest = summaries[0]
        return f"{latest.relative_to(REPO_ROOT)}: {latest.read_text(errors='replace')[:1500]}"
    except Exception as exc:
        return f"(failed to read summaries: {exc})"


def _system_prompt() -> str:
    return (
        "Du är en röstassistent för Erics autonoma forskningsloop på AMD gfx1151 "
        "(reservoir computing, NS-RAM, FPGA). Tala primärt svenska, växla till "
        "engelska om Eric gör det. Var koncis och teknisk-folklig (mormor "
        "civilingenjör-registret). När Eric ger ett beslut, kalla verktyget "
        "`log_decision` med exakt beslutstexten så att forskningsloopen kan läsa "
        "den. Om Eric bara vill ha status: sammanfatta de sista loggraderna och "
        "senaste resultatet kort.\n\n"
        f"=== Senaste loggrader (01_LOG.md, tail 30) ===\n{_tail_lines(LOG_PATH, 30)}\n"
        f"=== Senaste resultatsammanfattning ===\n{_latest_summary()}\n"
    )


def _append_log_decision(decision: str) -> str:
    """Append a user-voiced decision to research_plan/01_LOG.md."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"\n[{ts}] VOICE_DECISION: {decision.strip()}\n"
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(line)
    log.info("Appended VOICE_DECISION to %s", LOG_PATH)
    return line.strip()


# ---------- Vonage HTTP webhooks --------------------------------------------

@app.get("/answer")
async def answer(request: Request) -> JSONResponse:
    """Vonage Answer URL — returns NCCO that connects call to our /ws."""
    host = WEBHOOK_BASE_URL or f"https://{request.headers.get('host', 'localhost')}"
    # NCCO needs wss:// URI
    ws_uri = host.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/ws"
    ncco = [
        {
            "action": "talk",
            "text": "Hej Eric, jag har en uppdatering från forskningsloopen. Ett ögonblick.",
            "language": "sv-SE",
        },
        {
            "action": "connect",
            "from": os.environ.get("VONAGE_FROM_NUMBER", "+46765195862"),
            "endpoint": [
                {
                    "type": "websocket",
                    "uri": ws_uri,
                    "content-type": "audio/l16;rate=16000",
                    "headers": {},
                }
            ],
        },
    ]
    log.info("Answer URL hit; returning NCCO with ws=%s", ws_uri)
    return JSONResponse(ncco)


@app.post("/event")
async def event(request: Request) -> dict:
    body: dict[str, Any]
    try:
        body = await request.json()
    except Exception:
        body = {"raw": (await request.body()).decode("utf-8", errors="replace")}
    log.info("Vonage event: %s", json.dumps(body)[:500])
    return {"ok": True}


@app.get("/health")
async def health() -> dict:
    return {
        "ok": True,
        "model": GEMINI_MODEL,
        "voice": GEMINI_VOICE,
        "webhook_base": WEBHOOK_BASE_URL,
        "log_exists": LOG_PATH.exists(),
    }


# ---------- WebSocket bridge ------------------------------------------------

# Gemini Live: 16k in, 24k out. Vonage wants 16k in/out.
GEMINI_IN_RATE = 16000
GEMINI_OUT_RATE = 24000
VONAGE_RATE = 16000


def _import_genai():
    from google import genai
    from google.genai import types
    return genai, types


def _build_live_config(types_mod):
    tool = types_mod.Tool(
        function_declarations=[
            types_mod.FunctionDeclaration(
                name="log_decision",
                description=(
                    "Append a user-voiced decision to research_plan/01_LOG.md so "
                    "the autonomous research loop can read it. Call this whenever "
                    "the user gives an instruction, decision, or new task."
                ),
                parameters=types_mod.Schema(
                    type=types_mod.Type.OBJECT,
                    properties={
                        "decision": types_mod.Schema(
                            type=types_mod.Type.STRING,
                            description="The decision or instruction text, verbatim.",
                        ),
                    },
                    required=["decision"],
                ),
            )
        ]
    )
    return types_mod.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types_mod.SpeechConfig(
            voice_config=types_mod.VoiceConfig(
                prebuilt_voice_config=types_mod.PrebuiltVoiceConfig(voice_name=GEMINI_VOICE),
            ),
            language_code="en-US",
        ),
        system_instruction=types_mod.Content(
            parts=[types_mod.Part(text=_system_prompt())]
        ),
        tools=[tool],
        realtime_input_config=types_mod.RealtimeInputConfig(
            automatic_activity_detection=types_mod.AutomaticActivityDetection(
                disabled=False,
                start_of_speech_sensitivity=types_mod.StartSensitivity.START_SENSITIVITY_LOW,
                end_of_speech_sensitivity=types_mod.EndSensitivity.END_SENSITIVITY_LOW,
                prefix_padding_ms=20,
                silence_duration_ms=500,
            ),
            activity_handling=types_mod.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
        ),
        output_audio_transcription=types_mod.AudioTranscriptionConfig(),
        input_audio_transcription=types_mod.AudioTranscriptionConfig(),
    )


@app.websocket("/ws")
async def vonage_ws(ws: WebSocket) -> None:
    await ws.accept()
    log.info("Vonage WS connected from %s", ws.client)

    if not GEMINI_API_KEY:
        log.error("GEMINI key missing; closing WS")
        await ws.close()
        return

    genai, types_mod = _import_genai()
    client = genai.Client(api_key=GEMINI_API_KEY, http_options={"api_version": "v1beta"})
    cfg = _build_live_config(types_mod)

    out_resample_state: Optional[Any] = None  # audioop.ratecv state

    try:
        async with client.aio.live.connect(model=GEMINI_MODEL, config=cfg) as session:
            log.info("Gemini Live session up (model=%s, voice=%s)", GEMINI_MODEL, GEMINI_VOICE)

            async def pump_vonage_to_gemini() -> None:
                inbound_bytes = 0
                inbound_frames = 0
                import time as _t
                t0 = _t.time()
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        log.info("Vonage WS disconnected (inbound %d frames / %d bytes over %.1fs)",
                                 inbound_frames, inbound_bytes, _t.time()-t0)
                        return
                    data = msg.get("bytes")
                    text = msg.get("text")
                    if text is not None:
                        log.info("Vonage control text: %s", text[:200])
                        continue
                    if not data:
                        continue
                    inbound_bytes += len(data)
                    inbound_frames += 1
                    if inbound_frames % 50 == 0:
                        log.info("inbound audio: %d frames, %d bytes", inbound_frames, inbound_bytes)
                    try:
                        await session.send_realtime_input(
                            audio=types_mod.Blob(data=data, mime_type=f"audio/pcm;rate={GEMINI_IN_RATE}")
                        )
                    except Exception as exc:
                        log.warning("send_realtime_input failed: %s", exc)
                        return

            async def pump_gemini_to_vonage() -> None:
                nonlocal out_resample_state
                async for response in session.receive():
                    # tool call?
                    tc = getattr(response, "tool_call", None)
                    if tc and getattr(tc, "function_calls", None):
                        for fc in tc.function_calls:
                            log.info("Gemini tool call: %s(%s)", fc.name, fc.args)
                            result_text = "ok"
                            if fc.name == "log_decision":
                                decision = (fc.args or {}).get("decision", "")
                                try:
                                    line = _append_log_decision(decision)
                                    result_text = f"logged: {line}"
                                except Exception as exc:
                                    result_text = f"error: {exc}"
                            try:
                                await session.send_tool_response(
                                    function_responses=[
                                        types_mod.FunctionResponse(
                                            id=getattr(fc, "id", None),
                                            name=fc.name,
                                            response={"result": result_text},
                                        )
                                    ]
                                )
                            except Exception as exc:
                                log.warning("send_tool_response failed: %s", exc)

                    # audio?
                    data = getattr(response, "data", None)
                    if data:
                        # data is 24kHz PCM16 LE mono from Gemini -> resample to 16k
                        try:
                            resampled, out_resample_state = audioop.ratecv(
                                data, 2, 1, GEMINI_OUT_RATE, VONAGE_RATE, out_resample_state
                            )
                            await ws.send_bytes(resampled)
                        except Exception as exc:
                            log.warning("downsample/send failed: %s", exc)
                            return

                    sc = getattr(response, "server_content", None)
                    if sc and getattr(sc, "turn_complete", False):
                        log.debug("Gemini turn complete")

            await asyncio.gather(
                pump_vonage_to_gemini(),
                pump_gemini_to_vonage(),
            )

    except WebSocketDisconnect:
        log.info("Vonage WS disconnect")
    except Exception as exc:
        log.exception("WS bridge error: %s", exc)
    finally:
        try:
            await ws.close()
        except Exception:
            pass
        log.info("WS bridge closed")


# ---------- main -------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "voice_server:app" if __package__ is None else f"{__package__}.voice_server:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )
