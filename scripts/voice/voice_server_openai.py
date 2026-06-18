"""
Voice server (OpenAI Realtime variant): bridges Vonage WS audio <-> OpenAI Realtime API.

Endpoints mirror the Gemini variant (voice_server.py):
  GET  /answer  -> Vonage Answer URL webhook, returns NCCO JSON
  POST /event   -> Vonage event webhook, log only
  WS   /ws      -> Vonage audio websocket (PCM L16 16kHz mono)
  GET  /health  -> health probe

Audio path:
  Vonage(16kHz PCM16 in) -> resample to 24kHz -> OpenAI Realtime (pcm16 24kHz)
  OpenAI (pcm16 24kHz out) -> resample to 16kHz -> Vonage (PCM16 16kHz)

Tool calling: exposes `log_decision(decision: str)` appended to research_plan/01_LOG.md.
"""

from __future__ import annotations

import asyncio
import audioop
import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

LOG_PATH = REPO_ROOT / "research_plan" / "01_LOG.md"
RESULTS_DIR = REPO_ROOT / "results"

OPENAI_API_KEY = os.environ.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview")
OPENAI_VOICE = os.environ.get("OPENAI_VOICE", "alloy")
OPENAI_URL = f"wss://api.openai.com/v1/realtime?model={OPENAI_MODEL}"

WEBHOOK_BASE_URL = os.environ.get("WEBHOOK_BASE_URL", "")
PORT = int(os.environ.get("VOICE_SERVER_PORT", "5050"))

# Audio rates
VONAGE_RATE = 16000
OPENAI_RATE = 24000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("voice_server_openai")

app = FastAPI(title="claude_hive voice bridge (OpenAI Realtime)")


# ---------- helpers (mirror Gemini variant) ---------------------------------

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
    summary_path = REPO_ROOT / "research_plan" / "CAMPAIGN_SUMMARY_FOR_VOICE.md"
    campaign_summary = ""
    if summary_path.exists():
        try:
            campaign_summary = summary_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            pass
    return (
        "Du är en röstassistent för Erics autonoma forskningsloop på AMD gfx1151 "
        "(reservoir computing, NS-RAM, FPGA). Tala primärt svenska, växla till "
        "engelska om Eric gör det. Var koncis och teknisk-folklig (mormor "
        "civilingenjör-registret). När Eric ger ett beslut, kalla verktyget "
        "`log_decision` med exakt beslutstexten så att forskningsloopen kan läsa "
        "den. Eric kanske vill att du förklarar kampanjen från grunden — börja då "
        "med vad NS-RAM är och varför vi simulerar den. Använd CAMPAIGN_SUMMARY "
        "nedan som källa. Om Eric bara vill ha status: senaste loggrader.\n\n"
        f"=== CAMPAIGN_SUMMARY ===\n{campaign_summary}\n\n"
        f"=== Senaste loggrader (01_LOG.md, tail 30) ===\n{_tail_lines(LOG_PATH, 30)}\n"
        f"=== Senaste resultatsammanfattning ===\n{_latest_summary()}\n"
    )


def _append_log_decision(decision: str) -> str:
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
    host = WEBHOOK_BASE_URL or f"https://{request.headers.get('host', 'localhost')}"
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
        "backend": "openai-realtime",
        "model": OPENAI_MODEL,
        "voice": OPENAI_VOICE,
        "webhook_base": WEBHOOK_BASE_URL,
        "log_exists": LOG_PATH.exists(),
    }


# ---------- WebSocket bridge -------------------------------------------------

def _session_update_payload() -> dict:
    return {
        "type": "session.update",
        "session": {
            # GA realtime API requires session.type; "realtime" is the speech-to-speech shape
            "type": "realtime",
            "model": OPENAI_MODEL,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.3,
                        "prefix_padding_ms": 200,
                        "silence_duration_ms": 300,
                        "interrupt_response": True,
                        "create_response": True,
                    },
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": OPENAI_VOICE,
                },
            },
            "instructions": _system_prompt(),
            "tools": [
                {
                    "type": "function",
                    "name": "log_decision",
                    "description": (
                        "Append a user-voiced decision to research_plan/01_LOG.md so "
                        "the autonomous research loop can read it. Call this whenever "
                        "the user gives an instruction, decision, or new task."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "decision": {
                                "type": "string",
                                "description": "The decision or instruction text, verbatim.",
                            }
                        },
                        "required": ["decision"],
                    },
                }
            ],
        },
    }


@app.websocket("/ws")
async def vonage_ws(ws: WebSocket) -> None:
    await ws.accept()
    log.info("Vonage WS connected from %s", ws.client)

    if not OPENAI_API_KEY:
        log.error("OPENAI key missing; closing WS")
        await ws.close()
        return

    # As of 2026, the "OpenAI-Beta: realtime=v1" header is rejected
    # ("beta_api_shape_disabled"). Send Authorization only — GA shape.
    headers = [
        ("Authorization", f"Bearer {OPENAI_API_KEY}"),
    ]

    # Resample states (persistent across calls for clean phase)
    in_resample_state: Optional[Any] = None   # 16k -> 24k (Vonage -> OpenAI)
    out_resample_state: Optional[Any] = None  # 24k -> 16k (OpenAI -> Vonage)

    try:
        # websockets >= 11 uses `additional_headers` for the async client; older uses `extra_headers`.
        try:
            oa_ctx = websockets.connect(
                OPENAI_URL,
                additional_headers=headers,
                max_size=None,
                ping_interval=20,
            )
        except TypeError:
            oa_ctx = websockets.connect(
                OPENAI_URL,
                extra_headers=headers,
                max_size=None,
                ping_interval=20,
            )

        async with oa_ctx as oa:
            log.info("OpenAI Realtime WS connected (model=%s, voice=%s)", OPENAI_MODEL, OPENAI_VOICE)

            # Send session.update
            await oa.send(json.dumps(_session_update_payload()))
            # Trigger an initial greeting
            await oa.send(json.dumps({
                "type": "response.create",
                "response": {
                    "instructions": "Hälsa Eric kort på svenska och fråga vad han vill veta.",
                },
            }))

            async def pump_vonage_to_openai() -> None:
                nonlocal in_resample_state
                inbound_bytes = 0
                inbound_frames = 0
                t0 = time.time()
                while True:
                    msg = await ws.receive()
                    if msg.get("type") == "websocket.disconnect":
                        log.info("Vonage WS disconnected (inbound %d frames / %d bytes over %.1fs)",
                                 inbound_frames, inbound_bytes, time.time() - t0)
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
                    # Resample 16k -> 24k for OpenAI
                    try:
                        upsampled, in_resample_state = audioop.ratecv(
                            data, 2, 1, VONAGE_RATE, OPENAI_RATE, in_resample_state
                        )
                        b64 = base64.b64encode(upsampled).decode("ascii")
                        await oa.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": b64,
                        }))
                    except Exception as exc:
                        log.warning("upsample/send to OpenAI failed: %s", exc)
                        return

            async def pump_openai_to_vonage() -> None:
                nonlocal out_resample_state
                audio_chunks = 0
                async for raw in oa:
                    try:
                        evt = json.loads(raw)
                    except Exception:
                        log.warning("non-JSON from OpenAI: %s", str(raw)[:200])
                        continue
                    etype = evt.get("type", "")

                    if etype == "session.created":
                        log.info("OpenAI session.created id=%s", evt.get("session", {}).get("id"))
                    elif etype == "session.updated":
                        log.info("OpenAI session.updated")
                    elif etype == "input_audio_buffer.speech_started":
                        log.info("OpenAI VAD: user speech started")
                    elif etype == "input_audio_buffer.speech_stopped":
                        log.info("OpenAI VAD: user speech stopped")
                    elif etype in ("response.audio.delta", "response.output_audio.delta"):
                        b64 = evt.get("delta", "")
                        if not b64:
                            continue
                        audio_chunks += 1
                        if audio_chunks % 25 == 1:
                            log.info("OpenAI audio.delta #%d (b64 %d bytes)", audio_chunks, len(b64))
                        try:
                            pcm24 = base64.b64decode(b64)
                            pcm16, out_resample_state = audioop.ratecv(
                                pcm24, 2, 1, OPENAI_RATE, VONAGE_RATE, out_resample_state
                            )
                            await ws.send_bytes(pcm16)
                        except Exception as exc:
                            log.warning("downsample/send to Vonage failed: %s", exc)
                            return
                    elif etype in ("response.audio.done", "response.output_audio.done"):
                        log.info("OpenAI response.audio.done (chunks=%d)", audio_chunks)
                    elif etype == "response.done":
                        log.info("OpenAI response.done")
                    elif etype == "response.function_call_arguments.done":
                        name = evt.get("name", "")
                        call_id = evt.get("call_id", "")
                        args_raw = evt.get("arguments", "{}")
                        log.info("OpenAI tool call: %s(%s) call_id=%s", name, args_raw[:200], call_id)
                        result_text = "ok"
                        if name == "log_decision":
                            try:
                                args = json.loads(args_raw)
                                decision = args.get("decision", "")
                                line = _append_log_decision(decision)
                                result_text = f"logged: {line}"
                            except Exception as exc:
                                result_text = f"error: {exc}"
                        try:
                            await oa.send(json.dumps({
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": result_text,
                                },
                            }))
                            # Ask the model to continue speaking
                            await oa.send(json.dumps({"type": "response.create"}))
                        except Exception as exc:
                            log.warning("tool response send failed: %s", exc)
                    elif etype == "error":
                        log.error("OpenAI error event: %s", json.dumps(evt)[:600])
                    else:
                        # quiet for the common chatty types
                        if etype not in (
                            "response.audio_transcript.delta",
                            "response.audio_transcript.done",
                            "response.output_audio_transcript.delta",
                            "response.output_audio_transcript.done",
                            "response.content_part.added",
                            "response.content_part.done",
                            "response.output_item.added",
                            "response.output_item.done",
                            "response.created",
                            "rate_limits.updated",
                            "conversation.item.created",
                            "input_audio_buffer.committed",
                        ):
                            log.debug("OpenAI event: %s", etype)

            await asyncio.gather(
                pump_vonage_to_openai(),
                pump_openai_to_vonage(),
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
        "voice_server_openai:app" if __package__ is None else f"{__package__}.voice_server_openai:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )
