"""FEEL Lab Radio - FastAPI server on port 8770.

- GET /            -> HTML player
- GET /stream.mp3  -> long-lived MP3 stream; concatenates queued narration mp3s
                     with brief silence between, ambient ping when nothing new
- GET /now         -> SSE: current narration text + status snapshot
- GET /history     -> JSON of last 50 narrations
- GET /state       -> JSON: budget, temp, queue size
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tts_engine import get_engine
from status_poller import snapshot, hashable_diff_key
from script_generator import generate

LOG = logging.getLogger("radio.server")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / "state.json"
TEMPLATE = (ROOT / "templates" / "index.html").read_text()

# Config
PORT = 8770
HOST = "0.0.0.0"
FULL_INTERVAL_S = 180   # full update every 3 min
AMBIENT_INTERVAL_S = 90  # ambient ping every 1.5 min when nothing changed
DAILY_BUDGET_S = 4 * 3600  # 4 hours of synth per day
HOT_PAUSE_C = 75.0
SILENCE_BETWEEN_S = 0.8
MAX_HISTORY = 50

# minimal MP3 frame of silence (~26ms at 64kbps 22050Hz). Pre-generated on first use.
_SILENCE_MP3: bytes = b""


def _silence_mp3() -> bytes:
    global _SILENCE_MP3
    if _SILENCE_MP3:
        return _SILENCE_MP3
    import subprocess
    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"anullsrc=r=22050:cl=mono",
         "-t", str(SILENCE_BETWEEN_S),
         "-c:a", "libmp3lame", "-b:a", "64k",
         "-f", "mp3", "pipe:1"],
        capture_output=True, check=True,
    )
    _SILENCE_MP3 = proc.stdout
    return _SILENCE_MP3


class RadioState:
    def __init__(self):
        self.history: deque[dict[str, Any]] = deque(maxlen=MAX_HISTORY)
        self.queue: deque[tuple[bytes, dict[str, Any]]] = deque(maxlen=20)
        self.subscribers: list[asyncio.Queue] = []
        self.budget_used_s: float = 0.0
        self.budget_day: str = time.strftime("%Y-%m-%d")
        self.last_full_ts: float = 0.0
        self.last_ambient_ts: float = 0.0
        self.last_diff_key: str | None = None
        self.last_snap: dict[str, Any] = {}
        self.current_text: str = "(initierar...)"
        self.current_mode: str = "full"
        self.load_state()

    def load_state(self):
        if STATE_FILE.exists():
            try:
                d = json.loads(STATE_FILE.read_text())
                if d.get("budget_day") == self.budget_day:
                    self.budget_used_s = d.get("budget_used_s", 0.0)
                hist = d.get("history", [])
                for h in hist[-MAX_HISTORY:]:
                    self.history.append(h)
            except Exception as e:
                LOG.warning("could not load state: %s", e)

    def save_state(self):
        try:
            STATE_FILE.write_text(json.dumps({
                "budget_day": self.budget_day,
                "budget_used_s": self.budget_used_s,
                "history": list(self.history),
            }, default=str))
        except Exception as e:
            LOG.warning("could not save state: %s", e)

    def reset_budget_if_new_day(self):
        today = time.strftime("%Y-%m-%d")
        if today != self.budget_day:
            self.budget_day = today
            self.budget_used_s = 0.0
            LOG.info("new day -> budget reset")

    async def broadcast(self, payload: dict[str, Any]):
        dead = []
        for q in self.subscribers:
            try:
                q.put_nowait(payload)
            except Exception:
                dead.append(q)
        for q in dead:
            self.subscribers.remove(q)


STATE = RadioState()


async def narrator_loop():
    """Background task: poll, decide full/ambient, synthesize, enqueue."""
    engine = get_engine()
    LOG.info("narrator loop started")
    while True:
        try:
            STATE.reset_budget_if_new_day()
            snap = snapshot()
            STATE.last_snap = snap

            apu = snap.get("apu_c", -1)
            if apu >= HOT_PAUSE_C:
                LOG.warning("APU hot %.1fC, pausing 60s", apu)
                await asyncio.sleep(60)
                continue

            if STATE.budget_used_s >= DAILY_BUDGET_S:
                LOG.info("daily budget exhausted (%.0fs), sleeping 5min", STATE.budget_used_s)
                await asyncio.sleep(300)
                continue

            now = time.time()
            cur_key = hashable_diff_key(snap)
            changed = STATE.last_diff_key != cur_key

            mode = None
            if changed or (now - STATE.last_full_ts) >= FULL_INTERVAL_S:
                mode = "full"
            elif (now - STATE.last_ambient_ts) >= AMBIENT_INTERVAL_S:
                mode = "ambient"

            if mode is None:
                await asyncio.sleep(10)
                continue

            text, used_mode = generate(snap, mode=mode,
                                       prev_key=STATE.last_diff_key,
                                       cur_key=cur_key)
            if not text:
                await asyncio.sleep(10)
                continue

            LOG.info("[%s] %s", used_mode, text[:100])

            mp3, dur = await asyncio.to_thread(engine.synthesize, text)
            STATE.budget_used_s += dur

            item = {
                "ts": now, "text": text, "mode": used_mode,
                "duration_s": round(dur, 2),
                "apu_c": apu,
            }
            STATE.history.append(item)
            STATE.queue.append((mp3, item))
            STATE.current_text = text
            STATE.current_mode = used_mode

            if used_mode == "full":
                STATE.last_full_ts = now
            STATE.last_ambient_ts = now
            STATE.last_diff_key = cur_key

            await STATE.broadcast({
                "text": text, "mode": used_mode,
                "ts": now, "duration_s": round(dur, 2),
                "apu_c": apu,
                "running_count": sum(snap.get("running", {}).values()),
                "budget_used_s": STATE.budget_used_s,
            })
            STATE.save_state()

            # pace by audio duration so the stream doesn't pile up
            await asyncio.sleep(max(dur + 1.0, 8.0))

        except Exception as e:
            LOG.exception("narrator loop error: %s", e)
            await asyncio.sleep(15)


app = FastAPI(title="FEEL Lab Radio")


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(narrator_loop())


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(TEMPLATE)


@app.get("/history")
async def history():
    return JSONResponse(list(STATE.history))


@app.get("/state")
async def state():
    return JSONResponse({
        "budget_used_s": STATE.budget_used_s,
        "budget_day": STATE.budget_day,
        "queue_size": len(STATE.queue),
        "subscribers": len(STATE.subscribers),
        "apu_c": STATE.last_snap.get("apu_c", -1),
        "current_text": STATE.current_text,
        "current_mode": STATE.current_mode,
    })


@app.get("/now")
async def now_sse(request: Request):
    q: asyncio.Queue = asyncio.Queue(maxsize=8)
    STATE.subscribers.append(q)

    async def gen():
        # emit current state immediately
        try:
            init = {
                "text": STATE.current_text, "mode": STATE.current_mode,
                "ts": time.time(),
                "apu_c": STATE.last_snap.get("apu_c", -1),
                "running_count": sum(STATE.last_snap.get("running", {}).values()),
                "budget_used_s": STATE.budget_used_s,
            }
            yield f"data: {json.dumps(init)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=20.0)
                    yield f"data: {json.dumps(payload, default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            if q in STATE.subscribers:
                STATE.subscribers.remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/stream.mp3")
async def stream_mp3(request: Request):
    """Long-lived MP3 stream. Plays queued narration sequentially with silence
    between. Newly subscribed clients receive only future audio (not history)."""
    sub_q: asyncio.Queue = asyncio.Queue(maxsize=32)

    # tap: when narrator appends, push mp3 bytes to this subscriber's queue
    last_seen = {"idx": len(STATE.queue)}

    async def feeder():
        sil = _silence_mp3()
        # initial silence so the audio element warms up
        await sub_q.put(sil)
        while True:
            if await request.is_disconnected():
                break
            # drain new queue items
            current_len = len(STATE.queue)
            # the deque is mutated by narrator; snapshot via list
            items = list(STATE.queue)
            if len(items) > last_seen["idx"]:
                new_items = items[last_seen["idx"]:]
                last_seen["idx"] = len(items)
                for mp3, _meta in new_items:
                    await sub_q.put(mp3)
                    await sub_q.put(sil)
            else:
                # push silence periodically so MP3 stream stays warm
                await sub_q.put(sil)
            await asyncio.sleep(1.0)

    feeder_task = asyncio.create_task(feeder())

    async def gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    chunk = await asyncio.wait_for(sub_q.get(), timeout=5.0)
                    yield chunk
                except asyncio.TimeoutError:
                    yield _silence_mp3()
        finally:
            feeder_task.cancel()

    headers = {
        "Cache-Control": "no-cache, no-store",
        "Content-Type": "audio/mpeg",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="audio/mpeg", headers=headers)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=HOST, port=PORT, log_level="info")
