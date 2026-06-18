# voice_webapp — real-time browser voice client (OpenAI Realtime API)

## Architecture

```
phone browser  ── WebRTC (audio + datachannel) ──>  OpenAI Realtime
     │                                                    ▲
     │ HTTPS /session  (ephemeral key, 60s)               │
     ▼                                                    │
ikaros:8080  ── POST /v1/realtime/sessions (real key) ────┘
```

The real `OPENAI_API_KEY` lives on ikaros only. The browser gets a 60-second
ephemeral `client_secret` which is what authenticates the SDP exchange.

## Files

- `scripts/voice_webapp.py` — FastAPI server: `/`, `/session`, `/health`
- `scripts/static/index.html` — single-page WebRTC client (vanilla JS)

## Setup

```bash
# in repo venv
source /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/venv/bin/activate
pip install fastapi uvicorn httpx

export OPENAI_API_KEY=sk-...

cd research_plan/SIDE_TELEGRAM_BRIDGE/scripts
uvicorn voice_webapp:app --host 0.0.0.0 --port 8080
```

Health check:

```bash
curl http://localhost:8080/health
# {"ok":true,"model":"gpt-4o-realtime-preview-2024-12-17","voice":"verse","has_key":true,...}
```

## Phone access (same Wi-Fi)

Find ikaros's LAN IP (`ip -4 addr show | grep inet`) and open
`http://<ikaros-lan-ip>:8080` on the phone. iOS Safari and Android Chrome both
need the page served over **plain HTTP from the LAN IP** (not mDNS hostname
unless your phone resolves `.local`) — mic permission works on `http://` for
private-IP origins.

If you need TLS or remote access:

```bash
# option A: cloudflared (free, no signup)
cloudflared tunnel --url http://localhost:8080

# option B: ngrok
ngrok http 8080
```

Use the resulting `https://…` URL on the phone.

## Expected latency

- WebRTC peer-to-peer to OpenAI edge: **150–350 ms** round-trip on good Wi-Fi.
- Server VAD turn end: +500 ms (`silence_duration_ms` setting).
- Total perceived: **~500–800 ms** from end-of-speech to AI audio start.

## Cost

- Input audio: $5 / 1M tokens, Output audio: $20 / 1M tokens
- ~1000 audio tokens/minute speaking; mixed conversation ≈ $0.06 /min
- **10 min/day ≈ $0.60/day, ~$18/month**

## Configurable

Environment variables:

- `REALTIME_MODEL` (default `gpt-4o-realtime-preview-2024-12-17`)
- `REALTIME_VOICE` (default `verse`; also: alloy, shimmer, echo, ash, ballad, coral, sage)

System prompt auto-injects the last 20 lines of
`research_plan/SIDE_TELEGRAM_BRIDGE/01_LOG.md` if present, so the model has
context on current research.

## Smoke test (no API call)

```bash
uvicorn voice_webapp:app --host 0.0.0.0 --port 8080 &
sleep 2
curl -s http://localhost:8080/health | grep -q '"ok":true' && echo "server OK"
curl -s http://localhost:8080/ | grep -q "gfx1151 voice"  && echo "html OK"
kill %1
```

`/session` is **not** tested offline because it makes a real OpenAI call.

## Notes

- Mic perms on iOS: must be a user gesture (the "Start Conversation" tap is one).
- Audio playback element has `autoplay playsinline` — required for iOS.
- Stop button properly closes PC + DataChannel + releases mic tracks.
