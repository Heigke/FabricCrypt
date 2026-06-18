# Voice bridge — autonomous research loop ↔ phone

Bridges the autonomous research campaign on this repo to the user's phone via
Vonage Voice + Gemini 2.0 Flash Live. The poller watches for kill-shot /
ambitious results, alerts, or idle gaps, and places an outbound call. The
server hands the call's audio to Gemini Live, which talks back to the user
(Swedish-primary, Puck voice) and can write decisions back to
`research_plan/01_LOG.md` via a tool call.

## Components

```
scripts/voice/
  voice_server.py    FastAPI + WS bridge (Vonage <-> Gemini Live)
  voice_bridge.py    poller / outbound-call dispatcher
  start_bridge.sh    boots cloudflared, server, poller
  stop_bridge.sh     clean shutdown
  logs/              runtime logs + .pid files (created at first run)
```

Costs: ~$0.10/min Gemini Live + ~$0.025/min Vonage Sweden mobile = **~$0.125/min total**.

## One-time setup

1. **Credentials** live in `.env` at repo root (already there, not committed):
   - `gemini_api_key`
   - `vonage_api_secret`

2. **Vonage application config** (hard-coded constants, override via env if needed):
   - `VONAGE_APPLICATION_ID = d4f497cc-a01f-40da-8218-be92c960580e`
   - `VONAGE_API_KEY = 43ae6f74`
   - `VONAGE_PRIVATE_KEY_PATH = scripts/private-2.key`
   - `VONAGE_FROM_NUMBER = +46765195862`
   - `USER_PHONE_NUMBER = +46704990616`

3. **Dependencies** (already installed in repo venv):
   ```
   venv/bin/pip install fastapi uvicorn websockets vonage google-genai python-dotenv
   ```

4. **Private key permissions** — `start_bridge.sh` chmods `scripts/private-2.key`
   to `600` automatically.

## Start

```
bash scripts/voice/start_bridge.sh
```

The script:
- installs `cloudflared` to `~/.local/bin/cloudflared` if missing,
- opens a `trycloudflare.com` tunnel to `localhost:5050`,
- starts `voice_server.py` on `:5050`,
- starts `voice_bridge.py` poller,
- prints the public URL.

Then **register the URL** in the Vonage dashboard:

> Voice → Applications → (your app) → Capabilities
> - Answer URL: `<URL>/answer`  (GET)
> - Event URL: `<URL>/event`    (POST)

Save. The tunnel URL changes each restart — re-paste it whenever you restart.

## Test it

```
venv/bin/python scripts/voice/voice_bridge.py --test
```

This places one outbound call to `USER_PHONE_NUMBER` immediately, ignoring
debounce. Answer the call; the NCCO `talk` action greets you ("Hej Eric…"),
then connects you to the Gemini Live websocket. Talk to it.

To exercise the tool path: say something like
*"Lägg till en task: kör z2300 med dt=0.005"*. Gemini should call
`log_decision`, and `research_plan/01_LOG.md` will receive a line:

```
[2026-05-15T18:42:11Z] VOICE_DECISION: kör z2300 med dt=0.005
```

## Triggers (poll every 30s)

- `results/z*/summary.json` newest contains `"KILL_SHOT": true`  → **emergency**, calls immediately
- … contains `"AMBITIOUS": true`                                 → calls (debounced)
- tail of `01_LOG.md` contains `BLOCKED on user`                 → **emergency**
- tail of `01_LOG.md` contains `ALERT`                           → calls (debounced)
- `01_LOG.md` mtime > 30 min ago                                 → idle call (debounced)

Debounce: at most one call per 10 min, except emergencies. Same trigger
hash will not re-fire within 6h.

State file: `/tmp/voice_bridge_state.json`.

## Stop

```
bash scripts/voice/stop_bridge.sh
```

Kills the three processes (PIDs in `scripts/voice/logs/*.pid`).

## Audio path

- Vonage → server: PCM16 LE mono **16 kHz** over WebSocket bytes frames.
- Server → Gemini Live: same data, sent as `Blob(mime_type="audio/pcm;rate=16000")`.
- Gemini Live → server: **24 kHz** PCM16 mono.
- Server → Vonage: downsampled to 16 kHz via `audioop.ratecv`.

Both directions are PCM16, so no float/byte-order conversion is needed —
only sample-rate conversion on the model-output side.

## Files written outside this folder

- Appends to `research_plan/01_LOG.md` (only when `log_decision` tool fires).
- Reads `results/z*/summary.json` (read-only).
- State file `/tmp/voice_bridge_state.json`.

## Known caveats

- The cloudflared free tunnel URL is ephemeral. Re-register it in Vonage on
  each restart, or move to a Cloudflare named tunnel later.
- Gemini Live `gemini-2.0-flash-exp` is the current model name; override via
  `GEMINI_LIVE_MODEL` env var if Google renames it (`gemini-live-2.5-flash-preview`
  etc.).
- Vonage WebSocket sends a JSON text frame at the start of the audio stream
  (call leg metadata) — the server logs it and ignores it.
- If audio is one-way, first check that Gemini-side output `data` chunks arrive
  in `voice_server.log` ("downsample/send failed" would indicate the back path).
