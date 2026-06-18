# Telegram bridge — setup

Minimal mobile bridge so Claude (running on ikaros) can ping Eric on his phone,
and Eric can talk back via text or voice while away from the desk.

## What you get

- `/ask <question>` -> routes to OpenAI/Gemini/Grok/DeepSeek -> text reply
- Plain text or voice note -> same routing (voice notes are transcribed first)
- `/voice <question>` -> reply as audio (OpenAI TTS, sent as Telegram voice note)
- CLI push: `python telegram_bridge.py push "I have a question about X"` -> phone notification

## One-time setup (user actions)

1. **Create a bot** in Telegram:
   - Open Telegram, search `@BotFather`, send `/newbot`
   - Pick a name + username (must end in `bot`, e.g. `eric_claude_bridge_bot`)
   - BotFather replies with a token like `1234567890:AAH...`

2. **Add token to `.env`** (in repo root, alongside existing keys):
   ```
   TELEGRAM_BOT_TOKEN=1234567890:AAH...
   ```

3. **Find your chat_id** (needed for `push` mode):
   - Open Telegram, find your new bot, press Start, send any message.
   - Then in a terminal on ikaros:
     ```bash
     source venv/bin/activate
     pip install requests python-dotenv
     curl "https://api.telegram.org/bot$(grep TELEGRAM_BOT_TOKEN .env | cut -d= -f2)/getUpdates"
     ```
   - Look for `"chat":{"id":123456789,...}`. Copy that number.
   - Add to `.env`:
     ```
     TELEGRAM_CHAT_ID=123456789
     ```

4. **(Optional) pick default backend** in `.env`:
   ```
   BRIDGE_BACKEND=openai   # or gemini, grok, deepseek
   ```

## Running

**Serve mode** (foreground; daemonise later via systemd or `nohup &`):
```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 python research_plan/SIDE_TELEGRAM_BRIDGE/scripts/telegram_bridge.py serve --backend openai
```

**Push from Claude** (one-shot):
```bash
python research_plan/SIDE_TELEGRAM_BRIDGE/scripts/telegram_bridge.py push "z2213 done, 9/16 PASS — want me to start z2214?"
python research_plan/SIDE_TELEGRAM_BRIDGE/scripts/telegram_bridge.py push "Audio version" --voice
```

Eric replies in Telegram; serve loop catches the reply, routes to LLM, sends answer.

## Daemonise (optional)

```bash
# In a tmux pane or screen session:
tmux new -s bridge
HSA_OVERRIDE_GFX_VERSION=11.0.0 python research_plan/SIDE_TELEGRAM_BRIDGE/scripts/telegram_bridge.py serve
# Ctrl-b d to detach
```

Or via systemd user unit (not included; ~10 lines, add later).

## Security notes

- Bot is publicly addressable by `t.me/<username>`. Anyone who finds the
  username can DM it.
- For privacy, add a `chat_id` allow-list in `handle_message` — drop any
  message whose `chat.id` isn't yours.
- Token is in `.env` — already gitignored (verify).

## Latency expectations

| Path | Latency |
|---|---|
| Text in -> text out (OpenAI gpt-4o-mini) | 1.5-3s |
| Voice in -> text out (Whisper + LLM) | 3-5s |
| Voice in -> voice out (Whisper + LLM + TTS) | 5-8s |
| Push notification arrival | <1s after `sendMessage` returns |

For sub-second voice (true conversation), see `evaluation.md` -> option D
(OpenAI Realtime API). That's a v2 upgrade — Telegram POC first.

## Files

- `scripts/telegram_bridge.py` — POC, ~250 lines, single file, no Telegram SDK
- `evaluation.md` — Telegram vs WhatsApp vs Discord vs Realtime vs Twilio
