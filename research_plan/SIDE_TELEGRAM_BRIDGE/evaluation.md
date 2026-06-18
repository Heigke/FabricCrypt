# Mobile bridge — option evaluation

Goal: Eric (mobile) <-> Claude (ikaros) real-time bridge. Push from Claude,
ask from Eric, ideally voice-capable.

## Comparison

| Opt | Channel | Setup time | Cost | Voice? | Push? | Verdict |
|---|---|---|---|---|---|---|
| **A** | Telegram bot | 5 min | free (API) | yes (voice notes + Whisper + TTS) | yes, built-in | **recommended** |
| B | WhatsApp Business | days (Meta review) | ~$0.005/msg + setup | yes (voice notes) | yes | only if WhatsApp mandatory |
| C | Discord bot | 10 min | free | yes (voice channels, complex) | weak on mobile | worse mobile UX than Telegram |
| D | OpenAI Realtime API | hours (own client) | ~$0.06/min audio in + $0.24/min audio out | yes, ~300ms | needs Telegram/PWA shell | v2 upgrade for true conversation |
| E | Twilio SMS/voice | 30 min | $0.0075/SMS, $0.013/min PSTN | yes (PSTN call) | yes | only if SMS/phone-call needed |

## Why Telegram (A) wins for POC

- Free, no review, no business account
- Voice notes in/out work out of the box (just file upload/download)
- Mobile push is built-in to Telegram client
- Bot token is single secret, long-poll = no public webhook server needed
- Trivial to upgrade later: keep Telegram as transport, swap LLM backend
  to OpenAI Realtime API when Eric wants <1s voice round-trip

## Path B: WhatsApp

Blocked: Meta Business verification (1-3 days), template message approval
for proactive (push) messages outside 24-hour customer window. Cost ~$0.005
per "service" message. Don't pick unless Eric strongly prefers WhatsApp.

## Path D: OpenAI Realtime (v2 upgrade)

`gpt-4o-realtime-preview` over WebSocket:
- ~300ms speech-to-speech
- No Whisper/TTS round-trips
- Needs: persistent WS client + audio capture (PWA, native app, or bridge
  through Telegram by streaming audio chunks — non-trivial)
- Cost: roughly $0.06/min input, $0.24/min output audio
- Phase 2 work after Telegram POC is validated

## Path E: Twilio

Real PSTN call to Eric's phone (no internet needed). Useful for genuine
phone-call experience, especially when Eric is driving or has no data.
~$0.013/min outbound + $0.0085/min inbound. Skip unless mobile-data is
unreliable.

## Cost per research-day (estimate)

Assume 30 Eric<->Claude turns/day, ~200 input tokens + 400 output tokens
per turn, gpt-4o-mini text only:

- OpenAI gpt-4o-mini: 30 * (200 * $0.15/1M + 400 * $0.60/1M)
  = 30 * ($0.00003 + $0.00024) = **~$0.008/day** (under 1 cent)
- Gemini 2.0 Flash: free tier covers 30 turns easily; paid ~$0.075/1M input
  -> ~$0.005/day
- Grok-2: ~$2/1M input, $10/1M output -> 30 * (0.0004 + 0.004) = **~$0.13/day**
- DeepSeek-chat: ~$0.27/1M input, $1.10/1M output -> ~**$0.015/day**

Add voice:
- Whisper STT: $0.006/min -> 30 voice notes * 10s = 5min -> **$0.03/day**
- OpenAI TTS-1: $15/1M chars -> 30 * 400 chars = 12K chars -> **$0.18/day**

**Total realistic research-day cost: ~$0.20-0.25/day with full voice both ways.**
Text-only: under $0.01/day. Negligible.

## Recommendation

Build A (Telegram, this POC). When validated, add D (Realtime) as opt-in
`/live` command that opens a WS session for true conversational voice.
Skip B/C/E unless requirements change.
