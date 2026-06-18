# Continuous status radio — cost analysis + architecture
Date: 2026-05-30

## Premise
Local webapp där vi kan ansluta och lyssna på en kontinuerlig "radio-broadcaster" som beskriver status + bakgrund av pågående experiment. LLM genererar manus från real-time data (loggar, results-JSONs, task-notifications), TTS gör det till tal, frontend streamar.

## TTS-priser (Q2 2026 actual, double-checked)

| Provider | Model | $/1M chars | Quality | Streaming? |
|---|---|---|---|---|
| **Google Cloud TTS** | Standard | **$4** | OK för status-radio | ja |
| Google Cloud TTS | Neural2 | $16 | naturligt | ja |
| OpenAI | gpt-4o-mini-tts | ~$12 (token-priset) | mycket bra | ja |
| OpenAI | tts-1 | $15 | bra | ja |
| OpenAI | tts-1-hd | $30 | excellent | ja |
| Gemini | 2.5 Flash native TTS | ~$10 inferens + $0.50/M audio out | bra, preview | ja |
| ElevenLabs | Turbo v2.5 | ~$30-100 (tier-beroende) | excellent men dyrt | ja |
| **Lokalt** | piper-tts | **$0** | OK, ingen internet | ja (CPU snabbt) |

## Förbrukningsberäkning

Talet: ~150 ord/min × 5 tecken/ord = 750 chars/min = **45 000 chars/timme** för 100% airtime.

Realistiska scenarier:

| Scenario | Active hours/day | Chars/månad | Kostnad (Google Standard) | Kostnad (OpenAI tts-1) | Kostnad (Gemini Flash) |
|---|---|---|---|---|---|
| 100% 24h continuous | 24 | 32 M | **$128/mån** | $480/mån | $320/mån |
| 50% 24h (pauses) | 12 | 16 M | **$64/mån** | $240/mån | $160/mån |
| **8h active per dag** | 8 | 5.4 M | **$22/mån** | $81/mån | $54/mån |
| **Som vi sannolikt vill ha** (8h × 30% airtime, mest "live now" + tystnad) | 2.4 | 1.6 M | **$6.50/mån** | $24/mån | $16/mån |
| Lokalt piper-tts | obegränsat | obegränsat | **$0/mån** | — | — |

## LLM-kostnad för att generera manuset

Manus från status (loggar, results, task-notifications):
- Uppdatering var 3 min = 20/h × 8h × 30 dagar = **4 800 uppdateringar/mån**
- Per uppdatering: ~2 000 input tokens (status-snapshot) + 500 output tokens
- Total: **9.6M input + 2.4M output / mån**

| LLM | $/1M in / out | Kostnad/mån |
|---|---|---|
| Claude haiku 4.5 | $1 / $5 | **$22/mån** |
| Claude sonnet 4.6 | $3 / $15 | $65/mån |
| Gemini 2.5 Flash | $0.30 / $2.50 | **$9/mån** |
| GPT-5-mini | $0.40 / $1.60 | **$8/mån** |
| Lokalt llama (via Ollama) | $0 | **$0/mån** (men sämre kvalité) |

## TOTAL KOSTNAD — realistiska val

| Stack | TTS | LLM | Total/månad |
|---|---|---|---|
| **Billigast cloud**: Gemini Flash + Google TTS Standard | $6.50 | $9 | **~$16/mån** |
| **Bästa kvalité-pris**: OpenAI gpt-4o-mini-tts + GPT-5-mini | ~$20 | $8 | **~$28/mån** |
| **Premium**: ElevenLabs + Claude sonnet | ~$50 | $65 | ~$115/mån |
| **Helt lokalt**: piper-tts + Ollama llama | $0 | $0 | **$0** (sämre kvalité, viss latens) |

### Rekommenderat
**Gemini 2.5 Flash native TTS + Gemini Flash LLM = ~$15/mån** för "always-on" status-radio under 8h aktiva timmar/dag, 30% airtime. Ett system. En API-nyckel. Inga konstiga shim-lager.

Eller **piper-tts lokalt + Ollama** = **$0** men sämre röst och kräver att vi setup:ar Ollama.

## Arkitektur

```
┌──────────────────────────────────────────────────────────────┐
│ Backend (FastAPI på ikaros, port 8765 eller 8766)            │
├──────────────────────────────────────────────────────────────┤
│ poller_loop:                                                 │
│   var 30s eller på task-notification event                   │
│   → läs research_plan/01_LOG.md tail                         │
│   → läs senaste results/IDENTITY_BENCHMARK_2026-05-30/*.json │
│   → ps aux | grep körande experiment                         │
│   → hopa allt till compact_status                            │
│                                                              │
│ narrator_loop:                                               │
│   var 3 min:                                                 │
│   → skicka compact_status + tone-prompt till Gemini Flash    │
│   → få tillbaka 2-4 meningars "broadcaster"-manus            │
│   → skicka manuset till Gemini TTS                           │
│   → fa tillbaka mp3/opus-chunk                               │
│   → append till queue                                        │
│                                                              │
│ HTTP endpoints:                                              │
│   GET  /                  → HTML player                      │
│   GET  /stream.mp3        → audio chunk stream (icecast/hls) │
│   GET  /now               → SSE för "now playing" text       │
│   GET  /history           → JSON lista av senaste 50 manus   │
└──────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────┐
│ Frontend (statisk HTML, ~100 LOC)   │
├─────────────────────────────────────┤
│ <audio autoplay src=/stream.mp3>    │
│ <div id=now>...nuvarande text</div> │
│ <ul id=history>...tidigare</ul>     │
│ SSE → uppdatera now + history       │
└─────────────────────────────────────┘
```

## Reuse av befintlig kod
- `scripts/voice/voice_bridge.py` (Vonage outbound POC) — har TTS-hantering, kan kannibaliseras
- SIDE-B realtime webapp (task #240, OpenAI Realtime) — har webbplayer-skelett
- `scripts/oracle_dispatch.py` — har Gemini API-wrapper, kan återanvändas för manus-LLM

## Skydd mot kostnads-skenande
- Max-cap: $5/dag → om TTS-anrop översläver, paus 1h
- Manus-cache: identiska status → samma manus → cached audio, ingen ny TTS-anrop
- Pause-on-no-change: om status inte uppdaterats på 5 min → bara "*ingen ny aktivitet, väntar*" en gång, sen tystnad
- Volume-gating: bara säg "discovery" "fail" "thermal-alert" om något faktiskt hänt — annars kort ambient stämning

## Build-estimate
- ~250 LOC FastAPI backend
- ~50 LOC HTML frontend
- 2-3 timmar att bygga + testa
- Kan börja med piper-tts lokalt för att verifiera flow, sen swap till Gemini TTS

## Beslut till user
1. Vilken stack? (Gemini all-in cloud / lokalt piper / hybrid)
2. Aktiva timmar per dag? (8h normalt, 24h ambient, on-demand)
3. Bygga nu eller efter identitet-spåret landat?
