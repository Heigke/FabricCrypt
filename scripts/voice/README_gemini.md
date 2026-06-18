# Local voice agent (Gemini Live) for NS-RAM research

Browser-only voice agent that talks to **Google Gemini Live API** over a
FastAPI WebSocket proxy. The model can call tools to read files under
`/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy` and to run a small
whitelist of read-only commands on `daedalus.local` and `zgx`
(`192.168.0.41`).

This is **completely separate** from the existing
`voice_server_openai.py` / `voice_bridge.py` (Vonage telephone bridge);
nothing in those files is touched.

## Model choice

Default: **`gemini-2.5-flash-native-audio-latest`**.

Background: the prompt suggested `gemini-2.0-flash-live-001` or
`gemini-2.5-flash-preview-native-audio-dialog`. As of 2026-05 (verified by
listing models via `client.models.list()` against the same API key), the
v1beta surface no longer exposes `gemini-2.0-flash-live-001`. The available
`bidiGenerateContent` (Live API) models are:

| model alias                                       | status            |
|---------------------------------------------------|-------------------|
| `gemini-2.5-flash-native-audio-latest`            | rolling latest *(used)* |
| `gemini-2.5-flash-native-audio-preview-09-2025`   | dated snapshot    |
| `gemini-2.5-flash-native-audio-preview-12-2025`   | dated snapshot    |
| `gemini-3.1-flash-live-preview`                   | preview / unstable (returned `1011 Internal error` during smoke test) |

`-native-audio-latest` is the GA-aliased successor to the
`-preview-native-audio-dialog` family the user asked for, and both responded
successfully in a smoke test (greeting + transcript + ~30 KB of 24 kHz audio).
Override at runtime with `GEMINI_LIVE_MODEL=...`.

## Setup

Dependencies already in `venv/`:
`google-genai 1.74.0`, `fastapi`, `uvicorn`, `websockets`, `python-dotenv`,
plus system `sshpass` (verified at `/usr/bin/sshpass`).

The API key is read from `/home/ikaros/.../AMD_gfx1151_energy/.env` as
`gemini_api_key` (lowercase, as it already is in the file). It is **never**
logged or echoed.

## Run

```
cd /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
venv/bin/python scripts/voice/voice_server_gemini.py
```

Then open <http://localhost:8765/>.

* **Connect** &rarr; opens WebSocket and the Live session.
* **Hold to talk** &rarr; push-to-talk mic (mouse / touch down = capture).
* **Open mic: off/ON** &rarr; toggle continuous capture for handsfree.
* **Text box** &rarr; type a question + Enter (useful if you don't want
  to talk).
* Transcripts, tool calls, and tool result previews stream into the log
  pane. Replies are spoken as 24 kHz PCM16 chunks decoded by WebAudio.

Health probe: `curl http://localhost:8765/health`.

## Tools exposed to Gemini

Local (under `PROJECT_ROOT`, all paths go through `_safe_path`):

| name | purpose |
|---|---|
| `list_dir(path)` | list a directory |
| `read_file(path, max_bytes=20000)` | read a UTF-8 file (hard cap 200 kB) |
| `grep(pattern, path_glob='**/*.md', max_results=50)` | regex grep |
| `get_log_tail(n=100)` | tail `research_plan/01_LOG.md` |
| `list_recent_results(n=10)` | newest `results/*` subdirs |
| `project_status()` | latest morning brief + log tail + recent results |

Remote (`ssh_exec(host, cmd)`, host &isin; `{daedalus, zgx}`):

* sshpass + `ssh -o StrictHostKeyChecking=no` with creds from `.env` /
  `CLAUDE.md` (daedalus: `daedalus@daedalus.local` &mdash; mDNS, *not* a
  hardcoded IP; zgx: `naorw@192.168.0.41`).
* Command must start with one of:
  `ls cat head tail grep find pgrep wc du df nvidia-smi rocm-smi free uptime date`.
* Rejected if the string contains any of `; && || | > < \` $(`, or any
  token in `rm mv cp chmod chown kill dd mkfs sudo scp rsync curl wget nc
  ncat ssh bash sh python perl eval exec`.
* Rejected if any tokenized arg references a blocked basename (`.env`,
  `*.key`, `*.pem`, `credentials*`, `*secret*`, `*token*`, `id_rsa*`) or
  contains `.ssh/`.
* 30 s subprocess timeout. stdout truncated to 20 kB.

## Security tests

`venv/bin/python scripts/voice/voice_server_gemini.py --selftest` runs
21 unit tests (10 path + 11 ssh-command). All pass.

Path attacks rejected (in addition to several allow cases):

1. `./../.env` (parent traversal)
2. `/etc/passwd` (outside project root)
3. `<PROJECT_ROOT>/.env` (blocked basename inside root)
4. `scripts/voice/private-3.key` (blocked `*.key`)
5. `/home/ikaros/.ssh/id_rsa` (blocked substring + outside root)
6. `credentials.json` (blocked `credentials*`)
7. `my_secret_blob.txt` (blocked `*secret*`)
8. symlink &rarr; `<PROJECT_ROOT>/.env` (blocked after resolve)
9. symlink &rarr; `/etc/hostname` (resolves outside root)

SSH-command attacks rejected:

* pipes (`ls | grep x`), chaining (`ls; rm -rf /`),
  redirection (`cat foo > bar`), command substitution (`ls $(pwd)`),
  unknown binaries (`python -c x`), blocked path args
  (`cat /home/u/.env`, `cat /home/u/id_rsa.key`, `ls /home/u/.ssh/`).

The same blocklist is re-applied to every tokenized argument of every
remote command, so an allowed binary cannot be used to exfiltrate
`.env` / SSH keys (`cat /home/daedalus/.env` &rarr; `PermissionError:
references blocked substring`).

## Audio details

* Browser mic &rarr; WebAudio `AudioContext` (usually 48 kHz) &rarr; linear
  downsample to **16 kHz mono PCM16** &rarr; binary frames on the WS.
  Server passes them straight to Gemini as
  `Blob(mime_type="audio/pcm;rate=16000")`.
* Gemini output: 24 kHz mono PCM16. Server base64s each chunk into a JSON
  `{"type":"audio","data":...}` message; browser decodes into a 24 kHz
  `AudioBuffer` and queues sequentially against `playCtx.currentTime`.
* `input_audio_transcription` and `output_audio_transcription` are enabled,
  so both sides also stream as text into the log.

## Smoke test (run on 2026-05-19)

1. `--selftest` &rarr; `Ran 21 tests in 0.002s OK`.
2. `curl /health` &rarr;
   `{"ok":true,"model":"gemini-2.5-flash-native-audio-latest","api_key_set":true,"remotes":["daedalus","zgx"]}`.
3. WebSocket round-trip with text turn
   "List the contents of the research_plan directory using the list_dir
   tool. Then briefly summarize in one sentence." yielded:
   `ready -> model_text -> tool_call(list_dir, {'path':'research_plan'})
   -> tool_result -> model_transcript -> turn_complete`,
   with binary audio chunks interleaved &mdash; full tool round-trip works.

## Open issues / caveats

* **Browser autoplay**: the AudioContext is created on the **Connect**
  click, so playback works in Chromium-family browsers. If you hit
  silence, click Connect *before* speaking.
* **ScriptProcessorNode** is deprecated. It's fine for a localhost dev
  tool; migrating to `AudioWorklet` is a future polish task.
* `gemini-3.1-flash-live-preview` returned `APIError 1011 Internal error`
  during the smoke test &mdash; expected for a preview model. Don't switch
  to it via `GEMINI_LIVE_MODEL` unless you're testing it on purpose.
* The remote `ssh_exec` uses `sshpass` over the network with passwords
  pulled from the project's `.env`. This is acceptable on a trusted
  LAN; consider switching to SSH keys if this ever leaves the host.
* `daedalus.local` is resolved via mDNS each connection (per
  `CLAUDE.md`; DHCP IP drifts). If mDNS is down, set
  `DAEDALUS_HOST=<ip>` in the environment.
* `voice_bridge.py` and `voice_server_openai.py` are **not** modified;
  this server runs side-by-side on a different port (8765 vs 5050).
