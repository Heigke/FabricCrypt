# FabricCrypt Demo v2 (Narrated)

3-minute explainer with TTS narration that **explains the comparison** (which
v1 lacked: viewers saw "I am ikaros" -> later "false" with no context).

## What v2 fixes vs v1

- **v1**: 90s, no narration. Viewer must infer the meaning of green/red badges.
- **v2**: 186s, OpenAI tts-1-hd voiceover. Each step is explained: what the
  signature is, what transplant means, why the verifier rejects, why three
  attacks fail, and how it differs from Apple/NVIDIA/Intel/TPM.

## Reproduce

```bash
# 1. Generate TTS audio (requires OPENAI_API_KEY in .env as openai_api_key)
venv/bin/python scripts/demo_embodied_ai/generate_tts_v2.py

# 2. Build the video (with thermal pacing; ~3 min APU time)
nice -n 19 ionice -c 3 venv/bin/python scripts/demo_embodied_ai/build_demo_video_v2.py

# 3. (optional) Re-encode the Twitter cut
ffmpeg -y -i paper_drafts/demo_video_v2/fabriccrypt_v2_with_narration.mp4 \
       -t 60 -c:v libx264 -preset ultrafast -threads 2 -crf 26 \
       -c:a aac -b:a 96k -movflags +faststart \
       paper_drafts/demo_video_v2/twitter_60s.mp4
```

## Outputs

| File                                  | Size  | Notes                                |
|---------------------------------------|-------|--------------------------------------|
| `fabriccrypt_v2_with_narration.mp4`   | 9.4M  | Full 3:06 narrated demo, 960x540@12  |
| `twitter_60s.mp4`                     | 3.6M  | First 60s, under Twitter 100MB cap   |
| `captions.srt`                        | 4.8K  | Sentence-level timed captions        |
| `audio/audio_S{0..7}.mp3`             | ~3.3M | Per-section TTS audio                |
| `audio_full.wav`                      | 16M   | Silence-padded mixed timeline        |

## Thermal constraints

The ikaros APU shares `thermal_zone0` with concurrent benchmarks. v1 spiked to
67C during a single-pass ffmpeg encode. v2 mitigations:

- Two-stage build: PIL renders PNG frames to disk first, then ffmpeg encodes.
- 960x540 @ 12fps (vs 1080p/30 in v1) - 6x less raw pixel throughput.
- Per-frame thermal pacing: sleep 1s @ >58C, 3s @ >62C, hard cool to 58C @ >66C.
- During ffmpeg encode pass: SIGSTOP/SIGCONT loop pauses encoder when APU >66C.
- `nice -n 19 ionice -c 3` to yield to any concurrent ML workload.

Build completed with one mid-encode SIGSTOP cycle (72C -> 52C, resumed cleanly).

## Vocabulary

Per the user's banned-word list:
- BANNED: die, kill, soul, feel, loyalty, sentient, alive
- USED: bound, substrate, fingerprint, attestation, verifier, challenge-response,
  per-die, hardware-encrypted

## TTS settings

- Model: `tts-1-hd` (highest quality OpenAI offers)
- Voice: `nova` (warm, clear, female; alternatives tried in script: shimmer, onyx)
- Total characters: 2,703
- Total cost: **$0.0811** (well under the $0.05 estimate)

## Data provenance

All numerics shown in the video are from:
- `results/IDENTITY_BENCHMARK_2026-05-30/embodiment14c/ikaros_spoof_v2.json`
- (honest 100.0%, static replay 0.6%, dynamic replay 1.2%, peer 2.0%, random 0.6%)

No fabricated values. The "99.4% rejected" in narration is `1 - 0.006 = 0.994`.
