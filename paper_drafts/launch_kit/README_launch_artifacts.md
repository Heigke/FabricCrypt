# FabricCrypt Launch Artifacts

Two viral-launch artifacts built on 2026-06-01.

## 1. One-page whitepaper PDF

`whitepaper_pdf/fabriccrypt_onepager.pdf`

Single A4 page, designed for sharing on Twitter / HN / arXiv landing pages.

Sections (top-down):
- **Title bar** with tagline "Two identical computers. Only one can run our AI."
- **The Problem** (Apple PCC / NVIDIA CC / Intel TDX / AMD SEV-SNP all vendor-rooted)
- **3-step solution** (13 fingerprint signals  ->  nonce-bound challenge  ->  AI bound to chip)
- **Key numbers**: 100% LOO, 60-80 bits security, 10/10 attacks blocked, 0.6% replay rate
- **Twin Droids** side-by-side mini-PC graphic with diverging signal traces and a REJECT verdict
- **Comparison table**: FabricCrypt vs PCC / CCM / TDX / SEV-SNP / TPM 2.0
- **Footer**: GitHub, arXiv placeholder, contact, repro command

### Rebuild

```bash
source venv/bin/activate
python3 paper_drafts/launch_kit/whitepaper_pdf/build_onepager.py
```

Requirements: `reportlab` (installed in repo venv).

## 2. Pedagogical explainer video

`explainer_video/fabriccrypt_explainer_5min.mp4` (5:48, 1080p H264 + AAC, ~17 MB)
`explainer_video/fabriccrypt_explainer_thumbnail.png` (YouTube-ready thumbnail)

Audience: technically-curious general public (ELI12). Tone: warm, accessible,
Star Wars droid analogy as the central metaphor.

### Section map

| # | Name             | Duration | Topic |
|---|------------------|----------|-------|
| 0 | hook             | 0:18     | "Imagine an AI you cannot copy" |
| 1 | problem          | 0:41     | Why vendor-rooted attestation isn't enough |
| 2 | fingerprints     | 1:02     | 13 micro-arch signals, 466-dim signature |
| 3 | crypto           | 1:09     | Nonce-bound challenge, RFE + cPUF + HMAC |
| 4 | personality      | 1:02     | Same model, different chip, divergent style (honest 0.664 < 0.75 null) |
| 5 | why              | 0:55     | Verifiable inference, AI insurance, federated learning |
| 6 | honest           | 0:30     | n=2, gate not passed, scope |
| 7 | close            | 0:11     | Tagline restate |

**Total: 5:48** (348 s)

### Build pipeline

1. `narration_sections.py` -- 8 paragraph strings, ELI12-level
2. `generate_tts.py` -- calls OpenAI `tts-1-hd` with voice `nova`, one MP3 per section
3. `build_explainer.py` -- renders one PIL slide per section, ffmpeg `-loop 1` muxes
   each slide+audio into an MP4, then concat into the final video

Light-compute design (no per-frame encoding): each section is a single static
slide held for the duration of the narration, with `-preset ultrafast`.
APU peaked at 56 C during build (well below 65 C thermal budget).

### Rebuild

```bash
source venv/bin/activate
# 1. (one-time) generate audio  (~$0.07 of OpenAI credits)
python3 paper_drafts/launch_kit/explainer_video/generate_tts.py
# 2. assemble video
python3 paper_drafts/launch_kit/explainer_video/build_explainer.py
```

Requires `ffmpeg`, `Pillow`, `numpy`, OpenAI API key in
`/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/.env`
under `openai_api_key=...`.

## Numerical claims sourced from `paper_drafts/fabriccrypt_v3.md`

- 466-dim signature, 13 micro-arch signals
- 100% LOO classification, n=2 chassis
- Sub-millisecond end-to-end (median 1.12 ms, p99 2.79 ms)
- 60-80 bit security (Tier-2 hardening: RFE + Controlled-PUF + ZK binding)
- 10/10 attack gates passed (v2.1 battery, includes O115 custom forgery)
- Personality gate honest NULL: observed 0.664 vs pre-registered 0.75
  (CI [0.619, 0.705], p << 0.001 vs chance, n=420)

No numerical claim is fabricated.

## Vocabulary discipline

Banned words (die, kill, soul, feel, loyalty, sentient, alive) are NOT
used in narration text. We use: bound, coupled, substrate-locked,
fingerprint, attestation, instrument-dependent.
