# demo_video_v2 fix audit — 2026-06-01

Two issues reported by user inspection of `fabriccrypt_v2_with_narration.mp4`:

## Issue 1: Signal channel labels clipped on left side
**Symptom:** "syscall p99" rendered as "all p99", "TSC inter-core offset" as "offset",
"C2C ping pong" as "g pong" — left chars cut off by frame edge.

**Root cause:** Frame width is `W=960`, but several panels exceeded the canvas:
- **S1 setup panel:** `pw=1080`, so `px = (960 - 1080)//2 = -60`. Labels at
  `px+20 = -40` were drawn off-screen on the left.
- **S2/S3 chip panels:** `pw=540, gap=40, 2*pw+gap = 1120 > 960`, so `left_x = -80`.
  Left panel labels at `left_x+18 = -62` were clipped.

The "TSC inter-core" / "C2C ping-pong" labels were also long-form strings that
would have crowded the trace area even at correct sizing.

**Fix in `scripts/demo_embodied_ai/build_demo_video_v2.py`:**
- S1: panel `pw` reduced from `1080 → 900`, height `380 → 320`, block_h `56 → 48`,
  font `18 → 16` for labels, `13 → 11` for descs. Trace starts at `px + label_w`
  with `label_w = 220` reserved for text.
- S2/S3 chip panels: `pw` reduced from `540 → 440`, gap `40 → 20`. Inner padding
  tightened (`18 → 14`), fonts shrunk proportionally (badge `26 → 20`, conf `18 → 15`),
  trace label column `150 → 110`. All chip-internal label text is now within frame.
- Channel labels shortened in `CH_LABELS`:
    - "cache ping-pong" → "Cache xfer"
    - "syscall p99.9"  → "Syscall tail"
    - "NVMe queue"     → "NVMe tail"
    - "DRAM refresh"   (kept)
    - "TSC offset"     (kept)

## Issue 2: Frame 2 (transplant) confidence numbers mismatch real data
**Symptom:** S2/S3 showed badges like `pre=97%, post=75%` but the real run in
`real_data/identity_samples.json` has `μ_pre=0.961, μ_post=0.845`.

**Root cause:** Per-chip badge confidences were hardcoded constants
(`badge_conf=0.97`, `badge_conf=0.98`, `badge_conf=0.75`) in `render_S2()` and
`render_S3()`. They were never linked to the JSON.

**Fix:**
- Load `real_data/identity_samples.json` at module import; expose
  `REAL_PRE`, `REAL_POST`, `REAL_PRE_MEAN`, `REAL_POST_MEAN`,
  `REAL_PRE_CORRECT`, `REAL_POST_CORRECT` as module constants.
- S2 badges cycle real per-sample values from `REAL_PRE` (40 actual values,
  range 0.868 – 0.998, mean 0.961). Index advances every 6 frames (0.5 s) so
  the displayed number visibly evolves while staying in the real distribution.
- S3 badges cycle real values the same way: `REAL_PRE` for pre/transfer phases
  (correct claim), `REAL_POST` for the post-transplant phase (range 0.585 – 0.994,
  mean 0.845, all 40 correctly flagged as wrong-host claims).
- **Aggregate overlay added top-right** of both S2 and S3:
  - S2: `AVG over 40 samples / pre-transplant: 96.1% / correct: 40/40`
  - S3: `AVG over 40 samples (real) / pre mean 96.1% → 40/40 ok /
        post mean 84.5% → 0/40 ok (rejected)`
  The post-mean line turns red once the transplant phase begins.
- Removed redundant `40 / 40 correct identifications` bar in S2 (replaced by
  the corner aggregate which is more precise).
- Removed S3 mismatch banner at `y=510` (overlapped the caption strip at the
  smaller 540p layout; the caption already says "REJECTED").

## Real numbers used (from `real_data/identity_samples.json`)
| metric              | value      |
|---------------------|------------|
| pre  n              | 40         |
| pre  mean conf      | 0.9607     |
| pre  min / max      | 0.868 / 0.998 |
| pre  correct        | 40 / 40    |
| post n              | 40         |
| post mean conf      | 0.8449     |
| post min / max      | 0.585 / 0.994 |
| post correct (host) | 0 / 40 (all correctly flagged as wrong-host) |

## Verification frames
`/tmp/video_inspect_v2/`
- `verify_t15.png` — S1 mid-reveal, first label "TSC offset" fully visible at left.
- `verify_t30.png` — S1 full panel, all 5 labels intact: TSC offset / Cache xfer /
  DRAM refresh / Syscall tail / NVMe tail. None clipped.
- `verify_t45.png` — S2 both green, per-sample badges (91.4 % / 92.9 %), aggregate
  corner shows `pre 96.1 %, correct 40/40`.
- `verify_t80.png` — S3 post phase, daedalus red ("claims: ikaros [WRONG]" /
  confidence 87.4 %), ikaros green (97.2 %), aggregate corner shows the real
  means including the red post line "post mean 84.5 % → 0/40 ok (rejected)".

## Re-render details
- Frame cache cleared for S1+S2+S3 region (frames 168–1163 of 2232), other
  sections reused from cache.
- Thermal: peak APU 69 °C during encode (one SIGSTOP/SIGCONT cycle), no abort.
- Final encode used existing audio track from `audio_full.wav`; mux re-ran.
- Twitter cut re-rendered after final mp4 was created.

## Deliverables (overwritten in place)
- `paper_drafts/demo_video_v2/fabriccrypt_v2_with_narration.mp4` (3 min 06 s,
  ~10 MB) — full narrated video.
- `paper_drafts/demo_video_v2/twitter_60s.mp4` (60 s, ~3.9 MB).
- `paper_drafts/demo_video_v2/captions.srt` — unchanged (text content didn't
  change).
- `/tmp/video_inspect_v2/verify_t{15,30,45,80}.png` — 4 verification frames.

## Files changed
- `scripts/demo_embodied_ai/build_demo_video_v2.py`
  - Added real-data loader block (after `OUT.mkdir`).
  - `CH_LABELS` shortened.
  - `draw_chip_panel` (pw 540→440, font sizes, padding).
  - `render_S1` (pw 1080→900, fonts).
  - `render_S2` (real per-sample confidence, aggregate overlay,
     removed redundant stats line).
  - `render_S3` (real per-sample confidence for all phases, aggregate
     overlay, removed mismatch banner).
