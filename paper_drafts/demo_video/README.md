# `paper_drafts/demo_video/` — FabricCrypt 90-second demo

| File                          | What it is                              | Size  | Codec / spec               |
| ----------------------------- | --------------------------------------- | ----- | -------------------------- |
| `raw_recording.mp4`           | 90-s cut, no section markers            | 21 MB | h.264, 1920×1080, 30 fps   |
| `fabriccrypt_demo_90s.mp4`    | 90-s cut, with section + time overlays  | 20 MB | h.264, 1920×1080, 30 fps   |
| `transplant_moment.gif`       | 20-s loop of the transplant moment      | 1.9 MB | gif, 600×338, 10 fps      |
| `STORYBOARD.md`               | what each second of the video shows     | —     | —                          |
| `README.md`                   | this file                               | —     | —                          |

Companion figure dropped at `paper_drafts/figures/spoof_defense_bars.png`
(standalone version of the bar chart used in section 4).

## How it was made

The video is **synthesised** by `scripts/demo_embodied_ai/build_demo_video.py`,
not screen-recorded. The reason: we don’t have two physical chassis in front
of a camera, no headless X session for `ffmpeg -f x11grab`, and no Playwright
on this box. Frames are rendered with PIL and streamed straight to ffmpeg
via stdin (no PNGs on disk).

Real data feeds it:

* Identity confidence values come from 80 live `/api/identity` responses
  collected against `demo_server.py` running on **ikaros** with own/peer sigs
  from `results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/`. 40 pre-transplant
  samples (40/40 correct, conf 87–99.8 %) and 40 post-transplant (0/40 correct,
  conf 58–99.4 %).
* The spoof bars use the T1 block of
  `results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/ikaros_spoof.json`.
* The five signature channels (`pkg_uW`, `temp_mC`, `tsc_mean`, `ns_mean`,
  `cstate2`) are the same ones the dashboard pulls from `LiveSignature`.

The signal traces themselves are seeded per-channel synthetic noise (we did
not freeze 2700 real WebSocket samples for this render). The post-transplant
right panel uses a different noise seed + DC offset so the visual mismatch
matches what the heads are reacting to.

## Reproduce

Prereqs: `ffmpeg`, the project `venv/`, PIL (already in the venv), and the
sig files under `results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/`.

```sh
cd /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy

# 1. Capture fresh identity samples from a real demo_server run.
HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python \
    scripts/demo_embodied_ai/demo_server.py \
    --host-name ikaros --port 8770 \
    --own-sigs  results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/ikaros_sigs.npz \
    --peer-sigs results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/daedalus_sigs.npz \
    > /tmp/demo_server.log 2>&1 &
sleep 10

venv/bin/python - <<'PY'
import requests, json, time
samples = {'pre': [], 'post': []}
for _ in range(40):
    samples['pre'].append(requests.get('http://localhost:8770/api/identity').json())
    time.sleep(0.25)
requests.post('http://localhost:8770/api/transplant')
for _ in range(40):
    samples['post'].append(requests.get('http://localhost:8770/api/identity').json())
    time.sleep(0.25)
json.dump(samples, open('/tmp/demo_samples.json','w'))
PY

pkill -f demo_server.py

# 2. Render the video (writes into paper_drafts/demo_video/).
venv/bin/python scripts/demo_embodied_ai/build_demo_video.py
```

The render uses `-preset ultrafast -threads 2` and inserts micro-sleeps
between frames whenever APU > 55 °C (1.5 s sleep > 60 °C, 20 s pause > 65 °C,
matching CLAUDE.md thermal rules). The whole pipeline took ~85 s on ikaros
and never went above 62 °C in the second run.

## What to change if you want to re-record from a real screen

The shape of the build script is small and deliberate:

* `render_section1..5(frame_idx, sec_frame)` each return a single 1920×1080
  PIL frame — swap any one of them for a function that loads frames from an
  actual screen-recorded segment (`ffmpeg -ss <t> -t <dt> -i screen.mp4`).
* If you have a real screen recording of one panel, you can do the
  side-by-side compositing inside `render_section2/3` instead of synthesising
  the dashboard.
* Section markers + time index are added in `encode_final()` via two
  `drawtext` filters; tweak font / position there.

## Caveats called out in section 5 of the video

* N = 2 chassis tested so far.
* Vendor-key-free per-die attestation; replay-attack defense (nonce mixing)
  is implemented in `signature_io.py` but the full audience-challenge protocol
  is still being hardened.
* Reproduction script will live at `github.com/[redacted]/fabriccrypt` once
  the repo is public.
