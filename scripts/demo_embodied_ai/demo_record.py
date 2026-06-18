"""Record a 30-second loop of the demo from both ikaros and daedalus.

Two modes:
  1. SCRIPTED  (default) - drives the demo through the canonical 30s sequence
                           and captures screenshots via Playwright (if installed).
  2. INSTRUCTIONS         - print exact shell commands to record manually with
                           ffmpeg + x11grab on Linux; no extra deps.

Usage:
    venv/bin/python scripts/demo_embodied_ai/demo_record.py instructions

The 30-second sequence (per the brief):
   0-5s  cold open: two browser windows side by side, both showing "I am X" green
   5-10s "I am ikaros" (left), "I am daedalus" (right), embodied green/correct
  10-15s POST /api/stress on LEFT only; anomaly bar spikes red on LEFT
  15-20s POST /api/transplant on RIGHT (peer model moves to right)
  20-25s right now shows ikaros's model on daedalus -> identity flips, anomaly
        wanders
  25-30s caption frame
"""
from __future__ import annotations
import sys, os, time, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))

INSTRUCTIONS = r"""
=== Manual record using ffmpeg + x11grab (Linux, no extra deps) ===

# 1. start both demo servers (one on each machine)
#    ikaros:
ikaros$  HSA_OVERRIDE_GFX_VERSION=11.0.0 \
         venv/bin/python scripts/demo_embodied_ai/demo_server.py \
           --host-name ikaros --port 8770 \
           --own-sigs  results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/ikaros_sigs.npz \
           --peer-sigs results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/daedalus_sigs.npz

#    daedalus (via ssh; CLAUDE.md vars):
daedalus$ HSA_OVERRIDE_GFX_VERSION=11.0.0 \
          venv/bin/python scripts/demo_embodied_ai/demo_server.py \
            --host-name daedalus --port 8770 \
            --own-sigs  results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/daedalus_sigs.npz \
            --peer-sigs results/IDENTITY_BENCHMARK_2026-05-30/embodiment14b/ikaros_sigs.npz

# 2. open two browser windows, place side by side at known coords:
#    left  (ikaros):   http://localhost:8770/
#    right (daedalus): http://192.168.0.37:8770/   (or daedalus.local)

# 3. record 30 seconds of the area spanning both windows
ffmpeg -y -framerate 25 -f x11grab -video_size 1920x540 -i :0.0+0,200 \
       -t 30 -c:v libx264 -pix_fmt yuv420p demo_30s.mp4

# 4. while recording, type this sequence (use another shell):
#    t=05  curl -s -X POST http://localhost:8770/api/stress?duration=4
#    t=15  curl -s -X POST http://192.168.0.37:8770/api/transplant
#    t=25  (no action - frame settles)

=== End ===
"""

def scripted():
    """Use Playwright to drive a real headed browser; skip if not installed."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        print("playwright not installed; falling back to instructions.")
        print(INSTRUCTIONS)
        return 1

    ik = os.environ.get('IKAROS_URL', 'http://localhost:8770/')
    da = os.environ.get('DAEDALUS_URL', 'http://192.168.0.37:8770/')
    out_dir = os.environ.get('OUT_DIR', os.path.join(ROOT, '_recordings'))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[record] capturing {ik} (left) + {da} (right) -> {out_dir}")
    with sync_playwright() as pw:
        br = pw.chromium.launch(headless=False, args=['--window-size=960,720'])
        ctx = br.new_context(viewport={'width': 960, 'height': 720},
                             record_video_dir=out_dir, record_video_size={'width': 1920, 'height': 720})
        p_ik = ctx.new_page(); p_ik.goto(ik); time.sleep(2)
        p_da = ctx.new_page(); p_da.goto(da); time.sleep(2)
        # 0-5s settle
        time.sleep(5)
        # 5s: stress LEFT
        import requests
        try: requests.post(ik.rstrip('/') + '/api/stress?duration=4', timeout=3)
        except Exception as e: print(f"stress failed: {e}")
        time.sleep(10)  # to t=15
        # 15s: transplant RIGHT
        try: requests.post(da.rstrip('/') + '/api/transplant', timeout=3)
        except Exception as e: print(f"transplant failed: {e}")
        time.sleep(10)  # to t=25
        # 25s: hold caption
        time.sleep(5)
        ctx.close(); br.close()
    print(f"[record] done. videos in {out_dir}/")
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'instructions':
        print(INSTRUCTIONS); sys.exit(0)
    sys.exit(scripted())
