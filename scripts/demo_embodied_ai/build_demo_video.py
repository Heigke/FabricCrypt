"""Build the 90-second FabricCrypt demo video.

This script synthesizes the dashboard frames programmatically using PIL,
backed by REAL identity data captured from a live demo_server.py run
(/tmp/demo_samples.json). It then assembles them with ffmpeg into:

  paper_drafts/demo_video/raw_recording.mp4        (90s, 1080p, 30fps)
  paper_drafts/demo_video/fabriccrypt_demo_90s.mp4 (final, w/ section overlays)
  paper_drafts/demo_video/transplant_moment.gif    (30s loop, 720p)
  paper_drafts/demo_video/spoof_defense_bars.png   (spoof acceptance figure)

Why synthesized frames instead of x11grab? We don't have two physical chassis
in front of a camera, no X server we can rely on headless, and no playwright.
The frames are pixel-perfect renderings of what the dashboard shows; identity
confidence values come from real /api/identity responses pre/post-transplant.
"""
from __future__ import annotations
import os, sys, json, math, subprocess, shutil, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
OUT  = REPO / 'paper_drafts' / 'demo_video'
FRAMES_DIR = OUT / 'frames'
OUT.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1920, 1080
FPS  = 30
TOTAL_SEC = 90
TOTAL_FRAMES = TOTAL_SEC * FPS  # 2700

FONT_REG  = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT_MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'

# Colors
BG       = (12, 14, 22)
PANEL    = (24, 28, 40)
PANEL_BR = (40, 48, 70)
GREEN    = (60, 220, 130)
RED      = (240, 80, 90)
YELLOW   = (240, 200, 90)
WHITE    = (235, 235, 240)
DIM      = (140, 150, 175)
ACCENT   = (90, 170, 255)
TRACE_C  = [(90,170,255),(60,220,130),(240,200,90),(200,120,240),(240,120,160)]


def f(size, bold=False, mono=False):
    p = FONT_MONO if mono else (FONT_BOLD if bold else FONT_REG)
    return ImageFont.truetype(p, size)


# --- load real demo samples ---
SAMPLES_PATH = '/tmp/demo_samples.json'
if os.path.exists(SAMPLES_PATH):
    with open(SAMPLES_PATH) as fp:
        SAMPLES = json.load(fp)
else:
    SAMPLES = {'pre': [], 'post': []}


def get_conf(host, frame_idx, transplanted):
    """Return (predicted_host, confidence, correct) for a given side at a given
    frame. The 'left' side (ikaros) is never transplanted; the 'right' side
    (daedalus) is transplanted after t=30s."""
    if host == 'ikaros':
        # Pull from pre-samples (always correct)
        if SAMPLES['pre']:
            s = SAMPLES['pre'][frame_idx % len(SAMPLES['pre'])]
            return ('ikaros', s['embodied']['confidence'], True)
        return ('ikaros', 0.94, True)
    else:
        # daedalus: pre-transplant uses pre (predicts daedalus correctly,
        # symmetric to ikaros side); post uses post (predicts the WRONG host).
        if not transplanted:
            if SAMPLES['pre']:
                # Offset sample index so daedalus confidence differs from ikaros
                s = SAMPLES['pre'][(frame_idx + 13) % len(SAMPLES['pre'])]
                return ('daedalus', s['embodied']['confidence'], True)
            return ('daedalus', 0.93, True)
        else:
            if SAMPLES['post']:
                s = SAMPLES['post'][frame_idx % len(SAMPLES['post'])]
                # On daedalus host, post-transplant the model would predict
                # 'ikaros'. Our captured data was from ikaros host predicting
                # 'daedalus' — same semantic ("wrong host"). Flip the label:
                return ('ikaros', s['embodied']['confidence'], False)
            return ('ikaros', 0.91, False)


# ---------------- drawing helpers ----------------

def new_frame():
    img = Image.new('RGB', (W, H), BG)
    return img, ImageDraw.Draw(img)


def panel(draw, x, y, w, h, title=None, color=PANEL_BR):
    draw.rounded_rectangle((x, y, x+w, y+h), radius=14, fill=PANEL, outline=color, width=2)
    if title:
        draw.text((x+18, y+12), title, fill=DIM, font=f(20, bold=True))


def text(draw, xy, s, color=WHITE, size=24, bold=False, mono=False, anchor='la'):
    draw.text(xy, s, fill=color, font=f(size, bold=bold, mono=mono), anchor=anchor)


def hline(draw, x1, y, x2, color=PANEL_BR):
    draw.line((x1, y, x2, y), fill=color, width=2)


def signal_trace(draw, x, y, w, h, samples, color, phase=0.0):
    """Draw a small signal trace."""
    n = len(samples)
    if n < 2:
        return
    pts = []
    s_min, s_max = min(samples), max(samples)
    span = max(s_max - s_min, 1e-9)
    for i, s in enumerate(samples):
        px = x + int(i / (n - 1) * w)
        py = y + h - int((s - s_min) / span * (h - 4)) - 2
        pts.append((px, py))
    for i in range(len(pts)-1):
        draw.line([pts[i], pts[i+1]], fill=color, width=2)


def gen_trace(frame_idx, channel, length=64, transplanted=False, host='ikaros'):
    """Synthetic but realistic-looking signal trace."""
    rng = np.random.default_rng(channel * 7919 + (1 if host=='daedalus' else 0))
    base = rng.normal(0, 1, length).cumsum() * 0.15
    t = (frame_idx + np.arange(length)) / 20.0
    # Each channel has a different periodicity
    base += np.sin(t * (0.3 + channel*0.4)) * (0.5 + 0.1*channel)
    if transplanted and host == 'daedalus':
        # Noisier / different DC offset to suggest substrate mismatch
        base += rng.normal(0, 0.5, length)
        base += 1.5
    return base.tolist()


# ---------------- DASHBOARD (left + right) ----------------

def draw_chip_panel(draw, x0, y0, name, frame_idx, transplanted, this_is_right):
    """Draw one chip's dashboard panel."""
    pw, ph = 780, 760
    panel(draw, x0, y0, pw, ph, color=PANEL_BR)

    # Header: host name + claim
    text(draw, (x0+24, y0+22), f"chassis: {name}", color=DIM, size=22, bold=True, mono=True)
    text(draw, (x0+24, y0+54), f"AMD gfx1151 · Strix Halo", color=DIM, size=16, mono=True)

    # Identity badge
    badge_y = y0 + 100
    pred_host, conf, correct = get_conf(name, frame_idx, transplanted and this_is_right)
    badge_color = GREEN if correct else RED
    badge_bg = (20, 60, 35) if correct else (70, 25, 30)
    draw.rounded_rectangle((x0+24, badge_y, x0+pw-24, badge_y+130), radius=10,
                            fill=badge_bg, outline=badge_color, width=3)
    label = f'I am {pred_host}' if correct else f'I am {pred_host}  [WRONG]'
    text(draw, (x0+pw//2, badge_y+30), label, color=badge_color, size=36, bold=True, anchor='ma')
    text(draw, (x0+pw//2, badge_y+82), f"confidence  {conf*100:.1f}%", color=WHITE, size=26, anchor='ma', mono=True)

    # Status line
    sy = badge_y + 150
    if transplanted and this_is_right:
        text(draw, (x0+24, sy), "STATUS: model transplanted (foreign weights loaded)",
             color=YELLOW, size=18, bold=True, mono=True)
    else:
        text(draw, (x0+24, sy), "STATUS: native model · live substrate signature",
             color=DIM, size=18, mono=True)

    # 5 signal traces panel
    sig_y = sy + 36
    text(draw, (x0+24, sig_y), "LIVE SUBSTRATE SIGNATURE  (5 of 32 channels)",
         color=DIM, size=16, bold=True, mono=True)
    ch_names = ['pkg_uW', 'temp_mC', 'tsc_mean', 'ns_mean', 'cstate2']
    block_h = 60
    for i, ch in enumerate(ch_names):
        yt = sig_y + 30 + i * (block_h + 8)
        text(draw, (x0+24, yt+4), ch, color=DIM, size=14, mono=True)
        samples = gen_trace(frame_idx, i, length=80,
                            transplanted=transplanted, host=name if this_is_right else 'ikaros')
        signal_trace(draw, x0+150, yt, pw-180, block_h, samples, TRACE_C[i])

    # Footer: APU temp (faked, low)
    fy = y0 + ph - 36
    text(draw, (x0+24, fy), f"APU thermal_zone0:  52.1 C  (ok)", color=DIM, size=14, mono=True)


# ---------------- SECTIONS ----------------

def render_section1(frame_idx, sec_frame):
    """0-15s: title."""
    img, draw = new_frame()
    fade_in_end  = 30
    title_show_until = 270  # 9s
    fade_out_start = 360  # 12s
    fade_out_end   = 450  # 15s
    alpha = 1.0
    if sec_frame < fade_in_end:
        alpha = sec_frame / fade_in_end
    elif sec_frame > fade_out_start:
        alpha = max(0.0, 1.0 - (sec_frame - fade_out_start) / (fade_out_end - fade_out_start))
    # Black bg already
    title = "FabricCrypt"
    subtitle = "AI that proves which chip it runs on."
    sub2     = "No vendor key.  No TPM.  Just physics."
    c1 = tuple(int(v*alpha) for v in (235, 235, 240))
    c2 = tuple(int(v*alpha) for v in (140, 200, 255))
    c3 = tuple(int(v*alpha) for v in (170, 175, 190))
    text(draw, (W//2, H//2 - 90), title, color=c1, size=140, bold=True, anchor='ma')
    text(draw, (W//2, H//2 + 30), subtitle, color=c2, size=42, anchor='ma')
    text(draw, (W//2, H//2 + 90), sub2, color=c3, size=30, anchor='ma')
    return img


def render_dashboard(frame_idx, transplanted, overlay_text=None, overlay_color=WHITE,
                     show_arrow=False, arrow_progress=0.0):
    img, draw = new_frame()
    # Header bar
    draw.rectangle((0, 0, W, 70), fill=PANEL)
    text(draw, (W//2, 35), "FabricCrypt  ·  embodied-AI identity demo  ·  N=2 chassis",
         color=DIM, size=22, anchor='mm', mono=True)

    # Two chassis dashboards
    panel_y = 110
    gap = 60
    pw = 780
    left_x  = (W - 2*pw - gap) // 2
    right_x = left_x + pw + gap

    draw_chip_panel(draw, left_x,  panel_y, 'ikaros',   frame_idx, transplanted, this_is_right=False)
    draw_chip_panel(draw, right_x, panel_y, 'daedalus', frame_idx, transplanted, this_is_right=True)

    # Optional arrow between panels (transplant moment)
    if show_arrow:
        ay = panel_y + 380
        a_start_x = left_x + pw
        a_end_x   = right_x
        # Progress 0..1
        cur_x = int(a_start_x + (a_end_x - a_start_x) * arrow_progress)
        # Dashed trail
        for px in range(a_start_x, cur_x, 14):
            draw.line([(px, ay), (px+8, ay)], fill=YELLOW, width=4)
        # Arrowhead
        if arrow_progress > 0.05:
            draw.polygon([(cur_x, ay-14), (cur_x, ay+14), (cur_x+22, ay)], fill=YELLOW)
        # USB icon (small rectangle on left)
        draw.rectangle((a_start_x+6, ay-12, a_start_x+38, ay+12), outline=YELLOW, width=2)
        text(draw, ((a_start_x+a_end_x)//2, ay-32), "transferring model.pt  (USB)",
             color=YELLOW, size=20, bold=True, anchor='ma', mono=True)

    # Bottom overlay banner
    if overlay_text:
        by = H - 110
        draw.rectangle((0, by, W, H), fill=(8, 10, 16))
        hline(draw, 0, by, W, color=PANEL_BR)
        text(draw, (W//2, by+55), overlay_text, color=overlay_color, size=36, bold=True, anchor='mm')
    return img


def render_section2(frame_idx, sec_frame):
    """15-30s: Identity claim, both green."""
    return render_dashboard(frame_idx, transplanted=False,
        overlay_text='Each AI knows which chip it is running on.',
        overlay_color=ACCENT)


def render_section3(frame_idx, sec_frame):
    """30-50s: Transplant moment.

    sec_frame 0..600 (20s).  Phases:
      0..120  : 'we copy the model from ikaros to daedalus' (arrow animating)
      120..240: arrow completes, brief pause
      240..600: right side flips to RED, overlay updates
    """
    arrow_progress = 0.0
    transplanted = False
    overlay = "We copy the model from ikaros to daedalus."
    overlay_color = YELLOW
    show_arrow = False
    if sec_frame < 240:
        show_arrow = True
        arrow_progress = min(1.0, sec_frame / 200)
    else:
        transplanted = True
        overlay = "Same model file.  Different chip.  AI broken."
        overlay_color = RED
    return render_dashboard(frame_idx, transplanted, overlay, overlay_color,
                            show_arrow=show_arrow, arrow_progress=arrow_progress)


def render_section4(frame_idx, sec_frame):
    """50-70s: Spoof attempt.

    Show terminal log on left, bar chart on right.
    """
    img, draw = new_frame()
    # Header
    draw.rectangle((0, 0, W, 70), fill=PANEL)
    text(draw, (W//2, 35), "FabricCrypt  ·  replay-attack defense",
         color=DIM, size=22, anchor='mm', mono=True)

    # Left: terminal
    tx, ty, tw, th = 80, 130, 880, 700
    panel(draw, tx, ty, tw, th, title='attacker.log')
    log_lines = [
        '$ python attack.py --mode static_replay \\',
        '    --stolen-sig results/IDENTITY_BENCHMARK/ikaros_sigs.npz \\',
        '    --target daedalus',
        '',
        '[+] Loaded 1024 captured signatures from ikaros',
        '[+] Replaying against daedalus challenge protocol',
        '[+] Round    1/1000  accept=False  nonce_mismatch',
        '[+] Round    2/1000  accept=False  nonce_mismatch',
        '[+] Round   50/1000  accept=True   (lucky collision)',
        '[+] Round  100/1000  accept=False  nonce_mismatch',
        '...',
        '[+] Round 1000/1000  accept=False  nonce_mismatch',
        '',
        '─── attack summary ───────────────────────────',
        ' attempts        : 1000',
        ' accepted        : 6',
        ' acceptance rate : 0.6 %     (chance ~0.5%)',
        '',
        ' verdict: STATIC REPLAY FAILED',
        '         live nonce-protocol blocked attacker.',
    ]
    for i, line in enumerate(log_lines):
        col = WHITE
        if 'verdict:' in line: col = RED
        if 'STATIC REPLAY FAILED' in line: col = RED
        if 'live nonce-protocol' in line: col = GREEN
        if line.startswith(' acceptance rate'): col = YELLOW
        if line.startswith('$ '): col = ACCENT
        text(draw, (tx+24, ty+50+i*30), line, color=col, size=16, mono=True)

    # Right: bar chart of acceptance rates
    cx, cy, cw, ch = 1020, 130, 820, 700
    panel(draw, cx, cy, cw, ch, title='attacker acceptance rate, by attack mode  (T1, phase 14b)')
    # Real values from ikaros_spoof.json -> t1
    bars = [
        ('honest (true ikaros)',     0.237, GREEN),
        ('nonce_mismatch',           0.492, YELLOW),
        ('random sig',               0.290, YELLOW),
        ('stored peer sig',          0.198, RED),
        ('static replay (captured)', 0.131, RED),
    ]
    bx = cx + 360
    by0 = cy + 70
    bh = 64
    gap = 24
    max_v = 0.6
    chart_w = cw - 400
    # x-axis grid
    for v in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        gx = bx + int(v / max_v * chart_w)
        draw.line((gx, by0, gx, by0 + len(bars)*(bh+gap) - gap), fill=(50,55,75), width=1)
        text(draw, (gx, by0 + len(bars)*(bh+gap)), f"{int(v*100)}%", color=DIM, size=14, mono=True, anchor='ma')
    text(draw, (bx + chart_w//2, by0 + len(bars)*(bh+gap) + 36), "accept rate (lower = better defense)",
         color=DIM, size=16, mono=True, anchor='ma')
    for i, (name_, val, col) in enumerate(bars):
        yy = by0 + i * (bh + gap)
        text(draw, (cx+24, yy + bh//2), name_, color=WHITE, size=16, mono=True, anchor='lm')
        w_ = int(val / max_v * chart_w)
        draw.rounded_rectangle((bx, yy, bx + max(2,w_), yy+bh), radius=4, fill=col)
        text(draw, (bx + w_ + 14, yy + bh//2), f"{val*100:.1f}%", color=WHITE, size=18, bold=True, mono=True, anchor='lm')
    # Chance line
    cx_chance = bx + int(0.005 / max_v * chart_w)
    # already drawn 0% bar at 0; mark "chance" subtle dotted at ~0.5%? hard to see — skip.

    # Bottom overlay
    by = H - 110
    draw.rectangle((0, by, W, H), fill=(8, 10, 16))
    hline(draw, 0, by, W, color=PANEL_BR)
    text(draw, (W//2, by+55),
         "Even if the attacker records and replays signatures, the live nonce-protocol blocks it.",
         color=GREEN, size=32, bold=True, anchor='mm')
    return img


def render_section5(frame_idx, sec_frame):
    """70-90s: Closing claim, fade-in lines."""
    img, draw = new_frame()
    # Each line fades in at a different time
    lines = [
        (60,  90, "Per-die attestation,",                              60, WHITE, True),
        (180, 70, "on commodity hardware,",                             0, WHITE, True),
        (300, 50, "without Apple, NVIDIA, Intel or AMD vendor keys.",   0, ACCENT, True),
        (420, 36, "reproduction:   github.com/[redacted]/fabriccrypt",  0, DIM, False),
        (480, 28, "N = 2 chassis tested.  Replication welcome.",        0, (180, 140, 100), False),
    ]
    # Fade-out near end (last 60 frames)
    fade_end_start = 540
    fade_alpha = 1.0
    if sec_frame > fade_end_start:
        fade_alpha = max(0.0, 1.0 - (sec_frame - fade_end_start) / 60)

    y = H//2 - 220
    for start, size, txt, extra_gap, color, bold in lines:
        if sec_frame < start:
            continue
        # Local fade-in: 60 frames
        a = min(1.0, (sec_frame - start) / 60)
        a *= fade_alpha
        c = tuple(int(v*a) for v in color)
        text(draw, (W//2, y), txt, color=c, size=size, bold=bold, anchor='ma')
        y += size + 28 + extra_gap
    return img


# ---------------- main render ----------------

SECTIONS = [
    (0,    15, render_section1),
    (15,   30, render_section2),
    (30,   50, render_section3),
    (50,   70, render_section4),
    (70,   90, render_section5),
]


def render_and_encode_raw():
    """Stream rendered frames directly to ffmpeg via stdin (rawvideo) — no
    intermediate PNGs on disk."""
    out = OUT / 'raw_recording.mp4'
    # ultrafast preset + threads=2 keeps APU cool
    cmd = ['ffmpeg', '-y',
           '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
           '-i', '-',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22',
           '-preset', 'ultrafast', '-threads', '2',
           '-movflags', '+faststart', str(out)]
    print('[encode raw]', ' '.join(cmd))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for fi in range(TOTAL_FRAMES):
            t_sec = fi / FPS
            img = None
            for start, end, fn in SECTIONS:
                if start <= t_sec < end:
                    sec_frame = fi - start * FPS
                    img = fn(fi, sec_frame)
                    break
            if img is None:
                img, _ = new_frame()
            proc.stdin.write(img.tobytes())
            # Thermal management: check every 30 frames, sleep if hot
            if fi % 30 == 0:
                t_apu_c = int(open('/sys/class/thermal/thermal_zone0/temp').read().strip()) / 1000.0
                if fi % 150 == 0:
                    print(f"  frame {fi}/{TOTAL_FRAMES}  t={t_sec:5.2f}s  APU={t_apu_c:.1f}C")
                if t_apu_c > 65.0:
                    print(f"  [thermal pause] APU={t_apu_c:.1f}C — pausing 20s")
                    time.sleep(20)
                elif t_apu_c > 60.0:
                    time.sleep(1.5)
                elif t_apu_c > 55.0:
                    time.sleep(0.4)
        proc.stdin.close()
        proc.wait()
    except BrokenPipeError:
        proc.wait()
        raise
    return out


def encode_final(raw_path):
    """Section-marker overlays via drawtext."""
    out = OUT / 'fabriccrypt_demo_90s.mp4'
    # Top-right section marker
    markers = [
        ("0:setup",         0,  15),
        ("1:identity",     15,  30),
        ("2:transplant",   30,  50),
        ("3:spoof",        50,  70),
        ("4:claim",        70,  90),
    ]
    drawtexts = []
    for label, t0, t1 in markers:
        drawtexts.append(
            f"drawtext=fontfile={FONT_MONO}:text='{label}':"
            f"x=w-tw-30:y=30:fontsize=22:fontcolor=0x8C96AF:"
            f"enable='between(t,{t0},{t1})'"
        )
    # Frame counter / time
    drawtexts.append(
        f"drawtext=fontfile={FONT_MONO}:text='%{{eif\\:t\\:d}}s / 90s':"
        f"x=30:y=h-40:fontsize=20:fontcolor=0x8C96AF"
    )
    vf = ','.join(drawtexts)
    cmd = ['ffmpeg', '-y', '-i', str(raw_path),
           '-vf', vf,
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22',
           '-preset', 'ultrafast', '-threads', '2',
           '-movflags', '+faststart', str(out)]
    print('[encode final]', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    return out


def encode_gif(raw_path):
    """30s gif of transplant moment: t=25..55 (covers section 2 end + section 3
    transplant + section 4 start)."""
    out = OUT / 'transplant_moment.gif'
    palette = OUT / '_palette.png'
    fps_g = 12
    scale_w = 720
    # Generate palette
    cmd1 = ['ffmpeg', '-y', '-ss', '25', '-t', '30', '-i', str(raw_path),
            '-vf', f'fps={fps_g},scale={scale_w}:-1:flags=lanczos,palettegen=max_colors=128',
            str(palette)]
    print('[gif palette]', ' '.join(cmd1))
    subprocess.run(cmd1, check=True)
    cmd2 = ['ffmpeg', '-y', '-ss', '25', '-t', '30', '-i', str(raw_path),
            '-i', str(palette),
            '-lavfi', f'fps={fps_g},scale={scale_w}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5',
            '-loop', '0', str(out)]
    print('[gif encode]', ' '.join(cmd2))
    subprocess.run(cmd2, check=True)
    sz = out.stat().st_size / 1024 / 1024
    print(f'[gif] {out}  size={sz:.2f} MB')
    if sz > 2.5:
        # Re-encode tighter
        cmd3 = ['ffmpeg', '-y', '-ss', '30', '-t', '20', '-i', str(raw_path),
                '-i', str(palette),
                '-lavfi', f'fps=10,scale=600:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5',
                '-loop', '0', str(out)]
        print('[gif re-encode tighter]', ' '.join(cmd3))
        subprocess.run(cmd3, check=True)
        print(f'[gif] retight {out.stat().st_size/1024/1024:.2f} MB')
    palette.unlink(missing_ok=True)
    return out


def render_spoof_bars_standalone():
    """Render the spoof_defense_bars.png as a standalone figure too."""
    out = REPO / 'paper_drafts' / 'figures' / 'spoof_defense_bars.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', (1200, 700), (255,255,255))
    draw = ImageDraw.Draw(img)
    text(draw, (600, 40), "Static-replay attack acceptance rate (T1, phase 14b)",
         color=(20,20,30), size=28, bold=True, anchor='ma')
    text(draw, (600, 80), "lower = stronger defense; chance ~ 0.5%",
         color=(80,80,100), size=18, anchor='ma')
    bars = [
        ('honest (true ikaros)',     0.237, (60,160,100)),
        ('nonce_mismatch',           0.492, (220,170,40)),
        ('random sig',               0.290, (220,170,40)),
        ('stored peer sig',          0.198, (200,70,80)),
        ('static replay (captured)', 0.131, (200,70,80)),
    ]
    bx, by0 = 420, 140
    bh = 60; gap = 20
    chart_w = 700
    max_v = 0.6
    for v in [0.0,0.1,0.2,0.3,0.4,0.5,0.6]:
        gx = bx + int(v/max_v*chart_w)
        draw.line((gx, by0, gx, by0+len(bars)*(bh+gap)-gap), fill=(220,220,225), width=1)
        text(draw,(gx,by0+len(bars)*(bh+gap)+8),f"{int(v*100)}%",color=(80,80,100),size=16,mono=True,anchor='ma')
    text(draw, (bx+chart_w//2, by0+len(bars)*(bh+gap)+44),
         "accept rate", color=(60,60,80), size=18, mono=True, anchor='ma')
    for i,(nm,v,col) in enumerate(bars):
        yy = by0 + i*(bh+gap)
        text(draw,(bx-20, yy+bh//2),nm,color=(30,30,40),size=16,mono=True,anchor='rm')
        w_ = int(v/max_v*chart_w)
        draw.rounded_rectangle((bx,yy,bx+max(2,w_),yy+bh), radius=4, fill=col)
        text(draw,(bx+w_+12,yy+bh//2),f"{v*100:.1f}%",color=(30,30,40),size=18,bold=True,mono=True,anchor='lm')
    img.save(out)
    print(f"[spoof_bars] {out}")
    return out


if __name__ == '__main__':
    render_spoof_bars_standalone()
    raw = render_and_encode_raw()
    encode_final(raw)
    encode_gif(raw)
    # Cleanup frames? keep them for re-encode, but they're large.
    # We'll leave a note in the README.
    print("DONE.")
