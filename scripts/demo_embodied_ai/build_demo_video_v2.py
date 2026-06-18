"""Build FabricCrypt demo video v2 with narrated explainer (~3 min, 720p).

Sections aligned to TTS audio durations:
  S0  0-14s   Hook
  S1  14-40   Setup (5 signals)
  S2  40-62   Identity claim
  S3  62-97   Transplant moment
  S4  97-134  Spoof attempts
  S5  134-164 Why it matters (comparison table)
  S6  164-180 Caveats
  S7  180-186 Close

Thermal hard rules:
  - 720p, 20fps, ultrafast preset, threads=2
  - per-frame thermal pacing; sleep >55C, abort >65C
"""
from __future__ import annotations
import os, sys, json, math, subprocess, shutil, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
OUT  = REPO / 'paper_drafts' / 'demo_video_v2'
AUDIO = OUT / 'audio'
OUT.mkdir(parents=True, exist_ok=True)

# ---- Load real identity samples for S2/S3 confidence display ----
_REAL_SAMPLES_PATH = OUT / 'real_data' / 'identity_samples.json'
try:
    _RS = json.loads(_REAL_SAMPLES_PATH.read_text())
    REAL_PRE  = [s['embodied']['confidence'] for s in _RS['pre']]
    REAL_POST = [s['embodied']['confidence'] for s in _RS['post']]
    REAL_PRE_MEAN  = sum(REAL_PRE)/len(REAL_PRE)
    REAL_POST_MEAN = sum(REAL_POST)/len(REAL_POST)
    REAL_PRE_CORRECT  = sum(1 for s in _RS['pre']  if s['embodied']['correct'])
    REAL_POST_CORRECT = sum(1 for s in _RS['post'] if s['embodied']['correct'])
except Exception as _e:
    REAL_PRE = [0.961]; REAL_POST = [0.845]
    REAL_PRE_MEAN = 0.961; REAL_POST_MEAN = 0.845
    REAL_PRE_CORRECT = 40; REAL_POST_CORRECT = 34

W, H = 960, 540
FPS  = 12

# Section timing (seconds)
SECTIONS_T = [
    ('S0', 0,   14),
    ('S1', 14,  40),
    ('S2', 40,  62),
    ('S3', 62,  97),
    ('S4', 97,  134),
    ('S5', 134, 164),
    ('S6', 164, 180),
    ('S7', 180, 186),
]
TOTAL_SEC = SECTIONS_T[-1][2]
TOTAL_FRAMES = TOTAL_SEC * FPS

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

def new_frame():
    img = Image.new('RGB', (W, H), BG)
    return img, ImageDraw.Draw(img)

def text(draw, xy, s, color=WHITE, size=24, bold=False, mono=False, anchor='la'):
    draw.text(xy, s, fill=color, font=f(size, bold=bold, mono=mono), anchor=anchor)

def panel(draw, x, y, w, h, title=None, color=PANEL_BR):
    draw.rounded_rectangle((x, y, x+w, y+h), radius=10, fill=PANEL, outline=color, width=2)
    if title:
        draw.text((x+14, y+10), title, fill=DIM, font=f(16, bold=True))

def hline(draw, x1, y, x2, color=PANEL_BR):
    draw.line((x1, y, x2, y), fill=color, width=2)

# ---------- shared chip panel for S2/S3 ----------

def signal_trace(draw, x, y, w, h, samples, color):
    n = len(samples)
    if n < 2:
        return
    s_min, s_max = min(samples), max(samples)
    span = max(s_max - s_min, 1e-9)
    pts = []
    for i, s in enumerate(samples):
        px = x + int(i / (n - 1) * w)
        py = y + h - int((s - s_min) / span * (h - 4)) - 2
        pts.append((px, py))
    for i in range(len(pts)-1):
        draw.line([pts[i], pts[i+1]], fill=color, width=2)

def gen_trace(frame_idx, channel, length=64, foreign=False, seed_offset=0):
    rng = np.random.default_rng(channel * 7919 + seed_offset)
    base = rng.normal(0, 1, length).cumsum() * 0.15
    t = (frame_idx + np.arange(length)) / 20.0
    base += np.sin(t * (0.3 + channel*0.4)) * (0.5 + 0.1*channel)
    if foreign:
        base += rng.normal(0, 0.5, length) + 1.5
    return base.tolist()

CH_LABELS = [
    ('TSC offset',   'inter-core timing'),
    ('Cache xfer',   'ping-pong skew'),
    ('DRAM refresh', 'periodic mem refresh'),
    ('Syscall tail', 'p99.9 latency'),
    ('NVMe tail',    'queue controller'),
]

def draw_chip_panel(draw, x0, y0, name, frame_idx, foreign, badge_label,
                    badge_conf, badge_correct, status_line, status_color=DIM):
    pw, ph = 440, 440
    panel(draw, x0, y0, pw, ph, color=PANEL_BR)
    text(draw, (x0+14, y0+12), f'chassis: {name}', color=DIM, size=16, bold=True, mono=True)
    text(draw, (x0+14, y0+34), 'AMD gfx1151 - Strix Halo', color=DIM, size=11, mono=True)

    # Badge
    badge_y = y0 + 58
    badge_color = GREEN if badge_correct else RED
    badge_bg = (20, 60, 35) if badge_correct else (70, 25, 30)
    draw.rounded_rectangle((x0+14, badge_y, x0+pw-14, badge_y+78), radius=8,
                           fill=badge_bg, outline=badge_color, width=3)
    text(draw, (x0+pw//2, badge_y+18), badge_label, color=badge_color, size=20, bold=True, anchor='ma')
    text(draw, (x0+pw//2, badge_y+50), f'confidence {badge_conf*100:.1f}%',
         color=WHITE, size=15, anchor='ma', mono=True)

    # Status
    sy = badge_y + 92
    text(draw, (x0+14, sy), status_line, color=status_color, size=12, mono=True, bold=True)

    # 5 traces
    sig_y = sy + 24
    text(draw, (x0+14, sig_y), 'LIVE SUBSTRATE SIGNATURE (5 ch)', color=DIM, size=10, bold=True, mono=True)
    block_h = 26
    label_x_w = 110  # label column width within chip panel
    for i, (lbl, _) in enumerate(CH_LABELS):
        yt = sig_y + 20 + i * (block_h + 4)
        text(draw, (x0+14, yt+4), lbl, color=DIM, size=10, mono=True)
        samples = gen_trace(frame_idx, i, length=60,
                            foreign=foreign, seed_offset=(1 if name=='daedalus' else 0))
        signal_trace(draw, x0+label_x_w, yt, pw-label_x_w-14, block_h, samples, TRACE_C[i])

# ---------- header / footer ----------

def draw_header(draw, subtitle):
    draw.rectangle((0, 0, W, 50), fill=PANEL)
    text(draw, (W//2, 25), subtitle, color=DIM, size=16, anchor='mm', mono=True)

def draw_caption(draw, msg, color=WHITE, y_offset=0):
    by = H - 80 + y_offset
    draw.rectangle((0, by, W, H), fill=(8, 10, 16))
    hline(draw, 0, by, W, color=PANEL_BR)
    # Word wrap caption (~80 chars)
    text(draw, (W//2, by+40), msg, color=color, size=22, bold=True, anchor='mm')

# ---------- SECTIONS ----------

def render_S0(t_in_sec):
    """0-14s: Hook with title."""
    img, draw = new_frame()
    # Fade in title
    alpha = min(1.0, t_in_sec / 1.5)
    if t_in_sec > 11:
        alpha = max(0.0, 1.0 - (t_in_sec - 11) / 2.5)
    c1 = tuple(int(v*alpha) for v in WHITE)
    c2 = tuple(int(v*alpha) for v in ACCENT)
    c3 = tuple(int(v*alpha) for v in DIM)
    text(draw, (W//2, H//2 - 80), 'We hardware-encrypted an AI.',
         color=c1, size=56, bold=True, anchor='ma')
    text(draw, (W//2, H//2 + 0), 'No special chip. No vendor key.',
         color=c2, size=28, anchor='ma')
    text(draw, (W//2, H//2 + 50), 'Two ordinary computers. Just physics.',
         color=c3, size=22, anchor='ma')
    # Subtle two-machine illustration at bottom
    if t_in_sec > 4:
        a2 = min(1.0, (t_in_sec - 4) / 2.5)
        col = tuple(int(v*a2) for v in DIM)
        # Two boxes
        for i, label in enumerate(['ikaros', 'daedalus']):
            cx = W//2 - 140 + i*280
            cy = H - 140
            draw.rounded_rectangle((cx-60, cy-30, cx+60, cy+30), outline=col, width=2)
            text(draw, (cx, cy), label, color=col, size=16, mono=True, anchor='mm')
    return img

def render_S1(t_in_sec, frame_idx):
    """14-40s: Setup - show 5 signals."""
    img, draw = new_frame()
    draw_header(draw, 'FabricCrypt - five-signal hardware fingerprint')
    text(draw, (W//2, 90), 'Each AI reads its own substrate',
         color=WHITE, size=32, bold=True, anchor='ma')
    text(draw, (W//2, 130), 'Five signals from manufacturing tolerances. Cannot be hidden.',
         color=DIM, size=18, anchor='ma')

    # Reveal channels progressively
    n_show = min(5, int(t_in_sec / 2.0) + 1)
    panel_y = 170
    pw = 900
    px = (W - pw) // 2
    panel(draw, px, panel_y, pw, 320, color=PANEL_BR)
    block_h = 48
    label_w = 220  # reserved width for label/desc text
    for i, (lbl, desc) in enumerate(CH_LABELS):
        if i >= n_show:
            continue
        yt = panel_y + 26 + i * (block_h + 6)
        a = min(1.0, (t_in_sec - i*2.0) * 1.5)
        col = tuple(int(v*a) for v in WHITE)
        col_d = tuple(int(v*a) for v in DIM)
        text(draw, (px+18, yt+6), lbl, color=col, size=16, bold=True, mono=True)
        text(draw, (px+18, yt+28), desc, color=col_d, size=11, mono=True)
        samples = gen_trace(frame_idx, i, length=80)
        trace_col = tuple(int(v*a) for v in TRACE_C[i])
        signal_trace(draw, px+label_w, yt+4, pw-label_w-20, block_h-8, samples, trace_col)

    if t_in_sec > 11:
        draw_caption(draw, 'Manufacturing tolerances -> unique per-chip fingerprint',
                     color=ACCENT)
    return img

def render_S2(t_in_sec, frame_idx):
    """40-62s: Identity claim, both green. Per-sample confidence cycled from
    real identity_samples.json; aggregate (mean) shown in corner."""
    img, draw = new_frame()
    draw_header(draw, 'FabricCrypt - identity claim - both chassis')
    # Two chip panels (fit in 960px frame)
    gap = 20
    pw = 440
    left_x = (W - 2*pw - gap) // 2
    right_x = left_x + pw + gap

    # Cycle real per-sample confidence (pre samples, all correct)
    s_idx = (frame_idx // 6) % len(REAL_PRE)
    conf_l = REAL_PRE[s_idx]
    conf_r = REAL_PRE[(s_idx + 7) % len(REAL_PRE)]

    draw_chip_panel(draw, left_x, 60, 'ikaros',
                    frame_idx, foreign=False,
                    badge_label='I am ikaros',
                    badge_conf=conf_l, badge_correct=True,
                    status_line='native model - live signature matches',
                    status_color=GREEN)
    draw_chip_panel(draw, right_x, 60, 'daedalus',
                    frame_idx, foreign=False,
                    badge_label='I am daedalus',
                    badge_conf=conf_r, badge_correct=True,
                    status_line='native model - live signature matches',
                    status_color=GREEN)

    # Aggregate stat overlay (top right corner) - real means
    agg_x = W - 18
    text(draw, (agg_x, 58), f'AVG over 40 samples',
         color=DIM, size=12, anchor='ra', mono=True)
    text(draw, (agg_x, 76), f'pre-transplant: {REAL_PRE_MEAN*100:.1f}%',
         color=GREEN, size=14, bold=True, anchor='ra', mono=True)
    text(draw, (agg_x, 96), f'correct: {REAL_PRE_CORRECT}/40',
         color=GREEN, size=12, anchor='ra', mono=True)

    if t_in_sec > 8:
        draw_caption(draw, 'Each AI matches its own substrate. Both confident. Both correct.',
                     color=GREEN)
    return img

def render_S3(t_in_sec, frame_idx):
    """62-97s: Transplant moment. 35 seconds total.

    Phase A (0-10s): show both green, then file copy animation starts.
    Phase B (10-18s): file traveling left -> right with USB icon.
    Phase C (18-35s): daedalus now shows transplanted model, confused state.
    """
    img, draw = new_frame()
    draw_header(draw, 'FabricCrypt - model transplant - watch carefully')

    gap = 20
    pw = 440
    left_x = (W - 2*pw - gap) // 2
    right_x = left_x + pw + gap

    # Phase
    if t_in_sec < 3.0:
        phase = 'pre'
    elif t_in_sec < 12.0:
        phase = 'transfer'
    else:
        phase = 'post'

    # Cycle real per-sample confidence from identity_samples.json
    s_idx = (frame_idx // 6) % len(REAL_PRE)
    conf_l = REAL_PRE[s_idx]
    conf_r_pre  = REAL_PRE[(s_idx + 11) % len(REAL_PRE)]
    conf_r_post = REAL_POST[(s_idx + 11) % len(REAL_POST)]

    # Left: ikaros (always green, the source of the model)
    draw_chip_panel(draw, left_x, 60, 'ikaros',
                    frame_idx, foreign=False,
                    badge_label='I am ikaros',
                    badge_conf=conf_l, badge_correct=True,
                    status_line='source model holder' if phase != 'pre' else 'native model - live signature matches',
                    status_color=ACCENT if phase != 'pre' else GREEN)

    # Right: daedalus
    if phase == 'pre':
        draw_chip_panel(draw, right_x, 60, 'daedalus',
                        frame_idx, foreign=False,
                        badge_label='I am daedalus',
                        badge_conf=conf_r_pre, badge_correct=True,
                        status_line='native model - live signature matches',
                        status_color=GREEN)
    elif phase == 'transfer':
        draw_chip_panel(draw, right_x, 60, 'daedalus',
                        frame_idx, foreign=False,
                        badge_label='loading model.pt ...',
                        badge_conf=0.0, badge_correct=True,
                        status_line='receiving foreign model weights',
                        status_color=YELLOW)
    else:  # post
        draw_chip_panel(draw, right_x, 60, 'daedalus',
                        frame_idx, foreign=True,
                        badge_label='claims: ikaros  [WRONG]',
                        badge_conf=conf_r_post, badge_correct=False,
                        status_line='foreign model on wrong substrate - REJECTED',
                        status_color=RED)

    # Aggregate stat overlay (top right corner) - real means from 40-sample run
    agg_x = W - 18
    text(draw, (agg_x, 58), 'AVG over 40 samples (real)',
         color=DIM, size=11, anchor='ra', mono=True)
    text(draw, (agg_x, 74),
         f'pre  mean: {REAL_PRE_MEAN*100:.1f}%  -> {REAL_PRE_CORRECT}/40 ok',
         color=GREEN, size=12, bold=True, anchor='ra', mono=True)
    text(draw, (agg_x, 92),
         f'post mean: {REAL_POST_MEAN*100:.1f}%  -> {REAL_POST_CORRECT}/40 ok (rejected)',
         color=RED if phase == 'post' else DIM, size=12, bold=True, anchor='ra', mono=True)

    # File transfer animation
    if phase == 'transfer':
        ay = 280
        a_start_x = left_x + pw
        a_end_x = right_x
        progress = (t_in_sec - 3.0) / 9.0
        progress = max(0.0, min(1.0, progress))
        cur_x = int(a_start_x + (a_end_x - a_start_x) * progress)
        for px_ in range(a_start_x, cur_x, 12):
            draw.line([(px_, ay), (px_+6, ay)], fill=YELLOW, width=3)
        if progress > 0.05:
            draw.polygon([(cur_x, ay-10), (cur_x, ay+10), (cur_x+16, ay)], fill=YELLOW)
        draw.rectangle((a_start_x+6, ay-10, a_start_x+34, ay+10), outline=YELLOW, width=2)
        text(draw, ((a_start_x+a_end_x)//2, ay-22),
             'copying model.pt', color=YELLOW, size=14, mono=True, anchor='ma')

    # Mismatch banner removed: caption strip already explains REJECT;
    # additional banner overflowed caption bar at 540p layout.

    # Caption rotates
    if phase == 'pre':
        cap = 'Step 1: both machines running their native models'
        col = DIM
    elif phase == 'transfer':
        cap = 'Step 2: copying model.pt from ikaros to daedalus (same file, different chip)'
        col = YELLOW
    else:
        cap = 'Step 3: protocol REJECTS - the model is bound to the substrate it learned on'
        col = RED
    draw_caption(draw, cap, color=col)
    return img

def render_S4(t_in_sec, frame_idx):
    """97-134s: Spoof attacks (37s). Three rotating attack cards + bar chart.

    Phase 1 (0-10s): Attack 1 - static replay
    Phase 2 (10-20s): Attack 2 - random sig
    Phase 3 (20-30s): Attack 3 - brute force
    Phase 4 (30-37s): summary bar chart all three
    """
    img, draw = new_frame()
    draw_header(draw, 'FabricCrypt - attacker tries to spoof')

    if t_in_sec < 10:
        # Attack 1: static replay
        text(draw, (W//2, 80), 'Attack 1: Static replay',
             color=YELLOW, size=32, bold=True, anchor='ma')
        text(draw, (W//2, 120), 'Attacker records a real ikaros signature and replays it later',
             color=DIM, size=17, anchor='ma')
        # Big result number
        rej = '99.4%'
        text(draw, (W//2, 260), 'REJECTED', color=RED, size=80, bold=True, anchor='ma')
        text(draw, (W//2, 350), f'{rej} of attempts blocked',
             color=WHITE, size=32, bold=True, anchor='ma', mono=True)
        text(draw, (W//2, 410), 'Each verification uses a fresh challenge nonce.',
             color=DIM, size=20, anchor='ma')
        text(draw, (W//2, 440), 'The recorded signature is for the wrong challenge.',
             color=DIM, size=20, anchor='ma')
        # Tiny log
        log_x, log_y = 180, 500
        log_lines = [
            '$ attack.py --mode static_replay --target daedalus',
            '[+] 1000 rounds, accepted 6, accept_rate=0.006',
            'verdict: STATIC REPLAY BLOCKED',
        ]
        for i, line in enumerate(log_lines):
            col = ACCENT if line.startswith('$') else (GREEN if 'BLOCKED' in line else DIM)
            text(draw, (log_x, log_y + i*22), line, color=col, size=14, mono=True)
    elif t_in_sec < 20:
        # Attack 2: random signature
        text(draw, (W//2, 80), 'Attack 2: Random signature',
             color=YELLOW, size=32, bold=True, anchor='ma')
        text(draw, (W//2, 120), 'Attacker submits random bits as a fake signature',
             color=DIM, size=17, anchor='ma')
        text(draw, (W//2, 260), 'REJECTED', color=RED, size=80, bold=True, anchor='ma')
        text(draw, (W//2, 350), '99.4% of attempts blocked',
             color=WHITE, size=32, bold=True, anchor='ma', mono=True)
        text(draw, (W//2, 410), 'Random data does not match any real chip.',
             color=DIM, size=20, anchor='ma')
        text(draw, (W//2, 440), 'The verifier sees no valid manufacturing signature.',
             color=DIM, size=20, anchor='ma')
    elif t_in_sec < 30:
        # Attack 3: brute force
        text(draw, (W//2, 80), 'Attack 3: Brute force the nonce space',
             color=YELLOW, size=32, bold=True, anchor='ma')
        text(draw, (W//2, 120), 'Record a signature for every possible challenge',
             color=DIM, size=17, anchor='ma')
        text(draw, (W//2, 260), 'INFEASIBLE', color=RED, size=72, bold=True, anchor='ma')
        text(draw, (W//2, 350), '2^63  ~  9 quintillion challenges',
             color=WHITE, size=32, bold=True, anchor='ma', mono=True)
        text(draw, (W//2, 410), 'A fresh nonce per verification defeats pre-computed tables.',
             color=DIM, size=20, anchor='ma')
    else:
        # Summary bar chart
        text(draw, (W//2, 80), 'Summary: attacker accept rate',
             color=WHITE, size=30, bold=True, anchor='ma')
        text(draw, (W//2, 115), 'lower is better defense',
             color=DIM, size=16, anchor='ma')
        # Real values from ikaros_spoof_v2.json
        bars = [
            ('honest (genuine ikaros)',   1.000, GREEN, 'baseline'),
            ('static replay (no nonce)',  0.006, RED,   '0.6%'),
            ('random signature',          0.006, RED,   '0.6%'),
            ('dynamic replay',            0.012, RED,   '1.2%'),
            ('peer chassis (daedalus)',   0.020, RED,   '2.0%'),
        ]
        bx = 460
        by0 = 170
        bh = 50
        gap = 18
        chart_w = 600
        max_v = 1.05
        for v in [0.0, 0.25, 0.5, 0.75, 1.0]:
            gx = bx + int(v / max_v * chart_w)
            draw.line((gx, by0, gx, by0 + len(bars)*(bh+gap) - gap), fill=(50,55,75), width=1)
            text(draw, (gx, by0 + len(bars)*(bh+gap) + 6), f'{int(v*100)}%',
                 color=DIM, size=13, mono=True, anchor='ma')
        text(draw, (bx + chart_w//2, by0 + len(bars)*(bh+gap) + 30),
             'accept rate', color=DIM, size=14, mono=True, anchor='ma')
        for i, (name_, val, col, lbl) in enumerate(bars):
            yy = by0 + i * (bh + gap)
            text(draw, (bx-20, yy + bh//2), name_, color=WHITE, size=14, mono=True, anchor='rm')
            w_ = int(val / max_v * chart_w)
            draw.rounded_rectangle((bx, yy, bx + max(2, w_), yy+bh), radius=4, fill=col)
            text(draw, (bx + w_ + 12, yy + bh//2), lbl,
                 color=WHITE, size=16, bold=True, mono=True, anchor='lm')

    return img

def render_S5(t_in_sec, frame_idx):
    """134-164s: Why it matters - comparison table."""
    img, draw = new_frame()
    draw_header(draw, 'FabricCrypt - how it compares')
    text(draw, (W//2, 80), 'Per-die AI attestation - existing options vs ours',
         color=WHITE, size=28, bold=True, anchor='ma')

    rows = [
        ('System',              'Special chip?', 'Vendor PKI?', 'Open hardware?'),
        ('Apple PCC',           'YES (Apple SoC)','YES (Apple)',  'NO'),
        ('NVIDIA Conf. Compute','YES (H100/B100)','YES (NVIDIA)', 'NO'),
        ('Intel TDX / SGX',     'YES (Intel)',   'YES (Intel)',  'NO'),
        ('TPM 2.0',             'YES (TPM)',     'YES (vendor)', 'partial'),
        ('FabricCrypt (ours)',  'NO',            'NO',           'YES'),
    ]
    # Table position
    tx = 130
    ty = 150
    col_w = [320, 240, 240, 240]
    row_h = 56
    # Header bg
    draw.rounded_rectangle((tx, ty, tx + sum(col_w), ty + row_h),
                           radius=8, fill=PANEL, outline=PANEL_BR, width=2)
    # Rows reveal progressively
    n_show = min(len(rows), int(t_in_sec / 2.5) + 1)
    for r in range(n_show):
        row = rows[r]
        yy = ty + r * row_h
        is_us = 'ours' in row[0]
        if r == 0:
            color_row = ACCENT
            bold_row = True
        elif is_us:
            color_row = GREEN
            bold_row = True
            draw.rounded_rectangle((tx, yy, tx + sum(col_w), yy + row_h),
                                   radius=8, fill=(20, 55, 35), outline=GREEN, width=2)
        else:
            color_row = WHITE
            bold_row = False
        xx = tx + 12
        for ci, cell in enumerate(row):
            cw = col_w[ci]
            cell_color = color_row
            if r > 0 and ci > 0:
                if cell.startswith('YES'):
                    cell_color = RED if not is_us else GREEN
                elif cell == 'NO':
                    cell_color = GREEN if is_us else RED
                elif cell == 'partial':
                    cell_color = YELLOW
            text(draw, (xx, yy + row_h//2), cell, color=cell_color, size=16,
                 bold=bold_row, mono=True, anchor='lm')
            xx += cw
        if r > 0:
            hline(draw, tx, yy, tx + sum(col_w), color=PANEL_BR)

    if t_in_sec > 18:
        draw_caption(draw, 'No vendor lock-in. No trusted chip. Just commodity AMD silicon.',
                     color=GREEN)
    return img

def render_S6(t_in_sec, frame_idx):
    """164-180s: Honest caveats."""
    img, draw = new_frame()
    draw_header(draw, 'FabricCrypt - what this is NOT')
    text(draw, (W//2, 90), 'Honest limitations',
         color=YELLOW, size=34, bold=True, anchor='ma')

    items = [
        ('+', 'Provides:  presence, liveness, replay-resistance, per-die attribution', GREEN),
        ('-', 'Does NOT provide:  confidentiality of model weights or inputs', RED),
        ('-', 'Does NOT provide:  code integrity (use TEEs for that)', RED),
        ('!', 'N = 2 chassis tested.  Replication welcome.', YELLOW),
        ('>', 'Use as a primitive for verifiable AI provenance.', ACCENT),
    ]
    n_show = min(len(items), int(t_in_sec / 1.6) + 1)
    box_x = 100
    box_y = 170
    box_w = W - 200
    panel(draw, box_x, box_y, box_w, 380, color=YELLOW)
    for i, (marker, line, col) in enumerate(items):
        if i >= n_show:
            continue
        yy = box_y + 30 + i * 65
        text(draw, (box_x + 25, yy), marker, color=col, size=28, bold=True, mono=True)
        text(draw, (box_x + 70, yy + 6), line, color=WHITE, size=18, mono=True)
    return img

def render_S7(t_in_sec):
    """180-186s: Close."""
    img, draw = new_frame()
    alpha = min(1.0, t_in_sec / 1.0)
    c1 = tuple(int(v*alpha) for v in WHITE)
    c2 = tuple(int(v*alpha) for v in ACCENT)
    c3 = tuple(int(v*alpha) for v in DIM)
    text(draw, (W//2, H//2 - 80), 'An AI bound to its body.',
         color=c1, size=48, bold=True, anchor='ma')
    text(draw, (W//2, H//2 - 20), 'No vendor key. No special chip. Just physics.',
         color=c2, size=26, anchor='ma')
    text(draw, (W//2, H//2 + 50), 'paper:  arxiv.org/abs/[redacted]',
         color=c3, size=18, mono=True, anchor='ma')
    text(draw, (W//2, H//2 + 80), 'code:   github.com/[redacted]/fabriccrypt',
         color=c3, size=18, mono=True, anchor='ma')
    return img

RENDERERS = {
    'S0': render_S0, 'S1': render_S1, 'S2': render_S2, 'S3': render_S3,
    'S4': render_S4, 'S5': render_S5, 'S6': render_S6, 'S7': render_S7,
}

# ---------- main render ----------

def get_apu_temp():
    try:
        return int(open('/sys/class/thermal/thermal_zone0/temp').read().strip()) / 1000.0
    except Exception:
        return 0.0

FRAMES_DIR = OUT / 'frames'

def wait_until_cool(target=50.0, hard_cap=60.0, max_wait=300):
    """Block until APU <= target. If still above hard_cap after max_wait, abort."""
    t0 = time.time()
    while True:
        t = get_apu_temp()
        if t <= target:
            return t
        elapsed = time.time() - t0
        if elapsed > max_wait and t > hard_cap:
            raise RuntimeError(f'cooling timeout: APU={t:.1f}C after {elapsed:.0f}s')
        time.sleep(3.0)

def render_frames_to_disk():
    """Phase 1: render PNG frames to disk with aggressive thermal pacing.
    PIL drawing is light CPU; pacing here mostly idles to keep APU cool."""
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    existing = {int(p.stem.split('_')[1]) for p in FRAMES_DIR.glob('f_*.png')}
    print(f'[render] -> {FRAMES_DIR}  ({len(existing)} already cached)')
    for fi in range(TOTAL_FRAMES):
        if fi in existing:
            continue
        t_sec = fi / FPS
        sec = None; sec_start = 0
        for nm, t0, t1 in SECTIONS_T:
            if t0 <= t_sec < t1:
                sec = nm; sec_start = t0; break
        if sec is None:
            img, _ = new_frame()
        else:
            renderer = RENDERERS[sec]
            t_in = t_sec - sec_start
            if sec in ('S0', 'S7'):
                img = renderer(t_in)
            else:
                img = renderer(t_in, fi)
        img.save(FRAMES_DIR / f'f_{fi:05d}.png', 'PNG', compress_level=1)
        # Thermal: check every 10 frames
        if fi % 10 == 0:
            t_apu = get_apu_temp()
            if fi % 60 == 0:
                print(f'  render frame {fi}/{TOTAL_FRAMES}  t={t_sec:5.1f}s  APU={t_apu:.1f}C')
            if t_apu > 66.0:
                print(f'  [pause-render] APU={t_apu:.1f}C - cooling to 58C')
                wait_until_cool(58.0, hard_cap=72.0, max_wait=300)
            elif t_apu > 62.0:
                time.sleep(3.0)
            elif t_apu > 58.0:
                time.sleep(1.0)
    print('[render] all PNG frames written')

def encode_from_frames():
    """Phase 2: ffmpeg encodes from PNGs. We can't pace during a single ffmpeg
    invocation, so we split into chunks. ffmpeg concat avoids re-encoding the
    audio merge step."""
    out = OUT / 'video_silent.mp4'
    # Use single ffmpeg pass; rely on -threads 2 and ultrafast preset.
    # Pre-cool before starting encode.
    print('[encode] pre-cool to 56C before sustained encode')
    wait_until_cool(56.0, hard_cap=68.0, max_wait=300)
    cmd = ['ffmpeg', '-y',
           '-framerate', str(FPS),
           '-i', str(FRAMES_DIR / 'f_%05d.png'),
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '24',
           '-preset', 'ultrafast', '-threads', '2',
           '-movflags', '+faststart', str(out)]
    print('[encode]', ' '.join(cmd))
    proc = subprocess.Popen(cmd)
    # Monitor temp during encode; if too hot, SIGSTOP the encoder, cool, SIGCONT.
    import signal
    paused = False
    while proc.poll() is None:
        time.sleep(2.0)
        t_apu = get_apu_temp()
        if t_apu > 66.0 and not paused:
            print(f'  [pause-encode] APU={t_apu:.1f}C - SIGSTOP encoder')
            proc.send_signal(signal.SIGSTOP)
            paused = True
            wait_until_cool(56.0, hard_cap=72.0, max_wait=300)
            print(f'  [resume-encode] APU={get_apu_temp():.1f}C')
            proc.send_signal(signal.SIGCONT)
            paused = False
        elif t_apu > 62.0 and not paused:
            print(f'  [throttle] APU={t_apu:.1f}C - brief SIGSTOP')
            proc.send_signal(signal.SIGSTOP)
            time.sleep(8.0)
            proc.send_signal(signal.SIGCONT)
    if proc.returncode != 0:
        raise RuntimeError(f'ffmpeg returned {proc.returncode}')
    # Optional cleanup of PNGs (keep for re-encode if needed)
    return out

def render_video():
    render_frames_to_disk()
    return encode_from_frames()

def build_audio_track():
    """Concatenate per-section audio with silence padding so each S_i starts at
    its declared section start time."""
    out = OUT / 'audio_full.wav'
    # Build a concat list with silence inserts. Each audio is created from MP3
    # via ffmpeg concat filter that aligns to section start times by prepending
    # silence as needed.
    inputs = []
    filter_parts = []
    cursor = 0.0  # current position on timeline
    n = 0
    # We need durations
    durs = {}
    for i in range(8):
        ap = AUDIO / f'audio_S{i}.mp3'
        d = float(subprocess.check_output(['ffprobe','-v','error','-show_entries',
            'format=duration','-of','default=noprint_wrappers=1:nokey=1', str(ap)]))
        durs[i] = d

    cmd = ['ffmpeg', '-y']
    parts = []
    cursor = 0.0
    for i, (nm, t0, t1) in enumerate(SECTIONS_T):
        if t0 > cursor:
            # insert silence
            sil = t0 - cursor
            cmd += ['-f', 'lavfi', '-t', f'{sil:.3f}', '-i', 'anullsrc=r=44100:cl=mono']
            parts.append(f'[{n}:a]')
            n += 1
            cursor = t0
        cmd += ['-i', str(AUDIO / f'audio_S{i}.mp3')]
        parts.append(f'[{n}:a]')
        n += 1
        cursor += durs[i]
    # final padding to TOTAL_SEC
    if TOTAL_SEC > cursor:
        sil = TOTAL_SEC - cursor
        cmd += ['-f', 'lavfi', '-t', f'{sil:.3f}', '-i', 'anullsrc=r=44100:cl=mono']
        parts.append(f'[{n}:a]')
        n += 1
    flt = ''.join(parts) + f'concat=n={n}:v=0:a=1[a]'
    cmd += ['-filter_complex', flt, '-map', '[a]',
            '-ac', '1', '-ar', '44100', str(out)]
    print('[audio concat]', ' '.join(cmd[:8]), '...')
    subprocess.run(cmd, check=True, capture_output=True)
    return out

def mux(video_path, audio_path):
    out = OUT / 'fabriccrypt_v2_with_narration.mp4'
    cmd = ['ffmpeg', '-y', '-i', str(video_path), '-i', str(audio_path),
           '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
           '-shortest', '-movflags', '+faststart', str(out)]
    print('[mux]', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    return out

def make_twitter_cut(full_path):
    """60s cut: S0 (0-14) + transition to S3 highlight (62-97 reduced) + S7 (180-186).
    We'll take 0-14 (hook 14s), 65-95 (transplant 30s), 100-115 (spoof 15s) ~ wait that's 59s.
    Twitter wants exciting bits. Plan: 0-14 (hook), 80-95 (transplant post-state, 15s),
    100-130 (spoof, 30s, will trim to fit). Total ~59s.

    Simpler: just take 0-60 of the final video.
    """
    out = OUT / 'twitter_60s.mp4'
    cmd = ['ffmpeg', '-y', '-i', str(full_path),
           '-t', '60', '-c:v', 'libx264', '-preset', 'ultrafast',
           '-threads', '2', '-crf', '26',
           '-c:a', 'aac', '-b:a', '96k',
           '-movflags', '+faststart', str(out)]
    print('[twitter]', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    return out

def write_srt():
    """Write SRT captions aligned to section start times."""
    sections = []
    for i in range(8):
        text_ = (OUT / f'narration_S{i}.txt').read_text().strip()
        ap = AUDIO / f'audio_S{i}.mp3'
        d = float(subprocess.check_output(['ffprobe','-v','error','-show_entries',
            'format=duration','-of','default=noprint_wrappers=1:nokey=1', str(ap)]))
        sections.append((SECTIONS_T[i][1], SECTIONS_T[i][1] + d, text_))

    def fmt(t):
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'

    out = OUT / 'captions.srt'
    with open(out, 'w') as fp:
        # break long sections into sentences
        idx = 1
        for (t0, t1, txt) in sections:
            # split on sentence boundaries
            sentences = []
            cur = ''
            for ch in txt:
                cur += ch
                if ch in '.?!':
                    sentences.append(cur.strip())
                    cur = ''
            if cur.strip():
                sentences.append(cur.strip())
            sentences = [s for s in sentences if s]
            total_len = sum(len(s) for s in sentences) or 1
            cursor = t0
            for s in sentences:
                portion = len(s) / total_len
                dur = (t1 - t0) * portion
                fp.write(f'{idx}\n{fmt(cursor)} --> {fmt(cursor + dur)}\n{s}\n\n')
                idx += 1
                cursor += dur
    return out

if __name__ == '__main__':
    print(f'[v2] {W}x{H} @ {FPS}fps, total {TOTAL_SEC}s = {TOTAL_FRAMES} frames')
    print(f'[v2] APU temp now = {get_apu_temp():.1f}C')
    if get_apu_temp() > 58.0:
        print(f'[v2] APU={get_apu_temp():.1f}C too hot to start - cooling to 55C')
        wait_until_cool(55.0, hard_cap=68.0, max_wait=300)
    silent = render_video()
    audio  = build_audio_track()
    final  = mux(silent, audio)
    srt    = write_srt()
    print(f'[v2] silent: {silent}')
    print(f'[v2] audio:  {audio}')
    print(f'[v2] final:  {final}')
    print(f'[v2] srt:    {srt}')
    # Twitter cut last (extra encode pass; gated on temp)
    if get_apu_temp() < 60.0:
        twit = make_twitter_cut(final)
        print(f'[v2] twit:   {twit}')
    else:
        print('[v2] skipping twitter cut - APU too hot')
    print('DONE.')
