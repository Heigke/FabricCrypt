#!/usr/bin/env python3
"""Build the FabricCrypt pedagogical explainer video.

Pipeline:
  1. Read per-section narration MP3 durations.
  2. Render each section's slide(s) as PNG frames using PIL + matplotlib.
     Animation is light: one keyframe slide per section, with a few sub-frames
     that fade-in / scroll, to keep encoding cheap.
  3. ffmpeg concat slides + audio into 1080p H264+AAC mp4.
  4. Generate thumbnail.

Constraints: light compute, no model training. Frames generated as static
PNGs and assembled with ffmpeg using -loop 1 per slide segment (no per-frame
encoding work).
"""
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
AUDIO_DIR = HERE / "audio"
FRAMES_DIR = HERE / "frames"
WORK_DIR = HERE / "work"
OUT_MP4 = HERE / "fabriccrypt_explainer_5min.mp4"
THUMB_PNG = HERE / "fabriccrypt_explainer_thumbnail.png"

FRAMES_DIR.mkdir(exist_ok=True)
WORK_DIR.mkdir(exist_ok=True)

W, Hpx = 1920, 1080
FPS = 30  # final container fps (we use -loop 1 so this is the OUTPUT fps)

# Colors (RGB)
NAVY = (11, 37, 69)
ACCENT = (19, 196, 163)
WARM = (232, 179, 57)
RED = (215, 38, 61)
GREY = (60, 72, 88)
LIGHT = (244, 246, 250)
LINE = (195, 202, 217)
WHITE = (255, 255, 255)
BG = (8, 22, 40)

# Font discovery
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]
FONT_REG_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]
FONT_MONO_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
]


def font(size, bold=True, mono=False):
    candidates = FONT_MONO_CANDIDATES if mono else (FONT_CANDIDATES if bold else FONT_REG_CANDIDATES)
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


# ---------- helpers ----------
def new_canvas():
    img = Image.new("RGB", (W, Hpx), BG)
    return img, ImageDraw.Draw(img)


def gradient_bg(draw):
    """Subtle vertical gradient background."""
    for y in range(Hpx):
        t = y / Hpx
        r = int(8 + (11 - 8) * t)
        g = int(22 + (37 - 22) * t)
        b = int(40 + (69 - 40) * t)
        draw.line([(0, y), (W, y)], fill=(r, g, b))


def text_centered(draw, xy, txt, fnt, fill=WHITE):
    x, y = xy
    bbox = draw.textbbox((0, 0), txt, font=fnt)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((x - w / 2, y - h / 2), txt, font=fnt, fill=fill)


def draw_chip(draw, cx, cy, size=140, label="APU", color=ACCENT):
    s = size
    # body
    draw.rounded_rectangle([cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2],
                           radius=14, fill=NAVY, outline=LINE, width=2)
    # pins
    n = 7
    pin_len = 12
    for i in range(n):
        t = (i + 1) / (n + 1)
        # top pins
        x = cx - s / 2 + t * s
        draw.line([(x, cy - s / 2), (x, cy - s / 2 - pin_len)], fill=LINE, width=2)
        draw.line([(x, cy + s / 2), (x, cy + s / 2 + pin_len)], fill=LINE, width=2)
        y = cy - s / 2 + t * s
        draw.line([(cx - s / 2, y), (cx - s / 2 - pin_len, y)], fill=LINE, width=2)
        draw.line([(cx + s / 2, y), (cx + s / 2 + pin_len, y)], fill=LINE, width=2)
    # core dot
    draw.ellipse([cx - 18, cy - 18, cx + 18, cy + 18], fill=color)
    text_centered(draw, (cx, cy), label, font(14), fill=WHITE)


def draw_trace(draw, x0, y0, w, h, seed=0, color=ACCENT, n=160, width=3):
    rng = np.random.default_rng(seed)
    # 1/f-ish trace
    samples = rng.normal(0, 1, n)
    # Smoothed
    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    samples = np.convolve(samples, kernel, mode="same")
    base = np.sin(np.linspace(0, 6 * np.pi, n) + seed) * 0.5
    samples = samples * 0.6 + base
    samples = samples / max(abs(samples.min()), abs(samples.max()), 1e-6)
    pts = []
    for i, v in enumerate(samples):
        x = x0 + i * w / (n - 1)
        y = y0 + h / 2 + v * h / 2 * 0.85
        pts.append((x, y))
    draw.line(pts, fill=color, width=width)


def title_bar(draw, txt):
    draw.rectangle([0, 0, W, 110], fill=NAVY)
    draw.text((60, 30), "FabricCrypt", font=font(54), fill=WHITE)
    draw.text((60, 80), txt, font=font(22, bold=False), fill=ACCENT)
    # right corner small
    draw.text((W - 600, 50), "substrate-locked AI attestation",
              font=font(22, bold=False), fill=LIGHT)


def footer_bar(draw, hint=""):
    draw.rectangle([0, Hpx - 60, W, Hpx], fill=NAVY)
    draw.text((60, Hpx - 45), "github.com/ikaros-hive/AMD_gfx1151_energy  -  arXiv:2606.NNNNN [cs.CR]",
              font=font(18, bold=False), fill=LIGHT)
    if hint:
        draw.text((W - 700, Hpx - 45), hint, font=font(18, bold=False), fill=ACCENT)


# ---------- slides ----------
def slide_S0_hook():
    img, d = new_canvas()
    gradient_bg(d)
    # Big metaphor
    d.text((W // 2 - 720, 180), "Imagine an A.I.", font=font(76), fill=WHITE)
    d.text((W // 2 - 720, 280), "you cannot copy.", font=font(76), fill=ACCENT)
    # Two droids (abstract circles)
    cy = 620
    # droid 1
    d.ellipse([520, cy - 90, 700, cy + 90], outline=ACCENT, width=6)
    d.ellipse([580, cy - 40, 640, cy + 20], fill=ACCENT)
    d.text((565, cy + 110), "ikaros", font=font(36), fill=ACCENT)
    # droid 2
    d.ellipse([1220, cy - 90, 1400, cy + 90], outline=WARM, width=6)
    d.ellipse([1280, cy - 40, 1340, cy + 20], fill=WARM)
    d.text((1265, cy + 110), "daedalus", font=font(36), fill=WARM)
    # ARROW + REJECT
    d.line([(720, cy), (1200, cy)], fill=LINE, width=4)
    d.polygon([(1200, cy), (1180, cy - 12), (1180, cy + 12)], fill=LINE)
    d.text((W // 2 - 110, cy - 36), "copy AI", font=font(28, bold=False), fill=LIGHT)
    d.rounded_rectangle([W // 2 - 130, cy + 40, W // 2 + 130, cy + 90], radius=10, fill=RED)
    text_centered(d, (W // 2, cy + 65), "REJECTED", font(28), fill=WHITE)
    d.text((60, Hpx - 100), "FabricCrypt  -  two identical computers, only one runs the model",
           font=font(28, bold=False), fill=LIGHT)
    return img


def slide_S1_problem():
    img, d = new_canvas()
    gradient_bg(d)
    title_bar(d, "The everyday problem")

    d.text((80, 170), "You build an A.I.  Someone copies the file.",
           font=font(40), fill=WHITE)
    d.text((80, 230), "How do you prove WHICH chip ran the inference?",
           font=font(40), fill=ACCENT)

    # Vendor matrix
    schemes = [
        ("Apple PCC",      "needs Secure Enclave + vendor key"),
        ("NVIDIA CC",      "needs DICE + NVIDIA PKI"),
        ("Intel TDX",      "needs Intel silicon + Intel CA"),
        ("AMD SEV-SNP",    "needs PSP + AMD CA"),
        ("TPM 2.0",        "needs discrete TPM + EK cert"),
    ]
    y0 = 360
    for i, (k, v) in enumerate(schemes):
        y = y0 + i * 80
        d.rounded_rectangle([80, y, W - 80, y + 60], radius=10,
                            fill=(20, 38, 64), outline=LINE, width=1)
        d.text((110, y + 14), k, font=font(28), fill=WARM)
        d.text((430, y + 14), v, font=font(26, bold=False), fill=LIGHT)
        d.rounded_rectangle([W - 280, y + 10, W - 110, y + 50], radius=8,
                            fill=(40, 20, 30), outline=RED, width=1)
        text_centered(d, (W - 195, y + 30), "vendor-rooted", font(20), fill=RED)
    footer_bar(d, "We wondered: can we do this without trusting any vendor?")
    return img


def slide_S2_fingerprints():
    img, d = new_canvas()
    gradient_bg(d)
    title_bar(d, "15 fingerprint signals  ->  466-dim live signature  (5 HAL-bypass + 3 KS + 7 board)")
    sigs = [
        ("TSC offset",        "inter-core timing skew",  ACCENT, 11),
        ("Cacheline",         "ping-pong matrix",        WARM,   23),
        ("DRAM refresh",      "memory heartbeat jitter", ACCENT, 47),
        ("Syscall p99.9",     "worst-case kernel tail",  RED,    71),
        ("NVMe queue",        "storage tail latency",    WARM,   97),
        ("GPU clock jitter",  "shader engine drift",     ACCENT, 113),
        ("Multi-zone thermal","8 sensors, gradient",     WARM,   131),
        ("Jacobian dynamics", "temporal coupling",       RED,    149),
        ("PCI topology",      "bus tree hash",           ACCENT, 167),
        ("PCIe link state",   "lane width / speed",      WARM,   181),
        ("USB descriptor",    "vendor/product chain",    ACCENT, 199),
        ("DMI / SMBIOS",      "firmware identity",       WARM,   211),
        ("UCSI power",        "VBUS regulator id",       RED,    227),
        ("amdgpu descriptor", "driver/device topology",  ACCENT, 239),
        ("kernel-boot",       "boot-id / cmdline hash",  WARM,   251),
    ]
    cols, rows = 3, 5
    pad_x, pad_y = 80, 180
    cell_w = (W - 2 * pad_x) // cols
    cell_h = 120
    for i, (name, sub, color, seed) in enumerate(sigs):
        cx = pad_x + (i % cols) * cell_w
        cy = pad_y + (i // cols) * cell_h + 60
        d.rounded_rectangle([cx + 20, cy, cx + cell_w - 20, cy + cell_h - 14],
                            radius=10, fill=(18, 36, 60), outline=color, width=2)
        d.text((cx + 40, cy + 10), name, font=font(24), fill=color)
        d.text((cx + 40, cy + 40), sub, font=font(18, bold=False), fill=LIGHT)
        draw_trace(d, cx + 40, cy + 70, cell_w - 80, 30, seed=seed, color=color, n=120, width=2)
    # Big stat at bottom
    d.rounded_rectangle([80, Hpx - 200, W - 80, Hpx - 80], radius=14,
                        fill=(15, 34, 60), outline=ACCENT, width=2)
    d.text((140, Hpx - 180), "Result:  466-dim signature  ->  100% leave-one-out classification",
           font=font(34), fill=WHITE)
    d.text((140, Hpx - 130), "n = 2 chassis (ikaros + daedalus, AMD Ryzen AI Max+ 395 'Strix Halo')",
           font=font(22, bold=False), fill=LIGHT)
    footer_bar(d, "no software can hide manufacturing variation")
    return img


def slide_S3_crypto():
    img, d = new_canvas()
    gradient_bg(d)
    title_bar(d, "Nonce-bound challenge  -  the chip cannot pre-record an answer")
    # Verifier and prover
    d.rounded_rectangle([80, 200, 520, 600], radius=18, fill=(18, 36, 60), outline=ACCENT, width=2)
    d.text((110, 220), "VERIFIER", font=font(34), fill=ACCENT)
    d.text((110, 270), "picks random 64-bit nonce", font=font(22, bold=False), fill=LIGHT)
    d.text((110, 305), "decides sampling plan:", font=font(22, bold=False), fill=LIGHT)
    for j, line in enumerate(["- which CPU cores to probe",
                              "- which thermal zones",
                              "- which TSC core pairs",
                              "- which sleep durations"]):
        d.text((130, 345 + j * 30), line, font=font(20, bold=False), fill=LIGHT)
    # Nonce visual
    d.text((110, 510), "nonce_t  =  0x9F3E7D1A...", font=font(22, mono=True), fill=WARM)

    d.rounded_rectangle([W - 520, 200, W - 80, 600], radius=18, fill=(18, 36, 60), outline=WARM, width=2)
    d.text((W - 490, 220), "PROVER  (the chip)", font=font(34), fill=WARM)
    d.text((W - 490, 270), "runs the nonce-driven plan", font=font(22, bold=False), fill=LIGHT)
    d.text((W - 490, 305), "returns a 466-dim signature", font=font(22, bold=False), fill=LIGHT)
    d.text((W - 490, 340), "+ HMAC over (output, K_chip)", font=font(22, bold=False), fill=LIGHT)
    d.text((W - 490, 375), "+ Pedersen commitment for ZK", font=font(22, bold=False), fill=LIGHT)
    draw_chip(d, W - 300, 510, size=120, label="APU", color=WARM)

    # Arrow nonce -> prover
    d.line([(520, 350), (W - 520, 350)], fill=ACCENT, width=4)
    d.polygon([(W - 520, 350), (W - 540, 340), (W - 540, 360)], fill=ACCENT)
    d.text((W // 2 - 80, 320), "challenge", font=font(22), fill=ACCENT)
    # Arrow back signature
    d.line([(W - 520, 460), (520, 460)], fill=WARM, width=4)
    d.polygon([(520, 460), (540, 450), (540, 470)], fill=WARM)
    d.text((W // 2 - 90, 470), "signature + proof", font=font(22), fill=WARM)

    # Big security number
    d.rounded_rectangle([80, Hpx - 200, W - 80, Hpx - 80], radius=14,
                        fill=(15, 34, 60), outline=ACCENT, width=2)
    d.text((140, Hpx - 180), "Empirical attack-cost:  ~10^12 modeling samples  ->  random-Hamming distance",
           font=font(30), fill=WHITE)
    d.text((140, Hpx - 130), "Reverse Fuzzy Extractor + Controlled-PUF + HMAC binding  -  10/10 attack gates pass (incl. custom forgery)",
           font=font(22, bold=False), fill=LIGHT)
    footer_bar(d, "ask a witness about a specific moment -- they cannot fake what they did not see")
    return img


def slide_S4_personality():
    img, d = new_canvas()
    gradient_bg(d)
    title_bar(d, "Personality experiment  -  same architecture, same data, different chip")

    # Two side panels
    for i, (name, color, sample, stat) in enumerate([
        ("ikaros",
         ACCENT,
         ["The signal arrives, and the chip",
          "answers calmly: I am here, and I",
          "have been here for some time now.",
          "The pattern is mine to keep."],
         "avg length 198 tokens"),
        ("daedalus",
         WARM,
         ["Signal in.  Chip responds.",
          "I am here. Pattern is mine.",
          "",
          ""],
         "avg length 133 tokens"),
    ]):
        x0 = 80 + i * (W // 2 - 40)
        d.rounded_rectangle([x0, 180, x0 + (W // 2 - 120), Hpx - 280],
                            radius=14, fill=(18, 36, 60), outline=color, width=2)
        d.text((x0 + 30, 200), name, font=font(40), fill=color)
        d.text((x0 + 30, 250), "same model architecture, same data", font=font(20, bold=False), fill=LIGHT)
        # sample text
        for j, ln in enumerate(sample):
            d.text((x0 + 30, 320 + j * 40), '"' + ln + '"' if ln else "",
                   font=font(24, mono=True), fill=WHITE)
        # stat
        d.rounded_rectangle([x0 + 30, Hpx - 380, x0 + (W // 2 - 150), Hpx - 320], radius=10,
                            fill=(10, 24, 44), outline=color, width=1)
        d.text((x0 + 50, Hpx - 365), stat, font=font(26), fill=color)

    # Bottom: honest result
    d.rounded_rectangle([80, Hpx - 230, W - 80, Hpx - 80], radius=14,
                        fill=(40, 30, 20), outline=WARM, width=2)
    d.text((140, Hpx - 210), "Pre-registered gate: 75%  -  Observed: 66.4%  (CI 0.619-0.705, p << 0.001 vs chance)",
           font=font(28), fill=WARM)
    d.text((140, Hpx - 165), "Detectable divergence, but NOT a top-bar breakthrough. We report the null honestly.",
           font=font(22, bold=False), fill=LIGHT)
    d.text((140, Hpx - 125), "The substrate leaves a stylistic footprint. How strong is still open.",
           font=font(22, bold=False), fill=LIGHT)
    footer_bar(d, "identical twins raised in different rooms")
    return img


def slide_S5_why():
    img, d = new_canvas()
    gradient_bg(d)
    title_bar(d, "Why this matters  -  first per-die AI attestation primitive (n=2 chassis) on commodity HW")
    rows = [
        ("Verifiable inference origin",
         "prove this output came from chip X, not a stolen copy"),
        ("AI insurance / audit",
         "insurer can verify the deployment really runs where it claims"),
        ("Sybil-resistant federated learning",
         "each peer proves they are a distinct physical device, no SGX/TDX needed"),
        ("Substrate-locked models",
         "weights that, by physics, can only inference on one chassis"),
    ]
    y0 = 200
    for i, (title, body) in enumerate(rows):
        y = y0 + i * 140
        d.rounded_rectangle([80, y, W - 80, y + 110], radius=14,
                            fill=(18, 36, 60), outline=ACCENT, width=2)
        d.ellipse([110, y + 30, 170, y + 90], fill=ACCENT)
        text_centered(d, (140, y + 60), str(i + 1), font(36), fill=NAVY)
        d.text((200, y + 18), title, font=font(34), fill=WHITE)
        d.text((200, y + 65), body, font=font(24, bold=False), fill=LIGHT)
    # Bottom stat
    d.rounded_rectangle([80, Hpx - 160, W - 80, Hpx - 80], radius=14,
                        fill=(15, 34, 60), outline=ACCENT, width=2)
    d.text((140, Hpx - 140), "End-to-end sign + verify: median 1.12 ms, p99 2.79 ms",
           font=font(28), fill=WHITE)
    footer_bar(d, "no Secure Enclave, no TPM, no vendor key -- ~$700 Mini-PC")
    return img


def slide_S6_honest():
    img, d = new_canvas()
    gradient_bg(d)
    title_bar(d, "Honest limitations")

    items = [
        ("n = 2 chassis",
         "ikaros + daedalus. We need many more for a robust population claim."),
        ("Personality gate not passed",
         "pre-registered 0.75, observed 0.664. Reported as null."),
        ("Attestation, not confidentiality",
         "proves WHO ran the inference, not WHAT the inference was."),
        ("Strix Halo only",
         "validated on AMD Ryzen AI Max+ 395 / gfx1151. Other SKUs are future work."),
    ]
    y0 = 200
    for i, (k, v) in enumerate(items):
        y = y0 + i * 130
        d.rounded_rectangle([80, y, W - 80, y + 100], radius=12,
                            fill=(40, 30, 20), outline=WARM, width=2)
        d.text((120, y + 18), k, font=font(32), fill=WARM)
        d.text((120, y + 60), v, font=font(22, bold=False), fill=LIGHT)
    # CTA
    d.rounded_rectangle([80, Hpx - 220, W - 80, Hpx - 80], radius=14,
                        fill=(15, 34, 60), outline=ACCENT, width=2)
    d.text((140, Hpx - 200), "Please reproduce.  Break us.  Push us.",
           font=font(38), fill=ACCENT)
    d.text((140, Hpx - 145), "github.com/ikaros-hive/AMD_gfx1151_energy   -   arXiv:2606.NNNNN  [cs.CR]",
           font=font(22, mono=True), fill=LIGHT)
    footer_bar(d, "")
    return img


def slide_S7_close():
    img, d = new_canvas()
    gradient_bg(d)
    # Centered big closer
    d.text((W // 2 - 760, 220), "Two identical computers.", font=font(72), fill=WHITE)
    d.text((W // 2 - 760, 320), "Only one can run our A.I.", font=font(72), fill=ACCENT)

    # Two chips with diverging traces
    cy = 620
    draw_chip(d, 560, cy, size=180, label="ikaros", color=ACCENT)
    draw_chip(d, W - 560, cy, size=180, label="daedalus", color=WARM)
    draw_trace(d, 740, cy - 30, 440, 60, seed=11, color=ACCENT, width=3)
    draw_trace(d, 740, cy + 10, 440, 60, seed=29, color=WARM, width=3)

    d.text((W // 2 - 350, Hpx - 230), "FabricCrypt", font=font(64), fill=WHITE)
    d.text((W // 2 - 350, Hpx - 160), "substrate-locked AI attestation",
           font=font(28, bold=False), fill=ACCENT)
    d.text((W // 2 - 350, Hpx - 110), "no vendor key  -  no special chip  -  just physics",
           font=font(24, bold=False), fill=LIGHT)
    return img


SLIDE_BUILDERS = {
    "S0_hook":         slide_S0_hook,
    "S1_problem":      slide_S1_problem,
    "S2_fingerprints": slide_S2_fingerprints,
    "S3_crypto":       slide_S3_crypto,
    "S4_personality":  slide_S4_personality,
    "S5_why":          slide_S5_why,
    "S6_honest":       slide_S6_honest,
    "S7_close":        slide_S7_close,
}


def get_audio_duration(p: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(p)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def _apu_temp_c():
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()) / 1000.0
    except Exception:
        return 0.0


def _thermal_wait(cool_to=55.0, hot=70.0, timeout=180):
    import time
    t = _apu_temp_c()
    if t < hot:
        return t
    print(f"  [thermal] {t:.1f}C >= {hot}C, cooling to {cool_to}C...")
    start = time.time()
    while _apu_temp_c() > cool_to and (time.time() - start) < timeout:
        time.sleep(3)
    return _apu_temp_c()


def build_section(name, slide_img: Image.Image, audio: Path, out_mp4: Path):
    png = FRAMES_DIR / f"{name}.png"
    slide_img.save(png, "PNG", optimize=True)
    duration = get_audio_duration(audio)
    _thermal_wait()
    # Add small tail silence padding via -af apad? Easier: pad video to audio duration
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-threads", "2",
        "-loop", "1", "-framerate", str(FPS), "-i", str(png),
        "-i", str(audio),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-shortest",
        "-t", f"{duration + 0.4:.3f}",
        "-r", str(FPS),
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
    return duration


def main():
    if not AUDIO_DIR.exists():
        print("ERROR: audio dir missing. Run generate_tts.py first.", file=sys.stderr)
        sys.exit(1)

    section_clips = []
    durations = {}
    for name in SLIDE_BUILDERS:
        audio = AUDIO_DIR / f"{name}.mp3"
        assert audio.exists(), f"missing {audio}"
        out = WORK_DIR / f"{name}.mp4"
        # Skip if output already exists AND is newer than audio AND newer than this script
        script_path = Path(__file__)
        if (out.exists() and out.stat().st_size > 1000
                and out.stat().st_mtime > audio.stat().st_mtime
                and out.stat().st_mtime > script_path.stat().st_mtime):
            dur = get_audio_duration(audio)
            print(f"  skip {name} (cached): {dur:.2f}s")
        else:
            slide_img = SLIDE_BUILDERS[name]()
            dur = build_section(name, slide_img, audio, out)
            print(f"  built {name}: {dur:.2f}s -> {out.name}")
        durations[name] = dur
        section_clips.append(out)

    # Concat
    concat_list = WORK_DIR / "concat.txt"
    with concat_list.open("w") as f:
        for c in section_clips:
            f.write(f"file '{c}'\n")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy",
        str(OUT_MP4),
    ]
    subprocess.run(cmd, check=True)
    print(f"wrote {OUT_MP4}")

    # Thumbnail = S0 slide (the hook)
    thumb = SLIDE_BUILDERS["S0_hook"]()
    # Add a "WATCH" badge
    td = ImageDraw.Draw(thumb)
    td.rounded_rectangle([W - 460, 50, W - 60, 130], radius=18, fill=RED)
    text_centered(td, (W - 260, 90), "5 min explainer", font(36), fill=WHITE)
    thumb.save(THUMB_PNG, "PNG", optimize=True)
    print(f"wrote {THUMB_PNG}")

    # Stats
    total = sum(durations.values())
    meta = {
        "sections": durations,
        "total_seconds": total,
        "total_mmss": f"{int(total // 60)}:{int(total % 60):02d}",
        "output": str(OUT_MP4),
        "thumbnail": str(THUMB_PNG),
    }
    (HERE / "build_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"total runtime: {meta['total_mmss']}  ({total:.1f}s)")


if __name__ == "__main__":
    main()
