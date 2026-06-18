#!/usr/bin/env python3
"""Build the FabricCrypt single-page A4 whitepaper PDF.

Layout (top->bottom on A4 portrait):
  1. Title bar with tagline.
  2. The Problem (text + icon row).
  3. The 3-step solution (visual flow with icons).
  4. Key numbers (big bold callouts).
  5. Twin-Droids strip (two-chassis side-by-side, drawn).
  6. Comparison table (FabricCrypt vs vendor-rooted schemes).
  7. Footer: repo + arXiv placeholder + contact.

All numerical claims sourced from paper_drafts/fabriccrypt_v3.md.
No external download; uses local figures where appropriate.
"""
from pathlib import Path

from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

HERE = Path(__file__).resolve().parent
FIGS = HERE.parent.parent / "figures"
FRAMES = HERE.parent.parent / "demo_video_v2" / "frames_real"

OUT = HERE / "fabriccrypt_onepager.pdf"

# Palette
NAVY = HexColor("#0B2545")
ACCENT = HexColor("#13C4A3")
WARM = HexColor("#E8B339")
RED = HexColor("#D7263D")
GREY = HexColor("#3C4858")
LIGHT = HexColor("#F4F6FA")
LINE = HexColor("#C3CAD9")

W, H = A4  # 595 x 842 pt


def header(c):
    # Top title bar
    c.setFillColor(NAVY)
    c.rect(0, H - 80, W, 80, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 28)
    c.drawString(22 * mm, H - 38, "FabricCrypt")
    c.setFont("Helvetica", 12)
    c.setFillColor(ACCENT)
    c.drawString(22 * mm, H - 56, "Two identical computers. Only one can run our AI.")
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(LIGHT)
    c.drawRightString(
        W - 22 * mm, H - 38,
        "Software-discoverable, vendor-key-free, per-die AI attestation primitive (n=2 chassi)"
    )
    c.drawRightString(
        W - 22 * mm, H - 52,
        "AMD Strix Halo APU  -  Draft v3.1  -  2026-06-01"
    )


def problem(c):
    y = H - 110
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(22 * mm, y, "THE PROBLEM")
    c.setStrokeColor(ACCENT)
    c.setLineWidth(2)
    c.line(22 * mm, y - 3, 60 * mm, y - 3)
    c.setFillColor(GREY)
    c.setFont("Helvetica", 9.5)
    text = c.beginText(22 * mm, y - 16)
    text.setLeading(12)
    for line in [
        "Apple PCC, NVIDIA Confidential Compute, Intel TDX, and AMD SEV-SNP all attest AI inference,",
        "but each requires a vendor PKI rooted in a dedicated secure element (Secure Enclave, DICE, TPM EK).",
        "They authenticate the SKU class, not the individual die: two H100s look identical. And if you do",
        "not trust the vendor's key chain, you have no attestation at all.",
    ]:
        text.textLine(line)
    c.drawText(text)


def step_box(c, x, y, w, h, n, title, body, color):
    c.setStrokeColor(color)
    c.setLineWidth(1.4)
    c.setFillColor(LIGHT)
    c.roundRect(x, y, w, h, 8, fill=1, stroke=1)
    # Number badge
    c.setFillColor(color)
    c.circle(x + 14, y + h - 14, 9, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(x + 14, y + h - 17, str(n))
    # Title
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(x + 28, y + h - 18, title)
    # Body
    c.setFillColor(GREY)
    c.setFont("Helvetica", 7.8)
    t = c.beginText(x + 8, y + h - 34)
    t.setLeading(9.5)
    for line in body:
        t.textLine(line)
    c.drawText(t)


def solution(c):
    y = H - 175
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(22 * mm, y, "OUR 3-STEP SOLUTION")
    c.setStrokeColor(ACCENT)
    c.setLineWidth(2)
    c.line(22 * mm, y - 3, 76 * mm, y - 3)

    # Three side-by-side boxes
    margin = 22 * mm
    gap = 6 * mm
    box_w = (W - 2 * margin - 2 * gap) / 3
    box_h = 86
    box_y = y - 18 - box_h

    step_box(
        c, margin, box_y, box_w, box_h, 1,
        "15 signals (5+3+7)",
        [
            "5 HAL-bypass mu-arch:",
            " TSC, cacheline, DRAM,",
            " syscall p99.9, NVMe tail",
            "3 cross-host KS mu-arch:",
            " GPU jitter, thermal, Jacobian",
            "7 board-level deterministic:",
            " PCI/PCIe/USB/DMI/UCSI/",
            " amdgpu/kernel-boot",
            "-> 466-dim live signature.",
        ],
        ACCENT,
    )
    # Arrow
    c.setStrokeColor(LINE)
    c.setLineWidth(1.2)
    ax = margin + box_w + 2
    ay = box_y + box_h / 2
    c.line(ax, ay, ax + gap - 4, ay)
    c.line(ax + gap - 7, ay + 3, ax + gap - 4, ay)
    c.line(ax + gap - 7, ay - 3, ax + gap - 4, ay)

    step_box(
        c, margin + box_w + gap, box_y, box_w, box_h, 2,
        "Nonce-bound challenge",
        [
            "Verifier picks a fresh",
            "64-bit nonce per request.",
            "Nonce drives the sampling",
            "plan itself: which CPUs,",
            "which thermal zones, which",
            "core pairs, which durations.",
            "Replay & pre-record fail",
            "because the plan is unknown",
            "until challenge arrives.",
        ],
        WARM,
    )
    ax2 = margin + 2 * box_w + gap + 2
    c.line(ax2, ay, ax2 + gap - 4, ay)
    c.line(ax2 + gap - 7, ay + 3, ax2 + gap - 4, ay)
    c.line(ax2 + gap - 7, ay - 3, ax2 + gap - 4, ay)

    step_box(
        c, margin + 2 * (box_w + gap), box_y, box_w, box_h, 3,
        "AI bound to chip",
        [
            "Inference is HMAC-bound",
            "to the per-die key K_chip,",
            "derived via Reverse Fuzzy",
            "Extractor + Controlled-PUF",
            "wrap. Pedersen commitments",
            "give a ZK inference binding.",
            "Move the model to another",
            "chassis -> verification",
            "REJECTS. Per-die, not per-SKU.",
        ],
        RED,
    )


def big_number(c, x, y, w, h, value, label, color):
    c.setStrokeColor(color)
    c.setLineWidth(1.4)
    c.setFillColor(white)
    c.roundRect(x, y, w, h, 6, fill=1, stroke=1)
    c.setFillColor(color)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(x + w / 2, y + h - 22, value)
    c.setFillColor(GREY)
    c.setFont("Helvetica", 7.4)
    # word-wrap label into <=2 lines
    words = label.split()
    line1, line2 = "", ""
    for wd in words:
        if len(line1) + len(wd) + 1 <= 24:
            line1 = (line1 + " " + wd).strip()
        else:
            line2 = (line2 + " " + wd).strip()
    c.drawCentredString(x + w / 2, y + h - 35, line1)
    if line2:
        c.drawCentredString(x + w / 2, y + h - 44, line2)


def key_numbers(c):
    y = H - 295
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(22 * mm, y, "KEY NUMBERS  (n=2, AMD Ryzen AI Max+ 395)")
    c.setStrokeColor(ACCENT)
    c.setLineWidth(2)
    c.line(22 * mm, y - 3, 110 * mm, y - 3)

    margin = 22 * mm
    gap = 4 * mm
    boxw = (W - 2 * margin - 3 * gap) / 4
    boxh = 50
    boxy = y - 18 - boxh
    big_number(c, margin + 0 * (boxw + gap), boxy, boxw, boxh,
               "100%", "LOO per-die classification, 466-dim", ACCENT)
    big_number(c, margin + 1 * (boxw + gap), boxy, boxw, boxh,
               ">=10^12", "empirical attack-cost (no formal reduction)", WARM)
    big_number(c, margin + 2 * (boxw + gap), boxy, boxw, boxh,
               "10 / 10", "attack gates passed (v2.1 battery)", RED)
    big_number(c, margin + 3 * (boxw + gap), boxy, boxw, boxh,
               "0.6%", "replay acceptance rate (lower is better)", NAVY)


def twin_droids(c):
    """Simple side-by-side mini-PC graphic with diverging signal traces."""
    y_top = H - 410
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(22 * mm, y_top, "TWIN DROIDS")
    c.setStrokeColor(ACCENT)
    c.setLineWidth(2)
    c.line(22 * mm, y_top - 3, 50 * mm, y_top - 3)
    c.setFillColor(GREY)
    c.setFont("Helvetica-Oblique", 8.5)
    c.drawString(
        50 * mm, y_top,
        "Two physically identical Strix Halo APUs. Same SKU. Same firmware. Different fingerprints."
    )

    # Frame
    box_y = y_top - 95
    box_h = 80
    margin = 22 * mm
    panel_w = (W - 2 * margin - 6 * mm) / 2

    for i, (name, color) in enumerate([("ikaros", ACCENT), ("daedalus", WARM)]):
        x0 = margin + i * (panel_w + 6 * mm)
        c.setStrokeColor(LINE)
        c.setLineWidth(1)
        c.setFillColor(LIGHT)
        c.roundRect(x0, box_y, panel_w, box_h, 6, fill=1, stroke=1)

        # Mini PC body
        c.setFillColor(NAVY)
        c.roundRect(x0 + 10, box_y + 18, 50, 36, 4, fill=1, stroke=0)
        c.setFillColor(color)
        c.circle(x0 + 35, box_y + 36, 6, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 7)
        c.drawCentredString(x0 + 35, box_y + 34, "APU")
        # Label
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x0 + 68, box_y + box_h - 14, name)
        c.setFillColor(GREY)
        c.setFont("Helvetica", 7.4)
        c.drawString(x0 + 68, box_y + box_h - 25, "AMD Ryzen AI Max+ 395")
        c.drawString(x0 + 68, box_y + box_h - 35, "Strix Halo / gfx1151")

        # Mini signal trace
        import math
        c.setStrokeColor(color)
        c.setLineWidth(1.1)
        px = x0 + 68
        py = box_y + 14
        seg_w = (panel_w - 78) / 40
        seed = 11 if i == 0 else 29
        prev = py + 6
        for k in range(41):
            v = math.sin(k * 0.6 + seed) * 3 + math.sin(k * 1.7 + seed * 2) * 2
            nx = px + k * seg_w
            ny = py + 6 + v
            if k > 0:
                c.line(prev_x, prev, nx, ny)
            prev_x, prev = nx, ny

    # Verdict ribbon
    c.setFillColor(RED)
    c.roundRect(W / 2 - 70, box_y - 20, 140, 16, 4, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(W / 2, box_y - 9, "Cross-chassis verification: REJECT")


def comparison(c):
    y_top = H - 540
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(22 * mm, y_top, "FABRICCRYPT vs VENDOR-ROOTED ATTESTATION")
    c.setStrokeColor(ACCENT)
    c.setLineWidth(2)
    c.line(22 * mm, y_top - 3, 130 * mm, y_top - 3)

    rows = [
        ("Scheme",          "Vendor PKI",  "Dedicated chip", "Per-die",   "Commodity HW"),
        ("Apple PCC",       "Required",    "Secure Enclave", "No (SKU)",  "Apple only"),
        ("NVIDIA CC",       "Required",    "DICE / VCEK",    "No (SKU)",  "H100 / B100"),
        ("Intel TDX",       "Required",    "TDX module",     "No (SKU)",  "Intel only"),
        ("AMD SEV-SNP",     "Required",    "PSP",            "No (SKU)",  "EPYC only"),
        ("TPM 2.0",         "EK cert",     "Discrete TPM",   "Per chip",  "Limited"),
        ("FabricCrypt",     "None",        "None",           "YES",       "Any AMD APU"),
    ]
    margin = 22 * mm
    table_w = W - 2 * margin
    col_w = [table_w * 0.22, table_w * 0.18, table_w * 0.22, table_w * 0.16, table_w * 0.22]
    x0 = margin
    row_h = 14
    y = y_top - 22

    for r, row in enumerate(rows):
        # Background
        if r == 0:
            c.setFillColor(NAVY); c.setStrokeColor(NAVY)
            c.rect(x0, y - row_h, table_w, row_h, fill=1, stroke=0)
        elif row[0] == "FabricCrypt":
            c.setFillColor(HexColor("#E6FAF6"))
            c.rect(x0, y - row_h, table_w, row_h, fill=1, stroke=0)
        elif r % 2 == 1:
            c.setFillColor(LIGHT)
            c.rect(x0, y - row_h, table_w, row_h, fill=1, stroke=0)
        # Text
        cx = x0
        for ci, (cell, cw) in enumerate(zip(row, col_w)):
            if r == 0:
                c.setFillColor(white); c.setFont("Helvetica-Bold", 8.5)
            elif row[0] == "FabricCrypt":
                c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 8.4)
            else:
                c.setFillColor(GREY); c.setFont("Helvetica", 8.2)
            c.drawString(cx + 5, y - row_h + 4, cell)
            cx += cw
        # Bottom rule
        c.setStrokeColor(LINE); c.setLineWidth(0.4)
        c.line(x0, y - row_h, x0 + table_w, y - row_h)
        y -= row_h


def footer(c):
    # Honest caveats line + footer block
    cav_y = 92
    c.setFillColor(GREY)
    c.setFont("Helvetica-Oblique", 7.5)
    c.drawString(
        22 * mm, cav_y,
        "Honest scope: primitive at n=2 chassis (ikaros + daedalus). Empirical attack-cost; no formal cryptographic reduction. Exploratory stylometric divergence (sec 7.L6) is supplementary, not a headline claim. Replication welcome."
    )

    c.setFillColor(NAVY)
    c.rect(0, 0, W, 70, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(22 * mm, 50, "GitHub")
    c.drawString(22 * mm, 30, "arXiv")
    c.drawString(22 * mm, 14, "Contact")
    c.setFont("Helvetica", 9)
    c.drawString(45 * mm, 50, "github.com/ikaros-hive/AMD_gfx1151_energy  (release on camera-ready)")
    c.drawString(45 * mm, 30, "arXiv:2606.NNNNN  [cs.CR]  (submission in progress)")
    c.drawString(45 * mm, 14, "bergvall.eric@gmail.com")

    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", 10)
    c.drawRightString(W - 22 * mm, 50, "Reproduce in one command:")
    c.setFillColor(white)
    c.setFont("Courier-Bold", 8)
    c.drawRightString(W - 22 * mm, 36, "scripts/identity_benchmark/embodiment14d_crypto/run.sh")
    c.setFillColor(LIGHT)
    c.setFont("Helvetica-Oblique", 7)
    c.drawRightString(W - 22 * mm, 18, "FabricCrypt: substrate-locked attestation. No vendor key. No secure element. Just physics.")


def build():
    c = canvas.Canvas(str(OUT), pagesize=A4)
    c.setTitle("FabricCrypt: per-die AI attestation on commodity AMD silicon")
    c.setAuthor("Eric Bergvall")
    c.setSubject("Software-discoverable vendor-key-free per-die AI attestation")
    header(c)
    problem(c)
    solution(c)
    key_numbers(c)
    twin_droids(c)
    comparison(c)
    footer(c)
    c.showPage()
    c.save()
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
