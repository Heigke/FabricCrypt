"""Generate the brief v3 opener figure: co-design loop diagram showing
how Sebas's measurements + Mario's tape-out + Eric's pyport close a
loop. Output: figures/codesign_loop.pdf

Layout (left → right):
  [Pazos silicon]  →  [pyport]  →  [tape-out spec]
        ↑                              |
        +———[ measurement / refit ]——+

Each box has a one-line label and a small icon hint. The "pyport" box
in the middle shows what the differentiable simulator does: invert
measurements to silicon parameters, sweep cells/topologies, output
network-level performance numbers.

Style: clean, monospaced labels, subtle arrows, no clutter. Print-safe
in B&W (gray fills + black outlines + black arrows). Output PDF only
so it embeds cleanly in pdflatex.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=120)
ax.set_xlim(-2, 102)
ax.set_ylim(-2, 56)
ax.axis("off")

def box(ax, x, y, w, h, title, lines, fill="#f0f0f0", edge="black", lw=1.2):
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=0.8",
        linewidth=lw, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 4.5, title,
             ha="center", va="top", fontsize=10, fontweight="bold")
    for i, ln in enumerate(lines):
        ax.text(x + w/2, y + h - 9 - 4*i, ln,
                 ha="center", va="top", fontsize=7.5)

# Three boxes
box(ax, 4, 18, 24, 26,
    "Pazos silicon",
    ["• 130 nm 2T NS-RAM",
     "• 33-bias I–V family",
     "• transient pulses",
     "• extracted BSIM4 + NPN"],
    fill="#e8eef6")

box(ax, 38, 18, 24, 26,
    "pyport (this proposal)",
    ["• differentiable BSIM4",
     "• Gummel–Poon NPN",
     "• inverts: targets → spec",
     "• 5×10⁶ cell-evals / s"],
    fill="#fff4d6")

box(ax, 72, 18, 24, 26,
    "Mario tape-out",
    ["• cell parameters",
     "• topology / routing",
     "• thick-ox variant",
     "• next TSMC mask"],
    fill="#e6f0e6")

# Arrow: Pazos → pyport (top, "measurements")
arr1 = FancyArrowPatch((28, 30), (38, 30),
                        arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.4, color="black")
ax.add_patch(arr1)
ax.text(33, 31.5, "measurements", ha="center", va="bottom", fontsize=8, style="italic")

# Arrow: pyport → tape-out (top, "spec")
arr2 = FancyArrowPatch((62, 30), (72, 30),
                        arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.4, color="black")
ax.add_patch(arr2)
ax.text(67, 31.5, "spec", ha="center", va="bottom", fontsize=8, style="italic")

# Feedback arrow: tape-out → Pazos (bottom, "next-fab refit")
arr3 = FancyArrowPatch((84, 18), (84, 8),
                        arrowstyle="-", linewidth=1.0, color="dimgray", linestyle="--")
ax.add_patch(arr3)
arr4 = FancyArrowPatch((84, 8), (16, 8),
                        arrowstyle="-", linewidth=1.0, color="dimgray", linestyle="--")
ax.add_patch(arr4)
arr5 = FancyArrowPatch((16, 8), (16, 18),
                        arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.0, color="dimgray", linestyle="--")
ax.add_patch(arr5)
ax.text(50, 5, "fabricate → measure → refit",
         ha="center", va="top", fontsize=8, style="italic", color="dimgray")

# Top header
ax.text(50, 53, "Co-design loop closed by a differentiable cell simulator",
         ha="center", va="top", fontsize=11, fontweight="bold")

# Bottom byline
ax.text(50, 2, "Pazos: substrate · Eric (FEEL): pyport · Lanza: tape-out",
         ha="center", va="bottom", fontsize=7.5, style="italic", color="#444444")

plt.tight_layout(pad=0.2)
out_pdf = OUT / "codesign_loop.pdf"
out_png = OUT / "codesign_loop.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"saved {out_pdf}")
print(f"saved {out_png}")
