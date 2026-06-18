"""course_figs.py — generate all ~12 figures for the NS-RAM self-study course.

Run once: python scripts/course_figs.py
Outputs: docs/course/figures/fig_XX_*.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, FancyArrowPatch, Circle
from matplotlib.patches import FancyBboxPatch

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/docs/course/figures")
OUT.mkdir(parents=True, exist_ok=True)

GREY = "#666"; BLUE = "#2a6fb5"; RED = "#c0392b"; GREEN = "#27ae60"
YEL = "#f1c40f"; PURP = "#8e44ad"


# ──────────────────────────────────────────────────────────────
# Module 1 — semiconductor doping & band diagram
# ──────────────────────────────────────────────────────────────
def fig01_doping():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    # Left: band diagram
    Ec_i, Ev_i = 1.0, 0.0
    for s, (Ec, Ev, Ef, label) in zip([0, 1, 2], [
        (Ec_i, Ev_i, 0.55, "intrinsic"),
        (Ec_i, Ev_i, 0.85, "n-type"),
        (Ec_i, Ev_i, 0.15, "p-type"),
    ]):
        xs = [s*1.8, s*1.8+1.3]
        ax[0].plot(xs, [Ec, Ec], "k-", lw=2)
        ax[0].plot(xs, [Ev, Ev], "k-", lw=2)
        ax[0].plot(xs, [Ef, Ef], "--", color=RED, lw=1.5)
        ax[0].text(s*1.8+0.65, -0.15, label, ha="center", fontsize=9)
        if s == 0:
            ax[0].text(s*1.8+1.35, Ec, "$E_C$", va="center", fontsize=9)
            ax[0].text(s*1.8+1.35, Ev, "$E_V$", va="center", fontsize=9)
            ax[0].text(s*1.8+1.35, Ef, "$E_F$", va="center", color=RED, fontsize=9)
    ax[0].set_xlim(-0.3, 6.3); ax[0].set_ylim(-0.35, 1.25)
    ax[0].axis("off")
    ax[0].set_title("(a) Band diagram — doping shifts $E_F$")

    # Right: silicon lattice cartoon with donor/acceptor atoms
    ax[1].set_aspect("equal")
    for i in range(4):
        for j in range(3):
            ax[1].plot(i, j, "o", color="#aaa", ms=14)
    # donor (n-type)
    ax[1].plot(1.5, 2.5, "o", color=BLUE, ms=18)
    ax[1].text(1.5, 2.5, "P", ha="center", va="center", fontsize=9, color="white", weight="bold")
    ax[1].annotate("free $e^-$", xy=(1.7, 2.5), xytext=(2.3, 2.7),
                    arrowprops=dict(arrowstyle="->", color=BLUE), fontsize=8, color=BLUE)
    # acceptor (p-type)
    ax[1].plot(1.5, 0.5, "o", color=RED, ms=18)
    ax[1].text(1.5, 0.5, "B", ha="center", va="center", fontsize=9, color="white", weight="bold")
    ax[1].annotate("hole (missing $e^-$)", xy=(1.7, 0.5), xytext=(2.3, 0.3),
                    arrowprops=dict(arrowstyle="->", color=RED), fontsize=8, color=RED)
    ax[1].text(0, -0.6, "Si lattice: donor P $\\to$ n-type · acceptor B $\\to$ p-type", fontsize=9)
    ax[1].set_xlim(-0.5, 4.5); ax[1].set_ylim(-1, 3.3)
    ax[1].axis("off")
    ax[1].set_title("(b) Dopants in silicon")
    fig.tight_layout(); fig.savefig(OUT / "fig01_doping.png", dpi=130); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Module 2 — NMOS cross section + Id-Vgs
# ──────────────────────────────────────────────────────────────
def fig02_nmos():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    a = ax[0]
    # p-substrate
    a.add_patch(Rectangle((0, 0), 10, 2.8, color="#ffd6a0"))
    a.text(5, 0.4, "p-substrate (body)", ha="center", fontsize=9)
    # n+ source / drain
    a.add_patch(Rectangle((1, 1.9), 2, 0.9, color=BLUE, alpha=0.6))
    a.text(2, 2.3, "n$^+$ (S)", ha="center", fontsize=9, color="white", weight="bold")
    a.add_patch(Rectangle((7, 1.9), 2, 0.9, color=BLUE, alpha=0.6))
    a.text(8, 2.3, "n$^+$ (D)", ha="center", fontsize=9, color="white", weight="bold")
    # gate oxide
    a.add_patch(Rectangle((3, 2.8), 4, 0.15, color="#aaa"))
    a.text(5, 2.85, "oxide", ha="center", fontsize=7)
    # poly gate
    a.add_patch(Rectangle((3, 2.95), 4, 0.7, color="#444"))
    a.text(5, 3.3, "gate (G)", ha="center", fontsize=9, color="white")
    # terminals
    for (x, t) in [(2, "S"), (5, "G"), (8, "D")]:
        a.plot([x, x], [3.7 if t == "G" else 2.8, 4.2], "k-", lw=1.5)
        a.text(x, 4.4, t, ha="center", fontsize=11, weight="bold")
    # body
    a.plot([5, 5], [0, -0.4], "k-", lw=1.5)
    a.text(5, -0.7, "B (body)", ha="center", fontsize=10, weight="bold")
    # inversion channel (dashed)
    a.plot([3, 7], [2.8, 2.8], "--", color=GREEN, lw=2)
    a.text(5, 2.6, "channel (when $V_G>V_{th}$)", ha="center", fontsize=8, color=GREEN)
    a.set_xlim(-0.5, 10.5); a.set_ylim(-1.1, 4.9); a.axis("off")
    a.set_title("(a) NMOS cross-section — four terminals: D/G/S/B")

    # Right — Id vs Vg showing subthreshold + above-threshold
    a = ax[1]
    Vg = np.linspace(0, 1.2, 200)
    Vth = 0.5
    Ids = np.where(Vg < Vth,
                    1e-12 * np.exp((Vg - Vth) / 0.04),
                    1e-12 + 5e-4 * (Vg - Vth)**2)
    a.semilogy(Vg, Ids, BLUE, lw=2)
    a.axvline(Vth, color=RED, ls="--", lw=1)
    a.text(Vth + 0.02, 1e-11, "$V_{th}$", color=RED, fontsize=11)
    a.text(0.15, 1e-9, "subthreshold\n(exp)", fontsize=8, color=GREY)
    a.text(0.85, 1e-5, "strong\ninversion\n($\\propto V_{gt}^2$)", fontsize=8, color=GREY)
    a.set_xlabel("$V_{GS}$ [V]"); a.set_ylabel("$I_D$ [A]")
    a.set_ylim(1e-13, 1e-3); a.grid(alpha=0.3, which="both")
    a.set_title("(b) $I_D$–$V_{GS}$: the $V_{th}$ transition")
    fig.tight_layout(); fig.savefig(OUT / "fig02_nmos.png", dpi=130); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Module 3 — compact-model hierarchy + BSIM4 equations overview
# ──────────────────────────────────────────────────────────────
def fig03_compact():
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # Hierarchy: physics -> TCAD -> compact -> circuit
    boxes = [
        (0.5, 3.5, "Physics\n(drift-diffusion,\nPoisson eq.)", "#ffe0e0"),
        (3.5, 3.5, "TCAD\n(Sentaurus, DEVSIM)\nmesh, 2D/3D", "#fff1cc"),
        (6.5, 3.5, "Compact model\n(BSIM4, PSP, HiSIM)\nclosed-form eq.", "#d3f5d3"),
        (6.5, 0.8, "Circuit simulator\n(ngspice, spectre)\nmillions of devices", "#cce6ff"),
    ]
    for (x, y, txt, c) in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 2.8, 1.2, boxstyle="round,pad=0.1",
                                        facecolor=c, edgecolor="black"))
        ax.text(x+1.4, y+0.6, txt, ha="center", va="center", fontsize=9)
    # Arrows
    def arrow(x0, y0, x1, y1, label=None):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="->", lw=1.5))
        if label:
            ax.text((x0+x1)/2, (y0+y1)/2+0.1, label, ha="center", fontsize=8, color=GREY)
    arrow(3.3, 4.1, 3.5, 4.1, "approx")
    arrow(6.3, 4.1, 6.5, 4.1, "fit")
    arrow(7.9, 3.4, 7.9, 2.0, "embed")
    ax.text(5, 5.2, "Hierarchy of semiconductor modeling", ha="center", fontsize=12, weight="bold")
    ax.text(5, -0.3, "We live here: stock BSIM4 + custom KCL\n"
                       "Pazos/Lanza live here: TCAD + BSIM-SOI",
             ha="center", fontsize=9, color=GREY)
    ax.set_xlim(0, 10); ax.set_ylim(-1, 5.8); ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "fig03_compact.png", dpi=130); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Module 4 — floating body: history effect + kink
# ──────────────────────────────────────────────────────────────
def fig04_floatingbody():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    a = ax[0]
    # NMOS on SOI / floating-bulk
    a.add_patch(Rectangle((0, 0), 10, 1.2, color="#e0e0e0"))
    a.text(5, 0.3, "buried oxide / isolation", ha="center", fontsize=9)
    a.add_patch(Rectangle((0, 1.2), 10, 1.6, color="#ffd6a0"))
    a.text(5, 1.6, "floating p-body (no contact!)", ha="center", fontsize=9, weight="bold")
    a.add_patch(Rectangle((1, 1.9), 2, 0.9, color=BLUE, alpha=0.6))
    a.text(2, 2.3, "n$^+$ (S)", ha="center", fontsize=9, color="white", weight="bold")
    a.add_patch(Rectangle((7, 1.9), 2, 0.9, color=BLUE, alpha=0.6))
    a.text(8, 2.3, "n$^+$ (D)", ha="center", fontsize=9, color="white", weight="bold")
    a.add_patch(Rectangle((3, 2.8), 4, 0.15, color="#aaa"))
    a.add_patch(Rectangle((3, 2.95), 4, 0.7, color="#444"))
    # Charge accumulation
    for i, x in enumerate([4, 4.8, 5.6, 6.4]):
        a.plot(x, 2.4, "+", color=RED, ms=14, mew=2)
    a.text(5, 2.0, "holes accumulate → $V_b\\uparrow$", ha="center", fontsize=8, color=RED)
    a.set_xlim(-0.5, 10.5); a.set_ylim(-0.3, 3.8); a.axis("off")
    a.set_title("(a) Floating-body MOSFET — holes can build up in bulk")

    a = ax[1]
    Vd = np.linspace(0, 2, 200)
    # classic kink: smooth saturation, then kink at Vd~1.2V going up sharply
    Id_no_kink = 5e-5 * (1 - np.exp(-Vd/0.3)) + 1e-7
    Id_kink = Id_no_kink + 2e-5 * (Vd > 1.2) * (Vd - 1.2)**2
    a.plot(Vd, Id_no_kink*1e6, color=GREY, lw=2, ls="--", label="bulk MOSFET\n(no kink)")
    a.plot(Vd, Id_kink*1e6, color=BLUE, lw=2, label="floating-body\n(with kink)")
    a.axvline(1.2, color=RED, ls=":", lw=1)
    a.text(1.25, 10, "kink:\nBJT kicks in", color=RED, fontsize=9)
    a.set_xlabel("$V_{DS}$ [V]"); a.set_ylabel("$I_D$ [µA]")
    a.grid(alpha=0.3); a.legend(fontsize=8, loc="upper left")
    a.set_title("(b) The 'kink effect' — signature of floating body")
    fig.tight_layout(); fig.savefig(OUT / "fig04_floatingbody.png", dpi=130); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Module 5 — parasitic BJT inside NMOS
# ──────────────────────────────────────────────────────────────
def fig05_parasitic_bjt():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    a = ax[0]
    # Cross section with overlaid BJT symbol
    a.add_patch(Rectangle((0, 0), 10, 2.8, color="#ffd6a0"))
    a.text(5, 0.3, "p-body (base)", ha="center", fontsize=9, weight="bold")
    a.add_patch(Rectangle((1, 1.9), 2, 0.9, color=BLUE, alpha=0.6))
    a.text(2, 2.3, "n$^+$ source\n(emitter)", ha="center", fontsize=8, color="white", weight="bold")
    a.add_patch(Rectangle((7, 1.9), 2, 0.9, color=BLUE, alpha=0.6))
    a.text(8, 2.3, "n$^+$ drain\n(collector)", ha="center", fontsize=8, color="white", weight="bold")
    # BJT current arrow
    a.annotate("", xy=(7.2, 2.4), xytext=(2.8, 2.4),
                 arrowprops=dict(arrowstyle="->", color=RED, lw=2.5))
    a.text(5, 2.55, "NPN conduction", ha="center", color=RED, fontsize=9, weight="bold")
    a.text(5, 1.5, "If body is forward-biased ($V_B > 0$) → NPN turns on",
             ha="center", fontsize=8, color=GREY)
    a.set_xlim(-0.5, 10.5); a.set_ylim(-0.3, 3.4); a.axis("off")
    a.set_title("(a) The NMOS contains a hidden NPN (emitter=S, base=body, collector=D)")

    # Right: Gummel plot
    a = ax[1]
    Vbe = np.linspace(0.2, 0.9, 200)
    Vt = 0.0259
    IS = 5e-15
    BF = 100
    Ic = IS * np.exp(Vbe/Vt)
    Ib = Ic / BF
    a.semilogy(Vbe, Ic, BLUE, lw=2, label="$I_C$ (collector)")
    a.semilogy(Vbe, Ib, RED, lw=2, label="$I_B$ (base)")
    a.set_xlabel("$V_{BE}$ [V]"); a.set_ylabel("current [A]")
    a.set_ylim(1e-16, 1)
    a.grid(alpha=0.3, which="both"); a.legend(fontsize=9)
    a.text(0.25, 1e-4, "$\\beta_F = I_C/I_B = 100$", fontsize=9, color=GREY)
    a.set_title("(b) Gummel plot — BJT currents vs $V_{BE}$")
    fig.tight_layout(); fig.savefig(OUT / "fig05_parasitic_bjt.png", dpi=130); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Module 6 — impact ionization / Chynoweth
# ──────────────────────────────────────────────────────────────
def fig06_impact_ion():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    a = ax[0]
    # band bend + high-field region
    x = np.linspace(0, 10, 300)
    Ec = np.where(x < 4, 1.0, np.where(x < 6, 1.0 - 1.8*(x-4), -2.6))
    Ev = Ec - 1.1
    a.plot(x, Ec, "k-", lw=2); a.plot(x, Ev, "k-", lw=2)
    a.fill_between(x, Ec, Ec+0.05, color=BLUE, alpha=0.4)
    a.fill_between(x, Ev-0.05, Ev, color=RED, alpha=0.4)
    a.text(2, 1.2, "$E_C$", fontsize=10); a.text(2, -0.3, "$E_V$", fontsize=10)
    # incoming electron, gains energy
    a.annotate("", xy=(5.5, -0.7), xytext=(1, 1.05),
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
    a.text(3, 0.8, "hot electron\ngains KE", color=BLUE, fontsize=8)
    # creates EHP
    a.plot(5.7, -0.6, "o", color=BLUE, ms=12)
    a.plot(5.7, -1.7, "o", color=RED, ms=12)
    a.annotate("", xy=(7.5, -1.8), xytext=(5.7, -1.7),
                 arrowprops=dict(arrowstyle="->", color=RED, lw=2))
    a.text(6.5, -1.5, "hole → body\n(drives Iii)", color=RED, fontsize=8)
    a.set_xlim(0, 10); a.set_ylim(-3.0, 1.6); a.axis("off")
    a.set_title("(a) Impact ionization — hot electron creates e-h pair")

    # Right: Chynoweth Iii vs Vd
    a = ax[1]
    Vd = np.linspace(0.5, 3, 200)
    alpha0 = 7.8e-5; beta0 = 18
    dv = np.clip(Vd - 0.6, 1e-6, None)   # (Vds - Vdsat)
    Iii_norm = alpha0 * dv * np.exp(-beta0/dv)
    Ids = 1e-5 * np.ones_like(Vd)        # assume constant Ids
    Iii = Iii_norm * Ids
    a.semilogy(Vd, Iii, color=BLUE, lw=2)
    a.set_xlabel("$V_{DS}$ [V]"); a.set_ylabel("$I_{ii}$ [A]")
    a.grid(alpha=0.3, which="both")
    a.text(1.4, 1e-12, "$I_{ii} \\propto (V_{DS}-V_{dsat})\\cdot \\exp(-\\beta_0/(V_{DS}-V_{dsat}))$",
             fontsize=8, color=GREY)
    a.set_title("(b) Chynoweth-like $I_{ii}(V_{DS})$ — exponential-threshold")
    fig.tight_layout(); fig.savefig(OUT / "fig06_impact_ion.png", dpi=130); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Module 7 — BSIM-SOI model structure
# ──────────────────────────────────────────────────────────────
def fig07_bsimsoi():
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    # Central body node with four branches
    bx, by = 5, 2.5
    ax.add_patch(Circle((bx, by), 0.35, color="#ffd6a0"))
    ax.text(bx, by, "$V_b$", ha="center", va="center", fontsize=11, weight="bold")
    ax.text(bx, by-0.65, "body node", ha="center", fontsize=8)
    # Branches: Iii (source in), BJT base (sink), body diode (sink), cap to ground
    branches = [
        ((bx-2, by+1.5), "$I_{ii}$\n(imp.ion.)", RED, "in"),
        ((bx+2, by+1.5), "BJT $I_b$\n(base $\\to$ emitter)", BLUE, "out"),
        ((bx-2, by-1.2), "diode leak\n(body-source)", PURP, "out"),
        ((bx+2, by-1.2), "$C_b\\frac{dV_b}{dt}$\n(cap to ground)", GREEN, "out"),
    ]
    for (pos, txt, c, d) in branches:
        ax.add_patch(FancyBboxPatch((pos[0]-0.85, pos[1]-0.35), 1.7, 0.8,
                                        boxstyle="round,pad=0.05",
                                        facecolor="white", edgecolor=c, lw=2))
        ax.text(pos[0], pos[1], txt, ha="center", va="center", fontsize=8, color=c)
        if d == "in":
            ax.annotate("", xy=(bx-0.35, by), xytext=(pos[0]+0.6, pos[1]-0.3),
                         arrowprops=dict(arrowstyle="->", color=c, lw=1.5))
        else:
            ax.annotate("", xy=(pos[0]-0.6, pos[1]-0.3), xytext=(bx+0.35, by),
                         arrowprops=dict(arrowstyle="->", color=c, lw=1.5))
    # KCL equation
    ax.text(bx, 0.4,
             "KCL at body node:   $I_{ii} + I_{GIDL} - I_{b,BJT} - I_{diode} - C_b dV_b/dt = 0$",
             ha="center", fontsize=10,
             bbox=dict(facecolor="#fff8d0", edgecolor="black"))
    ax.text(bx, 5.2, "BSIM-SOI solves this KCL self-consistently\n"
                       "(we were solving it manually → reinventing the wheel)",
             ha="center", fontsize=10, weight="bold")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.8); ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "fig07_bsimsoi.png", dpi=130); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Module 8 — 1T-DRAM operation timeline
# ──────────────────────────────────────────────────────────────
def fig08_1t_dram():
    fig, axes = plt.subplots(3, 1, figsize=(11, 5.5), sharex=True)
    t = np.linspace(0, 20, 1000)
    Vg = np.zeros_like(t); Vd = np.zeros_like(t); state = np.zeros_like(t)
    # Write '1': high Vg, high Vd, impact ionization creates holes
    Vg[(t>1)&(t<3)] = 2.5; Vd[(t>1)&(t<3)] = 2.5
    state[t>=3] = 1
    # Hold
    # Read '1': moderate Vg, low Vd, BJT latches → high Id
    Vg[(t>5)&(t<7)] = 1.0; Vd[(t>5)&(t<7)] = 0.8
    # Write '0': high Vg, negative Vd → removes holes
    Vg[(t>9)&(t<11)] = 2.5; Vd[(t>9)&(t<11)] = -0.5
    state[t>=11] = 0
    # Read '0': moderate Vg, low Vd → no latch, low Id
    Vg[(t>13)&(t<15)] = 1.0; Vd[(t>13)&(t<15)] = 0.8
    # Id output during read
    Id = 1e-12 * np.ones_like(t)
    Id[(t>5)&(t<7)] = 5e-5     # read '1' high
    Id[(t>13)&(t<15)] = 1e-9   # read '0' low

    axes[0].plot(t, Vg, color=BLUE, lw=2); axes[0].set_ylabel("$V_G$ [V]")
    axes[0].grid(alpha=0.3); axes[0].set_ylim(-1, 3)
    axes[0].axvspan(1, 3, alpha=0.15, color="red"); axes[0].text(2, 2.7, "write '1'\n(II → holes)", ha="center", fontsize=7)
    axes[0].axvspan(5, 7, alpha=0.15, color="green"); axes[0].text(6, 2.7, "read '1'", ha="center", fontsize=7)
    axes[0].axvspan(9, 11, alpha=0.15, color="orange"); axes[0].text(10, 2.7, "write '0'\n(eject holes)", ha="center", fontsize=7)
    axes[0].axvspan(13, 15, alpha=0.15, color="green"); axes[0].text(14, 2.7, "read '0'", ha="center", fontsize=7)
    axes[1].plot(t, Vd, color=RED, lw=2); axes[1].set_ylabel("$V_D$ [V]")
    axes[1].grid(alpha=0.3); axes[1].set_ylim(-1.2, 3)
    axes[2].semilogy(t, Id, color=GREEN, lw=2); axes[2].set_ylabel("$I_D$ [A]")
    axes[2].set_xlabel("time [µs]"); axes[2].grid(alpha=0.3, which="both")
    axes[2].set_ylim(1e-13, 1e-3)
    fig.suptitle("1T-DRAM operation: write-hold-read cycles\n"
                   "Bit '1' = body charged (holes) → BJT latches → high $I_D$", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "fig08_1t_dram.png", dpi=130); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Module 9 — Pazos NS-RAM schematic + neuron/synapse modes
# ──────────────────────────────────────────────────────────────
def fig09_nsram():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    # Left: 2T cell schematic
    a = ax[0]
    a.add_patch(Rectangle((3.5, 3.5), 1.5, 1.0, facecolor="#eee", edgecolor="black"))
    a.text(4.25, 4.0, "M1", ha="center", fontsize=11, weight="bold")
    a.add_patch(Rectangle((3.5, 1.5), 1.5, 1.0, facecolor="#eee", edgecolor="black"))
    a.text(4.25, 2.0, "M2", ha="center", fontsize=11, weight="bold")
    # Gate labels
    a.text(2.0, 4.0, "$V_{G1}$", ha="right", fontsize=10); a.annotate("", xy=(3.5, 4.0), xytext=(2.1, 4.0), arrowprops=dict(arrowstyle="->"))
    a.text(2.0, 2.0, "$V_{G2}$", ha="right", fontsize=10); a.annotate("", xy=(3.5, 2.0), xytext=(2.1, 2.0), arrowprops=dict(arrowstyle="->"))
    # Drain (top)
    a.plot([4.25, 4.25], [4.5, 5.2], "k-", lw=1.5); a.text(4.25, 5.4, "$V_D$", ha="center", fontsize=10)
    # Between M1 and M2 — floating body and cap
    a.plot([4.25, 4.25], [3.5, 2.5], "k-", lw=1.5)
    a.plot([4.5, 5.3], [3.0, 3.0], "k-", lw=1)
    a.plot([5.3, 5.3], [3.0, 2.5], "k-", lw=1)
    a.plot([5.3, 5.3], [2.5, 2.0], "k-", lw=1)
    a.text(5.5, 2.7, "$C_b\\approx 1$fF", fontsize=9)
    # Parasitic NPN
    a.plot([5.0, 6.2], [4.0, 4.0], color=RED, lw=1.5)
    a.text(6.4, 4.0, "parasitic\nNPN Q1", color=RED, fontsize=8, va="center")
    # Source (bottom)
    a.plot([4.25, 4.25], [1.5, 0.8], "k-", lw=1.5); a.text(4.25, 0.5, "$V_S\\!=\\!0$", ha="center", fontsize=10)
    a.set_xlim(0, 9); a.set_ylim(0, 6); a.axis("off")
    a.set_title("(a) NS-RAM 2T cell (Pazos 2025)")

    # Right: neuron vs synapse modes
    a = ax[1]
    # Neuron mode: LIF-like — voltage ramp, fires periodically
    t = np.linspace(0, 10, 500)
    v_mem = 0.3 + 0.4*(t - np.floor(t/2)*2)
    fire = (t % 2) > 1.8
    v_mem[fire] = 0.9
    a.plot(t, v_mem, color=PURP, lw=1.5)
    a.set_xlabel("time"); a.set_ylabel("$I_D$", color=PURP)
    a.tick_params(axis='y', labelcolor=PURP)
    a.text(0.5, 0.8, "neuron mode\n(LIF-like firing)", color=PURP, fontsize=9)

    ax2 = a.twinx()
    # Synapse mode: charge trapping → weight change
    t2 = np.linspace(0, 10, 500)
    w = 0.3 + 0.15*np.tanh((t2-3)*1.5) + 0.2*np.tanh((t2-7)*1.5)
    ax2.plot(t2, w, color=GREEN, lw=2, ls="--")
    ax2.set_ylabel("synaptic weight", color=GREEN)
    ax2.tick_params(axis='y', labelcolor=GREEN)
    a.text(6, 0.2, "synapse mode\n(weight update via\ncharge trapping)",
             color=GREEN, fontsize=9)
    a.set_title("(b) Two modes of NS-RAM operation")
    fig.tight_layout(); fig.savefig(OUT / "fig09_nsram.png", dpi=130); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Module 10 — Pazos/Lanza method: TCAD + ngspice stack
# ──────────────────────────────────────────────────────────────
def fig10_method():
    fig, ax = plt.subplots(1, 1, figsize=(11, 5))
    # Three-stage pipeline
    stages = [
        (0.5, "Experimental\nmeasurement\n(probe station)", "#ffe0e0"),
        (3.5, "Sentaurus TCAD\ndevice simulation\n(drift-diffusion)", "#fff1cc"),
        (6.5, "ngspice compact\nmodel (BSIM-SOI\n+ parasitic BJT)", "#d3f5d3"),
        (9.5, "Paper / benchmark\n(write/read, pulse,\nretention)", "#cce6ff"),
    ]
    for (x, txt, c) in stages:
        ax.add_patch(FancyBboxPatch((x-0.2, 1.8), 2.6, 1.4, boxstyle="round,pad=0.1",
                                        facecolor=c, edgecolor="black"))
        ax.text(x+1.1, 2.5, txt, ha="center", va="center", fontsize=9)
    # Arrows
    for x0 in [3.0, 6.0, 9.0]:
        ax.annotate("", xy=(x0+0.3, 2.5), xytext=(x0, 2.5),
                     arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(5.5, 4.4, "Pazos/Lanza 2025 methodology (Nature 640)",
             ha="center", fontsize=12, weight="bold")
    ax.text(5.5, 0.8,
             "• 180 nm + 500 nm commercial CMOS (NOT 130 nm like Sebas's data here)\n"
             "• Sentaurus for physics → ngspice for circuit-level\n"
             "• Zenodo DOI 10.5281/zenodo.13843362 has their files\n"
             "• They did NOT publish extracted compact params — gap in the literature",
             ha="center", fontsize=9)
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 5.2); ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "fig10_method.png", dpi=130); plt.close(fig)


def main():
    fig01_doping()
    fig02_nmos()
    fig03_compact()
    fig04_floatingbody()
    fig05_parasitic_bjt()
    fig06_impact_ion()
    fig07_bsimsoi()
    fig08_1t_dram()
    fig09_nsram()
    fig10_method()
    print(f"Generated {len(list(OUT.glob('*.png')))} figures in {OUT}")


if __name__ == "__main__":
    main()
