"""z26_ngspice_gap.py — demonstrate the PTM130 vs Sebas data gap.

Runs Sebas's 2T schematic in ngspice with the public PTM130 card, overlays
against his measurement. The ~6-decade gap is the empirical evidence that
his NDA foundry card contains information absent from the public model.
This justifies why our fitted nsram simulator (with free VTH0/Rb/ALPHA0)
is necessary.
"""
from __future__ import annotations
import csv, re, subprocess, tempfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
             "data/sebas_2026_04_22")
OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z26_ngspice_gap")
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
                  "scripts/ngspice_models")
VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")


def run_ngspice_dc(vg1, vg2, m2_native=False):
    """Return Vd, Id arrays from ngspice DC sweep. m2_native flag replaces
    M2 with a low-Vt device to illustrate how Sebas's NDA card would change
    the picture."""
    m2_line = ("M2 Sint G2 0 0 NMOS_LVT L=1.8u W=0.36u" if m2_native
                 else "M2 Sint G2 0 0 NMOS L=1.8u W=0.36u")
    extra_model = (".model NMOS_LVT NMOS Level=14 Vth0=-0.1 Lint=1.969e-08 "
                    "toxe=4e-009 toxp=4e-009 toxm=4e-009 Rdsw=200 k1=0.4"
                    if m2_native else "")
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(f"""* z26 ngspice gap demo
.include {MODEL_DIR}/PTM130_ngspice.txt
.include {MODEL_DIR}/parasiticBJT_ngspice.txt
{extra_model}
VG1 G  0 DC {vg1:g}
VG2 G2 0 DC {vg2:g}
Vd  D  0 DC 0
M1 D G Sint B NMOS L=0.18u W=0.36u
{m2_line}
Q1 D B Sint parasiticBJT area=1u
C1 B 0 1f
.control
dc Vd 0 2 0.02
wrdata {f.name}.dat i(vd) v(b)
quit
.endc
.end
""")
        cir = f.name
    try:
        subprocess.run(["ngspice", "-b", cir], capture_output=True, timeout=30)
    except Exception:
        return None, None
    try:
        data = np.loadtxt(cir + ".dat")
    except Exception:
        return None, None
    Vd = data[:, 0]; Id = np.abs(data[:, 1])
    return Vd, Id


def load_meas(fn):
    rows = []
    with open(fn) as f:
        rdr = csv.reader(f); next(rdr)
        for r in rdr:
            try: rows.append((float(r[2]), float(r[0]), float(r[1])))
            except ValueError: continue
    rows.sort()
    Vd = np.array([r[1] for r in rows]); Id = np.array([r[2] for r in rows])
    peak = int(np.argmax(Vd))
    return Vd[:peak+1], Id[:peak+1]


# Three representative curves at VG2=0
picks = []
for sub in sorted(DATA.iterdir()):
    if not sub.is_dir(): continue
    for fn in sorted(sub.glob("*.csv")):
        m = VG_RE.search(fn.name)
        if m and abs(float(m.group(1))) < 0.01:
            picks.append((float(m.group(2)), float(m.group(1)), fn))
picks = sorted(set(picks))[:3]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for col, (vg1, vg2, fn) in enumerate(picks):
    Vd_m, Id_m = load_meas(fn)
    Vd_ng, Id_ng = run_ngspice_dc(vg1, vg2, m2_native=False)
    Vd_lvt, Id_lvt = run_ngspice_dc(vg1, vg2, m2_native=True)
    ax = axes[col]
    ax.semilogy(Vd_m, np.clip(Id_m, 1e-15, None), "k-", lw=2, label="Sebas meas")
    if Vd_ng is not None:
        ax.semilogy(Vd_ng, np.clip(Id_ng, 1e-22, None), "r--", lw=1.4,
                     label="ngspice (public PTM130)")
    if Vd_lvt is not None:
        ax.semilogy(Vd_lvt, np.clip(Id_lvt, 1e-22, None), "b:", lw=1.4,
                     label="ngspice (M2=low-Vt proxy)")
    # Compute gap at Vd=2V
    if Vd_ng is not None:
        id_meas_2v = Id_m[np.argmin(np.abs(Vd_m - 2.0))]
        id_ng_2v = Id_ng[-1]
        gap = id_meas_2v / max(id_ng_2v, 1e-30)
        print(f"VG1={vg1}: @Vd=2V  meas={id_meas_2v:.2e}  ngspice={id_ng_2v:.2e}  "
               f"gap={gap:.1e}× ({np.log10(gap):.1f} decades)")
    ax.set_title(f"VG1={vg1}  VG2={vg2:+.2f}")
    ax.set_xlabel("Vd [V]"); ax.set_ylabel("|Id| [A]")
    ax.set_ylim(1e-13, 1e-3)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
fig.suptitle("The gap: Sebas's public schematic cannot reproduce his data\n"
              "→ justifies calibrated nsram simulator with fitted VTH0/Rb/ALPHA0")
fig.tight_layout(); fig.savefig(OUT / "gap.png", dpi=140); plt.close(fig)
print(f"\nWrote {OUT/'gap.png'}")
