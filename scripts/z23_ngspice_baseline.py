"""z23_ngspice_baseline.py — run Sebas's 2T netlist via ngspice, overlay vs data.

This runs the ACTUAL circuit from 2tnsram_simple.asc (PTM130 MOS + parasiticBJT
+ 1fF body cap, no Rb) and compares the resulting Id vs Vd against measured.

Four sweep modes:
 (a) Pure DC sweep (ngspice .dc) — steady-state
 (b) Transient triangle sweep at 0.2 V/s — matches Sebas's measurement
 (c) Fast sweep (10 V/s) — should latch earlier if dynamic
 (d) Our Python model at same params — for cross-check
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
           "results/z23_ngspice_baseline")
OUT.mkdir(parents=True, exist_ok=True)
VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")


def load_raw(fn):
    rows = []
    with open(fn) as f:
        rdr = csv.reader(f); next(rdr)
        for r in rdr:
            try: rows.append((float(r[2]), float(r[0]), float(r[1])))
            except ValueError: continue
    rows.sort()
    t = np.array([r[0] for r in rows])
    Vd = np.array([r[1] for r in rows])
    Id = np.array([r[2] for r in rows])
    peak = int(np.argmax(Vd))
    return t, Vd, Id, peak


def run_ngspice(vg1, vg2, mode="dc", sweep_s=10.0):
    """mode: 'dc' = .dc sweep, 'tran' = transient triangle 0->2->0."""
    model_inc = str(DATA / "PTM130bulkNSRAM.txt")
    bjt_inc = str(DATA / "parasiticBJT.txt")
    # Key: Sebas's schematic has NO Rb. Body floats, only sinks via Q1 base.
    # Parasitic NPN: C=drain, B=body, E=source_internal (Sint between M1 and M2).
    # M2 is access transistor to ground, gated by VG2.
    tran_stop = 2 * sweep_s  # triangle 0->2V->0
    if mode == "dc":
        analysis = ".dc Vd 0 2 0.02\n.print dc i(Vd) v(B) v(Sint) v(D)"
    else:
        analysis = (f".tran 1m {tran_stop} uic\n"
                     f".print tran i(Vd) v(B) v(Sint) v(D) v(Dext)")
    if mode == "dc":
        vd_line = f"Vd Dext 0 DC 0"
    else:
        # triangle: PWL 0 0  sweep_s 2  2*sweep_s 0
        vd_line = (f"Vd Dext 0 PWL(0 0 {sweep_s} 2 {tran_stop} 0)")
    netlist = f"""* NS-RAM 2T cell — Sebas 2tnsram_simple.asc equivalent
.include {model_inc}
.include {bjt_inc}
.param Ln=0.18u Wn=0.36u CBpar=1f

VG1 G  0 DC {vg1:g}
VG2 G2 0 DC {vg2:g}
{vd_line}
Vd_mon Dext D 0    ; ammeter
M1 D  G  Sint B NMOS L={{Ln}}      W={{Wn}}
M2 Sint G2 0   0 NMOS L={{Ln*10}}   W={{Wn}}
Q1 D  B  Sint parasiticBJT area=1u
C1 B 0 {{CBpar}}

.ic V(B)={vg2:g}
{analysis}
.options reltol=1e-4 abstol=1e-14 gmin=1e-14
.end
"""
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(netlist); cir = f.name
    out_file = cir + ".out"
    # Use -b batch, -o for output, -r raw, but easier: use .print + capture stdout
    try:
        r = subprocess.run(["ngspice", "-b", "-o", out_file, cir],
                              capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return None, None, None
    # Parse .out file — ngspice writes column data below header lines
    try:
        with open(out_file) as f:
            txt = f.read()
    except FileNotFoundError:
        return None, None, None
    # Parse columns: look for lines beginning with index integer
    vds, ids, vbs = [], [], []
    in_data = False
    for line in txt.splitlines():
        parts = line.split()
        if len(parts) >= 4:
            try:
                idx = int(parts[0]); float(parts[1])  # first two numeric
                vds.append(float(parts[1]) if mode == "dc" else float(parts[1]))
                ids.append(float(parts[2]))
                vbs.append(float(parts[3]))
                in_data = True
            except ValueError:
                pass
    if not ids:
        return None, None, None
    return np.array(vds), np.array(ids), np.array(vbs)


# Pick three curves
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
    t, Vd_m, Id_m, peak = load_raw(fn)
    Vd_up, Id_up = Vd_m[:peak+1], Id_m[:peak+1]
    Vd_dn, Id_dn = Vd_m[peak:],   Id_m[peak:]

    print(f"\n=== VG1={vg1} VG2={vg2:+.2f} ===")
    Vd_dc, Id_dc, Vb_dc = run_ngspice(vg1, vg2, "dc")
    if Vd_dc is not None:
        print(f"  ngspice DC:    Id range [{np.min(np.abs(Id_dc)):.2e}, "
               f"{np.max(np.abs(Id_dc)):.2e}]  Vb [{Vb_dc.min():.2f},"
               f"{Vb_dc.max():.2f}]")
    else:
        print("  ngspice DC: FAILED")

    ax = axes[col]
    ax.semilogy(Vd_up, np.clip(Id_up, 1e-15, None), "k.-", lw=1.5, ms=3,
                 label="meas up")
    ax.semilogy(Vd_dn, np.clip(Id_dn, 1e-15, None), "k:", lw=1.2,
                 label="meas down")
    if Vd_dc is not None:
        ax.semilogy(Vd_dc, np.clip(np.abs(Id_dc), 1e-22, None), "g-", lw=1.5,
                     label="ngspice DC")
    ax.set_title(f"VG1={vg1}, VG2={vg2:+.2f}")
    ax.set_xlabel("Vd [V]"); ax.set_ylabel("|Id| [A]")
    ax.set_ylim(1e-13, 1e-3)
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
fig.suptitle("Sebas schematic via ngspice (DC) vs measured (up+down)")
fig.tight_layout(); fig.savefig(OUT / "ngspice_vs_meas.png", dpi=140); plt.close(fig)
print(f"\nWrote {OUT/'ngspice_vs_meas.png'}")
