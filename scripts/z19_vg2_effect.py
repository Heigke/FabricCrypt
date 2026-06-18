"""z19_vg2_effect.py — does VG2 shift the curves' shape or just scale?"""
from __future__ import annotations
import csv, re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
            "data/sebas_2026_04_22")
OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z19_vg2_effect")
OUT.mkdir(parents=True, exist_ok=True)

VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")

curves_by_vg1 = {0.2: [], 0.4: [], 0.6: []}
for sub in sorted(DATA.iterdir()):
    if not sub.is_dir(): continue
    for fn in sorted(sub.glob("*.csv")):
        m = VG_RE.search(fn.name)
        if not m: continue
        vg2 = float(m.group(1)); vg1 = float(m.group(2))
        rows = []
        with open(fn) as f:
            rdr = csv.reader(f); next(rdr)
            for r in rdr:
                try: rows.append((float(r[2]), float(r[0]), float(r[1])))
                except ValueError: continue
        rows.sort()
        Vd = np.array([r[1] for r in rows])
        Id = np.array([r[2] for r in rows])
        peak = int(np.argmax(Vd))
        Vd = Vd[:peak + 1]; Id = Id[:peak + 1]
        if vg1 in curves_by_vg1:
            curves_by_vg1[vg1].append((vg2, Vd, Id))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, vg1 in zip(axes, [0.2, 0.4, 0.6]):
    cs = sorted(curves_by_vg1[vg1])
    cmap = plt.cm.coolwarm(np.linspace(0, 1, len(cs)))
    for (vg2, vd, idd), c in zip(cs, cmap):
        ax.semilogy(vd, np.clip(idd, 1e-15, None), color=c, lw=1.2,
                     label=f"VG2={vg2:+.2f}")
    ax.set_title(f"VG1={vg1}  ({len(cs)} VG2 values)")
    ax.set_xlabel("Vd [V]")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6, loc="lower right", ncol=2)
axes[0].set_ylabel("|Id| [A]")
fig.suptitle("Measured I-V vs VG2 (back-gate regime selector)")
fig.tight_layout(); fig.savefig(OUT / "vg2_effect.png", dpi=140); plt.close(fig)
print(f"Wrote {OUT/'vg2_effect.png'}")

# Knee position per curve
print("\nKnee analysis: where is max d(log10 Id)/dVd?")
for vg1 in [0.2, 0.4, 0.6]:
    print(f"\nVG1={vg1}:")
    for vg2, vd, idd in sorted(curves_by_vg1[vg1]):
        log_id = np.log10(np.clip(idd, 1e-15, None))
        slope = np.gradient(log_id, vd)
        imax = np.argmax(slope)
        print(f"  VG2={vg2:+.2f}  knee@Vd={vd[imax]:.2f}V  "
               f"max slope={slope[imax]:.1f} dec/V  "
               f"Id(knee)={idd[imax]:.2e}  Id(max)={idd.max():.2e}")
