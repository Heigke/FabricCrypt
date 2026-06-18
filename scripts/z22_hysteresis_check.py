"""z22_hysteresis_check.py — is the sweep bidirectional? does it show hysteresis?"""
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
           "results/z22_hysteresis_check")
OUT.mkdir(parents=True, exist_ok=True)

VG_RE = re.compile(r"VG2=(-?\d+\.?\d*)_VG=(\d+\.\d+)")


def load_raw(fn):
    rows = []
    with open(fn) as f:
        rdr = csv.reader(f); next(rdr)
        for r in rdr:
            try: rows.append((float(r[2]), float(r[0]), float(r[1])))
            except ValueError: continue
    rows.sort()  # by time
    t = np.array([r[0] for r in rows])
    Vd = np.array([r[1] for r in rows])
    Id = np.array([r[2] for r in rows])
    return t, Vd, Id


# Pick 3 representative curves (one per VG1 at VG2=0)
picks = []
for sub in sorted(DATA.iterdir()):
    if not sub.is_dir(): continue
    for fn in sorted(sub.glob("*.csv")):
        m = VG_RE.search(fn.name)
        if not m: continue
        vg2, vg1 = float(m.group(1)), float(m.group(2))
        if abs(vg2) < 0.01:  # VG2 = 0.00
            picks.append((vg1, vg2, fn))

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for col, (vg1, vg2, fn) in enumerate(sorted(picks)[:3]):
    t, Vd, Id = load_raw(fn)
    peak = int(np.argmax(Vd))
    Vd_up, Id_up = Vd[:peak+1], Id[:peak+1]
    Vd_dn, Id_dn = Vd[peak:],   Id[peak:]
    print(f"VG1={vg1} VG2={vg2}: n_up={len(Vd_up)} n_dn={len(Vd_dn)} "
           f"tmax={t.max():.2f}s  Vd_peak={Vd.max():.2f}V")

    ax = axes[0, col]
    ax.semilogy(Vd_up, np.clip(Id_up, 1e-15, None), "b.-", lw=1, ms=3, label="up")
    ax.semilogy(Vd_dn, np.clip(Id_dn, 1e-15, None), "r.-", lw=1, ms=3, label="down")
    ax.set_title(f"VG1={vg1}  VG2={vg2:+.2f}")
    ax.set_xlabel("Vd [V]"); ax.set_ylabel("|Id| [A]")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    ax = axes[1, col]
    ax.plot(t, Vd, "k-", lw=1, label="Vd(t)")
    ax.axvline(t[peak], color="gray", ls="--", lw=0.5)
    ax.set_xlabel("time [s]"); ax.set_ylabel("Vd [V]")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("Raw data: up-sweep (blue) vs down-sweep (red) — any hysteresis?")
fig.tight_layout(); fig.savefig(OUT / "hysteresis.png", dpi=140); plt.close(fig)
print(f"\nWrote {OUT/'hysteresis.png'}")
