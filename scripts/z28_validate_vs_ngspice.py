"""z28_validate_vs_ngspice.py — compare our Python BSIM4 against ngspice's
official Berkeley BSIM4 (level 14) on the SAME parameters and operating
points. This is the validation we should have done first.

Test: simple NMOS, no parasitic BJT, no floating body — just drain current
as a function of Vds and Vgs at fixed Vbs=0. PTM130 model card.
Sweep Vd ∈ [0, 2V] at Vg ∈ {0.4, 0.6, 0.8, 1.0, 1.2}, compare Id.

If our Python is faithful, log-RMSE should be < 0.1 dec across regimes.
If it's not, the residual reveals which BSIM4 mechanisms we're missing.
"""
from __future__ import annotations
import csv, subprocess, tempfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nsram.bsim4 import BSIM4_PRESETS, drain_current_bsim

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z28_validate_vs_ngspice")
OUT.mkdir(parents=True, exist_ok=True)
MODELS = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
              "scripts/ngspice_models")


def run_ngspice_id_vd(vg, vbs=0.0):
    """Return (Vd_arr, Id_arr) from ngspice, simple NMOS, no BJT."""
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(f"""* Simple NMOS Id-Vd validation
.include {MODELS}/PTM130_ngspice.txt
VG G 0 DC {vg:g}
VS S 0 DC 0
VB B 0 DC {vbs:g}
VD D 0 DC 0
M1 D G S B NMOS L=0.18u W=0.36u
.control
dc Vd 0 2 0.02
wrdata {f.name}.dat i(vd) v(d)
quit
.endc
.end
""")
        cir = f.name
    subprocess.run(["ngspice", "-b", cir], capture_output=True, timeout=30)
    data = np.loadtxt(cir + ".dat")
    Vd = data[:, 2]; Id = -data[:, 1]   # i(vd) is into source; flip sign
    return Vd, Id


def python_id_vd(vg, vbs=0.0, n=101):
    """Same sweep through our nsram drain_current_bsim."""
    p = BSIM4_PRESETS["ns_ram_130nm_pazos"]
    Vd = np.linspace(0, 2, n)
    Vbs = np.full_like(Vd, vbs)
    Vgs = np.full_like(Vd, vg)
    Id, _ = drain_current_bsim(Vgs, Vd, Vbs, p)
    return Vd, np.asarray(Id)


def main():
    Vgs = [0.4, 0.6, 0.8, 1.0, 1.2]
    fig, axes = plt.subplots(2, len(Vgs), figsize=(15, 6.5),
                              sharex=True, sharey="row")
    print(f"{'Vg':>6}  {'Id_max ngspice':>15}  {'Id_max python':>15}  "
           f"{'log-RMSE':>10}  {'note':>30}")
    for col, vg in enumerate(Vgs):
        Vd_ng, Id_ng = run_ngspice_id_vd(vg)
        Vd_py, Id_py = python_id_vd(vg, n=len(Vd_ng))
        # Linear-Id plot
        ax = axes[0, col]
        ax.plot(Vd_ng, Id_ng * 1e6, "k-", lw=1.5, label="ngspice")
        ax.plot(Vd_py, Id_py * 1e6, "g--", lw=1.5, label="our python")
        ax.set_title(f"VG={vg}V"); ax.grid(alpha=0.3)
        if col == 0: ax.set_ylabel("Id [µA]")
        ax.legend(fontsize=7)
        # Log-Id plot
        ax = axes[1, col]
        ax.semilogy(Vd_ng, np.clip(Id_ng, 1e-15, None), "k-", lw=1.5)
        ax.semilogy(Vd_py, np.clip(Id_py, 1e-15, None), "g--", lw=1.5)
        ax.set_xlabel("Vd [V]"); ax.grid(alpha=0.3, which="both")
        if col == 0: ax.set_ylabel("|Id| [A]")
        # Log-RMSE
        m = (Id_ng > 1e-13) & (Id_py > 0)
        if m.sum() > 5:
            r = float(np.sqrt(np.mean((np.log10(Id_ng[m]) - np.log10(Id_py[m]))**2)))
        else:
            r = float("nan")
        # Saturation magnitude ratio
        ratio = Id_py.max() / max(Id_ng.max(), 1e-30)
        note = f"py/ng={ratio:.2f}× at Vd=2V"
        print(f"{vg:>6.2f}  {Id_ng.max():>15.2e}  {Id_py.max():>15.2e}  "
               f"{r:>10.3f}  {note:>30}")
    fig.suptitle("Validation: our Python BSIM4 vs ngspice Berkeley BSIM4 "
                  "(PTM130, Sebas's M1, Vbs=0)\n"
                  "If implementation is faithful: green ≈ black everywhere",
                  fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "validation.png", dpi=140); plt.close(fig)
    print(f"\nWrote {OUT/'validation.png'}")


if __name__ == "__main__":
    main()
