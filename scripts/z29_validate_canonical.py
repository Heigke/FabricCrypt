"""z29_validate_canonical.py — validate the new bsim4_canonical against ngspice."""
from __future__ import annotations
import csv, subprocess, tempfile
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from nsram.bsim4_canonical import (
    bsim4_drain_current, make_ptm130_nmos, BSIM4ModelParams,
)

OUT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
           "results/z29_validate_canonical")
OUT.mkdir(parents=True, exist_ok=True)
MODELS = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/"
              "scripts/ngspice_models")


def run_ngspice_id_vd(vg, vbs=0.0):
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(f"""* NMOS Id-Vd validation
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
    Vd = data[:, 2]; Id = -data[:, 1]
    return Vd, Id


def python_id_vd(vg, vbs=0.0, n=101):
    p = make_ptm130_nmos()
    Vd = torch.linspace(0, 2, n, dtype=torch.float64)
    Vg_t = torch.full_like(Vd, vg)
    Vb_t = torch.full_like(Vd, vbs)
    Id, _ = bsim4_drain_current(Vg_t, Vd, Vb_t, p)
    return Vd.numpy(), Id.numpy()


def main():
    Vgs = [0.4, 0.6, 0.8, 1.0, 1.2]
    fig, axes = plt.subplots(2, len(Vgs), figsize=(15, 6.5),
                              sharex=True, sharey="row")
    print(f"{'Vg':>6}  {'Id_max ngspice':>15}  {'Id_max python':>15}  "
           f"{'log-RMSE':>10}  {'note':>30}")
    for col, vg in enumerate(Vgs):
        Vd_ng, Id_ng = run_ngspice_id_vd(vg)
        Vd_py, Id_py = python_id_vd(vg, n=len(Vd_ng))
        Id_py = np.clip(Id_py, 0.0, None)  # type: ignore

        ax = axes[0, col]
        ax.plot(Vd_ng, Id_ng * 1e6, "k-", lw=1.5, label="ngspice")
        ax.plot(Vd_py, Id_py * 1e6, "g--", lw=1.5, label="canonical")
        ax.set_title(f"VG={vg}V"); ax.grid(alpha=0.3)
        if col == 0: ax.set_ylabel("Id [µA]")
        ax.legend(fontsize=7)
        ax = axes[1, col]
        ax.semilogy(Vd_ng, np.clip(Id_ng, 1e-15, None), "k-", lw=1.5)
        ax.semilogy(Vd_py, np.clip(Id_py, 1e-15, None), "g--", lw=1.5)
        ax.set_xlabel("Vd [V]"); ax.grid(alpha=0.3, which="both")
        if col == 0: ax.set_ylabel("|Id| [A]")

        m = (Id_ng > 1e-13) & (Id_py > 0)
        if m.sum() > 5:
            r = float(np.sqrt(np.mean((np.log10(Id_ng[m]) - np.log10(Id_py[m]))**2)))
        else:
            r = float("nan")
        ratio = Id_py.max() / max(Id_ng.max(), 1e-30)
        note = f"py/ng={ratio:.2f}× at Vd=2V"
        print(f"{vg:>6.2f}  {Id_ng.max():>15.2e}  {Id_py.max():>15.2e}  "
               f"{r:>10.3f}  {note:>30}")
    fig.suptitle("CANONICAL Python BSIM4 vs ngspice (PTM130, Vbs=0)")
    fig.tight_layout(); fig.savefig(OUT / "validation.png", dpi=140); plt.close(fig)
    print(f"\nWrote {OUT/'validation.png'}")


if __name__ == "__main__":
    main()
