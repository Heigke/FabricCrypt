"""z81_demo_2t_cell — Sanity demo for the 2T NS-RAM cell model.

Loads Sebas's PTM130 BSIM4 card + parasiticBJT card, runs the 2T cell
at a few representative biases, and prints Id values to verify:
  - Id increases monotonically with VG1 (more drive)
  - Id is ACTUALLY sensitive to VG2 (because M2 is now in the topology,
    not faked through a vth0 shift)
  - Newton converges everywhere in <30 iters

Usage:
    HSA_OVERRIDE_GFX_VERSION=11.0.0 python scripts/z81_demo_2t_cell.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

# Ensure local nsram package is importable
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "nsram"))

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import (
    NSRAMCell2TConfig, solve_2t_steady_state,
)


SEBAS_CARD = REPO / "data" / "sebas_2026_04_22" / "PTM130bulkNSRAM.txt"


def main():
    print(f"=== z81: 2T NS-RAM cell sanity demo ===")
    print(f"Card: {SEBAS_CARD}")
    model = BSIM4Model.from_spice(SEBAS_CARD.read_text(), model_type="nmos")
    bjt = GummelPoonNPN.from_sebas_card()
    cfg = NSRAMCell2TConfig(
        Ln=180e-9, Wn=360e-9, M2_length_factor=10.0,
        T_C=27.0,
        use_iii=True, use_gidl=True, use_bjt=True, use_igb=True, use_diode=True,
        newton_max_iters=30, newton_tol=1e-12,
    )

    print(f"\nGeometry: M1 L={cfg.Ln*1e9:.0f}nm W={cfg.Wn*1e9:.0f}nm | "
          f"M2 L={cfg.Ln*cfg.M2_length_factor*1e9:.0f}nm W={cfg.Wn*1e9:.0f}nm")
    print(f"Newton: max_iters={cfg.newton_max_iters} tol={cfg.newton_tol:.0e}\n")

    Vd_grid = [0.5, 1.0, 1.5]
    VG1_vals = [0.2, 0.4, 0.6]
    VG2_vals = [-0.2, 0.0, 0.2]

    fmt = "{:>5} {:>5} {:>6} | {:>11} {:>11} {:>11} {:>9} {:>5} {:>4}"
    print(fmt.format("Vd", "VG1", "VG2", "Id [A]", "Ids_M1", "Ic_Q1",
                     "Vsint", "Vb", "n"))
    print("-" * 90)

    for VG1 in VG1_vals:
        for VG2 in VG2_vals:
            for Vd in Vd_grid:
                Vd_t = torch.tensor([Vd], dtype=torch.float64)
                VG1_t = torch.tensor([VG1], dtype=torch.float64)
                VG2_t = torch.tensor([VG2], dtype=torch.float64)
                out = solve_2t_steady_state(cfg, model, bjt,
                                            Vd=Vd_t, VG1=VG1_t, VG2=VG2_t)
                Id = float(out["Id"])
                IdsM1 = float(out["Ids_M1"])
                Ic = float(out["Ic_Q1"])
                Vs = float(out["Vsint"])
                Vb = float(out["Vb"])
                n = out["niter"]
                conv = bool(out["converged"].all())
                tag = "" if conv else " *"
                print(fmt.format(
                    f"{Vd:.2f}", f"{VG1:.2f}", f"{VG2:+.2f}",
                    f"{Id:+.3e}", f"{IdsM1:+.3e}", f"{Ic:+.3e}",
                    f"{Vs:+.3f}", f"{Vb:+.3f}", f"{n}{tag}",
                ))

    print("\n* = Newton not converged at this point")
    print("\nExpected sanity checks:")
    print(" - Id increases with VG1 at fixed (Vd, VG2)")
    print(" - Id is *not* identical across VG2 values (real M2-gate effect)")
    print(" - Vsint is between 0 and Vd")
    print(" - All points converged in <30 iterations\n")


if __name__ == "__main__":
    main()
