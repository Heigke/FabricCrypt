"""Test whether the lumped solver's convergence stop level matters for Id.

Compare default lumped solver vs. tightened lumped solver (smaller Iabstol,
smaller xtol_v, more iters) at a representative subset of Sebas's biases
spanning the 33-row dataset. If Id moves significantly, the 0.654-dec
production fit may be partially a numerical artifact and the entire
baseline needs revisiting before we layer Plan B refactor on top.

Light: 9 biases × 2 solver configs ≈ 18 single-bias solves, ~5 sec wall.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nsram"))

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import (
    NSRAMCell2TConfig, solve_2t_steady_state,
)

SEBAS_M1 = ROOT / "data/sebas_2026_04_22/M1_130DNWFB.txt"
SEBAS_M2 = ROOT / "data/sebas_2026_04_22/M2_130bulkNSRAM.txt"


def _t(x): return torch.tensor(x, dtype=torch.float64)


def main():
    m1 = BSIM4Model.from_spice(SEBAS_M1.read_text(), model_type="nmos")
    m2 = BSIM4Model.from_spice(SEBAS_M2.read_text(), model_type="nmos")
    bjt = GummelPoonNPN.from_sebas_card()

    # Span of the 33-bias dataset: 3 VG1 levels × 3 (Vd, VG2) regimes
    biases = [
        # (label, Vd, VG1, VG2)
        ("VG1=0.2 lo",  0.5, 0.2, 0.0),
        ("VG1=0.2 mid", 1.0, 0.2, 0.15),
        ("VG1=0.2 hi",  1.5, 0.2, 0.3),
        ("VG1=0.4 lo",  0.5, 0.4, 0.0),
        ("VG1=0.4 mid", 1.0, 0.4, 0.15),
        ("VG1=0.4 hi",  1.5, 0.4, 0.3),
        ("VG1=0.6 lo",  0.5, 0.6, 0.0),
        ("VG1=0.6 mid", 1.0, 0.6, 0.3),
        ("VG1=0.6 hi",  1.5, 0.6, 0.5),
    ]

    cfg_default = NSRAMCell2TConfig()
    cfg_tight = NSRAMCell2TConfig(
        newton_max_iters=80,
        Iabstol=1e-15,
        Ireltol=1e-6,
        xtol_v=1e-10,
        min_iters=4,
    )

    print(f"{'bias':<20} | {'Id_default':>14}  {'maxR_def':>10} | {'Id_tight':>14}  {'maxR_tight':>10} | {'rel_chg':>10}")
    print("-" * 110)
    big_changes = 0
    for label, Vd, VG1, VG2 in biases:
        Vd_t, VG1_t, VG2_t = _t(Vd), _t(VG1), _t(VG2)
        out_d = solve_2t_steady_state(cfg_default, m1, bjt, Vd_t, VG1_t, VG2_t, model_M2=m2)
        out_t = solve_2t_steady_state(cfg_tight, m1, bjt, Vd_t, VG1_t, VG2_t, model_M2=m2)
        Id_d = float(out_d["Id"]); Id_t = float(out_t["Id"])
        Rd = float(out_d["R_Sint"].abs().max() + out_d["R_B"].abs().max()) if "R_Sint" in out_d else float("nan")
        # If R_Sint not in out, recompute via residual call:
        if torch.isnan(torch.tensor(Rd)):
            from nsram.bsim4_port.nsram_cell_2T import _residuals
            R_S, R_B, _ = _residuals(cfg_default, m1, bjt, Vd_t, VG1_t, VG2_t,
                                     out_d["Vsint"], out_d["Vb"], model_M2=m2)
            Rd = float(max(R_S.abs(), R_B.abs()))
            R_S, R_B, _ = _residuals(cfg_tight, m1, bjt, Vd_t, VG1_t, VG2_t,
                                     out_t["Vsint"], out_t["Vb"], model_M2=m2)
            Rt = float(max(R_S.abs(), R_B.abs()))
        else:
            Rt = float(out_t["R_Sint"].abs().max() + out_t["R_B"].abs().max())
        rel = (Id_t - Id_d) / max(abs(Id_d), 1e-15)
        flag = " <-- BIG" if abs(rel) > 0.10 else ""
        if abs(rel) > 0.10:
            big_changes += 1
        print(f"{label:<20} | {Id_d:14.4e}  {Rd:10.2e} | {Id_t:14.4e}  {Rt:10.2e} | {rel:+9.2%}{flag}")

    print("-" * 110)
    print(f"\n{big_changes}/{len(biases)} biases changed by > 10% under tighter convergence.")
    if big_changes > 0:
        print("=> Production 0.654-dec fit IS sensitive to solver convergence stop.")
        print("   Re-fitting with tightened solver may move the baseline before Plan B.")
    else:
        print("=> Production fit is solver-robust. Plan B refactor is safe.")


if __name__ == "__main__":
    main()
