"""Run multi-root probe with PRODUCTION BJT params (Bf=9000, Va=0.55, Is=1e-9).

Hypothesis: the multi-root behavior found in q2d_baseline_check.py was a
default-Bf artifact. At production params the lumped-vs-exact-zero gap
might disappear, in which case quasi-2D will converge to the correct
physical root and Plan A becomes testable.

Test: 9 representative biases × 3 solver configs (lumped default,
lumped tightened, quasi-2D stiff). Look at:
  - Are Id values stable across solver configs at production params?
  - At what residual does lumped stop vs. quasi-2D?
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nsram"))

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import (
    NSRAMCell2TConfig, solve_2t_steady_state, solve_2t_quasi2d_steady_state,
    _residuals,
)

SEBAS_M1 = ROOT / "data/sebas_2026_04_22/M1_130DNWFB.txt"
SEBAS_M2 = ROOT / "data/sebas_2026_04_22/M2_130bulkNSRAM.txt"


def _t(x): return torch.tensor(x, dtype=torch.float64)


def main():
    m1 = BSIM4Model.from_spice(SEBAS_M1.read_text(), model_type="nmos")
    m2 = BSIM4Model.from_spice(SEBAS_M2.read_text(), model_type="nmos")
    bjt = GummelPoonNPN.from_sebas_card()
    # Production overrides (from F6.v4 z91g run)
    bjt.Bf = 9000.0
    bjt.Va = 0.55
    bjt.Is = 1e-9
    print(f"BJT prod params: Bf={bjt.Bf} Va={bjt.Va} Is={bjt.Is}")

    biases = [
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

    cfg_l = NSRAMCell2TConfig()
    cfg_lt = NSRAMCell2TConfig(newton_max_iters=80, Iabstol=1e-15,
                                Ireltol=1e-6, xtol_v=1e-10, min_iters=4)
    cfg_q = NSRAMCell2TConfig(quasi2d_body=True, Rb_SD=1.0, iii_split_alpha=0.5)

    print(f"\n{'bias':<18} | {'lumped':>11} {'maxR_l':>8} | {'tight':>11} {'maxR_t':>8} | {'q2d':>11} {'maxR_q':>8} | flag")
    print("-" * 110)
    multi_root = 0
    for label, Vd, VG1, VG2 in biases:
        Vd_t, VG1_t, VG2_t = _t(Vd), _t(VG1), _t(VG2)
        try:
            out_l = solve_2t_steady_state(cfg_l, m1, bjt, Vd_t, VG1_t, VG2_t, model_M2=m2)
            R_S, R_B, _ = _residuals(cfg_l, m1, bjt, Vd_t, VG1_t, VG2_t,
                                     out_l["Vsint"], out_l["Vb"], model_M2=m2)
            Rl = float(max(R_S.abs(), R_B.abs()))
            Id_l = float(out_l["Id"])
        except Exception as e:
            Id_l, Rl = float("nan"), float("nan")
        try:
            out_t = solve_2t_steady_state(cfg_lt, m1, bjt, Vd_t, VG1_t, VG2_t, model_M2=m2)
            R_S, R_B, _ = _residuals(cfg_lt, m1, bjt, Vd_t, VG1_t, VG2_t,
                                     out_t["Vsint"], out_t["Vb"], model_M2=m2)
            Rt = float(max(R_S.abs(), R_B.abs()))
            Id_t = float(out_t["Id"])
        except Exception:
            Id_t, Rt = float("nan"), float("nan")
        try:
            out_q = solve_2t_quasi2d_steady_state(cfg_q, m1, bjt, Vd_t, VG1_t, VG2_t,
                                                    model_M2=m2)
            Id_q = float(out_q["Id"])
            Rq = float(max(out_q["R_Sint"].abs(), out_q["R_BS"].abs(), out_q["R_BD"].abs()))
        except Exception:
            Id_q, Rq = float("nan"), float("nan")
        # Multi-root if lumped and tight differ by >2x
        ratio = abs(Id_t / Id_l) if Id_l != 0 else float("inf")
        flag = "MULTI" if ratio > 2.0 or ratio < 0.5 else "ok"
        if flag == "MULTI":
            multi_root += 1
        print(f"{label:<18} | {Id_l:11.3e} {Rl:8.1e} | {Id_t:11.3e} {Rt:8.1e} | {Id_q:11.3e} {Rq:8.1e} | {flag}")

    print("-" * 110)
    print(f"{multi_root}/{len(biases)} biases show multi-root (lumped ≠ tight by >2x).")
    if multi_root == 0:
        print("=> At production params, lumped and tight converge to same root.")
        print("   Plan A is testable; quasi-2D should converge to this root too.")
    else:
        print("=> Multi-root persists at production params. Plan A needs deliberate")
        print("   investigation of which root is physical (ngspice cross-check).")


if __name__ == "__main__":
    main()
