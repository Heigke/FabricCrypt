"""B.3 tuning: sweep body-leak Rb from 1 GΩ to 10 TΩ at 3 snapback biases.

Goal: find Rb_leak that erases the alt-root without introducing
systematic Id offset. With branch-protect ALSO enabled (since they
should be combinable).
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
)
SEBAS_M1 = ROOT / "data/sebas_2026_04_22/M1_130DNWFB.txt"
SEBAS_M2 = ROOT / "data/sebas_2026_04_22/M2_130bulkNSRAM.txt"

def _t(x): return torch.tensor(x, dtype=torch.float64)

m1 = BSIM4Model.from_spice(SEBAS_M1.read_text(), model_type="nmos")
m2 = BSIM4Model.from_spice(SEBAS_M2.read_text(), model_type="nmos")
bjt = GummelPoonNPN.from_sebas_card()
bjt.Bf = 9000.0; bjt.Va = 0.55; bjt.Is = 1e-9

biases = [
    ("VG1=0.4 mid", 1.0, 0.4, 0.15),
    ("VG1=0.6 hi",  1.5, 0.6, 0.30),
    ("VG1=0.2 mid", 1.0, 0.2, 0.15),
]
rb_leaks = [0.0, 1e9, 1e10, 1e11, 1e12, 1e13]

for label, Vd, VG1, VG2 in biases:
    Vd_t, VG1_t, VG2_t = _t(Vd), _t(VG1), _t(VG2)
    cfg_l = NSRAMCell2TConfig()
    out_l = solve_2t_steady_state(cfg_l, m1, bjt, Vd_t, VG1_t, VG2_t, model_M2=m2)
    Id_l = float(out_l['Id'])
    print(f"\n{label}: lumped Id = {Id_l:.4e}")
    print(f"  {'Rb_leak':>10}  {'q2d Id':>12}  {'rel_to_lumped':>14}")
    for rb in rb_leaks:
        cfg_q = NSRAMCell2TConfig(quasi2d_body=True, Rb_SD=1e6, iii_split_alpha=0.7,
                                   q2d_branch_protect=True, q2d_branch_max_dvb=0.05,
                                   q2d_body_leak_R=rb)
        out_q = solve_2t_quasi2d_steady_state(cfg_q, m1, bjt, Vd_t, VG1_t, VG2_t, model_M2=m2)
        Id_q = float(out_q['Id'])
        rel = (Id_q - Id_l) / max(abs(Id_l), 1e-15)
        print(f"  {rb:10.0e}  {Id_q:12.4e}  {rel:+14.2%}")
