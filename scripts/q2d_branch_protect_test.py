"""Test B.2 branch-protection + B.3 body-leak regularizer.

Three configs at one snapback bias:
  (A) baseline quasi-2D (no protection, no leak) — finds alt-root
  (B) branch-protect ON (max ΔVb=50mV) — should stay near lumped's root
  (C) body-leak 50 GΩ ON (no protect) — alt-root should disappear
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

biases = [(0.5, 0.4, 0.0), (1.0, 0.4, 0.15), (1.5, 0.6, 0.30)]
print(f"{'bias':<24} {'lumped':>12} {'q2d_A':>12} {'q2d_B (protect)':>18} {'q2d_C (leak)':>14}")
for Vd, VG1, VG2 in biases:
    Vd_t, VG1_t, VG2_t = _t(Vd), _t(VG1), _t(VG2)
    cfg_l = NSRAMCell2TConfig()
    out_l = solve_2t_steady_state(cfg_l, m1, bjt, Vd_t, VG1_t, VG2_t, model_M2=m2)
    cfg_A = NSRAMCell2TConfig(quasi2d_body=True, Rb_SD=1e6, iii_split_alpha=0.7)
    out_A = solve_2t_quasi2d_steady_state(cfg_A, m1, bjt, Vd_t, VG1_t, VG2_t, model_M2=m2)
    cfg_B = NSRAMCell2TConfig(quasi2d_body=True, Rb_SD=1e6, iii_split_alpha=0.7,
                               q2d_branch_protect=True, q2d_branch_max_dvb=0.05)
    out_B = solve_2t_quasi2d_steady_state(cfg_B, m1, bjt, Vd_t, VG1_t, VG2_t, model_M2=m2)
    cfg_C = NSRAMCell2TConfig(quasi2d_body=True, Rb_SD=1e6, iii_split_alpha=0.7,
                               q2d_body_leak_R=5e10)
    out_C = solve_2t_quasi2d_steady_state(cfg_C, m1, bjt, Vd_t, VG1_t, VG2_t, model_M2=m2)
    print(f"Vd={Vd} VG1={VG1} VG2={VG2:>4}  "
          f"{float(out_l['Id']):12.4e} {float(out_A['Id']):12.4e} "
          f"{float(out_B['Id']):18.4e} {float(out_C['Id']):14.4e}")
