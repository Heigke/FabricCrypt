"""Test: does quasi-2D match lumped if we cap quasi-2D iters at lumped's count?

Theory: lumped stops at the physical root (not the lumped's max_iters cap).
If we limit quasi-2D Newton to take the SAME number of iters lumped did,
we should land at the same place (since quasi-2D's first 1-2 iters are
near-zero-step refinement of the body asymmetry, and the alt-root drift
only happens after 4-5 more iters).
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

biases = [(0.5, 0.4, 0.0), (1.0, 0.4, 0.15), (1.5, 0.6, 0.30)]
print(f"{'bias':<25} {'Id_lumped':>12} {'Id_q2d_full':>12} {'Id_q2d_capped':>14}")
for Vd, VG1, VG2 in biases:
    Vd_t, VG1_t, VG2_t = _t(Vd), _t(VG1), _t(VG2)
    cfg_l = NSRAMCell2TConfig()
    out_l = solve_2t_steady_state(cfg_l, m1, bjt, Vd_t, VG1_t, VG2_t, model_M2=m2)
    cfg_q = NSRAMCell2TConfig(quasi2d_body=True, Rb_SD=1.0, iii_split_alpha=0.5)
    out_q = solve_2t_quasi2d_steady_state(cfg_q, m1, bjt, Vd_t, VG1_t, VG2_t, model_M2=m2)
    cfg_q_cap = NSRAMCell2TConfig(quasi2d_body=True, Rb_SD=1.0, iii_split_alpha=0.5,
                                   newton_max_iters=2)  # cap iters
    out_qc = solve_2t_quasi2d_steady_state(cfg_q_cap, m1, bjt, Vd_t, VG1_t, VG2_t, model_M2=m2)
    print(f"Vd={Vd} VG1={VG1} VG2={VG2:>4}  "
          f"{float(out_l['Id']):12.4e} {float(out_q['Id']):12.4e} {float(out_qc['Id']):14.4e}")
