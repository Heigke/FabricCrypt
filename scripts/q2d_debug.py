"""Debug: residual evaluation at the lumped solution."""
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
    _residuals, _residuals_quasi2d,
)
SEBAS_M1 = ROOT / "data/sebas_2026_04_22/M1_130DNWFB.txt"
SEBAS_M2 = ROOT / "data/sebas_2026_04_22/M2_130bulkNSRAM.txt"

def _t(x): return torch.tensor(x, dtype=torch.float64)

m1 = BSIM4Model.from_spice(SEBAS_M1.read_text(), model_type="nmos")
m2 = BSIM4Model.from_spice(SEBAS_M2.read_text(), model_type="nmos")
bjt = GummelPoonNPN.from_sebas_card()
Vd, VG1, VG2 = _t(1.5), _t(0.6), _t(0.30)

cfg_l = NSRAMCell2TConfig()
out_l = solve_2t_steady_state(cfg_l, m1, bjt, Vd, VG1, VG2, model_M2=m2)
Vs_l, Vb_l = out_l["Vsint"], out_l["Vb"]
print(f"Lumped: Vsint={float(Vs_l):.6f}  Vb={float(Vb_l):.6f}  Id={float(out_l['Id']):.6e}")

# Eval lumped residual at lumped solution
R_S, R_B, comp = _residuals(cfg_l, m1, bjt, Vd, VG1, VG2, Vs_l, Vb_l, model_M2=m2)
print(f"Lumped residuals at lumped soln: R_S={float(R_S):.3e}  R_B={float(R_B):.3e}")
print(f"  Ids_M1={float(comp['Ids_M1']):.3e}  Iii_M1={float(comp['Iii_M1']):.3e}")
print(f"  I_well_body={float(comp['I_well_body']):.3e}")

# Eval quasi-2D at the lumped solution with various (Rb_SD, alpha)
print("\nEval quasi-2D residuals at lumped (Vb_S=Vb_D=Vb_lumped):")
for rb in [1e3, 1e6]:
    for a in [0.5, 0.7]:
        cfg_q = NSRAMCell2TConfig(quasi2d_body=True, Rb_SD=rb, iii_split_alpha=a)
        R_S, R_BS, R_BD, _ = _residuals_quasi2d(
            cfg_q, m1, bjt, Vd, VG1, VG2, Vs_l, Vb_l, Vb_l, model_M2=m2)
        print(f"  Rb={rb:.0e} α={a}: R_S={float(R_S):.3e}  R_BS={float(R_BS):.3e}  R_BD={float(R_BD):.3e}")

# Now verbose-solve quasi-2D from this warm start
print("\nVerbose quasi-2D solve:")
cfg_q = NSRAMCell2TConfig(quasi2d_body=True, Rb_SD=1.0, iii_split_alpha=0.5)
out_q = solve_2t_quasi2d_steady_state(
    cfg_q, m1, bjt, Vd, VG1, VG2, model_M2=m2,
    Vsint_init=Vs_l.clone(), Vb_S_init=Vb_l.clone(), Vb_D_init=Vb_l.clone(),
    verbose=True)
print(f"  final: Vsint={float(out_q['Vsint']):.6f}  Vb_S={float(out_q['Vb_S']):.6f}  Vb_D={float(out_q['Vb_D']):.6f}  Id={float(out_q['Id']):.3e}")

# Cross-check: evaluate lumped residual at quasi-2D's converged (Vsint, Vb_avg)
Vs_q = out_q['Vsint']; Vb_q = 0.5 * (out_q['Vb_S'] + out_q['Vb_D'])
R_S2, R_B2, comp2 = _residuals(cfg_l, m1, bjt, Vd, VG1, VG2, Vs_q, Vb_q, model_M2=m2)
print(f"\nLumped residuals at quasi-2D's solution (Vsint={float(Vs_q):.4f}, Vb={float(Vb_q):.4f}):")
print(f"   R_S={float(R_S2):.3e}  R_B={float(R_B2):.3e}")
Id_at_q = (comp2['Ids_M1'] + comp2['Ic_Q1'] + comp2.get('Ic_lat', 0.0)
           + comp2.get('Ic_avalanche', 0.0) + comp2['Igidl_M1'] - comp2['Ibd_M1'])
print(f"   Id (lumped formula) at quasi-2D's solution = {float(Id_at_q):.3e}")

# Run lumped solver from quasi-2D's initial guess (override warm start)
print(f"\nLumped solver from quasi-2D's warm start (no Vsint=0.5*Vd default):")
out_l2 = solve_2t_steady_state(
    cfg_l, m1, bjt, Vd, VG1, VG2, model_M2=m2,
    Vsint_init=Vs_q.detach().clone(), Vb_init=Vb_q.detach().clone())
print(f"   Vsint={float(out_l2['Vsint']):.4f}  Vb={float(out_l2['Vb']):.4f}  Id={float(out_l2['Id']):.3e}")
