"""M3c.2 path B test: does Ic_lat add measurable snapback at high Vd?

Tests at the bias rows where F1.v2 fails (VG1=0.6 VG2=0.0 — the high-residual
snapback row). Sweeps eta_lat ∈ [0, 1] and prints Id / Vb / lateral-collector
contribution as Vd → 2 V.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)

from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t
from nsram.bsim4_port.bjt import GummelPoonNPN

M1, M2 = v1.build_calibrated_models()
bjt = GummelPoonNPN.from_sebas_card(); bjt.Bf = 100.0

# Pick the snapback regime — high VG1, low VG2, sweep Vd to 2V.
VG1, VG2 = 0.6, 0.0
Vd_seq = torch.linspace(0.05, 2.0, 40, dtype=torch.float64)

print(f"Testing snapback at VG1={VG1} VG2={VG2}, Bf=100, Vd 0.05→2.0 V")
print(f"{'eta_lat':>8s} {'Id(Vd=2.0)':>14s} {'Vb(Vd=2.0)':>10s}  {'Id(Vd=1.5)':>14s}")
print("-"*60)
for eta_lat in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             newton_max_iters=50)
    cfg.eta_lat = eta_lat
    VG1_t = torch.tensor([VG1], dtype=torch.float64)
    VG2_t = torch.tensor([VG2], dtype=torch.float64)
    with torch.no_grad():
        out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt,
                          VG1=VG1_t, VG2=VG2_t, Vd_seq=Vd_seq)
    Id_2 = float(out["Id"][-1])
    Vb_2 = float(out["Vb"][-1])
    Id_15 = float(out["Id"][29])  # Vd=1.5 ish
    print(f"{eta_lat:>8.2f} {Id_2:>14.6e} {Vb_2:>10.4f}  {Id_15:>14.6e}")
