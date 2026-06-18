"""M3c.2 path C test: avalanche multiplier toggle.

(1) Gate: cfg.use_lateral_collector=False reproduces F1.v2 bit-identical.
(2) Sensitivity: with toggle=True at the failing snapback bias (VG1=0.6,
    VG2=0.0), sweep BV ∈ [3, 9] V and confirm Id changes meaningfully
    (not zero like path B alone).
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

print("=== Gate (toggle=False reproduces F1.v2) ===")
biases = [(0.6, 0.30), (0.4, 0.10), (0.2, 0.20)]
Vd_seq = torch.linspace(0.05, 1.5, 30, dtype=torch.float64)

cfg_off = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=50)
cfg_on  = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=50)
cfg_on.use_lateral_collector = False  # explicit
print(f"{'bias':>14s} {'Id_off':>14s} {'Id_explicit':>14s} {'reldiff':>10s}")
print("-"*60)
for VG1, VG2 in biases:
    VG1_t = torch.tensor([VG1], dtype=torch.float64)
    VG2_t = torch.tensor([VG2], dtype=torch.float64)
    o0 = forward_2t(cfg_off, model_M1=M1, model_M2=M2, bjt=bjt,
                     VG1=VG1_t, VG2=VG2_t, Vd_seq=Vd_seq)
    o1 = forward_2t(cfg_on, model_M1=M1, model_M2=M2, bjt=bjt,
                     VG1=VG1_t, VG2=VG2_t, Vd_seq=Vd_seq)
    a = float(o0["Id"][-1]); b = float(o1["Id"][-1])
    rd = abs(a - b) / max(abs(a), 1e-30)
    print(f"VG1={VG1} VG2={VG2:.2f} {a:>14.6e} {b:>14.6e} {rd:>10.2e}")

print()
print("=== Avalanche sensitivity at VG1=0.6 VG2=0.0, Bf=100, Vd=2.0 V ===")
Vd_seq2 = torch.linspace(0.05, 2.0, 40, dtype=torch.float64)
VG1_t = torch.tensor([0.6], dtype=torch.float64)
VG2_t = torch.tensor([0.0], dtype=torch.float64)
print(f"{'BV (V)':>8s} {'Id_off (toggle=False)':>22s} {'Id_on (toggle=True)':>22s} {'ratio':>8s}")
print("-"*70)
o_off = forward_2t(cfg_off, model_M1=M1, model_M2=M2, bjt=bjt,
                    VG1=VG1_t, VG2=VG2_t, Vd_seq=Vd_seq2)
Id_off = float(o_off["Id"][-1])
for BV in [3.0, 4.5, 6.0, 7.5, 9.0]:
    cfg_av = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=50)
    cfg_av.use_lateral_collector = True
    cfg_av.lat_BV = BV
    cfg_av.lat_N = 4.0
    o_on = forward_2t(cfg_av, model_M1=M1, model_M2=M2, bjt=bjt,
                       VG1=VG1_t, VG2=VG2_t, Vd_seq=Vd_seq2)
    Id_on = float(o_on["Id"][-1])
    ratio = Id_on / Id_off if Id_off else float("nan")
    print(f"{BV:>8.1f} {Id_off:>22.6e} {Id_on:>22.6e} {ratio:>8.3f}")
