"""Validate fix: forward_2t multi_init should now find high-Vb root."""
import sys
sys.path.insert(0, "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/nsram")
import torch, importlib.util
from pathlib import Path

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t
from nsram.bsim4_port.bjt import GummelPoonNPN

def build_base():
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=120)
    cfg.bjt_emitter_to_gnd = True
    cfg.body_pdiode_to = "vnwell"; cfg.use_well_diode = True; cfg.vnwell = 2.0
    cfg.body_pdiode_Js = 5.3675e-7 / 22e-12
    cfg.body_pdiode_n = 1.0535
    cfg.body_pdiode_Rs = 1.0e6
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    return cfg, M1, M2, bjt

VG1 = torch.tensor(0.6, dtype=torch.float64)
VG2 = torch.tensor(0.0, dtype=torch.float64)
Vd_seq = torch.tensor([0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0], dtype=torch.float64)

# BEFORE-EQUIV: multi_init=False (default forward_2t)
cfg, M1, M2, bjt = build_base()
out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_seq, VG1=VG1, VG2=VG2, warm_start=True, multi_init=False)
print("BEFORE (multi_init=False):")
for vd, vb, vs, idd in zip(Vd_seq, out["Vb"], out["Vsint"], out["Id"]):
    print(f"  Vd={float(vd):.2f}: Vb={float(vb):.4f} Vsint={float(vs):.4f} Id={float(idd):.3e}")

# AFTER: multi_init=True with hot_Vb_init=0.8
cfg, M1, M2, bjt = build_base()
out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_seq, VG1=VG1, VG2=VG2, warm_start=True, multi_init=True, hot_Vsint_init=0.2, hot_Vb_init=0.8)
print("\nAFTER (multi_init=True hot_Vb=0.8 hot_Vs=0.2 + IIIROUTE-FIX):")
for vd, vb, vs, idd in zip(Vd_seq, out["Vb"], out["Vsint"], out["Id"]):
    print(f"  Vd={float(vd):.2f}: Vb={float(vb):.4f} Vsint={float(vs):.4f} Id={float(idd):.3e}")
print("\nngspice ref at Vd=2.0: Vb=1.010 Vsint=0.476 Id=2.20e-6")
print(f"PASS gate: |log10(Id_pyport/Id_ngspice)| <= 0.5")
g = abs(float(torch.log10(torch.abs(out['Id'][-1])/2.20e-6)))
print(f"Final dec gap = {g:.3f}")
