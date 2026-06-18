"""Verify gap at apex (VG1=0.6, Vd=2.0V) - the original target bias."""
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

def build():
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=120)
    cfg.bjt_emitter_to_gnd = True
    cfg.body_pdiode_to = "vnwell"; cfg.use_well_diode = True; cfg.vnwell = 2.0
    cfg.body_pdiode_Js = 5.3675e-7 / 22e-12
    cfg.body_pdiode_n = 1.0535
    cfg.body_pdiode_Rs = 1.0e6
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Bf = 9000.0; bjt.Va = 0.55; bjt.Is = 1e-9
    return cfg, M1, M2, bjt

VG1 = torch.tensor(0.6, dtype=torch.float64)
VG2 = torch.tensor(0.0, dtype=torch.float64)
Vd_seq = torch.linspace(0.05, 2.0, 40, dtype=torch.float64)

cfg, M1, M2, bjt = build()
sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
sd_M1.scaled["alpha0"] = 7.83756e-4
sd_M2.scaled["alpha0"] = 7.83756e-4
sd_M1.scaled["k1"] = 0.53825

print("BEFORE fix (multi_init=False):")
out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_seq, VG1=VG1, VG2=VG2, warm_start=True)
print(f"  Vd=2.0V: Vb={float(out['Vb'][-1]):.4f}  Id={float(out['Id'][-1]):.3e}")

print("\nAFTER fix (multi_init=True hot_Vb=0.8):")
cfg, M1, M2, bjt = build()
sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
sd_M1.scaled["alpha0"] = 7.83756e-4
sd_M2.scaled["alpha0"] = 7.83756e-4
sd_M1.scaled["k1"] = 0.53825
out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_seq, VG1=VG1, VG2=VG2,
                 warm_start=True, multi_init=True, hot_Vsint_init=0.2, hot_Vb_init=0.8)
print(f"  Vd=2.0V: Vb={float(out['Vb'][-1]):.4f}  Id={float(out['Id'][-1]):.3e}")
print(f"  ngspice ref Id=2.20e-6 → gap = {abs(float(torch.log10(torch.abs(out['Id'][-1])/2.20e-6))):.3f} dec")

# ALPHA0 sweep at apex
print("\nALPHA0 sweep (multi_init=True hot_Vb=0.8) at VG1=0.6 Vd=2.0V:")
print("  ALPHA0_mult  ALPHA0       Vb         Id")
base_alpha = 7.842e-5
for mult in [1.0, 10.0, 100.0, 1000.0]:
    cfg, M1, M2, bjt = build()
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
    a = base_alpha * mult
    sd_M1.scaled["alpha0"] = a
    sd_M2.scaled["alpha0"] = a
    out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_seq, VG1=VG1, VG2=VG2,
                     warm_start=True, multi_init=True, hot_Vsint_init=0.2, hot_Vb_init=0.8)
    print(f"  {mult:7.1f}×    {a:.3e}  {float(out['Vb'][-1]):.4f}  {float(out['Id'][-1]):.3e}")
