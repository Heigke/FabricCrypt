"""Test: at each Vd, try BOTH the cascaded warm-start AND a fresh high-Vb init.
Pick whichever converges with higher |Id|."""
import sys
sys.path.insert(0, "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/nsram")
import torch, importlib.util
from pathlib import Path

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, solve_2t_steady_state
from nsram.bsim4_port.bjt import GummelPoonNPN

def build_base():
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=80)
    cfg.bjt_emitter_to_gnd = True
    cfg.body_pdiode_to = "vnwell"; cfg.use_well_diode = True; cfg.vnwell = 2.0
    cfg.body_pdiode_Js = 5.3675e-7 / 22e-12
    cfg.body_pdiode_n = 1.0535
    cfg.body_pdiode_Rs = 1.0e6
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    return cfg, M1, M2, bjt

cfg, M1, M2, bjt = build_base()
VG1 = torch.tensor(0.6, dtype=torch.float64)
VG2 = torch.tensor(0.0, dtype=torch.float64)
Vd_seq = [0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
HOT_VB = 1.05
HOT_VS = 0.05

print(f"Sweep with FRESH hot_Vb_init={HOT_VB} at EVERY Vd point (no cascade):")
Vs_cold = torch.tensor([0.025], dtype=torch.float64)
Vb_cold = torch.tensor([0.5], dtype=torch.float64)
for vd in Vd_seq:
    Vd_i = torch.tensor([vd], dtype=torch.float64)
    out_cold = solve_2t_steady_state(cfg, M1, bjt, Vd=Vd_i, VG1=VG1, VG2=VG2,
                                     Vsint_init=Vs_cold, Vb_init=Vb_cold, model_M2=M2)
    out_hot = solve_2t_steady_state(cfg, M1, bjt, Vd=Vd_i, VG1=VG1, VG2=VG2,
                                    Vsint_init=torch.tensor([HOT_VS], dtype=torch.float64),
                                    Vb_init=torch.tensor([HOT_VB], dtype=torch.float64),
                                    model_M2=M2)
    Id_c = float(torch.abs(out_cold["Id"])); Id_h = float(torch.abs(out_hot["Id"]))
    pick = "HOT" if Id_h > Id_c else "cold"
    out = out_hot if Id_h > Id_c else out_cold
    print(f"  Vd={vd:.2f}: cold[Vb={float(out_cold['Vb']):.3f} Id={Id_c:.3e}] "
          f"hot[Vb={float(out_hot['Vb']):.3f} Id={Id_h:.3e}] pick={pick} "
          f"conv={bool(out['converged'].all())}")
    # cold cascade only
    Vs_cold = out_cold["Vsint"].detach()
    Vb_cold = out_cold["Vb"].detach()
