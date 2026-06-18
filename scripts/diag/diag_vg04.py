"""Check VG1=0.4 at low Vd: should be low Id (~1e-13)."""
import sys
sys.path.insert(0, "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/nsram")
import torch, importlib.util
from pathlib import Path
ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, solve_2t_steady_state, forward_2t
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

VG1 = torch.tensor(0.4, dtype=torch.float64)
VG2 = torch.tensor(0.0, dtype=torch.float64)

cfg, M1, M2, bjt = build()
sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)
sd_M1.scaled["alpha0"] = 7.83756e-4
sd_M2.scaled["alpha0"] = 7.83756e-4

# At Vd=0.05 (subthreshold): ngspice 9.81e-13
for vd_t in [0.05, 0.1, 0.3, 1.0, 2.0]:
    Vd_i = torch.tensor([vd_t], dtype=torch.float64)
    print(f"\nVd={vd_t}:")
    for label, vs0, vb0 in [("COLD", 0.025, 0.5), ("HOT (Vb=0.8)", 0.2, 0.8)]:
        out = solve_2t_steady_state(cfg, M1, bjt, Vd=Vd_i, VG1=VG1, VG2=VG2,
                                    Vsint_init=torch.tensor([vs0], dtype=torch.float64),
                                    Vb_init=torch.tensor([vb0], dtype=torch.float64),
                                    model_M2=M2)
        print(f"  {label:14s}: Vb={float(out['Vb']):.4f} Vsint={float(out['Vsint']):.4f} Id={float(out['Id']):.3e} conv={bool(out['converged'].all())}")
