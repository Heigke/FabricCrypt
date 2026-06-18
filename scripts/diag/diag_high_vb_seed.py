"""Try forcing high-Vb seed via multi_init + various hot_Vb_init values."""
import sys
sys.path.insert(0, "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/nsram")
import torch, importlib.util
from pathlib import Path

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t, solve_2t_steady_state
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

VG1 = torch.tensor(0.6, dtype=torch.float64)
VG2 = torch.tensor(0.0, dtype=torch.float64)
Vd_seq = torch.tensor([0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0], dtype=torch.float64)

# Try DIRECT call to solve_2t_steady_state with Vb_init = 0.95
print("=== Direct solve_2t_steady_state at Vd=2.0V, various Vb_init ===")
cfg, M1, M2, bjt = build_base()
for vb0 in [0.0, 0.3, 0.7, 0.95, 1.05]:
    Vd_i = torch.tensor([2.0], dtype=torch.float64)
    out = solve_2t_steady_state(
        cfg, M1, bjt, Vd=Vd_i, VG1=VG1, VG2=VG2,
        Vsint_init=torch.tensor([0.5], dtype=torch.float64),
        Vb_init=torch.tensor([vb0], dtype=torch.float64),
        model_M2=M2,
    )
    print(f"  Vb_init={vb0:.2f}: Vb={float(out['Vb']):.4f} Vsint={float(out['Vsint']):.4f} Id={float(out['Id']):.4e} conv={bool(out['converged'].all())}")

# Now multi_init via forward_2t
print("\n=== forward_2t multi_init=True, hot_Vb_init sweep ===")
for hb in [0.5, 0.8, 0.95, 1.05]:
    cfg, M1, M2, bjt = build_base()
    out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_seq,
                     VG1=VG1, VG2=VG2, warm_start=True,
                     multi_init=True, hot_Vb_init=hb)
    print(f"  hot_Vb_init={hb}: Vb@2V={float(out['Vb'][-1]):.4f} Id@2V={float(out['Id'][-1]):.4e}")
