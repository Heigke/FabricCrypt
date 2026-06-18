"""Diagnose Iii routing into body KCL at VG1=0.6, VG2=0, Vd=2.0.

Prints:
- m1["Iii"], iii_gain, eta_lat, iii_to_body_factor
- Body residual contributions
- Final converged Vb, Id
"""
import os, sys, json
sys.path.insert(0, "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")
sys.path.insert(0, "/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/nsram")

import torch
import importlib.util
from pathlib import Path

ROOT = Path("/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy")

sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig, forward_2t, _residuals
from nsram.bsim4_port.bjt import GummelPoonNPN


def build_base():
    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True, newton_max_iters=40)
    cfg.bjt_emitter_to_gnd = True
    cfg.body_pdiode_to = "vnwell"
    cfg.use_well_diode = True
    cfg.vnwell = 2.0
    cfg.body_pdiode_Js = 5.3675e-7 / 22e-12
    cfg.body_pdiode_n = 1.0535
    cfg.body_pdiode_Rs = 1.0e6
    M1, M2 = v1.build_calibrated_models()
    bjt = GummelPoonNPN.from_sebas_card()
    return cfg, M1, M2, bjt


def run(alpha0_mult=1.0, label=""):
    cfg, M1, M2, bjt = build_base()
    sd_M1 = cfg.size_dep_M1(M1); sd_M2 = cfg.size_dep_M2(M2)

    base_alpha0 = sd_M1.scaled.get("alpha0", 7.84e-5)
    new_alpha0 = float(base_alpha0) * alpha0_mult
    sd_M1.scaled["alpha0"] = new_alpha0
    sd_M2.scaled["alpha0"] = new_alpha0

    Vd_seq = torch.tensor([0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
    VG1 = torch.tensor(0.6, dtype=torch.float64)
    VG2 = torch.tensor(0.0, dtype=torch.float64)
    out = forward_2t(cfg, model_M1=M1, model_M2=M2, bjt=bjt, Vd_seq=Vd_seq,
                     VG1=VG1, VG2=VG2, warm_start=True)
    Vd_final = Vd_seq[-1:]
    Vb_final = out["Vb"][-1:] if "Vb" in out else None
    Vsint_final = out["Vsint"][-1:] if "Vsint" in out else None

    print(f"\n--- {label}  ALPHA0={new_alpha0:.4e}  (mult={alpha0_mult}) ---")
    print(f"  Vd=2.0  Vb={float(Vb_final):.4f} V  Vsint={float(Vsint_final):.4f} V  Id={float(out['Id'][-1]):.4e} A")

    # Now decompose at converged Vsint, Vb
    R_Sint, R_B, comp = _residuals(
        cfg, M1, bjt,
        Vd=Vd_final, VG1=VG1, VG2=VG2,
        Vsint=Vsint_final, Vb=Vb_final,
        model_M2=M2,
    )
    print(f"  R_Sint={float(R_Sint):.4e}  R_B={float(R_B):.4e}  (should be ~0)")
    keys = ["Iii_M1", "Iii_M2", "Ids_M1", "Ids_M2",
            "Igidl_M1", "Igidl_M2", "Igb_M1", "Igb_M2",
            "Ibs_M1", "Ibd_M1", "Ibs_M2", "Ibd_M2", "I_well_body", "I_body_pdiode",
            "Ib_Q1", "Ib_Q2", "Ic_Q1", "Ib_lat_pair", "Ic_lat"]
    print(f"  expected ngspice Iii ~ 50e-6 A; Vb_target ~ 1.01 V")
    for k in keys:
        if k in comp:
            v = comp[k]
            try:
                vv = float(v[-1]) if hasattr(v, '__len__') else float(v)
                print(f"    {k:24s} = {vv:.4e}")
            except Exception:
                print(f"    {k:24s} = {v}")
    return float(Vb_final), float(out['Id'][-1]), comp


if __name__ == "__main__":
    for mult, lab in [(1.0, "baseline"), (10.0, "10x"), (100.0, "100x"), (1000.0, "1000x")]:
        run(alpha0_mult=mult, label=lab)
