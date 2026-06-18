"""A1l — Vb operating-point trace at three diagnostic biases.

For each bias point: solve via forward_2t_arclength_grad, then re-evaluate
_residuals at the converged (Vsint, Vb) AND at hypothetical Vb=0.7 (same
Vsint, Vd, gates) to test whether a high-Vb basin exists.

Run:
    cd /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
    source venv/bin/activate
    python research_plan/artifacts/A1l_demo.py
"""
from __future__ import annotations
import sys, json, importlib.util
from pathlib import Path

import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from nsram.bsim4_port.bjt import GummelPoonNPN, compute_bjt
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import (
    NSRAMCell2TConfig, _eval_mosfet, _residuals,
)
from nsram.bsim4_port.arclength import forward_2t_arclength_grad
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

spec = importlib.util.spec_from_file_location(
    "z91f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = importlib.util.module_from_spec(spec); spec.loader.exec_module(z91f)

DATA = ROOT / "data/sebas_2026_04_22"

# ---------------------------------------------------------------- biases ---
# (VG1, VG2, sebas-row dict, Id_meas @ Vd=1.5)
BIASES = [
    dict(label="WORST  (VG1=0.6,VG2=0.0)",
         VG1=0.6, VG2=0.0, Id_meas=2.07e-5,
         sebas=dict(ETAB=2.5, K1=0.41825, ALPHA0=7.842e-5, BETA0=20.0,
                    NFACTOR=6.0, mbjt=1.0, IS=5e-9, area=1e-6)),
    dict(label="MOD    (VG1=0.4,VG2=0.0)",
         VG1=0.4, VG2=0.0, Id_meas=1.02e-6,
         sebas=dict(ETAB=1.9, K1=0.53825, ALPHA0=7.842e-5, BETA0=19.0,
                    NFACTOR=6.0, mbjt=1.0, IS=5e-9, area=1e-6)),
    dict(label="BEST   (VG1=0.6,VG2=0.5)",
         VG1=0.6, VG2=0.5, Id_meas=9.64e-7,
         sebas=dict(ETAB=2.1, K1=0.41825, ALPHA0=7.842e-5, BETA0=20.0,
                    NFACTOR=1.25, mbjt=1.0, IS=5e-9, area=1e-6)),
]
VD_PT = 1.5


def trace_one(bias):
    SEBAS = bias["sebas"]
    VG1_PT = bias["VG1"]; VG2_PT = bias["VG2"]

    m1_text = (DATA / "M1_130DNWFB.txt").read_text()
    model_M1 = BSIM4Model.from_spice(m1_text, model_type="nmos")
    z91f.patch_model_values(model_M1, type_n=True)
    m2_text = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model_M2 = BSIM4Model.from_spice(m2_text, model_type="nmos")
    z91f.patch_model_values(model_M2, type_n=True)

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             use_diode=True, use_igb=True,
                             newton_max_iters=80, gmin_step=True)
    sd_M1 = compute_size_dep(model_M1, Geometry(L=cfg.Ln, W=cfg.Wn), T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model_M2,
                              Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn),
                              T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2

    P_M1 = {
        "etab":   torch.tensor(SEBAS["ETAB"]),
        "k1":     torch.tensor(SEBAS["K1"]),
        "alpha0": torch.tensor(SEBAS["ALPHA0"]),
        "beta0":  torch.tensor(SEBAS["BETA0"]),
    }
    P_M2 = {"nfactor": torch.tensor(SEBAS["NFACTOR"])}

    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Is = SEBAS["IS"]
    bjt.area = SEBAS["area"] * SEBAS["mbjt"]

    Vd_seq = torch.tensor([VD_PT], dtype=torch.float64)
    VG1 = torch.tensor(VG1_PT, dtype=torch.float64)
    VG2 = torch.tensor(VG2_PT, dtype=torch.float64)

    with torch.no_grad(), \
         z91f.patch_sd_scaled(sd_M1, P_M1), \
         z91f.patch_sd_scaled(sd_M2, P_M2):
        out = forward_2t_arclength_grad(
            cfg, model_M1=model_M1, model_M2=model_M2, bjt=bjt,
            Vd_seq=Vd_seq, VG1=VG1, VG2=VG2,
            P_M1=None, P_M2=None,
        )

        Vsint = out["Vsint"][:1]
        Vb    = out["Vb"][:1]
        Id    = out["Id"][:1]
        conv  = bool(out["converged"][0])

        Vd_b = torch.tensor([VD_PT])
        VG1_b = torch.tensor([VG1_PT])
        VG2_b = torch.tensor([VG2_PT])

        # 1) residuals/components at converged op-point
        R_S_c, R_B_c, comp_c = _residuals(cfg, model_M1, bjt,
            Vd_b, VG1_b, VG2_b, Vsint, Vb,
            P_M1=None, P_M2=None, model_M2=model_M2)

        # 2) residuals at hypothetical Vb=0.7 with same Vsint
        Vb_hi = torch.tensor([0.7], dtype=torch.float64)
        R_S_hi, R_B_hi, comp_hi = _residuals(cfg, model_M1, bjt,
            Vd_b, VG1_b, VG2_b, Vsint, Vb_hi,
            P_M1=None, P_M2=None, model_M2=model_M2)

        # BJT decomposed for both cases (emitter=GND topology)
        bjt_c  = compute_bjt(bjt, Vbe=Vb,    Vbc=Vb    - Vd_b, T_K=273.15 + cfg.T_C)
        bjt_hi = compute_bjt(bjt, Vbe=Vb_hi, Vbc=Vb_hi - Vd_b, T_K=273.15 + cfg.T_C)

    f = lambda t: float(t.detach().squeeze().item())
    rec = {
        "label": bias["label"], "converged": conv,
        "Vsint": f(Vsint), "Vb": f(Vb),
        "Id_pred": f(Id), "Id_meas": bias["Id_meas"],
        "log_resid_dec": (
            None if not (Id.item() > 0)
            else abs(torch.log10(Id).item() - torch.log10(torch.tensor(bias["Id_meas"])).item())
        ),
        # converged-state KCL residuals
        "R_Sint_at_conv": f(R_S_c), "R_B_at_conv": f(R_B_c),
        # components at converged Vb
        **{f"{k}_conv": f(v) for k, v in comp_c.items()},
        # hypothetical Vb=0.7 evaluation
        "R_Sint_at_Vb0p7": f(R_S_hi), "R_B_at_Vb0p7": f(R_B_hi),
        "Ic_at_Vb0p7": f(bjt_hi["Ic"]), "Ib_at_Vb0p7": f(bjt_hi["Ib"]),
        "Iii_M1_at_Vb0p7": f(comp_hi["Iii_M1"]),
        "Iii_M2_at_Vb0p7": f(comp_hi["Iii_M2"]),
        "Ibs_M2_at_Vb0p7": f(comp_hi["Ibs_M2"]),
        "Ibd_M2_at_Vb0p7": f(comp_hi["Ibd_M2"]),
        "Ibs_M1_at_Vb0p7": f(comp_hi["Ibs_M1"]),
        "Ibd_M1_at_Vb0p7": f(comp_hi["Ibd_M1"]),
    }
    return rec


def main():
    results = [trace_one(b) for b in BIASES]
    out_path = Path(__file__).with_name("A1l_vb_trace.json")
    out_path.write_text(json.dumps(results, indent=2))
    for r in results:
        print("=" * 70)
        print(r["label"])
        for k, v in r.items():
            if k == "label": continue
            if isinstance(v, float):
                print(f"  {k:24s} = {v:+.4e}")
            else:
                print(f"  {k:24s} = {v}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
