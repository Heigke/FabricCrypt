"""A1m — alpha0 / beta0 scale brute-force test.

At the worst diagnostic bias (VG1=0.6, VG2=0.0, Vd=1.5), scale alpha0
by 1, 10, 100, 1000, 10000 (applied to BOTH M1 and M2 sd.scaled["alpha0"]),
then sweep beta0 = 20, 10, 5, 2, 1, 0.5 (baseline alpha0). Repeat at
two reference biases.

Run:
    cd /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
    source venv/bin/activate
    python research_plan/artifacts/A1m_demo.py
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
    NSRAMCell2TConfig, _residuals,
)
from nsram.bsim4_port.arclength import forward_2t_arclength_grad
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

spec = importlib.util.spec_from_file_location(
    "z91f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = importlib.util.module_from_spec(spec); spec.loader.exec_module(z91f)

DATA = ROOT / "data/sebas_2026_04_22"

VD_PT = 1.5

BIASES = [
    dict(label="WORST(VG1=0.6,VG2=0.0)",
         VG1=0.6, VG2=0.0, Id_meas=2.07e-5,
         sebas=dict(ETAB=2.5, K1=0.41825, ALPHA0=7.842e-5, BETA0=20.0,
                    NFACTOR=6.0, mbjt=1.0, IS=5e-9, area=1e-6)),
    dict(label="REF1 (VG1=0.4,VG2=0.0)",
         VG1=0.4, VG2=0.0, Id_meas=1.02e-6,
         sebas=dict(ETAB=1.9, K1=0.53825, ALPHA0=7.842e-5, BETA0=19.0,
                    NFACTOR=6.0, mbjt=1.0, IS=5e-9, area=1e-6)),
    dict(label="REF2 (VG1=0.6,VG2=0.5)",
         VG1=0.6, VG2=0.5, Id_meas=9.64e-7,
         sebas=dict(ETAB=2.1, K1=0.41825, ALPHA0=7.842e-5, BETA0=20.0,
                    NFACTOR=1.25, mbjt=1.0, IS=5e-9, area=1e-6)),
]


def build(bias):
    SEBAS = bias["sebas"]
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

    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Is = SEBAS["IS"]
    bjt.area = SEBAS["area"] * SEBAS["mbjt"]

    return cfg, model_M1, model_M2, sd_M1, sd_M2, bjt, SEBAS


def run_one(bias, alpha0_val, beta0_val):
    cfg, model_M1, model_M2, sd_M1, sd_M2, bjt, SEBAS = build(bias)

    P_M1 = {
        "etab":   torch.tensor(SEBAS["ETAB"]),
        "k1":     torch.tensor(SEBAS["K1"]),
        "alpha0": torch.tensor(alpha0_val),
        "beta0":  torch.tensor(beta0_val),
    }
    P_M2 = {
        "nfactor": torch.tensor(SEBAS["NFACTOR"]),
        "alpha0":  torch.tensor(alpha0_val),
        "beta0":   torch.tensor(beta0_val),
    }

    Vd_seq = torch.tensor([VD_PT], dtype=torch.float64)
    VG1 = torch.tensor(bias["VG1"], dtype=torch.float64)
    VG2 = torch.tensor(bias["VG2"], dtype=torch.float64)

    rec = {"alpha0": alpha0_val, "beta0": beta0_val}
    try:
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
            VG1_b = torch.tensor([bias["VG1"]])
            VG2_b = torch.tensor([bias["VG2"]])
            R_S, R_B, comp = _residuals(cfg, model_M1, bjt,
                Vd_b, VG1_b, VG2_b, Vsint, Vb,
                P_M1=None, P_M2=None, model_M2=model_M2)
            bjt_c = compute_bjt(bjt, Vbe=Vb, Vbc=Vb - Vd_b,
                                T_K=273.15 + cfg.T_C)

        f = lambda t: float(t.detach().squeeze().item())
        rec.update({
            "converged": conv,
            "Vsint": f(Vsint), "Vb": f(Vb),
            "Id_pred": f(Id),
            "Iii_M1": f(comp["Iii_M1"]),
            "Iii_M2": f(comp["Iii_M2"]),
            "Ic_Q1":  f(bjt_c["Ic"]),
            "R_B":    f(R_B),
            "status": "ok",
        })
    except Exception as e:
        rec.update({"status": f"FAIL:{type(e).__name__}:{e}"})
    return rec


def main():
    BASE_ALPHA0 = 7.842e-5
    BASE_BETA0  = 20.0
    ALPHA_SCALES = [1, 10, 100, 1000, 10000]
    BETA_VALUES  = [20.0, 10.0, 5.0, 2.0, 1.0, 0.5]

    results = {"alpha0_sweep": {}, "beta0_sweep": {}}

    for bias in BIASES:
        lab = bias["label"]
        print("=" * 78); print(lab); print("=" * 78)
        # alpha0 sweep at beta0 = bias's baseline
        b0_base = bias["sebas"]["BETA0"]
        results["alpha0_sweep"][lab] = []
        print("\nALPHA0 SWEEP (beta0=%.1f baseline)" % b0_base)
        print(f"{'scale':>8} {'alpha0':>12} {'Vb':>10} {'Iii_M1':>12} "
              f"{'Iii_M2':>12} {'Ic':>12} {'Id':>12} {'conv':>5}")
        for sc in ALPHA_SCALES:
            r = run_one(bias, BASE_ALPHA0 * sc, b0_base)
            r["scale"] = sc
            results["alpha0_sweep"][lab].append(r)
            if r["status"] == "ok":
                print(f"{sc:>8d} {r['alpha0']:>12.3e} {r['Vb']:>+10.4f} "
                      f"{r['Iii_M1']:>12.3e} {r['Iii_M2']:>12.3e} "
                      f"{r['Ic_Q1']:>12.3e} {r['Id_pred']:>12.3e} "
                      f"{str(r['converged']):>5}")
            else:
                print(f"{sc:>8d} {r['alpha0']:>12.3e} {r['status']}")

        # beta0 sweep at baseline alpha0
        a0_base = bias["sebas"]["ALPHA0"]
        results["beta0_sweep"][lab] = []
        print("\nBETA0 SWEEP (alpha0=%.3e baseline)" % a0_base)
        print(f"{'beta0':>8} {'Vb':>10} {'Iii_M1':>12} "
              f"{'Iii_M2':>12} {'Ic':>12} {'Id':>12} {'conv':>5}")
        for b0 in BETA_VALUES:
            r = run_one(bias, a0_base, b0)
            results["beta0_sweep"][lab].append(r)
            if r["status"] == "ok":
                print(f"{b0:>8.2f} {r['Vb']:>+10.4f} "
                      f"{r['Iii_M1']:>12.3e} {r['Iii_M2']:>12.3e} "
                      f"{r['Ic_Q1']:>12.3e} {r['Id_pred']:>12.3e} "
                      f"{str(r['converged']):>5}")
            else:
                print(f"{b0:>8.2f} {r['status']}")

    out_path = Path(__file__).with_name("A1m_alpha0_scale_test.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
