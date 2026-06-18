"""A1d - trace M2 impact-ionization (Iii) at low-VG2 vs strong-VG2.

Goal: explain why Iii_M2 ~= 1e-16 at (VG1=0.6, VG2=0.0, Vd=1.5) while ALPHA0
is meaningful (7.842e-5). Print every factor in the BSIM4 §6.1 formula:

    Iii = (alpha0 + alpha1*Leff)/Leff * (Vds - Vdseff) *
          exp(-beta0/(Vds - Vdseff)) * Idsa

Run from project root with venv:
    cd /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
    source venv/bin/activate
    python research_plan/artifacts/A1d_demo.py
"""
from __future__ import annotations
import sys, json, math, importlib.util
from pathlib import Path

import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from nsram.bsim4_port.bjt import GummelPoonNPN
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.dc import compute_dc
from nsram.bsim4_port.leak import compute_iimpact
from nsram.bsim4_port.nsram_cell_2T import (
    NSRAMCell2TConfig, solve_2t_with_homotopy,
)
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

spec = importlib.util.spec_from_file_location(
    "z91f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = importlib.util.module_from_spec(spec); spec.loader.exec_module(z91f)

DATA = ROOT / "data/sebas_2026_04_22"

# Per-bias Sebas rows (from CSV; ALPHA0/BETA0 from CSV row).
BIAS_LOW = dict(
    name="LOW_VG2", VG1=0.6, VG2=0.0, Vd=1.5,
    sebas=dict(ETAB=2.5, K1=0.41825, ALPHA0=7.842e-5, BETA0=20.0,
               NFACTOR=6.0, mbjt=1.0, IS=5e-9, area=1e-6),
)
BIAS_HI = dict(
    name="HIGH_VG2", VG1=0.6, VG2=0.5, Vd=1.5,
    # placeholder-meaningful row; we just need a strong-on M2 regime, the
    # per-bias param values are still LDE-region-typical.
    sebas=dict(ETAB=2.5, K1=0.41825, ALPHA0=7.842e-5, BETA0=20.0,
               NFACTOR=6.0, mbjt=1.0, IS=5e-9, area=1e-6),
)


def run_one(bias):
    SEBAS = bias["sebas"]
    VD_PT, VG1_PT, VG2_PT = bias["Vd"], bias["VG1"], bias["VG2"]

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
                             Geometry(L=cfg.Ln * cfg.M2_length_factor,
                                      W=cfg.Wn), T_C=cfg.T_C)
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

    Vd = torch.tensor([VD_PT])
    VG1 = torch.tensor([VG1_PT])
    VG2 = torch.tensor([VG2_PT])

    with torch.no_grad(), \
         z91f.patch_sd_scaled(sd_M1, P_M1), \
         z91f.patch_sd_scaled(sd_M2, P_M2):
        out = solve_2t_with_homotopy(
            cfg, model_M1, bjt, Vd, VG1, VG2,
            P_M1=None, P_M2=None,
            Vsint_init=torch.tensor(0.5 * VD_PT),
            Vb_init=torch.tensor(0.7),
            model_M2=model_M2, verbose=False,
        )
        Vsint = out["Vsint"]
        Vb = out["Vb"]

        # --- M2 sub-block -------------------------------------------------
        # M2: Vg=VG2, Vd=Vsint, Vs=0, Vb=Vb
        Vgs2 = VG2 - 0.0
        Vds2 = Vsint - 0.0
        Vbs2 = Vb - 0.0

        dc_M2 = compute_dc(model_M2, sd_M2, Vgs=Vgs2, Vds=Vds2, Vbs=Vbs2)
        Iii_M2 = compute_iimpact(model_M2, sd_M2, dc_M2, Vds=Vds2)

        # Pull factors
        leff = float(sd_M2.geom.leff)
        a0 = float(sd_M2.scaled.get("alpha0", 0.0))
        a1 = float(sd_M2.scaled.get("alpha1", 0.0))
        b0 = float(sd_M2.scaled.get("beta0", 0.0))
        Vdsat = float(dc_M2.Vdsat.item())
        Vdseff = float(dc_M2.Vdseff.item())
        Vds_v = float(Vds2.item())
        diff = Vds_v - Vdseff
        T2 = (a0 + a1 * leff) / leff
        # exp factor (clamped diff)
        diff_safe = max(diff, 1e-30)
        T0 = -b0 / diff_safe
        exp_factor = math.exp(T0) if T0 > -700 else 0.0
        Idsa = float(getattr(dc_M2, "Idsa", dc_M2.Ids).item())
        Ids = float(dc_M2.Ids.item())

    return {
        "name": bias["name"],
        "VG1": VG1_PT, "VG2": VG2_PT, "Vd": VD_PT,
        "Vsint": float(Vsint.item()), "Vb": float(Vb.item()),
        # M2 sub-block bias
        "Vgs_M2": float(Vgs2.item()),
        "Vds_M2": Vds_v,
        "Vbs_M2": float(Vbs2.item()),
        # BSIM4 §6.1 factors
        "Vdsat_M2": Vdsat,
        "Vdseff_M2": Vdseff,
        "Vds_minus_Vdseff": diff,
        "alpha0_eff": a0,
        "alpha1_eff": a1,
        "beta0_eff": b0,
        "Leff_M2": leff,
        "T2 = (a0+a1*L)/L": T2,
        "neg_beta0_over_diff": T0,
        "exp(-beta0/diff)": exp_factor,
        "Idsa_M2": Idsa,
        "Ids_M2": Ids,
        "Iii_M2": float(Iii_M2.item()),
    }


def main():
    results = [run_one(BIAS_LOW), run_one(BIAS_HI)]
    out_path = Path(__file__).with_name("A1d_iimpact_trace.json")
    out_path.write_text(json.dumps(results, indent=2))
    for r in results:
        print(f"\n=== {r['name']}  VG1={r['VG1']} VG2={r['VG2']} Vd={r['Vd']} ===")
        for k, v in r.items():
            if k == "name": continue
            if isinstance(v, float):
                print(f"  {k:28s} = {v:+.4e}")
            else:
                print(f"  {k:28s} = {v}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
