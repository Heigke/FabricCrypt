"""A1c — Component-current trace at low-VG2 bias point.

Bias: VG1=0.6, VG2=0.0, Vd=1.5 V. Per-bias Sebas params from CSV row.

Calls solve_2t_steady_state directly (does NOT edit forward_2t), then
re-evaluates _eval_mosfet for M1/M2 and compute_bjt at the converged
(Vsint, Vb) so every component current is printable with sign.

Run from project root with the project venv:
    cd /home/ikaros/Documents/claude_hive/AMD_gfx1151_energy
    source venv/bin/activate
    python research_plan/artifacts/A1c_demo.py
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
    NSRAMCell2TConfig, solve_2t_steady_state, solve_2t_with_homotopy,
    _eval_mosfet,
)
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

# Reuse z91f helpers (patch_model_values, patch_sd_scaled, M2_STATIC_OVERRIDES)
spec = importlib.util.spec_from_file_location(
    "z91f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = importlib.util.module_from_spec(spec); spec.loader.exec_module(z91f)

DATA = ROOT / "data/sebas_2026_04_22"

# ---------------------------------------------------------------- Sebas row ---
# CSV row VG1=0.6, VG2=0.0
SEBAS = dict(ETAB=2.5, K1=0.41825, ALPHA0=7.842e-5, BETA0=20.0,
             NFACTOR=6.0, mbjt=1.0, IS=5e-9, area=1e-6)

# Bias point
VD_PT = 1.5
VG1_PT = 0.6
VG2_PT = 0.0


def main():
    # -- Load M1 and M2 cards ------------------------------------------------
    m1_text = (DATA / "M1_130DNWFB.txt").read_text()
    model_M1 = BSIM4Model.from_spice(m1_text, model_type="nmos")
    z91f.patch_model_values(model_M1, type_n=True)

    m2_text = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model_M2 = BSIM4Model.from_spice(m2_text, model_type="nmos")
    z91f.patch_model_values(model_M2, type_n=True)

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True,
                             use_diode=True, use_igb=True,
                             newton_max_iters=80, gmin_step=True)
    sd_M1 = compute_size_dep(model_M1, Geometry(L=cfg.Ln, W=cfg.Wn),
                              T_C=cfg.T_C)
    sd_M2 = compute_size_dep(model_M2,
                              Geometry(L=cfg.Ln * cfg.M2_length_factor,
                                       W=cfg.Wn), T_C=cfg.T_C)
    cfg._sd_M1 = sd_M1
    cfg._sd_M2 = sd_M2

    # M1 overrides from Sebas row
    P_M1 = {
        "etab":   torch.tensor(SEBAS["ETAB"]),
        "k1":     torch.tensor(SEBAS["K1"]),
        "alpha0": torch.tensor(SEBAS["ALPHA0"]),
        "beta0":  torch.tensor(SEBAS["BETA0"]),
    }
    # M2 override = NFACTOR only (the static k1/etab/beta0 baselines now live
    # in sd_M2 — z91g logic). DON'T re-apply M2_STATIC_OVERRIDES here since
    # we loaded the proper M2 card.
    P_M2 = {"nfactor": torch.tensor(SEBAS["NFACTOR"])}

    # BJT (Sebas card · area = area*mbjt = 1e-6)
    bjt = GummelPoonNPN.from_sebas_card()
    bjt.Is = SEBAS["IS"]
    bjt.area = SEBAS["area"] * SEBAS["mbjt"]

    # -- Solve at single bias point -----------------------------------------
    Vd = torch.tensor([VD_PT])
    VG1 = torch.tensor([VG1_PT])
    VG2 = torch.tensor([VG2_PT])

    with torch.no_grad(), \
         z91f.patch_sd_scaled(sd_M1, P_M1), \
         z91f.patch_sd_scaled(sd_M2, P_M2):
        # P_M1/P_M2 already applied via patch_sd_scaled context; pass None
        # so _eval_mosfet doesn't try the attribute-style _override_sd path.
        # Homotopy walk in gmin avoids the spurious flat root that the
        # plain Newton solver hits at this bias (Vb=0, Vsint=Vd/2 cold start
        # is near a saddle; residuals are <1e-12 there because all body
        # currents are sub-femtoamp). Source comments call out exactly this
        # pathology — homotopy is the recommended remedy.
        out = solve_2t_with_homotopy(
            cfg, model_M1, bjt, Vd, VG1, VG2,
            P_M1=None, P_M2=None,
            Vsint_init=torch.tensor(0.5 * VD_PT),
            Vb_init=torch.tensor(0.7),  # NPN turn-on guess
            model_M2=model_M2, verbose=True,
        )

        Vsint = out["Vsint"]
        Vb    = out["Vb"]
        conv  = bool(out["converged"].all())
        niter = out["niter"]
        Id    = out["Id"]

        # Re-evaluate components at converged voltages (overrides=None; sd
        # already patched).
        zero = torch.zeros_like(Vd)
        m1 = _eval_mosfet(model_M1, sd_M1, cfg, Vg=VG1, Vd=Vd, Vs=Vsint,
                          Vb=Vb, junctions=cfg._junctions_M1(), overrides=None)
        m2 = _eval_mosfet(model_M2, sd_M2, cfg, Vg=VG2, Vd=Vsint, Vs=zero,
                          Vb=Vb, junctions=cfg._junctions_M2(), overrides=None)
        Vbe = Vb - Vsint
        Vbc = Vb - Vd
        bjt_out = compute_bjt(bjt, Vbe=Vbe, Vbc=Vbc, T_K=273.15 + cfg.T_C)

    f = lambda t: float(t.item())
    rec = {
        "converged": conv, "niter": niter,
        "Vsint": f(Vsint), "Vb": f(Vb), "Vbe": f(Vbe), "Vbc": f(Vbc),
        "Id_node": f(Id),
        "Ids_M1": f(m1["Ids"]),
        "Ids_M2": f(m2["Ids"]),
        "Ic_Q1":  f(bjt_out["Ic"]),
        "Ib_Q1":  f(bjt_out["Ib"]),
        "Ibs_M2": f(m2["Ibs"]),  # body-source diode of M2 (the floating body)
        "Ibd_M2": f(m2["Ibd"]),
        "Ibs_M1": f(m1["Ibs"]),
        "Ibd_M1": f(m1["Ibd"]),
        "Igidl_M1": f(m1["Igidl"]), "Igisl_M1": f(m1["Igisl"]),
        "Igidl_M2": f(m2["Igidl"]), "Igisl_M2": f(m2["Igisl"]),
        "Iii_M1": f(m1["Iii"]), "Iii_M2": f(m2["Iii"]),
        "Igb_M1": f(m1["Igb"]), "Igb_M2": f(m2["Igb"]),
        "Id_meas_at_Vd1.5": 2.07e-5,   # from sebas CSV (interpolated row)
    }

    out_path = Path(__file__).with_name("A1c_trace.json")
    out_path.write_text(json.dumps(rec, indent=2))
    print("\n=== A1c component trace ===")
    for k, v in rec.items():
        if isinstance(v, float):
            print(f"  {k:14s} = {v:+.4e}")
        else:
            print(f"  {k:14s} = {v}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
