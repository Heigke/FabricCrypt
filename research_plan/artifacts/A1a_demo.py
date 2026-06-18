"""A1a — Diagnose whether sd.scaled["nfactor"] override reaches the
subthreshold-slope formula in compute_dc.

Calls compute_dc on M2 directly at (Vgs=-0.10, Vds=2.0, Vbs=0) with
nfactor in {1.58, 12.15} and prints Id, Vgsteff, Vth.
"""
from __future__ import annotations
from pathlib import Path
import sys
import torch

torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.dc import compute_dc

# Mirror z91f's loader/patch ----------------------------------------------- #
DATA = ROOT / "data/sebas_2026_04_22"

# SHARED_PARAM values copied from z91f patch_model_values (only the M2 ones
# we need; we also mirror the static M2 overrides applied to sd.scaled).
SHARED_PARAM_n = dict(vth0=0.40, vsat=9.0e4, lpe0=1.74e-7, lint=0.0,
                      wint=0.0, k3=80.0, pvth0=0.0, tox=2.7e-9)


def patch_model_values(model):
    pmap = {"vth0": SHARED_PARAM_n["vth0"], "vsat": SHARED_PARAM_n["vsat"],
            "lpe0": SHARED_PARAM_n["lpe0"], "lint": SHARED_PARAM_n["lint"],
            "wint": SHARED_PARAM_n["wint"], "k3": SHARED_PARAM_n["k3"],
            "pvth0": SHARED_PARAM_n["pvth0"], "toxe": SHARED_PARAM_n["tox"],
            "toxp": SHARED_PARAM_n["tox"], "toxm": SHARED_PARAM_n["tox"]}
    for k, v in pmap.items():
        model._values[k] = float(v)


M2_STATIC_OVERRIDES = {"k1": 0.63825, "k2": -0.070435,
                       "etab": -0.086777, "beta0": 18.0}


def main():
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model_M2 = BSIM4Model.from_spice(text_M2, model_type="nmos")
    patch_model_values(model_M2)

    # M2 geom: Ln*10x = 1.8 µm, W = 360 nm
    geom = Geometry(L=180e-9 * 10.0, W=360e-9)
    sd_M2 = compute_size_dep(model_M2, geom, T_C=27.0)

    # Apply static M2 overrides (k1/k2/etab/beta0) — same as z91f
    for k, v in M2_STATIC_OVERRIDES.items():
        sd_M2.scaled[k] = torch.tensor(float(v), dtype=torch.float64)

    Vgs = torch.tensor(-0.10, dtype=torch.float64)
    Vds = torch.tensor(2.0, dtype=torch.float64)
    Vbs = torch.tensor(0.0, dtype=torch.float64)

    default_nf = float(sd_M2.scaled.get("nfactor", float("nan")))
    print(f"[A1a] M2 default scaled['nfactor'] = {default_nf:.4f}")
    print(f"[A1a] bias: Vgs={float(Vgs):+.3f} Vds={float(Vds):.3f} "
          f"Vbs={float(Vbs):.3f}")

    results = {}
    for label, nf in [("low_1.58", 1.58), ("high_12.15", 12.15)]:
        saved = sd_M2.scaled.get("nfactor", None)
        sd_M2.scaled["nfactor"] = torch.tensor(nf, dtype=torch.float64)
        try:
            r = compute_dc(model_M2, sd_M2, Vgs, Vds, Vbs)
        finally:
            if saved is None:
                sd_M2.scaled.pop("nfactor", None)
            else:
                sd_M2.scaled["nfactor"] = saved
        Id = float(r.Ids)
        Vgsteff = float(r.Vgsteff)
        Vth = float(r.Vth)  # final assembled Vth
        results[label] = (nf, Id, Vgsteff, Vth)
        print(f"[A1a] nfactor={nf:6.2f}  Id={Id:.6e}  "
              f"Vgsteff={Vgsteff:.6e}  Vth={Vth:.6e}")

    nf_lo, Id_lo, Vg_lo, Vt_lo = results["low_1.58"]
    nf_hi, Id_hi, Vg_hi, Vt_hi = results["high_12.15"]
    if Id_lo > 0:
        ratio = Id_hi / Id_lo
        import math as _m
        decades = _m.log10(max(ratio, 1e-300))
    else:
        ratio = float("inf"); decades = float("inf")
    print(f"[A1a] Id ratio (high/low)   = {ratio:.4e}  ({decades:+.3f} dec)")
    print(f"[A1a] dVgsteff             = {Vg_hi - Vg_lo:+.4e}")
    print(f"[A1a] dVth                 = {Vt_hi - Vt_lo:+.4e}")


if __name__ == "__main__":
    main()
