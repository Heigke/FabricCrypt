"""z344 — Instrumentation: verify whether M1 card patch (lalpha0=0, alpha0=7.84e-4)
flows into pyport's IIMOD/Iii computation at flagship bias VG1=0.6, VG2=0.20.

Bias: VG1=0.6, VG2=0.20, Vd=2.0, Vsint=0.382, Vb=0.267 (ngspice OP values).

Outputs:
  - sd_M1.scaled["alpha0"] post-geometry scaling
  - sd_M1.scaled["lalpha0"] (should be 0 if patch applied — but lalpha0 NOT in
    SCALED_PARAMS, so it stays in model._values["lalpha0"])
  - sd_M1.geom.leff and Inv_L
  - binunit (override sets it to 1!)
  - compute_iimpact() at the OP bias
  - Compare to ngspice Iii ≈ 4e-7 A
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys, json, importlib.util
from pathlib import Path
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/z344_iimod_verify"
OUT.mkdir(parents=True, exist_ok=True)

DATA = ROOT / "data/sebas_2026_04_22"

from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.leak import compute_iimpact
from nsram.bsim4_port.dc import compute_dc
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

# Load helpers
sp = importlib.util.spec_from_file_location("v1", ROOT / "scripts/z96_narma10_pilot.py")
v1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(v1)
f = v1.f


def probe(card_filename: str, label: str) -> dict:
    text = (DATA / card_filename).read_text()
    M1 = BSIM4Model.from_spice(text, model_type="nmos")
    # raw values BEFORE patch
    raw_alpha0 = M1._values.get("alpha0")
    raw_lalpha0 = M1._values.get("lalpha0")
    raw_binunit = M1._values.get("binunit")

    f.patch_model_values(M1, type_n=True)
    # after patch
    post_alpha0 = M1._values.get("alpha0")
    post_lalpha0 = M1._values.get("lalpha0")
    post_binunit = M1._values.get("binunit")

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True)
    geom = Geometry(L=cfg.Ln, W=cfg.Wn, NF=1)
    sd = compute_size_dep(M1, geom, T_C=cfg.T_C)
    sc_alpha0 = sd.scaled.get("alpha0")
    sc_alpha1 = sd.scaled.get("alpha1")
    sc_beta0 = sd.scaled.get("beta0")
    leff = float(sd.geom.leff)
    Inv_L = float(sd.geom.Inv_L)

    # OP bias from ngspice (VG1=0.6 mode, VG2=0.20)
    # M1: Vg=VG1=0.6, Vd=2.0, Vs=0 (M1 drain to Vdd-ish but per Sebas topology,
    # M1's source is at GND, drain at Vsint=0.382, body Vb=0.267).
    # Actually M1 is the high-side switch — let me match to forward_2t topology:
    # In Sebas's NSRAM: M1 = read transistor, M2 = retention.
    # From R-26 ngspice: Vd=2.0 is the DRAIN of M1. Vsint=0.382 is the M1 SOURCE.
    Vg = torch.tensor(0.6)
    Vd_node = torch.tensor(2.0)
    Vs_node = torch.tensor(0.382)
    Vb_node = torch.tensor(0.267)
    Vgs = Vg - Vs_node
    Vds = Vd_node - Vs_node
    Vbs = Vb_node - Vs_node

    dc_result = compute_dc(M1, sd, Vgs=Vgs, Vds=Vds, Vbs=Vbs)
    Ids = float(dc_result.Ids)
    Vdseff = float(dc_result.Vdseff)
    Idsa = float(getattr(dc_result, "Idsa", torch.tensor(float("nan"))))

    Iii = compute_iimpact(M1, sd, dc_result, Vds=Vds)
    Iii_val = float(Iii)

    return {
        "label": label,
        "card_file": card_filename,
        "raw": {"alpha0": raw_alpha0, "lalpha0": raw_lalpha0, "binunit": raw_binunit},
        "after_patch": {"alpha0": post_alpha0, "lalpha0": post_lalpha0, "binunit": post_binunit},
        "geom": {"Ln_cfg": cfg.Ln, "leff": leff, "Inv_L": Inv_L},
        "scaled": {"alpha0": sc_alpha0, "alpha1": sc_alpha1, "beta0": sc_beta0},
        "op": {"Vgs": float(Vgs), "Vds": float(Vds), "Vbs": float(Vbs),
               "Ids": Ids, "Vdseff": Vdseff, "Idsa": Idsa},
        "Iii_A": Iii_val,
        "alpha0_eff_pyport": sc_alpha0,  # at binunit=1 lalpha0*Inv_L is tiny
    }


def main():
    results = {}
    results["original"] = probe("M1_130DNWFB.txt", "original M1 (lalpha0=-9.84e-12, alpha0=7.84e-5)")
    results["patched"] = probe("M1_130DNWFB_LALPHA0_FIX.txt", "patched M1 (lalpha0=0, alpha0=7.84e-4)")

    print(json.dumps(results, indent=2, default=str), flush=True)

    # Summary comparison
    o = results["original"]; p = results["patched"]
    print("\n=== SUMMARY ===", flush=True)
    print(f"binunit (post-patch): orig={o['after_patch']['binunit']}  patch={p['after_patch']['binunit']}", flush=True)
    print(f"leff: orig={o['geom']['leff']:.3e}  patch={p['geom']['leff']:.3e}", flush=True)
    print(f"Inv_L: orig={o['geom']['Inv_L']:.3e}  patch={p['geom']['Inv_L']:.3e}", flush=True)
    print(f"scaled[alpha0]: orig={o['scaled']['alpha0']:.3e}  patch={p['scaled']['alpha0']:.3e}  ratio={p['scaled']['alpha0']/o['scaled']['alpha0']:.2f}×", flush=True)
    print(f"Ids @ OP: orig={o['op']['Ids']:.3e}  patch={p['op']['Ids']:.3e}", flush=True)
    print(f"Iii @ OP: orig={o['Iii_A']:.3e}  patch={p['Iii_A']:.3e}  ratio={p['Iii_A']/max(abs(o['Iii_A']),1e-30):.2f}×", flush=True)
    print(f"ngspice target Iii @ OP (patched card): ~4e-7 A", flush=True)

    (OUT / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved {OUT/'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
