"""z98 — diff every BSIM4 intermediate: pyport vs literal C-to-Python port.

A.5.w (2026-05-02): pyport is 10.83× too high in sub-VT vs ngspice on
isolated M2. We've audited 9 high-prior hypotheses, all falsified.
Now: bisect by literal-port comparison.

Method:
  1. Build the parameter dict P (binned, temperature-shifted) by
     reading from pyport's own SizeDependParam (which we trust per
     A.5.p — every binning matches the C source verbatim).
  2. Run literal-port `bsim4_compute(P, vgs, vds, vbs)` — pure-Python
     translation of b4ld.c §1042-1336.
  3. Run pyport's compute_dc on the same OP, extract intermediates.
  4. Print side-by-side; flag any value differing by > 1e-6 relative.
  5. Hopefully one of: Vbseff/Phis/Xdep/Theta0/Vth/n/T10/T9/Vgsteff
     diverges. The first divergence pinpoints the bug.
"""
from __future__ import annotations
import math, sys
from pathlib import Path
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "bsim4_literal_port"))
from bsim4_literal import bsim4_compute

import importlib.util
sp = importlib.util.spec_from_file_location("f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
f = importlib.util.module_from_spec(sp); sp.loader.exec_module(f)

from nsram.bsim4_port.geometry import Geometry, compute_geometry
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.dc import compute_dc


def build_param_dict(model, sd, geom):
    """Pack pyport's binned + temp-shifted values into the literal port's
    parameter dict, using EXACTLY what pyport believes the values to be."""
    P = sd.scaled
    ctx = sd.model_ctx
    return {
        "vbsc": sd.vbsc, "phi": sd.phi, "sqrtPhi": sd.sqrtPhi,
        "Xdep0": sd.Xdep0, "factor1": ctx.factor1,
        "vbi": sd.vbi, "leff": geom.leff, "weff": geom.weff,
        "w0": P.get("w0", model.get("w0", 2.5e-6)),
        "toxe": ctx.toxe, "coxe": ctx.coxe,
        "vtm": ctx.vtm, "vtm0": ctx.Vtm0,
        "dvt0": P["dvt0"], "dvt1": P["dvt1"], "dvt2": P["dvt2"],
        "dvt0w": P["dvt0w"], "dvt1w": P["dvt1w"], "dvt2w": P["dvt2w"],
        "eta0": P["eta0"], "etab": P["etab"],
        "dsub": P.get("dsub", model.get("dsub", model.get("drout", 0.56))),
        "theta0vb0": sd.theta0vb0,
        "k1ox": sd.k1ox, "k1": P["k1"], "k2ox": sd.k2ox,
        "lpe0": model.get("lpe0", 1.74e-7),
        "lpeb": model.get("lpeb", 0.0),
        "kt1": model.get("kt1", -0.11),
        "kt1l": model.get("kt1l", 0.0),
        "kt2": model.get("kt2", 0.022),
        "type": float(model._values.get("type", 1)),
        "vth0": sd.vth0_T,
        "k3": P.get("k3", model.get("k3", 80.0)),
        "k3b": P.get("k3b", model.get("k3b", 0.0)),
        "dvtp0": P["dvtp0"], "dvtp1": P["dvtp1"],
        "dvtp4": float(model.get("dvtp4", 0.0)),
        "dvtp2factor": float(model.get("dvtp2factor", 0.0)),
        "nfactor": P["nfactor"],
        "cdsc": P["cdsc"], "cdscb": P["cdscb"], "cdscd": P["cdscd"],
        "cit": P["cit"],
        "voffcbn": sd.voffcbn, "mstar": sd.mstar, "cdep0": sd.cdep0,
        "Tnom": ctx.Tnom, "Temp": ctx.Temp,
    }


def main():
    DATA = ROOT / "data/sebas_2026_04_22"
    text = (DATA / "M2_130bulkNSRAM.txt").read_text()
    m = BSIM4Model.from_spice(text, model_type="nmos")
    f.patch_model_values(m, type_n=True)
    cfg = NSRAMCell2TConfig()
    g = Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn)
    sd = compute_size_dep(m, g, T_C=27.0)
    eg = compute_geometry(m, g)

    Vgs, Vds, Vbs = 0.30, 0.5, 0.0
    print(f"=== M2 OP @ Vgs={Vgs} Vds={Vds} Vbs={Vbs} ===\n")

    # Run literal port
    P = build_param_dict(m, sd, eg)
    lit = bsim4_compute(P, Vgs, Vds, Vbs)

    # Run pyport via compute_dc (returns DCResult with limited intermediates)
    Vg = torch.tensor([Vgs]); Vd = torch.tensor([Vds]); Vb = torch.tensor([Vbs])
    out = compute_dc(model=m, sd=sd, Vgs=Vg, Vds=Vd, Vbs=Vb)

    def get(attr):
        v = getattr(out, attr, None)
        if v is None: return None
        if hasattr(v, "item") and v.numel() == 1: return float(v)
        if hasattr(v, "__getitem__"): return float(v[0])
        return float(v)

    # Print side-by-side
    print(f"{'quantity':<20s}  {'literal-port':>16s}  {'pyport':>16s}  {'rel diff':>10s}")
    print("-" * 70)
    pyport_vals = {
        "Vbseff": get("Vbseff"),
        "Vth": get("Vth"),
        "n": get("n"),
        "Vgsteff": get("Vgsteff"),
        "Vgst": get("Vth") and (Vgs - get("Vth")),
    }
    for k, lv in lit.items():
        py = pyport_vals.get(k)
        if py is None:
            print(f"  {k:<18s}  {lv:>16.6e}  {'(not exposed)':>16s}")
            continue
        if abs(lv) < 1e-30:
            rel = float("inf") if abs(py) > 1e-30 else 0.0
        else:
            rel = abs(py - lv) / abs(lv)
        flag = " ⚠⚠⚠" if rel > 1e-3 else (" ⚠" if rel > 1e-6 else "")
        print(f"  {k:<18s}  {lv:>16.6e}  {py:>16.6e}  {rel:>10.2e}{flag}")

    # Also print Ids comparison from literal port via simple linear approx
    print(f"\n  Pyport Ids = {get('Ids'):.4e}")
    print(f"  Linear-approx Ids from literal Vgsteff = ", end="")
    Vgsteff_lit = lit["Vgsteff"]
    n_lit = lit["n"]
    Vth_lit = lit["Vth"]
    # Approximate: Id ≈ μ·Cox·W/L·Vgsteff·Vds·(1 - Vds/(2·Vdsat))
    # In subVT, Vdsat tiny — use Vdseff = Vds·sat factor. Skip detail; just
    # report Vgsteff ratio which IS the bug location.
    print(f"(skip — same Ids would result if Vgsteff matches)")
    print(f"\n  Vgsteff ratio pyport/literal = {get('Vgsteff') / Vgsteff_lit:.3f}×")
    print(f"  ⇒ if literal == ngspice (which it should, being a verbatim port),")
    print(f"    pyport bug = Vgsteff_pyport / Vgsteff_literal — printed above.")


if __name__ == "__main__":
    main()
