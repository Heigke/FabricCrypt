"""Stage 6b probe v1 — dump pyport's compute_size_dep effective values
on M2 (with full binning loaded, no patch_model_values), and cross-check
against the BSIM4 v4.8.3 manual formula. Localises whether the 0.87 dec
gap is in the binning arithmetic, the geometry path, or downstream.

Geometry: M2 is L = Ln*M2_length_factor = 180e-9 * 10 = 1800e-9 m,
W = Wn = 360e-9 m, NF=1.

Reference (BSIM4 v4.8.3 manual eq. for binnable parameters, binunit=2):
    P_eff = P + LP / Leff + WP / Weff + PP / (Leff * Weff)

Card values (post Stage-3a/Stage-5 ngspice-faithful parsing, NO patch):
    vth0   = 0.54153
    wvth0  = -1.6569e-08
    lvth0  = 0
    pvth0  = -1.45e-15
    voff   = -0.1368
    wvoff  = -5.6e-9
    voffl  = -5.5973e-9   (NOTE: BSIM4 has TWO L-ish coefs; voffl is its own
                            term, distinct from lvoff. Confirm in audit.)
    vsat   = 102230
    pvsat  = 1.03e-9
    lpe0   = 1.2439e-7
    pags   = 3e-13
"""
from __future__ import annotations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "nsram"))

from nsram.bsim4_port.model_card import BSIM4Model, parse_param_blocks
from nsram.bsim4_port.geometry import Geometry, compute_geometry
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig

DATA = ROOT / "data/sebas_2026_04_22"

# Load M2 the way Stage 5 does, with no patch.
text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
shared = parse_param_blocks(text_M2)
m2 = BSIM4Model.from_spice(text_M2, model_type="nmos", params=shared)

cfg = NSRAMCell2TConfig()
geom_M2 = Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn, NF=1)
print(f"M2 geometry: L={geom_M2.L*1e9:.1f} nm, W={geom_M2.W*1e9:.1f} nm")

# Effective geometry
eff = compute_geometry(m2, geom_M2)
print(f"  Lnew = {eff.Lnew*1e9:.4f} nm")
print(f"  Wnew = {eff.Wnew*1e9:.4f} nm")
print(f"  dl   = {eff.dl*1e9:.6f} nm")
print(f"  dw   = {eff.dw*1e9:.6f} nm")
print(f"  leff = {eff.leff*1e9:.4f} nm")
print(f"  weff = {eff.weff*1e9:.4f} nm")
print(f"  binunit (model) = {m2['binunit']}")
print(f"  Inv_L  = {eff.Inv_L:.6e}")
print(f"  Inv_W  = {eff.Inv_W:.6e}")
print(f"  Inv_LW = {eff.Inv_LW:.6e}")
print()

# Pyport effective parameters (full compute_size_dep, T=27 C nominal)
sd = compute_size_dep(m2, geom_M2, T_C=cfg.T_C)
print(f"=== Pyport compute_size_dep effective values ===")
for name in ("vth0", "voff", "vsat", "lpe0", "k1", "k2", "k3", "ags",
              "u0", "ua", "ub", "uc", "nfactor", "etab", "vfb"):
    v = getattr(sd, name + "_T", None) or getattr(sd, name, None)
    if v is not None:
        print(f"  {name:10s} = {v}")

# Manual binning formula cross-check (binunit=2 expected per Sebas card)
def binned(model, name, eff):
    base = model.get(name, 0.0)
    l = model.get("l" + name, 0.0)
    w = model.get("w" + name, 0.0)
    p = model.get("p" + name, 0.0)
    return base, l * eff.Inv_L, w * eff.Inv_W, p * eff.Inv_LW

print(f"\n=== Manual binning breakdown (b4set.c eq. 5.1) ===")
print(f"{'param':10s}  {'base':>15s}  {'+l/Leff':>15s}  {'+w/Weff':>15s}  "
      f"{'+p/(L·W)':>15s}  {'eff':>15s}")
for name in ("vth0", "voff", "vsat", "lpe0", "k1", "k2", "k3", "u0",
              "nfactor", "ags"):
    base, dL, dW, dLW = binned(m2, name, eff)
    eff_v = base + dL + dW + dLW
    print(f"  {name:10s} {base:>15.6g} {dL:>+15.6g} {dW:>+15.6g} "
           f"{dLW:>+15.6g} {eff_v:>15.6g}")

# voffl is a SEPARATE term, not via standard l/w/p binning
print(f"\nvoff special:")
voffl = m2.get("voffl", 0.0)
print(f"  voff (card)   = {m2.get('voff'):.6g}")
print(f"  voffl (card)  = {voffl:.6g}")
print(f"  voffl/Leff    = {voffl / eff.leff:.6g}  (b4ld.c voffcbn)")
print(f"  voff + voffl/Leff = {m2.get('voff') + voffl/eff.leff:.6g}")
