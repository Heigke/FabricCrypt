"""z369 — M2 card audit, mirror of z344_iimod_verify (R-51).

Apply the same audits to M2 that R-26/R-37 applied to M1:
  1. binunit handling in patch_model_values (M1 had binunit=2 in card, prior
     override forced binunit=1 which collapsed l-binning by 1e6).
  2. lalpha0 cancellation (M1 card: alpha0=7.84e-5 + lalpha0=-9.84e-12;
     in ngspice with binunit=2 these subtract because lalpha0*1/Leff has
     magnitude ~6.3e-5, nearly cancelling alpha0).
  3. lpe0 / toxe values (these are SHARED_PARAM-resolved; check pyport
     uses the same numbers ngspice settles on).

For each M2 card variant (original, LALPHA0_FIX), dump:
  - raw alpha0, lalpha0, binunit
  - patched values (after patch_model_values)
  - scaled.alpha0 (after compute_size_dep at M2 geometry: L = Ln*10)
  - leff, Inv_L
  - lpe0, toxe in model._values
  - Iii at flagship bias for M2 (VG2=0.20 is M2's gate, Vd=Vsint=0.382,
    Vs=0, Vb=Vb_M2=0 (m2_body_gnd default True))

Compare M2 Iii values across card variants; report whether cancellation /
binunit pathology repeats in pyport at M2 geometry.
"""
from __future__ import annotations
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

import sys, json, importlib.util
from pathlib import Path
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "nsram"))

OUT = ROOT / "results/z369_m2_audit"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data/sebas_2026_04_22"

from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.leak import compute_iimpact
from nsram.bsim4_port.dc import compute_dc
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.geometry import Geometry

# Pull patch_model_values from z91f
sp = importlib.util.spec_from_file_location(
    "z91f", ROOT / "scripts/z91f_validate_with_sebas_params.py")
z91f = importlib.util.module_from_spec(sp); sp.loader.exec_module(z91f)
patch_model_values = z91f.patch_model_values
M2_STATIC_OVERRIDES = z91f.M2_STATIC_OVERRIDES


def probe(card_filename: str, label: str) -> dict:
    text = (DATA / card_filename).read_text()
    M2 = BSIM4Model.from_spice(text, model_type="nmos")
    # raw values BEFORE patch
    raw = {
        "alpha0":  M2._values.get("alpha0"),
        "lalpha0": M2._values.get("lalpha0"),
        "alpha1":  M2._values.get("alpha1"),
        "beta0":   M2._values.get("beta0"),
        "binunit": M2._values.get("binunit"),
        "lpe0":    M2._values.get("lpe0"),
        "wlpe0":   M2._values.get("wlpe0"),
        "toxe":    M2._values.get("toxe"),
        "toxp":    M2._values.get("toxp"),
        "toxm":    M2._values.get("toxm"),
        "k1":      M2._values.get("k1"),
        "etab":    M2._values.get("etab"),
    }

    patch_model_values(M2, type_n=True)
    after = {
        "alpha0":  M2._values.get("alpha0"),
        "lalpha0": M2._values.get("lalpha0"),
        "alpha1":  M2._values.get("alpha1"),
        "beta0":   M2._values.get("beta0"),
        "binunit": M2._values.get("binunit"),
        "lpe0":    M2._values.get("lpe0"),
        "wlpe0":   M2._values.get("wlpe0"),
        "toxe":    M2._values.get("toxe"),
        "toxp":    M2._values.get("toxp"),
        "toxm":    M2._values.get("toxm"),
        "k1":      M2._values.get("k1"),
        "etab":    M2._values.get("etab"),
    }

    cfg = NSRAMCell2TConfig(use_iii=True, use_gidl=True, use_bjt=True)
    # M2 geometry: length = Ln * M2_length_factor (default 10×).
    L_M2 = cfg.Ln * cfg.M2_length_factor
    geom = Geometry(L=L_M2, W=cfg.Wn, NF=1)
    sd = compute_size_dep(M2, geom, T_C=cfg.T_C)
    sc = {
        "alpha0": sd.scaled.get("alpha0"),
        "alpha1": sd.scaled.get("alpha1"),
        "beta0":  sd.scaled.get("beta0"),
        "k1":     sd.scaled.get("k1"),
        "etab":   sd.scaled.get("etab"),
        "nfactor": sd.scaled.get("nfactor"),
    }
    leff = float(sd.geom.leff)
    Inv_L = float(sd.geom.Inv_L)

    # Apply M2_STATIC_OVERRIDES like forward_2t (k1, k2, etab, beta0 patch sd.scaled).
    sd_patched = {}
    for k, v in M2_STATIC_OVERRIDES.items():
        sd_patched[k] = float(v)

    # M2 OP at flagship VG1=0.6, VG2=0.20: M2.G=VG2=0.20, M2.D=Vsint, M2.S=0, M2.B=0
    # (m2_body_gnd default). Vsint from ngspice ≈ 0.382 V.
    Vg = torch.tensor(0.20)
    Vd_node = torch.tensor(0.382)
    Vs_node = torch.tensor(0.0)
    Vb_node = torch.tensor(0.0)  # m2_body_gnd
    Vgs = Vg - Vs_node
    Vds = Vd_node - Vs_node
    Vbs = Vb_node - Vs_node

    dc_result = compute_dc(M2, sd, Vgs=Vgs, Vds=Vds, Vbs=Vbs)
    Ids = float(dc_result.Ids)
    Vdseff = float(dc_result.Vdseff)
    Idsa = float(getattr(dc_result, "Idsa", torch.tensor(float("nan"))))

    Iii = compute_iimpact(M2, sd, dc_result, Vds=Vds)
    Iii_val = float(Iii)

    # Compute the cancellation magnitude: alpha0_eff_raw = alpha0 + lalpha0/Leff
    # at binunit=2 (ngspice native). If binunit override forces 1, lalpha0 collapses.
    alpha0_raw = raw["alpha0"] or 0.0
    lalpha0_raw = raw["lalpha0"] or 0.0
    # ngspice binunit=2: Inv_L = 1.0/Leff [m^-1] → contribution = lalpha0/Leff
    contrib_b2 = lalpha0_raw / leff
    # binunit=1: Inv_L = 1e-6/Leff [µm^-1] → contribution = lalpha0*1e-6/Leff
    contrib_b1 = lalpha0_raw * 1e-6 / leff
    alpha0_eff_b2 = alpha0_raw + contrib_b2
    alpha0_eff_b1 = alpha0_raw + contrib_b1

    return {
        "label": label,
        "card_file": card_filename,
        "raw": raw,
        "after_patch": after,
        "geom": {"Ln_cfg": cfg.Ln, "M2_length_factor": cfg.M2_length_factor,
                 "L_M2": L_M2, "leff": leff, "Inv_L": Inv_L},
        "scaled_pre_static": sc,
        "M2_STATIC_OVERRIDES_applied": sd_patched,
        "op": {"Vgs": float(Vgs), "Vds": float(Vds), "Vbs": float(Vbs),
               "Ids": Ids, "Vdseff": Vdseff, "Idsa": Idsa},
        "Iii_A": Iii_val,
        "cancellation_analysis": {
            "alpha0_raw": alpha0_raw,
            "lalpha0_raw": lalpha0_raw,
            "lalpha0_div_leff_(binunit=2_ngspice)": contrib_b2,
            "lalpha0_x_1e-6_div_leff_(binunit=1)": contrib_b1,
            "alpha0_eff_if_binunit2": alpha0_eff_b2,
            "alpha0_eff_if_binunit1": alpha0_eff_b1,
            "ratio_b2_over_b1": alpha0_eff_b2 / max(abs(alpha0_eff_b1), 1e-30),
        },
    }


def main():
    results = {}
    results["original"] = probe(
        "M2_130bulkNSRAM.txt",
        "original M2 (lalpha0=-9.84e-12, alpha0=7.84e-5)")
    results["patched"] = probe(
        "M2_130bulkNSRAM_LALPHA0_FIX.txt",
        "patched M2 (lalpha0=0)")

    print(json.dumps(results, indent=2, default=str), flush=True)

    o = results["original"]; p = results["patched"]
    print("\n=== SUMMARY ===", flush=True)
    print(f"binunit (raw):       orig={o['raw']['binunit']}  patch={p['raw']['binunit']}", flush=True)
    print(f"binunit (post):      orig={o['after_patch']['binunit']}  patch={p['after_patch']['binunit']}  (None means not overridden — pyport keeps card value)", flush=True)
    print(f"leff (M2):           orig={o['geom']['leff']:.3e}  patch={p['geom']['leff']:.3e}", flush=True)
    print(f"Inv_L (M2):          orig={o['geom']['Inv_L']:.3e}  patch={p['geom']['Inv_L']:.3e}", flush=True)
    print(f"alpha0 raw:          orig={o['raw']['alpha0']}  patch={p['raw']['alpha0']}", flush=True)
    print(f"lalpha0 raw:         orig={o['raw']['lalpha0']}  patch={p['raw']['lalpha0']}", flush=True)
    print(f"scaled[alpha0]:      orig={o['scaled_pre_static']['alpha0']:.6e}  patch={p['scaled_pre_static']['alpha0']:.6e}", flush=True)
    print(f"  alpha0_eff_b2 (ngspice native): {o['cancellation_analysis']['alpha0_eff_if_binunit2']:.3e}  vs patch {p['cancellation_analysis']['alpha0_eff_if_binunit2']:.3e}", flush=True)
    print(f"  alpha0_eff_b1 (pyport prior override): {o['cancellation_analysis']['alpha0_eff_if_binunit1']:.3e}  vs patch {p['cancellation_analysis']['alpha0_eff_if_binunit1']:.3e}", flush=True)
    print(f"lpe0 (raw/post):     raw_orig={o['raw']['lpe0']}  post_orig={o['after_patch']['lpe0']}", flush=True)
    print(f"toxe (raw/post):     raw_orig={o['raw']['toxe']}  post_orig={o['after_patch']['toxe']}", flush=True)
    print(f"Ids M2 @ OP:         orig={o['op']['Ids']:.3e}  patch={p['op']['Ids']:.3e}", flush=True)
    print(f"Iii M2 @ OP:         orig={o['Iii_A']:.3e}  patch={p['Iii_A']:.3e}", flush=True)
    if abs(o['Iii_A']) > 0:
        print(f"Iii ratio patched/orig: {p['Iii_A']/o['Iii_A']:.3f}×", flush=True)

    # Bug flags
    print("\n=== BUG FLAGS ===", flush=True)
    # Bug 1: binunit override? patch_model_values does NOT set binunit, so M2 retains
    # whatever the card said (binunit=2). But the M2 SCALED computation in
    # compute_size_dep uses Inv_L which depends on binunit interpretation.
    bug_binunit = (o['after_patch']['binunit'] != o['raw']['binunit'])
    print(f"  binunit altered by patch_model_values?  {bug_binunit}  "
          f"(raw={o['raw']['binunit']} → post={o['after_patch']['binunit']})", flush=True)

    # Bug 2: alpha0_eff cancellation in pyport scaled vs ngspice
    py_alpha0 = o['scaled_pre_static']['alpha0']
    ngspice_alpha0 = o['cancellation_analysis']['alpha0_eff_if_binunit2']
    ratio = py_alpha0 / max(abs(ngspice_alpha0), 1e-30)
    print(f"  pyport scaled.alpha0 vs ngspice (binunit=2) alpha0_eff:", flush=True)
    print(f"    pyport={py_alpha0:.3e}  ngspice_native={ngspice_alpha0:.3e}  ratio={ratio:.3f}×", flush=True)
    big_mismatch = abs(ratio) > 100 or abs(ratio) < 0.01
    print(f"  100×+ mismatch?  {big_mismatch}", flush=True)

    # Bug 3: M2_STATIC_OVERRIDES forces beta0=18 (correct, M2 card has 18 not 19 like M1).
    # The beta0 in scaled_pre_static is what compute_size_dep produced (no l-binning).
    # If lbeta0 exists in M2 card it could also be impacted by binunit.
    print(f"  scaled.beta0 (pre-static override): {o['scaled_pre_static']['beta0']}", flush=True)
    print(f"  M2_STATIC_OVERRIDES.beta0: {M2_STATIC_OVERRIDES['beta0']}", flush=True)

    (OUT / "m2_audit.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved {OUT/'m2_audit.json'}", flush=True)


if __name__ == "__main__":
    main()
