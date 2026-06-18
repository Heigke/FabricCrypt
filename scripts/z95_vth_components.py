"""z95 — A.5.c Vth aggregator decomposition for isolated M2.

z91l confirmed pyport Vth is uniformly ~57 mV LOW vs ngspice across
Vds∈{0.05, 0.5, 2.0}V (DIBL matches). The offset is a constant
additive bias somewhere in dc.py:349-356.

Method: for each line of the Vth assembly, compute the term inline
using the same SizeDependParam outputs (sd.vth0_T, sd.vbi, sd.phi,
sd.k1ox, ...) that compute_dc would use. Print each contribution
at Vbs=0, Vds=0.05V, M2 (L=1.8u, W=0.36u). Compare against ngspice's
@m1[vth] probe (operating point). Whichever line differs by ~57 mV
is the bug.
"""
from __future__ import annotations
import math, subprocess, tempfile, json
from pathlib import Path
import importlib.util
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/z95_vth_components"
OUT.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location(
    "z91j_mod", ROOT / "scripts/z91j_ngspice_isolated_m2.py")
z91j = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(z91j)

from nsram.bsim4_port.geometry import Geometry
from nsram.bsim4_port.nsram_cell_2T import NSRAMCell2TConfig
from nsram.bsim4_port.temp import compute_size_dep
from nsram.bsim4_port.model_card import BSIM4Model
from nsram.bsim4_port.smooth import safe_sqrt, safe_exp

DATA = ROOT / "data/sebas_2026_04_22"


def decompose_vth_pyport(model, sd, Vbs=0.0, Vds=0.05):
    """Replicate dc.py:140-403 Vth construction inline, term by term."""
    geom = sd.geom; ctx = sd.model_ctx; P = sd.scaled

    Leff = geom.leff; Weff = geom.weff
    toxe = ctx.toxe; coxe = ctx.coxe
    Vtm = ctx.vtm; epssub = ctx.epssub; factor1 = ctx.factor1

    type_n = float(model._values.get("type", 1))
    vth0  = sd.vth0_T
    k1    = P["k1"];   k2 = P["k2"]
    k3    = P.get("k3", model.get("k3", 80.0))
    k3b   = P.get("k3b", model.get("k3b", 0.0))
    w0    = P.get("w0", model.get("w0", 2.5e-6))
    dvt0  = P["dvt0"]; dvt1 = P["dvt1"]; dvt2 = P["dvt2"]
    dvt0w = P["dvt0w"]; dvt1w = P["dvt1w"]; dvt2w = P["dvt2w"]
    eta0  = P["eta0"]; etab = P["etab"]
    Xdep0 = sd.Xdep0; sqrtPhi = sd.sqrtPhi; phi = sd.phi
    vbi   = sd.vbi
    k1ox  = sd.k1ox; k2ox = sd.k2ox
    kt1   = model.get("kt1", -0.11); kt1l = model.get("kt1l", 0.0)
    kt2   = model.get("kt2", 0.022)
    lpe0  = model.get("lpe0", 1.74e-7); lpeb = model.get("lpeb", 0.0)
    dsub  = P.get("dsub", model.get("dsub", model.get("drout", 0.56)))

    # --- Vbseff (Vbs=0 ⇒ Vbseff=0) ---
    Vbseff = 0.0  # at Vbs=0 with Sebas's vbsc=-3..-5, this stays 0

    # --- Phis, sqrtPhis, Xdep ---
    Phis = phi - Vbseff
    sqrtPhis = math.sqrt(Phis)
    Xdep = Xdep0 * sqrtPhis / sqrtPhi

    # --- V0, lt1 ---
    V0 = vbi - phi
    T0 = dvt2 * Vbseff
    T1 = 1.0 + T0  # Vbseff=0 ⇒ T1=1
    lt1 = factor1 * math.sqrt(Xdep) * T1

    # --- Theta0, Delt_vth ---
    T0_th = dvt1 * Leff / max(lt1, 1e-30)
    Theta0 = math.exp(-T0_th) if T0_th < 30 else math.exp(-30)
    Delt_vth = dvt0 * Theta0 * V0

    # --- T2_narrow ---
    T0w = dvt2w * Vbseff
    T1w = 1.0 + T0w
    ltw = factor1 * math.sqrt(Xdep) * T1w
    T0_w = dvt1w * Weff * Leff / max(ltw, 1e-30)
    T5 = math.exp(-T0_w) if T0_w < 30 else 0.0
    T2_narrow = dvt0w * T5 * V0

    # --- Tlpe1, Vth_NarrowW ---
    TempRatio = ctx.Temp / ctx.Tnom - 1.0
    T0_lpe = math.sqrt(1.0 + lpe0 / Leff)
    Tlpe1 = (k1ox * (T0_lpe - 1.0) * sqrtPhi
             + (kt1 + kt1l/Leff + kt2*Vbseff) * TempRatio)
    Vth_NarrowW = toxe * phi / (Weff + w0)
    Lpe_Vb = math.sqrt(1.0 + lpeb / Leff)

    # --- DIBL_Sft ---
    T3_d = eta0 + etab * Vbseff
    T9_d = 1.0 / (3.0 - 2.0e4 * T3_d)
    T3_clamped = (2.0e-4 - T3_d) * T9_d if T3_d < 1.0e-4 else T3_d
    epsrox_t = ctx.epsrox
    EPS0 = 8.8542e-12
    tmp_dsub = math.sqrt(epssub / (epsrox_t * EPS0) * toxe * Xdep0)
    T0_dsub = dsub * Leff / max(tmp_dsub, 1e-40)
    theta0vb0_dsub = math.exp(-T0_dsub) if T0_dsub < 30 else 0.0
    DIBL_Sft = T3_clamped * theta0vb0_dsub * Vds

    # --- DITS (only if dvtp0 > 0) ---
    DITS = 0.0
    if float(model.get("dvtp0", 0.0)) > 0.0:
        dvtp0 = P["dvtp0"]; dvtp1 = P["dvtp1"]
        T0_p = -dvtp1 * Vds
        T2_p = math.exp(T0_p) if T0_p < 30 else math.exp(30)
        T3_p = Leff + dvtp0 * (1.0 + T2_p)
        T4_p = Vtm * math.log(Leff / T3_p)
        # n unknown analytically; approximate as 1.5 (subthreshold-ish)
        DITS = -1.5 * T4_p

    # --- DITS_Sft2 ---
    dvtp4 = float(model.get("dvtp4", 0.0))
    dvtp2factor = float(model.get("dvtp2factor", 0.0))
    DITS_Sft2 = dvtp2factor * math.tanh(dvtp4 * Vds) if (dvtp4 != 0 and dvtp2factor != 0) else 0.0

    # --- Final assembly (line by line) ---
    term_vth0       = type_n * vth0
    term_k1k2       = (k1ox * sqrtPhis - k1 * sqrtPhi) * Lpe_Vb
    term_k2ox       = -k2ox * Vbseff
    term_DVT        = -Delt_vth
    term_T2_narrow  = -T2_narrow
    term_k3_narrow  = (k3 + k3b * Vbseff) * Vth_NarrowW
    term_Tlpe1      = Tlpe1
    term_DIBL_Sft   = -DIBL_Sft
    term_DITS       = DITS
    term_DITS_Sft2  = -DITS_Sft2

    Vth = (term_vth0 + term_k1k2 + term_k2ox + term_DVT + term_T2_narrow
           + term_k3_narrow + term_Tlpe1 + term_DIBL_Sft
           + term_DITS + term_DITS_Sft2)

    return {
        "Vth_total": Vth,
        "vth0_T":             term_vth0,
        "(k1ox*√Φs - k1*√Φ)*Lpe_Vb": term_k1k2,
        "-k2ox*Vbseff":       term_k2ox,
        "-Delt_vth (DVT)":    term_DVT,
        "-T2_narrow":         term_T2_narrow,
        "+(k3+k3b*Vbs)*Vth_NarrowW": term_k3_narrow,
        "+Tlpe1 (lateral pkt)": term_Tlpe1,
        "-DIBL_Sft":          term_DIBL_Sft,
        "+DITS (n*Vtm*log)":  term_DITS,
        "-DITS_Sft2":         term_DITS_Sft2,
        # Reference quantities
        "_phi":      phi,
        "_sqrtPhi":  sqrtPhi,
        "_vbi":      vbi,
        "_V0=vbi-phi": V0,
        "_Xdep0":    Xdep0,
        "_lt1":      lt1,
        "_Theta0":   Theta0,
        "_T0_lpe":   T0_lpe,
        "_T0_dsub":  T0_dsub,
        "_theta0vb0_dsub": theta0vb0_dsub,
        "_lpe0":     lpe0,
        "_w0":       w0,
        "_k1":       k1, "_k2": k2, "_k1ox": k1ox, "_k2ox": k2ox,
        "_dvt0":     dvt0, "_dvt1": dvt1, "_dvt2": dvt2,
    }


def ngspice_op_probes(geom, Vds=0.05, Vgs=0.6, Vbs=0.0):
    """Pull operating-point probes for Vth and friends."""
    card = z91j.make_ngspice_card_inline()
    cir = f"""* z95 op probes
{card}
VD D 0 DC {Vds:g}
VG G 0 DC {Vgs:g}
VS S 0 DC 0
VB B 0 DC {Vbs:g}
M1 D G S B NMOSSEB L={geom.L:g} W={geom.W:g}
.options gmin=1e-15 reltol=1e-6 abstol=1e-16
.control
op
print @m1[vth] @m1[vfb] @m1[vbseff] @m1[phi] @m1[vfbsd] @m1[vdsat]
print @m1[xdep] @m1[gm] @m1[gds] @m1[gmbs]
quit
.endc
.end
"""
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(cir); cn = f.name
    res = subprocess.run(["ngspice", "-b", cn], capture_output=True, text=True, timeout=60)
    return res.stdout + "\n---STDERR---\n" + res.stderr


def main():
    text_M2 = (DATA / "M2_130bulkNSRAM.txt").read_text()
    model = BSIM4Model.from_spice(text_M2, model_type="nmos")
    z91j.z91f.patch_model_values(model, type_n=True)
    cfg = NSRAMCell2TConfig()
    geom = Geometry(L=cfg.Ln * cfg.M2_length_factor, W=cfg.Wn)
    sd = compute_size_dep(model, geom, T_C=27.0)

    print(f"[z95] M2 geom: L={geom.L:g}  W={geom.W:g}  Vds=0.05  Vbs=0")
    print(f"[z95] === pyport Vth decomposition ===")
    d = decompose_vth_pyport(model, sd, Vbs=0.0, Vds=0.05)
    Vth_total = d["Vth_total"]
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if k == "Vth_total":
            print(f"  {k:>40s} = {v:+.6f}")
        else:
            print(f"  {k:>40s} = {v:+.6f}    ({100*v/Vth_total:+6.1f}%)")

    print(f"\n[z95] === intermediate values ===")
    for k, v in d.items():
        if not k.startswith("_"):
            continue
        print(f"  {k:>20s} = {v}")

    print(f"\n[z95] === ngspice operating-point probes (M2, Vd=0.05, Vg=0.6) ===")
    op = ngspice_op_probes(geom, Vds=0.05, Vgs=0.6, Vbs=0.0)
    # Filter print/measurement lines
    for line in op.splitlines():
        if any(k in line for k in ["vth", "vfb", "phi", "xdep", "vbseff", "vdsat", "gm", "gds"]):
            print("  ", line)

    json.dump({k: float(v) for k, v in d.items()}, (OUT / "summary.json").open("w"), indent=2)
    (OUT / "ngspice_op.txt").write_text(op)
    print(f"\n[z95] saved {OUT}/summary.json + ngspice_op.txt")
    print(f"[z95] expected ngspice Vth ≈ +0.580 V (from z91l).")
    print(f"[z95] pyport Vth_total here = {Vth_total:+.4f} V; gap = {0.580 - Vth_total:+.4f} V")


if __name__ == "__main__":
    main()
