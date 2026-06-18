"""z72 — Worst-point trace at Vgs=0.094, Vds=1.5, Vbs=-0.6, T=-20°C.

Goal: pinpoint the equation block causing 88.8% rel error in Ids when Vth
is correct. Suspects: n (subthreshold ideality), Vgsteff bridge, DIBL_Sft,
Vbseff JX clamp, Coxeff vs coxe in n.

Steps:
  S1: Worst-point Python intermediates + ngspice OP (with tightened tolerances)
  S2: Vds sweep at Vg=0.094, Vbs=-0.6, T=-20C: monotonic err vs Vds → DIBL/Vdseff
  S3: Synthetic ablation: model card with DIBL/SCE zeroed
  S4: Manual recompute of n, DIBL_Sft, Vgsteff bridge from b4ld.c eqs

Writes: results/bsim4_port_validation/worst_point_diagnostic.md
"""
from __future__ import annotations
import sys, json, math, copy, re
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "bsim4_port"))
sys.path.insert(0, str(ROOT / "nsram"))

from spice_oracle import Bias, Geometry as SpiceGeom, Sweep, run_op, run_dc_sweep  # type: ignore
from nsram.bsim4_port.dc import compute_dc  # type: ignore
from nsram.bsim4_port.geometry import Geometry  # type: ignore
from nsram.bsim4_port.model_card import BSIM4Model  # type: ignore
from nsram.bsim4_port.temp import compute_size_dep  # type: ignore

GOLD = ROOT / "results/bsim4_port_validation/gold/sebas130.json"
OUT_MD = ROOT / "results/bsim4_port_validation/worst_point_diagnostic.md"

gold = json.loads(GOLD.read_text())
model_text = gold["model_text"]
model = BSIM4Model.from_spice(model_text, model_type="nmos")
model_name = "NMOS"

WP = dict(Vgs=0.094, Vds=1.5, Vbs=-0.6, T_C=-20.0)
LW = dict(L=1.3e-7, W=1.0e-5)


def py_full(Vgs, Vds, Vbs, T_C, model_obj=None, L=1.3e-7, W=1.0e-5):
    """Return DCResult plus key intermediates by re-running compute_dc and
    also probing internals from sd."""
    m = model_obj or model
    g = Geometry(L=L, W=W, NF=1)
    sd = compute_size_dep(m, g, T_C=T_C)
    Vgs_t = torch.tensor([Vgs], dtype=torch.float64)
    Vds_t = torch.tensor([Vds], dtype=torch.float64)
    Vbs_t = torch.tensor([Vbs], dtype=torch.float64)
    res = compute_dc(m, sd, Vgs=Vgs_t, Vds=Vds_t, Vbs=Vbs_t)
    out = dict(
        Ids=float(res.Ids),
        Vth=float(res.Vth),
        Vgsteff=float(res.Vgsteff),
        Vdsat=float(res.Vdsat),
        Vdseff=float(res.Vdseff),
        n=float(res.n),
        Abulk=float(res.Abulk),
        mueff=float(res.mueff),
        Vbseff=float(res.Vbseff) if res.Vbseff is not None else float("nan"),
        Vgs_eff=float(res.Vgs_eff) if res.Vgs_eff is not None else float("nan"),
    )
    out["sd_phi"] = float(sd.phi)
    out["sd_Xdep0"] = float(sd.Xdep0)
    out["sd_vbi"] = float(sd.vbi)
    out["sd_vbsc"] = float(sd.vbsc)
    out["sd_k1ox"] = float(sd.k1ox)
    out["sd_k2ox"] = float(sd.k2ox)
    out["sd_vth0_T"] = float(sd.vth0_T)
    out["sd_factor1"] = float(sd.model_ctx.factor1)
    out["sd_Vtm"] = float(sd.model_ctx.vtm)
    out["sd_Temp"] = float(sd.model_ctx.Temp)
    out["sd_coxe"] = float(sd.model_ctx.coxe)
    out["sd_voffcbn"] = float(sd.voffcbn)
    out["sd_mstar"] = float(sd.mstar)
    out["sd_cdep0"] = float(sd.cdep0)
    out["sd_epssub"] = float(sd.model_ctx.epssub)
    return out, sd


def ng_op(Vgs, Vds, Vbs, T_C, model_text_use=None):
    bias = Bias(Vd=Vds, Vg=Vgs, Vs=0.0, Vb=Vbs)
    sg = SpiceGeom(L=1.3e-7, W=1.0e-5, NF=1)
    out = run_op(model_text_use or model_text, model_name, sg, bias, temp_C=T_C)
    # BSIM4 OP doesn't expose @m1[ids]; derive from drain branch current
    if "ids" not in out and "i_vd" in out:
        out["ids"] = -out["i_vd"]
    return out


def ng_dc_sweep_pt(Vgs_query, Vds, Vbs, T_C, model_text_use=None,
                   vg_lo=0.05, vg_hi=0.30, vg_step=0.002):
    """Use a DC sweep instead of OP — more reliable in deep subthreshold."""
    bias = Bias(Vd=Vds, Vg=vg_lo, Vs=0.0, Vb=Vbs)
    sg = SpiceGeom(L=1.3e-7, W=1.0e-5, NF=1)
    res = run_dc_sweep(model_text_use or model_text, model_name, sg, bias,
                       Sweep("Vg", vg_lo, vg_hi, vg_step),
                       save_op=["vth", "vdsat", "gm", "gds", "gmbs"],
                       temp_C=T_C)
    Vg_arr = res["Vg"]
    Ids_arr = -res["i_vd"]
    # interpolate at Vgs_query
    idx = int(np.argmin(np.abs(Vg_arr - Vgs_query)))
    return {
        "ids": float(Ids_arr[idx]),
        "vth": float(res["vth"][idx]),
        "vdsat": float(res["vdsat"][idx]),
        "gm": float(res["gm"][idx]),
        "vg_actual": float(Vg_arr[idx]),
    }


def relerr(p, n):
    if n is None or abs(n) < 1e-30:
        return float("nan")
    return (p - n) / n


def fmt(x):
    if x is None:
        return "  -  "
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if abs(x) > 0 and (abs(x) < 1e-3 or abs(x) > 1e3):
            return f"{x:.4e}"
        return f"{x:+.6f}"
    return str(x)


def manual_recompute(py, sd, Vgs, Vds, Vbs):
    """Recompute n, DIBL_Sft, Vgsteff manually from b4ld.c equations."""
    # Use Python's intermediate Vbseff
    Vbseff = py["Vbseff"]
    Vth = py["Vth"]
    Vgs_eff = py["Vgs_eff"]
    n_py = py["n"]
    Vgsteff_py = py["Vgsteff"]

    coxe = py["sd_coxe"]
    Vtm = py["sd_Vtm"]
    epssub = py["sd_epssub"]
    Xdep0 = py["sd_Xdep0"]
    factor1 = py["sd_factor1"]
    voffcbn = py["sd_voffcbn"]
    mstar = py["sd_mstar"]
    cdep0 = py["sd_cdep0"]
    phi = py["sd_phi"]

    # Phis, Xdep
    Phis = phi - Vbseff
    sqrtPhis = math.sqrt(max(Phis, 0))
    sqrtPhi = math.sqrt(phi)
    Xdep = Xdep0 * sqrtPhis / sqrtPhi

    # n manually (b4ld.c §1133-1154)
    nfactor = float(model.get("nfactor", 1.0))
    cdsc = float(model.get("cdsc", 0.0))
    cdscb = float(model.get("cdscb", 0.0))
    cdscd = float(model.get("cdscd", 0.0))
    cit = float(model.get("cit", 0.0))
    dvt1 = float(model.get("dvt1", 0.53))
    Leff = LW["L"]  # crude (Lpe/dl correction skipped)

    tmp1 = epssub / Xdep
    tmp2 = nfactor * tmp1
    tmp3 = cdsc + cdscb * Vbseff + cdscd * Vds
    # Theta0 computed in dc.py — recompute approx
    lt1_T1 = 1.0 + dvt1 * Vbseff if (dvt1 * Vbseff) >= -0.5 else None  # use scaled
    # Actually use sd.theta0vb0 if present, else exp(-0.5 dvt1 Leff/lt1)
    # Best approach: pull via re-eval
    T0_th = dvt1 * Leff / max(factor1 * math.sqrt(Xdep) * (1 + float(model.get("dvt2", -0.032)) * Vbseff), 1e-30)
    Theta0 = math.exp(-T0_th) if T0_th < 34 else 0.0
    tmp4 = (tmp2 + tmp3 * Theta0 + cit) / coxe
    n_manual = 1.0 + tmp4 if tmp4 >= -0.5 else (1.0 + 3.0 * tmp4) / (3.0 + 8.0 * tmp4)

    # Vgsteff bridge (b4ld.c §1238-1296)
    Vgst = Vgs_eff - Vth
    T0v = n_py * Vtm
    T2v = mstar * Vgst / T0v
    if T2v > 34:
        T10 = mstar * Vgst
    elif T2v < -34:
        T10 = T0v * math.log1p(math.exp(-34))  # tiny
    else:
        T10 = T0v * math.log1p(math.exp(T2v))
    T1_off = voffcbn - (1.0 - mstar) * Vgst
    T2_off = T1_off / T0v
    if T2_off < -34:
        T3v = coxe * math.exp(-34) / cdep0
    elif T2_off > 34:
        T3v = coxe * math.exp(34) / cdep0
    else:
        T3v = coxe / cdep0 * math.exp(T2_off)
    T9v = mstar + n_py * T3v
    Vgsteff_manual = T10 / T9v

    return {
        "Vbseff": Vbseff,
        "Phis": Phis,
        "Xdep": Xdep,
        "Theta0_approx": Theta0,
        "tmp1=epssub/Xdep": tmp1,
        "tmp2=nfactor*tmp1": tmp2,
        "tmp3=cdsc+cdscb*Vbs+cdscd*Vds": tmp3,
        "tmp4": tmp4,
        "n_manual": n_manual,
        "n_py": n_py,
        "n_diff": n_manual - n_py,
        "Vgst=Vgs_eff-Vth": Vgst,
        "T0v=n*Vtm": T0v,
        "T2v=mstar*Vgst/T0v": T2v,
        "T10_num": T10,
        "T2_off": T2_off,
        "T3v_denom": T3v,
        "T9v=mstar+n*T3v": T9v,
        "Vgsteff_manual": Vgsteff_manual,
        "Vgsteff_py": Vgsteff_py,
        "Vgsteff_diff": Vgsteff_manual - Vgsteff_py,
    }


def make_synthetic_model_card(zeros: list[str]) -> str:
    """Take Sebas's model_text; force certain params to 0."""
    txt = model_text
    for k in zeros:
        # Replace existing param 'k=val' in card; case-insensitive, whole word
        # Use multiple patterns. NMOS model only.
        pattern = re.compile(rf"\b({re.escape(k)})\s*=\s*[-+]?\d*\.?\d+(e[-+]?\d+)?", re.IGNORECASE)
        txt = pattern.sub(rf"\1=0", txt)
    return txt


# ============================================================================
# STEP 1 — Worst-point block-level
# ============================================================================
print(f"=== STEP 1: WORST POINT {WP} ===")
py, sd = py_full(**WP, **LW)
ng = ng_op(**WP)
print("ngspice OP keys:", sorted(ng.keys()))
print(f"  Ids:    py={fmt(py['Ids'])}  ng={fmt(ng.get('ids'))}  rerr={fmt(relerr(py['Ids'], ng.get('ids')))}")
print(f"  Vth:    py={fmt(py['Vth'])}  ng={fmt(ng.get('vth'))}  rerr={fmt(relerr(py['Vth'], ng.get('vth')))}")
print(f"  Vdsat:  py={fmt(py['Vdsat'])}  ng={fmt(ng.get('vdsat'))}")
print(f"  von:    ng={fmt(ng.get('von'))}")
print(f"  gm:     ng={fmt(ng.get('gm'))}")
print(f"  gds:    ng={fmt(ng.get('gds'))}")
print(f"  gmbs:   ng={fmt(ng.get('gmbs'))}")

# ============================================================================
# STEP 2 — Vds sweep at Vg=0.094, Vbs=-0.6, T=-20C
# ============================================================================
print(f"\n=== STEP 2: Vds sweep at Vgs=0.094, Vbs=-0.6, T=-20°C ===")
S2 = []
for Vds in (0.05, 0.5, 1.0, 1.5):
    p, _ = py_full(Vgs=0.094, Vds=Vds, Vbs=-0.6, T_C=-20.0, **LW)
    n = ng_op(Vgs=0.094, Vds=Vds, Vbs=-0.6, T_C=-20.0)
    e = relerr(p["Ids"], n.get("ids"))
    S2.append((Vds, p["Ids"], n.get("ids"), e, p["n"], p["Vgsteff"], p["Vth"]))
    print(f"  Vds={Vds:.2f}  Ids_py={p['Ids']:.4e}  Ids_ng={n.get('ids',0):.4e}  rerr={e:+.4f}  n={p['n']:.3f}  Vgsteff={p['Vgsteff']:.4e}  Vth={p['Vth']:+.4f}")

# ============================================================================
# STEP 3 — Synthetic ablation: zero DIBL/SCE
# ============================================================================
print(f"\n=== STEP 3: synthetic ablations ===")
ablations = [
    ("baseline", []),
    ("noDIBL_eta", ["eta0", "etab"]),
    ("noDIBL_pdiblc", ["pdiblc1", "pdiblc2"]),
    ("noDVT", ["dvt0", "dvt1", "dvt0w"]),
    ("allSCE_off", ["eta0", "etab", "pdiblc1", "pdiblc2", "dvt0", "dvt0w", "pdits", "dvtp0"]),
]
S3 = []
for name, zeros in ablations:
    try:
        m_text = make_synthetic_model_card(zeros) if zeros else model_text
        m_obj = BSIM4Model.from_spice(m_text, model_type="nmos")
        p, _ = py_full(**WP, **LW, model_obj=m_obj)
        n = ng_op(**WP, model_text_use=m_text)
        e = relerr(p["Ids"], n.get("ids"))
        S3.append((name, p["Ids"], n.get("ids"), e, p["Vth"], n.get("vth")))
        print(f"  {name:18s}  Ids_py={p['Ids']:.4e}  Ids_ng={n.get('ids',0):.4e}  rerr={e:+.4f}  Vth_py={p['Vth']:+.4f} Vth_ng={n.get('vth',0):+.4f}")
    except Exception as exc:
        print(f"  {name:18s}  FAIL: {exc}")
        S3.append((name, None, None, None, None, None))

# ============================================================================
# STEP 4 — manual recompute at worst point
# ============================================================================
print(f"\n=== STEP 4: manual recompute at worst point ===")
mrec = manual_recompute(py, sd, **{k: WP[k] for k in ("Vgs","Vds","Vbs")})
for k, v in mrec.items():
    print(f"  {k:32s} = {fmt(v) if isinstance(v,float) else v}")

# ============================================================================
# STEP 4b — Manually compute the Va chain factors at the worst point
# ============================================================================
print(f"\n=== STEP 4b: Va chain decomposition ===")
import math as _m
Leff = LW["L"]; Weff = LW["W"]
Vds_ = WP["Vds"]; Vbs_ = WP["Vbs"]
# pull from py
Vgsteff = py["Vgsteff"]; n_v = py["n"]; Vdsat = py["Vdsat"]; Vdseff = py["Vdseff"]
Abulk = py["Abulk"]; mueff = py["mueff"]
Vth = py["Vth"]
Vtm = py["sd_Vtm"]
diffVds = Vds_ - Vdseff
Vgst2Vtm = Vgsteff + 2.0 * Vtm
# ngspice gm/gds
gm_ng = ng.get("gm"); gds_ng = ng.get("gds")
print(f"  Vgsteff={Vgsteff:.4e}  Vdsat={Vdsat:.4e}  Vdseff={Vdseff:.4e}")
print(f"  diffVds=Vds-Vdseff={diffVds:.4e}  Vgst2Vtm={Vgst2Vtm:.4e}  Abulk={Abulk:.4f}  mueff={mueff:.4e}")
# CoxeffWovL approximation: use coxe (no centroid in this regime)
coxe = py["sd_coxe"]
Esat = 2.0 * (lambda: 0)()  # not available, derive from Vdsat
# Try to extract Esat from the structure: EsatL = Vdsat·... (skip)
# Va ≈ Vasat + VACLM
# VADIBL formula
pdiblc1 = float(model.get("pdiblc1", 0.0))
pdiblc2 = float(model.get("pdiblc2", 0.0))
drout = float(model.get("drout", 0.56))
dsub = float(model.get("dsub", drout))
toxe = float(py["sd_coxe"] and 1.0)  # placeholder
# Use sd directly
toxe_val = float(sd.model_ctx.toxe)
epsrox = float(sd.model_ctx.epsrox); EPS0_ = 8.854e-12; epssub_ = py["sd_epssub"]
Xdep0 = py["sd_Xdep0"]
# Recompute Phis & Xdep at WP
Phis = py["sd_phi"] - py["Vbseff"]
Xdep = Xdep0 * _m.sqrt(Phis) / _m.sqrt(py["sd_phi"])
tmp_dsub = _m.sqrt(epssub_ / (epsrox*EPS0_) * toxe_val * Xdep0)
T0_dr = drout * Leff / tmp_dsub
T5_dr = _m.exp(-T0_dr) if T0_dr < 34 else 1.0/(34-2)
thetaRout = pdiblc1 * T5_dr + pdiblc2
T8_db = Abulk * Vdsat
T0_db = Vgst2Vtm * T8_db
T1_db = Vgst2Vtm + T8_db
VADIBL = (Vgst2Vtm - T0_db / T1_db) / max(thetaRout, 1e-30)
pdiblb = float(model.get("pdiblb", 0.0))
T7_db = pdiblb * py["Vbseff"]
T3_pdb = 1.0/(1.0+T7_db) if T7_db >= -0.9 else (17.0+20.0*T7_db)/(0.8+T7_db)
VADIBL = VADIBL * T3_pdb  # PvagTerm=1 if pvag=0
print(f"  thetaRout={thetaRout:.4e}  VADIBL={VADIBL:.4e}  factor=(1+diffVds/VADIBL)={1+diffVds/VADIBL:.4e}")
# VADITS
pdits = float(model.get("pdits", 0.0))
pditsd = float(model.get("pditsd", 0.0))
pditsl = float(model.get("pditsl", 0.0))
T1_dits = _m.exp(pditsd * Vds_) if pditsd*Vds_ < 34 else _m.exp(34)
T2_dits = 1.0 + pditsl * Leff
VADITS = (1.0 + T2_dits * T1_dits) / max(pdits, 1e-30) if pdits > 0 else float("inf")
fprout_v = float(model.get("fprout", 0.0))
if fprout_v > 0:
    FP = 1.0 / (1.0 + fprout_v * _m.sqrt(Leff) / Vgst2Vtm)
else:
    FP = 1.0
VADITS = VADITS * FP if pdits > 0 else float("inf")
print(f"  pdits={pdits} VADITS={VADITS:.4e}  factor=(1+diffVds/VADITS)={1+diffVds/VADITS:.4e}  FP={FP:.4e}")
# VASCBE
pscbe1 = float(model.get("pscbe1", 4.24e8))
pscbe2 = float(model.get("pscbe2", 1e-5))
litl = float(sd.litl)
T0_scbe = pscbe1 * litl / max(diffVds, 1e-12)
VASCBE = Leff * _m.exp(min(T0_scbe, 34)) / pscbe2
print(f"  litl={litl:.4e}  pscbe1={pscbe1:.3e}  pscbe2={pscbe2:.3e}  T0_scbe=p1*litl/diff={T0_scbe:.4e}")
print(f"  VASCBE={VASCBE:.4e}  factor=(1+diffVds/VASCBE)={1+diffVds/VASCBE:.4e}")
# Vasat & VACLM  — need Esat·L
vsattemp = float(sd.vsattemp)
Esat = 2.0 * vsattemp / mueff
EsatL = Esat * Leff
pclm = float(model.get("pclm", 1.3))
WVCox = Weff * vsattemp * coxe
# Rds: rdsmod=0
print(f"  vsattemp={vsattemp:.4e}  Esat={Esat:.4e}  EsatL={EsatL:.4e}")
# Vasat with rdsmod=1 simplification (approx): T0=EsatL+Vdsat, T1=2/Lambda-1
Lambda = 1.0
WVCoxRds = 0.0  # if rdsmod=1
tmp4_va = 1.0 - 0.5 * Abulk * Vdsat / Vgst2Vtm
T0_va = EsatL + Vdsat + 2.0 * WVCoxRds * Vgsteff * tmp4_va
T1_va = 2.0 / Lambda - 1.0 + WVCoxRds * Abulk
Vasat_app = T0_va / T1_va
T0_clm = 1.0 + 0.0  # Rds*Idl
T2_clm = Vdsat / Esat
T1_clm = Leff + T2_clm
Cclm_app = T0_clm * T1_clm / (pclm * litl)
VACLM = Cclm_app * max(diffVds, 1e-12)
Va = Vasat_app + VACLM
log_VaVasat = max(_m.log(max(Va/Vasat_app, 1.0)), 0)
clm_factor = 1.0 + log_VaVasat / Cclm_app
print(f"  Vasat={Vasat_app:.4e}  Cclm={Cclm_app:.4e}  VACLM={VACLM:.4e}  Va={Va:.4e}")
print(f"  log(Va/Vasat)/Cclm = {log_VaVasat/Cclm_app:.4e}  CLM_factor={clm_factor:.4e}")
chain_total = (1+diffVds/VADIBL) * (1+diffVds/VADITS) * clm_factor * (1+diffVds/VASCBE)
print(f"  CHAIN_PRODUCT = {chain_total:.4e}  (DIBL*DITS*CLM*SCBE)")
# rdsmod
print(f"  rdsmod={int(model.get('rdsmod',0))}")

# Now compute Idl directly
# beta = mueff * Coxeff * W / L; we approximate Coxeff~coxe (no centroid in subthresh? maybe)
# but dc.py uses Coxeff with centroid
toxp = float(sd.toxp); ados = float(sd.ados); bdos = float(sd.bdos)
vtfbphi2 = float(sd.vtfbphi2); coxp = float(sd.coxp)
T0_tc = max((Vgsteff + vtfbphi2) / (2e8 * toxp), 1e-30)
tmp3_tc = _m.exp(bdos * 0.7 * _m.log(T0_tc))
T1_tc = 1.0 + tmp3_tc
Tcen = ados * 1.9e-9 / T1_tc
Coxeff = epssub_ * coxp / (epssub_ + coxp * Tcen)
beta = mueff * Coxeff * Weff / Leff
T0_idl = 1.0 - 0.5 * Vdseff * Abulk / Vgst2Vtm
fgche1 = Vgsteff * T0_idl
fgche2 = 1.0 + Vdseff / EsatL
gche = beta * fgche1 / fgche2
# Rds estimation (rdsmod=0)
prwg = float(model.get("prwg", 0.0)); prwb = float(model.get("prwb", 0.0))
rdsw_v = float(model.get("rdsw", 0.0))
T0_rds = 1.0 + prwg * Vgsteff
sqrtPhis = _m.sqrt(Phis); sqrtPhi = _m.sqrt(py["sd_phi"])
T9_rds = sqrtPhis - sqrtPhi
T1_rds = prwb * T9_rds
T2_rds = 1.0/T0_rds + T1_rds
T3_rds = T2_rds + _m.sqrt(T2_rds*T2_rds + 0.01)
# rdstemp: skip, take from sd if available
rdstemp = float(getattr(sd, "rdstemp", rdsw_v))
nf = 1.0; weffCJ_um = LW["W"]*1e6
wr = float(model.get("wr", 1.0))
PowWeffWr = max(weffCJ_um, 1e-30) ** wr
rds0_val = rdstemp * nf / PowWeffWr
T4_rds = rds0_val * 0.5
Rds_est = T3_rds * T4_rds
Idl_my = gche / (1.0 + gche*Rds_est)
print(f"  Tcen={Tcen:.4e}  Coxeff={Coxeff:.4e}  coxe={coxe:.4e}  beta={beta:.4e}")
print(f"  fgche1={fgche1:.4e}  fgche2={fgche2:.4e}  gche={gche:.4e}  Rds_est={Rds_est:.4e}")
print(f"  Idl_my={Idl_my:.4e}  Vdseff={Vdseff:.4e}")
Ids_recomputed = Idl_my * Vdseff * chain_total
print(f"  Ids_recomp = Idl·Vdseff·chain = {Ids_recomputed:.4e}  (py={py['Ids']:.4e})")
# Now: what does ngspice equivalent say? Ids_ng = 1.124e-12 -> implied Idl_ng·Vdseff·chain
# If Idl_ng·Vdseff = Ids_ng / chain
print(f"  Implied Idl_ng·Vdseff = Ids_ng/chain = {ng['ids']/chain_total:.4e}")
print(f"  Implied Idl_ng = (Ids_ng/chain)/Vdseff = {ng['ids']/chain_total/Vdseff:.4e}")
print(f"  Ratio gche_ng/gche_py = {(ng['ids']/chain_total/Vdseff)/gche:.4f}  (should be ~1.0)")

# ============================================================================
# STEP 5 — Vgs sweep at Vds=1.5, Vbs=-0.6, T=-20C  (capture err at Vg=0.094)
# ============================================================================
print(f"\n=== STEP 5: Vgs sweep around worst point ===")
S5 = []
for Vgs in (0.05, 0.094, 0.15, 0.20, 0.30, 0.50):
    p, _ = py_full(Vgs=Vgs, Vds=1.5, Vbs=-0.6, T_C=-20.0, **LW)
    n = ng_op(Vgs=Vgs, Vds=1.5, Vbs=-0.6, T_C=-20.0)
    e = relerr(p["Ids"], n.get("ids"))
    S5.append((Vgs, p["Ids"], n.get("ids"), e, p["Vgsteff"], p["n"]))
    print(f"  Vg={Vgs:.3f}  Ids_py={p['Ids']:.4e}  Ids_ng={n.get('ids',0):.4e}  rerr={e:+.4f}  Vgsteff={p['Vgsteff']:.4e}  n={p['n']:.3f}")

# ============================================================================
# Persist markdown report
# ============================================================================
def md_table(headers, rows):
    out = "| " + " | ".join(headers) + " |\n"
    out += "| " + " | ".join("---" for _ in headers) + " |\n"
    for r in rows:
        out += "| " + " | ".join(fmt(v) for v in r) + " |\n"
    return out


lines = [
    "# z72 — Worst-point trace at Vgs=0.094, Vds=1.5, Vbs=-0.6, T=-20°C\n",
    f"**Bias**: {WP}, L=130nm, W=10µm; Sebas130 NMOS card.\n",
    "## Step 1 — Block-level comparison\n",
    f"- Ids: py={fmt(py['Ids'])}, ng={fmt(ng.get('ids'))}, rel_err={fmt(relerr(py['Ids'], ng.get('ids')))}\n"
    f"- Vth: py={fmt(py['Vth'])}, ng={fmt(ng.get('vth'))}, rel_err={fmt(relerr(py['Vth'], ng.get('vth')))}\n"
    f"- Vdsat: py={fmt(py['Vdsat'])}, ng={fmt(ng.get('vdsat'))}\n"
    f"- gm={fmt(ng.get('gm'))}, gds={fmt(ng.get('gds'))}, gmbs={fmt(ng.get('gmbs'))}\n",
    "\n## Step 2 — Vds sweep (Vg=0.094, Vbs=-0.6, T=-20°C)\n",
    md_table(["Vds", "Ids_py", "Ids_ng", "rel_err", "n", "Vgsteff", "Vth"], S2),
    "\n## Step 3 — Synthetic ablations at worst point\n",
    md_table(["card", "Ids_py", "Ids_ng", "rel_err", "Vth_py", "Vth_ng"], S3),
    "\n## Step 4 — manual recompute (n, Vgsteff)\n",
    md_table(["Var", "Value"], [(k, v) for k, v in mrec.items()]),
    "\n## Step 5 — Vgs sweep at Vds=1.5\n",
    md_table(["Vgs", "Ids_py", "Ids_ng", "rel_err", "Vgsteff", "n"], S5),
]
OUT_MD.parent.mkdir(parents=True, exist_ok=True)
OUT_MD.write_text("\n".join(lines))
print(f"\nWrote {OUT_MD}")
