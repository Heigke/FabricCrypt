"""B05_PHYSCONST + B06_TEMPADJ — physical constants and temperature dependencies.

Faithful port of b4temp.c phases 5+6:
  - oxide setup (coxe, factor1, cgdo, cgso)
  - bandgap Eg(T), intrinsic ni
  - Vtm0 (at Tnom), vtm (at op T)
  - temperature shifts on Vth0, u0, vsat, Rds, Js (per-instance)

Returns a SizeDependParam dict mirroring the C `bsim4SizeDependParam` struct
fields most relevant for our DC + transient port.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any

from .constants import (
    Charge_q, EPS0, EPSSI, KboQ, PI, TZEROK,
)
from .geometry import EffectiveGeom, Geometry, compute_geometry
from .model_card import BSIM4Model


# ---- Model-level (size-independent) physics constants ----------------------

@dataclass
class ModelTempCtx:
    """Computed once per model + temperature. Bit-for-bit faithful."""
    Tnom: float            # nominal temp [K]
    Temp: float            # operating temp [K]
    delTemp: float         # Temp - Tnom
    TRatio: float          # Temp / Tnom
    Vtm0: float            # KboQ * Tnom
    vtm: float             # KboQ * Temp
    Eg0: float             # bandgap at Tnom
    Eg: float              # bandgap at Temp
    ni: float              # intrinsic carrier density at Tnom
    epsrox: float          # gate-ox dielectric constant
    epssub: float          # substrate permittivity (F/m)
    toxe: float            # effective gate-ox thickness
    coxe: float            # gate-ox capacitance per area
    factor1: float         # sqrt(epssub / (epsrox*EPS0) * toxe)


def compute_model_temp(model: BSIM4Model, T_C) -> ModelTempCtx:
    """Faithful port of b4temp.c lines 162-235 (model-level temperature setup).

    T_C may be a Python float OR a torch.Tensor (for fitting). When a tensor,
    we keep math.* on Python-float cached results: the cache is per scalar T,
    so callers must pass a scalar at cache-build time. For tensor T_C inside
    autograd graphs, downstream layers (dc.py) re-derive Vtm from Temp.
    """
    # Defensive: detach to scalar for the cached scalar pipeline. Callers
    # wanting tensor-T fitting should detour via a tensor-aware recompute.
    try:
        T_C_scalar = float(T_C.detach().item()) if hasattr(T_C, "detach") else float(T_C)
    except Exception:
        T_C_scalar = float(T_C)
    Tnom = model["tnom"] + TZEROK
    Temp = T_C_scalar + TZEROK
    delTemp = Temp - Tnom
    TRatio = Temp / Tnom

    # Oxide / substrate (mtrlMod = 0 path for now — Sebas uses Si)
    if model["mtrlmod"] != 0:
        epsrox = 3.9
        toxe = model["eot"]
        epssub = EPS0 * model["epsrsub"]
    else:
        epsrox = model["epsrox"]
        toxe = model["toxe"]
        epssub = EPSSI

    coxe = epsrox * EPS0 / toxe
    factor1 = math.sqrt(epssub / (epsrox * EPS0) * toxe)
    Vtm0 = KboQ * Tnom
    vtm = KboQ * Temp

    if model["mtrlmod"] == 0:
        Eg0 = 1.16 - 7.02e-4 * Tnom * Tnom / (Tnom + 1108.0)
        ni = (1.45e10 * (Tnom / 300.15) * math.sqrt(Tnom / 300.15)
              * math.exp(21.5565981 - Eg0 / (2.0 * Vtm0)))
        Eg = 1.16 - 7.02e-4 * Temp * Temp / (Temp + 1108.0)
    else:
        Eg0 = (model["bg0sub"] - model["tbgasub"] * Tnom * Tnom
                / (Tnom + model["tbgbsub"]))
        T0_b = (model["bg0sub"] - model["tbgasub"] * 300.15 * 300.15
                 / (300.15 + model["tbgbsub"]))
        ni = (model["ni0sub"] * (Tnom / 300.15) * math.sqrt(Tnom / 300.15)
              * math.exp((T0_b - Eg0) / (2.0 * Vtm0)))
        Eg = (model["bg0sub"] - model["tbgasub"] * Temp * Temp
                / (Temp + model["tbgbsub"]))

    return ModelTempCtx(
        Tnom=Tnom, Temp=Temp, delTemp=delTemp, TRatio=TRatio,
        Vtm0=Vtm0, vtm=vtm, Eg0=Eg0, Eg=Eg, ni=ni,
        epsrox=epsrox, epssub=epssub, toxe=toxe, coxe=coxe, factor1=factor1,
    )


# ---- Per-instance scaled + temperature-adjusted params ---------------------

# A short list of scaled params we need for our DC port. Each is:
#   pParam->X = model->X + l_X·Inv_L + w_X·Inv_W + p_X·Inv_LW
# where l_X, w_X, p_X are the model's "lX", "wX", "pX" coefficients.
# Default coefficients are 0, so for un-binned PDKs (e.g. Sebas's),
# pParam->X == model->X.
SCALED_PARAMS = [
    "vth0", "k1", "k2", "k3", "k3b", "w0",
    "dvt0", "dvt1", "dvt2", "dvt0w", "dvt1w", "dvt2w",
    "u0", "ua", "ub", "uc", "ud", "up", "lp",
    # WAVE3-FIX (z214): ua1/ub1/uc1/ud1 also bin via lX/wX/pX (b4temp.c §757-770)
    # and feed the ua/ub/uc/ud temperature shift below.  Without these the
    # mobility at T≠Tnom is wrong → Esat wrong → Vdsat ~50 mV high.
    "ua1", "ub1", "uc1", "ud1",
    # Also missing previously: b0/b1 enter Abulk (`b0/(Weff+b1)`).
    "b0", "b1",
    "vsat", "a0", "ags", "a1", "a2", "at",
    "keta", "nfactor", "cit", "cdsc", "cdscb", "cdscd",
    "eta0", "etab", "fprout", "pdits", "pditsd",
    "pclm", "pdiblc1", "pdiblc2", "pdiblcb", "drout", "dsub",
    "pscbe1", "pscbe2", "pvag",
    "delta", "rdsw", "rsw", "rdw", "prwg", "prwb", "wr",
    "alpha0", "alpha1", "beta0",
    "agidl", "bgidl", "cgidl", "egidl", "fgidl", "kgidl", "rgidl",
    "agisl", "bgisl", "cgisl", "egisl", "fgisl", "kgisl", "rgisl",
    "aigc", "bigc", "cigc", "aigsd", "bigsd", "cigsd",
    "aigbacc", "bigbacc", "cigbacc", "aigbinv", "bigbinv", "cigbinv",
    "nigc", "nigbacc", "nigbinv", "ntox", "eigbinv", "pigcd", "poxedge",
    "xrcrg1", "xrcrg2",
    "lambda", "vtl", "xn", "lc",
    "vfb", "tnoia", "tnoib", "rnoia", "rnoib", "rnoic",
    "ntnoi",
    "voff", "voffl", "voffcv", "voffcvl", "minv", "minvcv",
    "lpe0", "lpeb",
    "phin", "ndep", "nsd", "ngate",
    "xt", "xj", "vbm",
    "dvtp0", "dvtp1", "dvtp2", "dvtp3", "dvtp4", "dvtp5",
]


@dataclass
class SizeDependParam:
    """Cached per-(model, geometry, T) scaled + temp-adjusted params.

    Fields are dynamically populated; we don't pre-declare 200+ slots.
    """
    geom: EffectiveGeom
    model_ctx: ModelTempCtx
    scaled: dict[str, float] = field(default_factory=dict)
    # Temp-adjusted shadows of selected scaled params
    vth0_T: float = 0.0
    u0temp: float = 0.0
    vsattemp: float = 0.0
    rdstemp: float = 0.0
    Vth_T: float = 0.0
    SourceSatCurDensity_T: float = 0.0
    DrainSatCurDensity_T: float = 0.0
    # Pre-computed Vth/Xdep quantities (from b4temp.c §1300-1520)
    phi: float = 0.0           # surface potential 2φF
    sqrtPhi: float = 0.0
    phis3: float = 0.0
    Xdep0: float = 0.0         # depletion width at zero bias
    sqrtXdep0: float = 0.0
    vbi: float = 0.0           # built-in pn voltage
    vbsc: float = 0.0          # body-bias saturation clamp
    vbm: float = 0.0           # body-bias min
    k1ox: float = 0.0          # k1 × toxe/toxm
    k2ox: float = 0.0          # k2 × toxe/toxm
    theta0vb0: float = 0.0     # short-channel DIBL prefactor
    litl: float = 0.0          # screening length
    # Vgsteff regularizers (b4temp.c §1373-1427)
    mstar: float = 0.5
    voffcbn: float = 0.0
    cdep0: float = 0.0
    # Tcen / Coxeff capMod=2 inputs (b4ld.c §1789-1805, b4temp.c §1786)
    vtfbphi2: float = 0.0      # 4·(vth0 - vfb - phi); clamped ≥0
    coxp: float = 0.0          # gate-ox cap using toxp (defaults to coxe)
    toxp: float = 0.0          # poly oxide thickness (defaults to toxe)
    ados: float = 1.0
    bdos: float = 1.0
    # Other cached scalars
    vfb_eff: float = 0.0       # type·vth0 - vfb - phi sign branch input


def compute_size_dep(model: BSIM4Model, geom: Geometry, T_C: float) -> SizeDependParam:
    """Faithful port of b4temp.c phases 3+4.

    Pipeline:
      1. compute geometry → EffectiveGeom (with Inv_L, Inv_W, Inv_LW)
      2. compute model temp ctx (Eg, ni, Vtm, coxe, factor1)
      3. for each scaled param X, apply: pParam.X = model.X + l_X·Inv_L + w_X·Inv_W + p_X·Inv_LW
      4. apply temperature shifts on selected scaled params (Vth0, u0, vsat, Js)
    """
    eff = compute_geometry(model, geom)
    ctx = compute_model_temp(model, T_C)

    scaled: dict[str, float] = {}
    for X in SCALED_PARAMS:
        base = model.get(X, 0.0)
        # Coefficient names are lX, wX, pX (lvth0, wvth0, pvth0, ...).
        l_X = model.get("l" + X, 0.0)
        w_X = model.get("w" + X, 0.0)
        p_X = model.get("p" + X, 0.0)
        scaled[X] = base + l_X * eff.Inv_L + w_X * eff.Inv_W + p_X * eff.Inv_LW

    # Temperature adjustments (b4temp.c §1208-1300, simplified subset).
    delTemp = ctx.delTemp
    TRatio = ctx.TRatio

    # Vth0 temperature shift LIVES IN dc.py (b4ld.c:1099-1103, the only place
    # in the C source where it appears). DO NOT also apply it here — that
    # double-counts. Diagnostic z71 confirmed this was the root cause of the
    # 109% subthreshold worst-case rel err (Wave 2 finding 2026-04-29).
    # Reference: b4temp.c never touches vth0 — kt1/kt1l/kt2 only enter via
    # b4ld.c §1099-1103 inside the per-bias Vth assembly.
    vth0_T = scaled["vth0"]
    Tm1 = TRatio - 1.0   # used by vsattemp and rdstemp below

    # Mobility: u0temp = u0 · Tratio^ute  (mobMod ≠ 3 path, b4temp.c:1283)
    ute = model["ute"]
    u0temp = scaled["u0"] * (TRatio ** ute)

    # WAVE3-FIX (z214): mobility-coefficient temperature shifts.  b4temp.c
    # §1202-1241.  tempmod=0 path:  ua += ua1·T0, ub += ub1·T0, uc += uc1·T0,
    # ud += ud1·T0.  Without these, ua/ub/uc are off by 30-300% at T≠Tnom →
    # Esat wrong → Vdsat off by ~50 mV.  We mutate the `scaled` dict in place
    # so dc.py picks up the temperature-shifted values via `P["ua"]`.
    T0_uab = Tm1   # = TRatio - 1, the "T0" the C source uses on §1196.
    if model["tempmod"] == 0:
        scaled["ua"] = scaled["ua"] + scaled["ua1"] * T0_uab
        scaled["ub"] = scaled["ub"] + scaled["ub1"] * T0_uab
        scaled["uc"] = scaled["uc"] + scaled["uc1"] * T0_uab
        scaled["ud"] = scaled["ud"] + scaled["ud1"] * T0_uab
    elif model["tempmod"] == 3:
        scaled["ua"] = scaled["ua"] * (TRatio ** scaled["ua1"])
        scaled["ub"] = scaled["ub"] * (TRatio ** scaled["ub1"])
        scaled["uc"] = scaled["uc"] * (TRatio ** scaled["uc1"])
        scaled["ud"] = scaled["ud"] * (TRatio ** scaled["ud1"])
    else:  # tempmod = 1, 2
        scaled["ua"] = scaled["ua"] * (1.0 + scaled["ua1"] * delTemp)
        scaled["ub"] = scaled["ub"] * (1.0 + scaled["ub1"] * delTemp)
        scaled["uc"] = scaled["uc"] * (1.0 + scaled["uc1"] * delTemp)
        scaled["ud"] = scaled["ud"] * (1.0 + scaled["ud1"] * delTemp)

    # Saturation velocity (b4temp.c:1208 mtrlMod=0 path):
    #   vsattemp = vsat - at · (Temp/Tnom - 1)         tempMod=0
    #   vsattemp = vsat * (1 - at·delTemp)             tempMod≠0
    # We use tempMod=0 form (matches what ngspice does for tempmod=0 default).
    at_v = scaled["at"]
    if model["tempmod"] == 0:
        vsattemp = scaled["vsat"] - at_v * Tm1
    else:
        vsattemp = scaled["vsat"] * (1.0 - at_v * delTemp)

    # Series resistance temp scaling: rds(T) = rdsw_T · (1 + prt·...)
    # Simplified: not actually used in our minimal port until b4ld.c rdsmod
    prt = model["prt"]
    rds_factor = 1.0 + prt * Tm1
    rdstemp = scaled["rdsw"] * rds_factor

    # Junction saturation current density temperature dependence
    # b4temp.c lines 1461-1510 region; we use the canonical Eg/Vtm shift:
    Js0 = model["jss"]
    # Js(T) = Js0 · (Tratio)^xtis · exp[(Eg0/Vtm0 - Eg/vtm)/Nj]
    xtis = model["xtis"]
    nj = model.get("njs", 1.0)
    if nj <= 0:
        nj = 1.0
    T0_eg = (ctx.Eg0 / ctx.Vtm0) - (ctx.Eg / ctx.vtm)
    js_temp_factor = (TRatio ** xtis) * math.exp(T0_eg / nj)
    SjctSatT = Js0 * js_temp_factor

    Js0_d = model["jsd"]
    xtid = model["xtid"]
    nj_d = model.get("njd", 1.0)
    if nj_d <= 0:
        nj_d = 1.0
    js_temp_factor_d = (TRatio ** xtid) * math.exp(T0_eg / nj_d)
    DjctSatT = Js0_d * js_temp_factor_d

    # ---- Pre-computed Vth/Xdep quantities (b4temp.c §1322-1520) -----------
    # phi = Vtm0·log(NDEP/ni) + phin + 0.4
    ndep = max(model["ndep"], 1e10)   # safety: never log(0)
    phi = ctx.Vtm0 * math.log(ndep / max(ctx.ni, 1e-30)) + model["phin"] + 0.4
    if phi <= 0:
        phi = 0.4   # fallback; should warn if Sebas's ndep produces this
    sqrtPhi = math.sqrt(phi)
    phis3 = sqrtPhi * phi
    # Xdep0 = sqrt(2·epssub/(q·NDEP·1e6)) · sqrtPhi  (NDEP in cm⁻³ → m⁻³ via 1e6)
    Xdep0 = math.sqrt(2.0 * ctx.epssub / (Charge_q * ndep * 1.0e6)) * sqrtPhi
    sqrtXdep0 = math.sqrt(Xdep0)
    # vbi = Vtm0·log(NSD·NDEP/ni²)
    nsd = max(model["nsd"], 1e10)
    vbi = ctx.Vtm0 * math.log(nsd * ndep / max(ctx.ni * ctx.ni, 1e-60))
    # vbsc body-bias clamp: simplified — use vbm if given, else -3.0
    vbm = model["vbm"]
    vbsc = vbm if vbm < 0 else -3.0
    # k1ox, k2ox: oxide-thickness scaling (b4temp.c §1516, 1802)
    # Same ref-default ordering issue as toxp: re-resolve toxm against toxe.
    if model.is_given("toxm"):
        toxm = model["toxm"] if model["toxm"] > 0 else ctx.toxe
    else:
        toxm = ctx.toxe
    k1ox = scaled["k1"] * ctx.toxe / toxm
    k2ox = scaled["k2"] * ctx.toxe / toxm
    # litl screening length (b4temp.c §1347)
    epsrox = ctx.epsrox
    litl = math.sqrt(3.0 * 3.9 / epsrox * model["xj"] * ctx.toxe)
    # theta0vb0 — DIBL prefactor at zero Vbs (b4temp.c §1538-1560)
    # Theta0 = exp(-0.5·dvt0·Leff/litl) approx (full form b4ld.c handles)
    # We compute a simple approximation here; full form computed in dc.py.
    theta0vb0 = math.exp(-0.5 * scaled["dvt0"] * eff.leff / max(litl, 1e-12))

    # ---- Vgsteff bridge regularizers (b4temp.c §1373, 1425-1427) ----------
    # mstar = 0.5 + atan(minv)/pi
    minv_v = scaled.get("minv", model.get("minv", 0.0))
    mstar = 0.5 + math.atan(minv_v) / math.pi
    # voffcbn = voff + voffl/Leff
    voff_v = scaled.get("voff", model.get("voff", -0.08))
    voffl_v = model.get("voffl", 0.0)
    voffcbn = voff_v + voffl_v / max(eff.leff, 1e-12)
    # cdep0 = sqrt(q·epssub·NDEP·1e6 / 2 / phi)   (b4temp.c §1373-1375)
    ndep_scaled = max(scaled.get("ndep", ndep), 1e10)
    cdep0 = math.sqrt(Charge_q * ctx.epssub * ndep_scaled * 1.0e6 / 2.0 / max(phi, 1e-3))

    # ---- Tcen / Coxeff inputs (b4ld.c §1789-1805, b4temp.c §1786, §180) ----
    # coxp = epsrox·EPS0 / toxp; if toxp not given, fall back to toxe.
    # NOTE: model_card.py resolves "ref" defaults BEFORE user overrides, so
    # toxp ends up frozen to the default-toxe value (3e-9) even if the user
    # set toxe=4e-9. Re-resolve here using is_given().
    if model.is_given("toxp"):
        toxp = model.get("toxp", 0.0)
    else:
        toxp = ctx.toxe
    if toxp <= 0:
        toxp = ctx.toxe
    coxp = ctx.epsrox * EPS0 / toxp
    # vtfbphi2 = 4·(type·vth0 - vfb - phi); clamped ≥0 (b4temp.c §1786-1788)
    type_n = float(model._values.get("type", 1)) if hasattr(model, "_values") else 1.0
    vfb_card = model.get("vfb", -1.0)
    T3_vfb = type_n * vth0_T - vfb_card - phi
    vtfbphi2 = max(4.0 * T3_vfb, 0.0)
    # ados/bdos defaults are 1.0 each
    ados = model.get("ados", 1.0)
    bdos = model.get("bdos", 1.0)

    return SizeDependParam(
        geom=eff,
        model_ctx=ctx,
        scaled=scaled,
        vth0_T=vth0_T,
        u0temp=u0temp,
        vsattemp=vsattemp,
        rdstemp=rdstemp,
        Vth_T=vth0_T,
        SourceSatCurDensity_T=SjctSatT,
        DrainSatCurDensity_T=DjctSatT,
        phi=phi, sqrtPhi=sqrtPhi, phis3=phis3,
        Xdep0=Xdep0, sqrtXdep0=sqrtXdep0,
        vbi=vbi, vbsc=vbsc, vbm=vbm,
        k1ox=k1ox, k2ox=k2ox,
        theta0vb0=theta0vb0, litl=litl,
        mstar=mstar, voffcbn=voffcbn, cdep0=cdep0,
        vtfbphi2=vtfbphi2, coxp=coxp, toxp=toxp,
        ados=ados, bdos=bdos,
        vfb_eff=T3_vfb,
    )
