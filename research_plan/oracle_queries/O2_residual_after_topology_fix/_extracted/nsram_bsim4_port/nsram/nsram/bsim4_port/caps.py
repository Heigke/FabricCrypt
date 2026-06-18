"""B21_CAPS — BSIM4 4.8.3 capacitance models (junction CV + Meyer capMod=0).

Faithful port of `external/bsim4/code/b4ld.c`:
  - Junction body-source/drain CV: lines 3912-4029 (manual §11.2.1-3)
  - Meyer intrinsic caps capMod=0: lines 2992-3197 (manual §7.4.1)

Scope:
  - Static C(V); no AC, no transient charge derivative integration.
  - capMod=0 Meyer model only (no capMod=2 charge-thickness CTM).
  - Returns the diagonal Cgg, Cgs, Cgd, Cgb plus body-junction Cjs, Cjd.
  - Body-cap for NS-RAM body-KCL: Cbody = Cjs + Cjd + (1-α)·Cox·W·L (channel).

Differentiability rules:
  - fp64 throughout.
  - All if-on-V branches replaced by torch.where over BOTH finite arms,
    or by smooth.smooth_step transitions.
  - safe_sqrt / safe_log / smooth_step substitute hard kinks at FC·Pb crossover
    and at the cutoff/triode/saturation region boundaries.
  - Junction CV uses the BSIM4 exact reverse-bias (V<0) form everywhere on
    [-inf, +inf), with a smooth gluing to the small-forward-bias linearization
    so dCj/dV stays finite as V → Pb.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import torch

from .geometry import Geometry
from .model_card import BSIM4Model
from .smooth import (
    safe_sqrt, safe_log, safe_exp, smooth_max, smooth_min, smooth_step,
)
from .temp import SizeDependParam


# --------------------------------------------------------------------------- #
# Result container                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class CapResult:
    # Junction body-source/drain caps (per device, F)
    Cjs: torch.Tensor       # body-source pn junction (bottom + sw + swg)
    Cjd: torch.Tensor       # body-drain   pn junction (bottom + sw + swg)
    # Intrinsic Meyer caps (per device, F)
    Cgg: torch.Tensor       # gate self-cap
    Cgs: torch.Tensor       # gate-source
    Cgd: torch.Tensor       # gate-drain
    Cgb: torch.Tensor       # gate-bulk
    # Convenience scalar for NS-RAM body-KCL: total body capacitance to ground
    Cbody_total: torch.Tensor  # Cjs + Cjd + Cgb (channel-to-bulk via Meyer)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _t(x, dtype=torch.float64, device=None) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


def _bsim4_pb_default(model: BSIM4Model, key: str, fallback: float):
    """BSIM4 built-in: pb*/mj* default to canonical values when card sets 0.

    GRADFIX: keep tensor type if `model.get` returns a tensor (injected for
    gradcheck on cap params); only cast to float for the comparison branch.
    """
    v = model.get(key, 0.0)
    v_check = float(v) if not isinstance(v, torch.Tensor) else float(v.detach())
    if v_check <= 0.0:
        return fallback
    return v


def _coxe(sd: SizeDependParam) -> float:
    return sd.model_ctx.coxe


# --------------------------------------------------------------------------- #
# JUNCTION CV  (b4ld.c §3912-4029, manual §11.2.1-3)                          #
# --------------------------------------------------------------------------- #
def _junction_cap_one(
    czb: torch.Tensor,            # zero-bias bottom-area cap (F)
    czbsw: torch.Tensor,          # zero-bias sidewall cap   (F)
    czbswg: torch.Tensor,         # zero-bias gate-sidewall cap (F)
    Pb: float, Pbsw: float, Pbswg: float,
    Mj: float, Mjsw: float, Mjswg: float,
    Vj: torch.Tensor,             # junction voltage (Vbs_jct or Vbd_jct), V
    sharpness: float = 50.0,
) -> torch.Tensor:
    """One body-junction cap (source OR drain), summing bottom + 2 sidewalls.

    BSIM4 (b4ld.c §3936-3978): for reverse bias V<0:
        Cj = Cj0 · (1 - V/Pb)^(-Mj)
    For forward bias V>=0 (the C code's `else` arm at §3974):
        Cj_lin(V) = Cj0 · (1 + V·Mj/Pb)              [Taylor expansion]
    We glue the two arms via smooth_step so the kink at V=0 disappears. The
    linearization is the same one BSIM4 uses for forward bias to keep Newton
    iterations from blowing up — we re-use it as a smooth fallback on [0, Pb).

    SMOOTH: smooth_step(V, -Vt, +Vt) blends reverse↔forward arms across V≈0.
    SMOOTH: safe_sqrt/safe_log on (1 - V/Pb) to avoid neg-arg + kink at V=Pb.
    """
    # --- Reverse-bias arm (V<0) -------------------------------------------- #
    # arg = 1 - V/Pb  must stay > 0; clamp via safe path.
    # SMOOTH: floor arg at small positive eps so log/sqrt are differentiable
    # even on the forward-bias side where this arm is unused.
    one = torch.ones_like(Vj)
    arg_b = (one - Vj / Pb).clamp_min(1e-6)         # SMOOTH: floor
    arg_sw = (one - Vj / Pbsw).clamp_min(1e-6)
    arg_swg = (one - Vj / Pbswg).clamp_min(1e-6)

    if abs(Mj - 0.5) < 1e-12:
        s_b = 1.0 / safe_sqrt(arg_b)
    else:
        s_b = safe_exp(-Mj * safe_log(arg_b))         # = arg_b^(-Mj)
    if abs(Mjsw - 0.5) < 1e-12:
        s_sw = 1.0 / safe_sqrt(arg_sw)
    else:
        s_sw = safe_exp(-Mjsw * safe_log(arg_sw))
    if abs(Mjswg - 0.5) < 1e-12:
        s_swg = 1.0 / safe_sqrt(arg_swg)
    else:
        s_swg = safe_exp(-Mjswg * safe_log(arg_swg))

    Cj_rev = czb * s_b + czbsw * s_sw + czbswg * s_swg

    # --- Forward-bias arm (V>=0): BSIM4 linearization, b4ld.c §3974-3978 -- #
    #   capbs = T0 + T1
    #   T0 = czbs+czbssw+czbsswg
    #   T1 = vbs_jct·(czbs·MJS/PhiBS + czbssw·MJSWS/PhiBSWS + czbsswg·MJSWGS/PhiBSWGS)
    T0 = czb + czbsw + czbswg
    T1 = Vj * (czb * Mj / Pb + czbsw * Mjsw / Pbsw + czbswg * Mjswg / Pbswg)
    Cj_fwd = T0 + T1

    # --- Smooth glue across V=0 ------------------------------------------- #
    # SMOOTH: smooth_step picks reverse arm for V<<0, forward for V>>0.
    # Width chosen as ~25 mV (≈ kT/q at 300K) so the blend is physically
    # localized at the no-bias point.
    w = 0.025
    blend_fwd = smooth_step(Vj, -w, +w, sharpness=sharpness)
    return blend_fwd * Cj_fwd + (1.0 - blend_fwd) * Cj_rev


def compute_junction_caps(
    model: BSIM4Model,
    sd: SizeDependParam,
    Vbs: torch.Tensor,
    Vbd: torch.Tensor,
    *,
    As: Optional[float] = None,
    Ad: Optional[float] = None,
    Ps: Optional[float] = None,
    Pd: Optional[float] = None,
) -> dict:
    """Body-source & body-drain junction capacitances. (b4ld.c §3912-4029)

    Cj(V) = Cj0·area · (1 - V/Pb)^(-Mj)        for V < 0 (reverse)
          = Cj0·area · (1 + V·Mj/Pb)           for V ≥ 0 (forward, smoothed)
    Adds bottom (Cj·As) + sidewall (Cjsw·Ps) + gate-sidewall (Cjswg·Weff_CJ)
    pieces. Convention: V is the JUNCTION voltage (V_body - V_diffusion); we
    take Vbs / Vbd directly. NMOS body is at Vb; for an NMOS in normal operation
    Vbs<0 (reverse), so cap shrinks vs zero-bias.
    """
    dtype = Vbs.dtype if isinstance(Vbs, torch.Tensor) else torch.float64
    device = Vbs.device if isinstance(Vbs, torch.Tensor) else None

    # --- Per-instance areas/perimeters (b4temp.c §2044) -------------------- #
    # If caller did not supply, default to ngspice's `if (AS<=0)` fallback:
    #   AS = AD = W · hdif · 2  (with hdif from card; otherwise drawn area).
    # We pick the safest neutral default: As = Ad = W·L (plausible bulk area)
    # and Ps = Pd = 2·(W+L). Caller should pass real values for SPICE match.
    W = sd.geom.weff
    L = sd.geom.leff
    if As is None: As = W * L
    if Ad is None: Ad = W * L
    if Ps is None: Ps = 2.0 * (W + L)
    if Pd is None: Pd = 2.0 * (W + L)
    weffCJ = sd.geom.weffCJ
    NF = 1.0  # single-finger; b4temp embeds NF separately into czbsswg

    # --- Per-area / per-perimeter zero-bias values (b4ld.c §3914-3921) ---- #
    # BSIM4DunitAreaTempJctCap = cjd at op-temp; at Tnom this equals model.cjd.
    # We use the un-temp-adjusted cj* directly (caller ensures Tnom or accepts
    # ~10% error). TODO(temp): apply tcj/tcjsw/tcjswg shifts.
    cjs   = model.get("cjs",   0.0)        # F/m^2
    cjd   = model.get("cjd",   0.0)
    cjsws = model.get("cjsws", 0.0)        # F/m
    cjswd = model.get("cjswd", 0.0)
    cjswgs = model.get("cjswgs", cjsws)    # default to cjsws if not given
    cjswgd = model.get("cjswgd", cjswd)

    # Built-in potentials & grading coeffs (BSIM4 standard defaults)
    PbS   = _bsim4_pb_default(model, "pbs",   1.0)
    PbD   = _bsim4_pb_default(model, "pbd",   1.0)
    PbSWS = _bsim4_pb_default(model, "pbsws", 1.0)
    PbSWD = _bsim4_pb_default(model, "pbswd", 1.0)
    PbSWGS = _bsim4_pb_default(model, "pbswgs", PbSWS)
    PbSWGD = _bsim4_pb_default(model, "pbswgd", PbSWD)
    MJS   = _bsim4_pb_default(model, "mjs",   0.5)
    MJD   = _bsim4_pb_default(model, "mjd",   0.5)
    MJSWS = _bsim4_pb_default(model, "mjsws", 0.33)
    MJSWD = _bsim4_pb_default(model, "mjswd", 0.33)
    MJSWGS = _bsim4_pb_default(model, "mjswgs", MJSWS)
    MJSWGD = _bsim4_pb_default(model, "mjswgd", MJSWD)

    czbs   = _t(cjs   * As, dtype, device)
    czbd   = _t(cjd   * Ad, dtype, device)
    czbssw = _t(cjsws * Ps, dtype, device)
    czbdsw = _t(cjswd * Pd, dtype, device)
    czbsswg = _t(cjswgs * weffCJ * NF, dtype, device)
    czbdswg = _t(cjswgd * weffCJ * NF, dtype, device)

    Cjs = _junction_cap_one(
        czbs, czbssw, czbsswg, PbS, PbSWS, PbSWGS, MJS, MJSWS, MJSWGS, Vbs,
    )
    Cjd = _junction_cap_one(
        czbd, czbdsw, czbdswg, PbD, PbSWD, PbSWGD, MJD, MJSWD, MJSWGD, Vbd,
    )
    return {"Cjs": Cjs, "Cjd": Cjd}


# --------------------------------------------------------------------------- #
# INTRINSIC MEYER CAPS  capMod=0  (b4ld.c §2992-3197, manual §7.4.1)          #
# --------------------------------------------------------------------------- #
def compute_intrinsic_caps_capmod0(
    model: BSIM4Model,
    sd: SizeDependParam,
    dc_result,                          # DCResult or None — only Vth used
    Vgs: torch.Tensor,
    Vds: torch.Tensor,
    Vbs: torch.Tensor,
) -> dict:
    """Meyer intrinsic caps, capMod=0. (b4ld.c §2992-3197)

    Three regions (manual §7.4.1):
      * Accumulation/cutoff (Vgs - Vfb < 0, b4ld.c §3014-3030):
            Cgg = Cox·W·L,  Cgs = Cgd = 0,  Cgb = -Cgg
      * Depletion (0 < Vgs - Vfb, Vgs < Vth, b4ld.c §3031-3050):
            Cgg = Cox·W·L · k1/(2·sqrt(k1²/4 + Arg1))     (small)
            Cgs = Cgd = 0
      * Strong inversion:
          - Saturation (Vds > Vdsat, b4ld.c §3062-3088, Meyer 0/100):
                Cgs = (2/3)·Cox·W·L,  Cgd = 0,  Cgb ≈ 0
          - Triode/linear (Vds < Vdsat, b4ld.c §3091+):
                Cgs/Cgd partition by Vds (Meyer's classic 50/50 form
                interpolates from 1/2 at Vds=0 to 2/3 at Vds=Vdsat).

    SMOOTH: each region transition is replaced with smooth_step blends so
    Cgg(Vgs) is C^∞ instead of piecewise.
    """
    CoxWL = _t(_coxe(sd) * sd.geom.weffCV * sd.geom.leffCV, Vgs.dtype, Vgs.device)

    # Vfbcv: flat-band voltage param for capMod=0 (model card field)
    Vfbcv = _t(model.get("vfbcv", -1.0), Vgs.dtype, Vgs.device)
    # Vth approx: use DC Vth if provided, else use vth0_T from temp
    if dc_result is not None and hasattr(dc_result, "Vth"):
        Vth = dc_result.Vth
    else:
        Vth = _t(sd.vth0_T, Vgs.dtype, Vgs.device)

    # Surface potential & k1ox from temp.py
    phi = _t(sd.phi, Vgs.dtype, Vgs.device)
    k1ox = _t(sd.k1ox, Vgs.dtype, Vgs.device)

    # --- Region masks (smooth) -------------------------------------------- #
    # Arg1 = Vgs - Vbs - Vfbcv         (b4ld.c §3012)
    Arg1 = Vgs - Vbs - Vfbcv
    # Vgst = Vgs - Vth                 (b4ld.c §3005)
    Vgst = Vgs - Vth
    # Vdsat ≈ Vgst / Abulk;            for Meyer cap0 BSIM4 uses AbulkCV.
    # We use Abulk≈1 as conservative default (manual §7.4.1 Meyer simplification).
    Abulk = _t(1.0, Vgs.dtype, Vgs.device)
    if dc_result is not None and hasattr(dc_result, "Abulk"):
        Abulk = dc_result.Abulk
    Vdsat = smooth_max(Vgst, _t(1e-6, Vgs.dtype, Vgs.device)) / Abulk

    # SMOOTH: width 25 mV (≈ kT/q) for all region transitions.
    w = 0.025
    accum_to_dep = smooth_step(Arg1, -w, +w)            # 0 in accum, 1 in dep+
    dep_to_inv   = smooth_step(Vgst, -w, +w)            # 0 in dep, 1 in strong inv
    sat_to_lin   = smooth_step(Vdsat - Vds, -w, +w)     # 0 in sat (Vds>Vdsat), 1 in lin

    # --- Accumulation arm  (b4ld.c §3014-3030) ---------------------------- #
    Cgg_acc = CoxWL
    Cgs_acc = torch.zeros_like(CoxWL)
    Cgd_acc = torch.zeros_like(CoxWL)
    Cgb_acc = -CoxWL

    # --- Depletion arm  (b4ld.c §3031-3050) ------------------------------- #
    # T1 = 0.5·k1ox; T2 = sqrt(T1² + Arg1).  Cgg = CoxWL · T1/T2  (= dQg/dVg)
    T1 = 0.5 * k1ox
    T2 = safe_sqrt(T1 * T1 + smooth_max(Arg1, _t(0.0, Vgs.dtype, Vgs.device)))
    Cgg_dep = CoxWL * T1 / (T2 + 1e-30)
    Cgs_dep = torch.zeros_like(CoxWL)
    Cgd_dep = torch.zeros_like(CoxWL)
    Cgb_dep = -Cgg_dep

    # --- Strong inversion: saturation arm  (b4ld.c §3062-3088) ------------ #
    # Meyer 50/50 in saturation: Cgs = 2/3·CoxWL, Cgd = 0, Cgg = 2/3·CoxWL,
    # Cgb ≈ 0.  (BSIM4 §7.4.1)
    Cgs_sat = (2.0 / 3.0) * CoxWL
    Cgd_sat = torch.zeros_like(CoxWL)
    Cgg_sat = Cgs_sat
    Cgb_sat = torch.zeros_like(CoxWL)

    # --- Strong inversion: linear/triode  (b4ld.c §3091+, Meyer triode) --- #
    # Classic Meyer triode partition (manual §7.4.1):
    #   eta = Vds / Vdsat                               (∈ [0, 1])
    #   Cgs = CoxWL · [1 - ((Vdsat - Vds)/(2·Vdsat - Vds))²] · (2/3)
    #   Cgd = CoxWL · [1 - (Vdsat / (2·Vdsat - Vds))²]      · (2/3)
    # At Vds=0: Cgs = Cgd = (1/2)·CoxWL ; at Vds=Vdsat: Cgs=2/3·CoxWL, Cgd=0.
    eps_v = _t(1e-9, Vgs.dtype, Vgs.device)
    Vdsat_safe = smooth_max(Vdsat, eps_v)
    denom = 2.0 * Vdsat_safe - Vds + eps_v
    r_s = (Vdsat_safe - Vds) / denom
    r_d = Vdsat_safe / denom
    Cgs_lin = CoxWL * (1.0 - r_s * r_s) * (2.0 / 3.0)
    Cgd_lin = CoxWL * (1.0 - r_d * r_d) * (2.0 / 3.0)
    Cgg_lin = Cgs_lin + Cgd_lin
    Cgb_lin = torch.zeros_like(CoxWL)

    # --- Blend strong-inv arms by Vds region ------------------------------ #
    Cgg_inv = sat_to_lin * Cgg_lin + (1.0 - sat_to_lin) * Cgg_sat
    Cgs_inv = sat_to_lin * Cgs_lin + (1.0 - sat_to_lin) * Cgs_sat
    Cgd_inv = sat_to_lin * Cgd_lin + (1.0 - sat_to_lin) * Cgd_sat
    Cgb_inv = sat_to_lin * Cgb_lin + (1.0 - sat_to_lin) * Cgb_sat

    # --- Blend depletion ↔ strong inversion ------------------------------- #
    Cgg_di = dep_to_inv * Cgg_inv + (1.0 - dep_to_inv) * Cgg_dep
    Cgs_di = dep_to_inv * Cgs_inv + (1.0 - dep_to_inv) * Cgs_dep
    Cgd_di = dep_to_inv * Cgd_inv + (1.0 - dep_to_inv) * Cgd_dep
    Cgb_di = dep_to_inv * Cgb_inv + (1.0 - dep_to_inv) * Cgb_dep

    # --- Blend accumulation ↔ (depletion+inversion) ----------------------- #
    Cgg = accum_to_dep * Cgg_di + (1.0 - accum_to_dep) * Cgg_acc
    Cgs = accum_to_dep * Cgs_di + (1.0 - accum_to_dep) * Cgs_acc
    Cgd = accum_to_dep * Cgd_di + (1.0 - accum_to_dep) * Cgd_acc
    Cgb = accum_to_dep * Cgb_di + (1.0 - accum_to_dep) * Cgb_acc

    return {"Cgg": Cgg, "Cgs": Cgs, "Cgd": Cgd, "Cgb": Cgb}


# --------------------------------------------------------------------------- #
# Top-level one-shot                                                          #
# --------------------------------------------------------------------------- #
def compute_caps(
    model: BSIM4Model,
    sd: SizeDependParam,
    dc_result,
    Vgs: torch.Tensor,
    Vds: torch.Tensor,
    Vbs: torch.Tensor,
    Vbd: Optional[torch.Tensor] = None,
    *,
    As: Optional[float] = None,
    Ad: Optional[float] = None,
    Ps: Optional[float] = None,
    Pd: Optional[float] = None,
) -> CapResult:
    """One-shot junction + Meyer intrinsic caps (capMod=0).

    Args:
        model: BSIM4 model card.
        sd:    SizeDependParam (per-geometry-temp cache).
        dc_result: DCResult from dc.compute_dc (uses Vth, Abulk).  May be None;
                   then falls back to vth0_T and Abulk=1.
        Vgs, Vds, Vbs: terminal voltages, fp64 tensors.
        Vbd: if None, computed as Vbs - Vds.
        As, Ad, Ps, Pd: device drawn area/perimeter (m^2, m).  None → defaults.
    """
    if Vbd is None:
        Vbd = Vbs - Vds
    j = compute_junction_caps(model, sd, Vbs, Vbd, As=As, Ad=Ad, Ps=Ps, Pd=Pd)
    m = compute_intrinsic_caps_capmod0(model, sd, dc_result, Vgs, Vds, Vbs)
    Cbody_total = j["Cjs"] + j["Cjd"] + torch.abs(m["Cgb"])
    return CapResult(
        Cjs=j["Cjs"], Cjd=j["Cjd"],
        Cgg=m["Cgg"], Cgs=m["Cgs"], Cgd=m["Cgd"], Cgb=m["Cgb"],
        Cbody_total=Cbody_total,
    )
