"""bsim4_port.diode — Source/drain body junction diodes (manual §11.1).

Faithful differentiable port of BSIM4 v4.8.3 body diode currents Ibs and Ibd.
These currents flow body↔source (Ibs) and body↔drain (Ibd) and enter the
body-node KCL with positive sign on the body side.

Source reference:
  - b4ld.c §654-852  : SourceSatCurrent / DrainSatCurrent + dioMod=0/1/2

We implement dioMod=1 (smooth-clamped exponential without breakdown), the
default for compact models.  The breakdown branch (xjbvs/bvs) is left as
TODO — relevant only at Vbs ≲ -BVS ≈ -10V which never occurs in NS-RAM ops.
TODO(reverse-bv): port dioMod=2 reverse-breakdown branch when needed.
TODO(TAT): trap-assisted tunneling not yet ported.

Components (manual §11.1):
    Is_total = As · Js  +  Ps · Jsws  +  Weff_CJ·NF · Jswgs        (source)
    Id_total = Ad · Jd  +  Pd · Jswd  +  Weff_CJ·NF · Jswgd        (drain)
    Ibs = Is_total · (exp(Vbs/(Nj·vt)) - 1)
    Ibd = Id_total · (exp(Vbd/(Nj·vt)) - 1)

Forward-bias safe; reverse bias clamps the exponent at -EXP_THRESHOLD via
safe_exp so the (exp - 1) saturates at exactly -1 (= reverse-bias saturation
current).
"""
from __future__ import annotations

import torch

from .model_card import BSIM4Model
from .smooth import safe_exp
from .temp import SizeDependParam


def compute_body_diodes(
    model: BSIM4Model,
    sd: SizeDependParam,
    Vbs: torch.Tensor | float,
    Vbd: torch.Tensor | float,
    *,
    As: float = 0.0,        # source bottom area [m²]
    Ad: float = 0.0,        # drain bottom area  [m²]
    Ps: float = 0.0,        # source perimeter (isolation edge) [m]
    Pd: float = 0.0,        # drain perimeter (isolation edge)  [m]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Body diode currents (Ibs, Ibd).  b4ld.c §654-852, manual §11.1.

    Geometry inputs (areas/perimeters) default to zero; in that case the
    currents reduce to the gate-edge component (weffCJ·NF·Jswg*).  For
    NS-RAM-style compact cells, callers should pass realistic As/Ad/Ps/Pd.

    The "dioMod=1 smooth" form is used: pure exp(V/(Nj·vt))-1, saturating at
    -A·Js for Vbs ≪ 0 via safe_exp clamp.

    NOTE: nj defaults — model_card defaults njs/njd to 0 (BSIM-style
    "givenness" flag).  When 0, the C code falls back to nj=1.0; we mirror
    that here.  Likewise jss/jsd default to 0; if all junction densities are
    zero we return zero current with `gmin·V` form (skipped here — caller is
    expected to add device-level gmin if needed).
    """
    # Coerce
    Vbs_t = torch.as_tensor(Vbs, dtype=torch.float64)
    Vbd_t = torch.as_tensor(Vbd, dtype=torch.float64)
    Vbs_b, Vbd_b = torch.broadcast_tensors(Vbs_t, Vbd_t)

    # ---- Temp-shifted current densities (from sd) ------------------------ #
    # GRADFIX: drop float() so injected tensor jss/jsd via sd.SourceSatCurDensity_T
    # propagate gradients. Floats still pass through unchanged.
    Js_s = sd.SourceSatCurDensity_T                    # [A/m²]
    Js_d = sd.DrainSatCurDensity_T                     # [A/m²]
    # Sidewall + gate-edge densities: temp-shift uses same xtis/xtid factor
    # as Js.  For first cut we treat them as zero unless the card supplies
    # them; in that case apply the same TRatio factor as Js.
    TRatio = float(sd.model_ctx.TRatio)
    xtis = float(model.get("xtis", 0.0))
    xtid = float(model.get("xtid", 0.0))
    # Junction emission factors  (model_card defaults give 0 ⇒ fall back to 1)
    nj_s_card = float(model.get("njs", 0.0))
    nj_s = nj_s_card if nj_s_card > 0 else 1.0
    nj_d_card = float(model.get("njd", 0.0))
    nj_d = nj_d_card if nj_d_card > 0 else 1.0

    # Sidewall densities (no TAT, simple Tratio^xti) — order-of-mag accurate.
    def _scale(jname: str, xti: float) -> float:
        j0 = float(model.get(jname, 0.0))
        if j0 == 0.0:
            return 0.0
        return j0 * (TRatio ** xti)

    jsws = _scale("jsws", xtis)
    jswgs = _scale("jswgs", xtis)
    jswd = _scale("jswd", xtid)
    jswgd = _scale("jswgd", xtid)

    # weffCJ × NF (use NF=1 if missing — Geometry stores it on geom)
    NF = float(getattr(sd.geom, "NF", 1)) if hasattr(sd.geom, "NF") else 1.0
    weffCJ = float(sd.geom.weffCJ)

    # ---- Saturation current per junction --------------------------------- #
    SourceSatI = As * Js_s + Ps * jsws + weffCJ * NF * jswgs
    DrainSatI = Ad * Js_d + Pd * jswd + weffCJ * NF * jswgd

    # Vt for diode emission
    vtm = float(sd.model_ctx.vtm)
    Nvtms = vtm * nj_s
    Nvtmd = vtm * nj_d

    # ---- Ibs ------------------------------------------------------------- #
    # safe_exp clamps the argument to ±EXP_THRESHOLD; at deep reverse bias,
    # safe_exp(Vbs/Nvtms) ≈ exp(-34) so (exp - 1) ≈ -1 exactly.
    if SourceSatI <= 0.0:
        Ibs = torch.zeros_like(Vbs_b)
    else:
        evbs = safe_exp(Vbs_b / Nvtms)
        Ibs = SourceSatI * (evbs - 1.0)

    # ---- Ibd ------------------------------------------------------------- #
    if DrainSatI <= 0.0:
        Ibd = torch.zeros_like(Vbd_b)
    else:
        evbd = safe_exp(Vbd_b / Nvtmd)
        Ibd = DrainSatI * (evbd - 1.0)

    return Ibs, Ibd
