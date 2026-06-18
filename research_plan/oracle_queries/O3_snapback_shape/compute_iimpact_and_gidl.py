"""bsim4_port.leak — Body-current leakage models (impact-ion, GIDL/GISL, Igb).

Faithful differentiable port of BSIM4 v4.8.3 sub-models that contribute to
the body-node KCL right-hand side. Critical for NS-RAM since the floating
bulk integrates these currents.

Source references:
  - b4ld.c §2047-2086  : Iii (impact-ionization, manual §6.1)
  - b4ld.c §2274-2370  : Igidl/Igisl pre-4.7 model (manual §6.2, gidlMod=0)
  - b4ld.c §2493-2538  : Voxacc / Voxdepinv setup
  - b4ld.c §2769-2882  : Igb = Igbacc + Igbinv (manual §4.3.1)

We DO NOT port Igc / Igs / Igd (gate-channel, gate-source, gate-drain) here
because those currents do NOT enter the body-node KCL. They flow gate→channel
ends.  TODO: port to a separate leak_gate.py if ever needed.

All exp(...) calls go through smooth.safe_exp (clamped at ±34) per project rule.
All denominators get clamp_min(epsilon) per project rule.
fp64 throughout.
"""
from __future__ import annotations

import torch

from .constants import DELTA_3, EPS0, EXP_THRESHOLD, MIN_EXP
from .model_card import BSIM4Model
from .smooth import safe_exp, safe_sqrt, smooth_max
from .temp import SizeDependParam


_DENOM_EPS = 1e-30        # absolute floor for /denominators
_GIDL_T2_CAP = 100.0      # b4ld.c "if (T2 < 100.0)" branch → use exp(-T2)
_TOX_FLOOR = 1.0e-12      # floor for toxe (1 pm)


def _t(x, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


# --------------------------------------------------------------------------- #
# Impact-ionization                                                           #
# --------------------------------------------------------------------------- #

def compute_iimpact(
    model: BSIM4Model,
    sd: SizeDependParam,
    dc_result,
    Vds: torch.Tensor | float,
) -> torch.Tensor:
    """Substrate (impact-ionization) current Isub. b4ld.c §2047-2086.

    Equation (manual §6.1):
        T2  = (alpha0 + alpha1·Leff) / Leff
        if (Vds-Vdseff) > beta0/EXP_THRESHOLD:
            T1 = T2·(Vds-Vdseff)·exp(-beta0/(Vds-Vdseff))
        else:
            T1 = T2·MIN_EXP·(Vds-Vdseff)            # tiny linear floor
        Iii = T1 · Idsa · Vdseff

    WAVE2-FIX-1 (Gap 2, b4ld.c §2069 vs §2089-2091):
      Iii uses the *pre-SCBE* ``Idsa·Vdseff`` quantity (``dc_result.Idsa``),
      NOT the post-SCBE ``dc_result.Ids``. SCBE is applied to Ids only AFTER
      Iii is computed in the C source. Using Ids inflates Iii by the SCBE
      factor (typically 1.0-1.3, i.e. up to ~30% rel err in saturation).
      Falls back to dc_result.Ids if Idsa is missing (legacy compat).
    """
    P = sd.scaled
    leff = float(sd.geom.leff)
    alpha0 = P.get("alpha0", 0.0)
    alpha1 = P.get("alpha1", 0.0)
    beta0 = P.get("beta0", 0.0)

    Vds_t = torch.as_tensor(Vds, dtype=dc_result.Ids.dtype, device=dc_result.Ids.device)
    Vdseff = dc_result.Vdseff
    Vds_b, Vdseff_b = torch.broadcast_tensors(Vds_t, Vdseff)
    diffVds = Vds_b - Vdseff_b
    # Guard against negative diffVds (shouldn't occur for Vds>0 NMOS) — clamp.
    diffVds = diffVds.clamp_min(0.0)

    tmp = alpha0 + alpha1 * leff
    if (tmp <= 0.0) or (beta0 <= 0.0):
        # Card disables impact-ion.
        return torch.zeros_like(diffVds)

    T2 = tmp / leff
    threshold = beta0 / EXP_THRESHOLD                       # b4ld.c branch
    diff_safe = diffVds.clamp_min(_DENOM_EPS)
    # Strong-bias arm: T1 = T2·diff·exp(-beta0/diff)
    T0 = -beta0 / diff_safe
    T1_strong = T2 * diff_safe * safe_exp(T0)
    # Weak-bias arm: T1 = T2·MIN_EXP·diff
    T1_weak = T2 * MIN_EXP * diff_safe
    T1 = torch.where(diffVds > threshold, T1_strong, T1_weak)

    # WAVE2-FIX-1 (Gap 2): use pre-SCBE Idsa·Vdseff (b4ld.c §2069). Fallback to
    # Ids preserves backward compatibility with any legacy DCResult lacking Idsa.
    Idsa_Vdseff = getattr(dc_result, "Idsa", None)
    if Idsa_Vdseff is None:
        Idsa_Vdseff = dc_result.Ids
    Iii = T1 * Idsa_Vdseff
    return Iii


# --------------------------------------------------------------------------- #
# GIDL / GISL                                                                 #
# --------------------------------------------------------------------------- #

def _gidl_one_side(
    *,
    weffCJ: torch.Tensor,
    toxe: torch.Tensor,
    a: float, b: float, c: float, e: float,
    V_drive: torch.Tensor,                     # Vd-Vg-egidl (or -Vd-Vg-egisl)
    Vbody: torch.Tensor,                       # vbd (GIDL) or vbs (GISL)
    body_disable: torch.Tensor,                # bool: vbd>0 (GIDL) / vbs>0 (GISL)
) -> torch.Tensor:
    """Shared kernel for GIDL/GISL. b4ld.c §2295-2324.

    Igidl = a · Weff_CJ · T1 · exp(-b/T1) · Vbody³/(Vbody³ + c)   (b4ld.c form)
    where T1 = (Vd-Vg-egidl)/(3·toxe) (already passed in via V_drive/(3·tox)).
    """
    # T0 = 3·toxe (b4ld.c §2277)
    T0 = 3.0 * toxe.clamp_min(_TOX_FLOOR)
    T1 = V_drive / T0                                       # may be ≤0
    # Disable conditions: a≤0, b≤0, c≤0, T1≤0, body_disable.
    if a <= 0.0 or b <= 0.0 or c <= 0.0:
        return torch.zeros_like(V_drive)

    T1_safe = T1.clamp_min(_DENOM_EPS)
    T2 = b / T1_safe                                        # may be huge
    # b4ld.c branches T2<100 vs ≥100; we use a single safe_exp(-T2) which
    # automatically saturates at exp(-34)≈MIN_EXP.  Mirrors the saturated
    # branch but smoothly differentiable.
    Igidl_pre = a * weffCJ * T1_safe * safe_exp(-T2)        # cylinder

    # Body-bias factor: Vbody³/(Vbody³ + c)  (b4ld.c §2315-2323)
    # b4ld.c writes T4=v*v, T5=-v*T4, T6=c+T5, T7=T5/T6  →  T7 = -v³/(c-v³)
    # Inspecting: GIDL conducts when vbd<0 (drain reverse-biased to body), so
    # T5 = -vbd·vbd² > 0 ⇒ T7 ∈ (0,1).  We replicate the C exactly.
    T4 = Vbody * Vbody
    T5 = -Vbody * T4                                        # -v³
    T6 = c + T5                                             # could be near 0
    T6_safe = T6.clamp_min(_DENOM_EPS)
    T7 = T5 / T6_safe                                       # body factor
    Igidl = Igidl_pre * T7

    # Hard zero where T1≤0 or body_disable (matches b4ld.c if-branch).
    valid = (T1 > 0.0) & (~body_disable)
    Igidl = torch.where(valid, Igidl, torch.zeros_like(Igidl))
    return Igidl


def compute_igidl_gisl(
    model: BSIM4Model,
    sd: SizeDependParam,
    Vgs: torch.Tensor,
    Vds: torch.Tensor,
    Vbs: torch.Tensor | float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (Igidl, Igisl).  b4ld.c §2274-2370.  Manual §6.2 (gidlMod=0)."""
    if int(model.get("gidlmod", 0)) != 0:
        # gidlMod=1 form not ported yet.
        raise NotImplementedError("Only gidlMod=0 (pre-4.7) GIDL/GISL ported.")

    P = sd.scaled
    weffCJ = torch.as_tensor(sd.geom.weffCJ, dtype=Vgs.dtype, device=Vgs.device)
    toxe = torch.as_tensor(sd.model_ctx.toxe, dtype=Vgs.dtype, device=Vgs.device)

    Vgs_b, Vds_b, Vbs_b = torch.broadcast_tensors(
        torch.as_tensor(Vgs, dtype=torch.float64),
        torch.as_tensor(Vds, dtype=torch.float64),
        torch.as_tensor(Vbs, dtype=torch.float64),
    )
    # vgs_eff ≈ Vgs (we don't carry vfbsd correction; b4ld.c uses BSIM4vgs_eff
    # which is bias-clamped; for body-KCL accuracy this is sufficient).
    vgs_eff = Vgs_b
    vgd_eff = Vgs_b - Vds_b
    vbd = Vbs_b - Vds_b
    vbs = Vbs_b

    # GRADFIX: drop float() cast so gradcheck can propagate gradients when the
    # caller injects parameter tensors into sd.scaled. Floats still pass-through
    # unchanged; tensors flow through subsequent arithmetic.
    egidl = P.get("egidl", model.get("egidl", 0.8))
    egisl = P.get("egisl", model.get("egisl", egidl))
    agidl = P.get("agidl", 0.0)
    bgidl = P.get("bgidl", 0.0)
    cgidl = P.get("cgidl", model.get("cgidl", 0.5))
    agisl = P.get("agisl", agidl)
    bgisl = P.get("bgisl", bgidl)
    cgisl = P.get("cgisl", cgidl)

    # GIDL drive: Vd-Vg-egidl  (mtrlMod=0)
    V_drive_d = Vds_b - vgs_eff - egidl
    Igidl = _gidl_one_side(
        weffCJ=weffCJ, toxe=toxe,
        a=agidl, b=bgidl, c=cgidl, e=egidl,
        V_drive=V_drive_d, Vbody=vbd,
        body_disable=(vbd > 0.0),
    )
    # GISL drive: -Vd-Vg-egisl   (vgd_eff = Vg-Vd  ⇒  -Vd-Vg = -(Vd+Vg) ≠ -vgd_eff!)
    # b4ld.c §2331: T1 = (-vds - vgd_eff - egisl)/T0
    V_drive_s = -Vds_b - vgd_eff - egisl
    Igisl = _gidl_one_side(
        weffCJ=weffCJ, toxe=toxe,
        a=agisl, b=bgisl, c=cgisl, e=egisl,
        V_drive=V_drive_s, Vbody=vbs,
        body_disable=(vbs > 0.0),
    )
    return Igidl, Igisl


# --------------------------------------------------------------------------- #
# Vfbeff / Voxacc / Voxdepinv  (Wave-2 Gap 1 + Gap 7)                          #
# --------------------------------------------------------------------------- #

def compute_vfbeff(
    Vgs_eff: torch.Tensor,
    Vbseff: torch.Tensor,
    vfb: float,
) -> torch.Tensor:
    """Smooth flat-band voltage Vfbeff per b4ld.c §2496-2504.

    Faithful port of the BSIM4 v4.8.3 form:

        V3     = Vfb - Vgs_eff + Vbseff - DELTA_3
        T0     = sqrt(V3**2 +/- 4*DELTA_3*Vfb)        (+ if Vfb>0 else -)
        Vfbeff = Vfb - 0.5*(V3 + T0)

    The branch on `Vfb<=0` is on a per-device scalar (vfbzb / vfb is bias-independent),
    so a Python `if` is fine -- no torch.where needed. We wrap `sqrt` in `safe_sqrt`
    for numerical safety; in the well-defined regime the discriminant is structurally
    non-negative.

    Used by:
      - leak.compute_igb (Voxacc / Voxdepinv) -- Gap 7
      - caps_capmod2 (CTM charge model)        -- future Gap 3
    """
    V3 = (vfb - Vgs_eff + Vbseff - DELTA_3)
    if vfb <= 0.0:
        T0 = safe_sqrt(V3 * V3 - 4.0 * DELTA_3 * vfb)   # b4ld.c §2498
    else:
        T0 = safe_sqrt(V3 * V3 + 4.0 * DELTA_3 * vfb)   # b4ld.c §2500
    Vfbeff = vfb - 0.5 * (V3 + T0)                       # b4ld.c §2502
    return Vfbeff


def compute_voxacc_voxdepinv(
    Vgs_eff: torch.Tensor,
    Vbseff: torch.Tensor,
    Vgsteff: torch.Tensor,
    vfb: float,
    k1ox: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Voxacc / Voxdepinv per b4ld.c §2506-2537.

        Voxacc    = max(0, Vfb - Vfbeff)               (line 2506-2510)
        T3        = Vgs_eff - Vfbeff - Vbseff - Vgsteff
        if k1ox == 0:        Voxdepinv0 = 0
        elif T3 < 0:         Voxdepinv0 = -T3
        else:                Voxdepinv0 = k1ox*(sqrt((k1ox/2)**2 + T3) - k1ox/2)
        Voxdepinv = Voxdepinv0 + Vgsteff               (line 2534)

    Smooth-replacement strategy (per spec):
      - `if (Voxacc < 0)` clamp -> smooth_max(Voxacc, 0).
      - `if k1ox == 0`           -> Python if on scalar (param-time branch).
      - `if (T3 < 0)` two-arm    -> torch.where; both arms finite via safe_sqrt.
    """
    Vfbeff = compute_vfbeff(Vgs_eff, Vbseff, vfb)

    # Voxacc -- accumulation oxide voltage, clamped at 0 (b4ld.c §2506-2510).
    Voxacc_raw = vfb - Vfbeff
    zero = torch.zeros_like(Voxacc_raw)
    Voxacc = smooth_max(Voxacc_raw, zero)

    # Voxdepinv -- depletion+inversion, b4ld.c §2512-2537.
    T3 = Vgs_eff - Vfbeff - Vbseff - Vgsteff
    if k1ox == 0.0:
        Voxdepinv0 = torch.zeros_like(T3)
    else:
        T0 = 0.5 * k1ox
        # T3<0 arm: -T3                                              (b4ld.c §2518)
        # T3>=0 arm: k1ox*(sqrt(T0**2 + T3) - T0)                    (b4ld.c §2525-2527)
        sqrt_arm = k1ox * (safe_sqrt(T0 * T0 + T3) - T0)
        neg_arm = -T3
        Voxdepinv0 = torch.where(T3 < 0.0, neg_arm, sqrt_arm)

    Voxdepinv = Voxdepinv0 + Vgsteff                    # b4ld.c §2534
    return Voxacc, Voxdepinv


# --------------------------------------------------------------------------- #
# Gate-to-body tunneling                                                       #
# --------------------------------------------------------------------------- #

def _igb_branch(
    *,
    weff: torch.Tensor, leff: torch.Tensor, ToxRatio: torch.Tensor,
    Vgs: torch.Tensor, Vbs: torch.Tensor,
    Vaux_input: torch.Tensor, n: float, Vt: torch.Tensor,
    aigb: float, bigb: float, cigb: float,
    Vox: torch.Tensor,
    T11_prefactor: float, T12_factor: float,
) -> torch.Tensor:
    """Common Igbacc / Igbinv kernel. b4ld.c §2769-2876.

      T0  = Vt·n
      Vaux = T0·log(1+exp(Vaux_input/T0))      (smooth softplus)
      T2  = (Vgs-Vbs)·Vaux
      T11 = T11_prefactor · weff · leff · ToxRatio
      T12 = T12_factor · toxe                  (passed implicitly via call site)
      T5  = T12·(aigb + (aigb·cigb - bigb)·Vox - bigb·cigb·Vox²)
      T6  = exp(T5)                             (clamped)
      Igb = T11 · T2 · T6
    """
    T0 = (Vt * n).clamp_min(_DENOM_EPS)
    # Smooth softplus: Vaux = T0 · log(1 + exp(VxNVt))   with VxNVt = T1/T0
    # safe_exp guarantees clamp ±34, log1p stays finite.
    VxNVt = Vaux_input / T0
    Vaux = T0 * torch.log1p(safe_exp(VxNVt))

    T2 = (Vgs - Vbs) * Vaux

    T3 = aigb * cigb - bigb
    T4 = bigb * cigb
    # T12 already includes ·toxe (sign + magnitude); see compute_igb caller.
    T5 = T12_factor * (aigb + T3 * Vox - T4 * Vox * Vox)
    T6 = safe_exp(T5)

    Igb = T11_prefactor * weff * leff * ToxRatio * T2 * T6
    return Igb


def compute_igb(
    model: BSIM4Model,
    sd: SizeDependParam,
    Vgs: torch.Tensor,
    Vbs: torch.Tensor | float = 0.0,
    *,
    dc_result=None,
) -> torch.Tensor:
    """Gate-to-body tunneling Igb = Igbacc + Igbinv.  b4ld.c §2769-2882.

    Voxacc (b4ld.c §2506-2510) and Voxdepinv (b4ld.c §2511-2537) are computed
    via the faithful Vfbeff form (Wave-2 Gap 7) when `dc_result` is supplied
    (it provides the poly-dep'd Vgs_eff, Vbseff, Vgsteff intermediates).

    Backward-compatibility: when `dc_result` is None, fall back to the
    simplified softplus form (max(0, Vfb-Vgs+Vbs) / max(0, Vgs-Vfb-Vbs)). This
    preserves existing call sites (tests) but loses the Vgsteff coupling and
    k1ox-weighted sqrt bridge -- callers that need fidelity (e.g. nsram_cell)
    should pass `dc_result`.
    """
    if int(model.get("igbmod", 0)) == 0:
        return torch.zeros_like(torch.as_tensor(Vgs, dtype=torch.float64))

    P = sd.scaled
    Vgs_b, Vbs_b = torch.broadcast_tensors(
        torch.as_tensor(Vgs, dtype=torch.float64),
        torch.as_tensor(Vbs, dtype=torch.float64),
    )
    weff = torch.as_tensor(sd.geom.weff, dtype=Vgs_b.dtype, device=Vgs_b.device)
    leff = torch.as_tensor(sd.geom.leff, dtype=Vgs_b.dtype, device=Vgs_b.device)
    toxe = torch.as_tensor(sd.model_ctx.toxe, dtype=Vgs_b.dtype, device=Vgs_b.device)

    # ToxRatio = (toxref/toxe)^ntox · (1/toxe²)  per b4temp.c §1377-1379
    # NOTE: BSIM4ToxRatio in C includes a 1/toxe^2 factor in some forms but
    # b4temp.c stores the bare exp(ntox·log(toxref/toxe)) form; b4ld.c then
    # multiplies T11 (which includes 4.97e-7) by ToxRatio.  We follow b4temp.c.
    # GRADFIX: tensor-pass-through for fitable params.
    def _t(x):
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(
            x, dtype=Vgs_b.dtype, device=Vgs_b.device)

    toxref = _t(model.get("toxref", 3.0e-9))
    ntox = _t(P.get("ntox", model.get("ntox", 1.0)))
    toxe_safe = torch.clamp(toxe, min=_TOX_FLOOR)
    ToxRatio = (toxref / toxe_safe) ** ntox / (toxe_safe * toxe_safe)

    Vfb = _t(model.get("vfb", -1.0))
    Vt = torch.as_tensor(sd.model_ctx.vtm, dtype=Vgs_b.dtype, device=Vgs_b.device)

    # ------------- Igbacc (b4ld.c §2769-2820) ------------- #
    # GRADFIX: keep tensors when injected via sd.scaled override.
    nigbacc = P.get("nigbacc", model.get("nigbacc", 1.0))
    aigbacc = P.get("aigbacc", model.get("aigbacc", 0.0136))
    bigbacc = P.get("bigbacc", model.get("bigbacc", 0.00171))
    cigbacc = P.get("cigbacc", model.get("cigbacc", 0.075))

    # WAVE2-FIX (Gap 7): Voxacc / Voxdepinv via faithful Vfbeff machinery
    # (b4ld.c §2493-2537) when dc_result is supplied.  Falls back to simplified
    # softplus form otherwise (preserves backward compat for existing tests).
    if dc_result is not None and dc_result.Vgs_eff is not None and dc_result.Vbseff is not None:
        # Broadcast dc intermediates against Vgs_b/Vbs_b shape.
        Vgs_eff_b = dc_result.Vgs_eff
        Vbseff_b = dc_result.Vbseff
        Vgsteff_b = dc_result.Vgsteff
        # vfb scalar from card; k1ox from sd (size-dep'd).  These match the C
        # `Vfb = vfbzb` / `pParam->BSIM4k1ox` reads at lines 2495 / 2512.
        # vfbzb proper would include vth0 and the k1·sqrtPhi shift (b4temp.c §1586,
        # §1805) -- using card.vfb is a known approximation, transitively validated
        # via the ngspice Igb diff once vfbzb is plumbed through (separate gap).
        vfb_scalar = float(model.get("vfb", -1.0))
        k1ox_scalar = float(P.get("k1ox", sd.k1ox))
        Voxacc, Voxdepinv = compute_voxacc_voxdepinv(
            Vgs_eff_b, Vbseff_b, Vgsteff_b, vfb_scalar, k1ox_scalar,
        )
    else:
        # SIMPLIFIED FALLBACK -- softplus·50.  No Vgsteff coupling, no k1ox sqrt.
        raw_Vacc = Vfb - Vgs_b + Vbs_b
        Voxacc = torch.nn.functional.softplus(raw_Vacc * 50.0) / 50.0
        raw_Vinv = Vgs_b - Vfb - Vbs_b
        Voxdepinv = torch.nn.functional.softplus(raw_Vinv * 50.0) / 50.0

    Vaux_input_acc = -Vgs_b + Vbs_b + Vfb                # T1 = -Vgs+Vbs+Vfb

    Igbacc = _igb_branch(
        weff=weff, leff=leff, ToxRatio=ToxRatio,
        Vgs=Vgs_b, Vbs=Vbs_b,
        Vaux_input=Vaux_input_acc, n=nigbacc, Vt=Vt,
        aigb=aigbacc, bigb=bigbacc, cigb=cigbacc,
        Vox=Voxacc,
        T11_prefactor=4.97232e-7,
        T12_factor=-7.45669e11 * toxe,
    )

    # ------------- Igbinv (b4ld.c §2822-2876) ------------- #
    # GRADFIX: tensor-pass-through.
    nigbinv = _t(P.get("nigbinv", model.get("nigbinv", 3.0)))
    aigbinv = _t(P.get("aigbinv", model.get("aigbinv", 0.0111)))
    bigbinv = _t(P.get("bigbinv", model.get("bigbinv", 0.000949)))
    cigbinv = _t(P.get("cigbinv", model.get("cigbinv", 0.006)))
    eigbinv = _t(P.get("eigbinv", model.get("eigbinv", 1.1)))

    # Voxdepinv already computed above (faithful path or fallback).
    Vaux_input_inv = Voxdepinv - eigbinv

    # b4ld.c §2849-2850: T11 *= 0.75610; T12 *= 1.31724
    Igbinv = _igb_branch(
        weff=weff, leff=leff, ToxRatio=ToxRatio,
        Vgs=Vgs_b, Vbs=Vbs_b,
        Vaux_input=Vaux_input_inv, n=nigbinv, Vt=Vt,
        aigb=aigbinv, bigb=bigbinv, cigb=cigbinv,
        Vox=Voxdepinv,
        T11_prefactor=4.97232e-7 * 0.75610,
        T12_factor=-7.45669e11 * toxe * 1.31724,
    )

    return Igbacc + Igbinv
