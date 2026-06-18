"""B07-B20 — Faithful BSIM4 4.8.3 DC drain-current port.

Ported from `external/bsim4/code/b4ld.c` lines 1002-2156.
Equation cross-references in section comments below as `# b4ld.c §<lines>`.

Scope (P3, first faithful pass):
  - mobMod = 1 path only
  - rdsMod ∈ {0, 1}: 0 = internal bias-dependent Rds (DEFAULT in BSIM4 per
    b4set.c:107-108; reduces Idsat & modifies Vdsat); 1 = external resistor.
  - mtrlMod = 0 (Si substrate)
  - tempMod = 0 (default)
  - No velocity overshoot (lambda branch skipped)
  - No source-end vtl limit (Fsevl skipped)
  - No quantum/bulk-charge centroid (Tcen, Coxeff = coxe directly)
  - No poly depletion Newton (Vgs_eff = Vgs)
  - No Weff_corr Newton (use sd.geom.weff directly)

Out of scope here (P3.5/P4):
  - Charge model (capMod), gate tunneling, GIDL/GISL, impact-ion, body diodes,
  - AC analysis, noise, NQS

Differentiability rules:
  - fp64 throughout.
  - All if-branches on tensors replaced with `torch.where` over both
    differentiable arms, OR substituted with smooth.py primitives.
  - All `exp` arguments are guarded via `safe_exp` (clipped at ±34) — matches
    BSIM4 DEXP MIN_EXP/MAX_EXP regularizer.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import torch

from .constants import Charge_q, EPSSI, EPS0, KboQ, MAX_EXP, MIN_EXP, EXP_THRESHOLD, PI
from .geometry import Geometry
from .model_card import BSIM4Model
from .smooth import safe_exp, safe_log, safe_sqrt, smooth_min, smooth_max
from .temp import SizeDependParam, compute_size_dep

import math as _math
# Pre-computed BSIM4 deep-subthreshold floor: log(1 + MIN_EXP) ≈ MIN_EXP.
_LOG1P_MIN_EXP = _math.log1p(MIN_EXP)


@dataclass
class DCResult:
    Ids: torch.Tensor       # drain current [A]  (positive for NMOS Vds>0)
    Vth: torch.Tensor       # threshold incl. all corrections used
    Vgsteff: torch.Tensor   # effective overdrive (smooth)
    Vdsat: torch.Tensor     # saturation drain voltage
    Vdseff: torch.Tensor    # effective Vds (smooth-min Vds, Vdsat)
    Abulk: torch.Tensor     # bulk-charge factor
    n: torch.Tensor         # subthreshold ideality
    mueff: torch.Tensor     # effective mobility
    Rds: Optional[torch.Tensor] = None   # internal Rds (rdsmod=0); None for rdsmod=1
    # WAVE2-FIX-1 (Gap 2): pre-SCBE channel current "T4 = Idsa·Vdseff" from b4ld.c §2069.
    # Iii (impact ionization) in leak.py must use this — NOT post-SCBE Ids — to match
    # the C-source impact-ionization formula faithfully.
    Idsa: Optional[torch.Tensor] = None
    # WAVE2-FIX (Gap 7): intermediates needed by leak.compute_igb (Vfbeff path).
    Vgs_eff: Optional[torch.Tensor] = None  # poly-dep'd Vgs (b4ld.c §1224-1296)
    Vbseff: Optional[torch.Tensor] = None   # body-bias clamp (b4ld.c §1002-1019)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _t(x, dtype=torch.float64, device=None) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


def _exp_threshold_branch(T0: torch.Tensor, *, ratio: float = 1.0) -> torch.Tensor:
    """BSIM4 Theta0-style stable rational form:
        if T0 < EXP_THRESHOLD: Theta0 = exp(T0) / ((exp(T0)-1)^2 + 2 exp(T0) MIN_EXP)
        else:                  Theta0 = 1 / (MAX_EXP - 2)
    Implemented faithfully via torch.where; both branches finite & differentiable.
    """
    # SMOOTH: replace if-on-tensor with torch.where over both arms; both finite.
    T1 = safe_exp(T0)                       # exp(min(T0, 34))  ⇒ never overflow
    T2 = T1 - 1.0
    T3 = T2 * T2
    T4 = T3 + 2.0 * T1 * MIN_EXP
    inner = T1 / T4
    saturated = torch.full_like(T0, 1.0 / (MAX_EXP - 2.0))
    return torch.where(T0 < EXP_THRESHOLD, inner, saturated)


# --------------------------------------------------------------------------- #
# Main DC entry point                                                          #
# --------------------------------------------------------------------------- #

def compute_dc(
    model: BSIM4Model,
    sd: SizeDependParam,
    Vgs: torch.Tensor,
    Vds: torch.Tensor,
    Vbs: torch.Tensor | float = 0.0,
    *,
    sharpness: float = 50.0,
) -> DCResult:
    """Faithful differentiable BSIM4 DC drain current.

    Returns DCResult with Ids and intermediate quantities. All tensors fp64.
    Bias inputs may be scalar or broadcastable tensors.
    """
    # ---- tensor coercion ------------------------------------------------- #
    if isinstance(Vgs, torch.Tensor):
        dtype = Vgs.dtype
        device = Vgs.device
    else:
        dtype = torch.float64
        device = None
    if dtype != torch.float64:
        # Promote — fp64 throughout per project rules.
        dtype = torch.float64

    def t(x):
        return _t(x, dtype=dtype, device=device)

    Vgs = t(Vgs)
    Vds = t(Vds)
    Vbs = t(Vbs)
    # Broadcast to a common shape so torch.where always works.
    Vgs, Vds, Vbs = torch.broadcast_tensors(Vgs, Vds, Vbs)

    geom = sd.geom
    ctx = sd.model_ctx
    P = sd.scaled

    # ---- Geometry / oxide / temp constants (scalars promoted to tensors) - #
    Leff = t(geom.leff)
    Weff = t(geom.weff)        # NOTE: Weff_corr Newton skipped this pass
    toxe = t(ctx.toxe)
    coxe = t(ctx.coxe)
    Vtm = t(ctx.vtm)
    Vtm0 = t(ctx.Vtm0)
    factor1 = t(ctx.factor1)
    epssub = t(ctx.epssub)

    type_n = float(model._values.get("type", 1))

    # ---- Scaled per-instance params (scalar floats → tensors) ------------ #
    vth0 = t(sd.vth0_T)               # T-shifted vth0
    k1 = t(P["k1"])
    k2 = t(P["k2"])
    k3 = t(P.get("k3", model.get("k3", 80.0)))
    k3b = t(P.get("k3b", model.get("k3b", 0.0)))
    w0 = t(P.get("w0", model.get("w0", 2.5e-6)))
    dvt0 = t(P["dvt0"])
    dvt1 = t(P["dvt1"])
    dvt2 = t(P["dvt2"])
    dvt0w = t(P["dvt0w"])
    dvt1w = t(P["dvt1w"])
    dvt2w = t(P["dvt2w"])
    eta0 = t(P["eta0"])
    etab = t(P["etab"])
    nfactor = t(P["nfactor"])
    cdsc = t(P["cdsc"])
    cdscb = t(P["cdscb"])
    cdscd = t(P["cdscd"])
    cit = t(P["cit"])
    voff = t(P["voff"])
    voffl = t(model.get("voffl", 0.0))
    minv = t(model.get("minv", 0.0))
    a0 = t(P["a0"])
    ags = t(P["ags"])
    a1 = t(P["a1"])
    a2 = t(P["a2"])
    keta = t(P["keta"])
    # WAVE3-FIX (z214): b0/b1 also bin via lX/wX/pX (b4temp.c).  Read from
    # the scaled per-instance dict so any future binning coefficients flow.
    b0 = t(P.get("b0", model.get("b0", 0.0)))
    b1 = t(P.get("b1", model.get("b1", 0.0)))
    xj = t(model.get("xj", 1.5e-7))
    dwg = t(model.get("dwg", 0.0))
    dwb = t(model.get("dwb", 0.0))
    pclm = t(P["pclm"])
    pdiblc1 = t(P["pdiblc1"])
    pdiblc2 = t(P["pdiblc2"])
    pdiblb = t(P.get("pdiblcb", model.get("pdiblcb", model.get("pdiblb", 0.0))))
    drout = t(P["drout"])
    pscbe1 = t(P["pscbe1"])
    pscbe2 = t(P["pscbe2"])
    pvag = t(P["pvag"])
    delta = t(P["delta"])
    fprout = t(P["fprout"])
    pdits = t(P["pdits"])
    pditsd = t(P["pditsd"])
    pditsl = t(model.get("pditsl", 0.0))
    dvtp0 = t(P["dvtp0"])
    dvtp1 = t(P["dvtp1"])
    dvtp4 = t(P.get("dvtp4", 0.0))
    dvtp2factor = t(model.get("dvtp2factor", 0.0))
    kt1 = t(model.get("kt1", -0.11))
    kt1l = t(model.get("kt1l", 0.0))
    kt2 = t(model.get("kt2", 0.022))
    lpe0 = t(model.get("lpe0", 1.74e-7))
    lpeb = t(model.get("lpeb", 0.0))
    ua = t(P["ua"])
    ub = t(P["ub"])
    uc = t(P["uc"])
    ud = t(P.get("ud", 0.0))
    u0temp = t(sd.u0temp)
    vsattemp = t(sd.vsattemp)
    Xdep0 = t(sd.Xdep0)
    sqrtPhi_pre = t(sd.sqrtPhi)
    phi_pre = t(sd.phi)
    vbi = t(sd.vbi)
    vbsc = t(sd.vbsc)
    k1ox = t(sd.k1ox)
    k2ox = t(sd.k2ox)
    litl = t(sd.litl)

    # Derived params now cached in SizeDependParam (b4temp.c §1373-1427).
    mstar = t(sd.mstar)
    voffcbn = t(sd.voffcbn)
    cdep0 = t(sd.cdep0)
    # Tcen / Coxeff inputs (b4ld.c §1789-1805 capMod=2 path)
    vtfbphi2 = t(sd.vtfbphi2)
    coxp = t(sd.coxp)
    toxp = t(sd.toxp)
    ados = t(sd.ados)
    bdos = t(sd.bdos)
    # Poly depletion inputs (b4ld.c §5170-5202)
    ngate = float(model.get("ngate", 0.0))
    epsrox_v = float(ctx.epsrox)
    coxe_f = float(ctx.coxe)

    # ===================================================================== #
    # 1. Vbseff  — body bias smooth saturation + JX forward correction      #
    #    b4ld.c §1002-1019    BSIM4 manual §3.5 (Vbseff)                    #
    # ===================================================================== #
    # WAVE2-FIX-2 (Gap 5): Strict ngspice validation for Vbseff is currently
    # NOT POSSIBLE: ngspice 42 does not expose `@m1[vbseff]` (probed
    # 2026-04-29 with all common name variants — vbseff/Vbseff/VBSeff/
    # vbs_eff/Vbs_eff/VBSEFF — all returned `Error: no such vector`). The
    # quantity is internal to BSIM4 and not part of the documented OP saves.
    # TODO Wave-3: validate via gm/gmbs ratio inference, OR patch ngspice to
    # expose vbseff in BSIM4dev.c, OR migrate to a newer ngspice version.
    # Original C: smooth-saturate Vbs to vbsc from below (T0 = Vbs - vbsc - 0.001).
    # The C splits on `T0 >= 0` and uses two algebraic forms; both are themselves
    # smooth — they differ by a continuous algebraic identity. We port the form
    # exactly, but use torch.where to keep both branches differentiable.
    T0 = Vbs - vbsc - 0.001
    T1 = safe_sqrt(T0 * T0 - 0.004 * vbsc)            # SMOOTH: safe_sqrt for grad at 0
    Vbseff_a = vbsc + 0.5 * (T0 + T1)                 # branch T0 >= 0
    T2 = -0.002 / (T1 - T0)                           # branch T0 < 0
    Vbseff_b = vbsc * (1.0 + T2)
    # SMOOTH: replace `if (T0 >= 0.0)` with torch.where over both arms.
    Vbseff = torch.where(T0 >= 0.0, Vbseff_a, Vbseff_b)

    # JX: correction to forward body bias  (b4ld.c §1014-1019)
    T9 = 0.95 * phi_pre
    T0 = T9 - Vbseff - 0.001
    T1 = safe_sqrt(T0 * T0 + 0.004 * T9)
    Vbseff = T9 - 0.5 * (T0 + T1)

    # ===================================================================== #
    # 2. Phis, sqrtPhis, Xdep   — b4ld.c §1020-1027   manual §2.4           #
    # ===================================================================== #
    Phis = phi_pre - Vbseff
    sqrtPhis = safe_sqrt(Phis)                         # SMOOTH: safe_sqrt
    Xdep = Xdep0 * sqrtPhis / sqrtPhi_pre

    # ===================================================================== #
    # 3. Vth core with DVT machinery   b4ld.c §1033-1130   manual §2.4-§3.0 #
    # ===================================================================== #
    T3 = safe_sqrt(Xdep)                               # SMOOTH: safe_sqrt
    V0 = vbi - phi_pre

    # --- lt1 ------------------------------------------------------------- #
    # b4ld.c §1037-1048: rational regularizer for dvt2*Vbs near -0.5
    # WAVE2-FIX (critique 7): the inactive arm 1/(3+8·T0) is singular at
    # T0=-3/8 (= -0.375) which sits INSIDE the active region (T0 >= -0.5).
    # Guard the denominator with a sign-preserving floor so backward never
    # produces NaN through the unused arm.
    T0 = dvt2 * Vbseff
    T1_a = 1.0 + T0                                    # branch: T0 >= -0.5
    _denom_b = 3.0 + 8.0 * T0
    _denom_b_safe = torch.where(_denom_b.abs() > 1e-6, _denom_b,
                                 torch.full_like(_denom_b, -1e-6))
    T4_b = 1.0 / _denom_b_safe                          # branch: T0 < -0.5
    T1_b = (1.0 + 3.0 * T0) * T4_b
    # SMOOTH: torch.where; both arms finite via _denom_b_safe.
    T1 = torch.where(T0 >= -0.5, T1_a, T1_b)
    lt1 = factor1 * T3 * T1

    # --- ltw  (b4ld.c §1050-1061) ---------------------------------------- #
    T0w = dvt2w * Vbseff
    T1w_a = 1.0 + T0w
    _denom_bw = 3.0 + 8.0 * T0w
    _denom_bw_safe = torch.where(_denom_bw.abs() > 1e-6, _denom_bw,
                                  torch.full_like(_denom_bw, -1e-6))
    T4w_b = 1.0 / _denom_bw_safe
    T1w_b = (1.0 + 3.0 * T0w) * T4w_b
    T1w = torch.where(T0w >= -0.5, T1w_a, T1w_b)
    ltw = factor1 * T3 * T1w

    # --- Theta0   b4ld.c §1063-1076  (body of §3.0 Vth long-channel) ----- #
    # Faithful: includes BSIM4's MIN_EXP regularizer in T4 = T2² + 2 T1 MIN_EXP.
    T0_th = dvt1 * Leff / lt1.clamp_min(1e-30)
    Theta0 = _exp_threshold_branch(T0_th)
    Delt_vth = dvt0 * Theta0 * V0

    # --- T5 (narrow-W via dvt0w/dvt1w)  b4ld.c §1081-1097 ---------------- #
    T0_w = dvt1w * Weff * Leff / ltw.clamp_min(1e-30)
    T5 = _exp_threshold_branch(T0_w)
    T2_narrow = dvt0w * T5 * V0   # corresponds to "T2" in C, narrow-W Vth shift

    # --- Lpe / temp / k3 narrow-W  b4ld.c §1099-1124 --------------------- #
    TempRatio = ctx.Temp / ctx.Tnom - 1.0
    T0_lpe = safe_sqrt(1.0 + lpe0 / Leff)
    Tlpe1 = (k1ox * (T0_lpe - 1.0) * sqrtPhi_pre
             + (kt1 + kt1l / Leff + kt2 * Vbseff) * TempRatio)
    Vth_NarrowW = toxe * phi_pre / (Weff + w0)

    # --- DIBL_Sft  (b4ld.c §1107-1117) ----------------------------------- #
    # Regularizer: when (eta0 + etab*Vbs) < 1e-4 use rational form to avoid
    # negative theta0vb0 contribution; we replicate it exactly.
    T3_d = eta0 + etab * Vbseff
    T9_d = 1.0 / (3.0 - 2.0e4 * T3_d)
    T3_clamped = torch.where(T3_d < 1.0e-4, (2.0e-4 - T3_d) * T9_d, T3_d)
    # b4temp.c §1531-1540 computes theta0vb0; we have it cached in sd.theta0vb0
    # (approximation). The C uses a slightly different form (with dsub):
    #   θ0vb0 = exp(dsub·Leff/√(εsub/(εrox·ε0)·tox·Xdep0)) / ...  rational form.
    # Recompute faithfully here so DIBL_Sft tracks dsub correctly.
    # AUTOGRAD-FIX: read dsub via the tensor-safe `t(...)` helper, preferring
    # the scaled per-instance dict (P) so external overrides — including
    # torch.Tensor leaves with requires_grad=True — flow through autograd.
    # The previous `float(...)` cast silently stripped gradients, which broke
    # stage-2 fitting of `dsub` in the v5 fitting script.
    dsub_v = t(P.get("dsub", model.get("dsub", model.get("drout", 0.56))))
    epsrox_t = t(ctx.epsrox)
    # SMOOTH: tensor-safe; keeps grads through epssub/toxe/Xdep0 if any becomes leaf.
    # Use very small eps in safe_sqrt: physical arg is ~1e-15 (epssub·toxe·Xdep0/epsrox);
    # the default 1e-12 floor would over-clamp by 3 orders of magnitude.
    tmp_dsub = torch.sqrt((epssub / (epsrox_t * EPS0) * toxe * Xdep0).clamp_min(1e-40))
    T0_dsub = dsub_v * t(geom.leff) / tmp_dsub.clamp_min(1e-40)
    # SMOOTH: replace if-on-tensor + math.exp with tensor _exp_threshold_branch
    theta0vb0 = _exp_threshold_branch(T0_dsub)

    DIBL_Sft = T3_clamped * theta0vb0 * Vds
    Lpe_Vb = safe_sqrt(1.0 + lpeb / Leff)

    # --- Final Vth assembly  b4ld.c §1121-1124 --------------------------- #
    Vth = (type_n * vth0
           + (k1ox * sqrtPhis - k1 * sqrtPhi_pre) * Lpe_Vb
           - k2ox * Vbseff
           - Delt_vth
           - T2_narrow
           + (k3 + k3b * Vbseff) * Vth_NarrowW
           + Tlpe1
           - DIBL_Sft)

    # ===================================================================== #
    # 4. Subthreshold n   b4ld.c §1133-1154    manual §3.2                  #
    # ===================================================================== #
    tmp1 = epssub / Xdep
    tmp2 = nfactor * tmp1
    tmp3 = cdsc + cdscb * Vbseff + cdscd * Vds
    tmp4 = (tmp2 + tmp3 * Theta0 + cit) / coxe
    # b4ld.c §1141-1154: regularize when tmp4 < -0.5 (n must stay > 0).
    # WAVE2-FIX (critique 7): inactive arm 1/(3+8·tmp4) singular at -3/8 ∈ (-0.5, 0).
    n_a = 1.0 + tmp4
    _ndenom = 3.0 + 8.0 * tmp4
    _ndenom_safe = torch.where(_ndenom.abs() > 1e-6, _ndenom,
                                torch.full_like(_ndenom, -1e-6))
    n_b = (1.0 + 3.0 * tmp4) / _ndenom_safe
    n = torch.where(tmp4 >= -0.5, n_a, n_b)

    # ===================================================================== #
    # 5. Pocket DITS Vth correction   b4ld.c §1158-1187   manual §3.0 (DITS)#
    # ===================================================================== #
    # Only active if dvtp0 > 0  (scalar param, so plain Python branch is fine).
    if float(model.get("dvtp0", 0.0)) > 0.0:
        T0_p = -dvtp1 * Vds
        T2_p = safe_exp(T0_p)                           # SMOOTH: safe_exp guards MIN_EXP
        T3_p = Leff + dvtp0 * (1.0 + T2_p)
        # tempMod < 2 path → use Vtm
        T4_p = Vtm * safe_log(Leff / T3_p)              # SMOOTH: safe_log on positive ratio
        Vth = Vth - n * T4_p

    # WAVE2-FIX-3 (Gap 6): v4.7 DITS_SFT2  b4ld.c §1189-1205  (only if both nonzero).
    # The C form `(exp(2·dvtp4·Vds) - 1)/(exp(2·dvtp4·Vds) + 1)` is identically
    # tanh(dvtp4·Vds): (e^(2x) - 1)/(e^(2x) + 1) = tanh(x).
    # SMOOTH: replace the C exp-rational form with torch.tanh — bounded, smooth,
    # avoids exp overflow for large positive Vds, exact algebraic equivalent.
    if (float(model.get("dvtp4", 0.0)) != 0.0
            and float(model.get("dvtp2factor", 0.0)) != 0.0):
        DITS_Sft2 = dvtp2factor * torch.tanh(dvtp4 * Vds)
        Vth = Vth - DITS_Sft2

    # ===================================================================== #
    # 6. Vgsteff  — smooth subthreshold↔strong-inversion bridge             #
    #    b4ld.c §1238-1296    manual §3.3                                   #
    # ===================================================================== #
    # ---- Poly Gate Si Depletion (BSIM4polyDepletion, b4ld.c §5170-5202) -- #
    # Closed-form (NOT Newton): for ngate in (1e18, 1e25) and Vgs > phi:
    #   T1 = 1e6·q·epsgate·ngate / coxe²
    #   T8 = Vgs - phi
    #   T4 = sqrt(1 + 2·T8/T1)
    #   T2 = 2·T8 / (T4 + 1)
    #   T3 = 0.5·T2² / T1               (Vpoly)
    #   T7 = 1.12 - T3 - 0.05
    #   T6 = sqrt(T7² + 0.224)
    #   T5 = 1.12 - 0.5·(T7 + T6)
    #   Vgs_eff = Vgs - T5
    # Differentiable as-is. We mask with torch.where on the active condition;
    # `epsgate` is BSIM4 epsrox·EPS0 in the C code path (T1 there).
    def _poly_dep(Vg_in: torch.Tensor) -> torch.Tensor:
        if not (1.0e18 < ngate < 1.0e25) or epsrox_v == 0.0 or coxe_f == 0.0:
            return Vg_in
        epsgate_f = epsrox_v * EPS0
        T1_pd = t(1.0e6 * Charge_q * epsgate_f * ngate / (coxe_f * coxe_f))
        T8_pd = Vg_in - phi_pre
        # Only apply when Vg > phi; below threshold, return Vg unchanged (smooth via where)
        active = T8_pd > 0.0
        # SMOOTH: clamp T8>=0 inside sqrt to keep grad finite when inactive
        T8_safe = T8_pd.clamp_min(0.0)
        T4_pd = safe_sqrt(1.0 + 2.0 * T8_safe / T1_pd)
        T2_pd = 2.0 * T8_safe / (T4_pd + 1.0)
        T3_pd = 0.5 * T2_pd * T2_pd / T1_pd
        T7_pd = 1.12 - T3_pd - 0.05
        T6_pd = safe_sqrt(T7_pd * T7_pd + 0.224)
        T5_pd = 1.12 - 0.5 * (T7_pd + T6_pd)
        return torch.where(active, Vg_in - T5_pd, Vg_in)

    Vgs_eff = _poly_dep(Vgs)
    Vgd_eff = _poly_dep(Vgs - Vds)   # Vgd path (b4ld.c line 1221)
    # Vds_eff for poly-depletion = Vgs_eff - Vgd_eff (BSIM4 line 1224 area)
    # For DC current we only consume Vgs_eff downstream (Vgd appears only in
    # symmetric-Vds capMod paths we skip).
    _ = Vgd_eff
    Vgst = Vgs_eff - Vth

    T0v = n * Vtm
    T1v = mstar * Vgst
    T2v = T1v / T0v.clamp_min(1e-30)
    # b4ld.c §1242-1263: faithful 3-branch numerator (T10).
    #   T2 >  EXP_THR  →  T10 = T1 = mstar·Vgst        (strong inversion linear)
    #   T2 < -EXP_THR  →  T10 = n·Vtm·log(1+MIN_EXP)   (deep subthreshold floor)
    #   else            →  T10 = n·Vtm·log(1+exp(T2))  (canonical bridge)
    # All three arms are evaluated with safe primitives so torch.where does NOT
    # propagate NaN even from the un-selected branch.
    ExpVgst = safe_exp(T2v)                                  # safe in all arms
    T10_bridge = n * Vtm * torch.log1p(ExpVgst)
    T10_strong = T1v
    T10_deep = n * Vtm * _LOG1P_MIN_EXP
    # Compose: pick deep when T2<-EXP_THR, strong when T2>EXP_THR, else bridge.
    T10v = torch.where(T2v > EXP_THRESHOLD, T10_strong,
            torch.where(T2v < -EXP_THRESHOLD, T10_deep, T10_bridge))

    # b4ld.c §1265-1291: faithful 3-branch denominator (T9).
    # T1_off = voffcbn - (1-mstar)·Vgst ;  T2_off = T1_off/T0
    #   T2_off < -EXP_THR → T3 = coxe·MIN_EXP/cdep0
    #   T2_off >  EXP_THR → T3 = coxe·MAX_EXP/cdep0
    #   else               → T3 = coxe/cdep0 · exp(T2_off)  (canonical)
    # Then T9 = mstar + n·T3.
    T1_off = voffcbn - (1.0 - mstar) * Vgst
    T2_off = T1_off / T0v.clamp_min(1e-30)
    coxe_over_cdep0 = coxe / cdep0.clamp_min(1e-30)
    ExpOff = safe_exp(T2_off)                                # safe in all arms
    T3_bridge = coxe_over_cdep0 * ExpOff
    T3_low = coxe_over_cdep0 * MIN_EXP
    T3_high = coxe_over_cdep0 * MAX_EXP
    T3v = torch.where(T2_off > EXP_THRESHOLD, T3_high,
           torch.where(T2_off < -EXP_THRESHOLD, T3_low, T3_bridge))
    T9v = mstar + n * T3v
    Vgsteff = T10v / T9v.clamp_min(1e-30)

    # ===================================================================== #
    # 6b. Weff correction  b4ld.c §1298-1311                                #
    # ===================================================================== #
    # Weff = Weff0 - 2·(dwg·Vgsteff + dwb·(sqrtPhis - sqrtPhi))
    # Plus a discontinuity guard for Weff < 2e-8.
    T9_w = sqrtPhis - sqrtPhi_pre
    Weff = Weff - 2.0 * (dwg * Vgsteff + dwb * T9_w)
    # Discontinuity guard (b4ld.c §1305-1311):
    #   if Weff < 2e-8: Weff = 2e-8·(4e-8 - Weff)/(6e-8 - 2·Weff)
    # SMOOTH: torch.where over both arms; both differentiable.
    T0_w = 1.0 / (6.0e-8 - 2.0 * Weff).clamp_min(1e-30)
    Weff_clamp = 2.0e-8 * (4.0e-8 - Weff) * T0_w
    Weff = torch.where(Weff < 2.0e-8, Weff_clamp, Weff)

    # ===================================================================== #
    # 7. Abulk   b4ld.c §1338-1395    manual §5.1                           #
    # ===================================================================== #
    T9_a = 0.5 * k1ox * Lpe_Vb / sqrtPhis.clamp_min(1e-30)
    T1_a = T9_a + k2ox - k3b * Vth_NarrowW

    # SMOOTH: safe_sqrt — but EPS_SQRT=1e-12 is far too coarse here. Physical
    # xj·Xdep ~ (1e-7)·(1e-7) = 1e-14; the default 1e-12 floor inflates T9_xj
    # by 100×, which propagates to Abulk0 (T5 collapses to ~0), making Abulk
    # ~10% low and Vdsat ~52 mV high. Use a fp64-safe 1e-30 floor (matches the
    # epssub/(εrox·ε0)·toxe·Xdep0 path in §3 DIBL_Sft, which has the same
    # numerical scale).  z214 finding 2026-04-30.
    T9_xj = torch.sqrt((xj * Xdep).clamp_min(1e-30))
    tmp1 = Leff + 2.0 * T9_xj
    T5_a = Leff / tmp1.clamp_min(1e-30)
    tmp2_a = a0 * T5_a
    tmp3_a = Weff + b1
    tmp4_a = b0 / tmp3_a.clamp_min(1e-30)
    T2_a = tmp2_a + tmp4_a
    T7_a = T5_a * T5_a * T5_a   # T6 = T5²; T7 = T5·T6

    Abulk0 = 1.0 + T1_a * T2_a
    T8_a = ags * a0 * T7_a
    dAbulk_dVg = -T1_a * T8_a
    Abulk = Abulk0 + dAbulk_dVg * Vgsteff

    # b4ld.c §1363-1375: rational regularizer when Abulk0 / Abulk < 0.1
    Abulk0 = torch.where(
        Abulk0 < 0.1,
        (0.2 - Abulk0) / (3.0 - 20.0 * Abulk0),
        Abulk0,
    )
    Abulk = torch.where(
        Abulk < 0.1,
        (0.2 - Abulk) / (3.0 - 20.0 * Abulk),
        Abulk,
    )

    # b4ld.c §1378-1393: keta body-bias scaling with -0.9 regularizer.
    T2_k = keta * Vbseff
    T0_a = torch.where(
        T2_k >= -0.9,
        1.0 / (1.0 + T2_k),
        (17.0 + 20.0 * T2_k) / (0.8 + T2_k),
    )
    Abulk = Abulk * T0_a
    Abulk0 = Abulk0 * T0_a   # tracked but not used downstream in DC-only path

    # ===================================================================== #
    # 8. Mobility   b4ld.c §1416-1578  (mobMod = 1 path)    manual §5.2     #
    # ===================================================================== #
    # mtrlMod=0 ⇒ T14 = 0
    T14 = t(0.0)
    T0_mu = Vgsteff + 2.0 * Vth - T14
    T2_mu = 1.0 + uc * Vbseff
    T3_mu = T0_mu / toxe
    T4_mu = T3_mu * (ua + ub * T3_mu)

    # ud term (Coulombic): T8 = ud · (toxe/(Vgsteff+2|Vth|))² · Vth
    T12_mu = safe_sqrt(Vth * Vth + 1.0e-4)
    T9_mu = 1.0 / (Vgsteff + 2.0 * T12_mu).clamp_min(1e-30)
    T10_mu = T9_mu * toxe
    T8_mu = ud * T10_mu * T10_mu * Vth
    T6_mu = T8_mu * Vth
    T5_mu = T4_mu * T2_mu + T6_mu

    # b4ld.c §1561-1571: Denomi rational regularizer at T5 < -0.8
    Denomi = torch.where(
        T5_mu >= -0.8,
        1.0 + T5_mu,
        (0.6 + T5_mu) / (7.0 + 10.0 * T5_mu),
    )
    mueff = u0temp / Denomi.clamp_min(1e-30)

    # ===================================================================== #
    # 8b. Rds(Vgsteff, Vbseff)   b4ld.c §1313-1336                          #
    # ===================================================================== #
    # rdsmod = 0 (DEFAULT in BSIM4 per b4set.c:107-108) → Rds is INTERNAL,
    #   bias-dependent, modifies Idsat AND Vdsat.
    # rdsmod = 1 → Rds is EXTERNAL (lumped resistor at S/D nodes), so the
    #   internal expressions take Rds=0.
    rdsmod_v = int(model.get("rdsmod", 0))

    # Internal rds0/rdswmin per b4temp.c §1255: rds*T10*nf/(weffCJ*1e6)^wr.
    # We get rdstemp = rdsw_scaled · (1 + prt·delT) from temp.py, plus rdswmin.
    nf_v = float(model.get("nf", 1.0))
    weffCJ_um = float(geom.weffCJ) * 1.0e6
    wr_v = float(P.get("wr", model.get("wr", 1.0)))
    PowWeffWr = (max(weffCJ_um, 1e-30) ** wr_v) if wr_v != 0.0 else 1.0
    PowWeffWr = max(PowWeffWr, 1e-30)
    # rdstemp already has the (1+prt·delT) factor. rdswmin temp-scales identically
    # (prt·delT factor); reconstruct from prt.
    prt_v = float(model.get("prt", 0.0))
    delTemp_v = float(ctx.Temp - ctx.Tnom)
    if int(model.get("tempmod", 0)) == 0:
        rds_temp_factor = 1.0 + prt_v * (ctx.TRatio - 1.0)
    else:
        rds_temp_factor = 1.0 + prt_v * delTemp_v
    rds0_val = sd.rdstemp * nf_v / PowWeffWr
    rdswmin_val = float(P.get("rdswmin", model.get("rdswmin", 0.0))) * rds_temp_factor * nf_v / PowWeffWr

    if rdsmod_v == 0:
        # b4ld.c §1316-1328  — full bias-dependent Rds.
        prwg = t(P.get("prwg", model.get("prwg", 1.0)))
        prwb = t(P.get("prwb", model.get("prwb", 0.0)))
        # T9 = sqrtPhis - sqrtPhi_pre  (b4ld.c §1299)
        T9_rds = sqrtPhis - sqrtPhi_pre
        T0_rds = 1.0 + prwg * Vgsteff
        T1_rds = prwb * T9_rds
        T2_rds = 1.0 / T0_rds.clamp_min(1e-30) + T1_rds
        # SMOOTH: safe_sqrt for the +0.01 regularizer (b4ld.c §1322)
        T3_rds = T2_rds + safe_sqrt(T2_rds * T2_rds + 0.01)
        T4_rds = t(rds0_val) * 0.5
        Rds = t(rdswmin_val) + T3_rds * T4_rds
        Rds = Rds.clamp_min(0.0)   # physical guard
    else:
        # rdsmod = 1: external resistor; internal expressions see Rds=0.
        Rds = torch.zeros_like(Vgsteff)

    # ===================================================================== #
    # 9. Vdsat   b4ld.c §1580-1679                                          #
    #    manual §5.6.1-§5.6.2                                               #
    # ===================================================================== #
    Esat = 2.0 * vsattemp / mueff.clamp_min(1e-30)
    EsatL = Esat * Leff
    # WVCox = Weff·vsattemp·coxe;  WVCoxRds = WVCox·Rds  (b4ld.c §1581-1582)
    WVCox = Weff * vsattemp * coxe
    WVCoxRds = WVCox * Rds   # zero tensor if rdsmod=1

    # Lambda: a1, a2 dependence  (b4ld.c §1591-1609)
    a1_v = float(P.get("a1", model.get("a1", 0.0)))
    if a1_v == 0.0:
        Lambda = a2  # tensor (likely 1.0)
    elif a1_v > 0.0:
        T0_l = 1.0 - a2
        T1_l = T0_l - a1 * Vgsteff - 1e-4
        T2_l = safe_sqrt(T1_l * T1_l + 4e-4 * T0_l)
        Lambda = a2 + T0_l - 0.5 * (T1_l + T2_l)
    else:
        T1_l = a2 + a1 * Vgsteff - 1e-4
        T2_l = safe_sqrt(T1_l * T1_l + 4e-4 * a2)
        Lambda = 0.5 * (T1_l + T2_l)

    Vgst2Vtm = Vgsteff + 2.0 * Vtm
    Lambda_safe = Lambda.clamp_min(1e-12)

    # b4ld.c §1620-1635 (simple): Rds=0 AND Lambda=1
    # b4ld.c §1636-1679 (full quadratic): all other cases.
    # We always compute the full quadratic; it reduces continuously to the
    # simple form as (Rds, 1-Lambda) → 0.
    # SMOOTH: use full quadratic everywhere with safe_sqrt; only divide by
    # T0 with clamp_min in the Lambda≈1 ∧ Rds=0 limit (then fall back to simple).
    T9_q = Abulk * WVCoxRds                                  # b4ld.c §1638
    T7_q = Vgst2Vtm * T9_q                                   # b4ld.c §1640
    T6_q = Vgst2Vtm * WVCoxRds                               # b4ld.c §1641
    T0_vd = 2.0 * Abulk * (T9_q - 1.0 + 1.0 / Lambda_safe)   # b4ld.c §1642
    T1_vd = Vgst2Vtm * (2.0 / Lambda_safe - 1.0) + Abulk * EsatL + 3.0 * T7_q  # §1649
    T2_vd = Vgst2Vtm * (EsatL + 2.0 * T6_q)                  # b4ld.c §1658
    # Discriminant T3 = sqrt(T1² - 2·T0·T2); SMOOTH: safe_sqrt
    disc = T1_vd * T1_vd - 2.0 * T0_vd * T2_vd
    T3_vd = safe_sqrt(disc)
    # Avoid 0/0 when T0→0 (Rds=0 ∧ Lambda=1): use simple form there.
    T0_safe = torch.where(T0_vd.abs() < 1.0e-12,
                          torch.full_like(T0_vd, 1.0e-12),
                          T0_vd)
    Vdsat_full = (T1_vd - T3_vd) / T0_safe
    Vdsat_simple = EsatL * Vgst2Vtm / (Abulk * EsatL + Vgst2Vtm).clamp_min(1e-30)
    use_full = T0_vd.abs() > 1.0e-9
    Vdsat = torch.where(use_full, Vdsat_full, Vdsat_simple)

    # ===================================================================== #
    # 10. Vdseff   b4ld.c §1682-1719   manual §5.6.3                        #
    # ===================================================================== #
    # This is BSIM4's smooth-min implementation; port bit-faithfully.
    T1_v = Vdsat - Vds - delta
    T2_v = safe_sqrt(T1_v * T1_v + 4.0 * delta * Vdsat)
    Vdseff_a = Vdsat - 0.5 * (T1_v + T2_v)             # T1 >= 0  (Vds < Vdsat-δ)
    T4_v = (2.0 * delta) / (T2_v - T1_v).clamp_min(1e-30)
    T5_v = 1.0 - T4_v
    Vdseff_b = Vdsat * T5_v                            # T1 < 0
    Vdseff = torch.where(T1_v >= 0.0, Vdseff_a, Vdseff_b)
    # b4ld.c §1712: clamp at Vds=0
    Vdseff = torch.where(Vds == 0.0, torch.zeros_like(Vds), Vdseff)
    # b4ld.c §1718-1719: hard cap Vdseff <= Vds
    Vdseff = smooth_min(Vdseff, Vds, sharpness=1000.0)  # SMOOTH: faithful cap

    diffVds = Vds - Vdseff

    # ===================================================================== #
    # 11. Idl  b4ld.c §1790-1844 (Coxeff=coxe simplification)               #
    #     manual §5.6.4                                                     #
    # ===================================================================== #
    # ---- Tcen / Coxeff centroid (b4ld.c §1789-1805 capMod=2 path) ------- #
    # T0 = (Vgsteff + vtfbphi2) / (2e8·toxp)
    # tmp3 = exp(bdos·0.7·log(T0)) = T0^(0.7·bdos)
    # T1 = 1 + tmp3
    # Tcen = ados·1.9e-9 / T1
    # Coxeff = epssub·coxp / (epssub + coxp·Tcen)
    tmp2_tc = (2.0e8 * toxp).clamp_min(1e-30)
    T0_tc_raw = (Vgsteff + vtfbphi2) / tmp2_tc
    # T0 must be > 0 for log; in deep subthreshold Vgsteff~0 ⇒ T0 small but >0
    T0_tc = T0_tc_raw.clamp_min(1e-30)
    tmp3_tc = safe_exp(bdos * 0.7 * safe_log(T0_tc))    # SMOOTH: safe primitives
    T1_tc = 1.0 + tmp3_tc
    Tcen = ados * 1.9e-9 / T1_tc.clamp_min(1e-30)
    Coxeff = epssub * coxp / (epssub + coxp * Tcen).clamp_min(1e-30)
    CoxeffWovL = Coxeff * Weff / Leff
    beta = mueff * CoxeffWovL

    AbovVgst2Vtm = Abulk / Vgst2Vtm.clamp_min(1e-30)
    T0_idl = 1.0 - 0.5 * Vdseff * AbovVgst2Vtm
    fgche1 = Vgsteff * T0_idl
    fgche2 = 1.0 + Vdseff / EsatL.clamp_min(1e-30)
    gche = beta * fgche1 / fgche2.clamp_min(1e-30)

    # b4ld.c §1843-1844:  Idl = gche / (1 + gche·Rds)
    # When Rds = 0 (rdsmod=1) this collapses to Idl = gche.
    Idl = gche / (1.0 + gche * Rds).clamp_min(1e-30)

    # ===================================================================== #
    # 12. DIBL / CLM / SCBE / DITS — combine Va contributions               #
    #     b4ld.c §1851-2110     manual §5.7                                 #
    # ===================================================================== #
    # FP — pocket-implant Rout degradation factor (b4ld.c §1853-1861)
    fprout_v = float(model.get("fprout", 0.0))
    if fprout_v <= 0.0:
        FP = torch.ones_like(Vgst2Vtm)
    else:
        T9_fp = fprout * safe_sqrt(Leff) / Vgst2Vtm.clamp_min(1e-30)
        FP = 1.0 / (1.0 + T9_fp)

    # PvagTerm — pvag pocket modifier (b4ld.c §1864-1880)
    T8_pv = pvag / EsatL.clamp_min(1e-30)
    T9_pv = T8_pv * Vgsteff
    PvagTerm = torch.where(
        T9_pv > -0.9,
        1.0 + T9_pv,
        (0.8 + T9_pv) / (17.0 + 20.0 * T9_pv),
    )

    # --- VACLM    b4ld.c §1882-1911    manual §5.7.1 -------------------- #
    pclm_v = float(P.get("pclm", model.get("pclm", 1.3)))
    if pclm_v > MIN_EXP:
        # b4ld.c §1883:  T0 = 1 + Rds·Idl  (Rds-coupled CLM denominator).
        # T1 = Leff + Vdsat/Esat = Leff + Vdsat·Leff/EsatL.
        T0_clm = 1.0 + Rds * Idl
        T2_clm = Vdsat / Esat.clamp_min(1e-30)
        T1_clm = Leff + T2_clm
        Cclm = FP * PvagTerm * T0_clm * T1_clm / (pclm * litl).clamp_min(1e-30)
        # diffVds≈0 case — guard with floor; result is huge VACLM (channel-length
        # modulation off in linear region) which is correct.
        diffVds_safe = diffVds.clamp_min(1e-12)
        VACLM = Cclm * diffVds_safe
    else:
        VACLM = torch.full_like(Vds, MAX_EXP)
        Cclm = torch.full_like(Vds, MAX_EXP)

    # --- VADIBL    b4ld.c §1913-1957    manual §5.7.2 ------------------- #
    # thetaRout = pdiblc1 · _exp_threshold_branch(drout·Leff/tmp_dsub) + pdiblc2
    # SMOOTH: tensor-safe; tmp_dsub is now a tensor.
    T0_dr = drout * t(geom.leff) / tmp_dsub.clamp_min(1e-40)
    T5_dr = _exp_threshold_branch(T0_dr)
    thetaRout = pdiblc1 * T5_dr + pdiblc2

    # SMOOTH: branch on tensor via torch.where to keep grads through thetaRout
    T8_db = Abulk * Vdsat
    T0_db = Vgst2Vtm * T8_db
    T1_db = Vgst2Vtm + T8_db
    VADIBL_active = (Vgst2Vtm - T0_db / T1_db.clamp_min(1e-30)) / thetaRout.clamp_min(1e-30)
    # Pocket pdiblb body-bias correction (b4ld.c §1934-1951)
    T7_db = pdiblb * Vbseff
    T3_db = torch.where(
        T7_db >= -0.9,
        1.0 / (1.0 + T7_db),
        (17.0 + 20.0 * T7_db) / (0.8 + T7_db),
    )
    VADIBL_active = VADIBL_active * T3_db * PvagTerm
    VADIBL = torch.where(
        thetaRout > MIN_EXP,
        VADIBL_active,
        torch.full_like(Vds, MAX_EXP),
    )

    # --- VADITS    b4ld.c §1969-1990    manual §5.7.3 ------------------- #
    T0_dits = pditsd * Vds
    T1_dits = safe_exp(T0_dits)                        # SMOOTH: safe_exp clipped
    pdits_v = float(P.get("pdits", model.get("pdits", 0.0)))
    if pdits_v > MIN_EXP:
        T2_dits = 1.0 + pditsl * Leff
        VADITS = (1.0 + T2_dits * T1_dits) / pdits.clamp_min(1e-30)
        VADITS = VADITS * FP
    else:
        VADITS = torch.full_like(Vds, MAX_EXP)

    # --- VASCBE    b4ld.c §1992-2011    manual §5.7.4 ------------------- #
    pscbe2_v = float(model.get("pscbe2", 1e-5))
    pscbe1_v = float(model.get("pscbe1", 4.24e8))
    if pscbe2_v > 0.0 and pscbe1_v >= 0.0:
        # SMOOTH: clamp diffVds away from 0 to avoid 1/0; safe_exp handles huge T0
        diffVds_scbe = diffVds.clamp_min(1e-12)
        T0_scbe = pscbe1 * litl / diffVds_scbe
        VASCBE = Leff * safe_exp(-T0_scbe) / pscbe2     # exp(-x)·... reformulated
        # Wait — b4ld.c writes VASCBE = Leff * exp(T0) / pscbe2 where T0 is
        # NEGATIVE (note: BSIM4 uses pscbe1·litl/diffVds positive but the term
        # appears in 1/Va as 1/VASCBE = pscbe2 · exp(-pscbe1·litl/(Vds-Vdseff))
        # / Leff per the manual §5.7.4 — so VASCBE = Leff·exp(+pscbe1·litl/diff)/pscbe2.
        # Re-derive: 1/Va_scbe = (pscbe2/Leff) · exp(-pscbe1·litl/diffVds).
        # ⇒ VASCBE = Leff·exp(+pscbe1·litl/diffVds) / pscbe2.   Match C: T0 = +x.
        VASCBE = Leff * safe_exp(T0_scbe) / pscbe2
    else:
        VASCBE = torch.full_like(Vds, MAX_EXP)

    # --- Vasat (b4ld.c §1765-1788)  manual §5.6 (extrinsic case) -------- #
    # Vasat = T0 / T1 where:
    #   tmp4 = 1 - 0.5·Abulk·Vdsat/Vgst2Vtm
    #   T0 = EsatL + Vdsat + 2·WVCoxRds·Vgsteff·tmp4
    #   T1 = 2/Lambda - 1 + WVCoxRds·Abulk
    # When WVCoxRds = 0 (rdsmod=1) and Lambda=1 → T0=EsatL+Vdsat, T1=1, Vasat=EsatL+Vdsat.
    # That is NOT equal to Vdsat in general — old port collapsed this incorrectly.
    Vgst2Vtm_safe = Vgst2Vtm.clamp_min(1e-30)
    tmp4_va = 1.0 - 0.5 * Abulk * Vdsat / Vgst2Vtm_safe
    T0_va = EsatL + Vdsat + 2.0 * WVCoxRds * Vgsteff * tmp4_va
    T1_va = 2.0 / Lambda_safe - 1.0 + WVCoxRds * Abulk
    Vasat = T0_va / T1_va.clamp_min(1e-30)
    Va = Vasat + VACLM

    # ===================================================================== #
    # 13. Final Ids — chain-multiply DIBL/DITS/CLM/SCBE  b4ld.c §2013-2091   #
    #     manual §5.6.4 + §5.7  (cdrain = Ids·Vdseff)                       #
    # ===================================================================== #
    # Faithful BSIM4 chain (NOT a parallel-resistor lump-sum):
    #   Idsa  = Idl · (1 + diffVds/VADIBL)
    #   Idsa *= (1 + diffVds/VADITS)
    #   Idsa *= (1 + log(Va/Vasat)/Cclm)         ← CLM as logarithm, not 1/V
    #   Ids   = Idsa · (1 + diffVds/VASCBE)
    # All "1 + small" factors stay >0 because diffVds≥0 and the V's are large.
    # Note: `Idl` in our port is gche (S); cdrain = Ids·Vdseff is computed below.
    Idsa = Idl * (1.0 + diffVds / VADIBL.clamp_min(1e-30))
    Idsa = Idsa * (1.0 + diffVds / VADITS.clamp_min(1e-30))
    # CLM term: log(Va/Vasat)/Cclm. Va≥Vasat ⇒ log≥0; Cclm>0 by guard above.
    Vasat_safe = Vasat.clamp_min(1e-30)
    log_VaVasat = safe_log((Va / Vasat_safe).clamp_min(1.0))   # never negative
    Idsa = Idsa * (1.0 + log_VaVasat / Cclm.clamp_min(1e-30))
    # WAVE2-FIX-1 (Gap 2): snapshot pre-SCBE Idsa·Vdseff for impact-ionization (Iii).
    # b4ld.c §2069: T4 = Idsa·Vdseff; Isub = T1·T4. SCBE is applied only to the final
    # Ids (line 2089-2091), not to Iii. Storing the current form (Idsa_chan * Vdseff)
    # so leak.compute_iimpact can directly multiply by T1.
    Idsa_Vdseff = Idsa * Vdseff
    Ids_chan = Idsa * (1.0 + diffVds / VASCBE.clamp_min(1e-30))
    # cdrain = Ids·Vdseff   (channel current per BSIM4 convention)
    Ids = Ids_chan * Vdseff

    # NMOS sign convention: positive Ids for Vds>0.
    return DCResult(
        Ids=Ids,
        Vth=Vth,
        Vgsteff=Vgsteff,
        Vdsat=Vdsat,
        Vdseff=Vdseff,
        Abulk=Abulk,
        n=n,
        mueff=mueff,
        Rds=Rds if rdsmod_v == 0 else None,
        Idsa=Idsa_Vdseff,
        Vgs_eff=Vgs_eff,
        Vbseff=Vbseff,
    )


# --------------------------------------------------------------------------- #
# Convenience wrapper                                                          #
# --------------------------------------------------------------------------- #

def compute_dc_simple(model: BSIM4Model, geom: Geometry, T_C: float,
                      Vgs, Vds, Vbs=0.0) -> DCResult:
    """One-shot helper that builds SizeDependParam and calls compute_dc."""
    sd = compute_size_dep(model, geom, T_C)
    return compute_dc(model, sd, Vgs=Vgs, Vds=Vds, Vbs=Vbs)
