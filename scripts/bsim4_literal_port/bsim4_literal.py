"""Literal C-to-Python port of ngspice b4ld.c §1042-1336.

A.5.w (2026-05-02): pure-Python (no torch, no smoothing, no abstraction)
translation of ngspice's BSIM4 device equations from Vbseff through
Vgsteff. Each variable named identically to the C source; each branch
preserved verbatim. Goal: bit-exact reproduction of what ngspice
WOULD compute, so we can diff intermediates against pyport and find
the bug.

Input: a dict of binned per-instance parameters (matching pParam->BSIM4X
naming) + Vgs, Vds, Vbs.
Output: dict of every named intermediate (Vbseff, Phis, ..., Vgsteff).
"""
from __future__ import annotations
import math

EPS0 = 8.854187817e-12
EPSSI = 11.7 * EPS0
EXP_THRESHOLD = 34.0
MIN_EXP = math.exp(-EXP_THRESHOLD)
MAX_EXP = math.exp(EXP_THRESHOLD)


def bsim4_compute(P, vgs, vds, vbs):
    """Faithful translation of b4ld.c lines 1042-1336.

    P is a dict with keys (all binned, temperature-shifted as needed):
      vbsc, phi, sqrtPhi, Xdep0, factor1, vbi, leff, weff, w0, toxe,
      coxe, vtm, vtm0, dvt0, dvt1, dvt2, dvt0w, dvt1w, dvt2w,
      eta0, etab, dsub, theta0vb0, k1ox, k1, k2ox, lpe0, lpeb,
      kt1, kt1l, kt2, type, vth0, k3, k3b, dvtp0, dvtp1, dvtp2factor,
      dvtp4, nfactor, cdsc, cdscb, cdscd, cit, voffcbn, mstar, cdep0,
      Tnom, Temp
    """
    out = {}

    # === Vbseff (b4ld.c §1042-1052) ===
    T0 = vbs - P["vbsc"] - 0.001
    T1 = math.sqrt(T0 * T0 - 0.004 * P["vbsc"])
    if T0 >= 0.0:
        Vbseff = P["vbsc"] + 0.5 * (T0 + T1)
    else:
        T2 = -0.002 / (T1 - T0)
        Vbseff = P["vbsc"] * (1.0 + T2)
    out["Vbseff_pre_jx"] = Vbseff

    # JX correction (§1054-1059)
    T9 = 0.95 * P["phi"]
    T0 = T9 - Vbseff - 0.001
    T1 = math.sqrt(T0 * T0 + 0.004 * T9)
    Vbseff = T9 - 0.5 * (T0 + T1)
    out["Vbseff"] = Vbseff

    Phis = P["phi"] - Vbseff
    sqrtPhis = math.sqrt(Phis)
    out["Phis"] = Phis
    out["sqrtPhis"] = sqrtPhis

    Xdep = P["Xdep0"] * sqrtPhis / P["sqrtPhi"]
    out["Xdep"] = Xdep

    Leff = P["leff"]
    Vtm = P["vtm"]
    Vtm0 = P["vtm0"]

    # === Vth machinery (§1073-) ===
    T3 = math.sqrt(Xdep)
    V0 = P["vbi"] - P["phi"]
    out["V0"] = V0

    # lt1 (§1077-1088)
    T0 = P["dvt2"] * Vbseff
    if T0 >= -0.5:
        T1 = 1.0 + T0
    else:
        T4 = 1.0 / (3.0 + 8.0 * T0)
        T1 = (1.0 + 3.0 * T0) * T4
    lt1 = P["factor1"] * T3 * T1
    out["lt1"] = lt1

    # ltw (§1090-1101)
    T0 = P["dvt2w"] * Vbseff
    if T0 >= -0.5:
        T1 = 1.0 + T0
    else:
        T4 = 1.0 / (3.0 + 8.0 * T0)
        T1 = (1.0 + 3.0 * T0) * T4
    ltw = P["factor1"] * T3 * T1
    out["ltw"] = ltw

    # Theta0 (§1103-1116)
    T0 = P["dvt1"] * Leff / lt1
    if T0 < EXP_THRESHOLD:
        T1 = math.exp(T0)
        T2 = T1 - 1.0
        T3_ = T2 * T2
        T4 = T3_ + 2.0 * T1 * MIN_EXP
        Theta0 = T1 / T4
    else:
        Theta0 = 1.0 / (MAX_EXP - 2.0)
    out["Theta0"] = Theta0
    Delt_vth = P["dvt0"] * Theta0 * V0
    out["Delt_vth"] = Delt_vth

    # T5 (narrow-W) (§1121-1134)
    T0 = P["dvt1w"] * P["weff"] * Leff / ltw
    if T0 < EXP_THRESHOLD:
        T1 = math.exp(T0)
        T2 = T1 - 1.0
        T3_ = T2 * T2
        T4 = T3_ + 2.0 * T1 * MIN_EXP
        T5 = T1 / T4
    else:
        T5 = 1.0 / (MAX_EXP - 2.0)
    out["T5_narrow"] = T5
    T2_narrow = P["dvt0w"] * T5 * V0
    out["T2_narrow"] = T2_narrow

    # Tlpe1 (§1139-1146)
    TempRatio = P["Temp"] / P["Tnom"] - 1.0
    T0 = math.sqrt(1.0 + P["lpe0"] / Leff)
    Tlpe1 = (P["k1ox"] * (T0 - 1.0) * P["sqrtPhi"]
             + (P["kt1"] + P["kt1l"] / Leff
                + P["kt2"] * Vbseff) * TempRatio)
    out["Tlpe1"] = Tlpe1
    Vth_NarrowW = P["toxe"] * P["phi"] / (P["weff"] + P["w0"])
    out["Vth_NarrowW"] = Vth_NarrowW

    # DIBL_Sft (§1147-1157)
    T3 = P["eta0"] + P["etab"] * Vbseff
    if T3 < 1.0e-4:
        T9 = 1.0 / (3.0 - 2.0e4 * T3)
        T3 = (2.0e-4 - T3) * T9
    DIBL_Sft = T3 * P["theta0vb0"] * vds
    out["DIBL_Sft"] = DIBL_Sft

    Lpe_Vb = math.sqrt(1.0 + P["lpeb"] / Leff)

    # === Final Vth assembly (§1161-1164) ===
    Vth = (P["type"] * P["vth0"]
           + (P["k1ox"] * sqrtPhis - P["k1"] * P["sqrtPhi"]) * Lpe_Vb
           - P["k2ox"] * Vbseff - Delt_vth - T2_narrow
           + (P["k3"] + P["k3b"] * Vbseff) * Vth_NarrowW
           + Tlpe1 - DIBL_Sft)
    out["Vth_pre_DITS"] = Vth

    # === n calculation (§1173-1194) ===
    tmp1 = EPSSI / Xdep  # epssub assumed = EPSSI for non-mtrlMod
    tmp2 = P["nfactor"] * tmp1
    tmp3 = P["cdsc"] + P["cdscb"] * Vbseff + P["cdscd"] * vds
    tmp4 = (tmp2 + tmp3 * Theta0 + P["cit"]) / P["coxe"]
    if tmp4 >= -0.5:
        n = 1.0 + tmp4
    else:
        T0_ = 1.0 / (3.0 + 8.0 * tmp4)
        n = (1.0 + 3.0 * tmp4) * T0_
    out["n"] = n
    out["tmp1_n"] = tmp1
    out["tmp2_n"] = tmp2
    out["tmp3_n"] = tmp3
    out["tmp4_n"] = tmp4

    # === DITS Vth correction (§1198-1244) ===
    if P["dvtp0"] > 0.0:
        T0 = -P["dvtp1"] * vds
        T2 = MIN_EXP if T0 < -EXP_THRESHOLD else math.exp(T0)
        T3_ = Leff + P["dvtp0"] * (1.0 + T2)
        # tempMod < 2 path
        T4 = Vtm * math.log(Leff / T3_)
        Vth -= n * T4
    if P["dvtp4"] != 0.0 and P["dvtp2factor"] != 0.0:
        T1 = 2.0 * P["dvtp4"] * vds
        if T1 < -EXP_THRESHOLD:
            T0_ = MIN_EXP
        elif T1 > EXP_THRESHOLD:
            T0_ = MAX_EXP
        else:
            T0_ = math.exp(T1)
        DITS_Sft2 = P["dvtp2factor"] * (T0_ - 1) / (T0_ + 1)
        Vth -= DITS_Sft2
    out["Vth"] = Vth

    # === Vgsteff bridge (§1278-1336) ===
    # For DC analysis we use Vgs_eff = Vgs (no poly depletion solver here).
    # That's correct since Sebas's card doesn't enable poly-Si gate depletion.
    Vgs_eff = vgs
    Vgst = Vgs_eff - Vth
    out["Vgst"] = Vgst

    T0 = n * Vtm
    T1 = P["mstar"] * Vgst
    T2 = T1 / T0
    out["T0_bridge"] = T0
    out["T1_bridge_num"] = T1
    out["T2_bridge_num"] = T2

    if T2 > EXP_THRESHOLD:
        T10 = T1
    elif T2 < -EXP_THRESHOLD:
        T10 = Vtm * math.log(1.0 + MIN_EXP)
        T10 *= n
    else:
        ExpVgst = math.exp(T2)
        T3_ = Vtm * math.log(1.0 + ExpVgst)
        T10 = n * T3_
    out["T10_numerator"] = T10

    T1 = P["voffcbn"] - (1.0 - P["mstar"]) * Vgst
    T2 = T1 / T0
    out["T1_off"] = T1
    out["T2_off"] = T2

    if T2 < -EXP_THRESHOLD:
        T3_ = P["coxe"] * MIN_EXP / P["cdep0"]
        T9 = P["mstar"] + T3_ * n
    elif T2 > EXP_THRESHOLD:
        T3_ = P["coxe"] * MAX_EXP / P["cdep0"]
        T9 = P["mstar"] + T3_ * n
    else:
        ExpOff = math.exp(T2)
        T3_ = P["coxe"] / P["cdep0"]
        T4 = T3_ * ExpOff
        T9 = P["mstar"] + n * T4
    out["T3_bridge"] = T3_ if T2 <= -EXP_THRESHOLD or T2 >= EXP_THRESHOLD else T3_ * math.exp(T2)
    out["T9_denominator"] = T9

    Vgsteff = T10 / T9
    out["Vgsteff"] = Vgsteff

    return out
