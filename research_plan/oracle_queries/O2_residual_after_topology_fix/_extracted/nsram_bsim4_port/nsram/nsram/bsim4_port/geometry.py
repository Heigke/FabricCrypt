"""B03_GEOM — effective channel geometry from drawn dimensions.

Faithful port of b4temp.c lines 450-545.
"""
from __future__ import annotations
from dataclasses import dataclass

from .model_card import BSIM4Model


@dataclass
class Geometry:
    L: float       # channel length [m]
    W: float       # channel width [m]
    NF: int = 1


@dataclass
class EffectiveGeom:
    Lnew: float
    Wnew: float
    dl: float
    dlc: float
    dw: float
    dwc: float
    dwj: float
    leff: float
    weff: float
    leffCV: float
    weffCV: float
    weffCJ: float
    Inv_L: float
    Inv_W: float
    Inv_LW: float


def compute_geometry(model: BSIM4Model, geom: Geometry) -> EffectiveGeom:
    """Faithful port of b4temp.c:455-545.

    Equations (BSIM4 manual §1):
        Lnew = L + xl
        Wnew = W/NF + xw
        dl   = Lint + Ll/Lnew^Lln + Lw/Wnew^Lwn + Lwl/(Lnew^Lln · Wnew^Lwn)
        leff = Lnew - 2·dl
        ...
    """
    L, W, NF = geom.L, geom.W, geom.NF
    Lnew = L + model["xl"]
    Wnew = W / NF + model["xw"]

    T0 = Lnew ** model["lln"] if model["lln"] != 0 else 1.0
    T1 = Wnew ** model["lwn"] if model["lwn"] != 0 else 1.0
    tmp1 = model["ll"] / T0 + model["lw"] / T1 + model["lwl"] / (T0 * T1)
    dl = model["lint"] + tmp1
    tmp2 = model["llc"] / T0 + model["lwc"] / T1 + model["lwlc"] / (T0 * T1)
    dlc = model["dlc"] + tmp2

    T2 = Lnew ** model["wln"] if model["wln"] != 0 else 1.0
    T3 = Wnew ** model["wwn"] if model["wwn"] != 0 else 1.0
    tmp1 = model["wl"] / T2 + model["ww"] / T3 + model["wwl"] / (T2 * T3)
    dw = model["wint"] + tmp1
    tmp2 = model["wlc"] / T2 + model["wwc"] / T3 + model["wwlc"] / (T2 * T3)
    dwc = model["dwc"] + tmp2
    dwj = model["dwj"] + tmp2

    leff = Lnew - 2.0 * dl
    weff = Wnew - 2.0 * dw
    if leff <= 0.0:
        raise ValueError(f"BSIM4 geometry: leff={leff} <= 0")
    if weff <= 0.0:
        raise ValueError(f"BSIM4 geometry: weff={weff} <= 0")

    leffCV = Lnew - 2.0 * dlc
    weffCV = Wnew - 2.0 * dwc
    weffCJ = Wnew - 2.0 * dwj

    if model["binunit"] == 1:
        Inv_L = 1.0e-6 / leff
        Inv_W = 1.0e-6 / weff
        Inv_LW = 1.0e-12 / (leff * weff)
    else:
        Inv_L = 1.0 / leff
        Inv_W = 1.0 / weff
        Inv_LW = 1.0 / (leff * weff)

    return EffectiveGeom(
        Lnew=Lnew, Wnew=Wnew, dl=dl, dlc=dlc, dw=dw, dwc=dwc, dwj=dwj,
        leff=leff, weff=weff, leffCV=leffCV, weffCV=weffCV, weffCJ=weffCJ,
        Inv_L=Inv_L, Inv_W=Inv_W, Inv_LW=Inv_LW,
    )
