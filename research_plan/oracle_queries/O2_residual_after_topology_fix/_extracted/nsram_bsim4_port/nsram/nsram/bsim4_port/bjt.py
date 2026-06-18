"""bjt — differentiable Gummel-Poon NPN port.

DC-only Gummel-Poon (SPICE) sufficient for the NS-RAM body-KCL use case where
the parasitic NPN sits with floating base = body, collector = drain,
emitter = source. Junction caps (cje, cjc) and transit times (tf, tr, itf,
vtf, xtf) are intentionally omitted — they affect transient/AC dynamics, not
the DC body current that closes the body-KCL fixed point.

References: ngspice manual ch. on BJT; SPICE Gummel-Poon model equations.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch

from .constants import KboQ
from .smooth import safe_exp, safe_sqrt


@dataclass
class GummelPoonNPN:
    """Gummel-Poon NPN parameters (subset matching Sebas's parasiticBJT card)."""
    Is: float = 5e-9        # transport saturation current [A]
    Va: float = 100.0       # forward Early voltage [V]
    Vb: float = 1e30        # reverse Early voltage [V]   (Vaf/Var; default = no effect)
    Bf: float = 10000.0     # ideal max forward beta
    Br: float = 100.0       # ideal max reverse beta
    Nf: float = 1.0         # forward emission coefficient
    Nr: float = 1.0         # reverse emission coefficient
    Nc: float = 2.0         # base-collector leakage emission
    Ne: float = 1.5         # base-emitter leakage emission
    Ikf: float = 1e30       # forward knee (no high-injection if huge)
    Ikr: float = 0.1        # reverse knee   ← Sebas: 100m
    Ise: float = 0.0        # B-E leakage saturation
    Isc: float = 0.0        # B-C leakage saturation
    Re:  float = 0.1        # emitter ohmic    (DC: not iterated, kept for API)
    Rc:  float = 0.1        # collector ohmic  (DC: not iterated, kept for API)
    Rb:  float = 0.0        # base ohmic       (DC: not iterated, kept for API)
    area: float = 1.0       # SPICE BJT instance multiplier (scales Is, Ikf, Ikr, Ise, Isc)

    @classmethod
    def from_sebas_card(cls) -> "GummelPoonNPN":
        """Sebas's parasiticBJT.txt parameters (data/sebas_2026_04_22/parasiticBJT.txt).

        .model parasiticBJT NPN(is=5E-9 va=100 bf=10000 br=100 nc=2 ikr=100m
                                rc=0.1 vje=0.7 re=0.1 cjc=1e-15 fc=0.5
                                cje=0.7e-15 ne=1.5 ise=0
                                tr=20e-12 tf=25e-12 itf=0.03 vtf=7 xtf=2)

        Junction caps / transit times skipped (DC port).  Nf, Nr, Bf default in
        SPICE when not specified ⇒ Nf=Nr=1, Br given.  Ikf not in card ⇒ infinity.
        """
        return cls(
            Is=5e-9,
            Va=100.0,
            Bf=10000.0,
            Br=100.0,
            Nf=1.0,
            Nr=1.0,
            Nc=2.0,
            Ne=1.5,
            Ikf=1e30,
            Ikr=0.1,
            Ise=0.0,
            Isc=0.0,
            Re=0.1,
            Rc=0.1,
            Rb=0.0,
            area=1e-6,  # schematic 2tnsram_simple.asc: SYMATTR Value2 area=1u
        )


def compute_bjt(
    bjt: GummelPoonNPN,
    Vbe: torch.Tensor,
    Vbc: torch.Tensor,
    T_K: float = 300.15,
) -> dict[str, torch.Tensor]:
    """SPICE Gummel-Poon DC currents (no ohmic-resistance iteration, no caps).

    Returns dict with keys: 'Ic', 'Ib', 'Ie', 'Icc', 'Iec', 'kqb'.

    Equations (ngspice manual):
        vt   = k·T/q
        Icc  = Is·(exp(Vbe/(Nf·vt)) − 1)
        Iec  = Is·(exp(Vbc/(Nr·vt)) − 1)
        Ibe_n= Icc/Bf,           Ibe_l = Ise·(exp(Vbe/(Ne·vt)) − 1)
        Ibc_n= Iec/Br,           Ibc_l = Isc·(exp(Vbc/(Nc·vt)) − 1)
        q1   = 1/(1 − Vbc/Va − Vbe/Vb)
        q2   = Icc/Ikf + Iec/Ikr
        kqb  = (q1/2)·(1 + sqrt(1 + 4·q2))
        Ic   = (Icc − Iec)/kqb − Ibc_n − Ibc_l
        Ib   = Ibe_n + Ibe_l + Ibc_n + Ibc_l
        Ie   = −(Ic + Ib)

    Sign convention: NPN — Vbe>0 forward biases B-E; current flows from C to E.
    For the NS-RAM parasitic NPN (floating base = body):
      Vbe = V_body − V_source
      Vbc = V_body − V_drain
    The collector current Ic is the "I_BJT" entering the body-KCL.

    fp64 throughout; uses safe_exp (clamped ±34) and safe_sqrt (eps=1e-12).
    """
    # promote to fp64 on whichever device the inputs live on
    Vbe = Vbe.to(torch.float64)
    Vbc = Vbc.to(torch.float64)

    vt = torch.as_tensor(KboQ * T_K, dtype=torch.float64, device=Vbe.device)
    # SPICE BJT area multiplier: scales Is, Ikf, Ikr, Ise, Isc (and divides
    # Rb/Re/Rc, but those aren't iterated in this DC port).
    area_ = torch.as_tensor(bjt.area, dtype=torch.float64, device=Vbe.device)
    Is_ = torch.as_tensor(bjt.Is, dtype=torch.float64, device=Vbe.device) * area_
    Ise_ = bjt.Ise * area_
    Isc_ = bjt.Isc * area_
    Ikf_ = bjt.Ikf * area_
    Ikr_ = bjt.Ikr * area_

    # -- Transport currents (block: Icc / Iec) ---------------------------------
    Icc = Is_ * (safe_exp(Vbe / (bjt.Nf * vt)) - 1.0)
    Iec = Is_ * (safe_exp(Vbc / (bjt.Nr * vt)) - 1.0)

    # -- Base currents (block: ideal + non-ideal leakage) ----------------------
    Ibe_n = Icc / bjt.Bf
    Ibc_n = Iec / bjt.Br
    if bjt.Ise > 0.0:
        Ibe_l = Ise_ * (safe_exp(Vbe / (bjt.Ne * vt)) - 1.0)
    else:
        Ibe_l = torch.zeros_like(Vbe)
    if bjt.Isc > 0.0:
        Ibc_l = Isc_ * (safe_exp(Vbc / (bjt.Nc * vt)) - 1.0)
    else:
        Ibc_l = torch.zeros_like(Vbc)

    # -- Base-charge factor kqb (block: Early + high-injection knee) -----------
    # q1 includes Early-effect denominator.  Va and Vb (=Vaf/Var) are large
    # ⇒ q1 ≈ 1.  Guard the denominator with a soft floor so q1 stays finite
    # even when Vbc → Va (deep saturation).
    inv_q1 = 1.0 - Vbc / bjt.Va - Vbe / bjt.Vb
    inv_q1 = inv_q1.clamp_min(1e-4)        # prevents divide-by-zero
    q1 = 1.0 / inv_q1

    # q2 = high-injection ratio.  Ikf and Ikr default huge ⇒ q2 ≈ 0.
    q2 = Icc / Ikf_ + Iec / Ikr_

    # kqb = (q1/2)·(1 + sqrt(1 + 4 q2)).  Using safe_sqrt on the discriminant
    # gives a smooth knee at q2≈0 with finite gradient; saturates as
    # kqb ~ q1·sqrt(q2)  for q2 ≫ 1, which damps Icc once high-injection hits.
    disc = 1.0 + 4.0 * q2
    kqb = 0.5 * q1 * (1.0 + safe_sqrt(disc))

    # -- Terminal currents -----------------------------------------------------
    Ic = (Icc - Iec) / kqb - Ibc_n - Ibc_l
    Ib = Ibe_n + Ibe_l + Ibc_n + Ibc_l
    Ie = -(Ic + Ib)

    return {
        "Ic": Ic, "Ib": Ib, "Ie": Ie,
        "Icc": Icc, "Iec": Iec, "kqb": kqb,
    }
