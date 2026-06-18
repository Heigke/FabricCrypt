"""nsram_cell — body-KCL wrapper for the NS-RAM 2T floating-body cell.

Integrates the BSIM4 port (DC + impact-ion + GIDL/GISL + Igb + body diodes +
junction caps) with a parasitic Gummel-Poon NPN and an external body-leak
resistor into a single differentiable cell model.

Body node KCL (positive = current INTO the body):

    C_body · dVb/dt =
        + Iii(Vb, Vd, VG1, VG2)           # impact-ionization, channel → body
        - Ibd(Vb, Vd)                     # body→drain diode (Ibd>0 forward → leaves body)
        - Ibs(Vb, 0)                      # body→source diode (likewise)
        - Igidl(Vb, Vd, VG1)              # GIDL: drain edge band-to-band
        - Igisl(Vb, VG1)                  # GISL: source edge band-to-band
        + Igb(VG1, Vb)                    # gate→body tunneling (positive ⇒ enters body)
        - I_BJT(Vb, Ve=0, Vc=Vd)          # parasitic NPN collector current
        - Vb / Rb_leak                    # external resistor to ground

Body capacitance:
    C_body = Cjs + Cjd + C_extra

VG2 (bottom gate / well) is treated, by NS-RAM convention, as a bias that
shifts the effective threshold of the BSIM4 top transistor:
    vth0_eff(VG2) = vth0_T + gamma_VG2 · VG2          (default gamma_VG2 = 0.3)

This is a single-knob proxy for the second-gate body-effect coupling — VG2
thus selects between the three regimes (BISTABLE / SOFT / INTEGRATOR) by
moving the impact-ionization knee.

Sign convention for the diode block (compute_body_diodes):
    Ibs, Ibd as returned are positive when the junction is forward biased
    (Vbs>0 / Vbd>0), corresponding to current flowing OUT of the body. We
    subtract them from I_into_body.

The whole `kcl_body` is differentiable end-to-end (no Python branches on
tensor values) and verified by torch.autograd.gradcheck in the tests.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch

from .bjt import GummelPoonNPN, compute_bjt
from .caps import compute_caps
from .dc import compute_dc
from .diode import compute_body_diodes
from .geometry import Geometry
from .leak import compute_iimpact, compute_igidl_gisl, compute_igb
from .model_card import BSIM4Model
from .temp import compute_size_dep, SizeDependParam


# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class NSRAMCellConfig:
    """Static config for one NS-RAM cell.

    Holds the BSIM4 model, geometry, parasitic NPN and a few cell-level
    knobs. The temperature-dependent SizeDependParam is computed lazily and
    cached on first use.
    """
    bsim4_model: BSIM4Model
    geometry: Geometry
    bjt_params: GummelPoonNPN
    Rb_leak: float = 5e8         # external body-leak resistor [Ω]
    C_extra: float = 0.0         # extra body cap [F]
    T_C: float = 27.0            # operating temp [°C]
    gamma_VG2: float = 0.3       # NS-RAM convention: VG2 → vth0 shift
    # Junction geometry. None = auto-default to W·L (area) and 2·(W+L) (perim);
    # explicit 0.0 = caller wants zero junction caps (rarely correct).
    # WAVE2-FIX (2026-04-29 critique): defaulting to 0 silently zeros junction
    # caps and makes Cbody dominated by abs(Cgb), which is physically wrong for
    # NS-RAM retention. Auto-default uses geometry W·L estimate when None.
    As: Optional[float] = None   # source bottom area [m²]; None → W·L
    Ad: Optional[float] = None   # drain  bottom area [m²]; None → W·L
    Ps: Optional[float] = None   # source perimeter   [m];  None → 2·(W+L)
    Pd: Optional[float] = None   # drain  perimeter   [m];  None → 2·(W+L)
    # Toggle bits (defaults match physical NS-RAM cell)
    use_iii: bool = True
    use_gidl: bool = True
    use_igb: bool = True
    use_diode: bool = True
    use_bjt: bool = True
    # Optional NS-RAM-specific lateral-BJT punch-through body-charging term.
    # BSIM4 doesn't model lateral-BJT punch-through avalanche at low Vd, but
    # measurements of NS-RAM 2T cells (Pazos+, 130nm) clearly show snapback
    # at Vd ≈ 0.7-1V — far below where standard BSIM4 alpha0/beta0=18 would
    # trigger. SOI-style extension; standard practice when modelling
    # floating-body cells with BSIM4 (alternative: switch to BSIM-SOI).
    #
    # Vb-COUPLED form (P7v4): trigger threshold lowered by accumulated body
    # voltage — captures the regenerative loop where body charge advances
    # the snapback voltage:
    #     I_PT = I_PT0 · scale · softplus((Vd + k_Vb·Vb − V_PT_th) / scale)
    # k_Vb > 0 means: more positive Vb → effective drive higher → triggers
    # at lower Vd. This is what makes VG2 (which sets Vb steady state)
    # actually shift the snapback voltage in the model.
    use_punchthrough: bool = False
    I_PT0: float = 1.0e-6      # punch-through pre-factor [A]
    V_PT_th: float = 0.7       # trigger Vd [V]
    V_PT_scale: float = 0.05   # ramp sharpness [V]
    k_Vb_PT: float = 0.0       # Vb coupling strength [unitless, ≥0]
    # Cached SizeDependParam (lazy)
    _sd: Optional[SizeDependParam] = field(default=None, init=False, repr=False)

    def size_dep(self) -> SizeDependParam:
        if self._sd is None:
            self._sd = compute_size_dep(self.bsim4_model, self.geometry,
                                        T_C=self.T_C)
        return self._sd

    def invalidate(self) -> None:
        """Force a re-compute of the SizeDependParam cache (after editing the
        model card)."""
        self._sd = None


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #

@contextmanager
def _vth0_shifted(sd: SizeDependParam, vth0_eff):
    """Temporarily replace sd.vth0_T with `vth0_eff` (tensor or float).

    compute_dc reads `sd.vth0_T` once via `t(sd.vth0_T)` (as_tensor); passing
    a fp64 tensor flows gradients through. We restore the original on exit.
    """
    saved = sd.vth0_T
    sd.vth0_T = vth0_eff
    try:
        yield
    finally:
        sd.vth0_T = saved


def _as_t(x, ref: torch.Tensor) -> torch.Tensor:
    """Coerce x to fp64 tensor on the same device as ref."""
    return torch.as_tensor(x, dtype=torch.float64, device=ref.device)


# --------------------------------------------------------------------------- #
# Body-KCL                                                                    #
# --------------------------------------------------------------------------- #

def kcl_body(
    cfg: NSRAMCellConfig,
    Vb: torch.Tensor,
    Vd: torch.Tensor,
    VG1: torch.Tensor,
    VG2: torch.Tensor,
    *,
    use_iii: Optional[bool] = None,
    use_gidl: Optional[bool] = None,
    use_igb: Optional[bool] = None,
    use_diode: Optional[bool] = None,
    use_bjt: Optional[bool] = None,
) -> dict[str, torch.Tensor]:
    """Compute body-KCL residual for the NS-RAM cell.

    All bias inputs are torch.Tensors (fp64). Sign convention: positive
    currents flow INTO the body.

    Returns:
        dict with
          - 'I_total'    [A]   sum of currents into body
          - 'C_body'     [F]   total body capacitance (Cjs + Cjd + C_extra)
          - 'dVb_dt'     [V/s] = I_total / C_body
          - 'components' dict of individual currents
    """
    # Resolve toggles (per-call override > cfg default)
    use_iii = cfg.use_iii if use_iii is None else use_iii
    use_gidl = cfg.use_gidl if use_gidl is None else use_gidl
    use_igb = cfg.use_igb if use_igb is None else use_igb
    use_diode = cfg.use_diode if use_diode is None else use_diode
    use_bjt = cfg.use_bjt if use_bjt is None else use_bjt

    sd = cfg.size_dep()
    model = cfg.bsim4_model

    # fp64 + broadcast all four biases to a common shape.
    Vb_t = Vb.to(torch.float64)
    Vd_t = _as_t(Vd, Vb_t)
    VG1_t = _as_t(VG1, Vb_t)
    VG2_t = _as_t(VG2, Vb_t)
    Vb_t, Vd_t, VG1_t, VG2_t = torch.broadcast_tensors(Vb_t, Vd_t, VG1_t, VG2_t)

    # Standard NS-RAM bias mapping
    Vgs = VG1_t                       # top gate voltage relative to source=0
    Vds = Vd_t
    Vbs = Vb_t                        # body relative to source=0
    Vbd = Vb_t - Vd_t

    # VG2 → vth0 shift (NS-RAM convention)
    vth0_T0 = sd.vth0_T
    if not isinstance(vth0_T0, torch.Tensor):
        vth0_T0 = torch.as_tensor(vth0_T0, dtype=torch.float64,
                                  device=Vb_t.device)
    vth0_eff = vth0_T0 + cfg.gamma_VG2 * VG2_t

    # ------- DC drain current + intermediates needed for Iii / caps ------- #
    with _vth0_shifted(sd, vth0_eff):
        dc = compute_dc(model, sd, Vgs=Vgs, Vds=Vds, Vbs=Vbs)

        # Impact-ionization (uses Vdseff from dc)
        if use_iii:
            Iii = compute_iimpact(model, sd, dc, Vds=Vds)
        else:
            Iii = torch.zeros_like(Vb_t)

        # GIDL / GISL
        if use_gidl:
            Igidl, Igisl = compute_igidl_gisl(model, sd, Vgs=Vgs, Vds=Vds, Vbs=Vbs)
        else:
            Igidl = torch.zeros_like(Vb_t)
            Igisl = torch.zeros_like(Vb_t)

        # Gate-to-body tunneling
        if use_igb:
            Igb = compute_igb(model, sd, Vgs=Vgs, Vbs=Vbs, dc_result=dc)
        else:
            Igb = torch.zeros_like(Vb_t)

        # WAVE2-FIX (2026-04-29 critique): auto-default junction geometries
        # to W·L (area) and 2·(W+L) (perimeter) when None. Defaulting to 0
        # silently zeros junction caps and produces wrong NS-RAM retention.
        W = cfg.geometry.W
        L = cfg.geometry.L
        WL = W * L
        WLp = 2.0 * (W + L)
        As_eff = WL  if cfg.As is None else cfg.As
        Ad_eff = WL  if cfg.Ad is None else cfg.Ad
        Ps_eff = WLp if cfg.Ps is None else cfg.Ps
        Pd_eff = WLp if cfg.Pd is None else cfg.Pd

        # Body diodes (positive when forward-biased = leaving body).
        if use_diode:
            Ibs, Ibd = compute_body_diodes(model, sd, Vbs=Vbs, Vbd=Vbd,
                                           As=As_eff, Ad=Ad_eff,
                                           Ps=Ps_eff, Pd=Pd_eff)
        else:
            Ibs = torch.zeros_like(Vb_t)
            Ibd = torch.zeros_like(Vb_t)

        # Junction capacitances (Cjs + Cjd) — depend on Vbs, Vbd.
        cap = compute_caps(model, sd, dc, Vgs=Vgs, Vds=Vds, Vbs=Vbs, Vbd=Vbd,
                           As=As_eff, Ad=Ad_eff, Ps=Ps_eff, Pd=Pd_eff)

    # Parasitic NPN: floating base = body, emitter = source(=0), collector = drain.
    # Vbe = Vb - Vsrc = Vb;   Vbc = Vb - Vd
    #
    # WAVE2-FIX (2026-04-29 critique): use Ib (base current), NOT Ic. The
    # body IS the base node — Ic flows collector→emitter through the BJT
    # transport mechanism, it does NOT terminate at the body. Only the base
    # current Ib (recombination + minority injection) charges/discharges the
    # body. Previous version used Ic, which with Bf=10000 drained the body
    # 10000× too fast in the BJT-on regime → wrong NS-RAM retention.
    if use_bjt:
        bjt = compute_bjt(cfg.bjt_params, Vbe=Vb_t, Vbc=Vbd,
                          T_K=273.15 + cfg.T_C)
        # SPICE NPN convention: Ib positive = INTO base from external circuit.
        # For floating-body, that current can only come from the body charge,
        # so Ib > 0 means the body is sourcing it (leaves body).
        Ib_bjt = bjt["Ib"]
    else:
        Ib_bjt = torch.zeros_like(Vb_t)

    # External body-leak resistor (to ground): I_leak_out = Vb / Rb_leak.
    Ileak_out = Vb_t / cfg.Rb_leak

    # Optional lateral-BJT punch-through body-charging term (NS-RAM specific).
    # P7v4 Vb-coupled form: I_PT depends on (Vd + k_Vb·Vb), so a charged body
    # advances the trigger → captures regenerative snapback dependence on VG2.
    if cfg.use_punchthrough:
        import torch.nn.functional as F
        V_scale_t = torch.as_tensor(cfg.V_PT_scale, dtype=Vd.dtype, device=Vd.device)
        k_Vb_t = torch.as_tensor(cfg.k_Vb_PT, dtype=Vd.dtype, device=Vd.device)
        drive = Vd + k_Vb_t * Vb_t - cfg.V_PT_th
        I_PT = (cfg.I_PT0 * V_scale_t
                * F.softplus(drive / V_scale_t.clamp_min(1e-6)))
    else:
        I_PT = torch.zeros_like(Vb_t)

    # ------- Sum into body (positive = entering body) ------- #
    # Iii    : channel → body                       → +
    # Ibd,Ibs: body → S/D when forward-biased        → -
    # Igidl  : drain → body when reverse-biased drain (it CHARGES body) but the
    #          model's sign for Igidl in BSIM4 is the magnitude; convention in
    #          ngspice is that Igidl flows OUT of the drain INTO the body, i.e.
    #          enters the body. We follow the diff_canonical convention: enter.
    # Igb    : gate → body                          → +
    # I_BJT  : The body IS the base of the parasitic NPN. Ib (positive into
    #          base in SPICE) drains the body when BE forward-biased (carriers
    #          recombine in base region). Ic flows collector→emitter through
    #          the device and does NOT terminate at the body — it must NOT
    #          enter body-KCL.
    # Ileak  : Vb / Rb_leak leaves the body          → -

    # Sign convention: I_total > 0 means net current ENTERING body.
    # Per b4ld.c §443 Ibtot = cbs+cbd - Igidl - Igisl - csub  (leaving body),
    # so currents ENTERING body are: Iii (csub), Igidl, Igisl, Igb;
    # currents LEAVING body are: Ibd, Ibs (junction diodes), Ib_bjt, Ileak.
    I_total = (
        Iii                       # +  (impact ionization → body)
        - Ibd                     # -  (body→drain diode)
        - Ibs                     # -  (body→source diode)
        + Igidl                   # +  (drain→body via GIDL)
        + Igisl                   # +  (source→body via GISL)
        + Igb                     # +  (gate→body tunneling)
        - Ib_bjt                  # -  (parasitic BJT base current; was Ic — bug)
        - Ileak_out               # -  (external Rb_leak)
        + I_PT                    # +  (lateral-BJT punch-through, if enabled)
    )

    # Body capacitance — include |Cgb| (gate-bulk intrinsic).
    # In accumulation (low Vgs) Cgb ≈ -CoxWL, dominating the NS-RAM hold regime.
    # BSIM4's sign convention can give negative Cgb (rate of Qb wrt Vg);
    # the contribution to total body cap is its magnitude.
    C_body = cap.Cjs + cap.Cjd + torch.abs(cap.Cgb) + cfg.C_extra
    # Guard against pathological zero (junctions all-zero geometry → tiny floor)
    C_body = C_body + 1e-30

    dVb_dt = I_total / C_body

    components = {
        "Iii": Iii,
        "Ibd": -Ibd,         # signed-into-body for clarity
        "Ibs": -Ibs,
        "Igidl": Igidl,      # enters body
        "Igisl": Igisl,      # enters body
        "Igb": Igb,
        "Ibjt": -Ib_bjt,
        "Ileak": -Ileak_out,
    }
    return {
        "I_total": I_total,
        "C_body": C_body,
        "dVb_dt": dVb_dt,
        "components": components,
    }


# --------------------------------------------------------------------------- #
# Transient                                                                   #
# --------------------------------------------------------------------------- #

def transient_step(
    cfg: NSRAMCellConfig,
    Vb: torch.Tensor,
    Vd: torch.Tensor,
    VG1: torch.Tensor,
    VG2: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """One forward-Euler step.  Returns Vb_new = Vb + dt · dVb/dt."""
    f = kcl_body(cfg, Vb, Vd, VG1, VG2)
    return Vb + dt * f["dVb_dt"]


def transient(
    cfg: NSRAMCellConfig,
    Vb0: torch.Tensor,
    Vd_seq: torch.Tensor,
    VG1: torch.Tensor,
    VG2: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Forward-Euler transient.

    Args:
        Vb0:    initial body voltage (scalar tensor)
        Vd_seq: drain bias trajectory, shape (T,)
        VG1, VG2: scalar tensors (constant over the transient)
        dt:     time step [s]

    Returns:
        Vb_history of shape (T,) — Vb after applying each Vd_seq[i].
    """
    Vd_seq = Vd_seq.to(torch.float64)
    T = int(Vd_seq.shape[0])
    Vb = Vb0.to(torch.float64)
    out = []
    for i in range(T):
        Vb = transient_step(cfg, Vb, Vd_seq[i], VG1, VG2, dt)
        out.append(Vb)
    return torch.stack(out, dim=0)
