"""nsram_cell_2T — Differentiable 2T NS-RAM cell with proper topology.

Replaces the 1T proxy in `nsram_cell.py` (which collapses VG2 into a
``vth0_eff = vth0 + gamma·VG2`` shift) with the FULL 2T topology faithful
to Sebas's schematic ``data/sebas_2026_04_22/2tnsram_simple.asc``::

        D ──┬─────────────┬── (drain pin)
            │             │
          M1.D          Q1.C
            │             │
   VG1 → M1.G           Q1.B ── B  (floating body, shared by M1 & M2)
            │             │
          M1.S ── Sint ── Q1.E
                    │
                  M2.D
                    │
   VG2 → M2.G       │
                    │
                  M2.S ── 0  (ground)

Two NMOS (M1 short, M2 long) share floating body B. The internal node
Sint is the M1 source / M2 drain / Q1 emitter. Two unknown internal
voltages (Vsint, Vb) are solved by Newton-Raphson at each (Vd, VG1, VG2)
bias point so Sint-KCL = 0 and Body-KCL = 0.

Newton residuals (currents INTO each node):

    R_Sint(Vsint, Vb) =
        + Ids_M1(VG1−Vsint, Vd−Vsint, Vb−Vsint)            # M1 source ejects into Sint
        − Ids_M2(VG2,         Vsint,    Vb)                 # M2 drain absorbs from Sint
        + Ie_Q1(Vb−Vsint, Vb−Vd)                            # BJT emitter ejects into Sint
        + Ibs_diode_M1(Vb−Vsint)                            # forward body→Sint diode of M1
        − Ibd_diode_M2(Vb)                                  # forward body→drain(=Sint) of M2 leaves Sint

    R_B(Vsint, Vb) =
        + Iii_M1 + Iii_M2                                   # impact-ion holes → body
        + Igidl_M1 + Igisl_M1 + Igidl_M2 + Igisl_M2         # BTBT
        + Igb_M1 + Igb_M2                                   # gate→body tunnel
        − Ibd_diode_M1(Vb−Vd) − Ibs_diode_M1(Vb−Vsint)      # M1 junction leaks LEAVE body
        − Ibd_diode_M2(Vb)    − Ibs_diode_M2(Vb)            # M2 junction leaks LEAVE body
        − Ib_Q1(Vb−Vsint, Vb−Vd)                            # BJT base current leaves B

Drain terminal current at the D pin (positive into device):
    Id = Ids_M1 + Ic_Q1 + Igidl_drain_M1 + Ibd_diode_M1

VG2 is now a *real* gate to M2 (not a proxy threshold shift); body-effect
on M1 enters naturally via Vbs_M1 = Vb − Vsint.

Differentiability: simplest correct path. Newton iterations live INSIDE
autograd (no implicit-function-theorem trick yet). Each iteration is a
single forward of the full BSIM4 stack (~30 calls per bias point worst
case, double precision). For 33×~10 sweep points that's still tractable.

WARNING: do NOT add arbitrary clipping to "fix" Newton divergence — that
was the v4 mistake. Diagnose with `verbose=True`.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch

from .bjt import GummelPoonNPN, compute_bjt
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
class NSRAMCell2TConfig:
    """Static config for a 2T NS-RAM cell.

    Geometry + toggles + Newton solver knobs. Two SizeDependParam objects
    (one per MOSFET) are computed lazily.
    """
    Ln: float = 180e-9                  # M1 channel length [m]
    Wn: float = 360e-9                  # both channels' width [m]
    M2_length_factor: float = 10.0      # M2 length = Ln * factor (Sebas: 10x)
    Cbody: float = 1e-15                # body cap [F] (transient only; from CBpar)
    T_C: float = 27.0                   # operating temperature

    # Junction geometry per MOSFET. None → auto W·L / 2(W+L).
    As_M1: Optional[float] = None
    Ad_M1: Optional[float] = None
    Ps_M1: Optional[float] = None
    Pd_M1: Optional[float] = None
    As_M2: Optional[float] = None
    Ad_M2: Optional[float] = None
    Ps_M2: Optional[float] = None
    Pd_M2: Optional[float] = None

    # Toggle physics
    use_iii: bool = True
    use_gidl: bool = True
    use_bjt: bool = True
    use_igb: bool = True
    use_diode: bool = True

    # Deep-N-well bias on M1 (130nm DNWFB device).
    # ──────────────────────────────────────────────────────────────────
    # IMPORTANT (2026-05-01, A.1.n finding): Sebas's measurement and
    # SPICE deck apply +2 V to the deep-N-well terminal of M1. The
    # well/body PN junction is forward-biased (well at +2 V vs floating
    # body at ~0 V), pumping current into the body. THIS is the missing
    # body-charging path that explains our 5-decade Id under-prediction
    # at low VG2. The schematic doesn't show the well node; the bias is
    # applied externally on the package pin, with the well capacitance
    # and series resistance internal.
    use_well_diode: bool = True
    vnwell: float = 2.0              # deep-N-well voltage [V]
    # Series-R production default = 1e9 Ω. Grid search (A.2 z91h_grid)
    # found tighter median at Rs=3e9 (0.79) and Rs=1e10 (0.69) but those
    # come with coverage loss — the arclength solver loses the snapback
    # fold mid-trace when vnwell coupling is strong. A.1.s solver work
    # (dual-direction sweep, branch detection) needed to unlock those
    # settings. Until then, Rs=1e9 gives the full 25/25 coverage at
    # honest median 1.19 / p90 2.88.
    vnwell_Rs: float = 1.0e10
    vnwell_area: float = 1.0e-12     # well-body junction area [m²] (~1 µm²; tiny)
    vnwell_Js: float = 3.4089e-7     # saturation current density [A/m²] (jss)
    vnwell_n: float = 1.017          # diode emission factor (njs)
    # mbjt-tracking: Sebas's CSV mbjt column scales the parasitic-NPN
    # area; physically the well-body junction belongs to the same parasitic
    # bipolar structure, so it should track the same multiplier. At
    # mbjt=0.001 (VG1=0.2 in his data) the well coupling effectively
    # disappears; at mbjt=1.0 (VG1=0.4/0.6) it's fully present.
    vnwell_mbjt: float = 1.0
    # Physical defaults injected when card has jss=jsd=0 (Sebas's PTM130 card
    # leaves these unset, which leaves the body diodes silent and lets Vb run
    # away unbounded under Iii injection — root cause of v6 fit explosion).
    # Typical 130nm CMOS pn junction: Js ≈ 1e-4 A/m². With AS = W·L = 360n·180n
    # = 6.5e-14 m², Is_diode ≈ 6.5e-18 A; at Vbs = 0.7V forward, Ibs ≈ 1.1e-5 A
    # → naturally clamps Vb at body-source diode turn-on voltage.
    default_jss: float = 1e-4    # A/m² source-bottom junction
    default_jsd: float = 1e-4    # A/m² drain-bottom junction

    # Newton solver
    newton_max_iters: int = 30
    newton_tol: float = 1e-12        # max(|R_Sint|, |R_B|) in Amperes (legacy)
    newton_damping: float = 1.0
    newton_min_damping: float = 1.0 / 64.0
    # Per-iteration relative voltage step cap (helps in steep regions w/o
    # masking divergence). Set to a large number to disable. Keep modest;
    # purpose is convergence, not "papering over" non-physics.
    max_step_V: float = 0.5

    # Oracle-recommended Newton hardening (gmin shunt + relative tol +
    # min-iter guard prevents the "spurious-root at iter 1" pathology where
    # Vb=0 initial guess lands all body currents at ~1e-17 A which is below
    # the absolute residual tolerance even though the true root is at
    # Vb~0.77 V).
    gmin: float = 1e-15              # shunt conductance on body+Sint KCL
    # Lowered from oracle-suggested 1e-12: at 1e-12 gmin shunts dominate
    # over physically zero body diodes (jss=jsd=0 in Sebas card) and pull
    # Vb to Vd/4, forward-biasing M1's body-source junction and doubling
    # Id. 1e-15 is small enough not to distort while still providing the
    # Jacobian slope to escape the spurious flat root at Vb=0.
    Iabstol: float = 1e-12           # absolute current tolerance
    Ireltol: float = 1e-3            # relative tolerance vs |I_physical|
    xtol_v: float = 1e-7             # voltage step infinity-norm tolerance
    min_iters: int = 2               # require >= this many Newton iters
    # gmin homotopy: if enabled, first cold-start solve walks gmin from
    # gmin_start down to `gmin` in factor-of-10 steps before declaring done.
    gmin_step: bool = False
    gmin_start: float = 1e-9

    # Lazy SizeDependParam caches
    _sd_M1: Optional[SizeDependParam] = field(default=None, init=False, repr=False)
    _sd_M2: Optional[SizeDependParam] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ #
    def _geom_M1(self) -> Geometry:
        return Geometry(L=self.Ln, W=self.Wn, NF=1)

    def _geom_M2(self) -> Geometry:
        return Geometry(L=self.Ln * self.M2_length_factor, W=self.Wn, NF=1)

    def size_dep_M1(self, model: BSIM4Model) -> SizeDependParam:
        if self._sd_M1 is None:
            self._sd_M1 = compute_size_dep(model, self._geom_M1(), T_C=self.T_C)
        return self._sd_M1

    def size_dep_M2(self, model: BSIM4Model) -> SizeDependParam:
        if self._sd_M2 is None:
            self._sd_M2 = compute_size_dep(model, self._geom_M2(), T_C=self.T_C)
        return self._sd_M2

    def invalidate(self) -> None:
        self._sd_M1 = None
        self._sd_M2 = None

    def _junctions_M1(self) -> tuple[float, float, float, float]:
        WL = self.Wn * self.Ln
        WLp = 2.0 * (self.Wn + self.Ln)
        return (
            WL  if self.As_M1 is None else self.As_M1,
            WL  if self.Ad_M1 is None else self.Ad_M1,
            WLp if self.Ps_M1 is None else self.Ps_M1,
            WLp if self.Pd_M1 is None else self.Pd_M1,
        )

    def _junctions_M2(self) -> tuple[float, float, float, float]:
        L2 = self.Ln * self.M2_length_factor
        WL = self.Wn * L2
        WLp = 2.0 * (self.Wn + L2)
        return (
            WL  if self.As_M2 is None else self.As_M2,
            WL  if self.Ad_M2 is None else self.Ad_M2,
            WLp if self.Ps_M2 is None else self.Ps_M2,
            WLp if self.Pd_M2 is None else self.Pd_M2,
        )


# --------------------------------------------------------------------------- #
# Param-override context for SizeDependParam                                  #
# --------------------------------------------------------------------------- #

@contextmanager
def _override_sd(sd: SizeDependParam, overrides: Optional[dict]):
    """Temporarily replace selected SizeDependParam fields (for fitting).

    Useful so optimizer can flow grads through ``sd.vth0_T`` etc. without
    rebuilding the whole SizeDependParam each iteration.
    """
    if not overrides:
        yield
        return
    saved: dict = {}
    try:
        for k, v in overrides.items():
            saved[k] = getattr(sd, k)
            setattr(sd, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(sd, k, v)


# --------------------------------------------------------------------------- #
# Per-MOSFET physics evaluator                                                #
# --------------------------------------------------------------------------- #

def _eval_mosfet(
    model: BSIM4Model,
    sd: SizeDependParam,
    cfg: NSRAMCell2TConfig,
    Vg: torch.Tensor,
    Vd: torch.Tensor,
    Vs: torch.Tensor,
    Vb: torch.Tensor,
    junctions: tuple[float, float, float, float],
    overrides: Optional[dict] = None,
) -> dict:
    """Compute Ids, Iii, Igidl, Igisl, Igb, Ibs, Ibd for one NMOS at given
    *terminal* voltages. Bias mapping (NMOS, source-referenced):

        Vgs = Vg - Vs,   Vds = Vd - Vs,   Vbs = Vb - Vs,   Vbd = Vb - Vd

    Returned dict uses the convention native to each sub-call:
        - Ids: drain-to-source channel current (positive in saturation, NMOS)
        - Iii: positive INTO body (channel impact-ion)
        - Igidl: positive INTO body (drain edge BTBT, "drain → body")
        - Igisl: positive INTO body (source edge BTBT)
        - Igb: positive INTO body (gate → body tunneling)
        - Ibs, Ibd: junction diode currents, *positive when forward biased*
                   (current flows OUT of body INTO source/drain).
    """
    Vgs = Vg - Vs
    Vds = Vd - Vs
    Vbs = Vb - Vs
    Vbd = Vb - Vd

    with _override_sd(sd, overrides):
        dc = compute_dc(model, sd, Vgs=Vgs, Vds=Vds, Vbs=Vbs)

        if cfg.use_iii:
            Iii = compute_iimpact(model, sd, dc, Vds=Vds)
        else:
            Iii = torch.zeros_like(dc.Ids)

        if cfg.use_gidl:
            Igidl, Igisl = compute_igidl_gisl(model, sd, Vgs=Vgs, Vds=Vds, Vbs=Vbs)
        else:
            Igidl = torch.zeros_like(dc.Ids)
            Igisl = torch.zeros_like(dc.Ids)

        if cfg.use_igb:
            Igb = compute_igb(model, sd, Vgs=Vgs, Vbs=Vbs, dc_result=dc)
        else:
            Igb = torch.zeros_like(dc.Ids)

        if cfg.use_diode:
            As_, Ad_, Ps_, Pd_ = junctions
            # Inject physical Js defaults when card has zero (Sebas card
            # bug — root cause of Vb runaway). See cfg comments.
            js_overrides = {}
            try:
                if float(sd.SourceSatCurDensity_T) == 0.0 and cfg.default_jss > 0:
                    js_overrides["SourceSatCurDensity_T"] = cfg.default_jss
                if float(sd.DrainSatCurDensity_T) == 0.0 and cfg.default_jsd > 0:
                    js_overrides["DrainSatCurDensity_T"] = cfg.default_jsd
            except Exception:
                pass
            if js_overrides:
                with _override_sd(sd, js_overrides):
                    Ibs, Ibd = compute_body_diodes(model, sd, Vbs=Vbs, Vbd=Vbd,
                                                   As=As_, Ad=Ad_, Ps=Ps_, Pd=Pd_)
            else:
                Ibs, Ibd = compute_body_diodes(model, sd, Vbs=Vbs, Vbd=Vbd,
                                               As=As_, Ad=Ad_, Ps=Ps_, Pd=Pd_)
        else:
            Ibs = torch.zeros_like(dc.Ids)
            Ibd = torch.zeros_like(dc.Ids)

    return {
        "Ids": dc.Ids,
        "Iii": Iii,
        "Igidl": Igidl,
        "Igisl": Igisl,
        "Igb": Igb,
        "Ibs": Ibs,         # >0 ⇒ leaves body INTO source
        "Ibd": Ibd,         # >0 ⇒ leaves body INTO drain
        "Vds": Vds,
        "Vbs": Vbs,
        "Vbd": Vbd,
    }


# --------------------------------------------------------------------------- #
# Residual                                                                    #
# --------------------------------------------------------------------------- #

def _residuals(
    cfg: NSRAMCell2TConfig,
    model_M1: BSIM4Model,
    bjt: GummelPoonNPN,
    Vd: torch.Tensor,
    VG1: torch.Tensor,
    VG2: torch.Tensor,
    Vsint: torch.Tensor,
    Vb: torch.Tensor,
    P_M1: Optional[dict] = None,
    P_M2: Optional[dict] = None,
    model_M2: Optional[BSIM4Model] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Compute (R_Sint, R_B, components) at current (Vsint, Vb) guess.

    `model_M2` defaults to `model_M1` (single-model legacy behaviour).
    """
    if model_M2 is None:
        model_M2 = model_M1
    sd_M1 = cfg.size_dep_M1(model_M1)
    sd_M2 = cfg.size_dep_M2(model_M2)
    j_M1 = cfg._junctions_M1()
    j_M2 = cfg._junctions_M2()

    # Ground reference is V_S (source of M2) = 0.
    zero = torch.zeros_like(Vd)

    # M1: D=Vd, G=VG1, S=Vsint, B=Vb
    m1 = _eval_mosfet(model_M1, sd_M1, cfg, Vg=VG1, Vd=Vd, Vs=Vsint, Vb=Vb,
                      junctions=j_M1, overrides=P_M1)
    # M2: D=Vsint, G=VG2, S=0, B=Vb
    m2 = _eval_mosfet(model_M2, sd_M2, cfg, Vg=VG2, Vd=Vsint, Vs=zero, Vb=Vb,
                      junctions=j_M2, overrides=P_M2)

    # Parasitic NPN: collector=D, base=B, emitter=GND.
    # ──────────────────────────────────────────────────────────────────
    # IMPORTANT (2026-05-01, A.1.i finding): Sebastian's LTSpice schematic
    # `2tnsram_simple.asc` wires the parasitic NPN with **emitter to
    # ground**, not to Sint. This is the "complementary bipolar current"
    # he refers to in his Apr-17 email — its purpose is to provide a
    # body-charging path that fires when Vb climbs (Vbe = Vb − 0 = Vb,
    # not Vb − Vsint ≈ small). With emitter=Sint the BJT would never
    # turn on at low VG2 because Vb tracks Vsint. With emitter=GND, Vbe
    # tracks Vb directly and the NPN switches at Vb ~0.6 V.
    if cfg.use_bjt:
        Vbe = Vb                 # emitter = ground
        Vbc = Vb - Vd            # collector = drain
        bjt_out = compute_bjt(bjt, Vbe=Vbe, Vbc=Vbc, T_K=273.15 + cfg.T_C)
        Ic_Q1 = bjt_out["Ic"]    # collector current (drain → emitter = GND)
        Ib_Q1 = bjt_out["Ib"]    # base current (INTO base from external)
        Ie_Q1 = bjt_out["Ie"]    # emitter current at GND (= −(Ic+Ib))
    else:
        Ic_Q1 = torch.zeros_like(Vd)
        Ib_Q1 = torch.zeros_like(Vd)
        Ie_Q1 = torch.zeros_like(Vd)

    # ---- Sint KCL: currents INTO Sint --------------------------------- #
    # M1 channel current Ids_M1 flows D→S — INTO Sint (M1 source). → +Ids_M1
    # M2 drain is Sint; M2 channel sinks current FROM drain → −Ids_M2
    # BJT emitter is now GND, NOT Sint — BJT no longer touches Sint node.
    # M1 junction: Ibs_M1 >0 ⇒ leaves body INTO source(=Sint). → +Ibs_M1
    # M2 junction: Ibd_M2 >0 ⇒ leaves body INTO drain(=Sint). → +Ibd_M2
    R_Sint = (
        m1["Ids"]
        - m2["Ids"]
        + m1["Ibs"]
        + m2["Ibd"]
    )

    # Deep-N-well to body diode (A.1.n: this is the missing body-charging path).
    # ──────────────────────────────────────────────────────────────────
    # When vnwell > Vb, the N-well/P-body junction forward-biases and pumps
    # current INTO the body. Modelled as a Shockley diode with series R:
    #
    #     I_ideal  = Js·A · (exp((vnwell − Vb)/(n·Vt)) − 1)
    #     I_Rs     = (vnwell − Vb) / Rs   (when forward biased)
    #     I_well_b = harmonic_mean(I_ideal, I_Rs)   smooth transition
    #
    # Reverse-bias contribution is tiny (Js·A ~1e-15 A) — included for
    # completeness so derivatives are continuous through Vb crossing vnwell.
    if cfg.use_well_diode:
        Vt = 0.02585 * (273.15 + cfg.T_C) / 300.0   # thermal voltage at T
        V_drive = cfg.vnwell - Vb
        # Clamp exponent to avoid overflow when V_drive >> Vt
        exp_arg = (V_drive / (cfg.vnwell_n * Vt)).clamp(max=40.0)
        I_ideal = cfg.vnwell_Js * cfg.vnwell_area * (torch.exp(exp_arg) - 1.0)
        # Series-R limited current (only forward; reverse bias = 0 here)
        I_Rs = torch.relu(V_drive) / cfg.vnwell_Rs
        # Smooth min via harmonic mean (differentiable, transitions at the
        # smaller of the two without a hard kink)
        eps = 1e-30
        I_well_body = (I_ideal * I_Rs) / (I_ideal.abs() + I_Rs + eps)
        # Scale by mbjt — the well-body junction belongs to the same
        # parasitic bipolar structure as Q1, so it follows the same
        # device-multiplier. Without this scaling, VG1=0.2 (where
        # mbjt=0.001 keeps the BJT off) would still see full well
        # coupling and the body would float high.
        I_well_body = I_well_body * cfg.vnwell_mbjt
    else:
        I_well_body = torch.zeros_like(Vd)

    # ---- Body KCL: currents INTO B ------------------------------------ #
    # Iii, Igidl, Igisl, Igb are already signed +INTO-body in the helpers.
    # Body junction diodes: Ibs and Ibd are POSITIVE-LEAVING-body, so we
    # subtract them.
    # BJT base current Ib (positive INTO base from external) — for the
    # floating body, the only external current into the base IS the body
    # node itself. Ib>0 ⇒ body sources current → leaves body. → −Ib_Q1
    # Well-body diode I_well_body is +INTO body (well pumps body up). → +I_well_body
    R_B = (
        m1["Iii"] + m2["Iii"]
        + m1["Igidl"] + m1["Igisl"] + m2["Igidl"] + m2["Igisl"]
        + m1["Igb"] + m2["Igb"]
        - m1["Ibs"] - m1["Ibd"]
        - m2["Ibs"] - m2["Ibd"]
        - Ib_Q1
        + I_well_body
    )

    # Oracle-recommended gmin shunts — ngspice-style parallel conductance
    # in PARALLEL with each pn junction, NOT a single shunt to ground.
    # This is what gives the body a tendency to track (Vd+Vs)/2 in absence
    # of other forces, matching ngspice's behavior.
    #   I_gmin_bd = gmin * (Vd - Vb)   flows INTO body from drain
    #   I_gmin_bs = gmin * (Vs - Vb) = -gmin * Vb (since Vs=0)
    #                                   flows INTO body from source
    #   I_gmin_bsi = gmin * (Vsint - Vb)  body↔Sint via M1's body-source
    #                                      and M2's body-drain (both at Sint)
    # Sum into R_B (currents INTO B). Similar for Sint node.
    gmin = getattr(cfg, "gmin", 0.0)
    if gmin > 0.0:
        # Body node: junctions B↔D, B↔S(=0), B↔Sint (counted once: M1 body-source
        # and M2 body-drain are both at Sint, so 2× weight)
        R_B = R_B + gmin * (Vd - Vb) + gmin * (-Vb) + 2.0 * gmin * (Vsint - Vb)
        # Sint node: gmin shunt to ground (Sint↔S=0 via M2 channel parasitic)
        # plus to body. Mainly to keep Jacobian non-singular at Sint=0.
        R_Sint = R_Sint + gmin * (-Vsint) + 2.0 * gmin * (Vb - Vsint)

    components = {
        "Ids_M1": m1["Ids"], "Ids_M2": m2["Ids"],
        "Ic_Q1": Ic_Q1, "Ib_Q1": Ib_Q1, "Ie_Q1": Ie_Q1,
        "Iii_M1": m1["Iii"], "Iii_M2": m2["Iii"],
        "Igidl_M1": m1["Igidl"], "Igisl_M1": m1["Igisl"],
        "Igidl_M2": m2["Igidl"], "Igisl_M2": m2["Igisl"],
        "Igb_M1": m1["Igb"], "Igb_M2": m2["Igb"],
        "Ibs_M1": m1["Ibs"], "Ibd_M1": m1["Ibd"],
        "Ibs_M2": m2["Ibs"], "Ibd_M2": m2["Ibd"],
        "I_well_body": I_well_body,
    }
    return R_Sint, R_B, components


# --------------------------------------------------------------------------- #
# Newton solve                                                                #
# --------------------------------------------------------------------------- #

def _solve_jac_2x2(R_S: torch.Tensor, R_B: torch.Tensor,
                   J: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve 2x2 system J · [dVs, dVb]^T = -[R_S, R_B]^T element-wise.

    J has shape (..., 2, 2). Returns (dVsint, dVb), each shape (...).

    Degenerate case handling: when all body physics is disabled
    (Iii=Igidl=Igb=Ibd=Ibs=BJT off), R_B ≡ 0 and the second row of J is
    zero. The 2D system is singular but the 1D problem in Vsint is
    well-posed. We detect this (R_B ≈ 0 AND row-2 of J ≈ 0) and reduce
    to dVs = -R_S / a, dVb = 0.
    """
    a = J[..., 0, 0]; b = J[..., 0, 1]
    c = J[..., 1, 0]; d = J[..., 1, 1]

    # Detect degenerate body row (R_B identically 0 ⇒ no info about Vb)
    body_dead = (c.abs() < 1e-30) & (d.abs() < 1e-30) & (R_B.abs() < 1e-30)

    det = a * d - b * c
    # Keep det away from 0 numerically; sign-preserving floor.
    sign = torch.where(det >= 0, torch.ones_like(det), -torch.ones_like(det))
    det_safe = torch.where(det.abs() < 1e-30, sign * 1e-30, det)
    rhs0 = -R_S
    rhs1 = -R_B
    dVs_full = (d * rhs0 - b * rhs1) / det_safe
    dVb_full = (-c * rhs0 + a * rhs1) / det_safe

    # 1-D fallback when body is dead
    a_safe = torch.where(a.abs() < 1e-30, sign * 1e-30, a)
    dVs_1d = -R_S / a_safe
    dVb_1d = torch.zeros_like(dVs_1d)

    dVs = torch.where(body_dead, dVs_1d, dVs_full)
    dVb = torch.where(body_dead, dVb_1d, dVb_full)
    return dVs, dVb


def _jacobian_finite_diff(
    cfg, model_M1, bjt, Vd, VG1, VG2, Vsint, Vb, P_M1, P_M2, h: float = 1e-6,
    model_M2=None,
) -> torch.Tensor:
    """Finite-difference 2x2 Jacobian ∂(R_Sint, R_B)/∂(Vsint, Vb).

    Vectorized over leading dims of Vsint/Vb. Returns shape (..., 2, 2).
    Computed under torch.no_grad — used inside the Newton loop only for the
    *step direction*; the autograd path through the converged solution
    flows via the iterative updates themselves (since they're under grad).
    """
    with torch.no_grad():
        # Central differences on Vsint
        Rsp_s, Rbp_s, _ = _residuals(cfg, model_M1, bjt, Vd, VG1, VG2,
                                     Vsint + h, Vb, P_M1, P_M2, model_M2=model_M2)
        Rsm_s, Rbm_s, _ = _residuals(cfg, model_M1, bjt, Vd, VG1, VG2,
                                     Vsint - h, Vb, P_M1, P_M2, model_M2=model_M2)
        dRs_dVs = (Rsp_s - Rsm_s) / (2 * h)
        dRb_dVs = (Rbp_s - Rbm_s) / (2 * h)
        # Central differences on Vb
        Rsp_b, Rbp_b, _ = _residuals(cfg, model_M1, bjt, Vd, VG1, VG2,
                                     Vsint, Vb + h, P_M1, P_M2, model_M2=model_M2)
        Rsm_b, Rbm_b, _ = _residuals(cfg, model_M1, bjt, Vd, VG1, VG2,
                                     Vsint, Vb - h, P_M1, P_M2, model_M2=model_M2)
        dRs_dVb = (Rsp_b - Rsm_b) / (2 * h)
        dRb_dVb = (Rbp_b - Rbm_b) / (2 * h)
    J = torch.stack([
        torch.stack([dRs_dVs, dRs_dVb], dim=-1),
        torch.stack([dRb_dVs, dRb_dVb], dim=-1),
    ], dim=-2)
    return J


def solve_2t_steady_state(
    cfg: NSRAMCell2TConfig,
    model: BSIM4Model,
    bjt: GummelPoonNPN,
    Vd: torch.Tensor,
    VG1: torch.Tensor,
    VG2: torch.Tensor,
    P_M1: Optional[dict] = None,
    P_M2: Optional[dict] = None,
    Vsint_init: Optional[torch.Tensor] = None,
    Vb_init: Optional[torch.Tensor] = None,
    verbose: bool = False,
    model_M2: Optional[BSIM4Model] = None,
) -> dict:
    """Solve the 2T cell at quasi-static (Vd, VG1, VG2).

    Returns dict with: Id, Vsint, Vb, components, R_Sint, R_B, niter, converged.

    Newton step uses *finite-difference* Jacobian (no_grad). Voltage
    updates themselves are inside the autograd graph, so gradients of Id
    w.r.t. fit params flow through the Newton iterates. This is slower
    than implicit-diff but correct and simpler.
    """
    # Coerce inputs to fp64 broadcastable tensors
    Vd = torch.as_tensor(Vd, dtype=torch.float64)
    VG1 = torch.as_tensor(VG1, dtype=torch.float64)
    VG2 = torch.as_tensor(VG2, dtype=torch.float64)
    Vd, VG1, VG2 = torch.broadcast_tensors(Vd, VG1, VG2)
    Vd = Vd.contiguous(); VG1 = VG1.contiguous(); VG2 = VG2.contiguous()

    if Vsint_init is None:
        Vsint = (0.5 * Vd).detach().clone()
    else:
        Vsint = Vsint_init.detach().clone().to(torch.float64).expand_as(Vd).contiguous()
    if Vb_init is None:
        # Cold-start at Vb=0. Note: oracle consensus recommended Vb=0.5
        # but in this model (PTM 130nm bulkNSRAM card) Iii=0 at typical
        # biases, so the high-Vb root is not an attractor and Newton
        # drifts back. Default Vb=0 matches legacy behaviour. Use the
        # `Vb_init=` kwarg explicitly when you know your bias is in the
        # impact-ion regime.
        Vb = torch.zeros_like(Vd)
    else:
        Vb = Vb_init.detach().clone().to(torch.float64).expand_as(Vd).contiguous()

    # Initial residual (need it grad-tracked for IFT-free autograd flow)
    R_S, R_B, comp0 = _residuals(cfg, model, bjt, Vd, VG1, VG2, Vsint, Vb, P_M1, P_M2,
                                 model_M2=model_M2)
    prev_resid_norm = (R_S.detach().abs() + R_B.detach().abs()).max()

    def _physical_scale(comp: dict) -> torch.Tensor:
        """Build a per-bias physical-current magnitude from KCL components.
        Used for relative-tolerance convergence — residual must be small
        relative to the current actually flowing in the device, not relative
        to the residual itself (circular)."""
        keys = ["Ids_M1", "Ids_M2", "Ic_Q1", "Ib_Q1",
                "Iii_M1", "Iii_M2", "Igidl_M1", "Igidl_M2",
                "Ibs_M1", "Ibd_M1", "Ibs_M2", "Ibd_M2"]
        scale = torch.zeros_like(R_S.detach())
        for k in keys:
            if k in comp:
                scale = scale + comp[k].detach().abs()
        return scale

    # Tolerances
    iabstol = getattr(cfg, "Iabstol", cfg.newton_tol)
    ireltol = getattr(cfg, "Ireltol", 0.0)
    xtol_v  = getattr(cfg, "xtol_v", 0.0)
    min_iters = getattr(cfg, "min_iters", 1)

    converged = torch.zeros_like(Vd, dtype=torch.bool)
    niter = 0
    last_dV_inf = torch.tensor(float("inf"), dtype=torch.float64)
    cur_comp = comp0
    for it in range(cfg.newton_max_iters):
        niter = it + 1
        # Convergence check (oracle hardening):
        #   - residual: |R| < max(Iabstol, Ireltol * |I_physical|)
        #     where I_physical = Σ|component currents|
        #   - step:     |dV|_inf < xtol_v
        #   - guard:    require >= min_iters AND the residual must have
        #               actually decreased once (or we've passed iter 1)
        residual_max = torch.maximum(R_S.detach().abs(), R_B.detach().abs())
        I_scale = _physical_scale(cur_comp)
        tol_eff = torch.maximum(torch.full_like(I_scale, iabstol), ireltol * I_scale)
        residual_ok = bool((residual_max < tol_eff).all())
        step_ok = bool((last_dV_inf < xtol_v).all()) if xtol_v > 0 else False
        cur_norm = (R_S.detach().abs() + R_B.detach().abs()).max()
        # min_iters: never declare convergence before this many iterations
        # have actually been taken (it counts the iteration *just executed*;
        # we must have done at least min_iters of them, i.e. it >= min_iters).
        if it >= min_iters and (residual_ok or step_ok):
            converged = residual_max < tol_eff
            if verbose:
                print(f"  Newton converged in {it} iter; max R = {residual_max.max():.3e} "
                      f"|dV|_inf = {float(last_dV_inf):.3e}")
            break
        prev_resid_norm = cur_norm

        # FD Jacobian (no_grad), step direction (no_grad). The implicit
        # function theorem is applied AFTER convergence to attach gradients
        # — see the IFT block at the end of this function.
        J = _jacobian_finite_diff(cfg, model, bjt, Vd, VG1, VG2,
                                  Vsint.detach(), Vb.detach(),
                                  P_M1, P_M2, model_M2=model_M2)
        dVs, dVb = _solve_jac_2x2(R_S.detach(), R_B.detach(), J)

        # Step-size cap (per-iteration relative-step limiter)
        max_abs = torch.maximum(dVs.abs(), dVb.abs())
        scale = torch.where(max_abs > cfg.max_step_V,
                            cfg.max_step_V / max_abs.clamp_min(1e-30),
                            torch.ones_like(max_abs))
        dVs = dVs * scale
        dVb = dVb * scale

        # Damped step + backtracking on residual norm (Armijo-style halving)
        damping = cfg.newton_damping
        prev_norm = R_S.detach().abs() + R_B.detach().abs()
        accepted = False
        while damping >= cfg.newton_min_damping:
            Vsint_try = Vsint + damping * dVs
            Vb_try = Vb + damping * dVb
            R_S_try, R_B_try, comp_try = _residuals(cfg, model, bjt, Vd, VG1, VG2,
                                             Vsint_try, Vb_try, P_M1, P_M2,
                                             model_M2=model_M2)
            new_norm = R_S_try.detach().abs() + R_B_try.detach().abs()
            # Strict decrease (mean over batch) accepted; or fall through at
            # min damping. The 0.999 factor demands genuine descent — at
            # min_damping we accept whatever we have.
            if (new_norm.mean() < prev_norm.mean() * 0.999) or damping <= cfg.newton_min_damping:
                Vsint = Vsint_try
                Vb = Vb_try
                R_S = R_S_try
                R_B = R_B_try
                cur_comp = comp_try
                accepted = True
                # Track step size for xtol convergence
                last_dV_inf = torch.maximum(
                    (damping * dVs).abs().max(),
                    (damping * dVb).abs().max(),
                )
                break
            damping *= 0.5
        if verbose:
            rmax = torch.maximum(R_S.detach().abs(), R_B.detach().abs()).max()
            print(f"  iter {it}: damping={damping:.3f} max|R|={rmax:.3e} "
                  f"|dVs|={dVs.abs().max():.3e} |dVb|={dVb.abs().max():.3e}")
        if not accepted:
            break

    # ----- Implicit Function Theorem (IFT) attachment -----
    # At convergence, R(x*, theta) ≈ 0 numerically, but x* (Vsint, Vb) has
    # been computed under no_grad — so it carries no gradient back to theta.
    # IFT says dx*/dtheta = -J^-1 · ∂R/∂theta. We can encode this in the
    # autograd graph by replacing x* with an "attached" version:
    #     x_attached = x*.detach() - J^-1 @ R(x*.detach(), theta)
    # At convergence R≈0 so x_attached ≈ x* in value, but its gradient w.r.t.
    # theta is exactly the IFT result because J^-1 is detached and R has
    # gradient through theta (via compute_dc, compute_iimpact, ...).
    Vsint_d = Vsint.detach()
    Vb_d = Vb.detach()
    R_S_at, R_B_at, _ = _residuals(cfg, model, bjt, Vd, VG1, VG2, Vsint_d, Vb_d, P_M1, P_M2,
                                   model_M2=model_M2)

    # CRITICAL: only apply IFT correction at biases where Newton ACTUALLY
    # converged. The IFT formula  x* = x*_d - J^-1 R(x*_d, theta)  assumes
    # R ≈ 0; if Newton failed, R can be huge, and J near-singular at that
    # bias would produce a spurious gradient that Adam misreads as a strong
    # signal — root cause of the v6/v7 stage 3 explosion.
    # When residual is too large, ZERO out the IFT delta at that bias →
    # gradient flows through theta-only paths, no broken Vb-loop signal.
    with torch.no_grad():
        J_final = _jacobian_finite_diff(cfg, model, bjt, Vd, VG1, VG2,
                                        Vsint_d, Vb_d, P_M1, P_M2,
                                        model_M2=model_M2)
    delta_s, delta_b = _solve_jac_2x2(R_S_at, R_B_at, J_final)

    # SMOOTH bound IFT delta via tanh — passes gradient through ALL bias
    # points (including non-converged ones), but compresses the magnitude so
    # Adam doesn't see exploding signal. Hard clamp would zero gradient at
    # boundary; tanh is differentiable everywhere.
    #   delta_smooth = D_MAX * tanh(delta_raw / D_MAX)
    # For |delta| << D_MAX: delta_smooth ≈ delta (full IFT signal)
    # For |delta| >> D_MAX: delta_smooth ≈ ±D_MAX, gradient ∝ sech²(.) → 0
    # This is the same effective bound but with smooth gradient transition.
    DELTA_BOUND = 0.3  # V — generous to allow real physics, not just to clip
    delta_s = DELTA_BOUND * torch.tanh(delta_s / DELTA_BOUND)
    delta_b = DELTA_BOUND * torch.tanh(delta_b / DELTA_BOUND)

    # 5th-oracle fix: at non-converged points, the IFT correction is meaningless
    # (Newton never reached a valid root) and would mutate Vsint/Vb away from
    # the un-corrected detached value. The function-level docstring promised we
    # don't apply IFT to non-converged points, but the code did. Gate it now.
    conv_mask = converged.detach()
    delta_s = torch.where(conv_mask, delta_s, torch.zeros_like(delta_s))
    delta_b = torch.where(conv_mask, delta_b, torch.zeros_like(delta_b))

    Vsint = Vsint_d - delta_s
    Vb = Vb_d - delta_b
    R_S, R_B, comp = _residuals(cfg, model, bjt, Vd, VG1, VG2, Vsint, Vb, P_M1, P_M2,
                                model_M2=model_M2)

    # Drain terminal current (positive INTO the D pin):
    #   Id = Ids_M1 (drain absorbs Ids from external) +
    #        Ic_Q1  (collector current absorbed from D) +
    #        Igidl_M1 leaves drain INTO body — but at the D pin this is a
    #        current LEAVING the drain to body, so the external D pin sees
    #        an EXTRA −Igidl_M1 inflow. We add it as a positive contribution
    #        because the convention "Igidl > 0 means current flows from drain
    #        into body via BTBT" implies the external supply drives that
    #        extra current INTO D. Same sign as the channel.
    #   The body diode Ibd_M1 is current LEAVING the body INTO drain, so
    #        from the D pin's perspective it FLOWS OUT to ground via M1
    #        substrate path → contributes −Ibd_M1 to Id (current leaves D).
    #
    # In the typical NS-RAM operating regime, |Ibd_M1|, |Igidl_M1| ≪ Ids_M1
    # so the dominant term is Ids_M1; SCBE / impact-ion shows up via Ic_Q1.
    Id = (
        comp["Ids_M1"]
        + comp["Ic_Q1"]
        + comp["Igidl_M1"]
        - comp["Ibd_M1"]
    )

    residual_max = torch.maximum(R_S.detach().abs(), R_B.detach().abs())
    I_scale_final = (R_S.detach().abs() + R_B.detach().abs()).clamp_min(iabstol)
    tol_final = torch.maximum(torch.full_like(I_scale_final, iabstol),
                              ireltol * I_scale_final)
    converged_final = residual_max < tol_final

    return {
        "Id": Id,
        "Vsint": Vsint,
        "Vb": Vb,
        "Ids_M1": comp["Ids_M1"],
        "Ids_M2": comp["Ids_M2"],
        "Ic_Q1": comp["Ic_Q1"],
        "Ib_Q1": comp["Ib_Q1"],
        "R_Sint": R_S,
        "R_B": R_B,
        "components": comp,
        "niter": niter,
        "converged": converged_final,
    }


# --------------------------------------------------------------------------- #
# gmin homotopy (z89): standard SPICE technique for snapback/bistable cells.  #
# --------------------------------------------------------------------------- #
@contextmanager
def _override_gmin(cfg: NSRAMCell2TConfig, value: float):
    """Temporarily override cfg.gmin (used by `_residuals` shunts)."""
    saved = cfg.gmin
    try:
        cfg.gmin = float(value)
        yield
    finally:
        cfg.gmin = saved


def solve_2t_with_homotopy(
    cfg: NSRAMCell2TConfig,
    model: BSIM4Model,
    bjt: GummelPoonNPN,
    Vd: torch.Tensor,
    VG1: torch.Tensor,
    VG2: torch.Tensor,
    P_M1: Optional[dict] = None,
    P_M2: Optional[dict] = None,
    Vsint_init: Optional[torch.Tensor] = None,
    Vb_init: Optional[torch.Tensor] = None,
    gmin_schedule: Optional[list] = None,
    verbose: bool = False,
    model_M2: Optional[BSIM4Model] = None,
) -> dict:
    """Solve 2T cell using gmin homotopy (oracle consensus recommendation).

    Standard SPICE technique for bistable / snapback / S-shaped I-V circuits:
    start with a LARGE gmin (linearizes the circuit, Newton always converges
    because every node has a strong shunt to its neighbours) and use that
    solution as a warm-start for the next smaller gmin. Repeat until the
    target gmin (= cfg.gmin) is reached.

    Implementation:
      * `gmin_schedule` defaults to [1e-3, 1e-5, 1e-8, 1e-12] followed by
        the cfg-specified target gmin (so the FINAL solve uses exactly the
        gmin that the IFT-attached delta sees, and gradients flow normally).
      * Each step calls existing `solve_2t_steady_state` with a temporarily
        overridden cfg.gmin and the previous solution as warm-start. We do
        NOT change `solve_2t_steady_state` so its IFT machinery is untouched.
      * The final returned dict comes from the last call (target gmin).

    NOTE: gmin shunts are physical-style conductances added in `_residuals`.
    They distort the solution slightly at large values; the homotopy walks
    that distortion smoothly to zero. At the FINAL gmin (= cfg.gmin), the
    solution is identical to a direct solve (only the convergence path is
    different) so gradient flow through IFT is unchanged.
    """
    if gmin_schedule is None:
        # Walk down by ~1000x per step. Final target = cfg.gmin.
        gmin_schedule = [1e-3, 1e-5, 1e-8, 1e-12]
    target = float(cfg.gmin)
    # Always end with the target gmin so IFT delta is computed at it.
    schedule = [g for g in gmin_schedule if g > target] + [target]

    Vsint_warm = Vsint_init
    Vb_warm = Vb_init
    last_out = None
    for step, g in enumerate(schedule):
        with _override_gmin(cfg, g):
            out = solve_2t_steady_state(
                cfg, model, bjt,
                Vd=Vd, VG1=VG1, VG2=VG2,
                P_M1=P_M1, P_M2=P_M2,
                Vsint_init=Vsint_warm,
                Vb_init=Vb_warm,
                verbose=verbose and (step == len(schedule) - 1),
                model_M2=model_M2,
            )
        if verbose:
            conv = bool(out["converged"].all())
            print(f"  homotopy step {step}: gmin={g:.1e}  converged={conv}  "
                  f"niter={out['niter']}", flush=True)
        # Warm-start next step with current solution. Detach so we don't
        # accumulate the previous step's autograd graph (the FINAL solve at
        # target gmin still goes through IFT for gradient attachment).
        Vsint_warm = out["Vsint"].detach()
        Vb_warm = out["Vb"].detach()
        last_out = out
    return last_out


# --------------------------------------------------------------------------- #
# Forward sweep                                                               #
# --------------------------------------------------------------------------- #

def forward_2t(
    cfg: NSRAMCell2TConfig,
    model: Optional[BSIM4Model] = None,
    bjt: Optional[GummelPoonNPN] = None,
    Vd_seq: Optional[torch.Tensor] = None,
    VG1: Optional[torch.Tensor] = None,
    VG2: Optional[torch.Tensor] = None,
    P_M1: Optional[dict] = None,
    P_M2: Optional[dict] = None,
    verbose: bool = False,
    warm_start: bool = True,
    use_homotopy: bool = False,
    dense_vd_in_snapback: bool = False,
    snapback_vd_threshold: float = 1.4,
    snapback_vd_step: float = 0.025,
    *,
    model_M1: Optional[BSIM4Model] = None,
    model_M2: Optional[BSIM4Model] = None,
) -> dict:
    """Sweep Vd from low to high with warm-starting Vsint, Vb between points.

    Returns dict with stacked tensors (shape (T,)): Id, Vsint, Vb, niter,
    converged, plus components per sub-call.

    Args (z89 additions):
      use_homotopy: if True, calls `solve_2t_with_homotopy` per point (gmin
          homotopy, expensive but converges through snapback / bistability).
      dense_vd_in_snapback: if True, internally insert intermediate Vd points
          at `snapback_vd_step` spacing for any segment where Vd >= threshold
          (defaults: threshold=1.4 V, step=0.025 V → 4× denser than the
          z88 default 0.1 V grid). Intermediate points are solved purely for
          warm-starting; the returned arrays only contain values at the
          ORIGINAL Vd_seq points (so the loss never sees the intermediate
          biases — they're a numerical aid only).

    Two-model variant: pass `model_M1=` and `model_M2=` as kwargs to use
    distinct BSIM4 cards for M1 and M2. If only legacy `model` is given,
    both transistors use it (back-compat). Mixing legacy `model` with
    `model_M2=` is also allowed (model → M1, kwarg → M2).
    """
    # Resolve model_M1 / model_M2 from positional `model` and kwargs.
    if model_M1 is None:
        model_M1 = model
    if model_M1 is None:
        raise TypeError("forward_2t requires either positional `model` or `model_M1=` kwarg")
    if model_M2 is None:
        model_M2 = model_M1
    Vd_seq = Vd_seq.to(torch.float64)
    VG1 = torch.as_tensor(VG1, dtype=torch.float64)
    VG2 = torch.as_tensor(VG2, dtype=torch.float64)
    T = int(Vd_seq.shape[0])

    # Build augmented schedule with intermediate (warm-start-only) points.
    # `report_idx[k]` indexes into the augmented sequence and tells us which
    # entries correspond to original Vd_seq points (we only return those).
    if dense_vd_in_snapback and T >= 2:
        aug_vd: list = []
        report_idx: list = []
        prev = float(Vd_seq[0].item())
        aug_vd.append(Vd_seq[0])
        report_idx.append(0)
        for i in range(1, T):
            cur = float(Vd_seq[i].item())
            # Insert intermediate points only if both endpoints (or the
            # current segment top) are in the snapback region. Spacing is
            # `snapback_vd_step` (only inserts if larger gap exists).
            if cur >= snapback_vd_threshold and (cur - prev) > 1.5 * snapback_vd_step:
                n_insert = int((cur - prev) / snapback_vd_step) - 1
                if n_insert > 0:
                    for k in range(1, n_insert + 1):
                        v = prev + (cur - prev) * (k / (n_insert + 1))
                        aug_vd.append(torch.tensor(v, dtype=torch.float64))
            aug_vd.append(Vd_seq[i])
            report_idx.append(len(aug_vd) - 1)
            prev = cur
        Vd_aug = torch.stack(aug_vd)
        report_set = set(report_idx)
    else:
        Vd_aug = Vd_seq
        report_idx = list(range(T))
        report_set = set(report_idx)

    T_aug = int(Vd_aug.shape[0])

    Ids_list, Vs_list, Vb_list = [], [], []
    niter_list, conv_list = [], []
    Ids_M1_list, Ids_M2_list, Ic_Q1_list = [], [], []

    # Cold start at Vb=0.5V (oracle consensus: avoid spurious flat root at
    # Vb=0 where all body currents are sub-femtoamp and Newton "converges"
    # without moving). Vsint=Vd/2 as initial series-divider guess.
    # Then cascade the converged solution from each point as the seed for
    # the next when warm_start=True (default).
    Vsint_warm = torch.tensor(0.0, dtype=torch.float64)  # gets replaced below
    Vb_warm = torch.tensor(0.5, dtype=torch.float64)

    # We collect outputs at ALL augmented points then filter to report_idx
    # at the end. This keeps the inner loop simple.
    aug_outs: list = []
    for i in range(T_aug):
        Vd_i = Vd_aug[i].unsqueeze(0)
        if i == 0:
            Vsint_warm = (Vd_i * 0.5).squeeze(0).detach()
        if use_homotopy:
            out = solve_2t_with_homotopy(
                cfg, model_M1, bjt,
                Vd=Vd_i, VG1=VG1, VG2=VG2,
                P_M1=P_M1, P_M2=P_M2,
                Vsint_init=Vsint_warm.expand_as(Vd_i),
                Vb_init=Vb_warm.expand_as(Vd_i),
                verbose=verbose,
                model_M2=model_M2,
            )
        else:
            out = solve_2t_steady_state(
                cfg, model_M1, bjt,
                Vd=Vd_i, VG1=VG1, VG2=VG2,
                P_M1=P_M1, P_M2=P_M2,
                Vsint_init=Vsint_warm.expand_as(Vd_i),
                Vb_init=Vb_warm.expand_as(Vd_i),
                verbose=verbose,
                model_M2=model_M2,
            )
        aug_outs.append(out)

        # Warm-start next point with current solution (detached so warm
        # start doesn't accumulate the previous step's Newton graph).
        if warm_start:
            Vsint_warm = out["Vsint"].detach().squeeze(0)
            Vb_warm = out["Vb"].detach().squeeze(0)

    # Filter to original Vd_seq points only (preserves graph for those).
    for i in report_idx:
        out = aug_outs[i]
        Ids_list.append(out["Id"].squeeze(0))
        Vs_list.append(out["Vsint"].squeeze(0))
        Vb_list.append(out["Vb"].squeeze(0))
        Ids_M1_list.append(out["Ids_M1"].squeeze(0))
        Ids_M2_list.append(out["Ids_M2"].squeeze(0))
        Ic_Q1_list.append(out["Ic_Q1"].squeeze(0))
        niter_list.append(out["niter"])
        conv_list.append(bool(out["converged"].all()))

    return {
        "Id": torch.stack(Ids_list),
        "Vsint": torch.stack(Vs_list),
        "Vb": torch.stack(Vb_list),
        "Ids_M1": torch.stack(Ids_M1_list),
        "Ids_M2": torch.stack(Ids_M2_list),
        "Ic_Q1": torch.stack(Ic_Q1_list),
        "niter": niter_list,
        "converged": conv_list,
    }
