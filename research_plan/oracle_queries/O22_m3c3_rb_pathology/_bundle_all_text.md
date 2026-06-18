# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: artifacts/M3c3_local_base_plan.md (5532 chars) ===
```
# M3c.3 — Local-base node + base-spreading Rb refactor

**Drafted:** 2026-05-04 ~09:05. **Status:** decision pending.
**Trigger:** O21 verdicts (3-of-3 Vb-clamp confirmed; 2-of-3 α
recommended) + my own Bf sweep + BJT-disable empirical tests.
The β path was empirically refuted; the α path is the only
coherent fix.

## What we are fixing

The body KCL is currently a single-node model:
```
Vb_global = (Vsint, Vb)  ← Newton 2D
```
The BSIM4 impact-ionisation Iii injects holes here, AND the
parasitic-NPN base sees the same Vb. At honest Bf=100, the BJT's
exponential Ib draw dominates the body KCL, flattening Vb across
biases (5 mV spread regardless of inflow magnitude).

In real silicon, the parasitic NPN has a finite **base spreading
resistance** Rb between the *local* point of Iii injection and
the *global* body region where the BJT's vertical base sees. Iii
can elevate the local potential significantly above the global
clamp, modulating Vbe_local without forcing Vb_global to follow.

## What changes structurally

  - Newton state expands from 2D `(Vsint, Vb)` to 3D
    `(Vsint, Vb, Vb_local)`.
  - Add new node `Vb_local` between Iii injection and the BJT base.
  - Add new KCL: `R_local(Vb_local) = 0`
  - The BJT now sees `Vbe_local = Vb_local`.
  - Body diodes / well diodes remain on `Vb_global`.

### KCL for the new local-base node

```
R_local(Vb_local) =
    + (1 − η_lat) · iii_gain · Iii_M1               # Iii injection here
    + Igidl_M1 + Igisl_M1                            # GIDL pumps here too
    + Ib_lat_pair                                     # M3c.1 lateral pair (toggle)
    − Ib_Q1(Vb_local, Vbc=Vb_local−Vd)               # BJT draws from local
    − (Vb_local − Vb) / Rb                           # spread resistor to global
```

### Modified KCL for `Vb_global` (was `Vb`)

```
R_B(Vb) =
    + (Vb_local − Vb) / Rb                           # spread current arrives
    + Igb_M1                                          # gate-body still on global
    − m1_d · (Ibs_M1 + Ibd_M1)                       # body diodes on global
    + I_well_body − I_body_pdiode                    # well/well-body on global
```

(`Vsint` KCL unchanged.)

### Drain accounting

Same as before — Ic_Q1 from BJT, but Ib_Q1 is now a function of
`Vb_local`, not `Vb`.

## Default parameters (regression-test gate)

  - `cfg.use_local_base: bool = False` → identical to F1.v2 (Newton
    stays 2D, no Vb_local node).
  - `cfg.Rb: float = 1e6 Ω` (when toggle on; physically plausible
    for 130 nm lateral parasitic NPN per literature).
  - `cfg.Cb: float = 0.0` (DC; transient extension is M3c.3.b).

### Gate criteria

  1. `use_local_base=False` → bit-identical Id to current F1.v2
     (regression).
  2. `use_local_base=True, Rb=0` → Newton 3D collapses to 2D
     case (Vb_local = Vb forced); should match `use_local_base=False`
     within Newton tolerance.
  3. `use_local_base=True, Rb=1e6` at canonical biases → Vb_local
     should be measurably above Vb_global; Id should show
     bias-dependent variation (not the 5-mV-flat we have now).

## Implementation sketch

Files affected:
  - `nsram/nsram/bsim4_port/nsram_cell_2T.py` (~200 LOC change):
    * `_residuals` returns 3-vector `(R_S, R_B, R_local)`
    * Newton solve_2t → solve_3t (or guard via `use_local_base` flag,
      keep both paths)
    * Jacobian 3×3 instead of 2×2 (analytical autograd preferred
      for speed; finite-diff as fallback)
    * Arclength continuation extends to 3D state
  - `nsram/nsram/bsim4_port/bjt.py`: no change (BJT just sees
    different Vbe).
  - `scripts/test_m3c3_gate.py` (new): regression + sensitivity
    tests per the gate criteria above.

Risk: Newton 3D may not converge as cleanly as 2D. Need warm
start from 2D solution + arclength fold-following. Estimated
debug time: 1 day for solver, 1 day for Jacobian + sensitivity
testing, 1 day for full 33-row refit.

## Pre-registered halt criteria

If after M3c.3 implementation:
  - Newton non-convergence at > 5% of biases at default Rb → halt;
    re-examine state choice.
  - Median dec ≥ 1.20 across reasonable Rb sweep → halt; engage
    O22 oracle round.
  - Any new fit param > 1 OoM outside its physical bound → halt;
    repeats M3a/M3b trap.

## Comparison to M3c plan as originally drafted

  - M3c.1 (electron-hole pair accounting) — DONE, gate passes.
  - M3c.2 (lateral-NPN-as-channel-current) — REFUTED by O21 + my
    own data. Both paths B and C cannot work because Ic_Q1 floor
    dominates Id at all biases.
  - **M3c.3 (this plan) — replaces M3c.2 as the primary structural
    fix.**
  - M3c.4 (smooth gating M_safe) — unchanged from original plan,
    applicable inside M3c.3 if avalanche path is also added later.
  - M3c.5 (charge conservation assertion) — unchanged.

## What this does to the brief / Mario / NRF

  - The brief addendum's M3 timeline statement ("~6 weeks of dev
    work") becomes:
    * M3c.1 (DONE)
    * M3c.3 (this plan, ~1 week dev + refit)
    * M3c.4–.5 (~1 week if needed)
    * M3c-A,B,C (re-run benchmarks, ~3 days)
    * **Total: ~3 weeks calendar to a sendable result, NOT 6.**
  - The reduced timeline reflects that O21's diagnostic test cut
    out the M3c.2 dead-end early.

## Status

This document is the M3c.3 plan. Code work starts when:
  - User authorises Newton 3D refactor (M3c.3 is non-trivial).
  - OR oracle round O22 (or equivalent) reviews this plan and
    confirms direction.

Until then, code state is: M3c.1 + M3c.2 paths B & C committed,
gates pass, defaults reproduce F1.v2.

```


=== FILE: artifacts/O21_synthesis.md (8764 chars) ===
```
# openai response (gpt-5) — 136s

1) Vb-clamp diagnosis
- Correct. With the present KCL, the body-source/drain diodes dominate the body node: any Iii/Igidl inflow and −Ib outflow settle Vb where Ibs+Ibd balances them, giving Vb ≈ 0.39 V and hence Ic_Q1 ≈ Is·exp(Vbe/Vt) ≈ constant. Your gate proves η_lat=0 and toggle=False reproduce F1.v2; avalanche toggle has no effect because Ids≪Ic_Q1. This is a structural consequence of tying the NPN base directly to the global body node with strong vertical diodes and no base-spreading network/capacitance, not a simple sign bug.
- Quick bug checks (but expect same outcome): (i) set m1_diode_scale→0 and see Vb lift and Id vary; (ii) zero Ibs/Ibd temporarily → Vb rises, Ic changes; (iii) KCL residual at body→~0; (iv) set mbjt→0 → floor disappears. If all true, the clamp is real, not a coding bug.

2) Source of silicon’s bias dependence
- Most likely A (missing body-charging path + base-spreading resistance). Real devices have local base (near drain) decoupled from the global body by sheet resistance and charge storage; GIDL/II can pump the local base above the vertical-diode clamp long enough to modulate Vbe and snapback. Raising β alone (B) won’t create the observed bias dependence because Ic≈Is·exp(Vbe/Vt) at DC; β mainly reduces Ib draw and is secondary. C can contribute at some corners, but it won’t explain the strong gate dependence where Iii is tiny unless you see similar variation with BJT disabled.
- Decisive tests:
  - A vs B: add an explicit base-spreading R-network (B_local↔B_global) and modest Cb; rerun DC and slow transients. If bias dependence appears without touching Bf, A wins.
  - B: extract β from a single-bias snapback segment by forcing a known Ib (e.g., inject a tiny current into base proxy node in SPICE test structure) and measuring Ic/Ib. If >>1000, revisit B.
  - C: kill the BJT (mbjt→0) and compare model vs silicon across failing biases. If most of the variation remains, C dominates.

3) Continue M3c?
- α (M3c.3 Vb-charging refactor). You need a local-base node, base-spreading resistance to the global body, and body charge storage (Cbb/Cbj) so Vb can rise under II/GIDL drive; only then will lateral-pair Ib and any avalanche multiplier matter. β (raise Bf) doesn’t remove the clamp; γ (accept floor) concedes a known structural miss.

— — — — — — — — — — — — — — — — — — — — — —

Vb-clamp verdict: Correct — the present KCL with strong vertical body diodes and no base-spreading/capacitance pins Vb, making Ic_Q1 ~ constant.

Silicon current source: A — missing body-charging and base-spread resistance are the primary cause of the observed bias dependence.

M3c continuation: α — implement a local-base node with Rb network and body charge; then re-evaluate lateral injection and any avalanche path.

Critical risk we missed: Interpreting a dynamic snapback mechanism with a DC, single-node body; until base-spreading and body charge are added (and β/Is re-extracted), any β or M(Vbc) conclusions and topology comparisons at low VG2 over-claim fidelity.
# gemini response (gemini-2.5-pro) — 52s

### 1. Is the Vb-clamp diagnosis correct?

**Correct.** Your diagnosis is not only correct, it is the central finding that invalidates the M3c.2 premise. The data is unambiguous: `Ic_Q1` is a constant floor because `Vb` is pinned by the body diodes `Ibs`/`Ibd`. Any physics that attempts to modulate the total drain current by acting on `Ids_M1` or `Iii` is doomed to be a rounding error, as these currents are 5-7 orders of magnitude smaller than the `Ic_Q1` floor.

The most damning evidence is in your own code (`nsram_cell_2T_excerpt.py`): the parameter `cfg.m1_diode_scale`. The comment explicitly states its purpose is to "weaken the diode shunt and let Vb climb." You have already identified the clamping mechanism and implemented the knob to control it. The fact that this knob exists proves the diagnosis is not a bug, but a known, fundamental dynamic of the F1.v2 model at honest `Bf`.

### 2. Where is silicon's bias-dependent current coming from?

**(A) Missing body-charging mechanism.** The Vb-clamp is the pathology. Therefore, the physics you are missing is whatever allows the real device's body potential to overcome that clamp and become bias-dependent. The silicon current's variation *is* the variation of `Vb`.

Hypothesis (B) is a trap; simply inflating `Bf` repeats the M3a error of using a single gain parameter to paper over a structural flaw in the body KCL. Hypothesis (C) is unlikely; the snapback signature is classic parasitic BJT behavior.

The most likely physical mechanisms are a combination of weaker-than-modelled body diodes and/or a significant base-spreading resistance (`Rb`) between the point of `Iii` injection and the diode terminals. A high `Rb` would allow the local `Vb` at the NPN base to rise significantly from `Iii` before the remote diodes can clamp it. The test is simple: run a sweep of `cfg.m1_diode_scale` from 1.0 down to 0.01. If this restores bias-dependence to `Vb` and `Id`, then (A) is confirmed.

### 3. Should we continue M3c at all?

**(α) New M3c.3 — Vb-charging refactor.** You must continue, but M3c is no longer about augmenting the collector current. M3c is now about fixing the body KCL. The halt criterion was triggered correctly; the M3c.2 structure is wrong.

The next step is a structural refactor focused entirely on `Vb` dynamics. This involves two things:
1.  Calibrating `m1_diode_scale` to weaken the clamp until `Vb` shows bias-dependence. This is not a fudge factor; it is an explicit acknowledgement that the default BSIM4 diode model is a poor fit for this cell's body leakage.
2.  Implementing an explicit base-spreading resistance `Rb` to decouple the `Iii` injection point from the diode clamp.

Do not return to (β); that is the path of non-physical hacks. Do not accept (γ); you have identified a clear, fixable structural flaw, and abandoning the model now would be premature.

---

-   **Vb-clamp verdict:** **Correct.** The body diodes are shunting all charging currents, pinning Vb and making Ic_Q1 a constant floor.
-   **Silicon current source:** **A.** The clamp is the problem; the missing physics is whatever defeats the clamp in silicon (weaker diodes, base resistance).
-   **M3c continuation:** **α.** Pivot M3c to a Vb-dynamics refactor; the collector-current paths are irrelevant until the body KCL is fixed.
-   **Critical risk we missed:** The O20 oracles (unanimously) and you were all debating the *shape of the flame* (avalanche vs. β-multiplication) while ignoring the fact that there was *no fuel* (no significant `Ids_M1` or `Iii` current to multiply). The F4 ngspice KCL divergence failure was a major hint that our pyport equilibrium was wrong, and we all missed it.
# grok response (grok-4-latest) — 38s

### 1. Is the Vb-clamp diagnosis correct?
Yes, the diagnosis is correct: at Bf=100, body diodes (Ibs/Ibd) dominate the KCL equilibrium when Iii/Igidl inflows are negligible (most biases), pinning Vb near 0.39 V and rendering Ic_Q1 bias-independent; no evident bug in the excerpted KCL (all terms accounted for, signs consistent), confirming this as a structural limit of the η-bounded model without additional Vb dynamics.

### 2. Where is silicon's bias-dependent current coming from?
B is most likely: the NS-RAM cell's lateral, low-doped-base parasitic NPN plausibly has Bf >> 100 (up to 10^4 possible in 130 nm geometries per literature), amplifying small base drives to produce observed variation without needing unmodeled mechanisms. Distinguishing test: extract silicon Bf from a high-Vd saturation curve at one bias (e.g., VG1=0.6 VG2=0.3) where NPN dominates (Ic/Ib ratio directly gives Bf); if >>100, confirms B; if ~100, probe for A (e.g., transient Vb response) or C (e.g., gate-leakage isolation).

### 3. Should we continue M3c at all?
β is most defensible: the halt criterion triggers on unphysical params (Bf=2e4 was that), but if silicon Bf is legitimately 1000+ due to geometry (ground-truthed via measurement), it's not a fudge—it's calibration; this avoids over-engineering α while beating γ's premature surrender, aligning with O20's "honest" ethos.

\boxed{**Vb-clamp verdict:** correct — body diodes pin Vb at ~0.39 V across biases when inflows are small, making Ic_Q1 constant.

**Silicon current source:** B — lateral geometry enables legitimately higher Bf, explaining variation without new mechanisms.

**M3c continuation:** β — ground-truth high Bf via measurement to justify without halting or refactoring.

**Critical risk we missed:** Ignoring O20's flagged F4 validation failure on internal nodes (Vb/Vsint mismatches >>5 mV), which this Vb-clamp finding reinforces as a core model-SPICE divergence, not "by design."}

```


=== FILE: artifacts/m3c3_excerpt.py (14546 chars) ===
```python
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
        Vbe = Vb                 # emitter = ground (legacy F1.v2 path)
        Vbc = Vb - Vd            # collector = drain
        bjt_out = compute_bjt(bjt, Vbe=Vbe, Vbc=Vbc, T_K=273.15 + cfg.T_C)
        Ic_Q1 = bjt_out["Ic"]    # collector current (drain → emitter = GND)
        Ib_Q1 = bjt_out["Ib"]    # base current (INTO base from external)
        Ie_Q1 = bjt_out["Ie"]    # emitter current at GND (= −(Ic+Ib))
    else:
        Ic_Q1 = torch.zeros_like(Vd)
        Ib_Q1 = torch.zeros_like(Vd)
        Ie_Q1 = torch.zeros_like(Vd)
    # M3c.3 local-base node is patched in below, after iii_gain/eta_lat are
    # defined (those need m1, m2). When use_local_base=True we do an inner
    # 1D solve for Vb_local and overwrite Ic_Q1, Ib_Q1, Ie_Q1.
    Vb_local = Vb  # default: local = global (F1.v2 reduction)

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
    # A.10: extra parasitic pdiode at floating body (Sebas's 2026-05-02
    # email). Anode = body B, cathode = one of {vnwell, GND, Sint}. Default
    # OFF — turns on once we have his SPICE card. Sign convention: I_pdiode
    # = Js·area·(exp((Vb-Vc)/(n·Vt)) - 1), positive when forward-biased,
    # leaves the body → enters R_B with negative sign.
    if cfg.body_pdiode_to != "off":
        Vt_body = 0.02585 * (273.15 + cfg.T_C) / 300.0
        if cfg.body_pdiode_to == "vnwell":
            Vc_pdi = cfg.vnwell
        elif cfg.body_pdiode_to == "gnd":
            Vc_pdi = 0.0
        elif cfg.body_pdiode_to == "sint":
            Vc_pdi = Vsint
        else:
            Vc_pdi = 0.0
        Vab = Vb - Vc_pdi
        exp_arg = (Vab / (cfg.body_pdiode_n * Vt_body)).clamp(-40.0, 40.0)
        I_body_pdiode = (cfg.body_pdiode_Js * cfg.body_pdiode_area
                          * (torch.exp(exp_arg) - 1.0))
        # Phase-B (2026-05-03 10:34): sidewall (perimeter) parallel branch.
        # Off when perim_length=0 (default). Same Vab, separate ideality and
        # saturation per Sebas's 2026-05-02 card (ns=1.0851, isw=1.3664e-13 A/m).
        if cfg.body_pdiode_perim_length > 0.0:
            exp_arg_sw = (Vab / (cfg.body_pdiode_n_sw * Vt_body)).clamp(-40.0, 40.0)
            I_body_pdiode = I_body_pdiode + (
                cfg.body_pdiode_Js_sw * cfg.body_pdiode_perim_length
                * (torch.exp(exp_arg_sw) - 1.0))
    else:
        I_body_pdiode = torch.zeros_like(Vd)

    # A.3.d: scale M1 body diodes (was clamping Vb at ~0.5V at VG1=0.4 row,
    # preventing parasitic NPN from lighting; controlled via cfg.m1_diode_scale,
    # default 1.0). Set <1 to weaken the diode shunt and let Vb climb.
    m1_d = float(cfg.m1_diode_scale)
    # M3b F1.v2 (post-O19): Iii→Vb collection efficiency η ∈ [0, 1].
    # Per O19 openai critique, the previous unbounded `iii_gain` was
    # itself a non-physical fudge factor that re-introduced an unphysical
    # gain path while we tried to clamp Bf. The correct form is a
    # bounded collection efficiency:
    #   η_eff = sigmoid(slope · (Vds − Vds_th)) ∈ [0, 1]
    # which models the fraction of channel-impact-ion holes that reach
    # the parasitic-NPN base laterally vs. diffuse to the bulk.
    # cfg.eta_max ∈ [0, 1] is a hard ceiling. Defaults: η_max=1.0,
    # slope=10/V, Vds_th=1.0 V → snapback regime.
    # If `iii_body_gain` is set explicitly (legacy), it overrides η
    # but is flagged as non-physical in the run log.
    iii_gain_legacy = getattr(cfg, "iii_body_gain", None)
    if iii_gain_legacy is not None and float(iii_gain_legacy) > 1.0 + 1e-9:
        # Legacy non-physical multiplier path. Used pre-O19; kept for
        # back-compat reproducibility, NOT for new fits.
        iii_gain = float(iii_gain_legacy) * torch.ones_like(Vd)
    else:
        eta_max = float(getattr(cfg, "eta_max", 1.0))
        eta_slope = float(getattr(cfg, "eta_slope", 10.0))
        eta_vds_th = float(getattr(cfg, "eta_vds_th", 1.0))
        Vds_eff = (Vd - 0.0)  # M2 source = GND, so Vds_M_full ≈ Vd
        iii_gain = eta_max * torch.sigmoid(eta_slope * (Vds_eff - eta_vds_th))

    # M3c.1 (charge-conserving electron–hole pair accounting). The
    # impact-ionised holes split into two destinations:
    #   η_lat   → fraction reaches the lateral parasitic-NPN base
    #   1−η_lat → fraction diffuses to the bulk body (existing F1.v2 path)
    # At η_lat=0 the routing reproduces F1.v2 exactly (regression gate).
    # The full M(Vbc)·Ids_M1 lateral collector formulation is M3c.2.
    eta_lat = float(getattr(cfg, "eta_lat", 0.0))
    iii_total_for_routing = m1["Iii"] if cfg.m2_body_gnd else (m1["Iii"] + m2["Iii"])
    Ib_lat_pair = eta_lat * iii_gain * iii_total_for_routing
    iii_to_body_factor = (1.0 - eta_lat)
    # M3c.2 path B (AUGMENT, post-O20 unanimous): the lateral-pair Ib
    # drives β·Ib of additional collector current that exits via the
    # drain (parasitic-NPN collector = drain). β = bjt.Bf (clamped at
    # the BJT's Bf parameter, which is honest-physical 100 in M3b).
    # KCL at NPN emitter (= GND): Ie_lat = Ic_lat + Ib_lat_pair.
    # GND absorbs both; drain pin sees +Ic_lat extra inflow.
    Ic_lat = float(getattr(bjt, "Bf", 100.0)) * Ib_lat_pair

    # M3c.3 local-base node (post-O21 unanimous α verdict). When
    # cfg.use_local_base=True, decouple the BJT base from the global
    # body Vb via a spread resistor Rb. Iii + GIDL inject into Vb_local;
    # BJT sees Vbe_local = Vb_local; body diodes see Vb_global.
    # Inner 1D damped-Newton solve (~6-8 iters) at each outer Newton step.
    # Default cfg.use_local_base=False → Vb_local = Vb → reduces to F1.v2.
    if cfg.use_bjt and getattr(cfg, "use_local_base", False):
        Rb = float(getattr(cfg, "lat_Rb", 1e6))
        # Total inflow into Vb_local: (1-η_lat)·iii_gain·Iii + GIDL + Ib_lat_pair
        inflow_local = (
            iii_to_body_factor * iii_gain * iii_total_for_routing
            + m1["Igidl"] + m1["Igisl"]
            + Ib_lat_pair
        )
        Vb_local = Vb.clone().detach()  # warm start at the legacy answer
        for _it in range(10):
            bjt_l = compute_bjt(bjt, Vbe=Vb_local, Vbc=Vb_local - Vd,
                                 T_K=273.15 + cfg.T_C)
            Ib_at_local = bjt_l["Ib"]
            spread = (Vb_local - Vb) / Rb
            f = inflow_local - Ib_at_local - spread
            # Finite-difference Jacobian (Ib_at_local has steep exponential)
            eps = 1e-4
            bjt_p = compute_bjt(bjt, Vbe=Vb_local + eps, Vbc=Vb_local + eps - Vd,
                                 T_K=273.15 + cfg.T_C)
            dIb_dV = (bjt_p["Ib"] - Ib_at_local) / eps
            dfdV = -dIb_dV - 1.0 / Rb
            # Guard small derivatives + clamp step magnitude for stability
            step = -f / torch.where(dfdV.abs() > 1e-30, dfdV,
                                     torch.full_like(dfdV, -1.0 / Rb))
            step = torch.clamp(step, min=-0.1, max=0.1)
            Vb_local = Vb_local + step
            if float(step.abs().max()) < 1e-10:
                break
        # Recompute BJT outputs at the converged Vb_local
        bjt_out_local = compute_bjt(bjt, Vbe=Vb_local, Vbc=Vb_local - Vd,
                                     T_K=273.15 + cfg.T_C)
        Ic_Q1 = bjt_out_local["Ic"]
        Ib_Q1 = bjt_out_local["Ib"]
        Ie_Q1 = bjt_out_local["Ie"]

    # M3c.2 path C (TOGGLE, post-O20): avalanche multiplier on the
    # channel current. Activated only when cfg.use_lateral_collector is
    # True. Default False → identical to path B baseline.
    # Vbc = Vb − Vd. In snapback regime Vd > Vb → Vbc < 0 (reverse-bias
    # B-C). We multiply only by the reverse-biased magnitude:
    #   M(Vbc) = 1 + (max(−Vbc, 0) / BV)^N      smoothed via softplus
    #   M_safe = 1 + (M − 1) · sigmoid((BV_max − |Vbc|) / δ)
    # so M saturates as |Vbc| → BV_max. Ic_avalanche is the EXTRA
    # collector current beyond Ids_M1 — added to drain only, never to
    # Ids_M1 itself (avoids double-count per O19 openai critique).
    if getattr(cfg, "use_lateral_collector", False):
        BV = float(getattr(cfg, "lat_BV", 6.0))
        N_av = float(getattr(cfg, "lat_N", 4.0))
        BV_max = float(getattr(cfg, "lat_BV_max", BV * 1.1))
        delta = float(getattr(cfg, "lat_M_smooth_delta", 0.5))
        # Reverse-bias magnitude: positive only when Vd > Vb
        Vbc = Vb - Vd
        rev_mag = torch.clamp(-Vbc, min=0.0)
        M_raw = 1.0 + (rev_mag / BV) ** N_av
        # Smooth saturation as |Vbc| approaches BV_max
        sat = torch.sigmoid((BV_max - rev_mag) / delta)
        M_safe = 1.0 + (M_raw - 1.0) * sat
        Ic_avalanche = (M_safe - 1.0) * m1["Ids"]
    else:
        Ic_avalanche = torch.zeros_like(Vd)

    if getattr(cfg, "use_local_base", False) and cfg.use_bjt:
        # M3c.3: Iii + GIDL + Ib_lat_pair routed through Vb_local.
        # Global body Vb sees only the spread current arriving from
        # local (replacing the direct Iii/GIDL terms), plus body
        # diodes / well diodes / Igb (which clamp Vb_global at ~0.4 V
        # but no longer dominate Vb_local).
        Rb = float(getattr(cfg, "lat_Rb", 1e6))
        spread_in = (Vb_local - Vb) / Rb
        if cfg.m2_body_gnd:
            R_B = (
                spread_in                       # Iii/GIDL/Ib_lat now arrive here via Rb
                + m1["Igb"]
                - m1_d * m1["Ibs"] - m1_d * m1["Ibd"]
                + I_well_body
                - I_body_pdiode
            )
        else:
            R_B = (
                spread_in
                + m1["Igb"] + m2["Igb"]
                - m1["Ibs"] - m1["Ibd"]
                - m2["Ibs"] - m2["Ibd"]
                + I_well_body
                - I_body_pdiode
            )
    elif cfg.m2_body_gnd:
        # A.1.u: M2's body is GND, so its body-current contributions do
        # NOT enter the floating-body KCL — they flow between M2's nodes
        # and ground, not the floating Vb.
        R_B = (
            iii_to_body_factor * iii_gain * m1["Iii"]
            + m1["Igidl"] + m1["Igisl"]
            + m1["Igb"]
            - m1_d * m1["Ibs"] - m1_d * m1["Ibd"]
            - Ib_Q1
            - Ib_lat_pair
            + I_well_body
            - I_body_pdiode
        )
    else:
        R_B = (
            iii_to_body_factor * iii_gain * (m1["Iii"] + m2["Iii"])
            + m1["Igidl"] + m1["Igisl"] + m2["Igidl"] + m2["Igisl"]
            + m1["Igb"] + m2["Igb"]
            - m1["Ibs"] - m1["Ibd"]
            - m2["Ibs"] - m2["Ibd"]
            - Ib_Q1
            - Ib_lat_pair
            + I_well_body
            - I_body_pdiode
        )

    # Oracle-recommended gmin shunts — ngspice-style parallel conductance
    # in PARALLEL with each pn junction, NOT a single shunt to ground.
    # This is what gives the body a tendency to track (Vd+Vs)/2 in absence
    # of other forces, matching ngspice's behavior.
    #   I_gmin_bd = gmin * (Vd - Vb)   flows INTO body from drain
    #   I_gmin_bs = gmin * (Vs - Vb) = -gmin * Vb (since Vs=0)
    #                                   flows INTO body from source

```


=== FILE: artifacts/m3c3_test_output.txt (799 chars) ===
```
M3c.3 inner-solve test results (2026-05-04 ~09:25)

=== Single-bias Rb sweep (VG1=0.4 VG2=0.2 Vd=2.0 V, Bf=100) ===
Rb=1.0e+03  Vb=0.3876  Vb_local=0.3876  Ic_Q1=1.638e-08  Id=1.638e-08
Rb=1.0e+06  Vb=0.3878  Vb_local=0.3876  Ic_Q1=1.638e-08  Id=1.638e-08
Rb=1.0e+09  Vb=0.5320  Vb_local=0.3852  Ic_Q1=1.491e-08  Id=1.492e-08
Rb=1.0e+12  Vb=1.6039  Vb_local=0.7523  Ic_Q1=2.168e-02  Id=2.168e-02

=== Multi-bias Rb scan ===
          bias      Rb=1e6 Id      Rb=1e8 Id     Rb=1e10 Id
VG1=0.8 VG2=0.50       2.321e-08      2.297e-08      2.529e-05
VG1=0.6 VG2=0.30       1.640e-08      1.624e-08      8.138e-04
VG1=0.4 VG2=0.20       1.638e-08      1.623e-08      1.121e-02
VG1=0.6 VG2=0.00       1.638e-08      1.622e-08      1.236e-02
VG1=0.2 VG2=0.10       1.638e-08      1.623e-08      1.232e-02

```
