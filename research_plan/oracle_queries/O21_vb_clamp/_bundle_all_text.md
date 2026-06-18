# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: artifacts/M3c2_design_decision.md (6122 chars) ===
```
# M3c.2 design decision — REPLACE vs AUGMENT the BJT

**Drafted:** 2026-05-04 ~07:15. **Status:** decision pending.
**Trigger:** preparing M3c.2 implementation; identified conflict
between M3c plan and O19 openai oracle's quoted recommendation.

## The conflict

The M3c plan (`research_plan/M3c_structural_rewrite_plan.md`) states:

> Ic_Q1 = M(Vbc) · Ids_channel        # collector = channel current
> ...
> This is structurally different from the current model: the NPN
> collector is no longer a separate Gummel-Poon current; it's a
> multiplication factor on the existing channel current.

But O19 openai's verdict (quoted in the same plan) said:

> "Don't replace the BJT with a pure Ids gain. KEEP the BJT and
> refactor the drive: base current = η(Vds, Vgs, Vbs)·Iii with
> 0 ≤ η ≤ 1 plus a base-spreading resistance network; ensure
> charge conservation. If you implement 'Ids × gain', expect:
> double-counted conduction, broken gm/gds continuity,
> non-conservative charge (bad caps/transients), premature
> snapback/latch, and poor extrapolation at low-Vg."

These are **structurally incompatible.** The plan replaces;
O19 says don't.

## Why this matters

The M3a/M3b walk-back chain has a documented failure mode:
single-parameter inflation 100×–10000× to compensate for missing
physics (Bf=5×10⁴ → walk back; γ=1×10⁵ → walk back). The plan's
"M(Vbc)·Ids_M1" pattern is structurally similar — it's a
multiplier on a real current. If we calibrate `BV` and `N` (the
multiplier shape parameters) post-hoc to fit silicon, we are
re-introducing a fudge factor with two new knobs.

O19 openai's explicit warning ("expect double-counted
conduction") names exactly the pathology we'd hit if we naïvely
follow the plan's formulation: at M=1, the existing Gummel-Poon
Ic_Q1 is dropped, but the "real" silicon at low Vbc has a real
parasitic NPN with a real (small) Ic. The plan's M=1 limit is
**no NPN**, not "F1.v2 NPN".

## Three candidate interpretations

### (A) Replace literally (the plan)

```
Ic_Q1 = M(Vbc) · Ids_M1
Ib_Q1 = η_lat · G_pair
```

- Drops Gummel-Poon entirely.
- M=1 gives "channel-only" with collector = channel
  (double-count if Id_drain still adds Ic_Q1).
- Calibration: tune BV ∈ [3, 9] V and N ∈ [4, 6] to fit silicon.
- **Risk: structural fudge factor.** Two new knobs (BV, N) that
  are *fitted*, not measured. Repeats M3a/M3b pattern.
- **Risk: O19 quoted critique exactly.** "Double-counted
  conduction, broken gm/gds continuity, non-conservative charge."

### (B) Augment (O19 openai's preferred form)

Keep Gummel-Poon BJT exactly as F1.v2. Add lateral pair injection
on top:

```
Ib_Q1_total = Ib_Q1_GP(Vbe, Vbc) + η_lat · G_pair       # already in M3c.1
Ic_Q1 = Ic_Q1_GP(Vbe, Vbc)                               # unchanged from F1.v2
# NO M(Vbc) multiplier on Ids
```

- M3c.1 is the entire "structural" change. M3c.2 doesn't replace
  anything; it just makes the lateral path drive both Ib AND
  amplifies Ic via the standard Gummel-Poon Ic = β · Ib relationship.
- The 1.39 dec floor is whatever the augmented model hits.
- Calibration: tune η_lat shape (slope, V_th) only — no new BV/N.
- **Risk: may not hit < 1.0 dec.** The 1.39 dec was already with
  η ∈ [0, 1]; adding the lateral pair just routes some current
  through a different path. Magnitude unclear.
- **Safety: zero new fudge factors.** All knobs already in F1.v2.

### (C) Hybrid (toggle)

Add `cfg.use_lateral_collector: bool` (default False).

- False → identical to F1.v2 (and M3c.1 with η_lat=0).
- True → M(Vbc) lateral collector PLUS legacy Gummel-Poon, with
  an explicit charge-conservation assertion that catches
  double-counting at runtime.

This lets us evaluate (A) and (B) on the same codebase without
deletion, and run the O20-equivalent oracle review once we have
*results from both* before committing structurally.

## Recommendation

**Implement (B) first.** It honors O19's stated preference, has
zero new fudge factors, and gives us a defensible bound on what
the structural lateral-path achieves without adding the M(Vbc)
multiplier. If (B) hits ≤ 1.0 dec, M3c is closed. If (B) plateaus
at ~1.3 dec or worse, we have data showing that (A)'s extra
machinery is *necessary*, which is a much stronger argument for
adding it later than "the plan said so".

Then, gated on (B)'s outcome:
  - If (B) ≤ 1.0 dec: ship it. M3c done, no replacement of BJT.
  - If (B) > 1.2 dec: implement (C) toggle, run a careful
    multi-oracle review BEFORE adding fitted BV/N parameters.

**Don't implement (A) directly.** The M3c plan as written invites
the same trap M3a/M3b walked back from. The user explicitly
authorised "kör m3c" but did so before this conflict was made
explicit; running (A) without flagging the conflict would be
bad faith.

## Implementation sketch for (B)

The change is small:

  1. M3c.1's `Ib_lat_pair = eta_lat · iii_gain · iii_total` (already
     committed) feeds the parasitic NPN base via increased Ib_Q1.
  2. Currently M3c.1 just subtracts Ib_lat_pair from R_B (current
     leaves body via base). For (B), we need to additionally feed
     it into the Gummel-Poon Ic computation: Ic_Q1_eff = Ic_Q1_GP +
     β · Ib_lat_pair (the extra base drive sees Bf gain).
  3. Re-run z91g 33-row fit with eta_lat ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
     at Bf=100, η_max=1.0, and pick the best median.

Estimated time:
  - Code change: 30 minutes
  - Single-bias gate test: 5 minutes
  - 33-row sweep at 5 η_lat values: 30 minutes
  - Result analysis: 30 minutes
  - **Total: ~2 hours**

This is a far smaller scope than the 6-week M3c.1–.5 plan, but
it's the part that honours O19 directly. If the result is
encouraging we expand; if not, we know.

## Pre-registered halt criterion

If (B) median > 1.30 dec across reasonable η_lat sweep, halt code
work and return to oracle review before any further change. This
prevents the M3a/M3b chain repeating.

## Status

This document is a decision request to the user, not an
implementation. Code change blocked on user choice between (A),
(B), or (C). Until decision, M3c work is paused at M3c.1
(charge-conserving routing committed, gate passes).

```


=== FILE: artifacts/O20_synthesis.md (10592 chars) ===
```
# openai response (gpt-5) — 374s

1) Is M3b “closed”? No.
- F2: Id/gm/gds/gmb errors are within spec; caps are not (cgg p90 ≫ 5%). You can claim DC small‑signal agreement; don’t claim cap accuracy.
- F4: 2T op‑point cross‑check fails your own gate (Vsint/Vb deltas orders above 5 mV; only 1/48 full passes). Saying “by design ngspice lacks η” doesn’t rescue a validation that you set to compare node voltages. Either relax/replace the criterion (currents-only) or don’t call M3b validated.
- Baseline: 1.39‑dec median is defensible as “physical, η‑bounded, Bf=100” but only if labeled “25/33 biases evaluated” and with the F4/Tx caveats.
- VG1=0.6 V row: 2.25 dec is an honest regression vs the Bf=2e4 hack. It’s acceptable as evidence the cell needs M3c physics; it’s not a show‑stopper if you frame it that way.
- 8 NaN biases: do not bury them. Either (a) refit 33/33 with un‑overridden cards and report the 33‑row median in parallel, or (b) keep 25/33 and say so in every headline table. Without (a) or (b) clearly stated, it’s not defensible.

2) Topology ranking inversion — real or artefact?
- The flip is mostly real plus a normalization effect. Evidence: with the honest low‑gain cell, meshes/small‑world lose their implicit feedback advantage; ER_SPARSE/LAYERED rise when collinearity is harmful. But ρ‑normalization clearly shifts winners (rho_deg_norm crowns MESH/WS/HUB).
- Kill‑tests:
  - Equalize effective linear dynamics: scale each W by the spectral radius of the one‑step Jacobian J = diag(f′)·W at the operating point (rho_jac). If rankings stabilize across norms under rho_jac, A is right; if they reshuffle again, B is biting.
  - Gain‑headroom sweep: run η_max ∈ {0.6, 0.8, 1.0}. If rankings persist, C is false; if they collapse within ±0.5 dec, C holds.
  - κ×ρ micro‑grid at N=800 for ER_SPARSE vs MESH_4N (3×3): A real effect should show consistent separation across κ.
Recommendation: do η_max sweep (cheap) + rho_jac variant (one codepath) before changing the headline again.

3) Send now or wait for M3c?
- As “validation closed,” you cannot — F4 and cap checks fail. As a corrective addendum that walks back the Bf hack and reports the honest 1.39‑dec + inverted ranking with explicit caveats, you can.
- Pick A (send now) only if you: (i) relabel the validation scope to DC currents + small‑signal conductances, (ii) call out the failed node‑voltage check, caps pending, and (iii) stamp “25/33 biases” on every DC‑fit number. Otherwise, wait.
- If you want one fully coherent artifact with <1.0‑dec target and end‑to‑end physics, pick B (wait for M3c). It’s safer but costs 6 weeks.

4) M3c.2 — REPLACE vs AUGMENT the BJT
- Choose B (AUGMENT). It preserves KCL/KVL continuity, avoids double‑count/dud gm steps, adds no new fudge knobs, and is fast.
- Will B get <1.0 dec? Plausible: η_lat·G_pair drives Ib and is multiplied by β; with Rb you can shape snapback onset/magnitude without inventing M(Vbc)·Ids. If it stalls >1.2 dec, you’ll have strong evidence you actually need an avalanche multiplier — then add a guarded M(Vbc) path under a toggle.
- A’s “M(Vbc)·Ids minus Ids” still risks gm/gds kinks and charge mismatch unless you rewrite the charge partition and ensure continuity across the M→1 limit. That’s a bigger, riskier change; don’t lead with it.

5) Topology recommendation under cross‑norm spread
- The data show three different champions across norms; ER_SPARSE is the only norm‑stable performer (spread 0.85 dec).
- Recommend (i) ER_SPARSE as default for robustness, with an explicit note that if you insist on rho_deg_norm, HUB_SPOKE/WS/MESH win on MC but are norm‑sensitive. Drop WAVE from any rec — it’s collapsed to chance for all topologies at the honest cell.

— — — — — — — — — — — — — — — — — — — — — — —

M3c.2 verdict: B — AUGMENT. Keep Gummel–Poon; drive Ib with η_lat·G_pair and add Rb; fast, conservative, avoids new fudge multipliers.

Topology rec verdict: (i) — ER_SPARSE for cross‑norm robustness; it’s the only topology not flipping with normalization.

Send M3b addendum: A (now), but only as a corrective note with explicit “25/33 biases,” failed node‑voltage/cap validations, and no transient claims; otherwise wait.

Critical risk we missed: claiming “ngspice‑validated” while F4 and cap checks fail — unless you explicitly scope validation to DC currents/derivatives only, reviewers can (fairly) call the model unvalidated at the 2T‑cell level.
# gemini response (gemini-2.5-pro) — 73s

### 1. Is the M3b closure actually closed?

Yes, with one caveat. The 1.39-dec median is a defensible, honest baseline because it uses physically-bounded parameters (Bf≤100, η≤1). The regression in the VG1=0.6 V row is not a deal-breaker; it is a successful diagnostic, correctly revealing the structural model gap that the previous Bf hack was masking. This result is the correct justification for M3c.

The caveat is the 8/33 un-evaluated biases. The F1 plan specified refitting all 33, but the summary shows 8 were skipped. While not a blocker for M3b closure, the addendum must explicitly state the headline number is on 25/33 biases and that the remaining 8 correspond to un-extracted parameter regimes. This is a known limitation of the dataset, not a model failure.

### 2. The topology ranking inversion — is it real or another artefact?

The inversion is almost certainly a **(B) ρ-normalization artefact**. The fact that three different normalization schemes produce three different champions (LAYERED, RAND_GAUSS, HUB_SPOKE) is the smoking gun. The performance of highly structured topologies like MESH_4N and HUB_SPOKE is extremely sensitive to how their spectral properties are scaled.

The best test to distinguish these interpretations is to implement the fix suggested in the O18 review: perform a **per-topology effective spectral radius calibration**. Instead of forcing a nominal ρ=0.9 via a single method for all, find the optimal ρ for each topology via a small sweep, and then compare them at their individual best performance. This would provide a much fairer comparison of their computational capacity.

### 3. Should we send the addendum NOW or hold for M3c?

**A (send now).** The project's credibility is best served by immediate, transparent communication. Sending the addendum now with the 1.39 dec result, the inverted (but more honest) topology table, and a clear commitment to the M3c structural rewrite accomplishes three things: it corrects the record on the non-physical claims in the original brief, it demonstrates a rigorous and honest validation process, and it sets accurate expectations for the ~6 weeks of work required to achieve < 1.0 dec. A 6-week silence after sending a brief with now-invalidated numbers is a greater risk.

### 4. M3c.2 design decision — REPLACE vs AUGMENT the BJT (NEW)

**B (augment).** My O19 critique stands. The M3c plan's proposal to replace the Gummel-Poon model with a simple `M(Vbc)·Ids_channel` multiplier risks repeating the exact same "fudge factor" error from M3a/M3b, as correctly identified in `M3c2_design_decision.md`.

Path (B) — keeping the F1.v2 Gummel-Poon BJT and using the new lateral-pair physics to augment its base drive (`Ib_Q1`) — is the physically principled approach. It introduces new physics without discarding the valid, charge-conservative framework of the existing BJT model. It has zero new unphysical fit parameters. The `M3c2_design_decision.md` recommendation to implement (B) first as a ~2-hour experiment is the correct, leanest path to determine if this structure is sufficient. If it fails to reach the goal, the data from that failure will provide a much stronger justification for exploring more complex (and risky) models like (A) or (C).

### 5. Topology rec under the cross-norm finding (NEW)

**(i) ER_SPARSE for cross-norm robustness.** A recommendation for a physical tape-out cannot be contingent on a specific, abstract mathematical normalization used in a simulation. The results show that the performance of MESH_4N, HUB_SPOKE, and LAYERED is extremely brittle and dependent on the normalization scheme. ER_SPARSE is the only topology that performs in the top-tier across multiple schemes (3.55, 2.84, 2.70), with the smallest performance spread (0.85 dec). It is the safest, most robust recommendation. The brief's secondary "WAVE" axis is dead; the recommendation should be based purely on robust memory capacity.

---
- **M3c.2 verdict:** **B (Augment)**. It honors the O19 critique, avoids new fudge factors, and is the most physically principled next step.
- **Topology rec verdict:** **(i) ER_SPARSE**. It is the only topology demonstrating robust high performance across different normalization schemes.
- **Send M3b addendum:** **A (now)**. Immediate honesty is the most defensible path and correctly frames the need for M3c.
- **Critical risk we missed:** The **F4 2T-cell ngspice cross-check (`z143_summary.json`) FAILED** its acceptance criteria on internal nodes (Vb, Vsint) by a large margin. While Id agreement was better, the massive internal voltage mismatch (e.g., max ΔVb of 99mV vs 5mV threshold) indicates that `pyport`'s cell-level KCL is fundamentally diverging from a standard SPICE netlist, even at Bf=100. This is not "by design"; it's a model discrepancy that must be understood before building more physics (M3c) on top of it.
# grok response (grok-4-latest) — 51s

1. M3b is closed: 1.39 dec is defensible as honest baseline; VG1=0.6 regression to 2.25 dec is acceptable structural gap disclosure; 25/33 biases defensible with explicit caveat on 8 NaN (refit not required for closure).

2. Inversion is real (A): η-bounded low-gain favors decorrelated ER_SPARSE; test by running sweep at η_max=0.5 (further bound) to confirm ranking stability without flip.

3. A (send now): honest 1.39 dec + inverted ranking with caveats is defensible; M3c ETA frames forward plan without delaying addendum.

4. B: physically faithful to hit <1.0 dec without avalanche (η_lat drives β·Ib for snapback); propagation through Gummel-Poon gives magnitude via β, f-Vbc preserved; A's subtraction avoids double-count but remains fudge per O19.

5. (i): ER_SPARSE's low spread (0.85 dec) ensures robustness across norms; avoids fence-sitting or committing to unproven norm.

\boxed{**M3c.2 verdict:** B, honors O19 critique with zero new fudges and fast sims.

**Topology rec verdict:** (i), ER_SPARSE stable across norms for safe rec.

**Send M3b addendum:** A (now)

**Critical risk we missed:** Over-claiming topology rankings as definitive with n=5 seeds; sd overlaps suggest noise, not robust signal.}

```


=== FILE: artifacts/nsram_cell_2T_excerpt.py (5777 chars) ===
```python
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

```


=== FILE: artifacts/vb_clamp_test_output.txt (859 chars) ===
```
=== Gate (toggle=False reproduces F1.v2) ===
          bias         Id_off    Id_explicit    reldiff
------------------------------------------------------------
VG1=0.6 VG2=0.30   1.631039e-08   1.631039e-08   0.00e+00
VG1=0.4 VG2=0.10   1.630332e-08   1.630332e-08   0.00e+00
VG1=0.2 VG2=0.20   1.629325e-08   1.629325e-08   0.00e+00

=== Avalanche sensitivity at VG1=0.6 VG2=0.0, Bf=100, Vd=2.0 V ===
  BV (V)  Id_off (toggle=False)    Id_on (toggle=True)    ratio
----------------------------------------------------------------------
     3.0           1.638403e-08           1.638403e-08    1.000
     4.5           1.638403e-08           1.638403e-08    1.000
     6.0           1.638403e-08           1.638403e-08    1.000
     7.5           1.638403e-08           1.638403e-08    1.000
     9.0           1.638403e-08           1.638403e-08    1.000

```
