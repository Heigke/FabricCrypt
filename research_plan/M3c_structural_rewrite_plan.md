# M3c — Lateral-NPN structural rewrite plan

**Trigger:** M3b closure (F1.v2 honest result = median 1.39 dec) + O19
critical risk callout: "the model remains a phenomenological fit, not
a physical one." Even with η bounded ∈ [0, 1] and Bf ≤ 100, the
2T cell with a SEPARATE Gummel-Poon NPN cannot reach < 1.0 dec
because the snapback magnitudes silicon shows require a different
structural mechanism.

**Goal:** rebuild the parasitic-NPN model to capture the *lateral*
mechanism (channel current acts as the collector at high Vds in the
snapback regime) rather than the current *vertical* lumped GP NPN.
Per O19 openai's specific guidance:

> "Don't replace the BJT with a pure Ids gain. Keep the BJT and
>  refactor the drive: base current = η(Vds, Vgs, Vbs)·Iii with
>  0 ≤ η ≤ 1 plus a base-spreading resistance network; ensure
>  charge conservation (electron–hole pair accounting). If you
>  implement 'Ids × gain', expect: double-counted conduction,
>  broken gm/gds continuity, non-conservative charge (bad
>  caps/transients), premature snapback/latch, and poor
>  extrapolation at low-Vg."

**This is M3c, not M3b.** M3b closes with the honest 1.39 dec
result. M3c is the next major work item, scoped at ~6 weeks dev.

---

## What's wrong with the current model (precise statement)

The current `_residuals` body KCL treats the parasitic NPN as:

  - Vertical Gummel-Poon `Q1` from `bjt.py`
  - Drive: `Ic_Q1, Ib_Q1, Ie_Q1 = compute_npn_currents(bjt, Vbe, Vbc)`
  - Body charge: `R_B includes -Ib_Q1`
  - Channel current: `Ids_M1` is computed independently of NPN state

This is a **two-device** model where:
  - M1 BSIM4 channel current flows D→S unaffected by NPN
  - Q1 NPN current flows independently, base fed by Ib_Q1
  - The two interact ONLY through Vb (NPN base voltage)

In real silicon, the lateral parasitic NPN's collector current IS
M1's channel current at high Vds. Both come from the same
electrons. The current model double-counts: Ids_M1 + Ic_Q1, when
in reality at snapback Ic_Q1 ≈ Ids_M1 (the same charge carriers).

This is why we needed Bf=2×10⁴ (z139) or γ=1×10⁵ (F1) to fit:
to inflate Ic_Q1 to match silicon's *total* observed current, when
the proper model would just route Ids_M1 through the NPN's
amplification at the right Vds.

---

## Five-component restructure (the M3c work)

### M3c.1 — Charge-conserving electron-hole accounting

Add explicit pair-generation rate at impact-ionisation site:

  G_pair(Vds, Vgs, Vbs) = α₀ · |Ids_channel| · exp(-β₀ / Vds_eff)

(BSIM4 §6.1 form, already in `compute_iimpact`.)

For each pair:
  - 1 electron joins the channel (Ids_channel → Ids_M1 unchanged)
  - 1 hole flows: fraction η_lat to lateral-NPN base, fraction
    (1-η_lat) to bulk diffusion (Iii into body)

Constraint: η_lat + η_bulk = 1, both in [0, 1].

Total body hole current: η_bulk · G_pair (replaces current Iii term).
Total NPN base current: η_lat · G_pair (NEW path; replaces current
unbounded γ multiplier).

Default: η_lat = 0.5 + sigmoid(slope · (Vds - Vds_th)) · 0.5
(Vds-modulated: at low Vds most go to bulk, at high Vds most go
to lateral NPN base).

### M3c.2 — Lateral NPN with channel-collector

Replace the standalone Gummel-Poon NPN drive with:

  Ic_Q1 = M(Vbc) · Ids_channel        # collector = channel current
  Ib_Q1 = η_lat · G_pair              # base = lateral hole current
  Ie_Q1 = Ic_Q1 + Ib_Q1               # KCL at emitter
  M(Vbc) = 1 + (Vbc / BV)^N            # avalanche multiplier when
                                       #   reverse-biased base-collector

`BV` (breakdown voltage) ≈ 6 V for 130 nm bulk parasitic NPN.
`N` (multiplication exponent) ≈ 4–6.

Body KCL at Vb becomes:

  G_pair · η_bulk    + Igidl + Igb        # IN
  - Ib_Q1            - Ibs - Ibd          # OUT
  + I_well_body      - I_body_pdiode      # boundary
  = 0

This is structurally different from the current model: the NPN
collector is no longer a separate Gummel-Poon current; it's a
multiplication factor on the existing channel current.

### M3c.3 — Base-spreading resistance Rb

Add a series resistance between the body node Vb and the effective
NPN base node Vbase:

  Vbase = Vb - Ib_Q1 · Rb

Rb ≈ 100 kΩ — 1 MΩ for the lateral parasitic NPN (depends on
geometry). This adds ONE new fit parameter but it's well-bounded.
Rb prevents the body from collapsing to a single voltage; the
effective base voltage that drives M(Vbc) is slightly different.

Per O19 openai: this is the "base-spreading resistance network"
explicitly required.

### M3c.4 — Smooth gating, no latching

The avalanche multiplier `M(Vbc) = 1 + (Vbc/BV)^N` blows up at
Vbc = BV. Need a smooth saturation:

  M_safe(Vbc) = 1 + (Vbc/BV)^N · sigmoid((BV_max - Vbc) / δ)

so that as Vbc → BV_max, M smoothly saturates rather than diverging.
This keeps Newton happy and avoids latching.

### M3c.5 — Charge conservation check

At every converged operating point, verify:

  Σ I_node = 0  for {drain, source, body, sint, well, ground}

Within Newton tolerance. Add an assertion in `_residuals` that
catches any new term that violates KCL.

---

## Implementation order + checkpoints

| step      | what                                  | gate criterion                                          |
|-----------|---------------------------------------|---------------------------------------------------------|
| M3c.1     | electron-hole pair accounting in body | F1.v2 numbers reproduce when η_lat=0 (regression test)  |
| M3c.2.a   | replace Ic_Q1 with M(Vbc)·Ids_M1      | M(0)=1 reproduces F1.v2; sweep BV ∈ [3,9] V             |
| M3c.2.b   | full lateral NPN + base accounting    | VG1=0.6 V row drops < 1.0 dec without unphysical params |
| M3c.3     | base-spreading Rb                     | VG1=0.4 V row drops < 1.0 dec; doesn't break VG1=0.6    |
| M3c.4     | smooth gating M_safe                  | Newton converges at all 33 biases (no NaN, no div)      |
| M3c.5     | charge conservation assertion         | any new physics term verified non-violating             |
| M3c-A     | full 33-row refit                     | overall median < 1.0 dec; report row-wise histogram     |
| M3c-B     | re-run F2 ngspice grid                | still ≤ 2 % single-MOSFET; new 2T-cell delta documented |
| M3c-C     | re-run F3 z142 topology               | n=5 × 3-ρ-norm; with new cell — does MESH_4N regain     |
|           |                                       | championship at honest physical params?                 |
| M3c-D     | new addendum (M3c-addendum)           | < 1.0 dec headline + structurally faithful model        |

Estimated wall time:
  - M3c.1–.5 dev: 1–2 weeks calendar
  - M3c-A refit: 1 day compute (overnight)
  - M3c-B/C reruns: 2–3 days
  - M3c-D writeup + O21 dispatch: 2 days
  - Total: **~6 weeks calendar to a sendable < 1.0 dec result**

This timeline is what the Mario follow-up email references as
"~6 weeks of dev work, not 2."

## Pre-registered halt criteria (avoid the fudge trap)

If after M3c.1–.5:
  - Overall median ≥ 1.0 dec → engage O21 oracles before further work
  - Any new fit param > 1 order of magnitude outside its physical
    bound → halt; the structure is still wrong
  - Newton non-convergence at > 5 % of biases → halt; solver fragile

The lesson from M3a/M3b: don't keep adding parameters when the
structure is wrong. M3c IS the structural fix; if it doesn't work,
the conclusion is "this 2T port cannot match silicon at honest
parameters" and we communicate that.

## What this changes in the brief / Mario / NRF

- The brief's M3 timeline (currently "≤ 2 weeks calendar") becomes
  M3 (M3a + M3b closure, ~3 days, **DONE**) plus M3c (~6 weeks).
- The brief's headline 1.00 dec becomes 1.39 dec for M3b closure
  and an open commitment for M3c.
- The architectural rec (MESH_4N) is on hold pending z142 + M3c
  re-run.

## Blocked-on dependencies

- **M3c.3 base-spreading Rb**: needs measured snapback I-V slope
  characterisation from Sebas. Currently a fit parameter; physical
  ground-truthing requires Sebas's high-Vd transient data
  (BLOCKED, task #128, #90).

- **M3c-C z142 rerun**: requires F3 (z142 at honest cell) to land
  first. Currently in flight.

## Status

This document is the M3c plan. M3c work itself starts when:
  - M3b closes (z142 + O20 SEND)
  - User authorises M3c kickoff
  - Brief addendum sends with current honest 1.39 dec

Until then, the plan is read-only.
