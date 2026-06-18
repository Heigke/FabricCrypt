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
