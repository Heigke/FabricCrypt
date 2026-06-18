# O12 — Final send-gate review (Mario brief, T-3 days)

This is the final pre-send oracle quality gate for the Mario Lanza
NRF brief (deadline 2026-05-06). Since O11 (~01:33 our time) we have:

  1. **Applied 6 of 7 O11 wording micro-edits** to the brief
     (MC absolute-bound caveat, XOR weak-positive framing,
     Hopfield "working hypothesis at small N", NARMA absolute-error
     caveat + "largest paired improvement", Limitations periphery
     bullet, quadrant caption rewrite).
  2. **Issued C.3 v2** (research_plan/C3_tapeout_recommendation_v2.md)
     with the 7th O11 recommendation: explicit small-signal
     κ ↔ R_bulk numerical mapping
       κ=0.003 ⇔ R_bulk≈600 MΩ;  κ=0.03 ⇔ ≈66 MΩ;  κ=0.3 ⇔ ≈6.6 MΩ
     plus 1 MΩ–1 GΩ digital-pot specification (was wrong by 3
     decades in v1) and Risk #3 (NARMA-10 deferral) marked RESOLVED.
  3. **Inserted single-line C.3 v2 callout** in the brief.
  4. **z108 — Hopfield N-scaling** at κ=0:
     N=10 → 0.686 ± 0.070,  N=30 → 0.970 ± 0.028,
     **N=50 → 1.000 ± 0.000** (5 seeds, M=3, p_flip=0.20).
     Substrate-alone advantage confirmed as architectural property.
     Brief Limitations bullet 2 upgraded from "deferred" to
     "scale-confirmed".
  5. **z109 — multi-class waveform** B.5 fifth benchmark CLOSES
     5/5 grid:  κ=0 acc 0.567 ± 0.090, κ=0.003 acc 0.595 ± 0.091,
     paired Δ +0.028 ± 0.026 (t=+1.05). Substrate clears chance
     by ~8 SE; recurrence neutral.
  6. **Inserted refined dichotomy table** in brief — five-point
     monotonic ordering by temporal-memory horizon:

| benchmark        | memory horizon       | recurrence effect |
|------------------|----------------------|-------------------|
| memory capacity  | multi-step           | essential (+0.88, t=+7.4) |
| NARMA-10 (N=100) | ~10 steps            | essential (−0.13 NRMSE, t=−9.4) |
| temporal-XOR(2)  | 2 steps              | beneficial (+0.13, t=+2.7) |
| multi-class wave | 1 step + context     | neutral (+0.03, n.s.) |
| Hopfield (M=3)   | instantaneous        | harmful (−0.11, t=−2.45) |

  7. **Brief metrics:** 5 pages, 374 KB. Page count stable.
     Content-saturated within 5 pages.

## Send-readiness checklist

  - [x] DC fidelity closure documented + 5-bug catalogue.
  - [x] Transient + throughput + GPU 5× target.
  - [x] **5/5 B.5 benchmarks reported** with 5-seed paired-t.
  - [x] **Monotonic dichotomy table** by memory horizon.
  - [x] Limitations section (5 bullets, all bounded or resolved).
  - [x] Quadrant chart fixed (2 markers, periphery caveat).
  - [x] All O11 oracle micro-edits applied.
  - [x] C.3 v2 issued and referenced from brief.
  - [ ] **User authorization to send (only remaining gate).**

## What we want from O12

This is the **final quality gate**. The user wants a green-light
or a single specific blocker before they authorize sending.
Specifically:

1. **Is the brief now defensible against the dichotomy
   "over-confidence" critique** that O11 raised? It is now backed
   by a 5-point monotonic ordering, not a 2-point claim.

2. **Is the 5/5 B.5 closure clean enough** that no reviewer
   will flag a missing benchmark or a swept-under-rug null?
   Multi-class waveform has a NULL result on the κ test
   (paired t=+1.05, n.s.) — is naming that null in the brief
   the right move?

3. **C.3 v2 specifies R_bulk = 1 MΩ–1 GΩ** instead of v1's wrong
   1 kΩ–1 MΩ. This is a 3-decade correction made by explicit
   numerical κ↔τ_coupling derivation. **Is this number right?**
   We used C_body_eff ≈ 5–10 fF (Pazos thin-ox), dt = 10 ns.
   Any obvious gotcha?

4. **Brief is 5 pages.** Mario's brief target was historically
   one-pager-ish; we expanded to 4 with O10 and now to 5 with the
   table. **Is 5 pages acceptable for an NRF brief, or is the
   table costing more reviewer credibility than it adds?**

5. **Limitations bullet about ngspice bug catalogue not being
   upstreamed yet** — should this be hardened into "we will
   submit a note to SimuCAD/ngspice maintainers within 30 days
   of brief sign-off", or kept soft?

6. **GREEN-LIGHT or SINGLE BLOCKER:** if you had to give a
   one-sentence verdict, do you say "send it" or "fix X first"?

## Files attached

- `nsram_proposal_short.tex`
- `nsram_proposal_short.pdf` (5 pages, 374 KB)
- `quadrant_nsram_vs_edge.png`
- `C3_tapeout_recommendation_v2.md`
- `LOG_tail.md` — last ~250 lines

Bullet your top-3 critiques and your top-3 recommendations.
Time-budget: ≤72 h before deadline.
