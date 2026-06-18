# O11 — Post-NARMA brief review (4/5 benchmarks, T-3 days from deadline)

We are 3 days from the Mario Lanza NRF brief deadline (2026-05-06).
Since O10 (2026-05-02 ~21:30 our time), we have:

  1. Closed B.5.b memory-capacity multi-seed (z102):
     **MC = 1.10 ± 0.23 vs 0.22 ± 0.10** (paired t = +7.4, 5 seeds, N=10).
  2. Added temporal-XOR(τ=2) (z104):
     **acc 0.68 ± 0.13 vs 0.54 ± 0.04** (paired t = +2.7).
  3. Added Hopfield-style associative recall (z105):
     **substrate alone 0.69 ± 0.07** (chance 0.33); recurrence
     *hurts* (Δ = −0.11, t = −2.45). Yields a "task-class
     dichotomy" framing.
  4. NARMA-10 v1 at N=10 (z103) returned chance — deferred.
  5. NARMA-10 v2 at N=100, ρ=0.9, κ=0.03 (z106) over-drove
     into instability (NRMSE 1.93 ± 1.14 vs 1.07 ± 0.05) — diagnosed
     as canonical ESN edge-of-chaos overshoot.
  6. NARMA-10 v3 at N=100, ρ=0.9 with finer κ (z107) graduates:
     **NRMSE 0.946 ± 0.018 at κ=0.003** (paired t = −9.4),
     largest effect size and tightest std in the suite.
  7. Drafted C.3 tape-out recommendation v1 with isolated/
     coupled/hybrid array variants and 5 named risks.
  8. Quadrant chart fixed: NS-RAM now plotted at TWO operating
     granularities (per-cycle 21 fJ × 0.7 ns AND 1024-step
     inference projection 21.5 pJ × 0.7 µs) to be apples-to-apples
     with vendor markers — addressing your O10 critique directly.
  9. Mario brief expanded: 4 B.5 paragraphs in Status,
     a new Limitations section (5 visible caveats), a 2-marker
     quadrant chart, and an explicit forward-reference to C.3.
     PDF currently 5 pages, 333 KB.

## Where we stand on the original O10 critiques

**Convergent O10 recommendations and their current status:**

| O10 recommendation                                       | Status |
|----------------------------------------------------------|--------|
| Pivot immediately to finalize Mario brief                | Done — 4/5 benchmarks, Limitations, quadrant fix, C.3 forward-ref |
| Then M9 fan-out for Sebas                                 | Pending — drafted in C.3 v1 routing topology |
| Defer B.5.c topology coupling                             | Implemented as software recurrence via VG2 modulation; SUCCESSFUL on 3 of 4 temporal benchmarks |
| Reframe as "Ground Truth Simulator that debugs ngspice"   | Reflected in Status §, 5-bug catalogue |
| Aggressively ask Sebas for 7-rate transient data         | Still A.12 (BLOCKED on Sebas) |
| Add Limitations subsection                                | Done (5 bullets) |

## What we want from O11

Brief is in "send-ready, awaiting user authorization" state. We want
a **last-line oracle review** before the user authorizes the email
to Mario. Specifically:

1. **Is the brief now defensible against the apples-to-oranges
   quadrant critique?** The chart shows two NS-RAM markers; the
   1024-step projection is explicitly labeled "(proj.)" with a
   linear-energy-scaling assumption. Is this honest enough, or
   would NRF reviewers still flag it?

2. **Is the task-class dichotomy framing (recurrence helps temporal,
   hurts spatial) over-claiming?** It is built on N=10 Hopfield
   with M=3 prototypes — small numbers. Does the framing earn its
   place in the brief and in C.3?

3. **The four positive benchmarks: any blind spots?**
    - MC: paired t=+7.4 — convincing.
    - XOR: t=+2.7 — borderline, large σ.
    - Hopfield: substrate alone strong; recurrence hurts.
    - NARMA-10: t=−9.4 BUT κ-bracket is narrow.
   Would a NRF reviewer find any of these "marketed as positive
   but actually borderline"?

4. **Is the Limitations section the right level of disclosure?**
   We name 5 caveats (NARMA κ-sensitive, Hopfield small-scale,
   software-vs-silicon equivalence, thick-ox card pending,
   ngspice bugs not upstreamed). Too few? Too many? Does the
   silicon-equivalence one read as a deal-breaker?

5. **C.3 tape-out recommendation v1 — is it concrete enough?**
   It proposes isolated + coupled + hybrid arrays in 0.6 mm² of
   130 nm. Should it commit to specific bitcell-count / array
   sizes / DAC bit-widths in v1, or is hedging the right call
   given pending Sebas data?

6. **Brutal final critique: what's the weakest link in the
   send-ready brief?** The user wants Grok-style critical
   assessment here even though we're text-only on Grok and only
   dispatching openai+gemini in this packet — please be Grok-level
   skeptical.

## Files attached

  - `nsram_proposal_short.tex` — current brief LaTeX source
  - `nsram_proposal_short.pdf` — rendered 5-page PDF
  - `quadrant_nsram_vs_edge.png` — fixed 2-marker chart
  - `C3_tapeout_recommendation_v1.md` — tape-out recommendation
  - `LOG_tail.md` — last ~250 lines of 01_LOG.md (z102 → present)

Bullet your top-3 critiques and your top-3 recommendations.
Time-budget: ≤24 h before user check-in.
