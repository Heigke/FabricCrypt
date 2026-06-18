# O41 — 12h oracle gap-closing review (2026-05-11)

## State (last 12h, see context.md)

- Project compute-closed: V_G2 continuum CLOSED (z244/z244b failed honestly),
  NS-RAM-vs-ESN matrix CLOSED at 14 cells (0 NS-RAM wins, 12 ESN wins, 1 tie),
  ESN-fairness sweep (36 configs) and NS-RAM hyperparam sweep (27 configs)
  confirm no headroom. z254 30-seed polish flipped NARMA-20 tie to ESN strict.
- Mario brief v4.3 locked stateless. Onepager + zip packaged.
- Baseline watchdog PASS 4 consecutive days. APU stable 34–39°C.
- 8+ idle hourly check-ins. Weekly review logged today.
- **Newly surfaced**: git repo corruption (zero-byte objects dated Mar 14;
  push blocked). Not auto-fixed.
- HUMAN-side blockers: Sebas main email 7d unsent (past 5-day flag),
  Sebas thick-ox 4d, Mario v2 2d. A.4 transient + A.6 Julia cross-val
  blocked on these.

## Pre-registered acceptance gate (unchanged)

Per matrix cell: NS-RAM mean strictly better than ESN, non-overlapping
95% CIs, n≥5 seeds, matched N + same readout/W_in/pipeline.

## Three questions

**Q1.** Have any results in the last 12h crossed a decision gate (Week-1 CI,
scale-gap, area-matched)? If yes, what concrete action does the gate trigger?
If no, is the no-action posture correct?

**Q2.** Cherry-picking / stats pitfall audit on this window: with the full
matrix already null and z254 polish flipping a tie to ESN, is there any
residual risk I am over-interpreting the negative pattern (e.g. selection
bias on which cells got 30-seed polish, ESN-fairness sweep range too narrow,
NS-RAM hyperparam grid missing a regime)? Concrete tightening, not vague
concerns.

**Q3.** Next single highest-value experiment in the next 12h, given that
(a) compute plan is closed, (b) brief is locked, (c) Sebas silicon data is
the actual gating dependency, (d) git push is currently broken? Options:
do nothing (idle is correct), or pivot to a NS-RAM non-reservoir primitive
(PUF, stateful trigger, chaotic oscillator) — pick one with a one-paragraph
justification, or argue idle is right.

Be terse. ≤200 words per question. If you don't see signal, say so.
