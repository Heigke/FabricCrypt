# 12-hour gap-closing review (NS-RAM Path A — post comprehensive NS-RAM-vs-ESN matrix)

You are reviewing the last 12 hours of autonomous research progress on the
NS-RAM project. The context.md contains the tail of the master log.

## Headline summary of the window

The user asked, post-Mario-v4.3-brief, to find ways forward without
requesting new silicon data from collaborators. We ran two
systematic experimental campaigns under a pre-registered NO-CHEAT
discipline (gates set before running, never bent post-hoc, n ≥ 5
seeds per cell):

**Campaign 1 — V_G2 continuum and mixed-mode-fabric**
- Rate-dependent hysteresis (single cell, transient sim, triangular
  V_G2 sweep at multiple T_ramp ∈ {1ns..30ms}, n=5 init seeds):
  hysteresis is real and peaks at T_ramp ≈ 1ms (body-RC time constant
  τ = C_b·R_b, classical single-RC signature). But its CONTRAST vs a
  quasi-static baseline is only ~5×, not the 100× pre-registered gate.
  Honest FAIL.
- Mixed-mode-fabric (fraction f ∈ {0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}
  of cells V_G2-grounded vs floating, NARMA-10, n=5 seeds):
  best mixed point f=0.25 edges pure-floating by Δ=0.006 NRMSE, far
  below pre-registered margin max(1pp, std)=0.016 NRMSE. Pure-grounded
  (f=1.0) is chance-level. Honest FAIL.
- Conclusion: V_G2 morphing has measurable but soft dynamical
  content; mixed-population gives no architectural advantage over
  pure-floating cells.

**Campaign 2 — NS-RAM vs textbook ESN matrix (head-to-head)**
Matched N, identical input projection, identical ridge readout, n=5
seeds per cell. ESN = sparse tanh, spectral radius 0.9, leak 0.30.
Pre-registered gate per cell: NS-RAM CI95 upper < ESN CI95 lower.
- seq-MNIST cross-task (previous, 8 seeds): ESN +27pp vs NS-RAM +5pp
- NARMA-10 (previous, 30 seeds): NS-RAM 0.612, ESN 0.563
- NARMA-5 (n=5): NS-RAM 0.623, ESN 0.537 — ESN strict win
- NARMA-10 (n=5): NS-RAM 0.663, ESN 0.567 — ESN strict win
- NARMA-20 (n=5): NS-RAM 0.986, ESN 0.880 — tie (both chance-level)
- Memory Capacity total (n=5, k=1..100): NS-RAM 1.751, ESN 1.973 — ESN
- NARMA-10 at N ∈ {100, 200, 500, 1000} (n=5 each): 0/4 NS-RAM wins
- Mackey-Glass h=6 (n=5): NS-RAM 0.193, ESN 0.067 — ESN strict win
- Mackey-Glass h=12 (n=5): NS-RAM 0.074, ESN 0.049 — tie (overlap)
- Aggregate: 11 head-to-head matrix cells; 0 NS-RAM strict wins,
  8 ESN strict wins, 3 ties (all in regimes where both reservoirs
  flounder).

## What is locked

Mario brief v4.3 is final: silicon-energy 10× advantage, ESN-class
NARMA-10 accuracy at the silicon-energy floor (honest "close but not
better"), 3-source physics triangulation ≤ 0.51 dec. The V_G2-bridge
and reservoir-quality-claim hypothesis families are closed by honest
negative results.

## What is queued but unrun

STEP E of the matrix plan: NS-RAM hyperparameter sweep (g_VG2,
leak, dt) on the best-margin task. Unlikely to flip the pattern but
not yet exhausted.

## Three questions

**Q1.** Across the 11-cell head-to-head matrix and the two
V_G2-continuum tests, did anything CROSS a decision gate that should
trigger user action (send the brief, retract a claim, change the
ecosystem-fit framing)? The user-side bottleneck is sending Mario v2
+ Sebas requests, all unsent for 6+ days.

**Q2.** Strongest cherry-picking or statistical pitfall I might be
missing in this window? Specifically:
  (a) The ESN baseline used a fixed config (sparse 10%, spectral
      radius 0.9, leak 0.30, input gain 1.0). Could a fair-but-default
      ESN actually be *over-tuned* relative to NS-RAM's
      reservoir-default-state? Should I sweep ESN hyperparams too?
  (b) The NS-RAM reservoir uses a body-state surrogate that converged
      against pyport across reservoir biases. But the surrogate's V_b
      time-step is dt=500ns; that may be too coarse to capture the
      ~1ms body-RC peak we measured. Does this systematically
      handicap NS-RAM at task timescales shorter than the body-RC
      window?
  (c) The N=200 scale was chosen because it's where most past work
      ran. Should I sanity-check at small N (e.g. N=30, 50) where
      ESN can't keep up but a stateful single-cell-per-feature might?

**Q3.** Given the 11-cell consistent negative pattern, the V_G2
continuum closure, and the locked Mario v4.3 brief, what is the
single highest-value experiment for the next 12 hours? Options on
the table:
  (i) Run STEP E (NS-RAM hyperparam sweep) for completeness.
  (ii) Sweep ESN hyperparams too on one task to check that NS-RAM's
       loss is not because we accidentally over-tuned the ESN.
  (iii) Pivot to NS-RAM as a NON-reservoir primitive (compact
       stateful trigger, PUF, chaotic oscillator, programmable
       nonlinear filter) — a different research program.
  (iv) Skip compute, write the final Mario-send-decision document
       laying out the brief, the limitations, and the next steps for
       the user to action.

Pick ONE. Be candid: where is the case weak, and what should we be
defending vs not defending?
