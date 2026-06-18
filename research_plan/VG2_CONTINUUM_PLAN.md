# V_G2 Continuum Study — Multi-step Plan

Created 2026-05-10 after O39 oracle consensus. Authoritative plan for
the next ~2 working days of autonomous compute.

## Background (one-paragraph context)

User's hypothesis: a continuous V_G2 trajectory could morph a
computation gradually from "universal calculator" territory (V_G2
grounded, vanilla 0/1 MOSFET) into "hardware-rooted analog" territory
(V_G2 floating, body charge dynamics + parasitic-NPN spike generator).
The philosophical "identity rooting" framing matters less than the
concrete physical question: does smooth V_G2 traversal produce a
well-defined family of compute regimes that are smoothly parametrised
between fully-digital and fully-LIF, with measurable functional
differences from a step-switched version?

O39 3-oracle consensus said YES, the hypothesis is testable in-silico,
and converged on the same experimental program below.

## My own opinion (preserve through autonomous loop)

The most defensible near-term form of the bridge is NOT "morph an LLM's
thought in flight" — that's the philosophical framing and is hard to
ground. It's the **mixed-mode fabric** form: a network where some
tiles are deliberately grounded (digital memory, stable, predictable)
and others are floating (analog LIF, stateful, adaptive), wired
together. This is the credible architectural pitch and matches how
biological brains mix stable long-term storage with adaptive
short-term firing.

The hysteresis-rate test (STEP 1) is the right first experiment because
it decides whether there is ANY dynamical content to the continuum at
all, before we invest in the harder training/network experiments.

## Steps

### STEP 1 — Rate-dependent hysteresis (gemini O39 pick)
Script: `scripts/z244_vg2_hysteresis_rate.py` (to create).
Single-cell pyport at V_G1=0.4, V_d=0.5 DC. V_G2 triangular wave
[-0.2, +0.5] V swept up then down, T_ramp ∈ {1ns, 10ns, 100ns, 1µs, 10µs}.
Implicit-Euler transient solver; quasi-2D body; floating M2 body
(`m2_body_gnd=False`).
Output: hysteresis loop area in (V_G2, I_d) and (V_G2, V_b) projections,
plotted vs ramp rate (1/T).
Wall time: ~2 hours CPU.
**Acceptance**: monotonic shrinking loop area with longer T_ramp → PASS
(continuum has dynamical content). Identical up/down traces at all rates
→ FAIL (kill smooth-morph story, pivot to mixed-population).

### STEP 2 — Trainable smooth V_G2 schedule vs hard step (openai O39 pick)
**Only run if STEP 1 PASS**.
Script: `scripts/z245_vg2_trainable_schedule.py`.
Single-cell NARMA-10 reservoir. Smooth arm = cubic B-spline V_G2(t)
with 8-12 knots, Adam optimised through implicit solver. Step arm =
detached hard step at t=T/2.
Wall time: ~10-16 hours CPU.
**Acceptance**: smooth-arm test NRMSE strictly better than step arm
AND ||∂L/∂θ||₂ stable across iterations.

### STEP 3 — Mixed-population network (parallel with STEP 2)
Always run, independent of STEP 1 outcome (mixed-mode fabric is
robust to the continuum question).
Script: `scripts/z246_mixed_population.py`.
N=200 cells, fraction f ∈ {0, 0.25, 0.5, 0.75, 1.0} grounded
(V_G2=0, digital memory mode); rest floating (LIF mode). Wired in
sparse Erdős-Rényi recurrent fabric. Task: NARMA-10, 10 seeds per f.
Wall time: ~4 hours CPU.
**Acceptance**: f=0.5 or f=0.25 outperforms f=0 (pure analog) and
f=1.0 (pure digital) by ≥3 pp NRMSE → mixed-mode fabric is a real
architectural win, headline-worthy.

### STEP 4 — Synthesis note
After at least 2 of {STEP 1, 2, 3} complete, write
`research_plan/VG2_CONTINUUM_FINDINGS.md` summarising:
- What we tested
- What we measured (loop areas, NRMSEs, gradient norms)
- What this means for the ecosystem-fit pitch
- What it does NOT support (overclaim guard)

### STEP 5 — Mario brief v4.4 (conditional)
If STEP 3 PASS, add a forward-looking section to
`nsram/main-4.tex` about "mixed-mode NS-RAM fabrics" as the
post-NRF research direction. Do not lead the brief with it;
keep silicon-energy + NARMA-ESN-class + R-track as headlines.

## Acceptance gates summary

| Step | Wall | Pass gate | Outcome if PASS | Outcome if FAIL |
|---|---|---|---|---|
| 1 | 2h | Loop area shrinks with T_ramp | go to STEP 2 | kill smooth morph, do STEP 3 only |
| 2 | 10-16h | NRMSE_smooth < NRMSE_step + stable grads | major story upgrade | quiet failure, STEP 3 still matters |
| 3 | 4h | f∈(0,1) beats both pure modes by ≥3pp NRMSE | mixed-mode-fabric headline | discrete mixed-mode is not a real arch |
| 4 | 30min | always run | synthesis doc | — |
| 5 | 1h | only if STEP 3 PASS | v4.4 has mixed-mode forward section | brief stays at v4.3 |

## Cron management

Old crons that referenced the now-stale RESEARCH_PLAN_2026-05-07 work-hours
priority queue have been removed. New crons:
  - hourly :17 — lightweight check-in (5 min budget), monitors progress
  - every 2h work-hours (09,11,13,15,17,19,21 :23) — V_G2 continuum runner

Both expire automatically after 7 days. Other audit crons (resource,
baseline watchdog, daily synth, GPU off-hours, oracle 12h) are
preserved.

## NO-CHEAT PRINCIPLE (added 2026-05-10 after STEP 1 v1)

Two rules that override convenience:

1. **Never bend an acceptance gate after seeing the result.** If a gate
   fails, it fails. If the physical interpretation argues the gate was
   misspecified, RE-RUN with a corrected, pre-specified gate. The
   reinterpretation does not become the result.
   - Action on z244 v1: pre-register a new gate
     ("hysteresis loop area exceeds noise floor for at least one
     T_ramp AND the peak T_ramp is within one decade of the predicted
     τ = C_b · R_b ≈ 1ms"), then run again with replication.

2. **Go full, no shortcuts.**
   - Multiple seeds per condition (n ≥ 5 minimum, 10–30 preferred).
   - Proper variance bars; report mean ± std and 95% CI.
   - Wider parameter sweeps than the minimum needed to "win" — at
     least one bracket above and below the expected sweet spot.
   - Never compress a "single-seed pilot" into a brief without
     explicit replication at proper n.

These two rules apply to every subsequent step in this plan and to
every brief/figure that comes out of them.

## STATUS (final)

- STEP 1v2 hysteresis-rate: ❌ FAIL (5x contrast vs 100x gate).
- STEP 2 trainable schedule: NOT RUN (blocked by STEP 1).
- STEP 3 mixed-population: ❌ FAIL (best-mix edges pure-floating
  by 0.006 NRMSE, far below 0.016 margin).
- STEP 4 synthesis: implicit in 01_LOG.md entry.
- STEP 5 v4.4 mixed-mode section: BLOCKED.

Plan superseded by NEXT_DIRECTION_PLAN.md (NS-RAM vs ESN matrix).
NO-CHEAT was enforced; both gates failed honestly; we accept and move on.
