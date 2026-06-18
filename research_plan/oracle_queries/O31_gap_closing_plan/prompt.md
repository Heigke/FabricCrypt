# O31 — Critique gap-closing plan; help lock cron schedule

## Context

Prior wake-ups found a candidate architecture win
(`ER_SPARSE + r=2/s=0.3 inhibition + ff + N=256` → best 1.000±0.000,
final 0.958±0.059 over 3 seeds). User pushed back: be CRITICAL —
this is not yet Mario-actionable due to 5 named gaps:

1. Toy benchmark (24 test samples)
2. NSRAMSurrogate (2 abstraction layers from silicon)
3. "Inhibition" is software W-pattern, needs chip mod for silicon
4. No realistic-task generalization tested
5. Statistical power weak (3 seeds)

The attached `plan.md` is a draft 6-track plan
(V/R/C/T/S/P) plus updated cron schedule. Goal: convert
the candidate win into a 1-page chip-mod recommendation Mario can act on.

THERMAL CONTEXT: my APU hits 100°C with 6 ProcessPool workers running
NSRAMSurrogate. Trip is 99°C → instant reboot. Track P aims to fix.

## Questions (be terse, < 600 words total)

1. **Track ordering**: I propose Day 1-2 = Track P (thermal-safe
   sweep wrapper) then Track V (10-seed validation). Is this right,
   or should something else be first?

2. **Track P branch**: GPU port of NSRAMSurrogate vs. safer-CPU
   wrapper (≤3 workers + thermal pause). GPU port is bigger lift
   but unblocks more. CPU wrapper is faster ramp but caps throughput.
   Which way?

3. **Validation power calibration**: I picked 10 seeds × 240 test
   samples. With expected effect size ~0.05 dec or ~5pp, is that
   enough power? What's the right N?

4. **Track C cost model**: I plan literature-cited area numbers
   for sign-inverter (~5 µm² in 130 nm). Is that the right
   defensible source? Better way to estimate cell-area cost without
   actually running place-and-route?

5. **Decision gate at end of Week 1**: I set it as "10-seed CI ≥
   +5pp = proceed". Right cutoff? Too lax / too strict?

6. **Cron schedule additions**: I want to add (5) GPU off-hours,
   (6) oracle feedback every 6h during exploration, (7) track-progress
   audit every 4h. Is this the right cadence, or am I over/under
   instrumenting? What single guardrail would you ADD that I'm missing?

7. **Killer omission**: what's the ONE most important item missing
   from this plan that I'm not seeing? Be brutal.

Goal: stamp "go" or specific fix-list. The plan must produce
something Mario can read in 5 minutes and decide on.
