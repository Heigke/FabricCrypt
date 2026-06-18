# O32 — Path forward after MC=1 discovery + run-through-night plan

## Recap of today's findings (single autonomous session, 8 wake-ups)

z211/212/213/214/216/217/218 chain:
- **z210/z211 inhibition "win" was lottery**: at 6 seeds saw +18pp;
  at 20 seeds (z213) it's +3.5pp p=0.44 ns. Confirmed at N=64/128/256/512
  (z214: null at all sizes).
- **NARMA-10 unsolvable** on current surrogate: 54 hyperparam combos all
  give train NRMSE ≥0.77. Working ESN gets 0.4-0.6.
- **Memory Capacity = 1.0** at N=200 (z217). Theoretical max = 100-200.
  Reservoir essentially memoryless.
- **Root cause**: NSRAMSurrogate is static lookup (VG1, VG2, Vd) → Id.
  No Vb state variable. Silicon NS-RAM uses body capacitance for memory
  via parasitic NPN.
- z218 PoC: passive body-state-feature only adds +0.6 MC (1.04→1.64).
  State variable must FEED BACK into cell, not run parallel.

Brief is sent + safe — Mackey-Glass result holds (short-mem task).
But brief implies edge-AI applicability that current model can't support.

## Question

Five candidate paths forward; rank them and recommend sequence:

**A** — Build 4D transient surrogate `(VG1, VG2, Vd, Vb) → (Id, Iii)`
       with time-stepped Vb dynamics. Use pyport's `solve_2t_steady_state`
       with `Vb_init` override. ~2 days. Parameters Cb, τ guessed from
       literature until Sebas data arrives.

**B** — Wait for Sebas measurements (Ic/Ib + pulsed-Vd τ) before A.
       Drafts ready 3 days, NOT YET SENT. Could be 1-4 weeks delay.

**C** — Pivot to hetero-cell exploration on MG-vs-sin (current surrogate
       works for short-mem). 1-2 days. Different axis (cell mix), tests
       gpt-5+gemini's O30 #1.

**D** — Use pyport directly (no surrogate) at small N=32. Slow (10x)
       but accurate. Validates surrogate-vs-direct delta in sanity check.

**E** — Reframe brief to clearly bound edge-AI claims; send Mario an
       update note. No new compute. Re-baseline expectations.

## Specific asks

1. **Ranked sequence for next 7 days**: which path first? In what order
   do A-E fire? Justify each.

2. **Parameters for path A** (transient surrogate): what defaults to
   use for Cb (body capacitance) and τ before Sebas data lands?
   Literature points for 130nm parasitic body cap?

3. **Cron strategy**: I'm running 10 crons covering work-hours, night-
   time-GPU, daily synthesis, mid-week risk, weekly review, baseline
   watchdog, oracle 12h, track audit 6h, resource audit, re-arm.
   Should something change for "run constant through night"? Right
   now nighttime has 03:23 GPU off-hours + 02:13 synthesis + 04:43
   watchdog. Add more density?

4. **Honest brief-update timing**: should I tell Mario about the
   MC=1 finding NOW (and what's the right framing) or wait until
   we have a working transient surrogate AND first realistic-task
   result? Risk of waiting: he commits chip-design choices on a
   misunderstanding. Risk of telling now: feels alarmist.

5. **Killer omission catch**: what's the one thing I'm NOT seeing
   in this MC=1 → 4D-surrogate pivot that I should be?

Be terse, < 600 words total. Decisions need to fire tonight.
