# O33 — 12h feedback review (auto-cron 75726554/69d4f51c)

Last 12h activity (see context.md): pivoted from architecture-search
on existing surrogate (which gave non-significant inhibition "win"
that didn't survive 20-seed validation) to building a 4D transient
surrogate that exposes Vb body-state.

Memory Capacity progression (target: ESN-class MC≈20-100, gate >5):
  z217 (no Vb feedback):           MC = 1.00
  z218 (passive Vb feature):       MC = 1.6
  z219 (5-pt Vb 4D, VG1 input):    MC = 2.5
  z220 (10-pt Vb 4D, VG1 input):   MC = 3.73
  z221 v3 (VG2 input):             MC = 4.46
  z221 fine-tuned (g=0.05, leak=0.30): MC = 5.13  ← gate >5 crossed

Brief unaffected — Mackey-Glass result still holds for short-mem.
Mario update note drafted but not sent (waiting user decision).
Sebas request packet still unsent (3+ days).

## Questions

1. **Has any decision gate crossed in last 12h that should change
   action? Specifically: MC>5 was the cron's "continue iterating"
   threshold; should we now (a) declare PoC done and write up,
   (b) push to MC>10 before NARMA-10 retry, or (c) skip MC and go
   directly to a realistic task?**

2. **Cherry-picking risk audit**: I tuned (Cb, dt, g_VG2, leak) until
   MC=5.13 was best. Each iteration changed multiple knobs. Is this
   p-hacking the reservoir? What's the proper way to validate the
   final config?

3. **Next single highest-value experiment in next 12h**: NARMA-10
   re-attempt? Bigger N (=400)? Stateful surrogate (GRU per gemini)?
   Pyport-direct N=32 sanity? Prioritize.

Be terse, < 400 words total per oracle. Decision needs to fire
within next work-hours wake-up.
