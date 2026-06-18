NS-RAM project critique. Today's R-1..R-7 topology rebuild sequence.

State: After 5 audits (R-1 mail, R-1b 31 images via vision, R-3 pyport code,
R-deep-A LTSpice vs pyport, R-deep-B oracle 3-way structural), we identified:
- D1: Q1 emitter mis-wired (GND vs LTSpice Sint) — FIXED
- D2: triple-counted diodes (LTSpice has 0 explicit) — FIXED  
- D9: Bf=10000 from card (we had 50/3000) — already correct
- v5 INFRASTRUCTURE WIN: 5/5 unit tests pass (flags wired)

THEN R-7 ran Stage 1 liveness ablation at V_G1=0.6:
- V1 control (D1+D2+D9 applied) = 3.248 dec
- V2 +well_diode = 3.248 BITWISE IDENTICAL
- V3 +iii_kill (alpha0=1e-20) = 3.248 bitwise
- → BODY BRANCH STILL DEAD. Vb≈Vsint≈0 fixed point where everything cancels.

z304's 0.99 dec is now revealed as SPURIOUS local optimum that relied on
non-physical Bf=3000 + avalanche crutch + K1(V_G2) DOF — REMOVED these
crutches per Sebas mail recipe → exposes structural incompleteness.

We conclude pyport's body-KCL has NO ACTIVE CURRENT SOURCE driving Vb up.
v4.4 brief stays on HDC + RNG headlines (network insensitive to DC bias
per z319 defense).

Three sharp questions:

Q1 FRAGILITY: Have we overclaimed today's diagnosis as "we know exactly
what's wrong"? Sebas's BSIM4 + parasiticBJT IS Mario's known-good 
combination — yet our implementation can't make body live. Either:
(a) Sebas's published combination only LOOKS sufficient — really 
    requires the avalanche/Chynoweth crutch z304 had
(b) Our pyport BSIM4 implementation is missing IMPACT_IONIZATION wiring
    to body (R-7 saw 0 effect from iii_kill — that's a clue)
(c) The "body branch" we're staring at is actually fine and the real bug
    is in M2 channel current vs measured
Which is most likely?

Q2 SINGLE FALSIFICATION: What is the cheapest 30-min experiment that
DISTINGUISHES Q1's three hypotheses (a/b/c)?

Q3 NO-CHEAT: Have we drifted today? Specifically: we declared 
"v4.4 stays valid on HDC + RNG" while announcing the model is 
"structurally incomplete". Is this consistent? Or have we silently 
re-defined "valid" to mean "we're not actually claiming the model 
works, just the networks do"?

Be sharp. ≤500 words per oracle.
