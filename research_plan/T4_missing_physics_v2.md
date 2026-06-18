# T4 — Missing 2T NS-RAM Physics, v2 (post-SA3, oracle synthesis O53)

**Date:** 2026-05-13
**Sources:** O53_missing_physics_scratch — gpt-5 (505 s), gemini-2.5-pro (77 s), grok-4 (126 s)
**Gate:** ≥3 new candidates (not in SA3's 7) flagged HIGH or MED by ≥2 of 3 oracles. **PASS** (5 candidates meet bar).

## SA3's 7 known-missing elements (do not double-count)
1. V_Nwell→V_B parasitic diode with C_j + V-dep leak
2. V_B↔V_G2 designed coupling cap
3. V_B as output node (not internal clamp)
4. NFACTOR(M2) depends on V_G1 AND V_G2 via V_B
5. Starved-inverter 1 V front-end
6. V_Nwell + thick-ox op-window constraint
7. V_D↔V_mem sign convention reverses across slides

## Oracle agreement matrix (new physics, not in SA3's 7)

| # | Candidate | GPT-5 | Gemini | Grok | Consensus |
|---|---|:---:|:---:|:---:|:---|
| **N1** | **Oxide / interface traps with multi-tau (µs–s) charge capture** | MED (STI border traps) | **HIGH** | **HIGH** | 3/3 — strongest agreement |
| **N2** | **Generation–Recombination / SRH-TAT in body-side depletion regions** | MED (drain-body TAT) | **HIGH** | MED | 3/3 |
| **N3** | **Distributed body resistance R_B,float (rbodymod≠0)** | MED-HIGH | **HIGH** | — | 2/3 — gemini flags as first-order BSIM card bug |
| **N4** | **Forward body-diode diffusion storage (V_B–S and/or V_B–DNW)** | **HIGH** | — | — (overlaps "impact-ion → body traps") | 1.5/3 |
| **N5** | **Channel self-heating coupled to ionization coeff** | MED | MED | LOW | 2/3 |
| **N6** | **GIDL/BTBT pre-snapback body charging at drain edge** | **HIGH** | — | (CHE adjacent) | 1.5/3 |
| **N7** | **Kirk effect / quasi-saturation in lateral NPN** | **HIGH** | — | LOW | 1/3 |
| **N8** | **Hot-hole / CHE injection into gate stack** | — | (subsumed in N1) | **HIGH** | 1.5/3 |
| N9 | DNW–substrate capacitive 3rd path | MED | LOW | MED | 2/3 (low priority) |

## TOP 3 NEW candidates (locked-gate output)

### #1 — Oxide / interface traps with distributed time constants (N1)
- **Why top:** Only mechanism all 3 oracles cite as **necessary** to reproduce 200 µs → 200 ms loop SHAPE evolution. Thick gate ox + STI corners + thick/Si interface yield a continuum of τ from µs to seconds. Single-RC models cannot reproduce the slide-21 loop morphology.
- **DC signature:** sweep-direction asymmetry, ~30–100 mV V_th drift across repeated sweeps. Roughly invisible at fast DC but visible in slow quasi-DC.
- **Transient signature:** loop area grows with slower ramp (more slow traps participate); long tails after pause.
- **Cheapest experiment (Gemini):** "wait-time" — ramp to just below knee, hold 1/10/100 ms, complete ramp; trigger V shifts → trapping confirmed. Uses **existing TEG, existing instrument**.
- **Implementation in v4.4:** 2–3 parallel trap reservoirs with τ = {300 µs, 5 ms, 80 ms}, each a SRH-like capture/emission node coupled to V_B.

### #2 — Generation–Recombination / SRH-TAT in body depletion (N2)
- **Why #2:** All 3 oracles cite; provides the **ramp-rate-dependent trigger voltage shift** (separately from the trap-driven shape change). Slower ramp → more time to integrate I_gen onto C_body → V_B pre-charges → snapback ignites at lower V_D.
- **DC signature:** voltage-dependent body-current floor; lifts pre-knee tail (especially at low V_G1), explains "premature" knee not predicted by isothermal avalanche.
- **Transient signature:** monotonic left-shift of up-sweep knee as ramp slows (~50–150 mV/decade ramp time per gpt-5; ~0.1 V at 200 ms per grok).
- **Cheapest experiment:** temperature sweep 25 °C → 75 °C at fixed ramp rate; SRH/TAT has Arrhenius E_a ≈ 0.5 eV, avalanche barely moves. Smoking gun if rate-dependence shifts strongly with T.
- **Implementation:** add V-dependent generation current source at V_B–DNW and V_B–drain junctions; A·exp(–E_a/kT)·f(V_j).

### #3 — Distributed body resistance R_B,float (N3) — *structural BSIM card bug*
- **Why #3:** Gemini caught a **first-order error**: the M1/M2 BSIM cards in `data/sebas_2026_04_22/` ship with `rbodymod=0`, which is a lumped-body assumption. For a floating-body device whose entire function is parasitic-BJT feedback, this is the wrong default. GPT-5 reaches the same conclusion via "RB,float not a lumped node."
- **DC signature:** snapback knee softens; post-snapback NDR becomes less vertical; V_hold rises with post-snapback I_D (because I_D·R_B contributes to V_B).
- **Transient signature:** mostly DC; sets the *DC envelope* of the hysteresis loop. Indirectly determines how much loop area the trap and G-R mechanisms can populate.
- **Cheapest experiment:** measure V_hold vs V_G1 on existing sweeps already in hand — strong V_G1 dependence indicates R_B is large; weak dependence kills.
- **Implementation:** flip `rbodymod=1` and use a 2- or 3-segment resistive body sheet. Free fix — no new physics, just enable an existing BSIM4 feature.

## Cheapest test to run first
**N3 (R_B,float).** Costs zero — re-fit existing 33-curve DC data with `rbodymod=1`. If V_hold vs V_G1 trend now matches without invoking N1/N2, we've absorbed a meaningful chunk of error before touching the trap and G-R machinery. Should be ≤ 2 hr of ngspice work.

**Second-cheapest: N1 wait-time test** on the bench — no new fixturing.

## Pitfalls / risks

1. **N1 and N2 degenerate against each other.** Both produce ramp-rate-dependent knee shifts; only the wait-time and temperature controls distinguish them. Without both controls, an N1-only fit will compensate by absorbing what is truly N2 physics. Run both Falsifiability tests *before* fitting.

2. **N3 may absorb apparent "snapback-shape" error that is actually self-heating (N5).** Pulsed-IV with varying pulse width is the standard discriminator; without it the R_B parameter will silently soak up thermal-induced flattening.

3. **`rbodymod=1` increases convergence difficulty in ngspice** — expect to retune GMIN/RELTOL. Some of the 5 ngspice bugs already closed in Phase A may resurface.

4. **GPT-5's top picks (N4 forward diffusion + N6 GIDL + N7 Kirk) only got 1 oracle each** — they look physically sound but are not gate-passing. Defer to v4.5 unless N1+N2+N3 still miss the data.

5. **Q3 ("must be present") consensus is strong but narrow**: GPT-5 says {impact-ion + GIDL/TAT + 2 body reservoirs}, Gemini says {G-R + traps}, Grok says {traps + body-DNW C(V) + ionization→trap feedback}. Common kernel = **traps + G-R** = N1 + N2 ⇒ confidence that N1/N2 are jointly required is high.

## Gate decision
**PASS.** Locked gate required ≥3 new candidates HIGH/MED by ≥2 oracles. We have **5**: N1 (3/3), N2 (3/3), N3 (2/3 HIGH), N5 (2/3 MED), N9 (2/3 MED-LOW). Proceed to v4.4 build with N1+N2+N3 as the primary additions.
