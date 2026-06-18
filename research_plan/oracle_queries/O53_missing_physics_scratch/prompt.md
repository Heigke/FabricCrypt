# O53: What 2T NS-RAM physics are we STILL missing beyond SA3's 7 elements?

You are a senior analog/device-physics oracle. Think **from scratch**. Do
not just rephrase what we already know — your value here is to surface
physics we have NOT yet identified.

## System under study
A 2T NS-RAM cell fabricated in imec 130nm, thick-oxide nMOS pair:
- **M1** = spike-generator, parasitic-BJT regime, snapback at V_d > ~2 V
- **M2** = readout transistor
- **Floating bulk** isolated via deep N-well (DNW), body charge holds memory
- Starved-inverter 1 V CMOS front-end on the chip
- Measured behavior includes snapback (DC) and multi-rate transient
  hysteresis (200 µs → 200 ms V_d ramps, slide 21)

Attached:
- `SA3_image_deep_extract.md` — deep extraction from the SA3 figure
- `M1_130DNWFB.txt`, `M2_130bulkNSRAM.txt` — BSIM4 cards (130 nm)

## What SA3 already told us is missing in our current pyport (do NOT repeat these)
SA3's 7 already-identified missing/wrong elements in our model:
1. V_Nwell → V_B parasitic diode with junction capacitance C_j and
   voltage-dependent leakage
2. Designed V_B ↔ V_G2 coupling capacitor
3. V_B exposed as an **output node**, not an internal clamp
4. NFACTOR(M2) depends on **both** V_G1 and V_G2 via V_B coupling
5. Starved-inverter 1 V CMOS front-end shaping the input pulses
6. V_Nwell + thick-oxide constraints on the operating window
7. V_D ↔ V_mem mapping **reverses** across slides (sign convention bug)

Your job: surface **additional** candidate physics BEYOND these 7. Focus
on (a) snapback shape at V_d > 2 V and (b) transient (200 µs to 200 ms)
hysteresis behavior.

## Three questions

### Q1 — Channel-side physics modulating snapback shape
What channel-level effects might shape the snapback curve that we are
not yet modeling? Candidates to evaluate (and add your own):
- DIBL at floating body
- Hot-electron channel injection (CHE)
- Drain-end impact-ionization **spatial profile** (not just rate)
- Velocity saturation feedback into V_DS,eff
- Gate-drain overlap capacitance modulation
- Punch-through / drain-induced punch-through assist
- Self-heating of the channel (localized lattice T rise)
- Substrate current feedback through R_B,float
- Quasi-saturation in the parasitic NPN

### Q2 — Body-side physics governing transient (ms) response
What body-side / bulk physics governs the ramp-rate-dependent
behavior? Candidates:
- Body-charge persistence between sweeps (memory effect)
- Body→N-well leakage with temperature dependence
- Multi-level / multi-time-constant traps in the thick gate oxide
- Hot-hole injection into the oxide (charge trapping in gate stack)
- Gate leakage at high V_G2
- DNW capacitive coupling to substrate ground (third coupling path)
- Generation–recombination in the depletion region (SRH / TAT)
- Surface-state traps at thick-ox / Si interface (fast traps)

### Q3 — Slide-21 ramp-rate hysteresis
Slide 21 (described in SA3_image_deep_extract.md) shows a hysteresis loop
that changes shape with V_d ramp rate from 200 µs to 200 ms. Which
physics MUST be present (not just "could be present") to reproduce that
loop shape? Be specific: which time constants, which charge reservoirs,
which feedback paths?

## Output format

For **each candidate** you propose (≥3 candidates each in Q1 and Q2, plus
your Q3 short list), report exactly these fields:

```
### CANDIDATE: <name>
- **Mechanism (1 sentence):**
- **DC signature** in the 33-curve DC sweep (V_G1, V_G2, V_Nwell ladder)
  — would this fix anything we currently can't reproduce? Be concrete.
- **Transient signature** — predicted shape of rate dependence (which
  way does the hysteresis loop open/close as ramp slows from 200 µs to
  200 ms?)
- **Falsifiability** — one cheap measurement that would confirm or kill
  this candidate. Prefer experiments doable on the existing TEG.
- **Priority for v4.4 model rebuild:** HIGH / MED / LOW, with one-line
  justification.
```

End with a `## SUMMARY` block: bullet list of your top-3 picks ranked,
each with one-line rationale.

## Rules

- **Think from scratch.** Do not anchor on SA3's 7. Surface things SA3
  missed.
- ≥3 new candidates per question (so ≥9 candidates minimum).
- If a candidate overlaps with SA3's 7, mark it `[OVERLAP: SA3#N]` and
  skip the deep evaluation — we want **new** physics.
- Be specific and quantitative where you can (time constants in ms,
  voltages in V, currents in A).
- No hedging. If you think something is LOW priority, say so.
