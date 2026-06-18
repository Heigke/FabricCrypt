# O77 — Physics Completion Strategy (3-way Oracle)

## Role
You are reviewing the NS-RAM cell physics-completion strategy for a paper aiming to publish a quantitative model of the IBM NS-RAM (neuromorphic SRAM) cell. The team has hit a wall: Sebas's transient measurement data (A.12) has been blocked for 3 weeks. We need a path forward.

## Context (attached files)
- `CAMPAIGN_SYNTHESIS_2026-05-16.md` — master synthesis of the current physics campaign
- `HONEST_BASELINE_2026-05-16.md` — what the cell model actually does today, no spin
- `z454_honest_analysis.md` — snapback integration result (ns-snap works)
- `z456_honest_analysis.md` — R_body reset KILL_SHOT under SB_HOT
- `z457_honest_analysis.md` — NPN gate works at V_db > V_knee
- `cap_breakdown.json` — parasitic cap audit (from z451)
- `01_LOG_tail.md` — last 250 lines of the campaign log
- `O76_*.md` — the previous critique round (prompt + 3 oracle responses) for context on what's already been argued

## Available materials (everything we have / can use)
- 33 IV curves DC (measured, used)
- M1/M2 BSIM4 v4.5 cards (fitted, used)
- Parasitic NPN card: Bf=10⁴, τ_F=25 ps (fitted, used)
- Mario slide 12 PWL I_pos coefficients (digitized — but maybe undersampled in the V_DS region of interest)
- Mario slide 21 qualitative oscillation waveform ~400 ns period (NOT yet used as a quantitative target)

## Blocked
- A.12 transient I(t)/V(t) traces from Sebas — 3 weeks no response, treat as permanently unavailable

---

## Q1 — Physics completion path WITHOUT A.12 data

We can never get A.12. Given ONLY existing materials + educated guesses from published 22FDX/14LPP NS-RAM literature, what is the **ONE highest-leverage experiment** that would close the most physics gaps?

Constraints on your answer:
- Pick **ONE concrete experiment** (script-runnable in our ngspice/Python flow). Don't list 5 things and call it a strategy.
- Justify why it **dominates** the other obvious candidates (re-fit PWL with finer V_DS sampling, sweep snap_Is grids, push BSIM4 v4.7 → v4.8, parasitic L_par sweeps, etc.)
- Give **concrete numerical targets** — what would PASS look like? What value of MC, NRMSE, or qualitative-match score?
- Be specific about which existing materials are reused, and which "literature priors" you would plug in for missing parameters (cite expected ranges).

## Q2 — Innate LIF closure (z458 design review)

Cell level today: integrate-and-fire works — ns-snap fires (z454), but self-reset fails (z456: R_body reset is KILL_SHOT under SB_HOT). z457 showed NPN gating works at V_db > V_knee.

**Proposed z458**: 4×4 grid (snap_Is × R_body) on top of v449_B + NX_1p8, hunting for a regime where:
- snap fires (V_db rises past V_knee → NPN turns ON)
- V_B is then drained through NPN
- post-snap V_db collapses → NPN turns OFF
- R_body then quietly drains V_B back toward V_SS (refractory)

Questions:
- Is the 4×4 grid the right resolution? Or do we need 8×8 / log-spacing?
- What is the **most likely failure mode**? Predict concretely:
  - (a) Latch-up — NPN never turns off, V_B pinned high
  - (b) Sub-knee snap — V_db never reaches V_knee, NPN never gates on
  - (c) Race condition — R_body drains V_B before NPN can fire
  - (d) Multi-fire / oscillation — but uncontrolled period
- Give a numerical prediction for the success-rate of the grid (out of 16 cells, how many produce a clean single LIF spike + return to baseline within 1 µs?).

## Q3 — No-A.12 publishability

If we accept that A.12 will never arrive, what is the **strongest defensible claim** we can publish about NS-RAM cell dynamics using ONLY existing materials + a **qualitative** match to Mario slide 21's ~400 ns oscillation?

Be specific:
- Which **figures/metrics** make it into the paper? (DC IV fit decade, snapback waveform overlay vs Mario, NPN gating region map, LIF spike waveform, …)
- Which **claims MUST we avoid** because they require transient ground-truth? (e.g., "we reproduce I(t) to within X% over 0-1 µs" — obviously out)
- What **language softening** do we apply to claims that are circumstantial? ("qualitative match", "consistent with", "to within order-of-magnitude")
- What is the realistic **venue** — workshop paper, short letter, full conference? If we omit transient validation entirely, does the paper still merit a top venue, or does it drop a tier?

---

## No-cheat rules
- **Concrete numbers, not generic advice.** "Do a sensitivity sweep" is useless. "Sweep snap_Is in [1e-9, 1e-6] A at 8 log-spaced points, target NRMSE < 0.15 on slide-12 PWL between V_DS=0.4-1.2 V" is useful.
- **Disagreements between oracles are valuable.** Don't try to be safe. Take a position.
- **Flag if any oracle hedges too much** — but since you don't see the other two answers, just commit to your own position firmly.
- **Cite literature priors when you invoke them** — paper + year + parameter range. We will fact-check.

Return your answer as 3 sections (Q1, Q2, Q3), each ending with a 1-line VERDICT (recommendation + confidence 0-1).
