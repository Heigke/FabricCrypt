# Oracle Critique Packet — z427 NS-RAM 2T Cell I-V Fit (DISCOVERY-gate crossing)

## Role for the oracle

You are an external referee. We crossed a DISCOVERY gate (cell-wide log-RMSE <2.0 dec)
on an NS-RAM 2T-cell I_D(V_D) fit against Sebas's 130nm measured data (33 bias points,
V_G1 ∈ {0.2, 0.4, 0.6} × V_G2 ∈ {0.0..0.5}). We want a critical, sceptical
referee read before we write this up. Please be blunt — the team has been on this
for ~12h and is at risk of self-confirmation bias.

## Quick context (campaign state)

Sebas measured 33 bias points on a 130nm 2T NS-RAM cell, with V_G2 acting as a
regime selector (empirically confirmed earlier — see project memory). pyport is
our Python port of the BSIM4+NPN (Gummel–Poon) cell model. Phase A previously
closed at 1.00 dec (the "honest" reading was later walked back to 1.39 dec at
Bf=100, η ≤ 1, see `nsram_m3b_corrections.md` in memory). The current campaign
adds three structural fixes on top of Phase A:

- **Mario Ipos PWL** injection at the BJT body node (substrate punch-through proxy)
- **Suppress M1 bulk forward diode** (was clamping V_B high)
- **BJT B-E one-way rectifier** (no reverse leakage when V_BE<0)

## Recent results timeline

- **z425 ALL_FLAGS_ON**: cell-wide RMSE = **3.928 dec** (KILL_SHOT). Even with all
  three fixes ON, V_B floats to 2.4V (high) and improvement is marginal.

- **z426 deep current decomposition** at V_D=2.0, V_G1=0.6:
  - V_B = 2.57 V
  - V_Sint = 2.00 V (runaway! tracks V_D)
  - V_BE = V_BC = 0.57 V → BJT in **deep saturation** → Ic≈0 by symmetry.

- **z427 V_Sint fix** — tested 4 hypotheses. The winning combo:
  - **H1**: 1 MΩ shunt M2.source → GND ("substrate-tap path") — value was *guessed*, not fit
  - **H2**: route GIDL current to Sint instead of GND
  - Combined: V_Sint drops 2.00 → 1.39 V, V_BC flips +0.57 → −0.04
    (saturation → forward-active), Ic_Q1 → +1.4 µA (right sign, right magnitude).
  - At bias point B4: per-point gap closes from 3.48 → 1.43 dec.

- **z427 COMBINED_H1_H2 cell-wide log-RMSE = 1.733 dec** (DISCOVERY <2.0 PASS;
  AMBITIOUS <1.0 NOT yet). Per-branch breakdown:

| V_G1 | Before (dec) | After H1+H2 (dec) | Δ |
|------|--------------|-------------------|----|
| 0.2  | 2.30         | **2.74**          | **+0.44 (REGRESSION)** |
| 0.4  | 3.70         | 0.56              | −3.14 (huge win) |
| 0.6  | 4.74         | 1.36              | −3.38 (huge win) |

- **Overlay-plot diagnosis** (not publication-quality yet):
  - High-V_G1: snapback shape now visible (was absent before — real physics gain).
  - Low-V_G1=0.2: model **overpredicts sub-threshold** by ~4 decades at V_D≈0.5V
    (model ~1e-9, measured ~1e-13).
  - V_G1=0.2 traces also show solver artifacts (non-converged spikes).

## OPEN QUESTIONS

### Q1 — Gate-crossing risk (statistical)

Is the headline "cell-wide log-RMSE = 1.733 dec, DISCOVERY-gate crossed" honest
given the per-branch breakdown?

- log-RMSE averages residuals on a log axis across 33 bias points; the huge wins
  at V_G1=0.4 and 0.6 (−3.14 and −3.38 dec) can easily mask a regression at
  V_G1=0.2 (+0.44 dec).
- If we computed instead a **median per-bias** residual, or a **max per-bias**
  residual, would the result still cross DISCOVERY (<2 dec)? At what
  per-bias-point granularity does the headline fall apart?
- What is the right metric for a 130nm NS-RAM I-V fit publication? log-RMSE,
  median absolute log error, max log error, or branch-stratified RMSE?

### Q2 — Cherry-pick risk (physical)

Are H1 (1 MΩ Sint→GND shunt) and H2 (GIDL→Sint routing) physically grounded, or
reverse-engineered to close the bias points where the gap was largest?

- H1's *topology* (substrate-tap path) is at least geometrically motivated, but
  the *value* (1 MΩ) was guessed, not fit. Does a 1 MΩ shunt at the body of a
  130nm device pass an order-of-magnitude smell test? What would TCAD or
  process docs predict for substrate resistance to the local tap?
- H2 (GIDL → Sint vs GIDL → GND) is geometrically reasonable (GIDL is generated
  at the drain-body junction, current physically flows through the body), but
  it's also exactly the modification that fixes V_Sint runaway. Is this a
  legitimate physics-first correction or a curve-fit?
- Should we worry about **over-fitting** with two structural mods that together
  happen to fix the two branches that were broken?

### Q3 — Next falsification experiment (single highest-value)

The remaining residual is dominated by sub-threshold V_D **over**prediction at
V_G1=0.2 (model ~1e-9 A, measured ~1e-13 A at V_D≈0.5 V; the model is 4 decades
HIGH in sub-threshold at the lowest gate bias).

What is the **single highest-value experiment** that:
- (a) tests whether H1+H2 are real physics or curve-fits,
- (b) closes (or at least identifies the mechanism behind) the V_G1=0.2 regression,
- (c) does **NOT** add more degrees of freedom (no new fit parameters)?

Concrete candidates we're considering — please rank and add your own:
1. Sweep H1 shunt resistance over 4 decades (100k, 1M, 10M, 100M Ω) with NO refit —
   does the cell-wide RMSE form a clean minimum, or is it monotonic? A clean
   minimum suggests fit; monotonic-with-plateau suggests physics.
2. Add a *blind* held-out bias group (e.g. an unseen V_G2 stripe) and re-test
   H1+H2 — do they generalise?
3. Switch H2 OFF only at V_G1=0.2 — does that branch recover (i.e. is GIDL
   routing the cause of the V_G1=0.2 regression?).
4. Replace H1+H2 with a single GIDL-only fix (no Sint shunt) — does V_Sint still
   stabilise? If yes, H1 is redundant.
5. TCAD-level structural probe (request from Sebas): what is the *measured*
   substrate-tap resistance at this geometry?

## Deliverable

For each of Q1, Q2, Q3 give us your honest read. We will diff the three oracle
responses against each other and look for points of convergence vs divergence.
Brevity preferred — 1 page of dense, technical critique per question is enough.
