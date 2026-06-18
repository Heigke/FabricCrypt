# O76 — Vision Audit: What physics are we missing on Mario's 2T NS-RAM?

## Context

We are modelling Mario/Sebas's NS-RAM 2T cell (TSMC 130nm, M1 + M2 series with floating bulk node, parasitic vertical NPN through deep N-well). Our compact model is BSIM4 channel for M1/M2 + Gummel-Poon NPN for the parasitic + body-charge ODE + impact-ionisation injection (Mario `Iion` PWL).

**Current best result (z432 pseudo-transient body integration):**
- Cell-wide log-RMSE: **1.027 dec (backward sweep)**, 1.349 dec (forward sweep)
- Per-branch best: VG1=0.4 backward = 0.521 dec; VG1=0.6 backward = 1.028 dec; VG1=0.2 backward = 1.353 dec
- Mean hysteresis log-gap forward↔backward: **0.45 dec**
- We are **~1.0 dec from publication-quality fit** (target ≤0.3 dec per branch)

**What works:**
- Snapback SHAPE is captured (knee is present)
- Bistability is present (forward ≠ backward in our model)
- Backward sweep is much closer to measured shape than forward

**What does not work:**
- Knee position is shifted **~0.2 V too low** in V_D
- Magnitude at high V_D is **~1 decade under** measured
- VG1=0.2 + high VG2 shows a **false snapback** in the model (measured: smooth monotonic, no knee)
- VG1=0.4 and VG1=0.6 forward sweeps are "skakigt" (jittery, multi-valued artefacts)

**Things we have already tried (~50 variants, none broke below 0.9 dec):**
- V_SINT_PIN (pin source-internal node)
- Mario `Iion` PWL digitisation
- BSIM4 GIDL (BSIM4 §6.2)
- Pseudo-transient body integration (z432 — current best)
- 2D PWL surface for `Iion(V_D, V_DS, V_B)`
- SCR / lateral PNP injection
- λ-homotopy (continuation in injection strength)
- Standalone snapback subcircuit
- M2 body-shunt term
- And many more — best cell-wide RMSE = **0.92 dec**

## Attached materials

1. `overlay_VG1_0p2.png` — model vs measured I_D(V_D) family at VG1 = 0.2 V (sweep VG2)
2. `overlay_VG1_0p4.png` — same at VG1 = 0.4 V
3. `overlay_VG1_0p6.png` — same at VG1 = 0.6 V
4. `hysteresis_check.png` — forward vs backward sweep overlaid for one bias
5. `mario_iion_formula_12_26.jpeg` — Mario's hand-written `Iion(V_D, c, f)` formula from Zoom slide
6. `2tnsram_simple.asc` — LTspice schematic netlist (text)

Please look carefully at the images (especially the overlays AND the formula slide).

## The three questions

### Q1 — WHERE EXACTLY is the residual coming from?

Looking at each overlay plot in turn (VG1 = 0.2, 0.4, 0.6), classify the dominant residual per VG1 branch as one of:

- (a) wrong knee position only (horizontal shift along V_D)
- (b) wrong magnitude scaling only (vertical shift in log I_D)
- (c) wrong functional shape (curvature, slope, asymptote wrong)
- (d) genuinely missing physics (a feature in measurement that model can't produce in principle)

Be **specific per VG1 branch**, and identify which VG2 sub-curves are worst.

### Q2 — Hysteresis: relaxation oscillator OR true bifurcation?

`hysteresis_check.png` shows forward sweep deviates from measured but backward sweep follows it. Two competing hypotheses:

- **(A) Relaxation-oscillator / DC-averaging artefact**: the floating body oscillates on µs–ms timescale; measurement averages, our pseudo-transient picks one branch
- **(B) True bistability / saddle-node bifurcation**: two stable DC solutions exist, sweep direction selects which one

How would you distinguish A vs B operationally — what targeted experiment or simulation diagnostic (e.g. AC small-signal, transient noise injection, eigenvalue of Jacobian, Lyapunov, sweep-rate sensitivity)?

### Q3 — Mario's `Iion` formula slide (image 12.26)

Look at the slide carefully. Our digitised PWL only uses a single `c` constant and a piecewise breakpoint at `V_D = -f`.

- Does our digitised PWL capture the FULL formula, or is there a subtlety we missed?
- Is the condition `if V_D > -f` interpreted correctly?
- Is `c` really constant, or might it be a function of V_DS, V_B, or temperature implicit in Mario's notation?
- Are there exponents, prefactors, or sign conventions we may have transcribed wrong?

Read the slide carefully and respond with what the formula actually says, then compare to our (PWL: `c · max(0, V_D + f)` with a knee at V_D = -f).

## Answer format

Please answer each of Q1/Q2/Q3 explicitly with a short heading. End your response with:

- **Top 3 concrete actions** to try next (ranked by expected dec reduction)
- **Confidence** in each diagnosis (low/med/high)
- One **kill-shot diagnostic** that would falsify your hypothesis quickly
