# S5C — Deep visual scan of Mario/Sebas Zoom screenshots for snapback physics
Date: 2026-05-15
Scan target: image-level features (annotations, cross-sections, IV kinks, schematics)
Source: research_plan/artifacts/Zoom/ and nsram/proposal_2026_05/figures/sebs_*

## Scope reviewed
- 25 JPEGs in `research_plan/artifacts/Zoom/` (2026-03-20 ×17, 2026-04-22 ×1, 2026-04-30 ×11 — several duplicates of slides also re-shipped as `figures/sebs_*`).
- 7 `sebs_*` figures in `nsram/proposal_2026_05/figures/`.
- PDFs of proposal/brief skipped here (R-55 already covered text-level).

## Cross-section / topology consensus from imagery
Slides 2026-03-20 12.29 (1T thick-ox) and 12.30 (2T thick-ox spiking cell) supply the canonical cross-section:
- **Triple-well 130 nm CMOS**: deep N-well encloses a **floating P-pocket** ("Floating P-substrate"). M1 is an NFET built INSIDE that floating P-pocket. The P-pocket sits in a deep N-well sitting in P-substrate.
- **M2 is a SECOND NFET whose drain is wired to the floating P-pocket of M1** (= V_B node in the 12.07 / 12.05 (1) schematics). M2's source goes to ground, gate is VG2.
- So M2 is NOT a body-bias source — it is a **bias-controlled leak resistor** from the floating P-body to GND. VG2 sets how fast injected body charge bleeds away (Mario writes literally on slide 12.30: *"VG2 = -0.5 V slows down the leaky behaviour"*).

This is consistent with the Nature 2023 line *"We adjust the resistance of the bulk terminal using a second transistor"* (slide 12.09).

## Findings ranked by snapback-relevance

### F1. Two-kink snapback per transient — IMPLIES additional positive-feedback branch beyond BSIM4 §6.1
**File**: `research_plan/artifacts/Zoom/Image 2026-04-30 at 13.25.jpeg`
**What it shows**: Top: ~MHz I/V transient with current pulses (red) lagging V_D ramps (blue) by tens of ns. Bottom: two log(I)-V trajectories of single transients with **red dots marking two distinct kinks per ramp** — one near 0.3–0.5 V, another near 1.5–1.8 V.
**Physical implication**: A single BSIM §6.1 II-current with α·(VDS-VDSAT)·exp(-β/(VDS-VDSAT)) gives ONE inflection. Two kinks means **two distinct positive-feedback channels active in different V_D bands**:
  - Low-V kink (~0.4 V): likely **floating-body forward-bias turn-on** of the source–body junction (V_B raised by capacitive coupling from V_D through the channel/CGB; once V_BS > ~0.4 V the source–body diode conducts → body-current bleed reverses sign → V_B latches up).
  - High-V kink (~1.6 V): the classic II / lateral-BJT snapback.
**Test**: ngspice transient with both (a) source–body diode `Dsb` between VB and source and (b) lateral NPN base=VB, emitter=S, collector=D; sweep ramp rate `dV/dt` from 1 V/µs to 100 V/µs; the LOW kink should appear only at high dV/dt (capacitive injection wins over leak) — the HIGH kink should be ramp-rate insensitive.

### F2. ETAB(M1) and NFACTOR(M2) extracted from silicon are NOT BSIM-physical
**File**: `research_plan/artifacts/Zoom/Image 2026-04-30 at 13.24.jpeg`
**What it shows**: MATLAB fitter outputs from Sebas. Four panels:
  - BETA0(M1) vs VG2 — non-monotonic peak around VG2 ≈ 0.05 V (family in VG1)
  - **ETAB(M1) vs VG2 — sweeps 0 → 2.5, strongly *decreasing* in VG2**
  - VT1(M1) vs VG1 — drops 0.58 → 0.42 V over VG1 ∈ [0.2, 0.6]
  - **NFACTOR(M2) vs VG2 — ranges 2 → 12** (BSIM4 NFACTOR is normally pinned 0.5–2)
**Physical implication**: To fit silicon, Sebas had to push ETAB(M1) (BSIM body-effect coefficient on Iion) to values *that depend on VG2*. Inside BSIM, ETAB is a single constant per MOSFET. **The extracted ETAB curve IS the unmodeled physics** — it is absorbing whatever real mechanism couples VG2 → M1.II strength. Likewise NFACTOR(M2)=2..12 means M2 is operating well outside subthreshold MOSFET regime; it is acting as a *forward-biased diode chain / punchthrough device*, not as a clean MOSFET. **This explains why our ngspice industrial baseline (R-55 KILL-SHOT) can't reproduce snapback: we put NFACTOR≈1 and ETAB constant.**
**Test**: Rebuild model with M2 replaced by a **diode-connected subthreshold device with explicit punchthrough current branch** (BSIM4 IGB + IGC + a manual GMIN-like leakage with eta-coefficient on VG2). Sweep NFACTOR_M2 ∈ [2, 12]; the high-V snapback breadth should track NFACTOR_M2 directly.

### F3. Self-relaxation snapback at V_D = constant after pulse — points to body-charge-driven relaxation oscillator, not steady-state II
**File**: `nsram/proposal_2026_05/figures/sebs_three_operating_regimes.jpeg` + Zoom slide 12.30
**What it shows**: IV families at three (VG1, VG2) points; slide 12.30 annotation states *"V_D pulses, Self relaxation only"* and shows hysteresis loops where the cell **fires AFTER the V_D pulse ends**.
**Physical implication**: Snapback here is NOT a DC I-V phenomenon (which BSIM §6.1 covers) but a **transient relaxation oscillator**: floating-body charge accumulates during the V_D pulse, then once the leak path through M2 closes (VG2 returning), V_B is trapped at high potential → M1's Vth drops → drain current surges → spike. The order of operations matters: it is a CHARGE-storage memory cell, and snapback is the discharge event.
**Test**: ngspice mixed-signal — apply 100 ns V_D pulse, then hold V_D = 0.3 V (sub-Vth) for 1 µs, varying VG2 from -0.5 V (slow leak) to +0.5 V (fast leak). With body-charge model: expect spike during the HOLD phase that delays with VG2 → -0.5 and disappears with VG2 → +0.5. Without body-charge model: no spike (BSIM4 default).

### F4. Triple-well layout suggests parasitic vertical NPN (drain → pwell → DNW), NOT lateral parasitic
**File**: `nsram/proposal_2026_05/figures/sebs_130nm_triple_well_layout.jpeg` and Zoom slide 12.29 / 12.30 cross-sections
**What it shows**: Cross-section drawing places N+ drain in floating P-well; under the P-well is the **Deep N-Well biased at +V_DNW (≈ V_DD)**.
**Physical implication**: The parasitic BJT we have been simulating (lateral channel-NPN, base=body, emitter=source, collector=drain) is only one path. There is **also a vertical NPN: emitter = drain, base = floating P-pocket, collector = Deep N-Well**. Because DNW is biased high (+VDD) and the P-pocket is *floating*, this vertical BJT can latch up when injected body current forward-biases drain–body. This is a path BSIM4 §6.1 entirely ignores (BSIM has no DNW model).
**Test**: Add explicit vertical BJT `Qv` in subcircuit with emitter=D, base=VB, collector=DNW; sweep V_DNW ∈ [0, 1.2 V]. If snapback intensity tracks V_DNW (gets stronger with higher DNW bias), the vertical NPN is the dominant mechanism. If it doesn't, lateral NPN suffices.

### F5. M2 NFACTOR up to 12 hints at GIDL / band-to-band tunneling at the M2 drain–body junction
**File**: `Image 2026-04-30 at 13.24.jpeg` (lower-right panel) — already discussed in F2 but with a distinct testable prediction.
**Physical implication**: NFACTOR > 5 in MOSFET subthreshold is typically a signature of **gate-induced drain leakage (GIDL)** — band-to-band tunneling at the gate–drain overlap of M2. Since M2.D = V_B = M1.body, **GIDL at M2 directly injects holes into M1.body**, raising V_B independently of M1's own impact ionization. This would explain why VG2 has such strong control even when M1.II should be saturated.
**Test**: Add BSIM4 GIDL parameters (AGIDL, BGIDL, CGIDL, EGIDL) to M2 only and sweep AGIDL ∈ [0, 1e-7]; check whether silicon's VG2 sensitivity at HIGH V_D (where M1.II is already strong) collapses without GIDL but matches with it.

## Findings of secondary value
- **F6 (12.07 schematics)**: V_B and V_O are TWO different nodes — V_B = floating body of M1 = drain of M2; V_O = "output" = drain of M1 = where we read I_D. Our SPICE netlist must keep them distinct (R-55 checked this is OK but worth re-verifying in the M3b code path).
- **F7 (slide 12.39 LIF / 13.33 self-reset)**: Mario already simulates the self-reset behaviour in **Brian2** with a leaky-integrate-and-fire surrogate. Their TIMESCALE_E = 50 ms, TIMESCALE_I = 25 ms, REFRACTORY = 50 ms — i.e. they observed ms-scale dynamics on silicon. This is **100× slower than V_D pulse RC** → implies the relaxation time is the **floating-body recombination / leak constant**, not RC. Body-charge lifetime τ_body ≈ 10–100 ms in floating P-well is the dynamics they fit.
- **F8 (slide 12.33 self-reset)**: shows the cell firing in continuous regime at fixed V_D — i.e. snapback can be RECURRENT, not one-shot. This requires a STABLE limit cycle in (V_B, I_D) phase space, which BSIM4 cannot produce because its II current is a memoryless function of V_DS.

## Top-3 most actionable experiments (recommendation for next ngspice round)

### Experiment T1 (highest priority): Body-charge-driven relaxation oscillator
Implement a 3-terminal subcircuit for M1 with:
- Standard BSIM4 NMOS for the channel
- Explicit `C_body` = 0.3–3 fF between floating V_B and substrate
- Source–body diode `Dsb(VB, S)` and drain–body diode `Ddb(VB, D)` with realistic IS, N, BV (avalanche at ~6 V)
- Vertical NPN `Qv(C=DNW, B=VB, E=D)` with low BF=1–3 (poor vertical BJT)
- Lateral NPN `Ql(C=D, B=VB, E=S)` (the one we already had)
- M2 replaced by a 2-terminal `R_leak(VG2)` whose conductance follows the measured NFACTOR(VG2) curve from 13.24 (look-up table, NOT a BSIM device)
Run transient with V_D step + hold; look for the predicted post-pulse spike. SUCCESS criterion: spike position vs VG2 matches slide 12.30 hysteresis within 30%.

### Experiment T2 (KO test for II vs body-charge): ramp-rate sweep
Same netlist but sweep dV_D/dt across 4 decades (10 V/µs … 1 mV/µs). Predictions:
- BSIM §6.1 II-only model: kink voltage independent of ramp rate (within MOSFET RC).
- Body-charge model: low-V kink VANISHES at slow ramps; high-V kink shifts to lower V_D at fast ramps (more capacitive injection).
Compare with 13.25 transient. If silicon shows ramp-rate dependence, we have falsified BSIM-only.

### Experiment T3 (DNW-bias diagnostic): split lateral vs vertical NPN
Take T1 netlist, sweep DNW bias V_DNW ∈ {0.3, 0.6, 0.9, 1.2} V. If snapback amplitude scales with V_DNW, vertical NPN through DNW dominates → recommend Sebas to characterise with V_DNW step (this is a simple bench experiment for them). If invariant, lateral NPN dominates → drop the DNW branch from our model.

## What we already had ("nothing new" list, for completeness)
- 2T-cell schematic with V_B node — known (slide 12.07, sebs_iv_family).
- BSIM §6.1 II equation — known.
- Body-bias / VG2 modulation of firing rate — known.
- Triple-well 130 nm CMOS process — known.

## What is NEW from this image-level scan (not in any text-only file)
- **F1** Two distinct kink voltages per transient → two-mechanism snapback (not in any prior write-up).
- **F2** Quantitative NFACTOR(M2) = 2–12, ETAB(M1) is VG2-dependent → BSIM cannot fit with constant parameters (R-55 only said "fit fails", not WHY).
- **F4** Vertical NPN through DNW as a candidate path → never simulated.
- **F5** GIDL at M2 drain → never simulated.
- **F8** Continuous-regime firing implies limit cycle → BSIM §6.1 cannot produce.

## File-path index (for downstream agents)
| Path | Key content |
|---|---|
| `research_plan/artifacts/Zoom/Image 2026-04-30 at 13.25.jpeg` | Two-kink transient (F1, F2) |
| `research_plan/artifacts/Zoom/Image 2026-04-30 at 13.24.jpeg` | NFACTOR(M2)=2–12, ETAB(M1) vs VG2 (F2, F5) |
| `research_plan/artifacts/Zoom/Image 2026-03-20 at 12.30.jpeg` | Thick-ox 2T cross-section with "VG2=-0.5 V slows leak" annotation (F3) |
| `research_plan/artifacts/Zoom/Image 2026-03-20 at 12.29.jpeg` | Thick-ox 1T cross-section (F4 DNW geometry) |
| `research_plan/artifacts/Zoom/Image 2026-03-20 at 12.33.jpeg` | Continuous self-reset firing (F8) |
| `nsram/proposal_2026_05/figures/sebs_130nm_triple_well_layout.jpeg` | Layout + DNW (F4) |
| `nsram/proposal_2026_05/figures/sebs_impact_ionization_fits.jpeg` | Piecewise-linear Iion empirical fit (supports F2) |
| `nsram/proposal_2026_05/figures/sebs_snapback_transient.jpeg` | VD-ramp drain-current hysteresis loops |
| `nsram/proposal_2026_05/figures/sebs_iv_family_with_2T_schematic.jpeg` | M1/M2 schematic with V_B/V_O nodes (F6) |
