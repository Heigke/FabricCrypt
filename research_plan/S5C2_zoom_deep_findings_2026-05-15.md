# S5C2 — Zoom Deep Findings (exhaustive pass)
Date: 2026-05-15
Author: research-agent (deep image+doc pass, in parallel to S9 snapback subcircuit)

This pass re-examines every file in the Zoom folders + the BSIMfitsBA bundle + the docs that surround the 30-Apr meeting. It is targeted at the open question: what concretely specifies Mario/Sebas's snapback model so that S9 can be parameterised from primary sources rather than BBO refits.

---

## 1. Files examined (complete enumeration)

### nsram/Zoom/ (mirror of docs/Zoom + the BSIMfitsBA bundle Sebas sent 30-Apr)
- mail.txt — full Gmail thread Mario↔Eric↔Sebas↔Robert, 20-Mar → 03-May. READ END TO END.
- pdiode.txt — Sebas's SPICE `.model pdiode diode` card (sent 1-May follow-up).
- image-2.png — Sebas's "Dynamic response (ramp rate dependence)" slide (the key transient slide).
- Image 2026-03-20 at 12.05.jpeg — Mario "Status of AI hardware" deck title.
- Image 2026-03-20 at 12.05 (1).jpeg — same deck, Mario CPU/GPU/transistor energy budget.
- Image 2026-03-20 at 12.06.jpeg — "AI on edge" power tiers, source: Innatera 2025 (qsysarch.com/posts/neuromorphic-ics-system-architecture).
- Image 2026-03-20 at 12.07.jpeg — Mario "Spiking Neural Networks" w/ LIF eq; cites X. Zhu, S. Pazos, M. Lanza et al., **Nature 618, 57–62 (2023)**.
- Image 2026-03-20 at 12.07 (1).jpeg — duplicate of 12.07 (slide capture).
- Image 2026-03-20 at 12.07 (2).jpeg — Mario "Memristor-based electronic neuron" cites Wang et al., Adv. Intell. Syst. 3, 2100007 (2021) and A. Mizoki, Nat. Commun. 15, 2812 (2024).
- Image 2026-03-20 at 12.08.jpeg — same memristor neuron slide w/ Wang/Mizoki refs.
- Image 2026-03-20 at 12.08 (1).jpeg — "Neuron implementation" with Adaptive expo IF, Mixed-signal IF, Analogue (8-element), **mentions US patent ?? for Newmorphic neuron** (Mario's startup). Plus Sentaurus reference under one block ("Source: Galarreta et al, IEDM ?, Patent ??") — too small to read precisely but the slide does invoke patent + Sentaurus.
- Image 2026-03-20 at 12.08 (2).jpeg — Mario memristor cycling: D. Alfaro et al., Mater. Sci. Eng. R: Reports 161, 100867 (2025) — cycling endurance plots.
- Image 2026-03-20 at 12.09.jpeg — Mario "Die-to-die variability" — NS-RAM card with bulk-terminal trim, cites S. Pazos, M. Lanza et al., **Nature 640, 65–70 (2025)** (this is the NS-RAM paper).
- Image 2026-03-20 at 12.26.jpeg — **★ Sebas slide "Semi-empirical model fits for impact-ionization bulk currents"** — gives explicit formula (transcribed below).
- Image 2026-03-20 at 12.27.jpeg — Sebas "Measurements vs SPICE simulation using semi-empirical bulk currents" — agreement curves Vdrain sweep, Vg2 family.
- Image 2026-03-20 at 12.27 (1).jpeg — Sebas "Floating body 2T NS-RAM cell under transient VD ramps" — VD0=0.5V, ramping VG2; multi-decade Id vs VD agreement.
- Image 2026-03-20 at 12.29.jpeg — Mario "NS-RAM implementation in standard triple-well CMOS (130 nm)" — 27 µm² cell, deep N-well, two-transistor floating well; "1000× improvement vs state-of-art neuron cells".
- Image 2026-03-20 at 12.29 (1).jpeg — Sebas cross-section + plots: standard CMOS deep-N-well NFET floating-body 1T (thick oxide), 8 µm², 100% yield, "firing window 7×–10⁹", high density 6 µm² per fully-contacted 1T neuron (180 nm CMOS).
- Image 2026-03-20 at 12.30.jpeg — Sebas "2T NS-RAM spiking neuron cell (thick oxide)" — 17 µm² cell, self-relaxation oscillation plots ("second transistor enables fully-floating behaviour (negative VTH driven from the leakage behaviour)"), "outstanding firing range exceeding 10⁵×".
- Image 2026-03-20 at 12.33.jpeg — Sebas "NSRAM Simple LIF in Brian2" — Vth_lif=1.345 V, Tau_dif=0.0026 s, REFRACTORY=0.0070 s, TIMESCALE=1e3, EXCIT_VALUE=2.6 V; fits SNN convergence at slowed timescale 10×.
- Image 2026-03-20 at 12.39.jpeg — "Simulating more physically realizable SNN in Brian2"; Poisson reference 86%, LIF (w/Poisson training) 72% (MNIST 10-class).
- Image 2026-04-22 at 19.57.jpeg — NUS Ariba supplier-registration screenshot (no science content).
- Image 2026-04-30 at 13.23.jpeg — **★ Sebas (30-Apr) M1/M2 SPICE fit family: 3 panels, VG1=0.2 / 0.4 / 0.6 V; lines=sim, symbols=meas; multi-decade Id from 1e-12 to 1e-4 A over Vdrain 0–2 V; clean snapback at all three VG1.**
- Image 2026-04-30 at 13.24.jpeg — **★★ Sebas's per-VG2 parameter trends: BETA0(M1) vs VG2, K1(M1) vs VG1, ETAB(M1) vs VG2, NFACTOR(M2) vs VG2.** This is the slide image (also the source of three_branch_params_extracted.json).
- Image 2026-04-30 at 13.25.jpeg — Sebas "transient self-reset oscillation" — Vread bias=2V, repeated sawtooth firing waveform (current pulses + ramping body voltage), period ~0.4 µs; two zoom panels show how the snapback knee carries current+voltage points (red dots) traversing the I-V family.
- Image 2026-04-30 at 13.28.jpeg — Sebas's M1/M2 final fit overlay (thick=meas, thin=sim) at three working points VG1/VG2 = (0.2,0), (0.4,0.25), (0.6,0.35).
- Image 2026-04-30 at 13.31.jpeg — Mario deck "Simple NS-RAM cell with integration (self-reset)" — energy ≈ 0.4 pJ per spike, spiking frequency configurable with 10× (clock?); slide ascribes the integration step.
- Image 2026-04-30 at 13.31 (1).jpeg — Mario "NS-RAM blocks for input neurons (soma without diode)" — area ≈ 46 µm²; energy ≈ 5.5 fJ (Idle), ≈21 fJ per cycle of action, ≈4.7 fJ spike generation, ≈30 fJ integration loss.
- Image 2026-04-30 at 13.33.jpeg — Mario "NSRAM firing with linear excitatory and inhibitory inputs" — VPD55 / VPD55(?)+inverter-based soma, thick-oxide stack for higher drain voltages.
- Image 2026-04-30 at 13.46.jpeg — Mario "AI on edge" slide (dup of 12.06 in newer deck).
- Image 2026-04-30 at 13.47.jpeg — Mario chip-stack micrographs (CMOS-fab inserts + h-BN memristor on 1T), cites previous publication; key text: **"deposition of resistive switching medium can be done in many different ways, but it cannot use temperatures above 400 °C otherwise the microchip will be destroyed. In our previous publication we transferred multilayer hBN on the chip, although we recommend the use of scalable methods."**
- Image 2026-04-30 at 13.50.jpeg — Mario SNN slide with Nature 618 (Zhu/Pazos/Lanza 2023) ref again.
- Image 2026-04-30 at 13.53.jpeg / 13.53 (1).jpeg — Mario's Outlook screenshots of an internal grant email and **NRF Mid-Sized Centre Grant ("interest in")** — funding-context only.
- schematic&modelCards/ — duplicates of M1/M2 + parasiticBJT + PTM130bulkNSRAM, plus 2tnsram_simple.asc.
- Slow I-Vs 2vHCa-2@VG2 VG1 vnwell=2 SRavg=0/ — the 33 raw measurement CSVs (StandardIV_HH_...), 82 lines each, columns `vdata,idata,tdata,Var4,vfixgdata,ifixdata`. Sweep span ~0 → 2.0 V, ~80 samples, **0.2 V/s average sweep rate (from mail.txt 20-Apr)**.

### docs/Zoom/2026-04-30 13.03.27 Zoom NSRAM/
- meeting_saved_closed_caption.txt — 2283 lines of Zoom auto-captioned Swedish/English; >95 % garbage transliteration. Only minor confirms ("Subthreshold Saturation [Bulk?] Effect", "Currentlangenglish", "Profit dependences" = "Parameter dependences"). No new technical content readable.

### docs/2026-04-30 BSIMfitsBA/
Same contents as nsram/Zoom/* model-card bundle (M1, M2, BSIM CSV, BA-plots .pptx + .xlsx).

### data/sebas_2026_04_22/
- 2Tcell_BSIM_param_DC.csv — **explicit per-(VG1, VG2) tabulated parameters** (ETAB, K1, ALPHA0, BETA0, NFACTOR, mbjt, IS, area, trise) for 33 bias rows. This is the table that drives the polynomial fits.
- 2tnsram_simple.asc — LTSpice schematic of the 2T cell (NMOS M1 + NMOS M2 + parasitic NPN Q1 + cap C1). Read below.
- M1_130DNWFB.txt / M2_130bulkNSRAM.txt — Sebas's two foundry-derived BSIM4 v4.5 cards (130 nm PTM-derived), full ~880 parameters.
- M{1,2}_..._LALPHA0_FIX.txt — fix variants: ALPHA0 inflated 10× (7.83756e-4 vs 7.83756e-5) and LALPHA0 zeroed. Diff shown below.
- parasiticBJT.txt — Sebas's NPN card (one line — transcribed below).
- PTM130bulkNSRAM.txt — public PTM 130 nm card for reference.

### data/sebas_2026_05_02/
- pdiode.txt — Sebas's body-junction diode card (sent 1-May). Verbatim above.
- three_branch_params_extracted.json — our prior extraction of the per-VG2 parameter trends from slides.
- image-2.png — duplicate of nsram/Zoom/image-2.png (dynamic ramp-rate slide).

### docs/ (top-level)
- NSRAM 20260429 Mario Seb.pptx — our (Eric's) status deck for the 30-Apr meeting; text extracted (no new device data).
- Team meeting 30 Apr 2026.pptx — Robert's status deck — Julia DEQ inner solver, BBO 10K iters loss=0.155, 8.2× improvement from defaults, **"1 of 7 parameters hit bounds on reasonable ranges"** — the snapback parameter is the one that hit the bound.
- NSRAM_Research_Proposal_2026-04-30.docx, FEEL_x_NSRAM_pitch.docx, NSRAM_samarbetsplan_*.txt — strategy docs (no new physics).
- 2026-04-29 NS-RAM I-V BA plots.pptx (in BSIMfitsBA bundle) — failed to load via python-pptx; same content rendered into the 13.23/24/25/28 Zoom screenshots above.

---

## 2. Verbatim transcriptions (the load-bearing pieces)

### 2.1 Sebas's `parasiticBJT.txt` (entire file)

```
* Simple bjt for floating bulk parasitic bipolar effect
* Pazos, S.
.model parasiticBJT NPN(is=5E-9 va=100 bf=10000 br=100 nc=2 ikr=100m rc=0.1 vje=0.7
                        re=0.1 cjc=1e-15 fc=0.5 cje=0.7e-15 ne=1.5 ise=0
                        tr=20e-12 tf=25e-12 itf=0.03 vtf=7 xtf=2)
```

**Key numbers:**
- Is = 5 nA (saturation current, large because area=1u in .asc)
- VA = 100 V (Early voltage)
- BF = 10000  (forward gain — *very* high; this is the snapback current-multiplier knob)
- BR = 100
- NC = 2 (B-C ideality)
- IKR = 100 mA (reverse knee)
- VJE = 0.7 V (emitter junction potential)
- CJC = 1 fF, CJE = 0.7 fF (BJT junction caps — small)
- TF = 25 ps, TR = 20 ps (transit times)
- ITF=0.03, VTF=7, XTF=2 (high-current transit modifiers)

### 2.2 Sebas's `2tnsram_simple.asc` (the schematic, key lines)

```
SYMBOL nmos4 464 112 R0    InstName M1   Value2 l='Ln' w='Wn' m=1
SYMBOL npn   736 112 R0    InstName Q1   Value parasiticBJT   Value2 area=1u
SYMBOL cap   688 288 R0    InstName C1   Value 'CBpar'        SpiceLine Rser=1m
SYMBOL nmos4 560 272 R0    InstName M2   Value2 l='Ln*10' w='Wn' m=1
TEXT 552 24 !.param Ln=0.18u\n.param Wn=0.36u\n.param CBpar=1f
TEXT 520 -64 !.inc PTM130bulkNSRAM.txt
TEXT 520 -40 !.inc parasiticBJT.txt
```

So the production schematic Sebas runs is:
- M1: NMOS Ln=0.18 µm, Wn=0.36 µm (drives the channel)
- M2: NMOS, **L = 10× M1 (Ln*10 = 1.8 µm)**, Wn = 0.36 µm (the long-channel "bulk biasing" transistor — this is what NFACTOR(M2) drift describes)
- Q1: NPN parasiticBJT, **area=1u** (1 µm² → Is becomes 5 nA × 1 µm² scaled inside SPICE)
- C1: CBpar = 1 fF parasitic body cap, Rser=1 mΩ
- nets: D, G, B (P-body float), Sint (internal), S, G2, Din

Body C1 = **1 fF** in the .asc — but the 1-May email says "5–10 fF" is the practical range; the new pdiode supersedes the linear cap.

### 2.3 Sebas's `pdiode.txt` (the body-diode card, sent 1-May, replaces C1)

```
.model pdiode diode
+level=1   tnom=25
+is=5.3675e-7   isw=1.3664e-13   rsw=0.46493   ns=1.0851
+nz=1.3664e-13  imax=1e30   imelt=1e30   bvj=1e31
+ik=97740       bv=11   ibv=97740   ikp=1.1946e5
+n=1.0535       rs=7.4155e-8
+cj=7.3279e-4   cjsw=1.0522e-10   vj=0.21918   vjsw=0.65166
+m=0.24097      fcs=0.5   mjsw=0.26029   fc=0.5
+xti=6.5        eg=1.11   tlev=1   tlevc=1
```

Email context (1-May): **"area 5 µm × 4.4 µm = 22 µm²"**, capacitance dependence on P-body voltage matters more than the diode current; **BV=11 V, so well above any operating point** (the diode is *not* the avalanche element).

### 2.4 The per-bias parameter table `2Tcell_BSIM_param_DC.csv`

Verbatim columns: `VG1, VG2, trise, ETAB, K1, ALPHA0, BETA0, NFACTOR, mbjt, IS, area`. **33 bias rows.** Worth pinning verbatim because this is what S9 should regress against:

- ALPHA0 is constant = 7.842e-5 across all rows. **It does NOT vary with VG1 or VG2.**
- BETA0 trends with VG1: ≈11–14 at VG1=0.2 V (rising with VG2), flat 19 at VG1=0.4 V, flat 20 at VG1=0.6 V.
- K1 only depends on VG1: 0.55825 / 0.53825 / 0.41825 for VG1=0.2/0.4/0.6 V.
- ETAB rises with VG1 (0.8 → 2.5) and modulates weakly with VG2.
- NFACTOR drops nearly linearly with VG2 in three branches (red 12.2 → 6.25; blue 6 → 2.75; black 6 → 1.25).
- mbjt = 0.001 for VG1=0.2 V branch, **= 1.0 for VG1=0.4 V and 0.6 V branches** (3-decade jump in BJT multiplier between VG1=0.2 and VG1≥0.4) — this is the snapback **on/off knob** in Sebas's fit.
- IS=5e-9 fixed; area=1e-6 fixed.
- trise = 9–13 (slow ramp parameter, units unclear from CSV, plausibly µs at the 0.2 V/s ramp).

### 2.5 The slide formula for the empirical impact-ionization bulk current (Image 12.26)

Sebas writes the bulk current as an **explicit sum of an exponential and a piecewise power law**, with all coefficients PWL functions of VG2:

```
Ipos  = Iexp + Ipow

Ipow  = a · (VD − y)^β     if VD > y
Ipow  = 0                  if VD ≤ y

Iexp  = c · exp(d · VD)

where  a, β, c, d, y  =  PWL(VG2),     b is constant
```

Three subplot insets on the slide show the PWL traces for the coefficients vs VG2.

Caption on the slide: "Transistor and bulk current model are fitted in MATLAB, then transferred to SPICE. … Two components in (mostly) inseparable voltage dependence with VG2: exponential term Iexp[c(VG2), d(VG2)], power-law term Ipow[a(VG2), β(VG2)]. Active when VD > y(VG2). Y(VG2) is the positive zero point of the function. Each parameter is extracted for different VG2. Iexp dependence is modelled in SPICE as a piecewise-linear PWL function. Ipow has the improved fit (smoothness) with piecewise polynomial of order Y (data fits)."

So **Mario/Sebas do NOT use a Verilog-A, TCAD, or Sentaurus snapback model.** The snapback is reproduced by:
1. The BSIM4 native impact-ionization stack (ALPHA0/BETA0/LALPHA0 in the model card) generating substrate current that lifts the floating P-body, lowering Vth, raising channel current → positive feedback.
2. The parasitic NPN Q1 (emitter=Sint, base=floating P-body, collector=drain) firing once V_BE forward-biases — `BF=10000` makes the multiplier huge.
3. **A semi-empirical add-on Ipos = Iexp + Ipow injected at the bulk node**, with PWL-in-VG2 coefficients, that captures the residual fit error after #1+#2.

The "model" Sebas uses is just the BSIM4 v4.5 card (M1+M2) + parasiticBJT NPN + (now) pdiode + Iexp+Ipow injection. Everything is foundry-style native SPICE — no behavioural .va, no TCAD output.

### 2.6 The dynamic-response slide (image-2.png)

Annotations on the slide:
- "Firing and relaxation transients (slopes S_rise, S_relax) relate to speed of bulk voltage following the drain voltage increase (how fast impact ionisation can generate carriers and increase the voltage of the floating body)."
- "Parasitic capacitances play a crucial role on the effective time constant."
- "Lines = ramped measurements" (Id-VD family at VD=2.0V→0V, ramp time series visible)
- Time constants annotated: **τ_rise = 100 ps, τ_relax = 200 µs (?? — illegible), τ_rise = 100 ps, τ_relax = 1 µs**, VG1=0.45 V, VG2=0.3 V.
- Right schematic: VG_bias / floating P-body / VG1 / VG2 / Vnwell=0 V — the **deep N-well is grounded** in this measurement (vnwell=2 in the slow-IV CSVs corresponds to the high-voltage characterisation; the dynamic measurement here is at Vnwell=0).
- Bottom-right annotation: **"SR and firing time experiments (3 through 7 in experiment list) are critical for fitting this dependence"** — so Sebas has experiments #3-#7 (pulsed/transient) explicitly earmarked for the τ extraction.

### 2.7 Robert's status deck (key constraint)

"BBO 10K iters loss=0.155, 8.2× from defaults, 1 of 7 parameters hit bounds on reasonable ranges, LDE physics missing." Combined with our prior `three_branch_params_extracted.json` comment ("Our BBO bound of 3.0 cuts off 4× of the realistic range. M3 unclamped refit should explore upper bound at least 15"), the parameter that hit the bound is **NFACTOR with cap=3 vs measured peak=12.2**. This is the gap that S9 has to close.

### 2.8 Mario's energy/area numbers from the 30-Apr deck

- 2T NS-RAM cell area: 17 µm² (130 nm, thick oxide).
- 1T (single transistor with DNW) variant: 8 µm², "firing window 7×–10⁹".
- Input-neuron block (soma without diode): 46 µm² total.
- Per-spike energy: **0.4 pJ**.
- Idle: **5.5 fJ**; per-cycle action: **21 fJ**; spike generation: **4.7 fJ**; integration loss: **30 fJ**.
- Self-reset oscillation period in the transient slide: ~0.4 µs at Vread=2 V.

### 2.9 Email-thread highlights (mail.txt)

- **17-Apr (Sebas):** "I've dropped the avalanche diode models (very annoying for convergence) that fire the bipolar parasitic effect in LTSpice, and I'm only including a **complementary bipolar current to capture the full swing of the firing mechanism**. … I'm focusing on adapting foundry models of body bias and impact ionization (in BSIM4) to fit the floating body behaviour of our 2T cells."
- **19-Apr (Eric):** confirmed BSIM4 §6.1 channel-HCI route fits ~4 decades RMS better than §10.1 junction breakdown — picks channel HCI.
- **20-Apr (Sebas):** "I-V curves in CSV files, **at an average sweep rate of 0.2 volts per second**, 3 different VG1 (0.2, 0.4, 0.6) and multiple VG2 (value enclosed within each filename). We are generating additional data now (multiple ramp rates for more data on dynamics, pulsed dynamics)."
- **30-Apr (Sebas):** "**some parameters change only for M1, while NFACTOR changes only for M2 (I attribute this to LDE)** — hence two separate model cards for each device." LDE = Layout-Dependent Effects.
- **1-May (Sebas):** parasitic-diode update — "pdiode with area 5 µm × 4.4 µm = 22 µm²", "could be replaced with a linear capacitor in the range 5–10 fF, although that would be less electrically accurate because it won't capture the capacitance dependence on P-body voltage". So C1=1 fF in the .asc is **stale** — replace with pdiode card.
- No GitHub URL, no arxiv, no DOI from Mario or Sebas. The cited papers are: **Pazos/Lanza Nature 640 65–70 (2025)** (the NS-RAM paper), **Zhu/Pazos/Lanza Nature 618 57–62 (2023)** (SNN with NS-RAM), Wang Adv. Intell. Syst. 3 2100007 (2021), Mizoki Nat. Commun. 15 2812 (2024), Alfaro Mater. Sci. Eng. R 161 100867 (2025).

---

## 3. Top 5 actionable findings for the snapback gap

Ranked by what most directly closes the gap.

### Finding 1 — Use Sebas's tabulated parameters DIRECTLY, do not BBO-fit them
We have `2Tcell_BSIM_param_DC.csv` with 33 rows containing exact ETAB, K1, ALPHA0, BETA0, NFACTOR, mbjt, IS values that Sebas's own SPICE fitting converged to. These are the ground-truth fit outputs, not free parameters. Robert's BBO loss=0.155 was hitting the NFACTOR bound (cap=3) because the CSV says NFACTOR peaks at 12.15 (VG1=0.2, VG2=−0.2). **S9 should hardcode the CSV table as the per-bias param look-up and benchmark unbounded fits against it.**

### Finding 2 — The snapback model is NOT a separate Verilog-A or TCAD subcircuit
It is fully captured by **BSIM4 v4.5 + parasitic NPN(BF=10000, VA=100) + a PWL-in-VG2 (Iexp + Ipow) injection at the bulk node**. The Mario/Sebas world has no `.va`, no Sentaurus output, no Synopsys deck. Every artefact is plain SPICE include files. S9 should implement the snapback as three independent additive currents into the body node:
- I_BSIM4_impact (channel HCI, native ALPHA0/BETA0)
- I_BJT_emitter (Q1 emitter current once V_BE > VJE = 0.7 V)
- I_PWL_residual = c(VG2)·exp(d(VG2)·VD) + a(VG2)·max(VD−y(VG2),0)^β(VG2)

with c, d, a, β, y read from the 13.24 slide / `three_branch_params_extracted.json`. The PWL residual term is the one we have been missing in our differentiable port.

### Finding 3 — The body-junction element is now a diode, not a capacitor
The .asc shows C1=1 fF, but Sebas's 1-May update replaces it with the **pdiode card (area 22 µm², BV=11, n=1.0535, cj=7.3279e-4 F/m², vj=0.219 V, m=0.24097)**. Linear-cap equivalent is 5–10 fF (not 1 fF). The capacitance-vs-Vbody nonlinearity is what controls the **τ_rise = 100 ps / τ_relax ~ 1 µs** dynamics on the image-2.png slide. Replace any 1 fF / 5 fF / 10 fF placeholder in S9's transient block with the full pdiode SPICE element.

### Finding 4 — M1 and M2 must be DIFFERENT model cards (LDE)
Both M1_130DNWFB.txt and M2_130bulkNSRAM.txt are 130 nm PTM-derived BSIM4 v4.5, but they differ in three ways visible in `diff`:
- M1 has k1 = 0.53825, M2 has k1 = 0.63825 (10% difference — gives the K1(M1) vs VG1 curve in 13.24).
- M1 has etab = 1.8 (positive), M2 has etab = −0.086777 (negative).
- M2 has rcjn=1, rcjswn=1, etc. — junction-cap scale knobs separately tuned.
- BETA0 differs (M1=19, M2=18).

Sebas's 30-Apr email confirms: "some parameters change only for M1, while NFACTOR changes only for M2 (LDE)". Our pyport must load two separate cards and bind each MOS instance to its own card. **If S9 is using a single shared card for M1 and M2, that alone explains a measurable chunk of fit residual.**

### Finding 5 — The LALPHA0_FIX variants tell us the convergence trick
`M{1,2}_..._LALPHA0_FIX.txt` differ from the originals in exactly two parameter values:
- ALPHA0 = 7.83756e-4 (×10 vs original 7.83756e-5)
- LALPHA0 = 0 (vs original −9.843026e-12)

The interpretation: Sebas turned off the length scaling of α₀ (which fights the bulk current at short L) and bumped α₀ itself by 10× to compensate. This is a known LTSpice convergence trick when the parasitic NPN fires before BSIM4's impact-ionisation generates enough body current. S9's snapback hunt should test both card variants — the FIX variants may be the ones used in production SPICE runs.

---

## 4. Specific URLs / papers / IPs to follow up on

- **S. Pazos, X. Zhu, M. Lanza et al., Nature 640, 65–70 (2025)** — the primary NS-RAM device paper. We should already have it; if not, fetch.
- **X. Zhu, S. Pazos, M. Lanza et al., Nature 618, 57–62 (2023)** — NS-RAM-based SNN demonstration.
- **Wang et al., Advanced Intelligent Systems 3, 2100007 (2021)** — single-device memristor neuron (LRS/HRS comparison).
- **A. Mizoki, Nat. Commun. 15, 2812 (2024)** — variability of integrated memristor neurons.
- **D. Alfaro et al., Mater. Sci. Eng. R: Reports 161, 100867 (2025)** — memristor cycling endurance.
- **PTM 130 nm reference card** — http://ptm.asu.edu/latest.html (cited inline in M1/M2 cards).
- **Innatera neuromorphic IC system architecture post** — https://qsysarch.com/posts/neuromorphic-ics-system-architecture/ — Mario's energy-tier slide source.
- **NUS Newmorphic startup** — Mario mentions an associated US patent on the slide ("Source: US Patent ?? Newmorphic"). Worth a uspto.gov search on "Newmorphic Lanza" once we are allowed to deanonymise.
- **No GitHub from Mario/Sebas.** No arxiv preprint shared. The "modeling work" lives entirely in Sebas's local LTSpice + MATLAB on a NUS workstation.

---

## 5. Parameter values to use DIRECTLY (not BBO-fit)

| Param | Value | Source |
|---|---|---|
| ALPHA0 (M1 & M2) | 7.83756e-5 (or 7.83756e-4 in FIX variant) | M{1,2}.txt |
| LALPHA0 (M1 & M2) | −9.843026e-12 (or 0 in FIX variant) | M{1,2}.txt |
| BETA0 (M1) | 19 | M1_130DNWFB.txt |
| BETA0 (M2) | 18 | M2_130bulkNSRAM.txt |
| LBETA0 | −9.5e-7 | both cards |
| K1 (M1) | 0.53825 | M1 |
| K1 (M2) | 0.63825 | M2 |
| ETAB (M1) | 1.8 | M1 |
| ETAB (M2) | −0.086777 | M2 |
| NFACTOR (M2, peak) | 12.15 (VG1=0.2, VG2=−0.2) | DC CSV |
| NFACTOR (M2, min)  | 1.25 (VG1=0.6, VG2=0.5)  | DC CSV |
| M1 geometry | L=0.18 µm, W=0.36 µm | .asc |
| M2 geometry | L=1.8 µm (10×M1), W=0.36 µm | .asc |
| Parasitic NPN: IS | 5 nA | parasiticBJT.txt |
| Parasitic NPN: BF | 10000 | parasiticBJT.txt |
| Parasitic NPN: BR | 100 | parasiticBJT.txt |
| Parasitic NPN: VA | 100 V | parasiticBJT.txt |
| Parasitic NPN: TF / TR | 25 ps / 20 ps | parasiticBJT.txt |
| Parasitic NPN: CJC / CJE | 1 fF / 0.7 fF | parasiticBJT.txt |
| Parasitic NPN: area | 1 µm² | .asc |
| Body cap CBpar (stale) | 1 fF | .asc (overridden by pdiode) |
| Body cap (linear approx) | 5–10 fF | 1-May email |
| Body pdiode area | 22 µm² (5 × 4.4) | 1-May email |
| Body pdiode BV | 11 V | pdiode.txt |
| Body pdiode CJ | 7.3279e-4 F/m² | pdiode.txt |
| Body pdiode VJ | 0.21918 V | pdiode.txt |
| Body pdiode N | 1.0535 | pdiode.txt |
| τ_rise | 100 ps | image-2.png |
| τ_relax | ~1 µs | image-2.png |
| Measurement sweep rate | 0.2 V/s | mail.txt 20-Apr |
| Vnwell (slow IV CSV set) | 2 V | folder name + filename |
| Vnwell (dynamic measurement) | 0 V | image-2.png schematic |
| Operating Vdrain range | 0 → 2.0 V (slow), 0 → 2.5 V (transient) | CSVs + image-2.png |
| Self-reset oscillation period | ~0.4 µs | 13.25 transient slide |
| Energy per spike | 0.4 pJ (Mario), 21 fJ per action (Mario's 13.31 slide) | 30-Apr deck |
| 2T cell area | 17 µm² (thick oxide) | 12.30 slide |
| 1T cell area | 8 µm² | 12.29 (1) slide |
| Technology | 130 nm bulk + DNW, thick oxide, triple-well | 12.29 slide |

PWL-in-VG2 coefficients (a, β, c, d, y) for the Ipos = Iexp + Ipow residual term: **we still need to digitise these from the three inset subplots in image 12.26**. That is a 30-min OriginLab / WebPlotDigitizer task; current `three_branch_params_extracted.json` only covers the four big panels of 13.24.

---

## 6. What we already had vs what is new in this pass

| Source | Already in S5-C / R-55 | New here |
|---|---|---|
| D3 zener + ETAB | yes (R-55) | confirmed not in production card; ETAB is BSIM-native, not external zener |
| M3 BSS145 | yes (R-55) | n/a — no third transistor exists in the production .asc; the "M3" we BBO-fit was a phantom |
| Vertical NPN to DNW | yes (S5-C image-level) | confirmed: parasiticBJT with **BF=10000, VA=100, IS=5 nA, area=1 µm²** — exact numbers |
| Two-kink snapback | yes (S5-C) | mechanism explicit: BSIM4 impact-ion + NPN parasitic + PWL residual |
| Transient relaxation oscillator | yes (S5-C) | τ_rise=100 ps, τ_relax~1 µs, period ~0.4 µs — pinned |
| **Per-bias ALPHA0/BETA0/NFACTOR/ETAB/K1/mbjt table** | NO | **2Tcell_BSIM_param_DC.csv = 33 rows, this is THE ground truth** |
| **M1 vs M2 split (LDE)** | NO | confirmed via diff(M1, M2) and 30-Apr email — must load two cards |
| **PWL-in-VG2 (Iexp + Ipow) injection** | NO | explicit formula transcribed above; coefficients still need digitisation |
| **mbjt step at VG1=0.2→0.4** | NO | 0.001 → 1.0 — 3-decade jump; this is the snapback on/off knob |
| **pdiode replacing C1** | NO | 22 µm² body diode, BV=11; not a capacitor |
| **LALPHA0_FIX variants** | NO | ALPHA0 × 10, LALPHA0 = 0 — Sebas's LTSpice convergence trick |
| **Measurement sweep rate** | NO | 0.2 V/s (slow), pulsed-dynamics dataset incoming |
| Verilog-A / TCAD / Sentaurus | speculative | **does not exist** — Mario/Sebas use plain SPICE only |
| GitHub / arxiv from NUS | speculative | **does not exist** — no public artefact from their side |

---

## 7. Concrete handoff to S9

If S9 wants to close the snapback gap without further BBO, do these four things in order:

1. **Load M1 and M2 as separate BSIM4 cards** in the differentiable pyport. Bind M1 to channel transistor, M2 to body-bias transistor (L = 10× M1).
2. **Hard-table the per-bias parameters** from `2Tcell_BSIM_param_DC.csv` — turn ETAB(VG1,VG2), K1(VG1), BETA0(VG1,VG2), NFACTOR(VG1,VG2), mbjt(VG1) into 2-D look-ups (or train polynomial fits with the table as targets, not via BBO on raw IV).
3. **Replace the body capacitor with the pdiode element** (or at minimum a 5 fF linear cap). Tie its area to 22 µm².
4. **Add the Ipos = Iexp + Ipow residual injection** at the body node, with PWL-in-VG2 coefficients. Digitise the inset subplots in image 12.26 first (30-min task).

Mario/Sebas have given us everything that is shareable. The remaining residual must come from one of:
- the pulsed-dynamics dataset Sebas is generating (post-30-Apr — not yet delivered);
- digitising the PWL traces in image 12.26 to nail (a, β, c, d, y)(VG2);
- the NDA'd foundry card, which is not shareable.

---

End of S5C2 deep findings pass.
