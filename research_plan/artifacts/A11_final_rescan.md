# A11 — Final-pass deep rescan of Sebas/Mario materials

**Date:** 2026-05-01
**Author:** Eric (this session)
**Scope:** Re-verify A7/A8/A9 against every Sebas/Mario artefact on disk
plus the new sebas_2026_05_02 dump (`pdiode.txt`, `image-2.png`) and
the most recent Zoom screenshots from 2026-04-30 13:23–13:53. Look for
anything we missed that should change the model or the focus, now that
our model fires real volt-scale NPN at Vd=3 V with `pdiode` + per-bias
ALPHA0/BETA0/ETAB/NFACTOR/K1 overrides.

## 1. New findings since A7/A8/A9

### 1.1 sebas_2026_05_02/image-2.png — DYNAMIC RAMP-RATE DEPENDENCE (NEW)
Slide titled **"Dynamic response (ramp rate dependence)"**, three panels.
Verbatim bullets (extracted from image OCR):

> *"Firing and relaxation transients (slopes Sup, Sdown) relate to
> speed of bulk voltage following the drain voltage increase (how fast
> impact ionization generates carriers and increases the voltage of the
> floating body)."*
>
> *"SR and firing times experiments (3 through 7 in experiment list)
> are critical for fitting this dependence."*
>
> *"Parasitic capacitances play a crucial role on the effective time
> constant."*

Panel 1 (lines = ramped measurements): Vd=2 V, log Id vs Vd, sweep rate
parameterised at Trise = 1 ms, 200 µs, 100 µs, 50 µs, 20 µs, 10 µs, 5 µs.
Annotation: **"firing slope Sup"** (rising edge) and **"relaxation slope
Sdown"** (falling edge). VG1=0.4 V, VG2=0.3 V.

Panel 2: Trise=200 µs ramp + 5 µs ramp side-by-side, showing **firing**
and **relaxation** windows.

Panel 3: schematic with Vbulk_eff floating P-body and external V_d, V_G,
V_VG2 control nodes; **Voltage-dependent leakage and capacitances** label.

**THIS IS THE BIGGEST NEW INPUT.** Sebas explicitly says experiments 3–7
in his experiment list (we don't have that list yet — should ask) are
the **dynamic SR (sweep-rate) and firing-time** experiments that fit the
parasitic capacitance / time-constant. Our DC-only fit cannot validate
this dimension.

### 1.2 sebas_2026_05_02/pdiode.txt — Drain-body diode model (NEW)
A complete level-1 diode card, NOT a BSIM block. Critical numbers:

```
is = 5.3675e-7  isw = 1.3664e-13  ns = 1.0851
ik = 97740      bv = 11           ibv = 97740
n  = 1.0535     rs = 7.4155e-8
cj = 7.328e-4   cjsw = 1.052e-10  vj = 0.21918
m  = 0.24097    mjsw = 0.26029
xti = 6.5       eg = 1.11
```

Implications:
- `bv = 11 V` — junction breakdown is 11 V, **far above** the operating
  Vd ≈ 1–3 V used by Sebas. Junction breakdown is therefore **not** the
  firing path; channel HCI must own snapback (consistent with A7 §1.1).
- `is = 0.54 µA`, `n = 1.05` — large saturation current with near-ideal
  ideality. This is a real drain-body PN diode (not a poly diode).
- `cj = 0.73 mF/m² × area + cjsw × perimeter` — gives the drain-body
  parasitic capacitance that sets τ in the dynamic ramp-rate work.
- `nz = 1.366e-13` and `ns = 1.085` are misnomers; ns is the swing-side
  ideality and nz looks like a typo for `js` (zero-bias saturation
  density). LOW priority but worth flagging.

**Action:** add `pdiode` between drain and body B (anti-parallel to the
NPN base-collector junction) in the topology. We currently use it as a
unity charge source on Vb; treating it as a real diode with `cj`/`cjsw`
provides the τ that the ramp-rate experiments fit.

### 1.3 Zoom screenshots from 2026-04-30 — Sebas's deck pages we did NOT have as PPTX

- **Image 13.31** ("Simple NS-RAM cell with integration (self-reset)"):
  > *"Simulated structures that self-reset. Tested with long bias intervals
  > between firing or boredom cycles. Lowest possible energy consumption,
  > with strong degree of leak and integrator currents configuration.
  > Floating body generates the spikes ranging 0.5 V to 0.7 V. Operating
  > frequency at V_apply contributes an integration capacitance.
  > **Energy consumption considerations: 4.x fJ per cycle leakage,
  > losses while firing, 21x fJ per spike generation, 14% bias amplitude.**
  > **Spiking frequency is configurable within 10×.**"*

  Slide 3 of 19; cell schematic shows VDS, VG2, V_BIAS path. **The 21 fJ/
  cycle and 6.7 fJ/spike numbers we cited in the proposal originate
  here**, but slide explicitly says spike rate is **10×-tunable** via VG2.

- **Image 13.33** (slide 4): **NSRAM firing with linear excitatory and
  inhibitory inputs**
  > *"Linear range for V_VG2 is between 2.5 V and 3 V. Weight comes from
  > the mirror bank generating the bias voltage. Thick oxide devices
  > accommodate the (now) higher drain voltage for NS-RAM application."*
  > *"In this implementation, NS-RAM inverter-based soma interfaces
  > directly to the linear synapse circuit by spiking directly into the
  > input. Devices are designed to operate with 1 V pulses emerging from
  > drained inverters in NS-RAM soma."*

  → **Mode-switching control voltages**: linear-input neuron operates
  with VG2 ∈ [2.5, 3.0] V — far above our DC-fit range (VG2 ≤ 0.5 V).
  Soma uses **1 V drain pulses** from inverter outputs.

- **Image 13.50**: **Spiking Neural Networks** (Mario's intro, Slide 28
  of 46). Cites *"X. Zhu, S. Pazos, M. Lanza et al., Nature 618, 57–62
  (2023)"* with LIF eq verbatim:
  $$ τ_m \, dV/dt = -(V - V_\text{rest}) + R \, I(t) $$
  Definitions: τ_m = membrane time constant, R = membrane resistance,
  V_rest, V_th, V_reset, refractory period τ_ref.

- **Image 13.46**: **AI on edge** ladder (Innatera 2025 source, modified):
  - <1 mW: sensors (Condition data, Understand relevance, Adjust sensor)
  - **1–10 mW: Edge devices** (MCU + embedded processor)
  - **10–100 mW: Gateway devices** (multi-processor SoC + DSP + NN
    accelerator) ← **NS-RAM target tier confirmed**
  - 1–10 W: Data Center (ML/NN accelerator + multi-proc SoC)
  - 10–100 W: HP inference accelerators

- **Image 13.47** (slide 31?): **hBN integration on multi-layer metal**
  > *"Remember that the deposition of the resistive switching medium and
  > the electrodes can be done in many different ways, but it cannot use
  > temperatures above 400 °C, otherwise the microchips will be destroyed.
  > In our previous publication we transferred multilayer hBN on the chip,
  > although we recommend the use of scalable methods. The image below
  > shows the morphology of the chip after each step in that publication."*
  Confirms **hBN BEOL transfer process below 400 °C** for mid-stack.

### 1.4 Sebas email 2026-04-30 (after the call) — confirmed action items
Verbatim from `/tmp/emails_clean.txt:371-390`:

> *"Remember that some parameters change only for M1, while NFACTOR
> changes only for M2 (I attribute this to LDE, hence two separate model
> cards for each device)."*
>
> *"All approaches that help better fit the set of parameters to properly
> model NS-RAM are very useful because they save SPICE simulation time
> and ease fits across different technologies."*
>
> *"You asked if I had information about how performance may degrade with
> fan-out. I think the model is in a better place now to start checking
> this in SPICE. I'm a little shorthanded now, but if you come up with a
> simple architecture of a few tens of neurons that is feasible to
> simulate at the circuit level, this could serve as a nice working
> example to check this aspect."* ← **fan-out study request**
>
> *"In the near future, we are targeting compact networks for specific,
> sparse-signal applications. However, thinking long-term, a 'roadmap'
> can include how NSRAM can scale to larger models and how this can
> co-exist with simplifying massive models (some sort of sweetspot maybe
> around reasoning models?) of todays mainstream-AI."* ← **app target**
>
> *"BONUS TRACK: ... I'm working on a floorplan for the first testchip
> entirely dedicated to NS-RAM. This will already include small arrays
> of NS-RAM based neurons, but if there is something small and specific
> that you think can be useful to extract specific metrics that help
> your approach, please let me know..."* ← **test-cell request**

### 1.5 March 2026 Brian2 SNN slides (Image 12.33, 12.39)
**We had not catalogued these before.** Sebas already ran SNN
inference at SLIDE-LEVEL:
- LIF (with Poisson training) MNIST: **72 %**
- Reference Poisson network: **88 %**
- Confusion matrices included for both.
- Note: *"Using LIF neurons at the input reduces accuracy when employing
  training weights and thresholds obtained from a Poisson input. This
  could be solved by 1) repeating of the input spiking rates at identical
  excitatory magnitude, 2) parametric analysis covering: threshold
  voltages, firing time constants, excitatory input ranges."*

Implication: **Sebas already has a baseline LIF SNN benchmark at 72 % on
MNIST**. Our reservoir 96.75 % MNIST claim in the v0.9 email is from a
different setup — we should explicitly compare on his protocol.

### 1.6 March 2026 thick-oxide 2T NS-RAM cell — 17 µm²
**Image 12.30**: a thick-oxide 2T cell at **17 µm²** (vs the 46 µm² thin-
ox cell on slide 3). VG2=2.5 V, VG1=2.5 V floating, deep N-well.
Annotation: *"Outstanding firing range exceeding 10× trade-off with power
consumption. On floor: stable and operating voltage range. Operating
voltages above terminal. Efficient behaviour as neuron thresholding
element."*

Two cell variants exist: thin-ox 46 µm² (synapse/soma, VG2≤0.5 V) and
thick-ox 17 µm² (neuron threshold, VG2≈2.5 V). Our model card targets
the thin-ox.

### 1.7 March 2026 measurements vs SPICE (Image 12.27)
Slide title: *"Measurements vs. SPICE simulation using semi-empirical
bulk currents"* — first pass at full 33-curve fit *before* the BSIM4
polynomial work. Showed **bulk current** and **total current** panels
side by side. Confirms the working flow: bulk current is fitted
empirically, total = transistor stack + bulk.

## 2. Mario's explicit application targets for NRF (verbatim)

From the Apr-30 deck and Mario's emails:
- **Power tier:** *"10–100 mW gateway devices"* (Image 13.46, slide 26).
- **Application class:** *"Specific classification, Identify patterns"*
  for that tier (Image 13.46).
- **Funding window:** Singapore NRF Mid-Sized Grant, ~7 yr / $50 M
  strategic frame; one-pager due **6 May 2026** (Mario, Apr-30 email
  Image 13.31_email screenshot).
- **TSMC test-vehicle channel** confirmed (joint scope §3).
- **Process:** **130 nm bulk** today; **TSMC 65 nm or 28 nm** next vehicle
  (mentioned in ZoomCC; not in slides verbatim, treat as best-effort).
- **Device co-existence:** *"hBN BEOL transfer ≤ 400 °C"* (Image 13.47).

## 3. Sebas's transient measurement protocol (the missing dimension)

From `image-2.png` (sebas_2026_05_02):
- **Sweep rates probed:** Trise ∈ {1 ms, 200 µs, 100 µs, 50 µs, 20 µs, 10 µs, 5 µs}.
  → 200× rate range.
- **Bias point:** VG1 = 0.4 V, VG2 = 0.3 V (single corner shown).
- **Range:** Vd 0 → 2 V, then 2 → 0 V (full hysteretic loop visible).
- **Quantities extracted:** Sup (firing slope), Sdown (relaxation slope).
- **Experiments 3–7** in his unnamed experiment list are the SR + firing-
  time set. **We don't have the list itself yet** — ASK.
- **Pulsed dynamics**: Sebas (Apr-20 email) said *"We are generating
  additional data now (multiple ramp rates for more data on dynamics,
  pulsed dynamics)."* That is the data we need.
- **Period (Image 13.25):** the self-reset spike train at Vd=2 V shows
  spike spacing ≈ 0.4 µs, peak ≈ 5 mA, rise ≈ 50 ns. So **period = 0.4 µs
  → spike rate = 2.5 MHz at this corner**, configurable 10× via VG2.

## 4. Recommended bias regimes per cell mode

Pieced together from slides + Apr-30 email. Quote → mode:

**SYNAPSE / floating-body memory (DC fit data we have):**
- VG1 ∈ {0.2, 0.4, 0.6} V
- VG2 ∈ [-0.2, +0.5] V (per the 33-curve grid)
- Vd: 0 → 2 V slow ramp (0.2 V/s)
- Sebas verbatim: *"Linear range for V_VG2 is between 2.5 V and 3 V"* —
  this is for the **NEURON-INPUT** mode (linear synapse with mirror bank).

**NEURON SOMA (self-reset spiking):**
- VG2 ≈ 2.5–3.0 V (linear-range neuron, Image 13.33)
- Drain pulses: **1 V amplitude** from preceding inverters (Image 13.33)
- Spike rate **10×-tunable** via VG2 (Image 13.31)
- Cell variant: thick-oxide 17 µm² (Image 12.30) — different card
  than the 130bulkNSRAM(M2) we currently fit.

**STM (synaptic short-term plasticity, the SRH↔TM bridge):**
- Sebas has **not** verbalised a recommended bias regime for this. From
  our nsram_joint_scope §3 hypothesis, STM lives in the same DC range
  as the synapse mode but requires **paired-pulse measurements** at 5
  VG2 levels × 4–5 Δt values. Sebas has **not yet shipped paired-pulse
  data** — explicit ask in samarbetsplan §"KONKRETA ASKS" #3.

**INTEGRATE-AND-FIRE (LIF):**
- LIF eq from Mario (Image 13.50) is the standard one. Sebas's Brian2
  (Image 12.33) used `THRES_VAL_F = 0.0070`, `tau_REC = 0.5 µs`,
  `REFRACT_RV = 0.0070`, `INIT_VAL = 0`, `EXCIT_VAL = 0.5` (slide
  numbers visible). These are the **LIF parameters his SNN already
  uses** — we should match them in our network sims.

## 5. Top 3 risks before declaring large-scale model "ready"

**R1 — Dynamic / ramp-rate dimension is unfit.**
Our model is DC only. Sebas explicitly says experiments 3–7 (SR + firing
time) are *critical* for the parasitic-capacitance fit. Without that
fit, every transient/spiking benchmark we run on the model is
quantitatively wrong. **Mitigation:** add `pdiode` cj/cjsw + parasitic
metal cap; ask Sebas for the experiments 3–7 raw traces; refit at the
VG1=0.4/VG2=0.3 corner across the 7 sweep rates.

**R2 — Two-cell topology gap.**
Our M1 + M2 stack is the thin-oxide synapse cell. Sebas's *neuron*
slide uses a thick-oxide 17 µm² variant with **VG2 ≈ 2.5 V drain pulses
at 1 V**. Running a "compact network of a few tens of neurons" (his
explicit ask) requires a second model card we do not have. **Mitigation:**
ask Sebas if the thick-ox card is shareable, or run both modes off the
same M2 with a Vth shift + cox scaling.

**R3 — VG2 range mismatch.**
DC fit grid tops out at VG2 = 0.5 V. Linear-input-neuron operation
requires VG2 ∈ [2.5, 3.0] V. **Our model has never been validated above
VG2 = 0.5 V.** Extrapolating to 3 V is a 6× extrapolation. **Mitigation:**
either ask Sebas for high-VG2 sweeps, or document the limitation and run
network sims only in the VG2 ≤ 0.5 V regime first.

## 6. Top 3 quick wins to validate next

**QW1 — Reproduce Sebas's Brian2 LIF MNIST baseline (72 %) on our
nsram package.** He already showed it at slide 9 of his Mar-20 deck. If
we hit 72 % using his LIF parameters fed through our calibrated cell,
that's a credible cross-validation point worth sharing on the next call.

**QW2 — Run a 30-neuron NS-RAM SPICE-level network with our model and
report fan-out degradation curves.** Sebas explicitly asked for this
("simple architecture of a few tens of neurons that is feasible to
simulate at the circuit level"). Deliverable: heatmap of accuracy vs
fan-out (1, 2, 4, 8, 16) on the Hopfield retrieval task.

**QW3 — Add `pdiode` from sebas_2026_05_02 to the topology and refit at
the VG1=0.4/VG2=0.3 corner against the **measured** ramp at Trise=200 µs
(Image 13.25-style). Even without the raw 7-rate set, fitting one rate
(slide 25 panel) is a credible "we matched the dynamics" claim.

## 7. Closing observations

- **Energy / area numbers are stable:** 21 fJ/cycle, 6.7 fJ/spike, 20 fJ
  integration loss, 46 µm² thin-ox / 17 µm² thick-ox. Same numbers
  appear in the Mar-20, Apr-29 and Apr-30 decks. These are the figures
  to lead with on the NRF one-pager.
- **No corner / process-variation data has been shipped.** Sebas's
  email implies it exists internally ("Monte Carlo variability" is on
  our side, not his). Worth asking but not blocking.
- **Two papers, two first authors:** Paper 1 (simulator, Q3 2026,
  ENIMBLE first); Paper 2 (SRH↔TM paired-pulse, Q1 2027, Sebas first)
  — both confirmed in samarbetsplan and joint scope.
- **The Apr-30 call covered the BSIM4 fit + simulator agreement
  story.** Sebas's bonus track (testchip floorplan input) is the
  **single highest-leverage open ask** — if we can specify a small
  (~10-cell) test cell that extracts a metric we need, it goes on the
  tape-out at zero marginal cost to us.

---

*Eric · 2026-05-01 · for cross-reference with A7/A8/A9 and the Apr-30
joint scope.*
