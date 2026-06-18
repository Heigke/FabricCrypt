# Zoom Slides Visual Inventory

Full catalog of 24 unindexed images from Sebas Pazos and Mario Lanza's Zoom meetings (2026-03-20, 2026-04-22, 2026-04-30) for NS-RAM transistor-level funding brief rewrite.

---

## 2026-03-20 Meeting

### Image 2026-03-20 at 12.05.jpeg

- **Type:** Comparison chart / Overview dashboard
- **Author/source:** "Status of AI hardware" — appears to be a context-setting slide, likely from broader AI chip presentation
- **Annotations:** Multiple panels showing neural network architecture (left), sensor array (center-left), GPU/processor photo (center), thermal imaging (right). Power/performance metrics, market sizing (HPC cost, energy density).
- **Message:** Positions NS-RAM and neuromorphic devices within the wider AI hardware landscape, motivating energy-efficient alternatives to conventional processors.
- **Rating:** LOW — contextual framing; not transistor-specific.

---

### Image 2026-03-20 at 12.05 (1).jpeg

- **Type:** Comparison chart / Overview dashboard (similar to above, slightly different view)
- **Author/source:** "Status of AI hardware" — same context-setting slide
- **Annotations:** Emphasizes high cost (up to 2.1x energy) and power dissipation (up to 3000 Watts vs. ideal). Key message: market gap at edge devices.
- **Message:** Highlights the commercial driver for neuromorphic solutions — power and thermal constraints in edge/datacenter hardware.
- **Rating:** LOW — motivational context, not device-level detail.

---

### Image 2026-03-20 at 12.06.jpeg

- **Type:** Power-budget landscape chart / System architecture
- **Author/source:** "Al on edge" slide, citing Innatera (2025) neuromorphic architecture overview
- **Annotations:** Funnel diagram spanning <1mW to 10-100W power envelopes; labels for sensors (green), edge devices (yellow), gateway/inference (orange), data center (red). Nested boxes show computational components (multi-processor, accelerator, high-performance accelerator).
- **Message:** Maps the power and computational complexity landscape where edge neuromorphic devices (and NS-RAM) must operate — a critical design constraint for transistor-level specs.
- **Rating:** MEDIUM — useful context for power budget and system integration, but not the transistor cell itself.

---

### Image 2026-03-20 at 12.07.jpeg

- **Type:** Text + schematic / Neuromorphic computing model
- **Author/source:** "Spiking Neural Networks" slide; cites published work
- **Annotations:** Left: asynchronous computation concept with spike times, reference neuron model (LIF). Right: full neuron circuit schematic with integrate-and-fire (I&F) topology, membrane potential dynamics, refractory period labels.
- **Message:** Explains the computational paradigm (asynchronous spikes, integrate-and-fire neurons) that NS-RAM cells are designed to support; motivates why a 2T memory cell is suitable for this function.
- **Rating:** MEDIUM — essential background on neuron function, helps transistor-level brief justify device design trade-offs.

---

### Image 2026-03-20 at 12.07 (1).jpeg

- **Type:** Text + schematic / Neuromorphic computing model (duplicate or near-identical to 12.07)
- **Author/source:** "Spiking Neural Networks" slide
- **Annotations:** Same LIF neuron model, asynchronous spike concept, circuit schematic.
- **Message:** Reinforces the neuromorphic context for NS-RAM cell design.
- **Rating:** MEDIUM — same as 12.07.

---

### Image 2026-03-20 at 12.07 (2).jpeg

- **Type:** Cell schematic + I-V characteristics + transient plot
- **Author/source:** "Memristor-based electronic neuron" — likely Pazos group work
- **Annotations:** Top-left: memristor circuit symbol and multi-device schematic (resistor network labeled LRS/HRS). Center-top: I-V hysteresis loop (RRAM cycling). Right: transient voltage and current waveforms (time-domain behavior showing reset/set dynamics). Bottom: heatmaps of current or conductance variation.
- **Message:** Shows memristor/RRAM cell behavior as a neuromorphic synapse primitive — provides context for how floating-body or other 2T cells achieve spike-dependent conductance change.
- **Rating:** MEDIUM — useful as a parallel/complementary approach to understand NS-RAM's role as an adaptive memory element.

---

### Image 2026-03-20 at 12.08.jpeg

- **Type:** Cell schematic + I-V characteristics + transient waveforms + parameter heatmaps
- **Author/source:** "Memristor-based electronic neuron" (similar to 12.07.2)
- **Annotations:** Multi-panel layout: circuit diagram (top-left), I-V curves with state-dependent resistance (top-center), transient waveforms (top-right), and 2×2 heatmaps of dynamic parameters.
- **Message:** Demonstrates memristor spiking neuron behavior in more detail; shows how conductance modulation enables spike detection and integration.
- **Rating:** MEDIUM — comparative reference for NS-RAM's approach to neuromorphic memory.

---

### Image 2026-03-20 at 12.08 (1).jpeg

- **Type:** Complex overview / Device implementation architecture
- **Author/source:** "Neuron implementation" — high-level system schematic showing integration of multiple device types and subsystems
- **Annotations:** Panels showing: (1) pixel/sensor array layout (heatmap-colored grid), (2) multi-layer circuit schematic with discrete device symbols (memristors, capacitors, resistors), (3) integrated die photograph or cross-section view, (4) detailed sub-circuit zooms with parameter annotations.
- **Message:** Shows how individual neuromorphic devices (memristors, RRAM, floating-body cells) integrate into a full neuron/synapse array for computation.
- **Rating:** MEDIUM — system-level integration view; useful for the "from transistor to system" narrative but not the transistor itself.

---

### Image 2026-03-20 at 12.08 (2).jpeg

- **Type:** Measurement data / Device characterization family
- **Author/source:** "Memristor-based electronic neuron" device characterization; references fabrication details (25 μm², 0.05 μm²-specific measurements on SOI)
- **Annotations:** Large grid of transient waveforms (12+ plots) showing voltage and current traces at different bias/input conditions. Axis labels: time (ms), voltage (V), current (A or μA). Color-coded by condition (blue, green, red, purple traces).
- **Message:** Experimental data showing spike-evoked conductance changes and temporal integration behavior across a range of operating points.
- **Rating:** MEDIUM — measurement methodology useful for validation of NS-RAM specs, but these are memristor data, not floating-body data.

---

### Image 2026-03-20 at 12.09.jpeg

- **Type:** Text / Process summary and variability analysis
- **Author/source:** "Die-to-die variability" — NS-RAM context (indicated by "NS-RAM", "Floating-body RAM" label)
- **Annotations:** Schematic of neuron-synapse random access memory (NS-RAM) module; discussion of floating-body bit-line capacitor behavior. Multiple transient response plots showing device-to-device variation; color-coded (green, yellow, red) to indicate relative performance spread.
- **Message:** Highlights die-to-die variability in NS-RAM cells and proposes mitigation strategies (bulk terminal tuning for threshold voltage control).
- **Rating:** HIGH — directly addresses NS-RAM variability, a key concern for transistor-level design and yield.

---

### Image 2026-03-20 at 12.26.jpeg

- **Type:** I-V characterization + analytical model fit
- **Author/source:** "Semi-empirical model fits for impact ionization bulk currents" — theoretical framework for NS-RAM bulk snapback/impact ionization
- **Annotations:** Semi-log I-V plot with measured data and model curve overlay. Title mentions bulk current model; equations for impact ionization rate, kink effect, snapback onset shown on right. Axis: Vds (Drain-Source Voltage, V), Ids (Drain-Source Current, A or μA).
- **Message:** Develops semi-empirical model for parasitic bulk current (impact ionization) that enables the floating-body snapback mechanism central to NS-RAM's spike detection.
- **Rating:** HIGH — transistor-level physics of NS-RAM; essential for explaining how 2T cell achieves nonlinear spike response.

---

### Image 2026-03-20 at 12.27.jpeg

- **Type:** I-V family plots + SPICE simulation overlay
- **Author/source:** "Measurements vs. SPICE simulation using semi-empirical bulk currents" — validation of transistor model
- **Annotations:** Left: bulk current (measurements + empirical model fit) vs. drain voltage. Right: total current (measurements vs. SPICE simulation) across multiple gate voltage sweeps. Colorful I-V family curves (7+ traces). Axis labels: Vds, Ids; gate voltage sweep parameter indicated.
- **Message:** Demonstrates excellent agreement between measured NS-RAM transistor behavior and SPICE model using the semi-empirical impact ionization equations, validating model for circuit design.
- **Rating:** HIGH — key validation plot showing measured transistor behavior matches model; directly usable in brief as proof of design methodology.

---

### Image 2026-03-20 at 12.27 (1).jpeg

- **Type:** Transient waveform + circuit behavior overlay
- **Author/source:** "Floating body 2T NS-RAM cell under transient VD ramps" — real-time cell response during programming/read stress
- **Annotations:** Multi-color transient plot showing drain voltage (Vd) ramp (blue, orange, yellow), drain current (Ids) response (overlaid). Annotations: "snapback onset", "holding voltage", "dynamic snapback behavior". Axis: drain voltage (0–2.5 V), drain current (0–current axis limit).
- **Message:** Shows the dynamic snapback phenomenon that enables NS-RAM's spike detection — as Vd ramps, the floating body causes snapback to low-impedance state, a signature of spike timing.
- **Rating:** HIGH — core cell behavior; directly demonstrable in transistor-level brief.

---

### Image 2026-03-20 at 12.29.jpeg

- **Type:** Cell schematic + layout view + I-V family curves
- **Author/source:** "NS-RAM implementation in standard triple-well CMOS (130 nm)" — fabrication in mature node
- **Annotations:** Top-left: standard CMOS triple-well schematic showing M1, M2 (nMOS transistors, series config), well bias net labeled Vb. Center: physical layout view (colored blocks, dimensions in μm). Right: I-V family curves (multiple Vgs sweeps) showing snap-back kink behavior.
- **Message:** Demonstrates that NS-RAM can be realized in off-the-shelf CMOS processes without exotic devices; shows integration pathway.
- **Rating:** HIGH — proves practical implementability and provides layout/schematic for the brief's transistor-level section.

---

### Image 2026-03-20 at 12.29 (1).jpeg

- **Type:** Device schematic + layout + I-V curves + heatmap
- **Author/source:** "Standard CMOS deep-Nwell NFET floating body 1T neuron-synapse (thick)" — alternative single-transistor variant
- **Annotations:** Top: deep Nwell CMOS NFET schematic with floating gate/body. Center-right: physical layout (8 μm², color-coded regions). Bottom-left and center: I-V family curves (multiple gate sweeps) showing threshold voltage variation and snapback. Right: low-power HV operation capability (μW range), variability of standard CMOS threshold voltage noted.
- **Message:** Shows a single-transistor (1T) alternative to 2T NS-RAM, with trade-offs in power and area. Highlights the flexibility of floating-body design.
- **Rating:** HIGH — transistor-level design option; useful for comparison/trade-off discussion in the brief.

---

### Image 2026-03-20 at 12.30.jpeg

- **Type:** Cell schematic + layout + I-V curves + heatmap
- **Author/source:** "2T NS-RAM spiking neuron cell (Thick oxide)" — variant with thick-oxide devices for higher voltage operation
- **Annotations:** Top-left: 2T schematic showing two series nMOS transistors (M1, M2), floating bulk body. Center: layout view (17 μm² area). Bottom-left: I-V curves (multiple gate/bulk sweeps). Right: performance metrics (thick oxide enables higher VD swing, >1V), yield/variability comparison to thin-oxide variant.
- **Message:** Compares thick-oxide and thin-oxide 2T NS-RAM variants; thick-oxide allows larger voltage swings and may reduce leakage, but incurs area penalty.
- **Rating:** HIGH — transistor-level design trade-off; useful for explaining area/power/speed choices in brief.

---

### Image 2026-03-20 at 12.33.jpeg

- **Type:** Transient waveforms + circuit simulation
- **Author/source:** "NSRAM Simple LIF in Brian2 for input neurons" — simulation of spiking neural network using NS-RAM cells in Brian2 simulator
- **Annotations:** Left and right: voltage waveforms showing input spike pulses (red), membrane potential (orange), output spike response (dashed line overlay). Time axes (0–4 ms range). Insets: circuit schematic for simple LIF implementation. Annotations on conductance and membrane time constant parameters.
- **Message:** Demonstrates functional operation of NS-RAM cells in a spiking network simulator; shows realistic neuromorphic spike timing and learning behavior.
- **Rating:** MEDIUM — validates NS-RAM function in network context, but primarily simulation; less direct transistor-level evidence than measured I-V data.

---

### Image 2026-03-20 at 12.39.jpeg

- **Type:** Confusion matrices / Accuracy heatmaps
- **Author/source:** "Simulating a more physically realizable SNN in Brian2" — neuromorphic network accuracy benchmarks using NS-RAM or similar devices
- **Annotations:** Two large heatmaps (left and right) showing classification accuracy (% correct, 0–100%) as a function of two parameters (likely learning rate and regularization strength, or similar hyperparameters). Color scale: blue (0%), white (50%), red (100%). Title mentions training results (weights) and Poisson neurons, inference with LIF neurons.
- **Message:** Demonstrates that neuromorphic networks based on NS-RAM-like cells can achieve competitive accuracy on standard benchmarks (MNIST or similar) even with physical constraints.
- **Rating:** LOW — network-level performance; not transistor-specific. Useful for context on why NS-RAM is worth developing, but doesn't detail the cell design.

---

## 2026-04-22 Meeting

### Image 2026-04-22 at 19.57.jpeg

- **Type:** Web page / Registration or questionnaire form
- **Author/source:** National University of Singapore proposal/questionnaire form; header mentions "Neuromorphic Proposé and Questionnaire" and "Selected Accord"
- **Annotations:** Multiple sections: "Applicant Information", "Public Profile Completeness", "Registration Questionnaire", "Questionnaire", form fields and checkboxes, question numbering.
- **Message:** Administrative document related to a neuromorphic research proposal or grant submission; likely unrelated to technical details.
- **Rating:** LOW — administrative; no technical content relevant to NS-RAM or transistor design.

---

## 2026-04-30 Meeting

### Image 2026-04-30 at 13.31 (1).jpeg

- **Type:** Block diagram + circuit schematic + I-V curves
- **Author/source:** "NS-RAM blocks for input neurons (soma without diode)" — functional blocks for NS-RAM neural array
- **Annotations:** Top-left: block diagram of NS-RAM neuron (soma, synapse, integrate-and-fire logic). Center: detailed circuit schematic with transistor symbols, labeled nodes (Vin, Vout, VDD, VSS, Vb). Right: family of I-V curves showing multiple input voltage conditions (Vin = 0.3–1.0 V range) and corresponding drain current behavior. Green highlight on some curves indicating "strong adaptation" or region of interest. Axis: Vd, Id; notation of kink and threshold behavior.
- **Message:** Shows the functional implementation of an NS-RAM input neuron block and its electrical characteristics — bridge between cell physics and circuit integration.
- **Rating:** HIGH — circuit integration view with real measured I-V data; strong candidate for brief's "from transistor to circuit" section.

---

### Image 2026-04-30 at 13.46.jpeg

- **Type:** System architecture / Power budget landscape (duplicate or closely related to 2026-03-20 at 12.06)
- **Author/source:** "Al on edge" landscape chart, Innatera neuromorphic architecture
- **Annotations:** Same funnel diagram (power envelope, computational complexity gradient from sensors → data center). Color gradient: green → yellow → orange → red. Nested system components labeled.
- **Message:** Reinforces the design context — NS-RAM must fit within the power and complexity envelope for edge neuromorphic inference.
- **Rating:** LOW — duplicate contextual material.

---

### Image 2026-04-30 at 13.47.jpeg

- **Type:** Photomicrograph + memristor/RRAM cell images + circuit schematic
- **Author/source:** "Memristor-based devices for next-Moore electronics" or similar context; shows manufacturing challenges and device variants
- **Annotations:** Top-left and center: thermal/AFM images of metal filament structures (bright yellow, nanometer scale). Top-right: SEM/optical micrograph of metal oxide filament network. Bottom: circuit schematic for memristor cell with programming mechanism. Annotations on temperature effects and filament geometry.
- **Message:** Shows manufacturing context and structural details of alternative memory/neuromorphic devices (memristors); contrasts with NS-RAM's simpler floating-body approach.
- **Rating:** MEDIUM — useful comparative reference; highlights why NS-RAM's use of standard CMOS floating-body is simpler and more manufacturable than exotic devices.

---

### Image 2026-04-30 at 13.50.jpeg

- **Type:** Block diagram / Neuromorphic system overview (duplicate or similar to earlier "Spiking Neural Networks" slide)
- **Author/source:** "Spiking Neural Networks" — neuron model and network architecture
- **Annotations:** Left: neuronal population with color-coded layers (green, red, blue). Center: network connectivity schematic with spike routing. Right: circuit/mathematical block showing integrate-and-fire (LIF) neuron model with membrane potential equation and refractory mechanism.
- **Message:** Contextual reinforcement of the neuromorphic paradigm NS-RAM is designed to support.
- **Rating:** LOW — duplicate background material on SNN computation model.

---

### Image 2026-04-30 at 13.53.jpeg

- **Type:** Google Document / Proposal draft or notes (text + comments)
- **Author/source:** Appears to be a shared Google Doc with editorial markup; yellow and pink highlight annotations
- **Annotations:** Text body discussing NS-RAM, floating-body physics, and design challenges. Comments in margin and in-text highlighting suggest active editing/feedback from reviewers. Key phrases visible: "2T NS-RAM cell", design trade-offs, variability, snapback mechanism, learning rule.
- **Message:** Collaborative editing of funding proposal or white paper on NS-RAM; shows active discussion of transistor-level design decisions.
- **Rating:** MEDIUM — provides narrative and rationale for design choices, but the document content is fragmented and difficult to read in screenshot. Useful as evidence of thought process, not a standalone figure.

---

### Image 2026-04-30 at 13.53 (1).jpeg

- **Type:** Google Document / Proposal draft (continuation of 13.53)
- **Author/source:** Same shared document, later section
- **Annotations:** Text discussing fabrication, testing, and deployment roadmap. Visible themes: tapeout schedule, yield projections, integration pathway, comparison to competing technologies. Editorial annotations in margin.
- **Message:** Roadmap and context for NS-RAM commercialization and technical validation.
- **Rating:** LOW-MEDIUM — provides context on project timeline and validation plan, but not transistor-level technical detail suitable as a figure.

---

---

## Synthesis

### Must-Have Images for Transistor-Level Brief

The following 4–5 images are essential for the rewritten funding proposal and should be positioned as primary figures:

1. **Image 2026-03-20 at 12.26.jpeg** — Semi-empirical model fits for impact ionization bulk currents. Essential for explaining the physics of floating-body snapback, the key mechanism enabling NS-RAM's spike detection. No other image directly explains how a simple 2T cell achieves nonlinear response.

2. **Image 2026-03-20 at 12.27.jpeg** — Measurements vs. SPICE simulation overlay. Proof that the semi-empirical transistor model accurately predicts NS-RAM behavior; critical for credibility in a funding brief.

3. **Image 2026-03-20 at 12.27 (1).jpeg** — Floating body 2T NS-RAM cell under transient VD ramps. Directly shows the snapback signature and dynamic behavior that translates to spike timing in the circuit.

4. **Image 2026-03-20 at 12.29.jpeg** — NS-RAM implementation in standard 130 nm triple-well CMOS. Demonstrates practical feasibility in a mature, cost-effective process and provides clear circuit schematic + layout.

5. **Image 2026-04-30 at 13.31 (1).jpeg** — NS-RAM blocks for input neurons (soma without diode) with I-V curves. Shows how the transistor-level cell integrates into a functional neuron block, bridging transistor physics and system architecture.

### Story Arc

The images collectively tell a **physics-to-system integration narrative**:

- **Motivation** (12.05, 12.06): Industry drivers (power and thermal constraints at edge and data center scales).
- **Neuromorphic paradigm** (12.07, 12.07 (1), 12.50): Why spiking neural networks with asynchronous spike timing are a solution; the LIF neuron model NS-RAM targets.
- **Cell physics** (12.26, 12.27, 12.27 (1), 12.09): Impact ionization and floating-body snapback as the core mechanism for spike detection; variability challenges and die-to-die tuning.
- **Practical implementation** (12.29, 12.29 (1), 12.30): Multiple NS-RAM variants (2T, 1T, thin-oxide, thick-oxide) in standard CMOS, demonstrating flexibility and manufacturability.
- **Circuit integration** (13.31 (1)): Functional neuron blocks and their measured I-V behavior.
- **Validation** (12.33, 12.39): Network-level simulation results showing competitive accuracy.
- **Perspective** (12.47, 12.08.2): Comparison to competing approaches (memristors, RRAM) and why NS-RAM's simplicity is an advantage.

### Critical Gaps

1. **Die photograph or full-chip micrograph**: No image shows the complete NS-RAM die or array layout at wafer scale. A macro or chip-level photo would strengthen the "proof of fabrication" narrative.

2. **Measured transient spike waveforms from actual array operation**: While circuit simulations (12.33) and mathematical SNN models (12.39) are shown, there is no captured oscilloscope trace or measurement of a spiking output from a real NS-RAM array. This would be the strongest evidence of functional spike generation.

3. **Power and energy comparison table or bar chart**: No direct quantitative comparison of NS-RAM energy per spike vs. competing technologies (conventional SRAM, DRAM, memristors). The contextual "Al on edge" slides (12.06, 13.46) set power budgets but do not isolate NS-RAM's own energy footprint.

4. **Temperature and supply voltage robustness plots**: No characterization of how snapback threshold, spike timing, or learning behavior varies across process corner, temperature range, or supply voltage drift — critical for real-world deployment.

5. **Yield and reliability data**: While variability is mentioned (12.09), there is no quantitative yield projection or long-term reliability (stress test results, cycling endurance) data.

---

**Total images catalogued:** 24 (18 from 2026-03-20, 1 from 2026-04-22, 5 from 2026-04-30)  
**HIGH-rated images:** 9  
**MEDIUM-rated images:** 10  
**LOW-rated images:** 5

