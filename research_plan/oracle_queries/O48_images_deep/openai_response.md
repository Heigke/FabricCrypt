# openai response (gpt-5) — 248s

Below I treat each of the 22 images in order of appearance. For every image I identify the circuit topology and ports, test configuration, any “extra” non‑standard elements, plot structure, key text/labels, and how it cross‑references the rest of the packet.

### Image 01
A. CIRCUIT TOPOLOGY
- Parameter extraction for the 2T NS‑RAM cell, specifically the leakage transistor M2.
- Port naming implied by later schematics: M2 gate = VG2, M2 drain = floating body node VB, M2 source = VNEG (normally 0 V).

B. TEST CONFIGURATION
- DC extraction: NFACTOR of M2 vs VG2, for three different VG1 biases on the main device M1.

C. NUMBERED/NON‑STANDARD ELEMENTS
- None drawn here; this is a parameter plot.

D. PLOT STRUCTURE
- Y: NFACTOR (for M2), unitless, range ~2 to ~12, linear scale.
- X: VG2 (V), −0.2 to 0.5 V, linear.
- Three dashed series: red (highest NFACTOR at negative VG2), blue (mid), black (lowest).

E. TEXT/ANNOTATIONS
- Axis labels explicitly say “NFACTOR (for M2)” and “VG2 (V).”

F. CROSS‑REFERENCE
- Matches the bottom‑right panel in Image 07 (same three colored branches labeled by VG1 = 0.20/0.40/0.60 V). Confirms M2 parameterization depends on VG2 and VG1.

### Image 02
A. CIRCUIT TOPOLOGY
- Parameter for main transistor M1 in the 2T cell.

B. TEST CONFIGURATION
- DC extraction: K1 (for M1) vs VG1.

C. NUMBERED/NON‑STANDARD ELEMENTS
- None.

D. PLOT STRUCTURE
- Y: K1 (for M1), unitless, ~0.41–0.56.
- X: VG1 (V), 0.2–0.6 V.
- Single red dashed curve decreasing with VG1.

E. TEXT/ANNOTATIONS
- “K1 (for M1)” and “VG1 (V).”

F. CROSS‑REFERENCE
- Same as the bottom‑left panel in Image 07 (“For all VG2”). This is the VG1‑only parameter used in the three‑branch fitting.

### Image 03
A. CIRCUIT TOPOLOGY
- Parameter of M1 (impact‑ionization branch dependence) vs VG2.

B. TEST CONFIGURATION
- DC extraction: ETAB (for M1) vs VG2, three VG1 branches.

C. NUMBERED/NON‑STANDARD ELEMENTS
- None.

D. PLOT STRUCTURE
- Y: ETAB f(for M1), ~0.8–2.5 (linear).
- X: VG2 (V), −0.2 to 0.5 V (linear).
- Three dashed colored series: red (≈0.8→1.1 rising), blue (≈1.9→1.6 falling), black (≈2.5 nearly flat then drop beyond ≈0.4–0.5 V).

E. TEXT/ANNOTATIONS
- “ETAB f(for M1).”

F. CROSS‑REFERENCE
- Same trends and colors as the upper‑right panel of Image 07.

### Image 04
A. CIRCUIT TOPOLOGY
- Parameter of M1 vs VG2: BETA0 (impact‑ionization strength scale).

B. TEST CONFIGURATION
- DC extraction: BETA0 (for M1) vs VG2, three VG1 branches.

C. NUMBERED/NON‑STANDARD ELEMENTS
- None.

D. PLOT STRUCTURE
- Y: BETA0 (for M1), ≈10–21.
- X: VG2 (V), −0.2 to 0.5 V.
- Three dashed colored sets: black ≈20 flat, blue ≈19 flat, red ≈10.7→~14 increasing with VG2.

E. TEXT/ANNOTATIONS
- Axis text explicitly identifies M1.

F. CROSS‑REFERENCE
- Matches the upper‑left panel of Image 07. These parameters feed the semi‑empirical Iion model used in Images 13–15.

### Image 05
A. CIRCUIT TOPOLOGY
- Left: 2T NS‑RAM schematic.
  - M1 (main NFET with floating P‑body) drain = VD, source = 0 V, gate = VG1, body = VB (floating).
  - M2 (leak/bleed NFET) drain = VB, source = VNEG (0 V), gate = VG2.
  - Explicit capacitor from VB to VG2 (gate‑controlled MOS capacitor used as body coupler).
- Right: Two ID–VD families of the same cell.

B. TEST CONFIGURATION
- DC sweeps: ramp VD (0→~3.5 V), log ID on Y.
- Families for fixed VG2 extremes: left panel says VG2 = 1.4 V; right panel says VG2 = 0.1 V.
- Within each family VG1 is stepped (0.25→0.45 V per annotation).

C. NUMBERED/NON‑STANDARD ELEMENTS
- VB–VG2 coupling capacitor.
- Floating‑body node explicitly drawn.
- No well contact on VB; body is intentionally floating.

D. PLOT STRUCTURE
- Two log(ID) vs VD plots.
- Left: higher VG2 shows earlier/steeper turn‑on and larger post‑avalanche current.
- Right: with VG2 = 0.1 V, S‑shaped region moves to higher VD; clear negative‑resistance “knee” location shifts.

E. TEXT/ANNOTATIONS
- Arrows indicating VG1 stepping in both plots.
- “VG2 = 1.4 V” (left) and “VG2 = 0.1 V” (right).

F. CROSS‑REFERENCE
- Schematic matches the symbol inset in Image 06, and the device cross‑section/topology in Images 18 and 21.
- The high VG2 = 1.4 V case is older/high‑oxide-bias data and differs from later 0.5 V max in Image 06/07.

### Image 06
A. CIRCUIT TOPOLOGY
- 2T NS‑RAM cell; inset schematic labels M1 (VG1), M2 (VG2), VB node between them; M1 drain is the swept “+VD”; sources at 0 V.

B. TEST CONFIGURATION
- Three DC ID–VD sweeps:
  - Left: VG1 = 0.6 V, VG2 stepped 0→0.5 V (0.05 V steps).
  - Middle: VG1 = 0.4 V, VG2 stepped 0→0.3 V.
  - Right: VG1 = 0.2 V, VG2 stepped −0.2→0.1 V.
- Symbols = measurements, Lines = simulations.

C. NUMBERED/NON‑STANDARD ELEMENTS
- Floating body node VB explicitly marked in the inset.
- No explicit well diode drawn here, but present in later Image 21.

D. PLOT STRUCTURE
- Log(Current) vs Voltage (VD).
- Distinct families per VG2; turn‑on knee shifts with VG2 and VG1.
- Simulation lines overlay measurements closely.

E. TEXT/ANNOTATIONS
- Stepping ranges and step size written inside each panel.
- Schematic labels “M1, M2, VG1, VG2, VB, +VD”.

F. CROSS‑REFERENCE
- These are the raw families from which the parameters in Images 01–04/07 were extracted.
- Matches the transient‑ramp comparison in Image 15.

### Image 07
A. CIRCUIT TOPOLOGY
- Consolidated parameter‑vs‑bias plots for the 2T cell.
- Parameters tagged to devices:
  - For M1: BETA0(VG2; VG1 branch), ETAB(VG2; VG1 branch), K1(VG1) “for all VG2”.
  - For M2: NFACTOR(VG2; VG1 branch).

B. TEST CONFIGURATION
- DC extraction from ID–VD families like Image 06.
- Three VG1 branches: 0.20 V (red), 0.40 V (blue), 0.60 V (black).

C. NUMBERED/NON‑STANDARD ELEMENTS
- None on the plots; but the underlying model is a “three‑branch” (impact‑ionization, channel, leakage) model.

D. PLOT STRUCTURE
- 2×2 grid:
  - Top‑left: BETA0 vs VG2, flat at high VG1, rising with VG2 at VG1 = 0.2 V.
  - Top‑right: ETAB vs VG2, magnitude depends strongly on VG1, opposite slopes for red vs blue/black.
  - Bottom‑left: K1 vs VG1, monotonic decrease ~0.56→0.41.
  - Bottom‑right: NFACTOR (M2) vs VG2, strong decrease with VG2; higher at lower VG1.
- Linear axes in all four.

E. TEXT/ANNOTATIONS
- Explicit legends on each panel: “VG1 = 0.20/0.40/0.60 V”, “for M1” or “for M2”.
- The phrase “For all VG2” on K1 panel.

F. CROSS‑REFERENCE
- This is the same content as Images 01–04 combined; likely the source for “three_branch_params_extracted.json”.

### Image 08
A. CIRCUIT TOPOLOGY
- Same 2T NS‑RAM cell under repeated VD ramps.
- No schematic shown here, but lower panels correspond to dynamic VD ramp on M1.

B. TEST CONFIGURATION
- Top: time‑domain stimulus (blue voltage ramp with rise/fall; red is measured current spike envelope).
- Bottom two: cloud of ID–VD points gathered during repetitive triangular ramps; selected markers (red dots) indicate reference points on the trajectory (start, pre‑knee, post‑knee).

C. NUMBERED/NON‑STANDARD ELEMENTS
- None explicit; parasitic capacitances implied by the dynamic behavior.

D. PLOT STRUCTURE
- Bottom panels: log(Current) vs Voltage with dense gray traces; superimposed thick black “mean” trace; red markers at three voltages.
- Left and right panels show two bias cases (not text‑labelled here), likely with/without well‑diode or different VG2 (consistent with Image 21).

E. TEXT/ANNOTATIONS
- Time axis “Time (µs)” on top plot; current axis in mA for the red trace and V for the blue.

F. CROSS‑REFERENCE
- Directly related to “ramp rate dependence” in Image 21 and to transient overlays in Image 15.

### Image 09
A. CIRCUIT TOPOLOGY
- 2T cell; measurement vs SPICE on ID–VD.

B. TEST CONFIGURATION
- DC VD sweep with fixed VG1, VG2 combinations:
  - Three annotated examples: (VG1, VG2) = (0.60, 0.35), (0.40, 0.25), (0.20, 0.00) V.

C. NUMBERED/NON‑STANDARD ELEMENTS
- None in the plot.

D. PLOT STRUCTURE
- Log(Current) vs Voltage; “Thick = measurements, Thin = simulations.”
- Good agreement of both knee location and post‑knee slopes.

E. TEXT/ANNOTATIONS
- The three bias labels are printed on the curves.

F. CROSS‑REFERENCE
- These three slices are consistent with families in Image 06 and parameters in Image 07.

### Image 10
A. CIRCUIT TOPOLOGY
- System cell: integrator + 2T NS‑RAM used as a self‑reset spiking soma.
  - Vmem node integrates I_excit through Cint (explicit capacitor 102 fF shown).
  - M1 gate = Vleak (sets passive leak), M2 gate = Vinteg (sets integration time/leak to ground through the body).
  - M1 drain (VD) is tied to Vmem (so the ramped “drain” is the membrane).
  - VB (the floating body/M2 drain) is the spiking node Vspike.
  - Sources at ground; VNEG = 0 V.

B. TEST CONFIGURATION
- Transient excitation: bursts of excitatory current pulses (blue raster at top).
- Measured/simulated Vspike waveforms vs time for different (VG1, VG2) = (Vleak, Vinteg) settings.

C. NUMBERED/NON‑STANDARD ELEMENTS
- The integrator capacitor Cint.
- The NS‑RAM floating body node used as a voltage output (Vspike).
- Load at Vmem contributes extra capacitance (note on the slide).

D. PLOT STRUCTURE
- Time‑domain plots: Vspike in 0–0.7 V range, multiple panels for different VG1/VG2.
- Top bar shows “LEAK” intervals between bursts.

E. TEXT/ANNOTATIONS
- “Self‑reset,” “Energy consumption as low as 0.2 pJ/spike,” device area notes (111 µm²).
- Explicit VG1/VG2 values beside each waveform panel.

F. CROSS‑REFERENCE
- The mapping of node names to device terminals matches Image 21’s schematic (M1 VD connected to the driving node, body as the observable spike).

### Image 11
A. CIRCUIT TOPOLOGY
- Larger NS‑RAM‑based soma macro (without explicit well diode), with mirrored biasing and starved inverter stacks.
  - Inputs: I_excitary, control biases Vtau, Vw_exc/Vs_exc and Vw_inh/Vs_inh.
  - Outputs: Vmem and Vspike.
  - Supplies: VPOS5 (thick‑oxide 5 V domain), VDD, Vint.

B. TEST CONFIGURATION
- Transient sims: energy accumulation vs time for several excitatory currents (500 pA→5 nA).
- Vspike traces at firing rates 60–360 kHz.

C. NUMBERED/NON‑STANDARD ELEMENTS
- Starved inverter branches providing 1 V pulses to the NS‑RAM soma.
- Multiple thick‑oxide devices to tolerate higher node voltages.

D. PLOT STRUCTURE
- Left: energy vs time (linear).
- Right: multiple Vspike vs time strips; insets of cumulative energy (fJ).

E. TEXT/ANNOTATIONS
- “VG2 = Vinteg = 0.275 V, VG1 = Vleak = 0 V.”
- Energy breakdown: ~21 fJ/spike, ~0.7 fJ spike generation, ~20 fJ integration; area ~60 µm².

F. CROSS‑REFERENCE
- Functionally compatible with Image 10’s simpler cell, sharing Vmem/Vspike naming and the two‑bias control (VG1/VG2).

### Image 12
A. CIRCUIT TOPOLOGY
- NS‑RAM soma interfaced to linear excitatory and inhibitory synapse blocks.
  - Two mirrored current DACs (weights via Vw_* and sources Vs_*), summed at a central node that drives the soma.
  - Thick‑oxide stacks to handle soma’s higher VD.

B. TEST CONFIGURATION
- Concept slide; no raw data plots here—intended operation is soma receiving two signed inputs.

C. NUMBERED/NON‑STANDARD ELEMENTS
- Explicit use of thick vs thin‑oxide transistors.
- Bias nodes Vtau, Vw_exc, Vw_inh, Vs_next.

D. PLOT STRUCTURE
- Schematic and conceptual spike cartoons only.

E. TEXT/ANNOTATIONS
- “Linear range for Vw_x is between 2.5 V and 3 V.”
- “Devices are designed to operate with 1 V pulses emerging from starved inverters.”

F. CROSS‑REFERENCE
- This is the system‑level wrapper around the same 2T cell as in Images 10–11.

### Image 13
A. CIRCUIT TOPOLOGY
- Model slide for bulk (impact‑ionization) current of M1. No circuit drawing beyond standard transistor.

B. TEST CONFIGURATION
- DC VD sweeps at fixed VG1; IB vs VD extracted; fit with semi‑empirical sum of exponential and power‑law terms.

C. NUMBERED/NON‑STANDARD ELEMENTS
- The model itself:
  - Iion = Iexp + Ipow, with Ipow = d(VD+f)^e for VD > −f; Iexp = a·exp{b(VD + c)}.
  - a,b,d,e,f are PWL functions of VG; c is constant.

D. PLOT STRUCTURE
- IB (A) vs VD (V) on log‑linear axes.
- Curves: measurement, exponential term, power‑law term, and sum.

E. TEXT/ANNOTATIONS
- “VG1 = 0.15 V” on example plot.
- Notes: “Dependence on body voltage VB not included in this revision.”

F. CROSS‑REFERENCE
- Parameters here correspond to BETA0/ETAB/K1 in Images 07 and 22.
- Used to build SPICE model compared in Image 14 and 15.

### Image 14
A. CIRCUIT TOPOLOGY
- Same M1/M2 cell; left panel isolates IB (bulk), right panel shows total ID.

B. TEST CONFIGURATION
- DC VD sweeps at many VG1 values (0.00→0.55 V, legend).
- VB fixed at 0 V for this dataset (explicitly stated).

C. NUMBERED/NON‑STANDARD ELEMENTS
- None; emphasizes that VB‑dependence is future work.

D. PLOT STRUCTURE
- Left: IB vs VD, log scale, colored symbols = measurements, dashed lines = SPICE model; VG1 increases downward/leftward.
- Right: ID vs VD total current; measurements vs simulations.

E. TEXT/ANNOTATIONS
- “Excellent fit with VB = 0 V.”
- Legend of VG1 values.

F. CROSS‑REFERENCE
- Provides validation for the semi‑empirical terms introduced in Image 13; agrees with ID families in Image 06 at similar biases.

### Image 15
A. CIRCUIT TOPOLOGY
- 2T floating‑body cell; focus on dynamic VD ramps.

B. TEST CONFIGURATION
- Transient VD ramps (triangle/linear) at fixed VG1 = 0.3 V, with VG2 swept.
- Overlaid simulations (dashed) vs measurements (squares).

C. NUMBERED/NON‑STANDARD ELEMENTS
- Dynamic behavior dominated by body charging; parasitic capacitances implicitly important (no explicit elements drawn here).

D. PLOT STRUCTURE
- Log(ID) vs VD from ~0 to 3.5 V.
- Many colored families; arrow “Increasing VG2.”

E. TEXT/ANNOTATIONS
- Title: “Floating body 2T NS‑RAM cell under transient VD ramps.”
- Note: “Improved model parameter fitting...” comment.

F. CROSS‑REFERENCE
- Matches tendencies in Image 06; the transient curves illustrate the same knee shift with VG2.

### Image 16
A. CIRCUIT TOPOLOGY
- Process implementation slide for 130 nm triple‑well CMOS.
  - Cross‑section shows isolated P‑well (floating P‑body) inside deep N‑well over P‑substrate.
  - Schematic at bottom: 2T NS‑RAM core inside dashed box, identical to Images 05–06.
  - Layout example with deep‑Nwell enclosure; min cell areas annotated.

B. TEST CONFIGURATION
- Not a measurement; technology/area description.

C. NUMBERED/NON‑STANDARD ELEMENTS
- Isolation wells (deep N‑well, isolated P‑well).
- Emphasizes the parasitic well diode that naturally exists (expanded later in Image 21).

D. PLOT STRUCTURE
- None beyond figures.

E. TEXT/ANNOTATIONS
- “Triple‑well CMOS (130 nm).”
- “A single transistor floating synapse/neuron can be isolated in a 3×3 µm² area.”

F. CROSS‑REFERENCE
- Confirms the device stack required for the floating body in Images 17–18 and the diode in Image 21.

### Image 17
A. CIRCUIT TOPOLOGY
- 1T NS‑RAM (thick‑oxide) in deep N‑well; floating P‑body.
  - Terminals: gate VG, drain VD (<3.5 V), source VS = 0 V, body floating.
  - Deep N‑well bias VNwell > 2.5 V; optional VNEG = 0 V.

B. TEST CONFIGURATION
- DC ID–VD sweeps for VG = 0.3→0.8 V at two VNwell values (3 V and 5 V).
- Notes about a −1 V pre‑write pulse widening dynamic range (not shown here).

C. NUMBERED/NON‑STANDARD ELEMENTS
- Parasitic N‑well diode is implied (see left cartoon).
- “Parasitic Nwell device” also indicated.

D. PLOT STRUCTURE
- Two log(ID) vs VD families (VNwell = 3 V and 5 V); knee shifts with both VG and VNwell.

E. TEXT/ANNOTATIONS
- Area “8 µm² per fully‑contacted 1T neuron (180 nm CMOS).”

F. CROSS‑REFERENCE
- Provides the single‑transistor limit case; compared with 2T cell in Image 18.

### Image 18
A. CIRCUIT TOPOLOGY
- 2T NS‑RAM spiking neuron (thick‑oxide).
  - Two NFETs sharing floating body VB (M1 main, M2 leak).
  - M1: VG1 < 0.8 V, VD < 3.5 V, VS = 0 V.
  - M2: VG2 < 0.5 V or floating; source at 0 V.
  - VNwell > 2 V; VNEG = 0 V.

B. TEST CONFIGURATION
- Left plot: VG2 floating, VNwell = 2 V; sweep VG1 = 0.1→0.8 V.
- Right plot: fixed VG1 = 0.2/0.4/0.6 V; sweep VG2 = −0.2→0.5 V.

C. NUMBERED/NON‑STANDARD ELEMENTS
- Thick‑oxide devices.
- Body “self‑relaxation” with no pre‑pulse.

D. PLOT STRUCTURE
- ID–VD log plots showing classic S‑knee and post‑knee slope modulation.

E. TEXT/ANNOTATIONS
- “Second transistor’s leakage limits body fully‑floating behaviour (negative VG2 slows down the leaky behaviour).”
- “Area 17 µm².”

F. CROSS‑REFERENCE
- This is the same topology used for parameterization in Images 06–07.

### Image 19
A. CIRCUIT TOPOLOGY
- Behavioral LIF model mapped to the NS‑RAM soma.
  - Small schematic bottom‑right: Cint integrates, VG1/VG2 bias the NS‑RAM block that generates Vspike; Vmem is the integrated node.

B. TEST CONFIGURATION
- Brian2 simulations with parameters chosen to emulate the silicon cell; overlaid “Spice Vmem” (dashed) indicates correspondence.

C. NUMBERED/NON‑STANDARD ELEMENTS
- None in hardware; this is a software neuron using NS‑RAM‑like parameters.

D. PLOT STRUCTURE
- v(t) traces, thresholds, and membrane potential vs time.

E. TEXT/ANNOTATIONS
- Parameters printed: VG1 = 550 mV, VG2 = 500 mV, Cint = 170 fF, TAU, REFRACTORY, etc.

F. CROSS‑REFERENCE
- Node naming and role of VG1/VG2 match Images 10–12 (hardware soma).

### Image 20
A. CIRCUIT TOPOLOGY
- Not a circuit; system‑level inference results (confusion matrices) for SNNs using LIF vs Poisson inputs.

B. TEST CONFIGURATION
- Inference accuracy comparison; no electrical setup.

C. NUMBERED/NON‑STANDARD ELEMENTS
- None.

D. PLOT STRUCTURE
- Two confusion matrices (percentages).

E. TEXT/ANNOTATIONS
- “Run a parametric analysis covering: threshold voltages, firing time constants, excitatory input ranges.”

F. CROSS‑REFERENCE
- Motivates the operating ranges/biasing established by Images 10–12.

### Image 21
A. CIRCUIT TOPOLOGY
- Detailed 2T NS‑RAM schematic emphasizing parasitics.
  - M1: gate = VG1, drain = VD (ramped), source = 0 V, body = VB (floating).
  - M2: gate = VG2, drain = VB, source = VNEG (0 V).
  - Explicit parasitic N‑well diode from VNwell to VB.
  - Parasitic capacitances (gate overlap, well capacitances) implicated.

B. TEST CONFIGURATION
- Transient VD ramps with specified timing:
  - Example annotated: Vset = 2.05–2.2 V, t_set = 0–1 µs, trise/tfall = 200 µs; VG1 = 0.45 V, VG2 = 0.30 V.
  - Comparisons with/without the N‑well diode; and different ramp rates (10 µs, 100 µs, 1 ms).

C. NUMBERED/NON‑STANDARD ELEMENTS
- Parasitic N‑well diode (explicit).
- Voltage‑dependent leakage and capacitance at the well/body junction.

D. PLOT STRUCTURE
- Left: ramped measurement, log(ID) vs VD with markers for “Sfiring” and “Srelax.”
- Top‑right: with vs without Nwell diode—diode accelerates/changes knee.
- Bottom‑right: simulations for different trise; faster ramp shifts apparent knee and slope.

E. TEXT/ANNOTATIONS
- “Parasitic capacitances play a crucial role on the effective time constant.”
- “SR and firing time experiments ... are critical for fitting this dependence.”

F. CROSS‑REFERENCE
- Explains the dynamic spreads seen in Image 08 and the transient curves in Image 15.
- Confirms the presence and importance of the well diode foreshadowed in Image 16/17.

### Image 22 (22_sebas_2026_05_02_image-2.png — “three-branch parameters” source)
A. CIRCUIT TOPOLOGY
- Same 2T NS‑RAM cell as Images 05–06; this plot collects fitted parameter dependences for the three branches of the semi‑empirical model:
  - M1 impact‑ionization branch parameters: BETA0(VG2; VG1) and ETAB(VG2; VG1).
  - M1 channel‑related coefficient: K1(VG1).
  - M2 subthreshold (leak) branch parameter: NFACTOR(VG2; VG1).

B. TEST CONFIGURATION
- DC extractions from families of ID–VD sweeps at three VG1 values (0.20, 0.40, 0.60 V) with VG2 stepped as in Image 06.

C. NUMBERED/NON‑STANDARD ELEMENTS
- None on the plots; but this figure encodes the three‑branch structure:
  - Branch 1: impact‑ionization (BETA0, ETAB for M1).
  - Branch 2: channel pre‑avalanche shaping (K1 for M1).
  - Branch 3: leakage through M2 (NFACTOR for M2).

D. PLOT STRUCTURE
- Four panels (2×2 grid):
  - Top‑left: BETA0 (for M1) vs VG2. Three overlaid dashed branches:
    - Black (VG1 = 0.60 V) ~20, nearly flat vs VG2.
    - Blue (VG1 = 0.40 V) ~19, nearly flat.
    - Red (VG1 = 0.20 V) increases with VG2 (~10.8→14).
  - Top‑right: ETAB (for M1) vs VG2. Three branches:
    - Black (VG1 = 0.60 V) ~2.45 then drops above ~0.4–0.5 V.
    - Blue (VG1 = 0.40 V) ~1.9→1.6, decreasing with VG2.
    - Red (VG1 = 0.20 V) ~0.8→1.1, increasing with VG2.
  - Bottom‑left: K1 (for M1) vs VG1 only: ~0.56 at 0.2 V, ~0.54 at 0.4 V, ~0.415 at 0.6 V. Labeled “For all VG2.”
  - Bottom‑right: NFACTOR (for M2) vs VG2. Three branches:
    - Red (VG1 = 0.20 V): ≈12→6 as VG2 goes −0.2→0.1 V.
    - Blue (VG1 = 0.40 V): ≈5.5→2.7 as VG2 goes 0.0→0.3 V.
    - Black (VG1 = 0.60 V): ≈6→1.5 as VG2 goes 0.0→0.5 V.

E. TEXT/ANNOTATIONS
- Each panel explicitly states “for M1” or “for M2” and the independent variable (VG1 or VG2).
- Color‑coded VG1 labels (0.20/0.40/0.60 V), same as Images 01–04.

F. CROSS‑REFERENCE
- This is the master summary from which Images 01–04 were exported.
- Consistent with the semi‑empirical formulation in Image 13 and the fit quality in Images 14–15.

--------------------------------
CROSS‑IMAGE CONSISTENCY MAP

- Same device/experiment sets:
  - 2T NS‑RAM DC/Transient characterization: Images 05–06, 08–09, 13–15, 18, 21 all describe the same 2T floating‑body cell with terminals VG1 (M1 gate), VG2 (M2 gate), VD (M1 drain), VB (floating body, tied to M2 drain), VS/VNEG = 0 V, VNwell bias, and the parasitic N‑well diode to VB.
  - Parameter extraction for the above: Images 01–04 and 07 (and 22) are the parameter vs bias summaries used in the three‑branch SPICE model.
  - System/soma use of the same cell: Images 10–12, 19–20 use that 2T cell as a spiking element in an integrator/LIF and larger SNN blocks.
  - Process/physical realization: Images 16–18 show cross‑sections and layouts that correspond to the very same 1T/2T floating‑body structures.

- Agreements across slides:
  - Node naming is stable: VG1 = M1 gate (leak), VG2 = M2 gate (integration/leak), VD is the ramped node (either a swept supply in device characterization or the membrane node Vmem in soma circuits), VB is floating body and also the spiking node Vspike (Images 10, 21).
  - The presence of a parasitic N‑well diode from VNwell to VB is explicitly shown (Image 21) and implied by cross‑sections (Images 16–18). Its effect on dynamics matches ramp‑rate sensitivity (Images 08, 15, 21).
  - Three‑branch parameterization (BETA0, ETAB, K1 for M1; NFACTOR for M2) consistently explains how VG1 and VG2 move the ID–VD knee and slopes (Images 06–07, 13–15, 22).

- Apparent discrepancies:
  - Image 05 uses VG2 up to 1.4 V, while later data (Images 06–07, 18, 22) cap VG2 near 0.5 V or below. This likely reflects an earlier/high‑oxide‑bias experiment vs the later thick‑oxide limit. The qualitative behavior is consistent.
  - Image 14’s bulk‑current fitting uses VB = 0 V (not floating); it’s a modelling step to isolate the Iion law, not a contradiction to the floating‑body operation elsewhere.

- Load‑bearing topology features (appear in ≥2 images):
  - 2T cell with VB floating and M2 drain tied to VB (05, 06, 09, 15, 18, 21).
  - Parasitic N‑well diode to VB (16–18 implicitly; 21 explicitly).
  - Use of VD ramp to excite impact ionization (05–06, 08–09, 14–15, 21).
  - Mapping the VD ramp to Vmem in system cells (10–12, 19).
  - Parameter vs bias maps for three VG1 branches (01–04, 07, 22).

- One‑off or recent revisions:
  - The four‑panel parameter page (Image 22 = new 2026‑05‑02) is the most recent and comprehensive parameter summary; it supersedes standalone panels (01–04).
  - Image 05’s very high VG2 case appears to be an older operating point not used in the later system proposals.

--------------------------------
SCHEMATIC/TOPOLOGY INSIGHTS THAT A PURE BSIM4 + GP 3‑BRANCH I–V MODEL WOULD MISS

1) Floating body as an observable node (Vspike) with explicit coupling to a tunable bleed path (M2) and to a ramped drain (which, in circuits, is actually the membrane node Vmem). See Image 10 (Vmem→M1 drain, VB→Vspike) and Image 21. A compact 3‑branch I–V lacking the explicit VB node cannot reproduce the self‑reset and spike amplitude behavior.

2) Parasitic N‑well diode from VNwell to VB adds a voltage‑dependent leakage and capacitance that alters both the firing knee and its ramp‑rate dependence (Image 21). This diode is not part of standard MOS models and must be added as a separate diode+cap element between VB and VNwell.

3) Deliberate body–gate coupling capacitance (VB–VG2 MOS capacitor) modulates the effective time constant and spike shape (Image 05 schematic). This capacitive injection is missing in simple transport models but is critical for matching transient waveforms (Images 08 and 21).

4) The leak transistor M2’s subthreshold slope factor NFACTOR depends on both VG2 and VG1 (Images 01 and 22). Cross‑device coupling via VB means M2’s “n” cannot be treated as a constant; it strongly shapes integration/leak dynamics and the knee position during a VD ramp.

5) Mapping of the “drain ramp” to a real circuit node (Vmem) introduces extra loading and capacitance that materially changes the timing (Image 10: “The loading circuitry at Vmem contributes as integrator capacitance”). Device‑only models miss this unless Vmem loading is explicitly included.

6) VNwell bias and device oxide thickness domains (thick vs thin) set allowable VD and VG2 ranges (Images 16–18). These process‑level constraints determine safe operating points and explain differences between older and newer datasets.

7) Inhibitory/excitatory front‑ends drive the soma with starved‑inverter pulses at ~1 V amplitude (Images 11–12). The soma must be modeled with its input drivers to capture firing thresholds and energy; a stand‑alone transistor model omits this pulse‑shaping path and its dynamic impedance.

These details, pulled from the schematics and cross‑sections, are required to reproduce both the static knees and the dynamic (ramp‑rate‑dependent) spiking behavior of the NS‑RAM cell and its soma wrappers.
