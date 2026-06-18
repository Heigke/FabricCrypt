# grok response (grok-4-latest) — 181s

## Image 01_NFACTOR_M2_vs_VG2
- [NUMERIC] Y-axis ticks 0,2,4,6,8,10,12 (linear); X-axis -0.3 to 0.6 V, ticks every 0.1 V; data points at approx. VG2 = -0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0,0.05,...0.6 V.
- [CONTRADICTION-OUR-EXTRACTION] Red branch peaks at ~12 (not 12.2) at VG2≈-0.3 V (not -0.2 V); black branch is lowest, likely VG1=0.60 V based on compilation in Image 07.

## Image 02_K1_M1_vs_VG1
- [NUMERIC] Y-axis 0.4 to 0.58, ticks every 0.02; X-axis 0.1 to 0.7 V, ticks every 0.1 V; curve has knee at VG1≈0.35 V, dropping from 0.56 to 0.42.

## Image 03_ETAB_M1_vs_VG2
- [NUMERIC] Y-axis 0.6 to 2.6, ticks every 0.2; black curve (highest) declines from 2.5 to 2.1; blue from 2.0 to 1.2; red from 0.7 to 1.1.

## Image 04_BETA0_M1_vs_VG2
- [NUMERIC] Y-axis 10 to 21, ticks every 1; red rises from ~10.5 at -0.3 V to ~14.5 at 0.6 V; blue flat ~18.5-19.5; black flat ~20.
- [CONTRADICTION-OUR-EXTRACTION] Red regime not 10.75→14 (visually 10.5→14.5); blue not exactly 19 (18.5-19.5); black not exactly 20 (~20).

## Image 05_IV_curves_schematic
- [NUMERIC] Middle plot: VG2=1.4 V, VG1=0.25 to 0.45 V (implied step ~0.05 V), x-axis 0-3.5 V; right plot: VG2=0.1 V (lowest) to 3.1 V (highest, arrow labeled), x-axis 0-3.5 V; y log 10^-9 to 10^-4 A for both.
- [ARCHITECTURAL] Schematic shows single NMOS with front gate VG1, back gate VG2, drain VD, source/bulk grounded.
- [CONTRADICTION-OUR-EXTRACTION] No p-body diode visible (contradicts slide 21 mention); not matching user-described slide 06 4-panel with VG1 including -0.2 V.

## Image 06_IV_family_measurements
- [NUMERIC] Left: VG1=0.6 V, VG2=0 to 0.5 V step 0.05 V, x 0-2 V; middle: VG1=0.4 V, VG2=0 to 0.3 V step 0.05 V; right: VG1=0.2 V, VG2=-0.2 to 0.1 V step 0.05 V; all y log 10^-12 to 10^-4 A.
- [CONTRADICTION-OUR-EXTRACTION] VG1 values are 0.2,0.4,0.6 V (no -0.2 V); VG2 sweeps vary per panel, not uniform 0→0.5 V; 3 panels + schematic, not 4.

## Image 07_parameter_compilation
- [NUMERIC] Explicit labels: BETA0 curves VG1=0.20 V (red rising), 0.40 V (blue flat ~19), 0.60 V (black flat ~20); ETAB VG1=0.20 V (red), 0.40 V (blue), 0.60 V (black); NFACTOR VG1=0.20 V (red high), 0.40 V (blue), 0.60 V (black low); K1 single for all VG2.
- [CONTRADICTION-OUR-EXTRACTION] BETA0 red starts ~10 at VG2=-0.3 V (not 10.75); no "3 flat-then-rising regimes" – red rising, others flat.

## Image 08_dynamic_response
- [NUMERIC] Top: current 0-6 mA, time 0-2.6 μs, voltage overlay 0-2 V; bottom left/middle: y log 10^-9 to 10^5 A (note 10^5, high current), x 0-2 V; data noisy with red points.
- [EXPERIMENTAL] Implies pulse-train recipes with alternating spikes (red/blue colors).

## Image 09_thick_thin_IV
- [MISSED] Comparison of thick oxide measurements (gray curves) vs thin oxide simulations (colored: cyan, gray, orange).
- [NUMERIC] Labels: VG1=0.60 V VG2=0.35 V (top curve), VG1=0.40 V VG2=0.25 V (middle), VG1=0.20 V VG2=0.0 V (bottom); y log 10^-12 to 10^-4 A, x 0-2 V.

## Image 10_simple_NS_RAM_cell
- [MISSED] Energy consumption 0.2 pJ/spike (steady state firing, 100 nA excitatory); area 111 μm² (with capacitor); spiking freq configurable within 10x.
- [NUMERIC] C_int=102 fF; plots with VG1=V_leak {0.35,0.375,0.4,0.425} V, VG2=V_integ {0.475,0.475,0.45,0.45} V; time ms scale, voltage 0-0.6 V.
- [ARCHITECTURAL] Self-reset via floating body generating spikes 0.5-0.7 V; V_mem loading circuitry as integrator capacitance.
- [CONTRADICTION-OUR-EXTRACTION] Area 111 μm² (not matching any listed variant); firing range 10x (not >10^4x).

## Image 11_NS_RAM_blocks_input_neurons
- [MISSED] Energy: I_exc=0.5 nA constant, 21 fJ/spike total (~0.7 fJ spike gen, ~20 fJ integration); area ~60 μm²; firing freq up to 360 kHz at 5 nA.
- [NUMERIC] Spike currents labeled 500 pA (60 kHz), 800 pA (120 kHz), 1.26 nA (180 kHz), 1.72 nA (260 kHz), 3.15 nA (340 kHz), 5 nA (360 kHz); energy (fJ) vs time (μs) up to 1000 fJ.
- [ARCHITECTURAL] Soma without diode; V_G2=V_integ=0.275 V, V_G1=V_leak=0 V; reset path V_rst.
- [CONTRADICTION-OUR-EXTRACTION] Linear range VG2=0.275 V (not [2.5,3.0] V); area 60 μm² new variant.

## Image 12_NS_RAM_firing_linear
- [ARCHITECTURAL] Bias-generation: Vw mirror bank for weights; VPOS5 (5 V supply implied); tau capacitors for timing; thick oxide for high V_DD, thin for logic; excitatory/inhibitory via starved inverters with 1 V pulses; soma interfaces directly to linear synapse by spiking into output.
- [CONTRADICTION-OUR-EXTRACTION] Linear range for V_w.x [2.5,3] V (not VG2 [2.5,3.0] V).

## Image 13_bulk_current_model
- [CONTRADICTION-OUR-EXTRACTION] Equation: Iion = Iexp + Ipow; Ipow = d (V_D + f)^e if V_D > -f else 0; Iexp = a * exp( b (V_D + c) ); params a,b,d,e,f = PWL(V_G), c=const (natural exp, not 10^; power law without linear +b term; f not j; no I_pwl = a V_d^c + b).
- [MISSED] Small plots show PWL fits for a (declining), b (rising), d (declining), e (U-shape), f (rising) vs VG ~ -0.5 to 0.5 V.
- [EXPERIMENTAL] Fitted in MATLAB then to SPICE; power law only active for V_D >0.

## Image 14_measurements_vs_SPICE
- [NUMERIC] VG2=0 V fixed; VG1 0.00 to 0.55 V step 0.05 V (11 curves); y log 10^-12 to 10^-6 A (bulk), 10^-12 to 10^4 A (total); x 0-3.5 V.
- [ARCHITECTURAL] Body voltage VB dependence not modeled yet; using RB=1 MΩ measurements for versioning.
- [ASK-SEBAS] Raw CSVs for bulk/total current measurements at VG2=0 V, including VB effects.

## Image 15_floating_body_2T
- [NUMERIC] VG1=0.3 V fixed; increasing VG2 (rainbow, ~20 curves); y log 10^-9 to 10^-4 A, x 0-3.5 V.
- [EXPERIMENTAL] Transient VD ramps; suggests ramp-rate dependence; improved fitting via ML search.
- [ASK-SEBAS] Pulse-mode protocols for transient VD ramps, including ramp rates.

## Image 16_NS_RAM_130nm
- [MISSED] ~1000x improvement to state-of-art neuron cores; single transistor synapse/neuron in 3x3 μm²; 2T cell in 5.5x6 μm² Deep N-Well.
- [CONTRADICTION-OUR-EXTRACTION] Area ~33 μm² (not 31.8 μm²).
- [ARCHITECTURAL] Triple-well or P-channel floating well; large body for high tuning flexibility.

## Image 17_deep_Nwell_1T
- [MISSED] Retention ~100 s with -1 V pre-write pulse; nW operation, 100% yield; firing window 7x to 10^4x.
- [NUMERIC] VNwell ≥2.5 V, VD <3.5 V, VG <0.8 V, VNEG=0 V, VS=0 V; plots for VNwell=3 V and 5 V, VG=0.3 to 0.8 V step ~0.1 V; y log 10^-13 to 10^-4 A, x 0-3.5 V.
- [EXPERIMENTAL] -1 V pre-write pulse widens dynamic range.
- [CONTRADICTION-OUR-EXTRACTION] Firing range 7x to 10^4x (not just >10^4x).
- [ASK-SEBAS] Retention protocols with -1 V pre-write pulse details.

## Image 18_2T_spiking_neuron
- [MISSED] Firing range >10^3x; operating voltages below nominal; efficient as thresholding element.
- [NUMERIC] Area 17 μm²; VG1=0.1 to 0.8 V, VG2=-0.2 to 0.5 V or floating; y log 10^-13 to 10^-3 A, x 0-2 V.
- [EXPERIMENTAL] Tested at slow sweep rates 0.2 V/s for high resolution.
- [ARCHITECTURAL] No pre-pulse, self-relaxation only; second transistor leakage limits full floating; negative VG2 slows leaky behavior.
- [CONTRADICTION-OUR-EXTRACTION] Firing range >10^3x (not >10^4x); VG ≈ -2 V not mentioned (image shows -0.2 V).

## Image 19_NS_RAM_LIF_Brian2
- [MISSED] Specific params: V_EXC_REST=2.1e3, THRESH_VAL=3.1e3, TAU_MEM=0.045, REFRACTORY=0.0079, TIMESCALE=1e5, EXCIT_VALUE=2.6 (left); right THRESH_VAL=3.14, TAU_MEM=0.0138, REFRACTORY=0.00599, TIMESCALE=1e5, EXCIT_VALUE=2.8; units mV, ms, scaled.
- [NUMERIC] VG1=550 mV, VG2=500 mV, C_int=170 fF (left); VG1=550 mV, VG2=500 mV, C_int=170 fF (right); spice Vmem 400 nA to 1 μA.
- [ASK-SEBAS] Brian2 script for LIF input neurons.

## Image 20_SNN_Brian2
- [MISSED] Confusion matrices: Poisson reference 89% acc (diagonal 98%,95%,88%,88%,86%,81%,97%,85%,89%,86%); LIF 72% acc (diagonal 80%,93%,67%,71%,67%,13%,83%,35%,89%,29%).
- [CONTRADICTION-OUR-EXTRACTION] Poisson 89% (not 85%); LIF 72%; no mention of 37% vs 22% differential pairs.
- [ASK-SEBAS] Raw results for parametric analysis on thresholds, firing constants, excitatory ranges.

## Image 21_dynamic_ramp_rate
- [NUMERIC] V_set=2.05 V, t_set=1 μs, t_rise=200 μs, t_fall=200 μs, VG1=0.45 V, VG2=0.3 V (left); sims with t_rise 1 ms, 10 μs, 100 μs (right); V_set=2.2 V in one sim.
- [ARCHITECTURAL] Schematic includes parasitic N-well diode (voltage-dependent leakage/capacitance); VNEG=0 V, VS=0 V; nodes: V_Nwell tied to VD? (dashed), floating P-body.
- [EXPERIMENTAL] Ramp rate dependence; SR/firing time experiments 5-7 (not 3-7); with/without Nwell diode.
- [CONTRADICTION-OUR-EXTRACTION] Experiments 5-7 (not 3-7); schematic has N-well diode (user said p-body diode).
- [ASK-SEBAS] Full "experiment list" including 5-7 SR/firing time protocols; transient traces for ramp rates.
