# gemini response (gemini-2.5-pro) — 166s

Of course. Here is a detailed numerical specification extraction from the 21 provided image artifacts, following your requested structure.

### Per-Slide Numerical Data Extraction

---
#### **Image 1: NFACTOR vs VG2**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| VG2 Range | -0.3 to 0.6 | V | 1 | measured-axis |
| NFACTOR Range | 0 to 12.5 | (unitless) | 1 | measured-axis |
| NFACTOR (Red, VG2=-0.18V) | ~12.2 | (unitless) | 1 | inferred-visual |
| NFACTOR (Red, VG2=0.1V) | ~6.2 | (unitless) | 1 | inferred-visual |
| NFACTOR (Black, VG2=0V) | ~6.0 | (unitless) | 1 | inferred-visual |
| NFACTOR (Black, VG2=0.5V) | ~1.2 | (unitless) | 1 | inferred-visual |
| NFACTOR (Blue, VG2=0.1V) | ~5.2 | (unitless) | 1 | inferred-visual |
| NFACTOR (Blue, VG2=0.3V) | ~2.8 | (unitless) | 1 | inferred-visual |
| Number of data series | 3 | - | 1 | inferred-visual |
| Parameter Name | NFACTOR (for M2) | - | 1 | label |

---
#### **Image 2: K1 vs VG1**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| VG1 Range | 0.1 to 0.7 | V | 2 | measured-axis |
| K1 Range | 0.4 to 0.58 | (unitless) | 2 | measured-axis |
| K1 (at VG1=0.2V) | ~0.56 | (unitless) | 2 | inferred-visual |
| K1 (at VG1=0.4V) | ~0.54 | (unitless) | 2 | inferred-visual |
| K1 (at VG1=0.6V) | ~0.42 | (unitless) | 2 | inferred-visual |
| Number of data points | 3 | - | 2 | inferred-visual |
| Parameter Name | K1 (for M1) | - | 2 | label |

---
#### **Image 3: ETAB vs VG2**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| VG2 Range | -0.3 to 0.6 | V | 3 | measured-axis |
| ETAB Range | 0.6 to 2.6 | (unitless) | 3 | measured-axis |
| ETAB (Red, VG2=-0.2V) | ~0.8 | (unitless) | 3 | inferred-visual |
| ETAB (Red, VG2=0.1V) | ~1.1 | (unitless) | 3 | inferred-visual |
| ETAB (Blue, VG2=0V) | ~1.9 | (unitless) | 3 | inferred-visual |
| ETAB (Blue, VG2=0.3V) | ~1.6 | (unitless) | 3 | inferred-visual |
| ETAB (Black, VG2=0 to 0.35V) | ~2.5 | (unitless) | 3 | inferred-visual |
| ETAB (Black, VG2=0.5V) | ~2.1 | (unitless) | 3 | inferred-visual |
| Parameter Name | ETAB (for M1) | - | 3 | label |

---
#### **Image 4: BETA0 vs VG2**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| VG2 Range | -0.3 to 0.6 | V | 4 | measured-axis |
| BETA0 Range | 10 to 21 | (unitless) | 4 | measured-axis |
| BETA0 (Red, VG2=-0.2V) | ~10.8 | (unitless) | 4 | inferred-visual |
| BETA0 (Red, VG2=0.1V) | ~14.0 | (unitless) | 4 | inferred-visual |
| BETA0 (Blue, VG2=0 to 0.3V) | ~19.0 | (unitless) | 4 | inferred-visual |
| BETA0 (Black, VG2=0 to 0.5V) | ~20.0 | (unitless) | 4 | inferred-visual |
| Parameter Name | BETA0 (for M1) | - | 4 | label |

---
#### **Image 5: Circuit Diagram and I-V Curves**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| VD Range | 0 to 3.5 | V | 5 | measured-axis |
| ID Range | 10⁻⁹ to 10⁻⁴ | A | 5 | measured-axis |
| VG2 (middle plot) | 1.4 | V | 5 | label |
| VG1 sweep (middle plot) | 0.25 to 0.45 | V | 5 | label |
| VG2 (right plot) | 0.1 | V | 5 | label |
| VG1 sweep (right plot) | 0.25 to 0.45 | V | 5 | label |
| Snapback Voltage (middle plot) | ~2.5 | V | 5 | inferred-visual |
| Snapback Voltage (right plot) | ~1.5 | V | 5 | inferred-visual |
| ON/OFF Ratio (approx) | > 10⁴ | - | 5 | inferred-visual |
| Number of transistors | 2 | - | 5 | inferred-visual |

---
#### **Image 6: Measurements vs Simulations I-V**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| Voltage Range | 0.0 to 2.0 | V | 6 | measured-axis |
| Current Range | 10⁻¹² to 10⁻⁴ | A | 6 | measured-axis |
| VG1 (left plot) | 0.6 | V | 6 | label |
| VG2 sweep (left plot) | 0 to 0.5 | V | 6 | label |
| VG2 step (left plot) | 0.05 | V | 6 | label |
| VG1 (middle plot) | 0.4 | V | 6 | label |
| VG2 sweep (middle plot) | 0 to 0.3 | V | 6 | label |
| VG2 step (middle plot) | 0.05 | V | 6 | label |
| VG1 (right plot) | 0.2 | V | 6 | label |
| VG2 sweep (right plot) | -0.2 to 0.1 | V | 6 | label |
| VG2 step (right plot) | 0.05 | V | 6 | label |
| Hysteresis window width (VG1=0.6V) | ~0.2 | V | 6 | inferred-visual |

---
#### **Image 7: Composite of 4 Parameter Plots**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| VG1 for BETA0/ETAB/NFACTOR (Red) | 0.20 | V | 7 | label |
| VG1 for BETA0/ETAB/NFACTOR (Blue) | 0.40 | V | 7 | label |
| VG1 for BETA0/ETAB/NFACTOR (Black) | 0.60 | V | 7 | label |
| K1 plot condition | For all VG2 | - | 7 | label |
| BETA0 (VG1=0.6V) | ~20 | (unitless) | 7 | inferred-visual |
| ETAB (VG1=0.6V, VG2=0.3V) | ~2.5 | (unitless) | 7 | inferred-visual |
| K1 (VG1=0.2V) | ~0.56 | (unitless) | 7 | inferred-visual |
| NFACTOR (VG1=0.2V, VG2=-0.1V) | ~11 | (unitless) | 7 | inferred-visual |
| NFACTOR (VG1=0.6V, VG2=0.5V) | ~1.2 | (unitless) | 7 | inferred-visual |

---
#### **Image 8: Transient Pulse and I-V Density**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| Time axis range | 0.0 to 2.6 | µs | 8 | measured-axis |
| Voltage pulse amplitude | 2.0 | V | 8 | measured-axis |
| Current pulse amplitude | ~5.5 | mA | 8 | inferred-visual |
| Pulse period | ~0.325 | µs | 8 | inferred-visual |
| Pulse frequency | ~3.08 | MHz | 8 | inferred-visual |
| Voltage pulse width (FWHM) | ~0.1 | µs | 8 | inferred-visual |
| I-V plot voltage range | 0.0 to 2.0 | V | 8 | measured-axis |
| I-V plot current range | 10⁻⁹ to 10⁻⁵ | A | 8 | measured-axis |
| Number of pulses shown | 8 | - | 8 | inferred-visual |

---
#### **Image 9: Measurements vs Simulations (Thick/Thin)**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| Voltage Range | 0.0 to 2.0 | V | 9 | measured-axis |
| Current Range | 10⁻¹² to 10⁻⁴ | A | 9 | measured-axis |
| Curve 1 VG1 | 0.2 | V | 9 | label |
| Curve 1 VG2 | 0.0 | V | 9 | label |
| Curve 2 VG1 | 0.40 | V | 9 | label |
| Curve 2 VG2 | 0.25 | V | 9 | label |
| Curve 3 VG1 | 0.60 | V | 9 | label |
| Curve 3 VG2 | 0.35 | V | 9 | label |
| Measurement line style | Thick | - | 9 | label |
| Simulation line style | Thin | - | 9 | label |
| Turn-on voltage (VG1=0.2V) | ~1.8 | V | 9 | inferred-visual |
| Turn-on voltage (VG1=0.6V) | ~1.3 | V | 9 | inferred-visual |

---
#### **Image 10: Simple NS-RAM cell (self-reset)**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| C_ext | 102 | fF | 10 | label |
| Energy consumption | As low as 0.2 | pJ / spike | 10 | label |
| Steady state firing current | 100 | nA | 10 | label |
| Spiking frequency configurability | 10x | - | 10 | label |
| Area | 111 | µm² | 10 | label |
| Floating body spike range | 0.5 to 0.7 | V | 10 | inferred-text |
| V_leak / V_integ (top trace) | 0.35 | V | 10 | label |
| V_G2 (top trace) | 0.475 | V | 10 | label |
| V_leak / V_integ (2nd trace) | 0.375 | V | 10 | label |
| V_leak / V_integ (3rd trace) | 0.4 | V | 10 | label |
| V_leak / V_integ (bottom trace) | 0.425 | V | 10 | label |
| Time axis range (waveforms) | 0 to 0.15 | ms | 10 | measured-axis |

---
#### **Image 11: NS-RAM blocks for input neurons**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| V_DD | 1.5 | V | 11 | label |
| VG2 (V_integ) | 0.275 | V | 11 | label |
| VG1 (V_leak) | 0 | V | 11 | label |
| I_excit (constant) | 0.5 | nA | 11 | label |
| Energy per spike | ~21 | fJ | 11 | label |
| Spike generation energy | ~0.7 | fJ | 11 | label |
| Integration time energy | ~20 | fJ | 11 | label |
| Area | ~60 | µm² | 11 | label |
| Firing frequency (bottom trace) | 360 | kHz | 11 | label |
| Input current (bottom trace) | 5 | nA | 11 | label |
| Firing frequency (top trace) | 60 | kHz | 11 | label |
| Input current (top trace) | 500 | pA | 11 | label |
| Time axis range (energy plot) | 0 to 50 | µs | 11 | measured-axis |

---
#### **Image 12: NSRAM firing with linear inputs**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| Linear range for Vw_i | 2.5 to 3 | V | 12 | inferred-text |
| Spike pulse amplitude | 1 | V | 12 | inferred-text |
| Number of transistors (Soma) | ~10 | - | 12 | inferred-visual |
| Oxide type 1 | Thick | - | 12 | label |
| Oxide type 2 | Thin | - | 12 | label |

---
#### **Image 13: Semi-empirical model fits**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| Model components | 2 (Exponential, Power law) | - | 13 | inferred-text |
| I_pow formula | d(VD + f)^e | - | 13 | fit-equation |
| I_exp formula | a * exp(b(VD+c)) | - | 13 | fit-equation |
| Parameter 'c' | constant | - | 13 | fit-equation |
| VG1 for example plot | 0.15 | V | 13 | label |
| Drain voltage range | 0 to 4.5 | V | 13 | measured-axis |
| Bulk current range | 10⁻¹³ to 10⁻⁵ | A | 13 | measured-axis |
| Knee voltage (Power law > Exp) | ~2.5 | V | 13 | inferred-visual |

---
#### **Image 14: Measurements vs. SPICE simulation (bulk currents)**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| Body voltage (VB) | 0 | V | 14 | inferred-text |
| Measurement series resistance (Rs) | 1 | MΩ | 14 | inferred-text |
| VG1 sweep range | 0.00 to 0.55 | V | 14 | label |
| Drain voltage range | 0 to 3.5 | V | 14 | measured-axis |
| Bulk current (IB) range | 10⁻¹³ to 10⁻⁵ | A | 14 | measured-axis |
| Total current (ID) range | 10⁻¹³ to 10⁻⁴ | A | 14 | measured-axis |
| Number of VG1 steps | 12 | - | 14 | inferred-visual |

---
#### **Image 15: Floating body 2T NS-RAM cell transient**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| Cell type | 2T NS-RAM | - | 15 | inferred-text |
| VG1 (fixed) | 0.3 | V | 15 | label |
| VG2 condition | Increasing | - | 15 | label |
| Drain voltage range | 0 to 3.5 | V | 15 | measured-axis |
| Total drain current range | 10⁻⁹ to 10⁻⁴ | A | 15 | measured-axis |
| Number of VG2 steps shown | ~15 | - | 15 | inferred-visual |
| Simulation line style | Dashed lines | - | 15 | label |
| Measurement line style | Squares | - | 15 | label |

---
#### **Image 16: NS-RAM implementation in standard CMOS**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| Technology | Standard triple-well CMOS | - | 16 | label |
| Technology node | 130 | nm | 16 | label |
| Minimum Deep N-Well size | 5.5x6 | µm² | 16 | label |
| Single transistor floating synapse/neuron area | 3x3 | µm² | 16 | label |
| Improvement over state-of-the-art | ~1000x | - | 16 | label |
| Number of transistors in neuron core | 2 | - | 16 | inferred-text |

---
#### **Image 17: Standard CMOS deep-Nwell NFET floating body 1T neuron**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| Technology node | 180 | nm | 17 | label |
| Cell area | 8 | µm² | 17 | label |
| V_Nwell | > 2.5 | V | 17 | label |
| V_D | < 3.5 | V | 17 | label |
| V_G | < 0.8 | V | 17 | label |
| V_S, V_NEG | 0 | V | 17 | label |
| Pre-pulse voltage | -1 | V | 17 | inferred-text |
| Retention time | ~100 | s | 17 | inferred-text |
| Firing window (range) | 7x to 10⁴x | - | 17 | label |
| Yield | 100 | % | 17 | label |
| VG sweep (plots) | 0.3 to 0.8 | V | 17 | label |
| V_Nwell (left plot) | 3 | V | 17 | label |
| V_Nwell (right plot) | 5 | V | 17 | label |

---
#### **Image 18: 2T NS-RAM spiking neuron cell (thick oxide)**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| Cell area | 17 | µm² | 18 | label |
| V_Nwell | > 2.5 | V | 18 | label |
| V_D | < 3.5 | V | 18 | label |
| V_G1 | < 0.8 | V | 18 | label |
| V_G2 | < 0.5 or floating | V | 18 | label |
| V_S, V_NEG | 0 | V | 18 | label |
| V_Nwell (left plot) | 2 | V | 18 | label |
| VG1 sweep (left plot) | 0.1 to 0.8 | V | 18 | label |
| VG2 sweep (right plot) | -0.2 to 0.5 | V | 18 | label |
| VG1 (right plot) | 0.4 | V | 18 | label |
| VG1 (middle plot) | 0.6 | V | 18 | label |
| Firing range trade-off | 10³x | - | 18 | label |
| Slow sweep rate | 0.2 | V/s | 18 | label |

---
#### **Image 19: NSRAM Simple LIF in Brian2**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| VG1 | 550 | mV | 19 | label |
| VG2 | 500 | mV | 19 | label |
| C_int | 170 | fF | 19 | label |
| Timescale slowdown factor | 10⁵ | - | 19 | label |
| **Set 1 (400nA input)** | | | | |
| V_EXC_REST | 2.1e3 | (arbitrary) | 19 | label |
| THRESH_VAL | 3.15e3 | (arbitrary) | 19 | label |
| TAU_MEM | 0.045 | (arbitrary) | 19 | label |
| REFRACTORY | 0.0079 | (arbitrary) | 19 | label |
| EXC_INPUT | 2.6 | (arbitrary) | 19 | label |
| **Set 2 (1uA input)** | | | | |
| THRESH_VAL | 3.4e3 | (arbitrary) | 19 | label |
| TAU_MEM | 0.0138 | (arbitrary) | 19 | label |
| REFRACTORY | 0.00599 | (arbitrary) | 19 | label |
| EXCIT_VALUE | 2.8 | (arbitrary) | 19 | label |

---
#### **Image 20: Simulating a more physically realizable SNN in Brian2**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| Poisson network accuracy | 89 | % | 20 | label |
| LIF network accuracy | 72 | % | 20 | label |
| Number of classes (digits) | 10 (0-9) | - | 20 | inferred-visual |
| LIF accuracy (class '1') | 93 | % | 20 | table-cell |
| LIF accuracy (class '9') | 62 | % | 20 | table-cell |
| Poisson accuracy (class '6') | 97 | % | 20 | table-cell |
| Poisson accuracy (class '5') | 81 | % | 20 | table-cell |

---
#### **Image 21: Dynamic response (ramp rate dependence)**

| Quantity | Value | Units | Slide ref | Source |
|---|---|---|---|---|
| **Static Ramp Condition** | | | | |
| V_set | 2.05 | V | 21 | label |
| t_set | 1 | µs | 21 | label |
| t_rise (static) | 200 | µs | 21 | label |
| t_fall (static) | 200 | µs | 21 | label |
| VG1 (static) | 0.45 | V | 21 | label |
| VG2 (static) | 0.3 | V | 21 | label |
| S_fire (firing slope) | ~500 | mV/decade | 21 | inferred-visual |
| S_relax (relaxing slope) | ~700 | mV/decade | 21 | inferred-visual |
| **Multi-Rate Conditions** | | | | |
| Number of distinct ramp rates | 3 | - | 21 | inferred-visual |
| t_rise (fastest) | 10 | µs | 21 | label |
| t_rise (medium) | 100 | µs | 21 | label |
| t_rise (slowest) | 1 | ms | 21 | label |
| Ramp rate (fastest, approx) | 2.5e5 (2.5V/10µs) | V/s | 21 | inferred-visual |
| Ramp rate (medium, approx) | 2.5e4 (2.5V/100µs) | V/s | 21 | inferred-visual |
| Ramp rate (slowest, approx) | 2.5e3 (2.5V/1ms) | V/s | 21 | inferred-visual |

---
### **Consolidated Section**

#### Top 10 Most Confident Quantities
These are values explicitly stated as text labels on the slides.

1.  **Technology Node:** 130 nm (Slide 16)
2.  **LIF Network Accuracy:** 72 % (Slide 20)
3.  **Poisson Network Accuracy:** 89 % (Slide 20)
4.  **1T Neuron Area (180nm):** 8 µm² (Slide 17)
5.  **Energy (Simple Cell):** As low as 0.2 pJ / spike (Slide 10)
6.  **Energy (Input Neuron):** ~21 fJ / spike (Slide 11)
7.  **Brian2 Input Capacitance (C_int):** 170 fF (Slide 19)
8.  **Brian2 VG1:** 550 mV (Slide 19)
9.  **Brian2 VG2:** 500 mV (Slide 19)
10. **Slow Sweep Rate for Test:** 0.2 V/s (Slide 18)

#### Top 5 Least Confident Quantities
These are values inferred from visual inspection of plots, often on logarithmic scales or involving slope estimation.

1.  **S_fire (Firing Slope):** ~500 mV/decade (Slide 21). This is a visual estimate of a slope on a log-linear plot and is highly subjective.
2.  **S_relax (Relaxing Slope):** ~700 mV/decade (Slide 21). Same reason as above.
3.  **Pulse Current Amplitude:** ~5.5 mA (Slide 8). Reading the peak of a fast transient from a graph without a clear marker is prone to error.
4.  **Knee Voltage (Power law > Exp):** ~2.5 V (Slide 13). Estimating the crossover point of two plotted functions is an approximation.
5.  **Pulse Frequency:** ~3.08 MHz (Slide 8). Inferred by counting peaks over a time interval, which can have off-by-one errors and assumes perfect periodicity.

---
### **Contradictory or Context-Dependent Quantities**

1.  **Technology Node:**
    *   **Slide 16:** States "**130 nm**" for the "NS-RAM implementation in standard triple-well CMOS".
    *   **Slide 17:** States "**180 nm CMOS**" for the "Standard CMOS deep-Nwell NFET floating body 1T neuron".
    *   **Flag:** This is a direct contradiction. It's likely two different devices or technology variants are being discussed across the slides, but this should be clarified.

2.  **Energy per Spike:**
    *   **Slide 10:** Claims "**As low as 0.2 pJ per spike**" for a "Simple NS-RAM cell with integration (self-reset)".
    *   **Slide 11:** Claims "**~21 fJ per spike**" for "NS-RAM blocks for input neurons (soma without diode)".
    *   **Flag:** These values differ by nearly 10x (200 fJ vs 21 fJ). This is not a direct contradiction but a strong **context-dependency**. The different circuit architectures ("self-reset" vs. "soma without diode") and operating conditions (100 nA vs 0.5 nA input current) lead to vastly different energy figures. The 21 fJ value also has a clear breakdown (0.7 fJ generation + 20 fJ integration).

3.  **Device Area:**
    *   **Slide 16:** Mentions a "**3x3 µm²**" single transistor area and a "**5.5x6 µm²**" Deep N-Well.
    *   **Slide 17:** Shows an "**8 µm²**" 1T neuron layout.
    *   **Slide 18:** Shows a "**17 µm²**" 2T neuron layout.
    *   **Flag:** These are not contradictory but are context-dependent, referring to different cell configurations (1T vs 2T) and layout rules (minimum vs. fully-contacted).

4.  **Brian2 Hyperparameters:**
    *   **Slide 19:** Shows two different sets of `THRESH_VAL`, `TAU_MEM`, and `REFRACTORY` values.
    *   **Flag:** This is context-dependent. The two sets correspond to different excitatory input currents (400 nA vs 1 uA), demonstrating the tunability of the neuron's response.
