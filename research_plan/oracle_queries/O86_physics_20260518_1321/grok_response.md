# grok response (grok-4-latest) — 15s

**Q1 — DC gap physics**

The 1.0–1.4 dec gap (33 biases) most likely stems from three BSIM4/model-stack deficiencies rather than pure fitting error.

1. **Missing DITS + RDSMOD=1 interaction with well-diode**. BSIM4 DITS (drain-induced threshold shift) and the external well-diode + Sebas pdiode create an unphysical subthreshold slope when V_b is swept; the pyport KCL does not enforce body-effect modulation of the diode forward current.  
   *Test (≤4 GPU-h)*: Fix V_b, sweep VG2 at two temperatures (25 °C / 85 °C) with RDSMOD=1 vs. 0; compare subthreshold swing and V_th roll-off against Mario Id-VG family. If swing error collapses >0.6 dec, DITS+diode is dominant.

2. **PCLM / VSAT under-modeling in saturation with Gummel-Poon NPN**. Early saturation current from the parasitic NPN (collector tied to drain) is not captured by BSIM4 PCLM; the external snapback subcircuit only triggers at high V_d. Result is ~0.4–0.6 dec excess current at VG1=0.4–0.6.  
   *Test*: Disable NPN, re-run DC family with PCLM=0.3/0.8 and VSAT=8e4/1.2e5; quantify Id error at V_d=1.2 V.

3. **Missing gate-drain overlap + fringing capacitance in transient calibration**. The 200 ns pulse data (t_rise=26 ns) forces the model to match Id_pk via C_gd overlap that BSIM4 CGDO/CGSL alone cannot supply once the well-diode is present. This inflates the apparent DC offset.  
   *Test*: Extract C_gd from 26 ns/76 ns edges at four VG2 biases; add explicit overlap capacitor and re-fit DC. Gap reduction >0.3 dec confirms the parasitic.

**Q2 — V3 DC-knee at VG1=0.4**

A sharp 1/V knee implies a voltage-dependent generation or tunneling current that turns on abruptly—none of the standard BSIM4 IIMOD + NPN combinations produce it.

1. **Band-to-band tunneling (BBT) at the drain–body junction** (not modeled by BSIM4 IIMOD=1/2). The well-diode + Sebas pdiode lack the Hurkx or local BBT term; at VG1=0.4 the surface potential places the junction in the BBT window.  
   *Falsify*: Add BSIM4 BBT (AGIDL/BGIDL) or explicit Hurkx diode; if knee rounds or shifts >50 mV the mechanism is confirmed.

2. **Impact-ionization-generated hole current feeding the parasitic NPN base** (M-factor + Gummel-Poon). The snapback subcircuit only triggers at high V_d; at low VG1 the M-factor produces a sharp current upturn that looks like 1/V.  
   *Falsify*: Sweep M-factor (or disable NPN) while monitoring base current; knee disappearance isolates this path.

3. **DITS + negative V_b feedback through the body diode**. DITS lowers V_th while the diode injects holes, creating a positive-feedback 1/V slope.  
   *Falsify*: Freeze body potential (remove diode) and re-run VG1 sweep; knee must vanish.

**Q3 — V7 knife-edge**

(a) 420 ns is marginally physical for a 130 nm 2T cell (RC time-scale of the body node + impact-ionization delay can reach hundreds of ns), but the exact 420 ns at τ_slow=800 ns, k_n=1e-4 is almost certainly a numerical artifact of the external FHN-trap wrapper. The mainline 3-state solver (V_b, q_F, q_R) plus external snapback cannot sustain the same Hopf once the card is locked; the 578 ns / –7.5 V result is the consistent physical solution.

(b) To widen the capture region to 30–50 % PVT robustness, two additional states/couplings are required: (i) an explicit slow body-charge state (q_body) coupled to the fast drain current via impact-ionization current, and (ii) a temperature-dependent VSAT or M-factor term that modulates the Hopf frequency. Without these the limit cycle remains knife-edge.

(c) The –7.5 V spike is direct evidence of a missing body–source diode forward-bias clamp. BSIM4 body diodes are reverse-only in the present stack; once V_b drops below –0.6 V the model extrapolates without clamp, producing the unphysical excursion. Adding an explicit anti-parallel diode or enforcing V_b ≥ –0.7 V in the wrapper removes the spike and collapses the Hopf.
