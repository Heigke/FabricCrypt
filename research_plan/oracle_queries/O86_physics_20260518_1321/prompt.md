# O86 — Hostile physics oracle: close DC gap, fix V3 knee, harden V7

We have ~1.0-1.4 dec DC fit (33 biases) on a 130nm 2T NS-RAM cell calibrated to Mario silicon. Required for tape-out guidance: 0.3-0.5 dec. V3 DC-knee at VG1=0.4 is a sharp 1/V behavior that current BSIM4 + IFT-snap model DOES NOT reproduce. V7 free oscillation works at one knife-edge point (τ_slow=800ns, k_n=1e-4 in FHN-trap wrapper) but vanishes when card-locked.

Mainline solver `nsram.bsim4_port.transient_real_v2` is 3-state (V_b, q_F, q_R) + external snapback_subcircuit; FHN-trap is NOT integrated, lives in mode-atlas wrapper. nsram_cell_2T.py uses BSIM4 v4.8.3 pyport + Gummel-Poon NPN + Sebas pdiode + well-diode.

Mario data we have: Id-VG family across 4 VG2 biases, 200ns pulse transient (Id_pk=4.8mA target), V_b=4V settle, t_rise=26ns, t_fall=76ns, period 430ns.

What we DON'T have: thick-ox cell card from Sebas, 7-rate transient sweep, mismatch data, RF small-signal.

**Q1 — DC gap physics**
What 3 most likely physics deficiencies account for 1.0-1.4 dec gap? Be specific about WHICH BSIM4 terms or missing devices/parasitics. For each: how to test in our pyport (≤4 GPU-hours).

**Q2 — V3 DC-knee at VG1=0.4**
This is a fundamental physics failure not a fitting issue. What physical mechanism produces a sharp 1/V relationship that BSIM4 IIMOD + NPN combinations can't capture? Top 3 hypotheses, each with falsification test.

**Q3 — V7 knife-edge**
z477c produced 420ns Hopf at τ=800ns, k_n=1e-4 but mode-atlas showed period→578ns when card-locked and V_b spikes to -7.5V (out of [-1, +1.5]V physical range). 
- (a) Is 420ns even physical for a 130nm 2T NS-RAM or are we chasing a numerical artifact? 
- (b) If physical, what additional state/coupling would broaden the Hopf capture region from a knife-edge to 30-50% PVT robustness? 
- (c) Can the V_b spike to -7.5V indicate a model deficiency (e.g. body-source diode forward-bias clamp not enforced)?

Be brutally honest. Cite mechanisms by name (e.g. "DITS", "VSAT", "DSUB", "PCLM"). Return ≤500 words per Q.
