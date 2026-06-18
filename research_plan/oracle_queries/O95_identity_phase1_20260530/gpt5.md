# openai response (gpt-5) — 90s

a) RTN-rate asymmetry: thermal/bandwidth artifact, not a silicon PUF. RTN capture/emission times are Arrhenius and bias activated; 10–15 °C shifts easily move corner frequencies orders of magnitude (Kirton & Uren, 1989; Simoen & Claeys, 2013; Grasser et al.). Your “0.000” on the hotter part is exactly what happens when switching escapes your detection band (too fast to resolve → looks Gaussian), while the cooler part sits in-band. A rate is not a stable identifier without tight T/V control and helper data (Maes, 2013).

b) Spatial CU correlation: overwhelmingly thermal/PDN, not “silicon layout.” Cross-CU correlations track shared supply droop, fan curve, and thermal gradients; they swing with DVFS governors and heatsink loading. A 6× jump (0.056→0.360) under a 15 °C ambient mismatch screams shared-envelope coupling, not die-specific fingerprints. Equalize Tdie and clamp clocks/voltage and this will collapse.

c) KL(PERF)=0.110 is not a “thermal-drift null.” PERF_SNAPSHOT counters are coarse, often normalized/quantized, and largely blind to low-frequency flicker/RTN kinetics. Thermal modulation that moves trap kinetics won’t necessarily perturb your aggregated perf histograms. Without a calibrated baseline for KL across matched ambients and repeats, 0.110 is uninterpretable. This null is invalid.

d) Phase 2: don’t. With the stable-bit channel dead (intra≈0.27, inter≈0.30), any “constitutive identity” you inject from process stats is an environment/servo fingerprint. You’ll just train the ESN to the thermostat and PDN. SW-matched RNG will likely erase any off-diagonal effect. Reframe as “process-statistics under controlled T/V” first; only call it identity after demonstrating T/V-invariance and helper-data-corrected reliability (Suh & Devadas, 2007; Maes, 2013).

e) Methodological holes:
- No DVFS/voltage clamping; fan PWM/fan curves unsynchronized; 15 °C Tdie mismatch.
- Single thermal regime; no cold/idle/warm sweeps; no Arrhenius fit of RTN time constants.
- Different rooms/PSUs/airflow; no common AC line or PSU swap control.
- No fuzzy extractor/bit selection; 70–76% “stability” is PUF-useless.
- CU mapping/dispatch not randomized; scheduler affinity, power gating, and occupancy confounds.
- Detector bandwidth/thresholds for RTN not calibrated; aliasing likely.
Most damning falsification: lock both chips to identical f/V, fixed fan PWM, same PSU/room, hold Tdie within ±0.5 °C, then temperature-ramp one device. If RTN-rate and spatial-corr follow temperature (or collapse when matched), the “silicon identity” claim is dead.

f) Publishable as-is? Barely as a negative/control note: “GPU SALU/timing ‘stable-bit’ PUF fails on RDNA3.5 under idle; RTN/spatial-corr are thermally dominated.” A figure showing RTN detection bandwidth/Arrhenius drift on RDNA3.5 could be a workshop cautionary result. “Twins show RTN-rate asymmetry” without T/V control is not publishable as identity. References: Kirton & Uren (1989); Simoen & Claeys (2013); Grasser et al.; Suh & Devadas (2007); Maes (2013).
