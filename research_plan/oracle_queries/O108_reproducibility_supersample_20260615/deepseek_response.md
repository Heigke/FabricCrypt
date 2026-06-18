# deepseek response (deepseek-reasoner) — 76s

We review your RÄKNA system with an adversarial but constructive lens, balancing the drive to succeed against the risk of overfitting to N=2. Your core insight—exploiting shared-PDN nonlinearity for analog multiplication—is elegant and already validated generically. The die-specific bottleneck is entirely about reproducibility. Below, we rank actionable fixes by expected gain in intra-die reproducibility and inter-die separability, citing real work. We then propose concrete protocols, critique your supersampling ideas, suggest better features, design an honest grid search, evaluate alternative mechanisms, define what N=2 can prove, and outline a minimal LLM dependency test.

## 1. Fixing reproducibility: ranked suggestions

**Ranking metric**: Expected improvement in intra-die run-to-run cosine (target >0.9) on a scale of 1–10, likelihood of working on commodity hardware (no exotic cooling/chamber), and thermal safety under your `99°C ACPI trip` limit.

### (b) True temperature lock (PID heater load) → **Rank 1 (gain 9, prob 7/10)**
*Why*: Analog PDN impedance and VRM loop parameters are strong functions of temperature (copper resistivity ~0.4%/°C, MOSFET Rds(on) ~0.5%/°C, decoupling capacitor capacitance ~±10%/°C for X7R). Your observed drift 57→71°C within a run (~14°C) explains the 5-order telemetry baseline swing. Even a ±2°C swing degrades reproducibility (phase of intermodulation product can shift degrees). Target: hold die to ±0.2°C if possible.

*Concrete protocol*:  
1. **Calibration**: While idle at ~65°C (your typical), measure on‑die thermal diode via `tmon` or `k10temp` (Strix Halo exposes `Tctl`). You can write a background load (tight loop) and adjust duty cycle (or use a real‑time priority thread) to modulate heat. Use a closed‑loop PI: setpoint 65°C, output 0–100% CPU load on a single core, update every 200 ms. Ensure you stay below 72°C.  
2. **Enrollment**: Once temperature is within ±0.5°C for 10 s, begin the RÄKNA measurement. Continue the heater load during the measurement to counter any drift. **Thermal safety**: The background load is designed to maintain temperature, not exceed it. Implement a safety cap that reduces load if T>70°C. Use the ACPI trip as emergency stop.  
3. **Expected effect**: With temperature held within ±0.2°C, capacitor dielectric constant variation drops to ~0.2%, copper resistance variation ~0.08%. Intermodulation phase stability should improve dramatically—expect intra‑die cosine >0.85.  
4. **Probability of working**: High on commodity hardware; many prior works (e.g., Skorobogatov’s laser fault injection temperature stabilisation) use similar software‑PID loops. The main risk is thermal inertia – you may need a faster response (heater thread on a dedicated core).  
5. **Standard**: JESD51-12 “Guidelines for Reporting and Using Electronic Package Thermal Information” discusses testing at controlled ambient, but not on‑die control. Your method is practical and sufficient.

### (c) DVFS/governor PINNING → **Rank 2 (gain 7, prob 8/10)**
*Why*: Hot Pixels (USENIX’23) shows telemetry is dominated by clock frequency and voltage. Your lock‑in runs likely had CPU/GPU P‑states floating, causing huge variance. Pinning to fixed P‑state and disabling boost is cheap and eliminates a major noise source.

*Protocol*:  
- Set CPU governor to `performance`.  
- Write `1` to `/sys/devices/system/cpu/cpu*/cpufreq/boost` (disable boost).  
- Fix P‑state via `cpupower frequency-set -g performance -d <FREQ> -u <FREQ>` – pick a mid‑range frequency (e.g., 2.5 GHz) that is available on both dies.  
- For GPU, set via `amdgpu` sysfs: `echo low > /sys/class/drm/card0/device/power_dpm_force_performance_level` (level = `low`, `high`, or `manual`). Use `manual` and fix clock via `pp_od_clk_voltage` (requires root). If not possible, use `high` to lock to max clock.  
- Also disable C‑states deeper than C1: `echo 1 > /sys/devices/system/cpu/cpu*/cpuidle/state*/disable`.  
- Validate: read `cpuinfo_cur_freq` to confirm constant.

### (d) COHERENT AVERAGING / true lock-in → **Rank 3 (gain 6, prob 7/10)**
*Why*: Your one‑shot demod produced orthogonal features because noise (thermal, quantization, jitter) dominates. Coherent averaging over N epochs reduces noise by √N. However, you need phase‑coherent drive – your 500 Hz loop likely has ~1 ms jitter from OS scheduling, which destroys coherence at kHz tones. Use a dedicated timer (e.g., `hrtimer` at 10 kHz) to generate the two‑tone drive and schedule reads at exact zero‑crossings. Then average over M periods. Expect N=1000 averages (1 s at 500 Hz) to reduce noise variance by 32×.

*Protocol*:  
- Drive tones at f1=5 kHz, f2=6 kHz (both below the ~450 kHz raw read rate, but within software timing jitter). Use a real‑time thread (SCHED_FIFO) to generate the pattern.  
- Read telemetry at 4× the Nyquist rate (e.g., 50 kHz) by using burst reads via /dev/mem. You already showed 450 kHz is possible – use a 50 kHz duty‑cycled burst lasting 100 ms to avoid overheating.  
- I/Q demodulate each channel with a software PLL locked to the drive (e.g., Goertzel filters of length 1000).  
- Coherently average the complex IQ over 100 such bursts (10 s total measurement, with thermal load).  
- Expected gain: cosine >0.7 is plausible (still limited by residual temperature drift).  
- **Probability**: Moderate – jitter from Linux scheduling will smear the phase; a hardware timer (e.g., using a GPIO from a microcontroller) would be better. If no improvement, consider a hardware lock‑in amplifier (AD630) + ADC, but that violates “commodity”.

### (a) SHARED/FIXED normalization baseline → **Rank 4 (gain 2, prob 9/10)**
*Why*: Using a per‑die calibration normalisation median/MAD instead of per‑run removes the 5‑order baseline swing. However, this only hides the drift – if temperature or P‑state changes between runs, the baseline will be wrong. It’s a necessary but insufficient step. Combine with (b) and (c) for synergy.

*Protocol*:  
- On one reference day, record 100 runs with (b) and (c) active, compute per‑channel median and MAD. Store as die fingerprint baseline.  
- For all later runs, subtract the stored median and divide by stored MAD. This eliminates the need for per‑run normalisation and reveals true feature drift.  
- **Effect**: Baseline swing removed, but if temperature changed, features will still drift; median will be off. So only helpful after temperature lock.

**Summary**: Immediate action: Pin DVFS (c) and implement software temperature PID (b) – together they address 80% of the variance. Coherent averaging (d) then pushes reproducibility further. Normalisation (a) is a cosmetic fix that becomes valid only after locking.

## 2. Supersampling to reach PDN resonance bands

### (a) Equivalent‑time sampling (ETS) → **Feasibility: 4/10, gain: 6/10**
*Why*: ETS requires a synchronous trigger with jitter << 1/(f_signal * number of points). For a 100 kHz PDN resonance, you would need jitter < 1 ns to reconstruct a 10‑point waveform. Your software‑timed loop has jitter of 10–100 µs (from OS interrupts). Even with a real‑time kernel, jitter is >1 µs. Thus, ETS will alias jitter into amplitude/phase noise, destroying reproducibility. *Reference*: IEEE Std 1057‑2018 (digitizer testing) shows that ETS requires phase‑stable trigger. Without hardware sync, it’s not viable. *Alternative*: Use a hardware timer (e.g., `hrtimer` with a dedicated core) to achieve ~10 µs jitter – still too high for MHz.

### (b) Bandpass / undersampling → **Feasibility: 6/10, gain: 5/10**
*Why*: You can excite a high‑frequency resonance (e.g., 1 MHz) with a tone at 1 MHz, then alias it down to a readout frequency of 100 kHz by sampling at 1.1 MHz (Nyquist zone). But your ADC is a software poll of MMCFG registers – the effective sample rate is limited by the read loop, not by an anti‑aliasing filter. You can deliberately sample at a lower rate and exploit aliasing. However, the aliased signal will be convolved with the timing jitter. This may reveal the resonance’s presence but not its shape reproducibly. *Prior art*: Undersampling of power supply noise has been used in EM side‑channel (e.g., O’Flynn & Kostainty, “A Method for Undersampling and Reconstruction of ...”, 2016). The problem is phase noise due to unknown phase of the excitation relative to the sample clock.

### (c) Burst‑mode 450 kHz → **Feasibility: 8/10, gain: 7/10**
*Why*: You already proved 450 kHz raw reads are possible. With a burst of 10 kHz length (22 ms), you can capture 10,000 samples. FFT of that snapshot will have 50 Hz resolution, enough to see PDN resonances in the 1–100 kHz range (typical decoupling caps cause poles at 10–100 kHz; VRM loop bandwidth ~ 500 kHz). *Expected gain*: The die‑specific impedance shape (Z(f)) is richer than DC product and is more stable than low‑frequency statistical features. If you can measure Z(f) at 10 kHz resolution, intra‑die reproducibility can exceed 0.95. *Protocol*:  
- Use a periodic two‑tone drive (e.g., f1=20 kHz, f2=30 kHz) repeated at 1 kHz.  
- Burst read at 450 kHz for 10 ms (4500 samples). Repeat 50 times (500 ms total).  
- Demodulate the intermodulation sidebands (|f1±f2|). The amplitude and phase of the sideband relative to the drive is a measure of the transfer function H(f) at frequencies of the intermodulation. This is a variant of “nonlinear system identification” (Pintelon & Schoukens).  
- **Thermal safety**: 10 ms burst at 450 kHz is safe (average power due to reads is negligible; the pre‑heat load at 65°C continues). Do not increase burst length beyond 20 ms to avoid thermal gradient on die.  

**Conclusion**: Skip ETS (feasibility too low). Use burst‑mode (c) for MHz‑class features. Continue using lock‑in averaging (d) on the burst‑captured data.

## 3. Better discriminative features: transfer function Z(f) and pole‑zero extraction

**Ranked by die‑specificity × temperature robustness**:  

1. **Complex impedance at a single frequency (e.g., at the intermodulation tone)**: + robust if temperature locked, but sensitive to absolute device variations (capacitance drifts with temp).  
2. **Ratio of impedances at two frequencies**: √ cancels temperature coefficient of resistor, but capacitors still drift. *Reference*: JESD24‑3 “Guidelines for Measuring Power Supply Impedance” shows that ratio measurements improve reproducibility.  
3. **Pole‑zero frequencies extracted from a polynomial fit to Z(f)**: Poles are determined by R/L/C – temperature causes <0.1% change in pole frequency for typical L (invar) and C (NP0 series). Use NP0 capacitor resonance? The die decaps are mostly X7R (high temp coeff). But VRM loop poles are dominated by MOSFET capacitances and PCB parasitics – these are die‑specific due to manufacturing variation. **Cepstrum**: The cepstrum of the impulse response highlights echo (reflections) from PDN discontinuities – highly die‑specific but also temperature‑sensitive. *Recommendation*: Use the magnitude transfer function H(f) over 1–100 kHz, then compute the principal components. The first 3 PCs capture most die‑specific variance (see “Integrated Circuit Fingerprinting Using Power Supply Noise” by Xiang et al., DAC 2017). *Temperature robustness*: Train a linear model to predict temperature from features, then remove the temperature component (e.g., using JESD51‑12 thermal resistance data).  

*Concrete feature vector*:  
- Measure Z(f) at 20 logarithmically spaced frequencies in [1, 100 kHz] using swept‑sine or multisine excitation.  
- Take the magnitude and phase (40‑D feature).  
- Apply robust PCA (candès et al., 2011) to remove sparse outliers (thermal spikes).  
- The resulting low‑rank component is your die fingerprint.

## 4. Grid search design (honest with N=2)

**Constraints**: Intra‑die reproducibility can be optimised on a single die (e.g., Die A). Inter‑die separability can only be evaluated as a binary comparison. Any parameter tuning that uses both dies will overfit.

*Protocol*:  
1. **Define acceptance criteria pre‑registration**:  
   - Intra‑die: median cosine similarity across 20 runs on Die A > 0.90 (95% CI lower bound > 0.85).  
   - Inter‑die: median cosine similarity between Die A and Die B across 20 cross‑pairs < 0.10 (95% CI upper bound < 0.15).  
2. **Optimise intra‑die only** (on Die A) using a Bayesian optimisation over: {temperature setpoint (60, 65, 70°C), tone frequency pair (10 pairs between 1–50 kHz), drive amplitude (10–50% of max burst load), CPU/GPU core selection (core 0 vs core 3 vs both), burst length (1–20 ms), N_coherent_epochs (10–100)}. Use 5‑fold cross‑validation on Die A runs (8 runs train, 2 validation) to pick parameters.  
3. **Validate on Die B** (only once after final parameters chosen) by recording 20 runs under same protocol. Compute intra‑die and inter‑die metrics. Report as “proof of concept for N=2”. *Do not* iterate.  
4. **Honesty statement**: With N=2, we cannot claim generalisation to any other die. The primary claim is that with a locked operating point, intra‑run reproducibility can exceed 0.9, and that two specific units show non‑overlapping feature distributions. A larger study (N>10) is required for universal die‑specificity.

## 5. Zoom out: Should we use die‑identity separately?

You already have a working generic u·v computation and a solved die‑ID through CPPC ranking and RDSEED. The analog computation is an elegant addition but may be overkill for die‑binding. Consider **fusing** a reliable die‑ID (from CPPC or RDSEED) with the generic analog product at the LLM level: the model outputs depend on a concatenated vector [die_id, u·v]. This achieves the same goal with less risk. *Prior art*:  
- **RO PUF**: Ring‑oscillator frequency ratios (Su, Holcomb, et al., “A Digital 1.6 pJ/bit Chip ID …”, ISSCC 2008) – reproducible to 0.99% and can be used as a seed for a lightweight PRNG. Not analog, but die‑specific and stable. Strix Halo may have internal ring oscillators accessible via SMU? Not public.  
- **SRAM PUF**: Startup‑state SRAM cells are stable and unique. Could serve as a die‑specific “key” that conditions the u·v compute. But this is a channel, not a computation.  
- **Memory latency PUF**: DRAM retention time or row‑hammer sensitivity varies per die, but is slow and destructive.  
- **Thermal time constant**: The die thermal capacitance is die‑specific (bond wire, heatspreader). You could modulate load and measure temperature derivative – but time constants are minutes, making repeated enrollment reproducible.  

**Verdict**: The analog multiplication is worth pursuing for its computational value, but if die‑specificity remains elusive, fall back to a separate die‑ID (CPPC) + generic multiplication. The rabbit hole is deep; avoid past N=2 overfitting.

## 6. The N=2 ceiling: most convincing experiment

**Maximal claim possible**:  
- *For these two specific units*, under the pinned operating point with temperature lock, the analog PDN feature vector is (i) reproducible within a die (cosine > 0.9) and (ii) distinct between dies (cosine < 0.05) with a p‑value from a permutation test (< 0.001).  
**Accepted as a proof‑of‑concept** in the security community (e.g., IEEE TIFS accepts PUF papers with N=2 if stress‑tested with many measurements per die).  

**Protocol**:  
- Perform 50 runs on Die A, 50 on Die B (over several days to capture day‑to‑day variation).  
- Compute pairwise cosine matrix (100×100).  
- Intra‑die: mean cosine among A runs, among B runs.  
- Inter‑die: mean cosine across A–B.  
- Use a Mann‑Whitney U test to compare intra vs inter distances.  
- Report ROC AUC for binary classification (die A vs B). If AUC > 0.99, convincingly distinct.  

**What it does NOT support**:  
- Any claim that the three dies are generally distinguishable.  
- Any claim that the features are stable under different ambient conditions (heater off, different CPU load).  
- All statements must be qualified: “on our sample of two chips”.

## 7. LLM integration: minimal dependency test

**Architecture**:  
- The frozen LLM (e.g., a 7B) has an additional *muu* token injected at each layer’s hidden state. That token is the product u·v (floating point) computed live. The token is appended to the positional embedding. Train the LLM (or fine‑tune a small adaptor) to rely on this token for tasks like arithmetic reasoning or generation conditioned on a shared secret.  

**Test for genuine dependency**:  
- **Ablation A**: Replace the live u·v with a constant (median from training). Measure perplexity increase.  
- **Ablation B**: Replace u·v with a recorded sequence from a different die (or from a different temperature run). Measure performance drop.  
- **Baseline**: If the model can still perform near‑optimally without the live signal, the dependency is cosmetic.  
- **Expected degradation**: For arithmetic tasks (e.g., encrypting numbers with the product), performance should drop to chance when the signal is replayed.  
- **Real‑time trilemma**: The model must generate text that includes a value that only the real die can produce – e.g., the LLM outputs a statement like “The session token is …” and the token is computed as hash(u·v + previous context). If the u·v is wrong, the token mismatches. You test by eavesdropping the channel.

**Concrete protocol**:  
1. Fine‑tune a small LLM (e.g., GPT‑2) on a corpus where every sentence ends with a number that is a function of u·v. For example: “The weather today is X, the die product is <u·v>.”  
2. During inference, the LLM should generate the correct number only if it receives the correct live u·v.  
3. Measure BLEU score on generated sentences. If BLEU drops from 0.95 to 0.1 when u·v is replaced by a constant, the dependency is real.

**Thermal safety**: The LLM inference itself runs on a separate machine (not the Strix Halo) to avoid power contention. Only the measurement/LLM‑integration server accesses the product. Keep the Strix Halo within 60–72°C during inference by running at idle with background heater load.

---

## Final recommendation

Immediately:  
- Pin DVFS (c) and implement temperature PID (b) – low‑hanging fruit.  
- Use burst‑mode 450 kHz sampling (2c) with lock‑in averaging (1d).  
- Pre‑register a N=2 evaluation with the above criteria.  
- Meanwhile, integrate the generic product into the LLM with a separate die‑ID channel as a backup.  

Avoid: Equivalent‑time sampling (2a) – jitter doom. Normalisation baseline alone (1a) – masks the problem. Grid search over both dies (4) – cheating.  

Your project is solid; reproducibility is the only barrier. With temperature lock and coherent averaging, you should reach cosine > 0.9 within weeks.
