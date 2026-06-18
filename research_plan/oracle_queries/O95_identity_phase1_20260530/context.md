# Oracle query O95 — Identity Phase 1 sanity check (4-way)

You are one of four oracles (GPT-5, Gemini 2.5 Pro, Grok-4, DeepSeek-R) being asked to **adversarially critique** a Phase 1 hardware-identity / PUF experiment on twin gfx1151 chassis. Be terse, ≤500 words. **Hostile / falsifying tone preferred — find what's wrong, don't validate.** Cite known PUF or RTN literature if you reference it.

---

## 1. Background

We tested whether the 32 enumerated hardware mechanisms on **AMD Ryzen AI Max+ PRO 395** (Radeon 8060S, gfx1151, RDNA3.5) can yield a PUF-grade fingerprint distinguishing **two physically-twin HP Z2 mini G1a chassis** (identical PCI_ID `1002:1586`, identical subsys `HP 103C:8D1D`). Goal: foundation for a "constitutive HW identity → non-fungibility → stake in survival" research program. Reference frame: PUF literature (Suh & Devadas 2007; Maes 2013), physical reservoir computing (Tanaka et al. 2019).

---

## 2. Protocol (Phase 1)

Per-device, N=500 reps of a fixed challenge kernel, per-CU readouts of `HW_ID`, `SHADER_CYCLES`, `PERF_SNAPSHOT`, atomic-ordering. Three channels:

- **Stable-bit channel** — fixed challenge kernel, per-CU output bits + per-CU SALU cycle count (hwreg(29)). Stable bits = those that don't flip within a device.
- **Process-stat channel** — 1/f spectral fit on cache-eviction latency time-series; spatial correlation matrix across 20 CUs; RTN telegraph detection on long-tail timing.
- **Pure-noise control** — `PERF_SNAPSHOT` (#13); expected ~identical across devices, used as null.

Intended gates: DISCOVERY = intra-HD ≤ 0.10 AND inter-HD ≥ 0.40. KILL = inter-HD ≤ intra-HD.

Phase 1 ran a **single 'idle' thermal regime only** (cold/warm regimes NOT yet executed). Devices were in different rooms at different ambient.

---

## 3. Results

### Stable-bit channel — NULL

| metric | ikaros | daedalus |
|---|---|---|
| n_cu | 80 | 80 |
| signature length (bits) | 640 | 640 |
| intra-HD mean | 0.2432 | 0.2965 |
| intra-HD min/max | 0.1922 / 0.2906 | 0.2547 / 0.3438 |
| bit_stability_mean | 0.7568 | 0.7035 |

**Cross-device:** Inter-HD = **0.2953** vs intra=0.2698. Barely above intra. DISCOVERY gate **not met**.

### Process-stat channel — apparent signal

| metric | ikaros | daedalus |
|---|---|---|
| knee_slope mean | 0.2018 | 0.0883 |
| RTN rate mean   | **0.0000** | **0.1149** |
| spatial_corr_mean_abs | **0.0563** | **0.3601** |

- KL(knee_slope distribution) = **6.54**
- KL(RTN rate distribution)   = **25.1**
- spatial-corr MSE            = **0.092**

### Noise control (PERF_SNAPSHOT)

- KL(PERF hist) = **0.110** (~50× smaller than process-stat KLs) — interpreted as "bulk noise stats not strongly ambient-modulated".

### CONFOUND

Ambient differed **~15 °C** between machines during the run:
- ikaros: 42 → 50 °C (APU)
- daedalus: 27 → 41 °C (APU)

---

## 4. Specific questions — answer each explicitly (a–f)

**a)** Is the **RTN-rate asymmetry (0.000 vs 0.115)** a known *silicon-origin* PUF-grade signature, or a known *thermal artifact* of telegraph noise (RTS trap capture/emission rates are famously Arrhenius-activated)?

**b)** Is the **spatial-CU-correlation asymmetry (0.056 vs 0.360)** more likely silicon (power-delivery layout) or thermal (heatsink loading / shared thermal envelope)?

**c)** Is **KL(PERF) = 0.110** small enough to function as a "thermal-drift null"? If thermal dominated, would we expect PERF_SNAPSHOT to drift too — or is PERF inherently insensitive to the kind of thermal modulation that affects RTN and spatial correlation?

**d)** Given the **stable-bit channel returned NULL** (inter ≈ intra ≈ 0.27, far from PUF-grade), is it even worth running **Phase 2** (transplantation matrix on NARMA-10 reservoir) on the process-stat channel alone? Or should we recategorize this as a "process-statistics fingerprint" project rather than a PUF-identity project?

**e)** Identify any **obvious methodological holes**:
- Should we have used DVFS clamping?
- Should we have run cold/idle/warm regimes before claiming any signature?
- What is the single most damning **adversarial test** that would falsify the silicon-identity claim?

**f)** Is there **any signal here that would be publishable as-is** (e.g. "twin HP Z2 G1a chassis show measurable RTN-rate asymmetry" as a side note in the FEEL paper), even if the silicon-identity claim does not survive?

---

## 5. Output format

Answer (a) through (f) in order. ≤500 words total. No throat-clearing, no validation; assume we already know the result is exciting and want you to break it.
