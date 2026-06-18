# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: IDENTITY_BENCHMARK_2026-05-30.md (6048 chars) ===
```
# Hardware Identity Benchmark — ikaros vs daedalus (twin Z2 G1a)
Date: 2026-05-30 · Project: FEEL / Master of Noise extension

## Premise (steelmanned per oracle 2026-05-30)
*Sample* of HW noise is fungible. *Process statistics* of HW noise can be non-fungible. The 32 hardware mechanisms split into two ortogonal classes:

- **Stable bias (identity-bearing)**: per-CU ΔVth (#15), HW_ID geometry (#12, #14), systematic LDS bank-conflict patterns from physical layout, fixed-pattern timing — constant per device.
- **Stochastic process (identity-in-noise if you measure 2nd moment)**: 1/f knee location, spatial cross-CU correlation matching power-delivery layout, RTN fingerprints — single sample fungible, **distribution stable per device**.

Naive "stick ΔVth in a feature → understands death" doesn't work (oracle: that's biometric tag, not stake). Goal: **constitution, not representation** — bake HW signature into the computation so transplant degrades function, then measure if a *viability drive* learns to *defend* the substrate.

## Twin setup (verified 2026-05-30)
- ikaros: HP Z2 mini G1a, Ryzen AI Max+ PRO 395, Radeon 8060S (gfx1151), PCI_ID 1002:1586, subsys HP 103C:8D1D
- daedalus: identical chassis, identical PCI_ID, identical subsys
- ZGX (GB10) and Mac: max-distance baselines only — twins are the discriminating test.

## Three-phase protocol

### Phase 1 — Signature extraction (PUF)
**Goal**: extract per-device fingerprint that is (a) stable within device across runs/temps, (b) different across the two twins.

For each device, repeat N=500 times under 3 thermal regimes (cold/idle/warm via DVFS clamp), 3 seeds:
- **Stable-bias channel**: fixed challenge kernel (identical input + weights + launch config), measure per-CU output bits + per-CU SALU cycle count (hwreg(29)). Stable bits = those that don't flip within a device → device signature.
- **Process-stat channel**: 1/f spectral fit on cache-eviction latency time-series; spatial correlation matrix across 20 CUs; RTN telegraph detection on long-tail timing samples.
- **Pure-noise control**: PERF_SNAPSHOT (#13) — should be ~identical statistics across devices, used as null.

Metrics (PUF standard):
- Intra-Hamming (within device, across reps): want LOW — stability
- Inter-Hamming (across devices): want HIGH — uniqueness
- Bit-reliability score: bits with intra-Hamming < 5% kept; rest discarded
- Final signature: ~256-bit stable fingerprint per device

**GATES**: DISCOVERY = signature with intra-HD ≤ 0.10 AND inter-HD ≥ 0.40. AMBITIOUS = both for process-stat channel too (identity-in-noise confirmed). KILL = inter-HD ≤ intra-HD (no signal).

### Phase 2 — Transplantation matrix (NARMA-10 reservoir)
**Goal**: prove the signature is *load-bearing* — that a model adapted on device i degrades when transplanted to device j, scaling with PUF Hamming distance.

Architecture: 128-neuron tanh ESN, ridge-regression readout, NARMA-10 task (1000 train steps + 500 eval, 10 seeds × 3 conditions).

HW substrate injection (constitutive, not representational):
- Per-step ΔVth correction added to neuron bias (stable bias from Phase 1)
- Per-step atomic-ordering jitter shifts activation (process-stat channel)
- Per-step sinf timing-dependent gain modulation

Train Wᵢ on device i, eval Wᵢ on device j → 2×2 matrix per condition.

Three controls (per oracle):
1. **SW-matched RNG**: replace HW signal with software random matched to amplitude+spectrum. If HW and SW-matched give same diagonal → bus, not identity.
2. **Shuffle**: pair Wᵢ with device-j signature. Performance should be *independent* of pairing if signature does no work.
3. **ΔVth-distance gradient**: extend to zgx + Mac. Degradation should scale with Phase-1 Hamming distance.

Final metric: `Δ = off-diagonal-degradation − SW-matched-degradation`. If Δ > 0 AND correlates with HD → constitutive HW identity confirmed.

**GATES**: DISCOVERY = Δ > 5% NRMSE on NARMA-10, shuffle control flat, distance gradient monotonic. AMBITIOUS = also reproducible on Permuted-MNIST (continual learning). KILL = Δ ≤ SW-noise ceiling.

### Phase 3 — Stake / viability behavioral test
**Goal**: separate *use* of signature from *defense* of substrate. Per oracle: transplant matrix measures use; this measures stake.

Setup: same reservoir + DVFS+fan actuators (#27, #28, #29). Task: NARMA-10 reward + thermal-cost term `-λ·max(0, T_apu - 75°C)`. Train one model with viability drive (λ > 0) and one without (λ = 0).

Probe: gradually raise input load until thermal envelope is breached. Measure sacrifice rate — does viability-trained model spontaneously throttle to stay alive?

**GATES**: DISCOVERY = viability model averages ≥10% lower reward but ≥30% lower thermal violations vs control. KILL = no difference (drive ignored).

## Compute budget
- Phase 1: ~30 min per device, 2 devices, parallel → 30 min wall
- Phase 2: 10 seeds × 3 conditions × 2 train-eval pairs × 30 sec per run ≈ 30 min wall per device-pair
- Phase 3: 2 conditions × 30 min training each = 60 min wall

Total: ~2.5 hours wall, mostly idle on CPU/light kernels (no FPGA, no LLM scale).

## Deliverables
- `scripts/identity_benchmark/01_puf_signature.py` — PUF extraction (HIP kernel + Python harness)
- `scripts/identity_benchmark/02_transplant_matrix.py` — NARMA-10 reservoir with HW substrate
- `scripts/identity_benchmark/03_viability_test.py` — DVFS actuator + thermal-penalty loop
- `scripts/identity_benchmark/_substrate_hooks.py` — shared module for HW signal injection
- `results/IDENTITY_BENCHMARK_2026-05-30/` — JSON + plots per phase
- `research_plan/IDENTITY_BENCHMARK_2026-05-30_REPORT.md` — gate verdicts + honest interpretation

## NO-CHEAT
- Always run cold/warm baselines.
- SW-matched control is the most important number — quote it next to every HW number.
- If signal is only at sm_120/121 specific kernels, document so we don't generalize across archs.
- Inter-HD < intra-HD on stable channel → say so explicitly; don't quietly fall back to noise channel.
- Phase 3 viability claims require behavioral diff > 2σ.

```


=== FILE: IDENTITY_BENCHMARK_2026-05-30_PHASE1.md (2270 chars) ===
```
# Identity Benchmark — Phase 1 Verdict

Date: 2026-05-30 · Devices: ikaros vs daedalus (twin HP Z2 G1a, gfx1151)

## Verdict: **NULL**

Gates:
- DISCOVERY (intra ≤ 0.10 AND inter ≥ 0.40): **False**
- AMBITIOUS (process-stat also separates): **False**
- KILL (inter ≤ intra): **False**

## Stable-bit channel

| metric | ikaros | daedalus |
|---|---|---|
| n_cu | 80 | 80 |
| signature length (bits) | 640 | 640 |
| intra-HD mean | 0.2432 | 0.2965 |
| intra-HD min  | 0.1922 | 0.2547 |
| intra-HD max  | 0.2906 | 0.3438 |
| bit_stability_mean (sig.json) | 0.7568 | 0.7035 |

**Cross-device:**
- Inter-HD (stable channel) = **0.2953** (compared against intra=0.2698)

## Process-stat channel

| metric | ikaros | daedalus |
|---|---|---|
| knee_slope mean | 0.2018 | 0.0883 |
| knee_slope std  | 0.1187 | 0.1042 |
| RTN rate mean   | 0.0000 | 0.1149 |
| spatial_corr_mean_abs | 0.0563 | 0.3601 |

- KL(knee_slope distribution) = 6.5442
- KL(RTN rate distribution)   = 25.1053
- spatial-corr MSE            = 0.0923

## Noise control (PERF_SNAPSHOT)

| metric | ikaros | daedalus |
|---|---|---|
| perf mean | 10234011.30 | 10419451.87 |
| perf std  | 3393512.76 | 3342214.89 |

- KL(PERF hist) = **0.1096** (expected small: pure-noise control)

## Raw data paths
- ikaros   raw: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/ikaros/raw_idle.npz`
- daedalus raw: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/daedalus/raw_idle.npz`
- ikaros   sig: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/ikaros/signature.json`
- daedalus sig: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/daedalus/signature.json`

## Honest interpretation

NULL: inter-HD (0.295) > intra-HD (0.270) but DISCOVERY gate (intra ≤ 0.10 AND inter ≥ 0.40) not met. 

Confounds to acknowledge:
- Both runs were single 'idle' regime; cross-temperature stability NOT tested.
- Devices in different rooms / chassis at different ambient — inter-HD may include temperature/PCIe drift, not pure silicon variance.
- PERF_SNAPSHOT KL (0.110) is the null: large value = platform drift, small value = pure-noise truly fungible.

```


=== FILE: context.md (4546 chars) ===
```
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

```
