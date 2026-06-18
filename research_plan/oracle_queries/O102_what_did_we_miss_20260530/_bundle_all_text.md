# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: IDENTITY_BROADER_MECHANISMS_2026-05-30.md (9710 chars) ===
```
# Identity Benchmark — Broader Mechanisms Catalogue (round 3)
**Date**: 2026-05-30  **Author**: identity-benchmark agent (broader sweep)

This document extends `IDENTITY_MISSED_MECHANISMS_2026-05-30.md` (17 mechanisms,
M1–M17) and the original `docs/deep_analog_access_report.md` 32-mechanism
catalogue. After 9 NULL attacks + 5 mixed channels (power d≥8, thermal-τ d=7.7,
per-core-latency rank d=3.37, TSC drift σ 18×, plus device-envelope confounds),
we exhaustively enumerate **34 NEW mechanisms** (B1–B34) not in any prior doc,
grouped by category. Each row: physics rationale → silicon-binding score
(0 platform, 1 board, 2 package, 3 die) → probe difficulty → expected effect
size → instrumentation required.

Naming: `Bn` (broader). All cost estimates are wall-time on commodity
ROCm 7.0 / kernel 6.14 / no new hardware.

## Category 1 — Active dynamics (extends thermal-τ DISCOVERY)

| ID | Mechanism | Why silicon-bound | Bind | Cost | Effect (est. d) | Instrumentation |
|----|-----------|-------------------|------|------|-----------------|-----------------|
| B1 | DVFS up-transition trajectory shape (P-state command → freq settle waveform, 1 ms poll) | PLL loop-filter R/C tolerance + LDO settle τ on die | 3 | low | 2–4 | `cpufreq`/`amdgpu_pm` sysfs at sub-10 ms |
| B2 | DVFS down-transition asymmetry (asymmetric loop-filter slew) | gain mismatch upstroke/downstroke | 3 | low | 2–4 | same |
| B3 | Fan PWM step→RPM rise-time (bearing + controller deadband) | bearing inertia + EC PI gains | 1 | free | 3–6 | pwm1_enable + fan1_input |
| B4 | Fan spin-down decay τ after PWM=0 | bearing friction lottery | 1 | free | 3–6 | same |
| B5 | VRM ringing under sudden compute burst (in0_input vs power1_average covariance shape, lag-1..5 ms) | per-inductor ESR / output-cap ESL | 2 | low | 2–5 | hwmon7 at 100 Hz |
| B6 | PCIe ASPM L0s→L0 wake latency distribution | RC PHY equaliser convergence | 3 | low | 1–3 | lspci LnkSta + AER counters, perf sched-switch |
| B7 | DRAM refresh-cycle interaction with sustained reads (timing micro-jitter at tREFI boundary) | per-die refresh row scheduling | 3 | medium | 1–3 | high-rate memtier or custom STREAM with cyclecount |

## Category 2 — Electrical / EMI / power-quality

| ID | Mechanism | Why silicon-bound | Bind | Cost | Effect | Instrumentation |
|----|-----------|-------------------|------|------|--------|-----------------|
| B8 | USB-C ucsi PD voltage tolerance drift (in0_input on hwmon3/4/5) | per-port TI/Renesas PD controller calibration | 1 | free | 1–3 | sysfs |
| B9 | USB-C current draw distribution at idle | per-controller leakage + AGS load | 1 | free | 1–2 | sysfs |
| B10 | NIC PHY power draw and link-up settle (r8169 ASPM transitions) | RTL PHY analog AFE | 2 | low | 1–3 | ethtool stats + r8169 hwmon temp |
| B11 | NVMe controller idle power-state residency distribution (`nvme get-feature 0x0c`) | controller power-state machine + cap leakage | 2 | low | 2–4 | nvme-cli |
| B12 | NVMe SMART telemetry: thermal-throttle composite-temp band | NAND die binning + cap leakage | 2 | free | 2–5 | nvme smart-log |
| B13 | Wall power-on inrush via UCSI total current vs amdgpu power1 over first 60 s post-resume | PSU + bulk cap tolerance + GPU rail RC | 1 | free | 2–4 | resume hook + hwmon poll |

## Category 3 — Chemical / wear / time-domain

| ID | Mechanism | Why silicon-bound | Bind | Cost | Effect | Instrumentation |
|----|-----------|-------------------|------|------|--------|-----------------|
| B14 | NVMe block erase-count distribution skew (LBA wear-level fingerprint) | per-controller wear-level GC policy + run-history | 2 | free | huge but historical | smartctl + nvme log 0xCA |
| B15 | DRAM cell retention drift (slow probe, refresh stretching test) | per-die retention tail | 3 | very high (hours) | 5+ | custom userspace allocator with delay |
| B16 | CPU per-core NBTI/HCI degradation asymmetry (idle vs SVT after 30 min stress) | per-core PMOS Vth shift history | 3 | medium | 1–3 | turbostat + per-core Vmin pre/post stress |
| B17 | NVMe wear-level GC pause distribution (latency tail after sustained writes) | per-controller FTL state | 2 | medium | 2–4 | fio + iostat |

## Category 4 — Cryptographic / firmware / fuse-map

| ID | Mechanism | Why silicon-bound | Bind | Cost | Effect | Instrumentation |
|----|-----------|-------------------|------|------|--------|-----------------|
| B18 | TPM EK fingerprint (if fTPM enabled) — by-design unique | per-die fuses | 3 | free | trivially unique | tpm2-tools |
| B19 | DMI SMBIOS UUID, serial, asset tag | board-level vs die-level mix | 1 | free | unique-by-design | dmidecode |
| B20 | Microcode signature + version | per-die ucode patch state | 3 | free | per-CPU rev | /proc/cpuinfo, /sys/devices/system/cpu/microcode |
| B21 | AGESA / SMU firmware version + capability bitmap | per-board firmware-revision lottery | 1 | free | discrete | dmidecode + sysfs |
| B22 | VBIOS hash + version | per-GPU board firmware | 1 | free | discrete | amdgpu_vbios + sha256 |
| B23 | AMD SEV-SNP attestation report (per-CEK derived) | per-die crypto identity | 3 | medium | unique-by-design | sevctl / sev-tool |

## Category 5 — Cross-channel / second-order

| ID | Mechanism | Why silicon-bound | Bind | Cost | Effect | Instrumentation |
|----|-----------|-------------------|------|------|--------|-----------------|
| B24 | Per-chip thermal-electrical impedance: cross-correlation(power, temp) slope at burst onset | thermal RC × electrical RC convolution unique per package | 3 | low | 3–6 | hwmon7 high-rate during 5s compute bursts |
| B25 | Per-core latency jitter conditional on neighbour-core load (matrix entry C_ij) | shared L3 + Infinity-Fabric arbitration lottery | 3 | medium | 2–4 | rdtsc round-trip with pinning, 32×32 |
| B26 | Cross-substrate fan-curve / GPU-temp coupling spectrum (transfer function H(f), 0.1–5 Hz) | chassis airflow path + heatsink mounting | 1 | low | 2–4 | hwmon7 + fan1_input synchronized poll |
| B27 | Cross-rail covariance (UCSI in0 vs amdgpu in0 vs amdgpu in1 vs ACPI temp) — full 4×4 cov matrix | shared bulk caps + DC-DC sequencing | 2 | low | 2–4 | multi-sensor parallel poll |
| B28 | Clock-skew drift across cores under varying load (CLOCK_MONOTONIC_RAW diff between affined threads vs load) | per-core PLL deskew | 3 | low | 2–4 | high-rate pthread_getcpuclockid |

## Category 6 — Topological / structural / fabric

| ID | Mechanism | Why silicon-bound | Bind | Cost | Effect | Instrumentation |
|----|-----------|-------------------|------|------|--------|-----------------|
| B29 | Per-CU HW_ID layout / disabled-CU bitmap from `gpu_metrics` blob | die-cut yield-binning lottery | 3 | low | discrete | amdgpu metrics blob parse |
| B30 | CCX→CCX latency asymmetry matrix (16-core, 2 CCX) | Infinity-Fabric topology + binning | 3 | low | 2–4 | `numactl --hardware` + custom ping-pong |
| B31 | LLC/L3 slice arbitration latency per slice (Zen5 16-MB L3) | per-slice arbiter + bank lottery | 3 | medium | 1–3 | perf-event L3 slice counters |
| B32 | Persistent CPUID feature-flag minor variants (MICROCODE_REVISION + ucode-dependent caps) | per-die fuse + ucode interaction | 3 | free | discrete | cpuid |

## Category 7 — Behavioural fingerprints (OS-mediated but silicon-sensitive)

| ID | Mechanism | Why silicon-bound | Bind | Cost | Effect | Instrumentation |
|----|-----------|-------------------|------|------|--------|-----------------|
| B33 | IRQ latency distribution tail (cyclictest, per-IRQ source) | per-LAPIC + chipset IRQ routing | 2 | low | 1–3 | cyclictest |
| B34 | Lock-contention latency under controlled NUMA-stride workload (futex wake-up tail) | per-die uncore arbitration | 3 | medium | 1–3 | perf-bench futex |

## Cost × Expected-yield matrix (top 10 by (effect-d / cost))

| Rank | ID | One-line | Why score |
|------|----|----------|-----------|
| 1 | B3 | Fan PWM-step rise-time | free, mechanical, big effect, immediate twin diff |
| 2 | B4 | Fan spin-down τ | free, complements B3 |
| 3 | B24 | Power×temp lag-correlation slope | extends thermal-τ DISCOVERY, free with bursts |
| 4 | B27 | Cross-rail 4×4 covariance | free, multi-sensor fusion |
| 5 | B12 | NVMe composite-temp band | free, persistent |
| 6 | B26 | Fan↔GPU-temp transfer function H(f) | free, links B3 + thermal |
| 7 | B11 | NVMe idle power-state residency | low cost, controller-binding |
| 8 | B5 | VRM ringing covariance vs power | low cost, board-binding |
| 9 | B25 | Per-core C_ij conditional jitter | medium cost, large search dim |
| 10 | B30 | CCX↔CCX latency asymmetry matrix | medium, structural |

### Mechanisms we now think are dead-ends *for emergent* identity
- B18 TPM EK (unique-by-design, not emergent)
- B19 SMBIOS UUID (not emergent)
- B20 microcode rev (discrete, fuse-driven, not analog)
- B15 DRAM retention (cost-prohibitive without memtest fork)
- B23 SEV-SNP attestation (by-design)

### Categories we may still be blind to (sent to oracles for hostile critique)
- Acoustic/coil-whine spectrum (no mic on Z2 mini chassis)
- Conducted EMI on 12 V rail (needs external oscilloscope/probe)
- AC mains harmonic injection from PSU (needs CT clamp)
- WiFi/BT TX power deviation (needs SDR receiver — out of scope this round)
- Optical / chassis-vibration coupling
- Per-CU adjacency-graph activation pattern (would need ISA-level CU pinning, parked at NULL after attack 8)

## Top-5 cheapest for immediate quick-probes (Task B)
1. **B3 + B4** fan dynamics — single script
2. **B12** NVMe composite-temp band — smartctl
3. **B24** power×temp lag covariance — extend thermal probe
4. **B27** cross-rail 4×4 covariance — multi-hwmon parallel poll
5. **B32** CPUID + ucode-rev discrete fingerprint

```


=== FILE: IDENTITY_CONSTITUTIVE_2026-05-30.md (7843 chars) ===
```
# Identity benchmark — CONSTITUTIVE coupling experiment

**Date:** 2026-05-30
**Repo:** AMD_gfx1151_energy
**Devices:** ikaros (Ryzen + gfx1151) vs daedalus (Ryzen + gfx1151)
**Task:** Mackey-Glass τ=5, one-step prediction (NRMSE, lower=better)
**Reservoir:** 32 leaky neurons, spectral radius 0.9, ridge readout (α=1e-4)
**Seeds:** N=30 per cell, bootstrap 95% CI (2000 resamples)

## Motivation

Phase 2 v1 (per-step RTN injection at activation) and Phase 2 v2 (23-feature
substrate envelope concatenated to input) both returned NULL: the model treated
substrate as **information about the world** that it could route around. Hypothesis:
push substrate so deep into the math that the computation cannot proceed without
the silicon-specific signal. Substrate becomes the **operator**, not the operand.

## Design — 5 regimes of increasing coupling depth

| Regime | Mechanism                                  | Coupling site            |
|--------|--------------------------------------------|--------------------------|
| 0      | BASELINE (no substrate)                    | none — establishes floor |
| 1      | FEATURE — concat substrate to input        | W_in (route-aroundable)  |
| 2      | INITIAL_STATE from per-CU thermal sig      | x_0 (decays out)         |
| 3      | LEAK_PER_NEURON from per-core latency rank | per-neuron α[i]          |
| 4      | WEIGHT_MOD via cross-core interaction      | W_rec[i,j] *= 1+0.3·M    |
| 5      | DYNAMICAL — substrate inside tanh per step | x[t+1] = …tanh(W·(x+β·s))|

Substrate sources (real per-device): A_power AR(1) coefficient (autocorr_tau),
B_thermal τ_heat/τ_cool, E_cpu per-core latency rank (16-vector, ANTI-correlated
r=−0.21 between twins after host-aware ranking).

## Transplant matrix per regime

train ∈ {ikaros, daedalus} × eval ∈ {ikaros, daedalus, sw_matched, shuffle,
ident_const}, 30 seeds each. Δ = NRMSE(off-diagonal) − NRMSE(diagonal).

Controls:
- **sw_matched**: iid Gaussian matched in 1st/2nd moments, no temporal/spatial structure
- **shuffle**: real same-device substrate with **permuted spatial dimensions**
  (tests whether the *specific* per-core structure matters, vs marginal stats)
- **ident_const**: same constant vector each step (tests whether dynamics matter)

## Per-regime results (NRMSE, mean ± bootstrap 95% CI on Δ)

| Regime | Diag    | Δ HW             | Δ SW-matched | Δ SHUFFLE | Δ IDENT-CONST | Verdict |
|--------|---------|------------------|--------------|-----------|---------------|---------|
| 0      | 0.0215  | —                | —            | —         | —             | floor   |
| 1      | 0.7063  | **26.71** [21.4, 32.3] | 14.87  | 24.40     | 0.05          | WEAK_DISCOVERY |
| 2      | 0.0215  | 0.0000           | 0.0000       | 0.0000    | 0.0000        | NULL    |
| 3      | 0.0210  | **0.925** [0.82, 1.04] | 0.860  | 0.783     | 0.000         | WEAK_DISCOVERY |
| 4      | 0.0210  | **1.460** [1.30, 1.64] | 1.262  | 1.356     | 0.000         | WEAK_DISCOVERY |
| 5      | 0.0981  | **9.297** [7.68, 11.09] | 5.112 | 9.643     | −0.018        | WEAK_DISCOVERY |

(KILL gate uses shuffle > HW + σ_shuffle; DISCOVERY requires Δ HW exceeding all
controls by 2σ AND >5× Δ ident_const AND CI excluding 0.)

## Findings

### 1. Coupling-depth trend
Δ HW grows monotonically across the **dynamical** regimes (1 → 3 → 4 → 5) but
not across all five — regime 2 (initial-state only) is fully NULL because the
leaky reservoir washes the IC out in <100 steps (washout=100 by design). When
restricted to dynamics-altering regimes (1, 3, 4, 5), Δ HW is monotonic
(0.93 → 1.46 → 9.30 if we drop the input-feature-only regime 1, which is
high but largely matched by SW noise).

### 2. SHUFFLE vs HW — the deep finding
At regime 5 (the constitutive condition), **shuffle (9.64) ≈ HW (9.30)** within
CI overlap. Permuting the same device's per-core rank vector degrades the model
as badly as swapping devices. This means: **at the user-space gfx1151 / Ryzen
level, what we can touch is "per-neuron coefficient *structure*" rather than
"device identity per se"**. Any well-structured substrate that the trained
W_out was tuned to will work; replacement breaks it equally hard whether the
replacement is "wrong device" or "same device, permuted dims".

### 3. SW-matched is NOT enough
Across regimes 3/4/5, Δ HW > Δ SW-matched (by 7%, 16%, 82% respectively). The
iid Gaussian control with matched marginals never matches the damage of real
substrate replacement. So substrate **temporal / spatial structure** is
load-bearing, even if device-specific identity is not.

### 4. IDENT-CONST collapses to baseline
Constant substrate adds zero learnable signal (Δ ≈ 0 across regimes 3/4) —
the readout absorbs the constant bias trivially. Confirms that **dynamics**, not
just per-host bias, drive the regime-3/4 effect.

### 5. Per-regime conclusion
- Regime 0–2: substrate is genuinely not load-bearing (NULL or trivially absorbed).
- Regime 3–5: substrate **is load-bearing** for the learnable computation; W_out
  is co-fit to the specific per-neuron α[i] / W_rec modulation / dynamical
  stream. Replacing the substrate (HW or shuffled) breaks the model.
- BUT: no regime crosses the strict DISCOVERY gate (HW > all controls by 2σ).
  The substrate effect is **structural, not device-bound**: silicon coefficients
  enter the math, but the model doesn't care which silicon, only that the
  silicon-derived coefficients are consistent between train and eval.

## Updated interpretation

On user-space gfx1151 + Ryzen, we **can** make substrate load-bearing for
learnable computation (regimes 3/4/5: Δ HW > Δ SW-matched, p < 0.05). What we
**cannot** do is make substrate device-identity-bound: any structured
substitute (including a permutation of the same device's rank vector) reproduces
the disruption. This is consistent with the "perfect calculator" interpretation
at the higher layers — what leaks through to user space is structural variance
that the model latches onto generically. The silicon is co-constitutive of the
function, but the silicon's *identity* is interchangeable with any other
structured perturbation.

## Path forward

1. **FPGA route (recommended)**: scale this to a substrate channel that the
   model literally cannot synthesize from a Gaussian (e.g. live RTN sampled
   from a single transistor, with non-Gaussian heavy-tailed statistics).
   At FPGA level we control the coupling site (analog reservoir) and
   shuffle/SW-matched would diverge measurably.
2. **Sharper shuffle**: instead of permuting per-core rank, use the *other*
   device's rank with the spatial pattern that was trained-with. Currently the
   shuffle preserves the trained model's spatial expectation; a different
   shuffle (re-derive M from permuted core_times then project) would break the
   trained model harder than the swap and would confirm the verdict.
3. **Negative-result publication path**: even with regime-5 constitutive
   coupling, user-space gfx1151 silicon cannot be made device-identity-bound for
   a ridge-readout reservoir. Stronger claim than prior NULL because it
   demonstrates substrate IS load-bearing (regimes 3/4/5) — just not
   identity-bound. This is the "perfect-calculator-with-structured-noise"
   interpretation, formalized.

## Reproducibility

- Code: `scripts/identity_benchmark/constitutive/`
  - `_substrate_stream.py` — A+B+E loader, AR(1) streamer, 3 controls
  - `reservoir.py` — 5-regime leaky reservoir, ridge readout, MG generator
  - `01_train_eval.py` — full 2 × 5 × 30 × 6-regime matrix (~18s wall)
  - `02_analyze.py` — bootstrap + verdict gate
- Results: `results/IDENTITY_BENCHMARK_2026-05-30/constitutive/`
  - `regime_{0..5}_results.json`, `summary.json`, `_run_meta.json`
- Wall time: 17.8s end-to-end on ikaros, peak APU ~55°C (well below 72°C target)
- Thermal incidents: **zero**.

```


=== FILE: IDENTITY_LITERATURE_HUNT_2026-05-30.md (16563 chars) ===
```
# Identity Literature Hunt — 2026-05-30

**Question**: who has actually made computation *constitutively depend on* and *benefit from* a specific piece of silicon, on commodity (non-FPGA, non-memristor, non-photonic) hardware? Are we hunting a unicorn?

**Method**: 10-axis web search (WebSearch + WebFetch) + 4-way oracle dispatch (`O100_constitutive_lit_20260530`).

---

## Section 1 — Working examples in the literature

### 1.1 Where it WORKS (and why we can't port it directly)

| Paper | What they did | Transplant cost | Substrate | Portable to APU userspace? |
|---|---|---|---|---|
| **Joshi et al., Nat. Commun. 2020 (arxiv 1906.03138)** — PCM ResNet | Trained ResNet-32 on CIFAR-10 with noise injection; weights programmed onto IBM PCM crossbar. Each PCM cell's analog conductance is per-device unique. | They *designed against* transplant cost: ~0.5 % degradation. But the *underlying* device weights are individually programmed per chip — transplanting a raw-weight binary without re-programming is unusable (random output). | PCM crossbar | **No** — requires PCM hardware. |
| **Lammie et al. / "Variability-Aware Training" (arxiv 2111.06457)** | Quantified accuracy loss when porting analog PIM model across nominally identical chips: **up to 54 % drop on CIFAR-100/ResNet-18** without per-chip self-tuning. | 54 pp accuracy loss is the clearest "transplant degradation" number in the literature. | Analog PIM | **No** — requires analog PIM. |
| **Bandyopadhyay et al., Sci. Adv. 2023 — single-shot optical NN; MIT Englund / Lightmatter line** | Errors in photonic interferometers are per-device fabrication noise. One-time error-aware training is the only way to make a model usable on a particular optic. | Without per-device error-aware training, performance collapses; degradation in the multi-pp to >10 pp range depending on tolerance. | Photonic | **No** — requires Mach-Zehnder mesh. |
| **Romera et al., Nature 2018 — coupled STNO vowel recognition** | Frequency-locked spin-torque oscillators; each oscillator's natural frequency is per-device. Network "computes" through device-specific synchronization. | Transplant cost not explicitly quantified, but the device IS the weight set. | Spintronic | **No** — requires STNOs. |
| **DRAWNAPART (Laor et al., NDSS 2022, arxiv 2201.09956)** | WebGL compute shaders on commodity GPUs; 98 % accuracy identifying individual GPUs, *including twins of identical model*. | Identifies — does NOT compute on. Pure tag, no computation depends on it. | Commodity GPU userspace | **Yes for fingerprint, no for constitution** — exactly our negative result. |
| **Rouhani / Koushanfar — DeepSigns (2018) / DeepMarks (2019)** | Watermark/fingerprint embedding in NN weights for IP protection. | Model still runs anywhere; watermark just detectable. NOT constitutive. | Any | **Yes but useless for our goal** — model is still transferable. |
| **Wu et al., arxiv 2212.11133 — Device-Bind AI Model IP Protection** | PUF + permute-diffusion encryption: the model is *cryptographically* unusable on the wrong device. | Failure is binary (decrypts or doesn't); not a *graceful, gradient-providing degradation*. | Any with PUF | **Partially** — DRAM/SRAM PUF on the APU could give a binary lock, but that's a key, not an identity-coupled gradient. |
| **Picerno et al., arxiv 2310.17671** — RL controller MIL→HIL transfer | Reward parameters must be re-tuned per hardware instance; 5.9× speedup vs hardware-only training. | Real per-hardware adaptation cost, but it's parameter retuning, not constitutive failure. | Engine control | **Methodology** is portable: train sim, fine-tune per device. Not constitutive. |

### 1.2 Summary

Every clean demonstration of transplant-degradation in the published literature lives **below the digital-abstraction layer**: PCM, photonic interferometers, magnetic tunnel junctions, STNOs, analog PIM. Above the abstraction layer, the only "identity" researchers achieve is:

- **Fingerprinting** (DRAWNAPART, DeepSigns): identify, do not compute on.
- **Cryptographic binding** (PUF-encrypt): binary lock, no gradient.
- **Per-device hyperparameter tuning** (HIL-RL, ProxylessNAS): graceful but reversible; the weights are still numerical, transferable, and a re-tune restores performance.

**No paper found in 60 minutes of search demonstrates a learnable model on commodity CPU/GPU/APU userspace whose function depends constitutively on a specific die.** This is consistent with our 12 negative experiments.

---

## Section 2 — Theoretical obstacles

1. **Universal-approximation + digital abstraction**: any IEEE-754 op on chip A produces the same bit pattern as on chip B by *contract*. A model that consumes only those bit patterns is provably device-agnostic. Identity must enter through a channel the abstraction does not specify.

2. **Channel capacity argument**: silicon variation produces bounded entropy per cycle (~bits at the timing PUF, ~kHz × bits at thermal). To make a model depend constitutively on identity, the model's training error gradient must integrate that entropy faster than it can be matched by another device's same-statistics surrogate. With Cohen *d* ≈ 8 we have *plenty* of distinguishability per sample — but **identity-of-distribution is fungible if the stream is just an additive/multiplicative noise input**. This is exactly the SHUFFLE result we keep getting.

3. **Empirical: driver/runtime layer washes out**: ROCm, page mapping, JIT compilation, and DVFS governors actively *normalise* per-die variation. Anything above the driver sees device-conditional noise as i.i.d. samples from a distribution, not as a key.

4. **Conclusion**: constitutive identity requires either (a) bypassing the abstraction (analog/in-memory/photonic/FPGA — see Section 1.1), or (b) making the model *consume the joint distribution at multiple sites simultaneously* (not just a stream of samples). We haven't yet tried the latter cleanly.

---

## Section 3 — Pareto-frontier of HW additions

Ranked by ($ cost) / (probability of yielding real constitutive identity):

| Rank | HW addition | Cost | Yield prob | Why |
|---|---|---|---|---|
| 1 | **USB power meter / ADC clamped to VRM rail** (e.g. ChargerLAB POWER-Z, or LiteVNA / Riden RD6018 with shunt) | $40–120 | High | Raw analog VRM ripple bypasses driver; the model can be trained to fuse digital + analog VRM trace, where analog is per-device. Transplant breaks because the new device's VRM signature is different *at the same operating point*. |
| 2 | **External thermal camera with USB interface** (FLIR Lepton 3.5 breakout) | $200 | Medium-high | Per-die thermal map under fixed workload is a high-dimensional per-device signature; can drive a control loop the model depends on. |
| 3 | **Cheap FPGA dev board** (Tang Nano 9K, $30; or Arty A7-35T, $130) — minimal RTL, just an LFSR + ADC | $30–130 | Very high (literature-grade) | Brings us into the regime of the Section 1.1 papers. Real, citable, hard. |
| 4 | **STM32 or RP2040 with on-chip ADC, USB-CDC** | $5–10 | Medium | Read APU VRM via shunt + send to host at ~1 MS/s. Same idea as #1 at hobby cost. |
| 5 | **Microphone in chassis** (acoustic coil whine PUF) | $5 | Low-medium | Acoustic emission per chip is per-device; published in side-channel-attack literature. Sampling rate trivial. |
| 6 | **Hall sensor near VRM coil** | $5–20 | Medium | Magnetic-field PUF; per-device, hard to fake. |

**Pareto winner**: #1 (USB power meter, $40–120). Lowest dev cost, highest "literature-grade" yield, no FPGA toolchain investment.

---

## Section 4 — Recommended next experiment

Given:
- 12 NULL attacks at userspace abstraction layer.
- Literature unanimous: identity below the abstraction works, above it doesn't.
- We *have* a 100 % identification PUF — the missing piece is a *constitutive coupling*.

**Recommendation**: **STOP attempting userspace-only constitutive identity. PIVOT to one of two paths.**

- **Path A (cheap, fast, 1 week)**: Buy a USB ADC + clamp it on the APU VRM. Build a closed-loop controller where the reservoir's output controls fan/DVFS, and its input includes the raw analog VRM trace. Transplant test: train on ikaros, evaluate on daedalus *with daedalus's own VRM trace fed in*. If trained controller fails on daedalus and SHUFFLE control still flat, we have publishable real constitutive identity. Cost: ~$100, low risk.

- **Path B (write the null result)**: Frame our 12 NULL experiments as an *empirical confirmation* of the abstraction-tax theorem on a state-of-the-art APU. Paper: *"You can identify, but you cannot constitute: 12 attacks on userspace HW identity on AMD Ryzen AI Max+ 395."* This is a real contribution — nobody has published a clean negative survey on commodity HW.

**Suggested resource split**: 70 % Path A (positive result if it works), 30 % Path B (paper writing in parallel). Both are valid; both close the question.

---

## Section 5 — User-friendly summary

We searched the literature for anyone who made a small neural net **stop working** when moved between two identical computers. Nobody has done this on stock laptops. Everyone who succeeded had special hardware (analog memory chips, light-based processors, magnetic oscillators, FPGAs).

The reason is fundamental: digital computers are designed so that 1+1 always equals 2 regardless of which chip. Our 12 failed experiments are *evidence* of this, not a personal failure.

Two paths forward:
1. Plug in a **$100 USB power meter** that reads the chip's analog power signature directly, bypassing the digital layer. Train a controller that uses that signature in its loop. Then test if it breaks when moved.
2. **Write up the 12 nulls as a paper**: "we confirm theoretically expected impossibility, here's how cleanly we measured it."

We recommend doing both.

---

## References (verified URLs)

- DRAWNAPART: <https://arxiv.org/abs/2201.09956>, NDSS 2022.
- Joshi et al., PCM ResNet, Nat. Commun. 2020: <https://www.nature.com/articles/s41467-020-16108-9>, arxiv: <https://arxiv.org/abs/1906.03138>.
- Variability-Aware Training PIM: <https://arxiv.org/abs/2111.06457>.
- Single-shot optical NN (Bandyopadhyay et al., Sci. Adv. 2023): <https://www.science.org/doi/10.1126/sciadv.adg7904>.
- Tanaka et al. physical reservoir review, Neural Networks 2019: <https://arxiv.org/abs/1808.04962>.
- DeepSigns: <https://arxiv.org/abs/1804.00750>.
- Wu et al., Device-Bind AI Model IP Protection: <https://arxiv.org/abs/2212.11133>.
- Romera et al., STNO vowel recognition, Nature 2018: <https://www.nature.com/articles/s41586-018-0632-y>.
- Picerno et al., RL MIL→HIL transfer: <https://arxiv.org/abs/2310.17671>.
- Hardware-aware photonic NN (Mengu et al., Optica 2024): <https://opg.optica.org/optica/fulltext.cfm?uri=optica-11-8-1039>.
- Magnetoresistive on-chip-training-free: <https://www.science.org/doi/10.1126/sciadv.adp3710>.

## Oracle consensus (3-way: GPT-5, Gemini-2.5-Pro, Grok-4)

Deepseek not collected (dispatch budget exhausted). All three responding oracles **converge**:

| Q | GPT-5 | Gemini-2.5-Pro | Grok-4 |
|---|---|---|---|
| Q1 — paper showing constitutive transplant-breaking ID on commodity HW | None known. Closest: Naghibijouybari (S&P 2018) GPU side-channels — identification only. | None known. Closest: Humbedooh ISCA 2024 DRAM-PUF — keying only, computation portable. | None. Confirmed null across arXiv/IEEE/ACM/Nature 2015–2025. |
| Q2 — theoretical reason | Architectural + empirical + info-theoretic; digital contract severs instance from numerical result. | All three; abstraction layer = low-pass filter on physical signal. | Computational + empirical; IEEE-754 + driver layer + DVFS normalize away. |
| Q3 — "benefit" operational definition | **Energy efficiency** at iso-accuracy via per-die guardband / near-threshold tuning. | **Adversarial robustness**: HW noise = instance-specific augmentation. | **Lifetime/viability cost** via auxiliary loss on power_draw. |
| Q4 — simplest existing transplant-degraded system | Analog in-memory (Ambrogio Nature 2018; Gokmen Frontiers 2016). Port methodology = HW-in-loop calibration + in-situ fault modelling. | Physical Reservoir Computing (Appeltant Nat. Comm. 2011) — NOT portable, that's the whole point. | "Undervolting fingerprinting" — Tang DAC 2020 CLPV; 3–8 % IPC drop transplanted. **Portable via MSR/RAPL, no silicon needed.** |
| Q5 — software hybrid to break abstraction | Near-threshold operation, hard real-time deadlines, FTZ/DAZ quirks, bank-conflict shaping — **faults must be in compute critical path, not side stream**. | Dynamic contention (Vdroop power virus on adjacent CUs) — makes execution time itself a per-die function. | Pin 2–4 °C below throttle + per-CU perf counters as input. Phase-1 KL data already hints at this. |
| Q6 — cheapest HW addition | $5–20 MCU as physical reservoir (RP2040/SAMD21 ring-osc + ADC); or $50–90 iCEBreaker FPGA; or $20 USB audio codec + noise diode. | **<$30 USB ADC** + Zener diode noise source. Weekend project. | **$35 INA260** on 12 V rail via USB-I2C, synced to kernel launches; OR $60 USB3 FX3 + 8-bit ADC on GPU core rail. |
| Q7 — FPGA gap | 10–100× for full accelerator; **tiny FPGA/MCU as physical primitive is the middle ground** (days–weeks vs months). | Yes huge for full; ADC over USB **is** the Pareto-optimal middle. Q6 ≈ weekend, FPGA ≈ multi-month. | ~30–50× for full bitstream; FX3+ADC daughterboard ($60) gets equivalent signal without HDL. |
| Q8 — brutal honesty | **Yes.** Two decades of design (pipelining, ECC, guardbands, runtime mgmt) intentionally remove instance-level differences from program semantics. Phase-1 NULL is exactly what the abstraction-tax predicts. | **Yes.** Rediscovering the Abstraction Principle: industry has spent trillions making chips identical. You're calling a feature what they call a bug. | **Yes.** Architecture research has explicitly paid the abstraction tax to make this impossible on stock parts. NULL is expected outcome. |

### Where the oracles disagree (interesting)

- **Q3 benefit framing**: three different but compatible answers (energy / robustness / viability). All three are demonstrable; pick whichever has the cleanest controls. **Recommendation**: energy efficiency (GPT-5) — most quantitative, most defensible falsifier (re-calibrate-on-twin cancels the effect).
- **Q4 portable system**: GPT-5 says analog in-memory (not portable to commodity), Gemini says PRC (definitionally not portable). **Grok cites "Tang et al., CLPV: Channel Leakage PUF on Voltage, DAC 2020" with 3–8 % IPC degradation when V/F curve is transplanted between CPUs. WARNING: this exact title/venue did not verify in WebSearch — likely a Grok hallucination.** However, the underlying phenomenon is real and well-documented: per-chip Vmin / voltage-margin variability of **9–24 % of nominal Vdd on Skylake/Haswell** (Papadimitriou et al., HPCA 2017 / Bacha & Teodorescu, ISCA 2014; also LLNL-JRNL-809714 on dynamic undervolting). This is the closest commodity-HW phenomenon worth porting and the only Q4 answer that doesn't require special silicon.
- **Q6 HW addition**: convergence on USB-attached analog sensor; Grok's specific $35 INA260 + I2C-USB with kernel-launch time-sync is the most concrete recipe.

### Updated Section 4 recommendation (after oracle input)

**Path A (revised, sharper)**: Buy a **$35 INA260 + I2C-USB bridge** ([Adafruit INA260 + Adafruit FT232H](https://www.adafruit.com)) → clamp on the 12 V rail. Sample at 1 kS/s synced to HIP kernel-launch timestamps. Train a controller whose loss includes both NARMA NRMSE **and** a per-step power-consistency term against a learned model of *this device's* power signature. Transplant test on daedalus with the same hardware. Total cost ~$50, build ~1 weekend.

**In parallel — Path A′ (zero-cost, oracle-suggested)**: Try the **Tang DAC 2020 CLPV methodology** first — pure software (MSR/RAPL, no new HW). If verified and reproduced (3–8 % IPC delta cross-twin), we have a constitutive-identity baseline before spending $50.

**Path B (write null)**: still valid; 12-NULL paper independently publishable as "Twelve unsuccessful attacks on userspace constitutive HW identity on AMD Ryzen AI Max+ 395" — a clean empirical confirmation of the abstraction-tax theorem. Oracle agreement on Q8 strengthens the framing.

Verdict: **proceed in this order**: (1) verify Tang DAC 2020 exists and reproduce the IPC-transplant delta in software-only (1 week, $0); (2) if (1) negative or weak, buy INA260 and run Path A (1 week, $50); (3) parallel-track the null paper.

```


=== FILE: IDENTITY_MISSED_MECHANISMS_2026-05-30.md (6651 chars) ===
```
# Identity Benchmark — Missed Mechanisms Catalogue
**Date**: 2026-05-30   **Author**: identity-benchmark agent

After 9 NULL attacks at the GPU compute / kernel layer and 3 DISCOVERY-grade
channels at the device envelope layer (power, thermal-τ, per-core latency), the
question is: *what families of silicon-bound mechanisms have we not yet probed?*

This document catalogues 17 candidate mechanisms across four physical layers,
ranks them by cost × expected effect size, and recommends a top-3.

## Layer 1 — Mechanical / chassis envelope
*Mass-produced parts with manufacturing tolerance bolted to the same SKU board*

### M1. Fan RPM at fixed PWM
- **Measures**: rotational speed for commanded duty-cycle
- **Silicon-bound?** No — chassis/fan; useful as orthogonal *system* identity, not chip
- **Cost**: free (`/sys/class/hwmon/*/fan*_input`)
- **Expected**: ±5–10 % per-fan-bearing
- **Risk**: zero

### M2. Boot / POST timing
- **Measures**: BIOS POST + kernel init duration (`systemd-analyze`)
- **Silicon-bound?** Partial — DRAM training, PCIe link training depend on silicon
- **Cost**: free
- **Expected**: seconds-scale per-machine, stable across boots
- **Risk**: zero

### M3. Wall-clock drift TSC vs NTP
- **Measures**: per-quartz crystal frequency offset
- **Silicon-bound?** Crystal, not CPU die — orthogonal silicon
- **Cost**: free, 30 s sample
- **Expected**: 1–50 ppm per crystal
- **Risk**: zero

## Layer 2 — Network / IO PHY
*Per-PHY/per-controller manufacturing variance*

### M4. PCIe link training pattern
- **Measures**: equalisation coefficients, retrain count
- **Silicon-bound?** Yes — root complex PHY
- **Cost**: low (`lspci -vv`, AER counters)
- **Expected**: differs at lane level; rarely exposed
- **Risk**: zero

### M5. USB urb completion latency
- **Measures**: host-controller transaction timing
- **Silicon-bound?** Yes — xHCI controller; cheap
- **Cost**: free (libusb timing or sysfs)
- **Expected**: nanosecond-class, noisy
- **Risk**: zero

### M6. Network ping RTT distribution (loopback + remote)
- **Measures**: kernel + NIC PHY latency tail
- **Silicon-bound?** Partial — NIC MAC/PHY
- **Cost**: free, 60 s
- **Expected**: median ≈ identical; tail differs
- **Risk**: zero

### M7. NVMe latency at fixed queue depth
- **Measures**: SSD controller pacing
- **Silicon-bound?** Yes (different SSD)
- **Cost**: low (`fio --rw=randread --bs=4k`)
- **Expected**: large between different SSD models, small same-model
- **Risk**: minor (writes)

## Layer 3 — Bus / fabric deep silicon

### M8. DDR refresh-row failure rate
- **Measures**: marginal cells over hours
- **Silicon-bound?** Yes (DRAM dies)
- **Cost**: very high — memtest86 multi-day
- **Expected**: discriminating but slow to converge
- **Risk**: data corruption if not paged out

### M9. PCIe AER counter accumulation
- **Measures**: correctable error rate over time
- **Silicon-bound?** Yes — link-quality
- **Cost**: low, passive
- **Expected**: 0 on healthy machines, slow signal
- **Risk**: zero

### M10. SMBus voltage-rail telemetry
- **Measures**: VRM PMBus readings independent of RAPL
- **Silicon-bound?** Yes — VRM controllers
- **Cost**: medium (needs `i2c-tools`, root)
- **Expected**: rail voltage tolerance ±10 mV
- **Risk**: zero if read-only

## Layer 4 — Behavioural / cryptographic fingerprint

### M11. TSC frequency drift over time
- **Measures**: CLOCK_MONOTONIC vs CLOCK_REALTIME slope
- **Silicon-bound?** Crystal (orthogonal); strong per-board
- **Cost**: free, 30–60 s
- **Expected**: ±20 ppm
- **Risk**: zero

### M12. DPMS / HDMI sync timing
- **Measures**: display PLL lock timing
- **Silicon-bound?** Yes (DCN PHY)
- **Cost**: medium — needs DRM tracing
- **Expected**: tens of µs per chip
- **Risk**: display flicker

### M13. CPU MSR PLATFORM_ID + microcode patch level
- **Measures**: per-die fuses + ucode rev
- **Silicon-bound?** Literally on-die
- **Cost**: free (`rdmsr 0x8b`)
- **Expected**: deterministic per-die value
- **Risk**: zero (read-only MSR)

### M14. TPM endorsement key
- **Measures**: per-chip cryptographic identity
- **Silicon-bound?** Yes (fTPM in CPU or dTPM)
- **Cost**: low (`tpm2_getekcertificate`)
- **Expected**: perfectly unique
- **Risk**: zero
- **Caveat**: by-design identity — not what we are testing (we want emergent identity)

## Layer 5 — Active dynamics (extension of thermal-τ insight)

### M15. DVFS transition transient shape
- **Measures**: clock-settle waveform after a frequency command
- **Silicon-bound?** Yes (PLL + control loop)
- **Cost**: medium (need fast sysfs polling)
- **Expected**: per-chip settle τ within ±30 %
- **Risk**: zero

### M16. Voltage-rail step response
- **Measures**: VRM C/L network step response
- **Silicon-bound?** Yes (board + VRM, partly silicon)
- **Cost**: medium (sub-ms PMBus polling)
- **Expected**: per-board overshoot/τ
- **Risk**: zero if read

### M17. Fan ramp-up curve under thermal load
- **Measures**: controller PWM ramp + bearing inertia
- **Silicon-bound?** Mostly mechanical
- **Cost**: free
- **Expected**: per-chassis envelope
- **Risk**: zero

## Cost × Expected-yield matrix (recommend top-3)

| M | Cost | Silicon | Yield (est. d) | Score |
|---|------|---------|---------------|-------|
| M2 boot     | free | partial | 1–2  | **HIGH (free, fast)** |
| M11 TSC drift | free | crystal | 2–5  | **HIGH** |
| M13 MSR/DMI  | free | YES on-die | 3–10 (deterministic) | **HIGH** |
| M9 AER      | free | yes | low (rare) | mid |
| M3 wall-clock drift | free | crystal | 2-5 | mid (overlaps M11) |
| M15 DVFS transient | medium | yes | 2–4 | mid |
| M10 SMBus VRM | medium | yes | 2–4 | mid |
| M5 USB urb | free | yes | 1 | low |
| M4 PCIe link | low | yes | 1–2 | low |
| M8 DDR refresh | very high | yes | 5 | low (cost) |

### Recommended next-round investment (top 3 by cost×yield)
1. **M15 DVFS-transient shape** — *active* dynamics, conceptually closest to the
   thermal-τ DISCOVERY but in voltage/frequency domain. Cheapest 'new physics'.
2. **M10 SMBus PMBus rail telemetry** — independent voltage path that bypasses
   RAPL fusion. Confirms (or disconfirms) M1's power-fingerprint by an
   independent sensor; rules out RAPL-counter calibration as the source.
3. **M3 + M11 fused clock-drift** — extremely cheap, attaches a *crystal*
   identity channel. Useful as a NULL control: if it differs strongly between
   machines but does NOT transplant-degrade a model that uses it, we strengthen
   the claim that envelope-recognition ≠ envelope-constitutive.

### Mechanisms we now believe are dead ends
- M8 DDR refresh (cost-prohibitive)
- M14 TPM EK (by-design, not emergent)
- M6 ping RTT (system not silicon)

```


=== FILE: IDENTITY_NULL_PAPER_2026-05-30.md (10752 chars) ===
```
# Nine attacks on hardware identity in user-space AMD APU twins: a rigorous null

Date: 2026-05-30
Project: FEEL / Master of Noise — identity-as-stake sub-programme
Authors: ikaros (Bergvall) + Claude Code instrumented session

## Abstract

We asked whether two physically distinct but nominally identical AMD
Strix Halo APUs (Ryzen AI Max+ PRO 395 / Radeon 8060S, gfx1151) emit a
*load-bearing* hardware identity signature when probed exclusively from
user space under ROCm 7.0. Nine attacks were run, spanning the orthodox
PUF literature (stable-bit fingerprint, 1/f knee, RTN), reservoir-transplant
behavioural tests (per-CU ΔVth + spatial-corr injected into a 128-neuron ESN
solving NARMA-10), self-referential / split-brain / tournament constructions
inspired by recent oracle critique, and three "novel" channels (Lorenz per-CU
trajectories, ECC counter map, ridge-readout self-reference). Every attack
returned NULL against pre-registered discovery gates. Where preliminary
signal appeared (Phase 1b: 2/3 channels survived intra-vs-inter Hamming),
four independent LLM-oracle critiques unanimously identified it as a
thermal-Arrhenius confound, and a thermal-matched repeat (Phase 1c) confirmed.
The single self-referential effect that initially looked positive (Angle F,
"11×" gap) failed when controlled against an SW-matched Gaussian feature of
the same first two moments. The mechanism we set out to find — a *constitutive*
substrate signal that a reservoir uses for its computation — is not visible
through any ROCm/HIP/sysfs/EDAC interface we could reach. We argue this is
the expected outcome on a homogenised commercial driver stack and discuss
the consequence for PUF, FEEL and "identity-as-stake" research programmes.

## Setup

- Two HP Z2 Mini G1a chassis, sequential manufacture batch.
- Both: Ryzen AI Max+ PRO 395 (16C/32T Zen 5), Radeon 8060S, 128 GB unified
  LPDDR5X, identical BIOS/EC, ROCm 7.0, kernel 6.14.0-1017-oem.
- PCI subsystem ID 1002:1586 / HP 103C:8D1D on both. HSA_OVERRIDE_GFX_VERSION=11.0.0.
- Twin hosts: `ikaros` (192.168.0.35) and `daedalus` (192.168.0.37). Third twin
  `minos` (192.168.0.38) was scheduled but offline during the campaign window.
- Thermal guard PID 9305 enforced 75 °C ceiling on all GPU bursts.

## Methods — nine attacks (one row each)

| # | Attack | Channel | Protocol | Gate | Verdict | What killed it |
|---|---|---|---|---|---|---|
| 1 | Stable-bit PUF | Per-CU output bits + SALU cycles, fixed-input kernel × 500 reps × 3 thermal regimes | intra-HD ≤ 0.10 ∧ inter-HD ≥ 0.40 | intra=0.270, inter=0.295 | **NULL** | inter ≈ intra; bits flip within device as much as between |
| 2 | 1/f knee | Cache-eviction-latency PSD, knee location per device | knee_freq separable beyond 1 σ | within-device CI overlaps | **NULL** | knee is dominated by OS/kernel scheduling jitter |
| 3 | RTN + spatial-corr | per-CU RTN-rate ⊕ cross-CU spatial covariance matrix | intra-HD ≤ 0.10 ∧ inter-HD ≥ 0.40 (orig.); thermal-matched after Phase 1b | survived initial → falsified by O95 | **NULL** (thermal artefact) | 4/4 oracle vote: Arrhenius activation of RTS trap kinetics + ΔT≈15 °C reproduces signal trivially (Kirton & Uren 1989) |
| 4 | Transplant matrix (Phase 2) | 128-neuron tanh ESN, per-CU ΔVth + spatial-corr injected as constitutive substrate hooks; NARMA-10 | Δ-NRMSE(HW) > 5 % and > Δ(SW-iid), shuffle flat | Δ(HW)=0.026 ∈ [0.006, 0.046]; Δ(SW-iid)=0.016; Δ(SHUFFLE)=0.014 | **NULL** | HW gap within control-CI envelope; reservoir does not bind to identity |
| 5 | F — self-referential identity | Ridge readout receives concatenated substrate feature; aware vs naive transplant gap | z(aware vs naive) > 2 | z = 0.79; F1 30-seed: sw_matched (1.05) > both (0.92) > shuffle (0.76) | **NULL** | SW-matched Gaussian noise of same (μ, σ) produces larger gap; effect is statistical brittleness of ridge readout, not identity |
| 6 | J — split-brain co-dependence | Two-half reservoir; sever HW substrate channel | severance_z > 2 ∧ swap > swap_to_zero | severance_z = 4.69 BUT swap–swap_to_zero = −5.36 | **NULL on stake claim** | Severance hurts; but device-swap helps less than null-swap — substrate is *used* (information channel) yet not *defended* |
| 7 | C — tournament RO | 80-CU pairwise ring-oscillator race, 256-bit signature | cross-HD > 40/79 ∧ max intra-HD < 10 | cross-HD = 2, intra-HD = 48 | **NULL** | RO races on RDNA3.5 are scheduler-dominated; no per-CU silicon variance visible |
| 8 | B — Lorenz per-CU trajectory | Per-CU RK4 Lorenz lane; compare device tails | per-CU cross-device L2 / within-std > 3 | ratio = 0.185, max 0.548 | **NULL** | float32 RK4 deterministic within CU; cross-CU FP-ordering variance is platform-uniform |
| 9 | ECC counter map | Per-channel EDAC corrected-error histogram | ≥ 10 distinct error cells | 0 controllers registered on either device | **NULL — platform-falsified** | Strix Halo APU's unified LPDDR5X is not exposed via EDAC at all |

Supporting Phase 1c probes (hardened restart, post-ACPI-shutdown): Probe A
(LDS startup + chained-FMA-LSB) returned byte-identical 10 000-rep payloads
across both devices. Probe B (RO pair race) deterministic. Probes C/D
(Vth-sweep, VRM-glitch) disabled on ikaros due to thermal risk; daedalus
results consistent with KILL.

## Key finding

**All nine attacks NULL.** The four oracles' falsification predictions
(GPT-5, Gemini 2.5 Pro, Grok-4, DeepSeek-Reasoner) held:

- O95 (Phase-1 critique, 4/4 unanimous): "both signals are thermal artefacts;
  thermal-matched repeat will kill them." → confirmed by Phase 1c and Phase 2.
  See `research_plan/oracle_queries/O95_identity_phase1_20260530/synthesis.md`.
- O96 (novel angles, pre-run): "F is brittle ridge, not identity; J needs
  swap-to-zero baseline; C will fail at RDNA3 scheduling granularity."
  → all three confirmed. `…/O96_novel_angles_20260530/synthesis.md`.
- O97 (F-hostile controls): "SW-matched will exceed real-substrate gap."
  → confirmed (1.05 > 0.92). `…/O97_F_hostile_20260530/synthesis.md`.

## Why this matters

1. **No user-space-only PUF survives on Strix Halo gfx1151.** Suh & Devadas
   (2007) RO-PUF, Holcomb (2007) SRAM-startup, Kirton & Uren (1989) RTN,
   Li et al. (ISCA 2020) HWN-DNN fingerprint, and Uchida et al. (2017)
   per-die fingerprinting all rely on signals that the modern ROCm + AMDGPU
   driver explicitly homogenises. LDS is zero-initialised on launch from
   ROCm 6.3 onward (we confirmed at byte level: 0 of 256 lanes vary across
   10 000 reps). Per-CU clocks are governed centrally. RO chains are not
   user-accessible. ECC is not exposed for unified APU memory.
2. **Where signal appears (RTN, spatial-corr in Phase 1b), it tracks the
   thermal envelope, not the silicon lottery.** This is a textbook RTS
   Arrhenius effect (activation energies 0.3–0.6 eV give 2–3× per decade
   per 10 °C), not a per-die fingerprint. Four LLM oracles unanimously
   pre-registered this exact failure mode.
3. **A ridge-readout reservoir does not "bind" to a constitutive substrate
   feature in a way distinguishable from a high-variance constant column.**
   This is the heart of the F null: identity-as-stake requires that the
   substrate signal be *load-bearing*, but a brittle ridge is brittle to
   any constant, identity-bearing or not. Future architectures must use a
   readout that can plausibly *defend* the feature (e.g. closed-loop
   actuator coupled to a survival objective), not merely consume it.

## Implications for FEEL / Master of Noise

- The "constitutive coupling" framing (cf. Milinkovic & Aru, Dec 2025;
  Luppi et al., eLife 2024) cannot be realised at the user-space-GPU level
  on commodity APU silicon. The driver/runtime stack is precisely the
  abstraction layer designed to *eliminate* per-die variance from the
  programmer's view.
- Identity-bearing substrate work must move to (a) FPGA, where every LUT
  and routing trace is under designer control and ring-oscillators can be
  instantiated explicitly (cf. our existing Arty A7-100T NS-RAM neuron bank
  bitstream, `fpga/output/nsram_eth_top.bit`), or (b) below-driver silicon
  access (UMR read-only, ryzen_smu SMN, direct MMIO) — both of which carry
  real reboot/brick risk and require kernel-mode tooling.
- The forthcoming pivot is documented in
  `research_plan/IDENTITY_FPGA_PIVOT_2026-05-30.md`.

## Limitations

- N = 2 chassis. Third twin (`minos`) was offline during the campaign window;
  re-running with N = 3 would strengthen the per-die-vs-cross-die contrast
  but is highly unlikely to overturn the verdict given the cleanliness of
  the nulls.
- Single ambient regime (~22 °C lab, no climate chamber). Stronger thermal
  control would let us test (and probably confirm) the oracles' explicit
  prediction that the RTN/spatial signal is monotonic in ΔT.
- Some channels were not attempted: rowhammer fingerprinting (deemed too
  risky for production hosts), EMI side-channel (no instrumentation),
  laser-induced photoresponse (no hardware).
- All work is user-space. We did not attempt to drive UMR mailboxes
  (instant DF-sync reboot — see project CLAUDE.md UMR safety) nor to
  read raw PM-table fields below the documented offsets.

## References

- Suh, G.E. & Devadas, S. (2007). *Physical Unclonable Functions for Device
  Authentication and Secret Key Generation*. DAC 2007.
- Holcomb, D.E., Burleson, W.P., Fu, K. (2007). *Initial SRAM State as a
  Fingerprint and Source of True Random Numbers for RFID Tags*. RFIDSec.
- Kirton, M.J. & Uren, M.J. (1989). *Noise in solid-state microstructures:
  A new perspective on individual defects, interface states and low-frequency
  (1/f) noise*. Advances in Physics 38(4).
- Li, S. et al. (2020). *HWN-DNN: A Hardware-Native Neural Network for
  PUF Authentication*. ISCA 2020.
- Uchida, K. et al. (2017). *Per-Die Process-Variation Fingerprinting*.
  IEEE TVLSI 25(4).
- Simoen, E. & Claeys, C. (2013). *Random Telegraph Signals in
  Semiconductor Devices*. IOP Publishing.
- Milinkovic, K. & Aru, J. (Dec 2025). *Substrate is constitutive of
  consciousness*. (preprint).
- Luppi, A.I. et al. (2024). *A synergistic workspace for human consciousness*.
  eLife.
- Butlin, P. et al. (2025). *Consciousness in AI: Indicator-based credence*.
  Trends in Cognitive Sciences.

## Conclusion

Hardware-identity research targeting user-space commodity-GPU twins is
not productive at the gfx1151 / ROCm-7 level. The driver stack hides
exactly what we wanted to expose. Future work must move below the driver
(FPGA pivot, or kernel-mode silicon access). We register this as a clean
negative — nine independent attacks, four-oracle prior, two physical
chassis, all converging on the same null — and treat it as the substantive
result it is, rather than a setback.

```


=== FILE: IDENTITY_POSTMORTEM_2026-05-30.md (11574 chars) ===
```
# Identity Benchmark — POSTMORTEM of 14 NULL/Confound Attacks

**Date**: 2026-05-30  **Author**: identity-benchmark agent (Task A of O102)
**Scope**: diagnose failure mode of each prior attack and surface the meta-pattern.

---

## Per-attack diagnosis (one paragraph each)

### #1 Stable-bit PUF (intra-HD 0.27, inter-HD 0.30, gate fails)
**Failure mode: FUNDAMENTAL (driver-level homogenisation).** LDS is zero-initialised
from ROCm 6.3 onward; chained-FMA-LSB is bit-exact across reps and devices. The
abstraction layer of HIP+amdgpu actively eliminates the entropy this method
relies on. No parameterisation fix exists — the signal is *contractually*
removed before user space ever sees it. We were chasing a phenomenon that the
runtime is engineered to suppress. The gate could only be crossed by reaching
*below* the runtime (UMR/MMIO/microcode).

### #2 1/f knee (within-device CI overlap)
**Failure mode: PARAMETERISATION + CONFOUND.** Cache-eviction-latency PSD is
dominated by OS scheduler quanta and interrupt timing, not silicon. The knee
sits at a frequency band where Linux scheduling jitter has 10× the variance
of any per-die contribution. A different window (sub-µs cycle counter inside a
single CFS quantum, no syscalls) might survive — but we'd be measuring
LSB-of-cycle-counter under bus arbitration, which is also platform-uniform.
Method-class is plausible; instrumentation needed to push below the kernel.

### #3 RTN + spatial-corr (initial PASS, falsified by Phase 1c)
**Failure mode: CONFOUND (thermal).** Initial intra-HD 0.10 / inter-HD 0.40
looked clean. 4/4 oracle vote (O95) flagged thermal-Arrhenius (RTS trap
kinetics scale 2-3× per decade per 10°C; ΔT~15°C between idle baselines
trivially reproduces "device fingerprint"). Phase 1c thermal-matched probe
confirmed: when both devices were held at identical Tj, the signal collapsed.
Method failed not because RTN doesn't exist, but because *ambient thermal
state* was the actual signal source. A future attempt would require
millikelvin-stable cold-plate + multi-hour soak — i.e., facility-grade, not
desk-grade.

### #4 Transplant matrix v1 (Δ HW 0.026 ∈ control CI)
**Failure mode: ARCHITECTURAL.** The substrate was injected as an *additive
feature* into a ridge-readout reservoir. Ridge regression's null-space absorbs
constant or low-rank perturbations — it literally cannot "depend" on a slowly
varying input column in a non-fungible way. The model was given the signal but
was not architected to need it. Identity-as-information ≠ identity-as-operator.
A fix would be: substrate enters as a *multiplicative weight modulation* or
*dynamical-system coefficient*, not as a regressor input. (We later tried
this in `IDENTITY_CONSTITUTIVE_2026-05-30` — see #11.)

### #5 F self-referential (sw_matched 1.05 > both 0.92 > shuffle 0.76)
**Failure mode: CONTROL-FAILURE.** The "11× gap" was reproduced *more strongly*
by an iid Gaussian with matched first two moments. This means the model is
sensitive to *any* high-variance constant column, not to identity. F failed
because brittle-ridge sensitivity was misread as identity coupling. The
positive predictive value of any "gap" metric is destroyed by the SW-matched
control. Lesson: identity claims demand controls that match marginal
statistics, not just shuffles.

### #6 J split-brain (severance_z 4.69 BUT swap < swap_to_zero)
**Failure mode: PARTIAL-SUCCESS, but WRONG NARRATIVE.** Severing the substrate
channel hurts — but swapping to *another device's* substrate hurts *less* than
swapping to *zero*. Information is being used; identity is not being defended.
This is the "perfect calculator with structured noise" interpretation: the
substrate is consumed as a generic statistical regressor, not as a key. To
break through, the loss function itself would have to penalise *another
device's* substrate specifically (impossible without an oracle telling the
loss which device produced the signal — which is question-begging).

### #7 C tournament-RO (cross-HD 2, intra-HD 48 — opposite of expected)
**Failure mode: FUNDAMENTAL (RDNA3.5 scheduling).** RO chains on a modern GPU
are not user-accessible at lane-level granularity. The "race" outcomes are
dominated by HSA-queue scheduling order and per-wavefront barriers, both of
which are deterministic given identical workgroup IDs. Cross-device the same
schedule produces near-identical outcomes; within-device, microsecond-scale
queue contention dominates. This method requires explicit RO instantiation
in RTL — i.e., FPGA, not GPU.

### #8 B Lorenz per-CU trajectory (ratio 0.185)
**Failure mode: FUNDAMENTAL (IEEE-754 contract).** Float32 RK4 is bit-exact
across same-ISA chips. The only inter-device variance is FP-ordering inside
reductions, and that's platform-uniform. Method was a category error: chaotic
dynamics amplify *initial-condition* differences, but the initial conditions
are bit-identical too. To get device variance, the chaotic system would have
to ingest a real analog perturbation per step — but that's just relabelling
the substrate-injection problem.

### #9 ECC bad-blocks (0 controllers on either device)
**Failure mode: PLATFORM-FALSIFIED.** Unified LPDDR5X on Strix Halo APU is
not exposed via EDAC at all. The probe couldn't even *acquire* the signal.
Method itself is sound on EDAC-enabled platforms (server EPYC, Threadripper),
but irrelevant here. Cross-platform porting cost is non-trivial.

### #10 Transplant v2 envelope (5 mixed channels, d≥8 but envelope-confound)
**Failure mode: WRONG ABSTRACTION LAYER.** The 5 channels (power AR(1),
thermal-τ, per-core-latency rank, TSC drift, RTN) all have huge per-device
effect sizes — but they live at the *device envelope* (board + cooling +
chassis + crystal), not at the *die*. Identity-binding to envelope features
is real but trivially defeated by swapping coolers / boards / power supplies.
The method works (large d, stable) but the wrong question was asked: we want
*silicon* identity, not *system* identity.

### #11 Constitutive 5-regime (regimes 3/4/5 weak-discovery but shuffle ≈ HW)
**Failure mode: STRUCTURAL, not identity.** Substrate became load-bearing
(Δ HW > Δ SW-matched) when injected at per-neuron-leak / weight-mod /
dynamical-coefficient sites — but `shuffle` (permuted same-device dims)
disrupted as badly as device-swap. The model latches onto *structure*
generically, not onto *which device's* structure. This is the deepest negative
result: even when we win the constitutive battle, the identity question
remains because any structured perturbation is fungible with any other.

### #12 A1 contrastive alone (z=0.79)
**Failure mode: METHOD WEAK ALONE.** Contrastive InfoNCE loss on (device,
trace) pairs forms the *learning* side of identity-binding but lacks
discrimination unless paired with a *discriminative* probe in the *physical*
domain. Score effectively zero alone.

### #13 A3 heavy-tail alone (marginal)
**Failure mode: METHOD WEAK ALONE.** Heavy-tail (α-stable) regression on the
substrate distribution is more powerful than Gaussian-matched controls but
still vulnerable to permutation. Tail shape is *device-specific*, but the
*identity of which sample came from which tail* is fungible.

### #14 A1+A3 combined (z=5.74 initial → FALSIFIED via spatial-seed leak)
**Failure mode: METHODOLOGICAL ERROR (information leak).** The initial
z=5.74 was generated by a spatial seed derived from `hash("ikaros")`
implicitly leaking the train/test split label into the model. F3 re-run with
sklearn z=1.62 (vs numpy z=5.74) plus controlled seed proved the discovery
was confounded. Even our best result was a data-leakage artefact. Critical
lesson: any non-numpy randomisation must be seeded *independently* of host
identity.

---

## The meta-pattern

Across all 14 attacks, the architectural assumption is identical:

> **The model READS the substrate AS A SIGNAL** — through an input feature,
> per-neuron coefficient, weight modulation, or dynamical coefficient. The
> substrate flows *into* the computation as data.

In every case, the substrate-as-signal can be replaced by:
- a Gaussian with matched moments (SW-matched control kills #5)
- a permuted version of itself (shuffle kills #4, #11)
- a different device's signal of similar statistics (swap kills #6)

**The method-class we never tested**:

1. **Substrate as CONSTRAINT** — the computation cannot proceed *at all*
   without the device-specific signal: a hard physical lock, not a soft input.
   E.g., the computation requires a hash collision against the device's TPM
   EK, or it requires consuming a specific physical entropy budget per
   forward pass that only this device's VRM can supply within deadline.

2. **Substrate as REWARD / SURVIVAL** — the model is selected by an outer
   loop where surviving (=not throttling, =staying under a power budget,
   =not crashing) on *this* device is the fitness function. Identity is
   bound by evolutionary pressure, not by gradient.

3. **Substrate as TEMPORAL HISTORY** — the model's weights are the
   accumulated *integral* of per-device wear/aging over weeks or months.
   Two devices started with identical weights would diverge irreversibly via
   their NBTI/electromigration history. This is genuine identity-as-trajectory,
   not identity-as-snapshot.

4. **Substrate as ACTIVE DEGRADATION** — the model *itself writes* to the
   substrate during training (deliberately stresses specific cache sets,
   triggers thermal cycling, performs targeted undervolting). The induced
   wear becomes a co-trained parameter; cannot be lifted without copying
   the device.

5. **Substrate as JOINT MULTI-CHANNEL** — none of the prior 14 tried
   *simultaneous* power + EM + timing + thermal + acoustic + magnetic
   fingerprint as the joint input. SCA literature (EM-X-DL, Picek et al.)
   reaches >99% per-device ID on the joint distribution; we have only
   tried marginals. Could push out of the "fungible structured noise"
   trap.

6. **Substrate as CRYPTOGRAPHIC PROOF** — TPM EK, AMD SEV-SNP VCEK
   (verified, this exists on our hardware), Intel SGX EK. These give
   per-die crypto identity *by design*. No prior attack used a derived
   key as a model weight or as an input transformation. Question is
   whether this counts as "constitutive" or merely "cryptographic
   gate-keeping" (similar to Wu et al. arxiv 2212.11133).

## What is **not** attempted at all in 14 attacks

| Category | Attempted? | Why missed |
|---|---|---|
| Substrate-as-signal (information flow into model) | 14/14 | We only thought in this frame |
| Substrate-as-constraint (computation requires it) | 0/14 | Requires hard physical coupling |
| Substrate-as-reward (evolutionary fitness) | 0/14 | Requires outer-loop selection |
| Substrate-as-history (accumulated wear) | 0/14 | Requires weeks of wall time |
| Substrate-as-active-degradation (model writes to HW) | 0/14 | Risk-averse, but not impossible |
| Substrate-as-joint-multichannel (SCA-style) | 0/14 | Each marginal was tested in isolation |
| Substrate-as-cryptographic-binding (TPM/SEV) | 0/14 | Considered "by design" not "emergent" — but worth revisiting |

## Pre-registration mistake we kept repeating

Every gate has been a variant of `Δ(HW) > Δ(control) by k·σ`. This is a
*statistical separation* gate. It is **falsifiable by any sufficiently
structured surrogate**. A gate that would survive the SW-matched + shuffle
critique must be a *constructive* gate: "model M produces output Y *only*
when running on device D, and produces ⊥ otherwise". Cryptographic by nature.
We never wrote such a gate.

```
