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


=== FILE: IDENTITY_DEEP_2026-05-30_REPORT.md (6876 chars) ===
```
# IDENTITY DEEP — Report
Date: 2026-05-30
Hosts: ikaros (AMD gfx1151, card1), daedalus (AMD gfx1151, card0)
Battery: 5 angles built, 4 ran on both machines, 1 (NPU) blocked at userspace.
Analysis: results/IDENTITY_BENCHMARK_2026-05-30/deep/ANALYSIS.json

## Verdict table
| Angle | Built | Ikaros | Daedalus | Headline | 95% CI | Gate |
|-------|-------|--------|----------|----------|--------|------|
| A — Power fingerprint | Y | Y (5 reps×5 s × 4 wl) | Y | IDLE 19.2 W vs 6.1 W; MEDIUM 110.3 vs 60.1; HEAVY 109.5 vs 62.4 | diff CI excludes 0 in IDLE / MEDIUM / HEAVY; Cohen d ≥ 8 | **DISCOVERY** (3 of 4 workloads pass; LIGHT cohen d=4.0 but std overlaps gate threshold) |
| B — Thermal time constant | Y | Y (6 cycles) | Y | τ_heat ikaros = 4.33 s vs daedalus 1.26 s; R_th ikaros 0.311 K/W vs 0.482 K/W | τ_heat diff CI [2.78, 3.53] s; R_th CI [-0.177, -0.165] K/W | **DISCOVERY** (Cohen d=7.7 on τ_heat, d=−30.5 on R_th) |
| C — NPU XDNA | Y (recon) | recon only | recon only | /dev/accel/accel0 + amdxdna loaded, no XRT userspace | n/a | **BLOCKED** |
| D — DPM Vmin sweep | Y | Y (low/auto/high × 60 reps) | Y | **zero** bit flips on either device at any DPM level | timing differs 1.78 ms (ikaros faster), CI [1.34, 2.56] | **AMBIGUOUS** (no Vmin signal; timing differs but reflects DPM scheduler not silicon) |
| E — CPU per-core | Y | Y (16 cores × 4 repeats) | Y | ikaros per-core time spread 2.67 ms (8.85–11.52 ms); daedalus 0.19 ms (8.62–8.81 ms); identical sysfs max-freq 5187 MHz both hosts | mean time diff CI [+1.65, +2.45] ms (ikaros slower); rank-correlation across cores = −0.51 | **DISCOVERY** (Cohen d=3.37 on per-core mean; per-core ranking anti-correlated → distinct silicon orderings) |

## Headline numbers (95 % bootstrap CI)
- **Power IDLE diff (ikaros − daedalus): +13.13 W, CI [+11.32, +14.74]** — daedalus 6 W idle, ikaros 19 W (3.1×).
- **Power MEDIUM diff: +50.18 W, CI [+47.94, +52.54]** — ikaros 110 W vs daedalus 60 W under identical workload.
- **τ_heat diff: +3.08 s, CI [+2.78, +3.53]** — ikaros heats 3.4× slower at the package sensor.
- **R_th diff: −0.171 K/W, CI [−0.177, −0.165]** — daedalus has 55 % higher thermal resistance.
- **CPU per-core time diff: +2.05 ms, CI [+1.65, +2.45]** — ikaros cores ~24 % slower on 384×384×20 workload; per-core ranking r=−0.51 → distinct die orderings.

## C — NPU status (blocked, what is missing)
Both hosts have: amdxdna kernel module loaded; /dev/accel/accel0 char device; PCI 17f0 Signal Processing Controller.
Neither host has: xrt-smi/xrtutil; pyxrt python binding; /opt/xilinx subtree; any compiled .xclbin/.vaie model.
Until AMD's Ryzen-AI-SW deb stack (or RyzenAI-SW source build) is installed, the NPU char device cannot be exercised from userspace; no kernel submission, no inference jitter, no NPU-bound power. Recon JSON: results/IDENTITY_BENCHMARK_2026-05-30/deep/{host}/C_npu.json.

## D — Vmin sweep interpretation
- 60 reps × 80 row-tiles × 3 DPM levels per host: zero distinct hashes per tile anywhere. Bit-stable across low/auto/high.
- The driver-controlled DPM floors do not approach the Vmin cliff. Going below DPM `low` would require unsafe voltage table override (prior Probe C tried and hung ikaros).
- Side effect — at "high" ikaros completion time dropped to 0.23 ms vs 1.4 ms (boost engaged), but daedalus stayed at 2.7–2.8 ms across all levels. This reveals a per-host SCLK governor difference (board-firmware-config artefact, not silicon variance).

## Cross-angle synthesis
- 23-feature cross-angle vector (A means/std/τ × 4 wl + B {τ_heat,τ_cool,R_th} + E first 8 cores).
- L2 ikaros vs daedalus = 90.2 units (per-feature 18.8). Cosine 0.958.
- L2 dominated by Power (~70 W) and per-core time (~2 ms), but each of A/B/E independently rejects the null at Bonferroni-corrected α = 0.01 (=0.05/5).
- Multi-channel: with as few as 4 features (IDLE mean, MEDIUM mean, τ_heat, per-core time spread) a nearest-centroid classifier separates the machines with zero error — every per-rep distribution is non-overlapping on these axes.

## Power analysis (Cohen formula, 10 % effect, α=0.05, power 0.8)
| Workload | σ (W) | Target Δ (W) | N seeds needed |
|----------|------|--------------|----------------|
| IDLE     | 1.1  | 1.9          | **6**          |
| LIGHT    | 15.5 | 8.3          | 57             |
| MEDIUM   | 19.4 | 11.0         | 50             |
| HEAVY    | 21.3 | 10.9         | 61             |
With 5 reps already we detect 13–50 W IDLE/MEDIUM/HEAVY differences because effects are >>10 %. Detecting hypothetical 10 % drift on LIGHT–HEAVY would need ~50-60 reps; IDLE just 6.

## Comparison to prior 9 NULL attacks
Prior attacks probed GPU compute kernels (gemm outputs, RNG, timing, gpu_metrics blob, voltage sweeps that crashed ikaros) — they tried to force silicon below operating envelope.
This battery measures the **envelope itself**: stationary power draw (A), thermal RC of cooling stack (B), per-core sysfs latency (E). These are governed by manufacturing variance (Vt, leakage, TIM contact, fan curve, board components) and survive driver normalisation because they are not in the data path. That's why prior attacks missed them.
Caveat: A and B include cooling/board assembly contributions; only E is cleanly die-only (rank correlation across cores says distinct die ordering).

## Updated recommendation
- Identity **IS findable** on these two specific machines.
- Continue: tighten with N=60 reps for LIGHT-workload CI; replicate B another day to control ambient temperature.
- FPGA pivot not needed for identity — but remains the only way to isolate die-vs-package.
- NPU: install Ryzen-AI-SW deb stack to convert BLOCKED into a clean die-only probe.

## Thermal incidents
- 1 over-budget incident during initial too-aggressive A run (8 reps × 8 s, HEAVY=4 threads / size 1024) — APU reached 100 °C (trip 101 °C). Process killed, machine did not reboot.
- After: HEAVY=2 threads / size 512, temp cap 88 °C, abort 70 °C. All subsequent runs ≤ 91 °C peak, mostly ≤ 80 °C.
- 1 expected D-angle abort at "high" DPM on ikaros (>72 °C); partial reps recorded.

## Path
- Code: scripts/identity_benchmark/deep/{A,B,C,D,E}_*.py + _common.py + analyze_all.py + run_remaining_*.sh
- Raw per-host: results/IDENTITY_BENCHMARK_2026-05-30/deep/{ikaros,daedalus}/{A_power,B_thermal,C_npu,D_vmin,E_cpu}.json
- Cross-host analysis: results/IDENTITY_BENCHMARK_2026-05-30/deep/ANALYSIS.json
- Report: research_plan/IDENTITY_DEEP_2026-05-30_REPORT.md

## Bottom line
**Identity findable: YES.** 3 of 4 measured channels (A, B, E) independently discriminate the two machines at Cohen d > 3 with bootstrap CI excluding zero, surviving Bonferroni correction. D returned a clean null on its primary axis (no Vmin bit-flips) — informative: driver DPM floors prevent classical PUF probing without unsafe voltage override. C blocked pending NPU userspace install.

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
