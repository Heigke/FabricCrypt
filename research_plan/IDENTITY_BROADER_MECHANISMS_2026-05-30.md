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
