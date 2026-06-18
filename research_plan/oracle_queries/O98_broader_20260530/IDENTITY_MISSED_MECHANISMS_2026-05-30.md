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
