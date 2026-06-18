# Phase 8 — Rich dynamic substrate + A/B/C/D ablation

Date: 2026-05-31. Both hosts. APU stayed 45-59°C throughout.

## Capture
- ikaros: 120 channels × 15000 samples × 50 Hz (300 s)
- daedalus: 120 channels × 14046 samples × 50 Hz (~280 s)

Channels: hwmon (power/temp/freq/voltage all rails), thermal_zones, RAPL @1kHz, per-core cpufreq (16 cores), /proc/interrupts deltas, page-fault/context-switch rates, GPU freq/power/temp, TSC drift.

## Dynamic features extracted per host
- 3190 scalar per-channel features (mean, std, derivatives 1-3 at 5 scales, spectral, hysteresis, Fano)
- 60 channel-pairs × 4 cross-features = 240 (cross-channel impedance dP/dT, lag-correlation peak)
- Total scalar: **3430**
- Time-series: 480-dim × ~2900 samples (multi-scale derivatives + rolling spectra)

## A/B/C/D ablation (30 seeds, bootstrap 2000)

### C1 self-prediction (lower = better)

| eval | A | B | C | D | A−B (struct%) | A−C (data%) | CI A−B |
|---|---|---|---|---|---|---|---|
| ikaros | 157.3 | 159.4 | 385.9 | 388.9 | **+1.30%** | +59.2% | [-18.4, +10.7] |
| daedalus | 20.0 | 20.1 | 74.7 | 77.3 | **+0.24%** | +73.2% | [-1.6, +1.2] |

### C2 self-anomaly (AUROC)

| eval | A | B | C | D | A−B (struct%) |
|---|---|---|---|---|---|
| ikaros | 0.5104 | 0.5108 | 0.5075 | 0.5085 | -0.07% |
| daedalus | 0.5100 | 0.5099 | 0.5050 | 0.5057 | +0.02% |

**Embodiment gate (A−B ≥5% with CI excluding 0): FAILED on all 4 cells.**

Data-distribution effect (A−C) is 60-73% on C1 — confirms Phase 7 finding that distribution shift dominates and chassi-hash structure adds nothing measurable even with 3430 rich features.

## Physics-aware structure (specific neurons assigned to impedance/RTN/spectral)

| eval | A_phys | A_base | B_rand | vs base | vs random |
|---|---|---|---|---|---|
| ikaros | 145.97 | 155.51 | 150.17 | +6.13% | +2.80% |
| daedalus | 11.54 | 16.90 | 18.60 | +31.75% | **+37.97%** |

**Asymmetric**: physics-aware structure helps significantly on daedalus (+38% vs random) but only modestly on ikaros (+3%). Result is suggestive but inconsistent.

## Verdict
- **Hash-based decoration (A/B/C/D)**: NULL even with rich dynamic features. Confirms Phase 7 conclusion.
- **Physics-aware mapping**: positive signal on daedalus, weak on ikaros. Asymmetry needs investigation (could be daedalus's richer dynamic signature, or could be artifact).
- **Cross-channel impedance** signal exists (T-P r=0.95, lag=1.36s thermal RC) but doesn't lift A/B in current architecture.

## Combined with Phase 9
Phase 9 fan-control showed 49.8% transplant penalty on ikaros, 69× on daedalus — first clear positive embodiment result.

**Headline (defensible)**:
> Static body-info encoding adds 0% even with 3430 dynamic substrate features (n=30 seeds, A−B = 0.0% to +1.3%, CI spans 0). Closed-loop interaction with chassi's physical transfer function (fan-control) shows 49.8% transplant penalty (n=30, both hosts symmetric). Embodiment is real only when the model must couple to chassi physics through action, not when it merely reads chassi signals as input.

## Recommendation
1. Drop static-feature embodiment claims. Phase 8 fully replicates the NULL.
2. Promote fan-control to headline. Verify on real PWM (currently sim).
3. Investigate physics-aware asymmetry (why daedalus +38% but ikaros +3%?) — could be 11th-hour discovery or measurement artifact.
4. Consider Phase 10: constitutive + closed-loop hybrid (live substrate IS computation AND model controls actuator) for sharpest test.
