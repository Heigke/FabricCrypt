# Identity Temporal Probe — Phase 1 Verdict (2026-05-31)

## Status: AMBIGUOUS (envelope-confounded discovery)

## Setup

- 50 Hz, 12 segments × 2 cycles ≈ 210 s wall, ~10 k samples per device
- Channels: T_apu, T_gpu, P_gpu, F_gpu, F_cpu (per-CU SHADER_CYCLES
  deferred — would need on-GPU kernel polling above SMU bandwidth)
- Load schedule: idle / cpu25 / cpu100 / idle / gpu_burst / cool ×2
- Devices: ikaros (local), daedalus (remote, idle GPU 8 W, ambient ~22 °C)
- Mid-segment cooling pause at 72 °C to respect APU thermal envelope
- n=1 trial per device (Cohen-d not directly estimable; use z_proxy =
  |a-b| / max(|a|,|b|))

## Critical envelope difference (not silicon)

- ikaros idle GPU: 20.2 W mean, 16-18 W p10-p90, 42-44 °C APU
- daedalus idle GPU: 7.9 W mean, 4-7 W p10-p90, 28-30 °C APU

This is a 2.5× P_gpu / ~14 °C ambient/cooling difference. ALL T1 static
features are confounded by this.

## Per-group top discriminators

| Group | Feature | ikaros | daedalus | z_proxy | Interpretation |
|---|---|---|---|---|---|
| T1 static | F_gpu_std | 3.4e5 | 0 | 1.00 | DPM state — daedalus stays in single P-state |
| T2 deriv | dT_apu_mean_cool | -22.5 | -32.1 K/s | 0.30 | cooling rate — daedalus cooler ambient gives faster cooling |
| **T3 cross-coupling** | **dP/dT_gpu_slope** | **0.068 W/K** | **0.020 W/K** | **0.70** | **3.4× thermal-electrical coupling difference** |
| T3 cross-coupling | T_P_xcorr_lag | 0 s | 0.54 s | 1.00 | phase lag |
| T4 hysteresis | P_at_cool_mean | 23.6 W | 11.3 W | 0.52 | envelope confound |
| T5 step | ringback_hz_std | 0.113 | 0.044 | 0.61 | step-response variability |
| T6 spectral | coh_TP_freq | 1.17 Hz | 6.64 Hz | 0.82 | T-P coherence peak frequency |
| T7 phase-space | recurrence_density | 0.32 | 0.73 | 0.56 | attractor density |

## Pre-registered gates

- **DISCOVERY** (any T2-T7 z_proxy ≥ 0.5 at matched thermal state): **YES**
  on raw probe — dP/dT, ringback, hysteresis, spectral, phase-space all
  pass z=0.5 threshold.
- **AT MATCHED THERMAL STATE**: **NOT YET TESTED** — the 14 °C ambient
  difference contaminates everything.
- **KILL** (all T2-T7 < 0.2): no, not the case.

→ **Verdict: AMBIGUOUS** — direction predicted by oracles (dP/dT
top-3 across all 4) is observed at z=0.70, but envelope confound
dominates and matched-state replication is not done.

## What the oracles said (O104)

3/4 voted WRITE (P ≤ 0.08). gemini voted DISPATCH (P=0.15).
All 4 converged on dP/dT, hysteresis, aging as top channels.
**Bias check Q3:** 3/4 explicitly acknowledged RLHF steering on
device-fingerprinting topics; all 4 stated the cross-oracle
"abstraction tax" convergence is genuine physics, not alignment artifact.
See `research_plan/oracle_queries/O104_temporal_bias_20260531/synthesis.md`.

## Honest interpretation

The strongest hit (dP/dT_gpu = 0.068 vs 0.020 W/K) is exactly what
oracles predicted as the most promising channel. However:

1. ikaros runs idle GPU at ~20 W, daedalus at ~8 W. The slope is
   estimated at fundamentally different operating points of the same
   nonlinear thermal-electrical R(T) curve.
2. ikaros and daedalus have different cooling (different chassis design,
   different ambient, different fan curves). dP/dT measured this way is
   chip + TIM + heatsink + chassis + ambient.
3. n=1 per device. No within-day repeats to estimate noise.

To upgrade AMBIGUOUS → DISCOVERY, we would need:
- Multiple back-to-back runs per device to bound within-device variance.
- Matched ambient (both machines in same room, same fan curve forced).
- DPM pinned to same P-state on both machines.
- Excitation at a single fixed frequency for a clean impedance fit.

## Recommendation

**WRITE the negative-result paper now** with this temporal probe as
"Phase 1C — temporal channels also envelope-confounded; dP/dT direction
matches oracle prediction but is not silicon-isolable on commodity
gfx1151 with software-only telemetry."

The substrate-as-dynamic-operator design
(`scripts/identity_benchmark/temporal/DYNAMIC_OPERATOR_DESIGN.md`) is
pre-registered for a possible Phase 2 if the matched-ambient
follow-up is funded.

## Updated P(silicon-bound identity reachable on gfx1151)

Phase 1 + Phase 1B (thermal-controlled static): ~0.10
Phase 1C (temporal, this work): **unchanged at ~0.10**.
The temporal probe surfaced exactly what oracles predicted, but at the
same low-bandwidth, package-dominated regime that the static probe
also lived in. To move the needle below 0.05 we'd need either:
- matched-environment temporal repeat (cheap, would settle envelope question)
- on-GPU cycle counters above SMU bandwidth (expensive — kernel design)

## Artifacts

- Raw: `results/IDENTITY_BENCHMARK_2026-05-30/temporal/{ikaros,daedalus}_temporal.npz`
- Features: `results/IDENTITY_BENCHMARK_2026-05-30/temporal/{ikaros,daedalus}_features.json`
- Compare: `results/IDENTITY_BENCHMARK_2026-05-30/temporal/compare.json`
- Probe: `scripts/identity_benchmark/temporal/probe.py`
- Extract: `scripts/identity_benchmark/temporal/extract_features.py`
- Compare: `scripts/identity_benchmark/temporal/compare.py`
- Oracle synthesis: `research_plan/oracle_queries/O104_temporal_bias_20260531/synthesis.md`
- Dynamic-operator design: `scripts/identity_benchmark/temporal/DYNAMIC_OPERATOR_DESIGN.md`
