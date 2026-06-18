# Identity Benchmark — Phase 2 (Transplant Matrix) Verdict

Date: 2026-05-30 · Devices: ikaros vs daedalus
Substrate hooks: Phase 1b raw_idle.npz (per-CU RTN-rate + spatial-corr)
Task: NARMA-10 in 128-neuron tanh ESN, ridge readout

## Verdict: **NULL**

HW transplant Δ in NARMA-10 NRMSE between ikaros and daedalus is **0.0260
[95% CI 0.0062, 0.0457]**, only marginally above the SW-iid control
(0.0158 [0.0009, 0.0408]) and the shuffle control
(0.0141 [0.0008, 0.0364]). The HW Δ confidence interval overlaps with both
controls' upper CI — we cannot reject the null hypothesis that "device
identity makes no transplantable difference to a downstream reservoir task."

| pair | Δ NRMSE | 95% CI |
|---|---|---|
| HW(ikaros) vs HW(daedalus) | 0.0260 | [0.0062, 0.0457] |
| SW-iid control             | 0.0158 | [0.0009, 0.0408] |
| SHUFFLE control            | 0.0141 | [0.0008, 0.0364] |

## Per-condition NRMSE (mean ± 95% CI bootstrap)

| device | control | mean | CI lo | CI hi | n |
|---|---|---|---|---|---|
| ikaros | HW | 0.6760 | 0.6600 | 0.6910 | 10 |
| ikaros | SW_iid | 0.6901 | 0.6701 | 0.7059 | 10 |
| ikaros | SHUFFLE | 0.6669 | 0.6498 | 0.6824 | 10 |
| daedalus | HW | 0.6499 | 0.6383 | 0.6606 | 10 |
| daedalus | SW_iid | 0.7058 | 0.6866 | 0.7201 | 10 |
| daedalus | SHUFFLE | 0.6529 | 0.6360 | 0.6686 | 10 |

## Interpretation

- Both per-device HW NRMSE (0.65–0.68) is statistically indistinguishable from
  the SHUFFLE control on the same device (0.65–0.67) — the spatial structure
  of the per-CU correlation matrix carries no information that the reservoir
  exploits.
- SW-iid is slightly WORSE on both devices (0.69–0.71), suggesting the
  per-CU RTN-rate magnitudes do something for the reservoir, but this effect
  is identical in shape across devices and survives shuffling.
- Combined: the Phase 1b "surviving channels" (RTN, spatial-corr) carry
  device-statistical structure but not transplantable identity.

## Gate verdict
- **Phase 2: NULL.** Device identity is not transplantable to a downstream
  task at the resolution of these substrate hooks. Combined with Phase 1c
  KILL, the overall identity benchmark verdict for the gfx1151 user-space
  channels we tested is **no identity beyond statistical-pattern artefacts**.

## Method

- 128-neuron tanh ESN, spectral radius 0.9, leak 0.3, ridge α=1e-4
- NARMA-10 task, T_train=2000, T_test=500, washout=100
- 10 seeds × 3 controls × 2 devices = 60 runs (pure CPU/numpy, no GPU)
- Substrate hooks at activation:
  - RTN-rate → sparse multiplicative gain perturbation per neuron (tiled over
    CU-index)
  - spatial-corr → colored additive noise via Cholesky factor of regularised
    80×80 device spatial-correlation matrix (PSD-floored at 1e-4)
- Controls:
  - **HW**: real Phase 1b signature of that device.
  - **SW-iid**: per-CU rates uniform over (rtn.min, rtn.max), identity spatial
    cov — destroys device shape but matches scale.
  - **SHUFFLE**: device's own rates and spatial cov, but CU index randomly
    permuted — destroys identity, preserves marginal distribution exactly.
- Bootstrap on the |mean(ikaros) − mean(daedalus)| statistic, 5000 resamples,
  95% interval.

## Raw data
- matrix: `results/IDENTITY_BENCHMARK_2026-05-30/phase2/matrix_results.json`
- verdict: `results/IDENTITY_BENCHMARK_2026-05-30/phase2/verdict.json`
- markdown: `results/IDENTITY_BENCHMARK_2026-05-30/phase2/verdict.md`

## Thermal incidents
- Zero. Phase 2 is pure CPU/numpy; no GPU kernels, no temperature spikes.
- Total wall: 4.4 s.
