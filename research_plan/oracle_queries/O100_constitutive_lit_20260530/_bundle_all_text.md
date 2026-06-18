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
