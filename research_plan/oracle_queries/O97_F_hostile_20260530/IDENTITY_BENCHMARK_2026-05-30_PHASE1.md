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
