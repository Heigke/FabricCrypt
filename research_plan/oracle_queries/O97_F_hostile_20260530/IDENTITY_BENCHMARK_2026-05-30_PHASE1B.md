# Identity Benchmark — Phase 1B Verdict (Thermal-Controlled)

Date: 2026-05-30 · Devices: ikaros vs daedalus
Phase 1B run: ikaros 2026-05-30T11:47:33 · daedalus 2026-05-30T11:55:27

## Verdict: **MIXED**

## Achieved-temperature matrix

| regime | ikaros (°C) | daedalus (°C) | Δ | ikaros in_band | daedalus in_band |
|---|---|---|---|---|---|
| cold | 48.0 | 46.1 | +1.8 | False | True |
| idle | 53.5 | 46.3 | +7.2 | True | False |

## Per-regime divergence

| regime | intra_a | intra_b | inter | KL(knee) | KL(RTN) | RTN_a | RTN_b | spatial_MSE | KL(perf) |
|---|---|---|---|---|---|---|---|---|---|
| cold | 0.243 | 0.296 | 0.295 | 12.691 | 25.105 | 0.0000 | 0.1086 | 0.0792 | 0.123 |
| idle | 0.243 | 0.296 | 0.289 | 1.316 | 25.105 | 0.0000 | 0.1111 | 0.0953 | 0.121 |

## Phase 1 baseline (unmatched temp, for reference)

- KL(knee) = 6.544
- KL(RTN) = 25.105
- spatial-corr MSE = 0.0923
- inter-HD = 0.295 vs intra-HD = 0.270
- RTN ikaros=0.0000 daedalus=0.1149
- spatial-corr ikaros=0.0563 daedalus=0.3601

## Justification

- KL(knee)[cold]=12.691 (194% of Phase 1 baseline 6.544)
- KL(knee)[idle]=1.316 (20% of Phase 1 baseline 6.544)
- RTN[cold] a=0.0000 b=0.1086 sign=-1
- RTN[idle] a=0.0000 b=0.1111 sign=-1
- spatial-MSE[cold]=0.0792
- spatial-MSE[idle]=0.0953
- Channel survival: knee=False rtn=True spatial=True -> 2/3

## Decomposition

Partial survival: some channels show silicon-driven divergence 
at matched temperatures, others collapse. **Phase 2 transplant 
matrix: CONDITIONAL — proceed using only the surviving 
channels listed above; document the thermal-sensitive channels 
as confounded.**

## Raw data paths

- ikaros/cold: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/ikaros/raw_cold.npz`
- ikaros/idle: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/ikaros/raw_idle.npz`
- ikaros/signature_thermal.json: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/ikaros/signature_thermal.json`
- daedalus/cold: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/daedalus/raw_cold.npz`
- daedalus/idle: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/daedalus/raw_idle.npz`
- daedalus/signature_thermal.json: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_BENCHMARK_2026-05-30/daedalus/signature_thermal.json`