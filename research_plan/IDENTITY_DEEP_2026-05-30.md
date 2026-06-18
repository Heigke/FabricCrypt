# IDENTITY DEEP — 5 Angle Silicon Fingerprint Battery
Date: 2026-05-30
Machines: ikaros (local, card1), daedalus (SSH, card0)
Prior: 9 NULL attacks documented in IDENTITY_NULL_PAPER_2026-05-30.md.
Hypothesis: prior attacks missed channels dominated by silicon manufacturing variance.
This battery hits 5 new channels with stronger statistical power.

## Channels
- A. Power-draw fingerprint at fixed workloads (IDLE/LIGHT/MEDIUM/HEAVY, 50Hz x 60s x 10 reps each)
- B. Thermal time-constant (rising τ_heat, falling τ_cool, R_th via 10 step cycles 5 min each)
- C. NPU XDNA recon + fingerprint if accessible
- D. DPM Vmin sweep (low→auto→high) with per-CU bit-stability scan
- E. CPU per-core fingerprint over 16 logical cores (freq, completion time, RAPL)

## Statistics
- Bootstrap 95% CI (1000 resamples) on every reported number
- Bonferroni multi-testing correction across 5 angles
- SHUFFLE control: cross-device pairings as null
- Pre-registered discovery gate: signal > 2σ vs control AND effect > 3× within-condition variance
- Honest power analysis: bootstrap-extrapolate required N for 10% effect at α=0.05

## Hard constraints
- APU temp < 75 °C; abort kernel if > 72 °C
- 2 s max kernel burst, 30 s cooldown
- 120 min wall budget total
- Thermal_guard PID 9305 already filters identity_benchmark/* — respect SIGSTOP, retry

## Output layout
- scripts/identity_benchmark/deep/{A..E}_*.py
- results/IDENTITY_BENCHMARK_2026-05-30/deep/{ikaros,daedalus}/{A..E}_*.json
- research_plan/IDENTITY_DEEP_2026-05-30_REPORT.md

## NPU recon (prior to coding C)
- /dev/accel/accel0 exists, ikaros in render group
- amdxdna kernel module loaded
- NO XRT userspace at /opt/xilinx, NO python xrt binding installed
- C-step1 will document blocker; C-step2/3 only if userspace appears
