# H7 first-pass — within-day cross-chassis discriminability

Source: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_H7_2026-06-09` (5 runs, hosts=['daedalus', 'ikaros'])
Pre-registration: `research_plan/H7_PREREG_2026-06-09.md`

This is a **first-pass** report. The pre-registered acceptance gates (block-CV AUC, matched-spectrum spoof, thermal match, replay) are NOT applied here — they will be enforced once we have ≥5 traces per (host, load) cell. The numbers below are the raw separability of each channel from a single 20-second idle baseline per chassis.

## TPM ground-truth
- daedalus: EK=000bfa5e7d54f8e4570c55ffeb025a8b1b6ebf3dc93edffd6f7bcc142bdb9264918c  PCR0=0xC0EA9099846E466A…
- daedalus: EK=000bfa5e7d54f8e4570c55ffeb025a8b1b6ebf3dc93edffd6f7bcc142bdb9264918c  PCR0=0xC0EA9099846E466A…
- ikaros: EK=000b359aefc948982dcfbc2e2f84db2c477909f8aee9e9222ab4b5c0ef423d477a14  PCR0=0xE2DDD6B9DF1E27DA…
- ikaros: EK=000b359aefc948982dcfbc2e2f84db2c477909f8aee9e9222ab4b5c0ef423d477a14  PCR0=0xE2DDD6B9DF1E27DA…
- ikaros: EK=000b359aefc948982dcfbc2e2f84db2c477909f8aee9e9222ab4b5c0ef423d477a14  PCR0=0xE2DDD6B9DF1E27DA…

## Channel table (sorted by discriminative AUC, highest first)
| channel | n_a | n_b | mean_a | mean_b | d | AUC | flag |
|---|---|---|---|---|---|---|---|
| C03_core00_thermal | 2241 | 3232 | 1.25e+06 | 1.57e+06 | -2.89 | 1.000 | ↑ promising |
| C03_core01_thermal | 2241 | 3232 | 1.26e+06 | 1.56e+06 | -2.89 | 1.000 | ↑ promising |
| C03_core02_thermal | 2241 | 3232 | 1.26e+06 | 1.58e+06 | -2.69 | 1.000 | ↑ promising |
| C03_core03_thermal | 2241 | 3232 | 1.26e+06 | 1.6e+06 | -2.63 | 1.000 | ↑ promising |
| C03_core04_thermal | 2241 | 3232 | 1.25e+06 | 1.6e+06 | -2.60 | 1.000 | ↑ promising |
| C03_core05_thermal | 2241 | 3232 | 1.25e+06 | 1.59e+06 | -2.52 | 1.000 | ↑ promising |
| C03_core06_thermal | 2241 | 3232 | 1.26e+06 | 1.58e+06 | -2.49 | 1.000 | ↑ promising |
| C03_core07_thermal | 2241 | 3232 | 1.25e+06 | 1.58e+06 | -2.51 | 1.000 | ↑ promising |
| C03_core08_thermal | 2241 | 3232 | 1.26e+06 | 1.58e+06 | -2.71 | 1.000 | ↑ promising |
| C03_core09_thermal | 2241 | 3232 | 1.25e+06 | 1.56e+06 | -2.81 | 1.000 | ↑ promising |
| C03_core10_thermal | 2241 | 3232 | 1.25e+06 | 1.55e+06 | -2.90 | 1.000 | ↑ promising |
| C03_core11_thermal | 2241 | 3232 | 1.25e+06 | 1.54e+06 | -2.99 | 1.000 | ↑ promising |
| C03_core12_thermal | 2241 | 3232 | 1.26e+06 | 1.56e+06 | -2.62 | 1.000 | ↑ promising |
| C03_core13_thermal | 2241 | 3232 | 1.26e+06 | 1.58e+06 | -2.42 | 1.000 | ↑ promising |
| C03_core14_thermal | 2241 | 3232 | 1.26e+06 | 1.6e+06 | -2.24 | 1.000 | ↑ promising |
| C03_core15_thermal | 2241 | 3232 | 1.26e+06 | 1.59e+06 | -2.41 | 1.000 | ↑ promising |
| C04_base_thermal_C | 2241 | 3232 | 79.2 | 107 | -1.59 | 1.000 | ↑ promising |
| C07_xtal_cntl | 2241 | 3232 | 1.25e+06 | 1.56e+06 | -2.82 | 1.000 | ↑ promising |
| C09_pm[1] | 225 | 325 | 6.25 | 20 | -5.53 | 1.000 | ★ candidate |
| C09_pm[5] | 225 | 325 | 7.46 | 33.4 | -1.54 | 1.000 | ↑ promising |
| C09_pm[3] | 225 | 325 | 7.47 | 67 | -1.20 | 0.997 | ↑ promising |
| C09_pm[31] | 225 | 325 | 1.47 | 0.79 | +0.23 | 0.925 | ↑ promising |
| C11_drift_ns_per_step | 2241 | 3231 | 5.47e+03 | 1.42e+04 | -0.56 | 0.866 | ↑ promising |
| C09_pm[110] | 225 | 325 | 1.35 | 1.32 | +0.89 | 0.746 | weak |
| C06_fast | 2241 | 3232 | 2.71e+09 | 1.65e+09 | +1.00 | 0.735 | weak |
| C09_pm[130] | 225 | 325 | 1.52 | 1.52 | +0.07 | 0.702 | weak |
| C05_e1 | 2241 | 3232 | 4e+05 | 5.91e+05 | -0.89 | 0.650 | weak |
| C05_e0 | 2241 | 3232 | 5.99e+05 | 8.64e+05 | -0.90 | 0.650 | weak |
| C09_pm[30] | 225 | 325 | 2.18e+03 | 2.1e+03 | +0.29 | 0.551 | — |
| C05_e2 | 2241 | 3232 | 2e+05 | 2e+05 | +0.05 | 0.515 | — |
| C18_gpu_clock_delta | 2241 | 3230 | -1.65e+16 | -1.71e+16 | +0.00 | 0.500 | — |
| C08_gfx_vid | 2241 | 3232 | 92 | 92 | +0.00 | 0.500 | — |
| C08_soc_vid | 2241 | 3232 | 50 | 50 | +0.00 | 0.500 | — |
| C09_pm[170] | 225 | 325 | 2e+03 | 2e+03 | +0.00 | 0.500 | — |
| C09_pm[194] | 225 | 325 | 0.945 | 0.945 | +0.00 | 0.500 | — |
| C19_CP_STAT | 2241 | 3230 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_GRBM_STATUS | 2241 | 3230 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_GRBM_STATUS2 | 2241 | 3230 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_GRBM_STATUS_SE0 | 2241 | 3230 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_GRBM_STATUS_SE1 | 2241 | 3230 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_RLC_GPM_STAT | 2241 | 3230 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_RLC_STAT | 2241 | 3230 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_SRBM_STATUS | 2241 | 3230 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |

## Notes
- **★ candidate** = both AUC≥0.95 AND |d|≥3 in this single-trace pair. That clears the *point-estimate* level of the pre-registered gate. It does NOT yet clear matched-spectrum spoofing, thermal-matching, or replay-from-log — those need more traces and the cross-temp set.
- **↑ promising** = AUC≥0.80 but not 0.95. Often these are chassis-confounds (PSU, fan, NVMe) that survive crude classification but are designed to fail the spoof+thermal gate.
- Channels at AUC≈0.5 are not carrying chassis identity in this trace.