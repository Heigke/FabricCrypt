# H7 first-pass — within-day cross-chassis discriminability

Source: `/home/ikaros/Documents/claude_hive/AMD_gfx1151_energy/results/IDENTITY_H7_2026-06-09` (11 runs, hosts=['daedalus', 'ikaros'])
Pre-registration: `research_plan/H7_PREREG_2026-06-09.md`

This is a **first-pass** report. The pre-registered acceptance gates (block-CV AUC, matched-spectrum spoof, thermal match, replay) are NOT applied here — they will be enforced once we have ≥5 traces per (host, load) cell. The numbers below are the raw separability of each channel from a single 20-second idle baseline per chassis.

## TPM ground-truth
- daedalus: EK=000bfa5e7d54f8e4570c55ffeb025a8b1b6ebf3dc93edffd6f7bcc142bdb9264918c  PCR0=0xC0EA9099846E466A…
- daedalus: EK=000bfa5e7d54f8e4570c55ffeb025a8b1b6ebf3dc93edffd6f7bcc142bdb9264918c  PCR0=0xC0EA9099846E466A…
- daedalus: EK=000bfa5e7d54f8e4570c55ffeb025a8b1b6ebf3dc93edffd6f7bcc142bdb9264918c  PCR0=0xC0EA9099846E466A…
- daedalus: EK=000bfa5e7d54f8e4570c55ffeb025a8b1b6ebf3dc93edffd6f7bcc142bdb9264918c  PCR0=0xC0EA9099846E466A…
- daedalus: EK=000bfa5e7d54f8e4570c55ffeb025a8b1b6ebf3dc93edffd6f7bcc142bdb9264918c  PCR0=0xC0EA9099846E466A…
- ikaros: EK=000b359aefc948982dcfbc2e2f84db2c477909f8aee9e9222ab4b5c0ef423d477a14  PCR0=0xE2DDD6B9DF1E27DA…
- ikaros: EK=000b359aefc948982dcfbc2e2f84db2c477909f8aee9e9222ab4b5c0ef423d477a14  PCR0=0xE2DDD6B9DF1E27DA…
- ikaros: EK=000b359aefc948982dcfbc2e2f84db2c477909f8aee9e9222ab4b5c0ef423d477a14  PCR0=0xE2DDD6B9DF1E27DA…
- ikaros: EK=000b359aefc948982dcfbc2e2f84db2c477909f8aee9e9222ab4b5c0ef423d477a14  PCR0=0xE2DDD6B9DF1E27DA…
- ikaros: EK=000b359aefc948982dcfbc2e2f84db2c477909f8aee9e9222ab4b5c0ef423d477a14  PCR0=0xE2DDD6B9DF1E27DA…
- ikaros: EK=000b359aefc948982dcfbc2e2f84db2c477909f8aee9e9222ab4b5c0ef423d477a14  PCR0=0xE2DDD6B9DF1E27DA…

## Channel table (sorted by discriminative AUC, highest first)
| channel | n_a | n_b | mean_a | mean_b | d | AUC | flag |
|---|---|---|---|---|---|---|---|
| C03_core00_thermal | 5229 | 6212 | 1.25e+06 | 1.54e+06 | -3.63 | 1.000 | ★ candidate |
| C03_core01_thermal | 5229 | 6212 | 1.26e+06 | 1.53e+06 | -3.66 | 1.000 | ★ candidate |
| C03_core02_thermal | 5229 | 6212 | 1.26e+06 | 1.55e+06 | -3.32 | 1.000 | ★ candidate |
| C03_core03_thermal | 5229 | 6212 | 1.26e+06 | 1.56e+06 | -3.20 | 1.000 | ★ candidate |
| C03_core04_thermal | 5229 | 6212 | 1.25e+06 | 1.56e+06 | -3.14 | 1.000 | ★ candidate |
| C03_core05_thermal | 5229 | 6212 | 1.25e+06 | 1.55e+06 | -3.03 | 1.000 | ★ candidate |
| C03_core06_thermal | 5229 | 6212 | 1.25e+06 | 1.54e+06 | -3.00 | 1.000 | ★ candidate |
| C03_core07_thermal | 5229 | 6212 | 1.25e+06 | 1.54e+06 | -3.04 | 1.000 | ★ candidate |
| C03_core08_thermal | 5229 | 6212 | 1.25e+06 | 1.54e+06 | -3.35 | 1.000 | ★ candidate |
| C03_core09_thermal | 5229 | 6212 | 1.24e+06 | 1.53e+06 | -3.54 | 1.000 | ★ candidate |
| C03_core10_thermal | 5229 | 6212 | 1.25e+06 | 1.52e+06 | -3.69 | 1.000 | ★ candidate |
| C03_core11_thermal | 5229 | 6212 | 1.25e+06 | 1.51e+06 | -3.86 | 1.000 | ★ candidate |
| C03_core12_thermal | 5229 | 6212 | 1.26e+06 | 1.53e+06 | -3.24 | 1.000 | ★ candidate |
| C03_core13_thermal | 5229 | 6212 | 1.26e+06 | 1.54e+06 | -2.91 | 1.000 | ↑ promising |
| C03_core14_thermal | 5229 | 6212 | 1.26e+06 | 1.55e+06 | -2.61 | 1.000 | ↑ promising |
| C03_core15_thermal | 5229 | 6212 | 1.26e+06 | 1.55e+06 | -2.87 | 1.000 | ↑ promising |
| C04_base_thermal_C | 5229 | 6212 | 78.9 | 100 | -1.59 | 1.000 | ↑ promising |
| C07_xtal_cntl | 5229 | 6212 | 1.24e+06 | 1.53e+06 | -3.54 | 1.000 | ★ candidate |
| C09_pm[1] | 525 | 625 | 6.2 | 20.4 | -8.11 | 1.000 | ★ candidate |
| C09_pm[5] | 525 | 625 | 6.95 | 27.1 | -1.59 | 1.000 | ↑ promising |
| C09_pm[3] | 525 | 625 | 7.1 | 44.5 | -0.98 | 0.997 | ↑ promising |
| C09_pm[31] | 525 | 625 | 1.36 | 0.411 | +0.42 | 0.959 | ↑ promising |
| C11_drift_ns_per_step | 5227 | 6207 | 5.54e+03 | 1.69e+04 | -0.70 | 0.882 | ↑ promising |
| C06_fast | 5229 | 6212 | 2.65e+09 | 1.58e+09 | +1.05 | 0.752 | weak |
| C09_pm[130] | 525 | 625 | 1.52 | 1.52 | +0.05 | 0.687 | weak |
| C09_pm[110] | 525 | 625 | 1.35 | 1.33 | +0.60 | 0.665 | weak |
| C05_e0 | 5229 | 6212 | 5.99e+05 | 7.37e+05 | -0.61 | 0.592 | — |
| C05_e1 | 5229 | 6212 | 3.99e+05 | 4.99e+05 | -0.61 | 0.592 | — |
| C09_pm[30] | 525 | 625 | 2.18e+03 | 2.14e+03 | +0.20 | 0.575 | — |
| C05_e2 | 5229 | 6212 | 2e+05 | 2e+05 | -0.03 | 0.508 | — |
| C18_gpu_clock_delta | 5227 | 6210 | -1.76e+16 | -1.78e+16 | +0.00 | 0.500 | — |
| C08_gfx_vid | 5229 | 6212 | 92 | 92 | +0.00 | 0.500 | — |
| C08_soc_vid | 5229 | 6212 | 50 | 50 | +0.00 | 0.500 | — |
| C09_pm[170] | 525 | 625 | 2e+03 | 2e+03 | +0.00 | 0.500 | — |
| C09_pm[194] | 525 | 625 | 0.945 | 0.945 | +0.00 | 0.500 | — |
| C19_CP_STAT | 5227 | 6210 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_GRBM_STATUS | 5227 | 6210 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_GRBM_STATUS2 | 5227 | 6210 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_GRBM_STATUS_SE0 | 5227 | 6210 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_GRBM_STATUS_SE1 | 5227 | 6210 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_RLC_GPM_STAT | 5227 | 6210 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_RLC_STAT | 5227 | 6210 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |
| C19_SRBM_STATUS | 5227 | 6210 | 4.29e+09 | 4.29e+09 | +0.00 | 0.500 | — |

## Notes
- **★ candidate** = both AUC≥0.95 AND |d|≥3 in this single-trace pair. That clears the *point-estimate* level of the pre-registered gate. It does NOT yet clear matched-spectrum spoofing, thermal-matching, or replay-from-log — those need more traces and the cross-temp set.
- **↑ promising** = AUC≥0.80 but not 0.95. Often these are chassis-confounds (PSU, fan, NVMe) that survive crude classification but are designed to fail the spoof+thermal gate.
- Channels at AUC≈0.5 are not carrying chassis identity in this trace.