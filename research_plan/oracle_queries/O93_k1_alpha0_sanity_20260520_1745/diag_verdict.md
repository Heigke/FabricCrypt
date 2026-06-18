# Track Diag — Asymmetry diagnosis of build_pyport_base() 1.163 dec gap

**Baseline**: LEGACY (enable_jts_dsd=False) per-bias rows from cached `results/Pillar_I_C3_jts_tat/summary.json` (same code path as JTS_OFF/JTS_ON; identical to canonical `build_pyport_base()`).

## TLDR numbers (n=33 biases, fwd+bwd each)

| metric | value |
|---|---|
| median(rmse_fwd) | **1.1508 dec** |
| median(rmse_bwd) | **1.1678 dec** |
| |fwd−bwd| of medians | **0.0171 dec** |
| median per-bias |fwd−bwd| | **0.0035 dec** |
| max per-bias |fwd−bwd|    | 1.0499 dec |
| Spearman ρ(fwd, bwd) over 33 biases | **0.950** (p=3.67e-17) |
| median measurement hysteresis | **0.0006 dec** (n=33) |
| max measurement hysteresis    | 2.8185 dec |
| top-5 outliers share of total | **25.2%** |
| (cached) median_dec_all | 1.1635 dec |

## Classification

- (a) **SYMMETRIC** — fwd≈bwd, ρ high. The 1.163 dec gap is *conventional static-physics shortfall* (missing parallel path or wrong subthreshold/triode physics), NOT memory.
- Measurement is *static* (median hysteresis 0.001 dec). The dataset is a fair static-DC target.

## Plain-words answer

The 1.163 dec gap is **symmetric** (fwd 1.151 vs bwd 1.168; per-bias |Δ| median 0.004 dec; ρ=0.95) AND the measurement itself is **non-hysteretic** (0.001 dec). The model is missing **static physics** (parallel leakage / subthreshold / triode regime), not memory effects. The Pazos cell may be a memory cell *in pulsed operation*, but the Sebas DC sweep at issue here is static-clean — the 1.163 dec gap is NOT explained by hysteresis.

## Top-5 outlier biases (dominate the 1.163 median)

| rank | VG1 | VG2 | rmse_fwd | rmse_bwd | |Δ| | Imeas_peak |
|---|---|---|---|---|---|---|
| 1 | 0.60 | -0.05 | 1.989 | 1.989 | 0.000 | 4.05e-05 |
| 2 | 0.60 | -0.10 | 1.989 | 1.989 | 0.000 | 4.05e-05 |
| 3 | 0.60 | -0.15 | 1.989 | 1.989 | 0.000 | 4.05e-05 |
| 4 | 0.60 | +0.00 | 1.987 | 1.988 | 0.001 | 4.05e-05 |
| 5 | 0.60 | +0.05 | 1.927 | 1.929 | 0.001 | 4.05e-05 |

## Files

- `per_bias_residuals.json` — full table with all 33 biases
- `scatter_fwd_bwd.png` — diagnostic plot (diagonal = symmetric)
- `hist_measurement_hysteresis.png` — fwd-vs-bwd hysteresis IN THE DATA
- `top5_outliers.md` — outlier breakdown
