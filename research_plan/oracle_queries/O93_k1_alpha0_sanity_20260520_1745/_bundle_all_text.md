# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: ablation.json (8094 chars) ===
```json
{
  "K1=0.41825__ALPHA0=7.8420e-05": {
    "label": "K1=0.41825__ALPHA0=7.8420e-05",
    "median_dec_all": {
      "median": 1.1634943709139112,
      "ci95_lo": 1.1337578092161436,
      "ci95_hi": 1.415315930063886,
      "n": 66
    },
    "median_dec_VG1=0.2": {
      "median": 0.6308014499376018,
      "ci95_lo": 0.48251684202586453,
      "ci95_hi": 0.7636531995065461,
      "n": 14
    },
    "median_dec_VG1=0.4": {
      "median": 1.4113726661745711,
      "ci95_lo": 0.32521385675487835,
      "ci95_hi": 1.4153746065191068,
      "n": 22
    },
    "median_dec_VG1=0.6": {
      "median": 1.7316692258861455,
      "ci95_lo": 1.1592930398671355,
      "ci95_hi": 1.9124829040054117,
      "n": 30
    },
    "median_dec_fwd": {
      "median": 1.1507672535287545,
      "ci95_lo": 0.522081779834151,
      "ci95_hi": 1.415453678576866,
      "n": 33
    },
    "median_dec_bwd": {
      "median": 1.1678254815701017,
      "ci95_lo": 1.0675774494653663,
      "ci95_hi": 1.4161133339804328,
      "n": 33
    },
    "triode_rmse_VG1=0.6": {
      "median": 1.1834782385036529,
      "ci95_lo": 1.182630411252866,
      "ci95_hi": 1.187836561516818,
      "n": 30
    },
    "k1_vg1_0p6": 0.41825,
    "alpha0": 7.842e-05,
    "nan_count": 0,
    "runtime_s": 142.69160771369934,
    "n_rows": 66,
    "n_finite": 66,
    "convergence_rate": 1.0,
    "worst_VG1=0.2": {
      "n": 14,
      "median_dec": 0.6308014499376018,
      "max_dec": 1.0675774494653663,
      "Imeas_over_Ipred_med": 288.77856398343147,
      "Imeas_over_Ipred_max": 767.2493986078798
    },
    "worst_VG1=0.4": {
      "n": 22,
      "median_dec": 1.4113726661745711,
      "max_dec": 1.4168249919311169,
      "Imeas_over_Ipred_med": 108.98973519847047,
      "Imeas_over_Ipred_max": 138.8783748785792
    },
    "worst_VG1=0.6": {
      "n": 30,
      "median_dec": 1.7316692258861455,
      "max_dec": 1.9891698823537478,
      "Imeas_over_Ipred_med": 45.79083797308658,
      "Imeas_over_Ipred_max": 65.54068326003804
    }
  },
  "K1=0.41825__ALPHA0=7.8376e-04": {
    "label": "K1=0.41825__ALPHA0=7.8376e-04",
    "median_dec_all": {
      "median": 1.163494370913838,
      "ci95_lo": 1.1329339279664041,
      "ci95_hi": 1.3756539243611947,
      "n": 66
    },
    "median_dec_VG1=0.2": {
      "median": 0.6348812228051308,
      "ci95_lo": 0.4795822477162961,
      "ci95_hi": 0.7636531995065461,
      "n": 14
    },
    "median_dec_VG1=0.4": {
      "median": 1.4108725325820122,
      "ci95_lo": 0.3252469593945868,
      "ci95_hi": 1.4148720724762485,
      "n": 22
    },
    "median_dec_VG1=0.6": {
      "median": 1.2953690020056006,
      "ci95_lo": 1.163494370913838,
      "ci95_hi": 1.4856479651695556,
      "n": 30
    },
    "median_dec_fwd": {
      "median": 1.150767253528553,
      "ci95_lo": 0.5302413255698362,
      "ci95_hi": 1.4123664606055888,
      "n": 33
    },
    "median_dec_bwd": {
      "median": 1.1678254815700537,
      "ci95_lo": 1.1401764754200494,
      "ci95_hi": 1.4127820160909499,
      "n": 33
    },
    "triode_rmse_VG1=0.6": {
      "median": 1.1834782385036107,
      "ci95_lo": 1.182630411252822,
      "ci95_hi": 1.1879518216508638,
      "n": 30
    },
    "k1_vg1_0p6": 0.41825,
    "alpha0": 0.000783756,
    "nan_count": 0,
    "runtime_s": 147.92579698562622,
    "n_rows": 66,
    "n_finite": 66,
    "convergence_rate": 1.0,
    "worst_VG1=0.2": {
      "n": 14,
      "median_dec": 0.6348812228051308,
      "max_dec": 1.064642136004478,
      "Imeas_over_Ipred_med": 139.18187429650948,
      "Imeas_over_Ipred_max": 376.1934879102288
    },
    "worst_VG1=0.4": {
      "n": 22,
      "median_dec": 1.4108725325820122,
      "max_dec": 1.4163112409938936,
      "Imeas_over_Ipred_med": 38.384044406271656,
      "Imeas_over_Ipred_max": 50.96297117630874
    },
    "worst_VG1=0.6": {
      "n": 30,
      "median_dec": 1.2953690020056006,
      "max_dec": 1.7428154263592281,
      "Imeas_over_Ipred_med": 17.589823037899862,
      "Imeas_over_Ipred_max": 25.50141614941974
    }
  },
  "K1=0.53825__ALPHA0=7.8420e-05": {
    "label": "K1=0.53825__ALPHA0=7.8420e-05",
    "median_dec_all": {
      "median": 0.8825963117849005,
      "ci95_lo": 0.4843776212331683,
      "ci95_hi": 1.0594171627420843,
      "n": 66
    },
    "median_dec_VG1=0.2": {
      "median": 0.6308014499376018,
      "ci95_lo": 0.483434618310185,
      "ci95_hi": 0.7636531995065461,
      "n": 14
    },
    "median_dec_VG1=0.4": {
      "median": 1.4113726661745711,
      "ci95_lo": 0.32459494499119135,
      "ci95_hi": 1.4153748614641386,
      "n": 22
    },
    "median_dec_VG1=0.6": {
      "median": 0.926545106206953,
      "ci95_lo": 0.4223698442199475,
      "ci95_hi": 1.104752476203208,
      "n": 30
    },
    "median_dec_fwd": {
      "median": 0.496448181705226,
      "ci95_lo": 0.4223271940032882,
      "ci95_hi": 1.1046412843709943,
      "n": 33
    },
    "median_dec_bwd": {
      "median": 1.013774564870955,
      "ci95_lo": 0.7485176268293419,
      "ci95_hi": 1.172357572653974,
      "n": 33
    },
    "triode_rmse_VG1=0.6": {
      "median": 0.42534249825046266,
      "ci95_lo": 0.4186414815628004,
      "ci95_hi": 0.4307048801518011,
      "n": 30
    },
    "k1_vg1_0p6": 0.53825,
    "alpha0": 7.842e-05,
    "nan_count": 0,
    "runtime_s": 143.21283745765686,
    "n_rows": 66,
    "n_finite": 66,
    "convergence_rate": 1.0,
    "worst_VG1=0.2": {
      "n": 14,
      "median_dec": 0.6308014499376018,
      "max_dec": 1.0675774494653663,
      "Imeas_over_Ipred_med": 288.77856398343147,
      "Imeas_over_Ipred_max": 767.2493986078798
    },
    "worst_VG1=0.4": {
      "n": 22,
      "median_dec": 1.4113726661745711,
      "max_dec": 1.4168249919311169,
      "Imeas_over_Ipred_med": 108.98973519847047,
      "Imeas_over_Ipred_max": 138.8783748785792
    },
    "worst_VG1=0.6": {
      "n": 30,
      "median_dec": 0.926545106206953,
      "max_dec": 1.1829390705245437,
      "Imeas_over_Ipred_med": 7.627941110469383,
      "Imeas_over_Ipred_max": 9.64991497333567
    }
  },
  "K1=0.53825__ALPHA0=7.8376e-04": {
    "label": "K1=0.53825__ALPHA0=7.8376e-04",
    "median_dec_all": {
      "median": 0.6648666388478048,
      "ci95_lo": 0.48437144596104653,
      "ci95_hi": 0.7686396833860188,
      "n": 66
    },
    "median_dec_VG1=0.2": {
      "median": 0.6348812228051308,
      "ci95_lo": 0.4795822477162961,
      "ci95_hi": 0.7631443295667053,
      "n": 14
    },
    "median_dec_VG1=0.4": {
      "median": 1.4108725325820122,
      "ci95_lo": 0.32438103446065014,
      "ci95_hi": 1.4149197336234858,
      "n": 22
    },
    "median_dec_VG1=0.6": {
      "median": 0.6171550768450804,
      "ci95_lo": 0.42258059920802316,
      "ci95_hi": 0.7252349080724478,
      "n": 30
    },
    "median_dec_fwd": {
      "median": 0.5059914934623251,
      "ci95_lo": 0.42353777576597285,
      "ci95_hi": 0.7250755515027967,
      "n": 33
    },
    "median_dec_bwd": {
      "median": 0.7626093637325138,
      "ci95_lo": 0.6172125744553156,
      "ci95_hi": 0.8976808112356878,
      "n": 33
    },
    "triode_rmse_VG1=0.6": {
      "median": 0.4253424982501913,
      "ci95_lo": 0.4187888931040771,
      "ci95_hi": 0.43081505945251336,
      "n": 30
    },
    "k1_vg1_0p6": 0.53825,
    "alpha0": 0.000783756,
    "nan_count": 0,
    "runtime_s": 145.74339938163757,
    "n_rows": 66,
    "n_finite": 66,
    "convergence_rate": 1.0,
    "worst_VG1=0.2": {
      "n": 14,
      "median_dec": 0.6348812228051308,
      "max_dec": 1.064642136004478,
      "Imeas_over_Ipred_med": 139.18187429650948,
      "Imeas_over_Ipred_max": 376.1934879102288
    },
    "worst_VG1=0.4": {
      "n": 22,
      "median_dec": 1.4108725325820122,
      "max_dec": 1.4163112409938936,
      "Imeas_over_Ipred_med": 38.384044406271656,
      "Imeas_over_Ipred_max": 50.96297117630874
    },
    "worst_VG1=0.6": {
      "n": 30,
      "median_dec": 0.6171550768450804,
      "max_dec": 0.9693116851061774,
      "Imeas_over_Ipred_med": 3.9182892878579887,
      "Imeas_over_Ipred_max": 4.877703193966324
    }
  }
}
```


=== FILE: alpha_verdict.md (2655 chars) ===
```
# Track ALPHA — ALPHA0 10× Fix Ablation (canonical baseline, 33-bias fwd+bwd)

Canonical baseline (CSV ALPHA0=7.842e-5): median_dec target reference = 1.163 dec (n=66)

Mario LALPHA0_FIX card value: ALPHA0=7.83756e-4 (10× larger)


## Sweep table

| ALPHA0 | median_dec (all) | CI95 | conv | VG1=0.2 | VG1=0.4 | VG1=0.6 | VG1=0.6 Imeas/Ipred (med) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0000e-05 | 1.163 | [1.134, 1.415] | 1.00 | 0.629 | 1.411 | 2.148 | 1.43e+02 |
| 7.8420e-05 | 1.163 | [1.133, 1.415] | 1.00 | 0.631 | 1.411 | 1.732 | 4.58e+01 |
| 2.5000e-04 | 1.163 | [1.133, 1.414] | 1.00 | 0.631 | 1.411 | 1.493 | 2.71e+01 |
| 7.8376e-04 | 1.163 | [1.134, 1.393] | 1.00 | 0.635 | 1.411 | 1.295 | 1.76e+01 |
| 2.5000e-03 | 1.149 | [1.132, 1.231] | 1.00 | 0.636 | 1.352 | 1.176 | 1.25e+01 |
| 1.0000e-03 | 1.163 | [1.132, 1.374] | 1.00 | 0.635 | 1.411 | 1.258 | 1.62e+01 |

## CSV (baseline) → CARD (fix) delta

- Δmedian_dec (all 33 biases, fwd+bwd) = -0.000 dec
- Δmedian_dec (VG1=0.6 subset)         = -0.436 dec
- Imeas/Ipred at VG1=0.6 (median): baseline 4.58e+01 → fix 1.76e+01

**Verdict: WEAK — Δ ∈ (−0.2, 0], one of multiple needed fixes**


## Did the 46× shortfall at VG1=0.6 close?
- Baseline (CSV) Imeas/Ipred median at VG1=0.6 = **4.58e+01×**
- Fix (10× ALPHA0)  Imeas/Ipred median at VG1=0.6 = **1.76e+01×**
- Gap closure factor = 2.60× (>>1 = closed)
- **KILLSHOT NOTE**: Even at the 10× ALPHA0 fix the VG1=0.6 saturation regime still under-predicts by orders of magnitude. ALPHA0 alone CANNOT close the 46× shortfall. Consistent with A1m_alpha0_scale_test.md verdict (case (ii): missing body-charging path). Need additional body-injection mechanism.

## Best ALPHA0 in sweep
- **ALPHA0=2.5000e-03** → median_dec = 1.149

## PWL(V_G) impact-ionization — documented as future work
- Sebas's 2Tcell_BSIM_param_DC.csv has ALPHA0 = 7.842e-5 CONSTANT across all 33 (VG1, VG2) rows (verified S5C2_zoom_deep_findings_2026-05-15.md, line 140).
- Mario slide 21 is about transient oscillation, NOT impact-ionization PWL.
- BSIM4 §6.1 standard ALPHA0 is a scalar; PWL(V_G) is NOT supported by the foundry card data.
- Therefore PWL implementation is out of scope for this track; deferred.

## Provenance
- Baseline builder: `scripts/pillar_I_C3_jts_tat.py::build_pyport_base()` (Bf=100, η≤1, JTS default)
- Source of CSV value: `data/sebas_2026_04_22/2Tcell_BSIM_param_DC.csv` (33 biases, ALPHA0 constant 7.842e-5)
- Source of CARD value: `data/sebas_2026_04_22/M1_130DNWFB_LALPHA0_FIX.txt` line: `alpha0 = 7.83756e-4`
- Prior art: `research_plan/artifacts/A1m_alpha0_scale_test.md` (single-bias 4-decade sweep, falsifying at WORST bias)

```


=== FILE: diag_verdict.md (2234 chars) ===
```
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

```
