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
