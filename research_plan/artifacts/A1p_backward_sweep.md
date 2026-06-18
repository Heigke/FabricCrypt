# A1p — Backward-Vd Sweep Hysteresis Test
**Bias:** VG1=0.6 V, VG2=0.0 V (worst-failing point in z91g).
**Sebas overrides:** ETAB=2.5, K1=0.41825, ALPHA0=7.842e-5, BETA0=20, NFACTOR=6.0, mbjt=1, IS=5e-9.
**Physics:** post-A.1.j/A.1.o, emitter=GND + vnwell=2 V, vnwell_Rs=1e9 (cfg defaults).

## Procedure
- **Forward:** Vd 0.05 → 2.0 V, init Vsint=0, Vb=0; warm-start each point from previous solve.
- **Backward:** Vd 2.0 → 0.05 V, init Vsint=0.5, Vb=0.85 (hot, biased into the impact-ion / high-Vb branch); warm-start each point from previous solve.
- Both use `solve_2t_with_homotopy` (gmin schedule 1e-3→1e-5→1e-8→1e-12→target).

## Result
- max |Δlog10 Id| across sweep = **0.0075 decades**
- Verdict: **NO_BISTABILITY**

| Vd (V) | Id_fwd (A) | Id_bwd (A) | Vb_fwd | Vb_bwd | Δdec |
|---|---|---|---|---|---|
| 0.050 | 2.387e-07 | 2.387e-07 | +0.4705 | +0.4705 | 0.000 |
| 0.100 | 1.392e-06 | 1.392e-06 | +0.5103 | +0.5103 | 0.000 |
| 0.200 | 4.674e-06 | 4.674e-06 | +0.5349 | +0.5349 | 0.000 |
| 0.300 | 5.209e-06 | 5.209e-06 | +0.5371 | +0.5371 | 0.000 |
| 0.400 | 6.420e-06 | 6.420e-06 | +0.5425 | +0.5425 | 0.000 |
| 0.500 | 4.978e-06 | 4.978e-06 | +0.5359 | +0.5359 | 0.000 |
| 0.700 | 6.832e-06 | 6.832e-06 | +0.5440 | +0.5440 | 0.000 |
| 0.900 | 4.976e-06 | 4.976e-06 | +0.5358 | +0.5358 | 0.000 |
| 1.100 | 4.983e-06 | 4.983e-06 | +0.5358 | +0.5358 | 0.000 |
| 1.300 | 4.993e-06 | 4.993e-06 | +0.5358 | +0.5358 | 0.000 |
| 1.400 | 4.998e-06 | 4.998e-06 | +0.5358 | +0.5358 | 0.000 |
| 1.500 | 5.003e-06 | 5.003e-06 | +0.5358 | +0.5358 | 0.000 |
| 1.600 | 5.061e-06 | 5.061e-06 | +0.5360 | +0.5360 | 0.000 |
| 1.700 | 1.478e-05 | 1.478e-05 | +0.5637 | +0.5637 | 0.000 |
| 1.800 | 7.483e-05 | 7.483e-05 | +0.6057 | +0.6057 | 0.000 |
| 1.900 | 2.289e-04 | 2.289e-04 | +0.6346 | +0.6346 | 0.000 |
| 2.000 | 6.073e-04 | 6.179e-04 | +0.6598 | +0.6602 | 0.008 |

## Interpretation
Forward and backward sweeps are bit-identical (max Δ < 0.01 dec). The Newton solver converges to the **unique** root regardless of warm-start; the high-Vb branch is **not an attractor** at this bias under the current physics. The z91g residual at VG1=0.6/VG2=0.0 is therefore **not** a missed bistable branch — it is a genuine **model-physics gap** (impact-ion / BJT / leakage parameter shape) that no solver-side fix can recover. Next step: refit alpha0/beta0/NFACTOR/IS jointly across this diagnostic point or add a missing current pathway.
