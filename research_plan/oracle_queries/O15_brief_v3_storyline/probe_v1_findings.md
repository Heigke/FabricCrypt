# Stage 6b probe v1 — pyport binning audit on M2

**Date:** 2026-05-03 14:38

**Setup.** Pyport `compute_size_dep` invoked on M2 (geometry L=1800 nm,
W=360 nm, NF=1) with all binning terms loaded from card (no
`patch_model_values()` applied; cross-file `.param` plumbing from
M2's own `.param` block per Stage 5 fix at 13:08).

**ngspice ground-truth (interactive `showmod m1` on the same M2 card,
verified at 13:30):** all binning coefficients load correctly:
`wvth0=-1.66e-8`, `pvth0=-1.45e-15`, `wvoff=-5.6e-9`, `voffl=-5.6e-9`,
`pvsat=1.03e-9`, `pags=3e-13`, `lpe0=1.244e-7`, `phin=0.05`, etc.

## Per-parameter binning breakdown (pyport, binunit=2)

`P_eff = base + l_X·(1/Leff) + w_X·(1/Weff) + p_X·(1/(Leff·Weff))`

| param | base | +l/Leff | +w/Weff | +p/(L·W) | effective | shift |
|-------|---:|---:|---:|---:|---:|---:|
| vth0 | 0.54153 | 0 | -0.05479 | -0.00270 | 0.48404 | -57 mV |
| voff | -0.1368 | 0 | -0.01852 | 0 | -0.15532 | -19 mV |
| vsat | 102230 | 0 | 0 | +1918 | 104148 | +1.9% |
| **ags** | **0.34914** | 0 | 0 | **+0.559** | **0.90785** | **+60%** |
| u0 | 0.04832 | 0 | 0 | -2.2e-4 | 0.04809 | -0.5% |
| nfactor, k1, k2, k3, lpe0, etc. | (no binning coefs in card) | | | | unchanged | 0 |

Plus `voffcbn = voff + voffl/Leff = -0.13995` (special term, not via
the standard l/w/p triplet).

## Findings

1. **Pyport's binning arithmetic matches the BSIM4 v4.8.3 manual
   formula exactly** for `binunit=2`. No off-by-µm scaling, no swapped
   Leff/Weff, no missed coefficients.

2. **`ags` is the loudest correction by far.** It controls the
   Vbs-dependent reduction of velocity saturation in BSIM4 §6.5.
   A +60% shift in `ags` would propagate strongly into `Vdsat` and
   thence into both saturation `Id` and the pre-saturation transition.

3. **Therefore the 0.87-decade gap (Stage 5 finding) is NOT a
   binning-arithmetic bug.** It is one of:
   - (i) ngspice's effective `ags` differs from pyport's
     (i.e., the C source applies a different formula for `pags` —
     unlikely given §5.1 is unambiguous, but worth one direct probe);
   - (ii) downstream consumption of the +60%-shifted `ags` differs
     between pyport's `dc.py` and ngspice's `b4ld.c`;
   - (iii) some other parameter-cascade we haven't audited.

## Next probe (Stage 6b v2)

Run ngspice with instance-level query `print @m1[ags]` on the same
M2 geometry — that reads the C struct's per-instance binned value.
- If it equals 0.90785 → divergence is downstream (option ii or iii).
- If it equals 0.34914 or other → divergence is at the binning layer.

Either outcome localises the gap in one query, ~10 min wall.

## Why this matters for the brief

The brief currently commits "1-2 days coding" for M3a closure. The
v1 probe localises the search space to ~3 candidate buckets. If
probe v2 confirms downstream consumption (most likely), the fix is
in `dc.py`'s `ags` consumer (Vdsat assembly). If it's at the binning
layer, the fix is more involved (BSIM4 manual cross-check + C source
audit). Either way, the 1-2 day estimate is still the right order
of magnitude — but the "verification + 33-bias regression re-runs"
phase (~3 days) will look identical regardless of which bucket it
turns out to be.
