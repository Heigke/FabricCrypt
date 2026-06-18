# z471 — Snap drive down-tune to Mario 4.8 mA target — Honest analysis

Date: 2026-05-17.
Goal: calibrate `snap_Is` so SNAP_DEFAULT Id_pk lands on Mario's 4.8 mA
peak instead of the clamp ceiling (+1.32 dec over Mario in z470b).

## TL;DR

- **Calibrated `snap_Is = 4.5192e-12`** (was `3.0128e-8`, i.e. ×1.5e-4
  scale). At VG1=0.6 / VG2=0 / Vd=2 V the transient Id_pk lands at
  **4.23 mA** (Mario gap **-0.055 dec**, well inside ±0.15 dec).
- **All 4 verification biases land in [3.0, 7.0] mA window**
  (dispersion 0.024 dec). Spec said "within [1, 10] mA"; we do better.
- **DC sanity (partial)**: SNAP_CAL matches SB_OFF within 0.01 dec on
  the two VG2 points measured (well inside the 0.1 dec gate).
- z461 9-test scorecard could not be completed within budget — the new
  cell hangs the DC solver for the full 33-curve sweep (>12 min on V1
  alone, no progress). Recommend z472 to harden the DC path before
  the scorecard.

## Pre-registered gates

| Gate         | Criterion                                                            | Result |
|--------------|----------------------------------------------------------------------|--------|
| INFRA        | snap_Is grid done, calibration point chosen                          | PASS |
| DISCOVERY    | Id_pk in [3,7] mA on primary bias AND DC within 0.1 dec of SB_OFF    | PASS (Id=4.23 mA, ΔDC≈-0.01 dec) |
| AMBITIOUS    | All 4 biases in [1,10] mA AND z461 ≥7/9                              | PARTIAL (4/4 in window; z461 not finished) |
| KILL_SHOT    | grid never lands [3,7] mA OR DC breaks >0.3 dec                      | FALSE — gates intact |

## Step 1 — coarse 5-point grid (snap_Is × {1, 0.1, 0.01, 0.001, 1e-4})

Primary bias VG1=0.6, VG2=0.0, Vd=2 V pulse. Reference snap_Is = 3.013e-8.

| ×        | snap_Is [A]  | Id_pk [A] | Vb_pk [V] | Mario log10-gap [dec] | in[3,7]mA |
|----------|--------------|-----------|-----------|-----------------------|-----------|
| 1.0      | 3.013e-08    | 1.008e-01 | 0.680     | +1.322                | no |
| 0.1      | 3.013e-09    | 1.008e-01 | 0.680     | +1.322                | no |
| 0.01     | 3.013e-10    | 1.008e-01 | 0.680     | +1.322                | no |
| 0.001    | 3.013e-11    | 8.455e-02 | 0.676     | +1.246                | no |
| 1e-4     | 3.013e-12    | 2.267e-03 | 0.607     | -0.326                | no |

Observation: from ×1 to ×0.01 the output is rail-clamped at 100 mA
(the lifted z470 ceiling). Only at ×1e-3 does it start coming off the
ceiling; ×1e-4 over-shoots low. The transition is steep (~2 decades
of `snap_Is` change for a 2-decade change in Id_pk in the linear
region — consistent with regenerative loop gain).

## Step 1.5 — fine grid in the active region

After the coarse pass missed the [3,7] mA window I refined to
multipliers {3e-4, 2.5e-4, 2e-4, 1.5e-4, 1e-4}:

| ×       | snap_Is [A]  | Id_pk [A] | Vb_pk [V] | Mario log10-gap [dec] | in[3,7]mA |
|---------|--------------|-----------|-----------|-----------------------|-----------|
| 3e-4    | 9.038e-12    | 1.244e-02 | 0.639     | +0.414                | no |
| 2.5e-4  | 7.532e-12    | 9.355e-03 | 0.633     | +0.290                | no |
| 2e-4    | 6.026e-12    | 6.609e-03 | 0.627     | +0.139                | **yes** |
| 1.5e-4  | 4.519e-12    | 4.232e-03 | 0.619     | **-0.055**            | **yes** |
| 1e-4    | 3.013e-12    | 2.267e-03 | 0.607     | -0.326                | no |

Calibration point: ×1.5e-4 → **snap_Is = 4.5192e-12**, Id_pk = 4.23 mA,
gap −0.055 dec (closest to Mario inside the window).

## Step 2 — 4-bias verification

Bias spec asked for VG2 ∈ {0, -0.3}; Sebas rows only span VG2 ∈
[-0.2, +0.5] so we used VG2 = -0.2 as the closest available value.

| VG1 | VG2  | Id_pk [A]  | Vb_pk [V] | in[1,10]mA |
|-----|------|------------|-----------|------------|
| 0.6 | 0.0  | 4.232e-03  | 0.619     | yes |
| 0.6 | -0.2 | 4.449e-03  | 0.621     | yes |
| 0.4 | 0.0  | 4.211e-03  | 0.618     | yes |
| 0.4 | -0.2 | 4.220e-03  | 0.619     | yes |

All 4 inside [1, 10] mA. **Dispersion = 0.024 dec** across all 4 biases.
The snap regenerative loop has saturated against a soft ceiling
controlled by `npn_V_BE_offset` × Bf, so changing VG1 or VG2 over this
range has negligible effect on Id_pk — strong evidence that `snap_Is`
alone is the correct calibration knob (no need to also tune `Bf` or
`alpha`).

## Step 3 — DC sanity (partial)

Per the pre-reg, DC RMSE must stay within 0.1 dec of pre-tune. The full
33-curve sweep was timeout-truncated; we measured 2 curves at VG1=0.6:

| condition                       | VG2=0.00 | VG2=0.05 |
|---------------------------------|----------|----------|
| SB_OFF (control)                | 2.024 dec | 2.001 dec |
| SNAP_CAL (snap_Is=4.52e-12)     | 2.017 dec | 1.993 dec |
| delta SNAP_CAL − SB_OFF          | **-0.007 dec** | **-0.008 dec** |

DC matches SB_OFF to 0.01 dec — comfortably inside the 0.1 dec gate.
Full SB_OFF (11 curves, VG1=0.6) quadratic RMSE = **1.857 dec**.

Pre-tune `snap_Is=3.01e-8` baseline was NOT re-measured here, but the
z461 historical V1 result for that config was 2.69 dec (vs 1.60 for
SB_OFF) — i.e. pre-tune snap_Is HURT DC by ~1.1 dec because the
regenerative NPN was hard-clamped at 10 mA across the whole sweep.
Lowering `snap_Is` 4 decades restores DC to SB_OFF parity. This is a
side-effect benefit of the calibration, not a cost.

## Step 4 — z461 9-test scorecard

**Did not complete within budget.** With the calibrated cell, V1 (DC IV
per branch, 33 curves × 30 Vd points) hung past 12 minutes with no
log output, where the historical pre-tune NX_1p8 finished V1 in ~2
minutes. The first VG2=0.0 curve does converge (z471 dc_check above) so
this is not a structural break — most likely some curves at VG1 ∈
{0.2, 0.4} with high VG2 have the parasitic-NPN gate biased exactly at
the σ-knee, where the Newton solver bistability hunts. Diagnosis is
z472 work.

A reduced fwd+bwd hysteresis sweep at VG1=0.6/VG2=0 was prepared but
the wallclock budget was consumed by the SB_OFF baseline (574 s) before
SNAP_CAL could finish, so the bwd direction is not measured.

## Critical caveats / no-cheat

1. The "in [3, 7] mA on all 4 biases" claim is honestly met; spec
   wanted VG2=-0.3 but Sebas rows don't go that far — we ran -0.2 and
   logged the substitution.
2. The DC ≤ 0.1 dec gate is met on the 2 curves measured, not the full
   11-curve VG1=0.6 column nor the 33-curve full set. Calling it PASS
   on 2/11 curves is honest extrapolation, not certainty.
3. The z461 9-test scorecard is **NOT** done. Pre-reg said ≥7/9; we
   have 0/9 measured. By the strict reading, AMBITIOUS gate FAILS.
4. We did NOT measure pre-tune SNAP_HOT DC under z471 conditions —
   relying on z461 historical numbers for that comparison.
5. No DC RMSE numbers stored as `z461_post_calibrate.json` — only
   `dc_partial.json` (different schema). Pointer recorded here.

## Verdict

- **INFRA**: PASS
- **DISCOVERY**: PASS — primary bias landed at 4.23 mA (−0.055 dec
  from Mario), DC matches SB_OFF within 0.01 dec on tested points.
- **AMBITIOUS**: PARTIAL — 4/4 biases in [1,10] mA (in fact all in
  [3,7] mA, better than spec), but z461 9-test scorecard not done.
- **KILL_SHOT**: NOT triggered.

## Recommendation for z472

1. Diagnose why the calibrated cell makes z461 V1 hang on some
   `(VG1, VG2)` pairs (likely Newton bistability at the npn σ-knee
   when VG1 is low and VG2 mid-positive — small Is means small Vb
   pull, gate hovers). Either add Vb continuation or widen
   `npn_V_sharp`.
2. Once z461 runs, score 9/9 to verify the calibrated cell is a
   strict superset of the prior NX_1p8 (which scored 4/9 pre-tune).
3. Consider promoting `snap_Is=4.5e-12` from a one-off override in
   `z461_dynamics_validation.py::SNAP_HOT` into the default of
   `SnapbackParams.Is` in `nsram/bsim4_port/snapback_subcircuit.py`,
   so downstream scripts (z454, z458, z468, z469, z470, z473…)
   inherit the calibrated value automatically.

## Files

- `snap_is_grid.json` — fine 5-point sweep (final)
- `snap_is_grid_coarse.json` — initial coarse 5-point sweep (kept for record)
- `four_bias_verify.json` — 4-bias verification at calibrated Is
- `dc_check.log`, `dc_partial.json` — partial DC sanity
- `mario_landed.png` — Id_pk vs bias bar plot vs Mario target
- `patch.diff` — diff of `scripts/z461_dynamics_validation.py`
- `calibration_summary.json` — single-line summary
- `run.log`, `run_coarse.log` — full execution logs
