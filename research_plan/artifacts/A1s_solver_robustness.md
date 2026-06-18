# A.1.s Arclength Solver Robustness

**Goal:** unlock `vnwell_Rs ≥ 3e9` (needed for the snapback knee) while keeping
full 25/25 curve coverage in `z91g_two_model_validation`.

## Baseline (v14 / v16, Rs=3e9)
- **Coverage**: 14–19 / 25 curves with `conv > 0`. Entire VG1=0.4 family was
  `conv=0/30` (interpreted by z91g as `rmse=inf`).
- **Median log-RMSE**: 0.894 (v14), 0.817 (v16) — but those numbers exclude
  the `inf` rows so they over-flatter the real fit quality.

## Root cause
The arclength corrector ran with `tol=1e-9 A` while the well-body diode at
`Rs=3e9, Vd=2V` injects `~6.7e-10 A`. The corrector was therefore happy to
declare a "root" at `Vb≈0` even though the body KCL was unbalanced by exactly
the well current (`R_B = 6.67e-10 ≈ I_well`). Per-bias `solve_2t_steady_state`
uses `Iabstol=1e-12, Ireltol=1e-3` — the warm-started "root" failed those, the
Newton loop couldn't find descent (it was already at a flat residual basin),
and z91g saw `converged=False` for all 30 Vd points → `rmse=inf`.

So this was **not** a Newton-overshoots-the-fold problem; it was a
silent-too-loose-tolerance problem that produced a non-physical off-branch
path. The fold-tracking machinery had already found a "valid enough" path at
`Vb≈0` that no per-bias polish could escape.

## Fix (`nsram/bsim4_port/arclength.py`)
1. **Tighten corrector / initial-point tolerances** from `1e-9` to `1e-13 A`.
   This forces `_newton_arclength_corrector` and `_solve_initial_point_single`
   to find an actually-physical root before adapting `ds`. Single highest-impact
   change.
2. **Multi-restart `_solve_initial_point`**: try 6 seeds covering both the cold
   off-branch (`Vb=0`) and the hot on-branch (`Vb=0.75–0.85`) families so a
   snapback-only initial point is reachable when the cold seed sits in a basin
   that the corrector can't cross.
3. **Skip-and-continue on persistent corrector failure**: instead of recording
   one bogus unconverged point and breaking, take a `ds_min` step along the
   tangent and try to re-acquire the path. Only abort after 6 consecutive
   failures. Keeps the post-failure portion of the curve traceable.
4. **Perturbed-Vb predictor restart**: when corrector fails at `ds_min`, jitter
   the predicted `Vb` by `±0.05–0.30 V` before bisecting further (snapback
   instability is dominantly in the body voltage; this is the cheap multi-start
   that brings the corrector back to a real root).
5. **Tangent-rotation step shrink**: shrink `ds` aggressively when the tangent
   rotates (`>30°` ⇒ `0.4× ds`, `>60°` ⇒ `4× ds_min`). Smooths approach to
   folds when they exist.
6. **Backward sweep + merge**: if the forward path doesn't reach `Vd_max`, run
   a backward pass from `Vd_max` with a hot on-branch seed (`Vb=0.85`) and
   concatenate. Interpolation is direction-agnostic so the segments stitch
   naturally.

## Result at Rs=3e9 (`run_v17_solver_robust.log`)
- **Coverage: 25/25 curves at 30/30 (100%)** — was 14/25 baseline.
- **Median log-RMSE: 0.896** (target ≤ 0.8 — close but not quite met).
- **p90 log-RMSE: 2.347** (target ≤ 1.7 — VG1=0.4 group is the residual; new
  RMSEs there are 2.0–2.5 because the per-bias solver now finds the *real*
  high-Vb on-branch but the bias-data row's BSIM params don't put the curve
  there — physics mismatch, not solver mismatch).

## Result at Rs=1e10 (`run_v17b_Rs1e10.log`)
Coverage 25/25 also, median 1.129, p90 1.898. Worse median than 3e9, so 3e9
is the better default.

## Tests
- Pre-existing failing tests: 6. After fix: **5** (`test_homotopy_converges_at_snapback_bias`
  now passes thanks to the tighter corrector tolerance). No new failures, no
  API breakage.

## Defaults left in place
- `NSRAMCell2TConfig.vnwell_Rs = 3.0e9` (was 1.0e9).
- `arclength.py` corrector tolerances 1e-13.

## Honest caveats
- `n_folds=0` is reported on every traced path even though the curves are
  visibly snapback. Either the well-Rs branch already pre-closed the fold
  (smooth Vb sweep → no `dVd/ds` sign change) or the tangent-sign detector
  isn't catching it. Worth a follow-up if a true bistable trace is needed.
- VG1=0.4 RMSE jumped from `inf` → 2.0–2.5. That's now a *physics*-side gap
  (BSIM α0/β0/etab from Sebas's CSV don't generate the right Iimpact at this
  bias). The user's earlier z91h grid search noted Bf=5e4·α0×10 closed it but
  cost coverage; with v17's solver we may now be able to revisit those grid
  points.
- Strict `≤ 0.7 dec` median criterion was not reached at Rs=3e9. Closer to it
  would require either physics tuning (β0/α0 sweep, now feasible) or moving
  beyond pure-arclength (PT continuation, deflation).
