# DC Solver Default Changed: Newton-DC → Pseudo-Transient Backward

**Date**: 2026-05-17
**Change ID**: z462b
**Scope**: `scripts/z429_multisolver_debug.py::run_vsint_pinned`

## What changed

The default DC solver for `run_vsint_pinned(cfg, ..., Vd_f, VG1_f, VG2_f, ...)`
flipped from **1-D damped Newton on V_B** to **pseudo-transient (PT) Euler
integration of `C_B·dV_B/dt = R_B(V_B)`** (the z432 reference settings:
`C_B=1e-18 F`, `dt=1 ns`, `N_steps=800`, `Vb ∈ [-0.2, 1.0]`, terminate when
`|ΔV_B|<1e-5` and `|R_B|<1e-4·max(|I_D|, 1pA)` after ≥100 steps).

The dispatcher reads `NSRAM_DC_SOLVER` env var (default `"pt"`). The legacy
Newton path is retained as `_run_vsint_pinned_newton` and invoked when
`NSRAM_DC_SOLVER=newton` or `method="newton"` is passed explicitly — it now
emits a one-shot `DeprecationWarning`.

New helper: `run_vd_sweep_pt_backward(cfg, ..., Vd_seq, ...)` — sweeps
V_D **high → low** with V_B warm-started from the previous high-V_D
attractor, returning results in the original `Vd_seq` order. This is the
recommended path for full I-V characterization (z432 style).

## Why

- Newton-DC on `R_B(V_B)` finds the **nearest root**, which on the cold-start
  path is always the low-current root → the snap-up at the bistability knee
  is never visible at the I-V level (z429, z461 V1 plot went flat at the
  knee).
- Real silicon (Sebas 130 nm data) shows a 2-3 decade jump at V_D ≈ 1.5-1.7 V
  for VG1=0.6.
- z432 demonstrated that PT integration with body-cap relaxation lands on the
  latched attractor when sweeping V_D from above (backward sweep RMSE 1.027
  dec vs forward Newton 1.349 dec on the same cell-wide grid).
- z461 V4-V7 transient validators already use BDF time integration and
  reproduce snap-up — only the DC characterization path was lagging.

## Migration notes

- **Signature unchanged**: same args, same return-dict keys (with an extra
  `niter` field added — non-breaking).
- **Numerical results will shift**: any script comparing absolute I_D values
  against frozen baselines from before 2026-05-17 must be re-baselined.
  Notably affected: cell-wide RMSE summaries in z429, z430, z433, z435,
  z437, z443, z446, z449, z454, z456, z458, z460, z461.
- **Convergence semantics**: PT returns `converged=True` when the relaxation
  has settled (`|ΔV_B|<1e-5` AND `|R_B|<rel_tol`) **or** when the final
  residual passes the legacy `|R_B|<1e-8` test. Scripts that filter out
  unconverged points should continue to work; the converged set will
  typically be larger.
- **Opt-out**: `export NSRAM_DC_SOLVER=newton` (emits DeprecationWarning) or
  call `run_vsint_pinned(..., method="newton")`.

## Performance

- Newton: ~80 evals × 2 (for finite-diff `dRdV`) ≈ 160 resid_pair calls per
  bias point.
- PT: up to 800 evals per point, typical 200-400 with early termination.
- **Estimated cost ratio ≈ 2× per bias** (matches sanity-test wall-time
  observation, see `results/z462b_pt_default/run.log`).

## Why no other files edited

All other scripts that import `z429.run_vsint_pinned` automatically inherit
the new default because the dispatcher lives at the entry point. No
behavioral assumptions in those scripts depend on Newton being used — they
all consume the returned `Id` / `Vb` / `converged` fields, which PT
populates with the same semantics.
