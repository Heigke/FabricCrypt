# M3c.3 — Local-base node + base-spreading Rb refactor

**Drafted:** 2026-05-04 ~09:05. **Status:** decision pending.
**Trigger:** O21 verdicts (3-of-3 Vb-clamp confirmed; 2-of-3 α
recommended) + my own Bf sweep + BJT-disable empirical tests.
The β path was empirically refuted; the α path is the only
coherent fix.

## What we are fixing

The body KCL is currently a single-node model:
```
Vb_global = (Vsint, Vb)  ← Newton 2D
```
The BSIM4 impact-ionisation Iii injects holes here, AND the
parasitic-NPN base sees the same Vb. At honest Bf=100, the BJT's
exponential Ib draw dominates the body KCL, flattening Vb across
biases (5 mV spread regardless of inflow magnitude).

In real silicon, the parasitic NPN has a finite **base spreading
resistance** Rb between the *local* point of Iii injection and
the *global* body region where the BJT's vertical base sees. Iii
can elevate the local potential significantly above the global
clamp, modulating Vbe_local without forcing Vb_global to follow.

## What changes structurally

  - Newton state expands from 2D `(Vsint, Vb)` to 3D
    `(Vsint, Vb, Vb_local)`.
  - Add new node `Vb_local` between Iii injection and the BJT base.
  - Add new KCL: `R_local(Vb_local) = 0`
  - The BJT now sees `Vbe_local = Vb_local`.
  - Body diodes / well diodes remain on `Vb_global`.

### KCL for the new local-base node

```
R_local(Vb_local) =
    + (1 − η_lat) · iii_gain · Iii_M1               # Iii injection here
    + Igidl_M1 + Igisl_M1                            # GIDL pumps here too
    + Ib_lat_pair                                     # M3c.1 lateral pair (toggle)
    − Ib_Q1(Vb_local, Vbc=Vb_local−Vd)               # BJT draws from local
    − (Vb_local − Vb) / Rb                           # spread resistor to global
```

### Modified KCL for `Vb_global` (was `Vb`)

```
R_B(Vb) =
    + (Vb_local − Vb) / Rb                           # spread current arrives
    + Igb_M1                                          # gate-body still on global
    − m1_d · (Ibs_M1 + Ibd_M1)                       # body diodes on global
    + I_well_body − I_body_pdiode                    # well/well-body on global
```

(`Vsint` KCL unchanged.)

### Drain accounting

Same as before — Ic_Q1 from BJT, but Ib_Q1 is now a function of
`Vb_local`, not `Vb`.

## Default parameters (regression-test gate)

  - `cfg.use_local_base: bool = False` → identical to F1.v2 (Newton
    stays 2D, no Vb_local node).
  - `cfg.Rb: float = 1e6 Ω` (when toggle on; physically plausible
    for 130 nm lateral parasitic NPN per literature).
  - `cfg.Cb: float = 0.0` (DC; transient extension is M3c.3.b).

### Gate criteria

  1. `use_local_base=False` → bit-identical Id to current F1.v2
     (regression).
  2. `use_local_base=True, Rb=0` → Newton 3D collapses to 2D
     case (Vb_local = Vb forced); should match `use_local_base=False`
     within Newton tolerance.
  3. `use_local_base=True, Rb=1e6` at canonical biases → Vb_local
     should be measurably above Vb_global; Id should show
     bias-dependent variation (not the 5-mV-flat we have now).

## Implementation sketch

Files affected:
  - `nsram/nsram/bsim4_port/nsram_cell_2T.py` (~200 LOC change):
    * `_residuals` returns 3-vector `(R_S, R_B, R_local)`
    * Newton solve_2t → solve_3t (or guard via `use_local_base` flag,
      keep both paths)
    * Jacobian 3×3 instead of 2×2 (analytical autograd preferred
      for speed; finite-diff as fallback)
    * Arclength continuation extends to 3D state
  - `nsram/nsram/bsim4_port/bjt.py`: no change (BJT just sees
    different Vbe).
  - `scripts/test_m3c3_gate.py` (new): regression + sensitivity
    tests per the gate criteria above.

Risk: Newton 3D may not converge as cleanly as 2D. Need warm
start from 2D solution + arclength fold-following. Estimated
debug time: 1 day for solver, 1 day for Jacobian + sensitivity
testing, 1 day for full 33-row refit.

## Pre-registered halt criteria

If after M3c.3 implementation:
  - Newton non-convergence at > 5% of biases at default Rb → halt;
    re-examine state choice.
  - Median dec ≥ 1.20 across reasonable Rb sweep → halt; engage
    O22 oracle round.
  - Any new fit param > 1 OoM outside its physical bound → halt;
    repeats M3a/M3b trap.

## Comparison to M3c plan as originally drafted

  - M3c.1 (electron-hole pair accounting) — DONE, gate passes.
  - M3c.2 (lateral-NPN-as-channel-current) — REFUTED by O21 + my
    own data. Both paths B and C cannot work because Ic_Q1 floor
    dominates Id at all biases.
  - **M3c.3 (this plan) — replaces M3c.2 as the primary structural
    fix.**
  - M3c.4 (smooth gating M_safe) — unchanged from original plan,
    applicable inside M3c.3 if avalanche path is also added later.
  - M3c.5 (charge conservation assertion) — unchanged.

## What this does to the brief / Mario / NRF

  - The brief addendum's M3 timeline statement ("~6 weeks of dev
    work") becomes:
    * M3c.1 (DONE)
    * M3c.3 (this plan, ~1 week dev + refit)
    * M3c.4–.5 (~1 week if needed)
    * M3c-A,B,C (re-run benchmarks, ~3 days)
    * **Total: ~3 weeks calendar to a sendable result, NOT 6.**
  - The reduced timeline reflects that O21's diagnostic test cut
    out the M3c.2 dead-end early.

## Status

This document is the M3c.3 plan. Code work starts when:
  - User authorises Newton 3D refactor (M3c.3 is non-trivial).
  - OR oracle round O22 (or equivalent) reviews this plan and
    confirms direction.

Until then, code state is: M3c.1 + M3c.2 paths B & C committed,
gates pass, defaults reproduce F1.v2.
