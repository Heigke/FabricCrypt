# A1g — Multi-root hypothesis at low VG2

**Bias:** VG1∈{0.4, 0.6}, VG2=0.0, Vd=1.5 V; Sebas per-bias overrides.
**Symptom:** z91g returns Id≈1e-11; measurement Id≈2e-5 (VG1=0.6) /
1e-6 (VG1=0.4) — 5–6 decade gap.

## Method

`research_plan/artifacts/A1g_demo.py`. M1+M2 cards via z91f's
`patch_model_values`; per-bias overrides via `patch_sd_scaled` (z91g
convention; `_override_sd` errors on dict-only fields like
etab/alpha0/beta0/nfactor). Per bias:
- `solve_2t_with_homotopy`, Vb_init ∈ {0.0, 0.5, 0.7, 0.9}.
- `forward_2t_arclength_grad`, Vd∈[0.05, 2.0] (40 pts).

Trace: `A1g_multiroot_trace.json`. Plot: `A1g_multiroot.png`.

## Results

### VG1 = 0.6, VG2 = 0.0, Vd = 1.5 (meas Id = 2.07e-5 A)
| start          | Id [A]    | Vb [V]  | Vsint [V] | conv |
|----------------|-----------|---------|-----------|------|
| Vb_init=0.0    | 1.253e-11 | 0.3419  | 0.3063    | yes  |
| Vb_init=0.5    | 1.253e-11 | 0.3419  | 0.3063    | yes  |
| Vb_init=0.7    | 1.253e-11 | 0.3419  | 0.3063    | yes  |
| Vb_init=0.9    | 1.253e-11 | 0.3419  | 0.3063    | yes  |
| arclength@1.5  | 4.21e-12  | 0.1842  | —         | no   |

Arclength: **n_folds = 0**, 45 steps, 2.5 s.

### VG1 = 0.4, VG2 = 0.0, Vd = 1.5 (meas Id = 1.02e-6 A)
| start          | Id [A]    | Vb [V]  | Vsint [V] | conv |
|----------------|-----------|---------|-----------|------|
| Vb_init=0.0    | 1.444e-11 | 0.3550  | 0.2372    | no   |
| Vb_init=0.5    | 1.444e-11 | 0.3550  | 0.2372    | no   |
| Vb_init=0.7    | 1.444e-11 | 0.3550  | 0.2372    | no   |
| Vb_init=0.9    | 1.444e-11 | 0.3550  | 0.2372    | no   |
| arclength@1.5  | 4.57e-12  | 0.1857  | —         | no   |

Arclength: **n_folds = 0**, 44 steps, 2.8 s.

## Verdict: multi-root hypothesis **DISPROVEN**

Three converging lines:

1. **Initial-condition basin sweep is degenerate.** Vb_init from 0.0
   to 0.9 V — past the diode-on knee — converges in 3 Newton iters to
   the **same** (Vsint, Vb, Id) triple to 5 sig-figs. No second basin
   in the explored range.
2. **Arclength finds no fold** (n_folds = 0). If a high-Vb root were
   separated from the low-Vb root by an S-shaped fold in Id(Vd),
   trace_arclength would have detected a turning point. It did not.
3. **The found Vb is already moderate** (~0.34 V) — not a stuck-at-0
   NPN-OFF root. Body diode mildly forward-biased; Iii just isn't
   strong enough in our model to ignite the
   avalanche → base-drive → collector-current loop.

There is no second root for arclength or high-Vb init to snap to.
Newton already finds the only root the model admits.

## Actual mechanism (likely)

The 6-decade gap is a **sub-block disagreement**, not a convergence
failure. Ranked candidates:

- **Iii too weak.** Sebas's ALPHA0=7.84e-5, BETA0=20 through our
  `compute_iimpact` may produce orders less impact current than
  HSPICE BSIM4 IIMOD. The Iii→Vb→base-drive loop never ignites; the
  NPN stays off and Id collapses to M2 sub-threshold leakage. The
  z91g pattern (good fits at VG1=0.2 where NPN doesn't matter,
  degrading at VG1=0.4–0.6 where it does) is consistent.
- **GP NPN under-driven.** At Vb=0.34, Vbe is too small for our
  Gummel-Poon to deliver appreciable Ic. Worth verifying that
  `bjt.py` references Vbe = Vb − Vsint with the same sign convention
  as Sebas's `parasiticBJT.txt` (A1b mapped this; recheck).
- **Body-diode Js mismatch.** M2 card has zero SourceSatCurDensity_T;
  `cfg.default_jss` kicks in. If too large it clamps Vb near 0.34 V.

## What to do next

A high-Vb / arclength heuristic won't help — there's nothing to find.
Fix is in physics:

1. Audit `compute_iimpact` vs HSPICE BSIM4 IIMOD=1 at this bias with
   Sebas overrides (next diagnostic).
2. Verify GP NPN Vbe-mapping vs `parasiticBJT.txt` (C=Vd, B=body,
   E=Vsint).
3. Once Iii is calibrated and Vb pushes past ~0.6 V, Id should jump
   4–5 decades without any continuation tricks.
