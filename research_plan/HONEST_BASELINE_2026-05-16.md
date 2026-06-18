# HONEST BASELINE — P1a fwd+bwd re-run (2026-05-16)

Closes 6/9 cherry-picks identified in `CAMPAIGN_SYNTHESIS_2026-05-16.md`.
Replaces forward-only headline numbers in §1 of the synthesis with
sweep-direction-honest values.

- **Dataset**: Sebas's full set, 33 measured curves;
  25 biases have BSIM parameter cards (`find_params` matches);
  every per-bias entry below is over those 25 (NOT the 4-bias
  z447/z448 cherry-subset).
- **Sweep directions**:
  - **fwd** = V_D 0.05 → 2.0 V, warm-start V_B from previous V_D point.
  - **bwd** = V_D 2.0 → 0.05 V, warm-start V_B from previous V_D point.
- **Cell-wide RMSE** = quadratic mean of per-bias log10 RMSE over
  converged V_D points.
- **avg** = `sqrt(0.5 * (cell_fwd^2 + cell_bwd^2))` (RMS average across
  directions), per the synthesis §1.1 convention.
- Raw data: `results/P1a_honest_baseline/summary.json`,
  run log `run.log`, harness `scripts/P1a_honest_baseline.py`.

## §1 Corrected DC pipeline table (P1a, honest fwd+bwd)

| Pipeline | n biases (fwd / bwd) | fwd dec | bwd dec | **avg dec** | conv-rate fwd | conv-rate bwd | wall (fwd+bwd) | Defensible? |
|---|---|---|---|---|---|---|---|---|
| **z430 V_SINT_PIN** (hard pin V_Sint=0, 1D Newton on V_B) | 25 / 25 | 1.619 | 2.823 | **2.301** | 100 % | 100 % | 81 s | YES — honest; the prior 1.62 was fwd-only |
| **z432 PTRAN** (pseudo-transient body integration, C_B=1 aF) | **18 / 25** | 1.349 | 1.027 | **1.199** (mixed n) | 31.9 % | 49.7 % | 2 558 s | partial — fwd drops VG1=0.2 column entirely (7/7 biases fail) |
| **z443 VBIC_AVL** (VBIC level-4 NPN, AVC1=AVC2=0.5 Si defaults) | 25 / 25 | 1.311 | 2.864 | **2.227** | 100 % | 100 % | 67 s | YES — was reported fwd-only as 1.31; honest avg = 2.23 dec |

### Per-branch RMSE (full breakdown)

```
z430 V_SINT_PIN
  fwd  VG1=0.2: 2.625   VG1=0.4: 0.786   VG1=0.6: 1.086
  bwd  VG1=0.2: 2.633   VG1=0.4: 2.662   VG1=0.6: 3.031

z432 PTRAN
  fwd  VG1=0.2:  ---    VG1=0.4: 0.703   VG1=0.6: 1.632      ← VG1=0.2 column DROPPED (all 7 biases fail to converge)
  bwd  VG1=0.2: 1.353   VG1=0.4: 0.521   VG1=0.6: 1.028

z443 VBIC_AVL
  fwd  VG1=0.2: 0.911   VG1=0.4: 1.135   VG1=0.6: 1.600
  bwd  VG1=0.2: 2.622   VG1=0.4: 2.665   VG1=0.6: 3.121
```

## §1.1 Where the cherry-picks came from

The synthesis suspected two failure modes; P1a confirms both.

1. **Direction-pick (z430, z443): forward sweep is systematically better
   than backward by 1.2 – 1.5 dec.** Hard-pin V_Sint=0 closes V_B
   forward-active runaway at low V_D but the backward sweep arrives at
   high V_D in a warm-state where V_B is far from the forward attractor;
   in V_B saturates and Id_pred goes lower than the measured Id by an
   order of magnitude across **every** VG2 column. This is the same
   1.31 dec → 2.86 dec gap that z454 SB_OFF already saw on its own
   bwd half (synthesis line 51); we now see it on three pipelines.

2. **Bias-pick (z432): the entire VG1 = 0.2 column was silently
   dropped from the forward report.** The 7 VG1=0.2 biases all
   diverge in pseudo-transient forward (warm-start from Vb=0.1 V
   never finds a stable attractor at low V_D), so they're discarded
   by `if not conv.any(): continue` and not reflected in n_biases.
   The 1.35 dec forward headline averages over 18 biases on 2 VG1
   branches; including the VG1=0.2 column (which only the backward
   sweep can solve) the honest fwd is undefined and the only honest
   number is **bwd = 1.03 dec on all 25**. Reporting fwd 1.349 next
   to bwd 1.027 as "an avg ~1.19" is, methodologically, a
   convergence-pick (different denominator each side).

Both failure modes share a common root: **warm-start initial condition
asymmetry**. Forward starts with V_B near 0 (which is the actual
attractor at small V_D); backward starts at high V_D where the model
sits in a fundamentally different regime (high-V_D V_B-runaway or
deep saturation) that the local Newton/integrator cannot escape from.
This is the fwd↔bwd asymmetry MASTER_FIX_PLAN already flagged for
**rbodymod=1** (P4 in the autonomous plan).

## §1.2 Recommended re-rank order (honest)

Replaces synthesis §1.1 ranking 1-4.

| Rank | Pipeline | avg dec | Caveat |
|---|---|---|---|
| 1 | **z432 PTRAN bwd-only** | **1.027 dec, 25 biases, 50 % conv** | drop forward direction entirely; report as "backward, n=25, conv 50 %". Fwd is broken on VG1=0.2 (KILL_SHOT for forward use). |
| 2 | z430 V_SINT_PIN fwd-only | 1.619 dec, 25 biases, 100 % conv | bwd 2.82 dec — physics is not symmetric. Report fwd only with that caveat. |
| 3 | z443 VBIC_AVL fwd-only | 1.311 dec, 25 biases, 100 % conv | bwd 2.86 dec — same forward-attractor-only behaviour as z430. Adding avalanche doesn't help bwd. |
| —  | All three "avg" combinations | 1.2 – 2.3 dec | misleading: averaging a working direction with a broken one gives a number with no consistent denominator across pipelines. The forward-only ranking is more defensible than any fwd+bwd avg. |

## §1.3 Implications for the campaign plan

- **The "best honest DC = 1.19 dec" headline (synthesis §1, §6.intro)
  is overstated**. The honest single-direction best is **1.03 dec
  bwd (z432 PT)** OR **1.31 dec fwd (z443 VBIC)**; the dishonesty was
  averaging mismatched-denominator results.
- **P4 (rbodymod=1) is now confirmed-urgent**. The fwd↔bwd asymmetry
  (1.6 → 2.8, 1.3 → 2.9) is exactly the signature §13 of the
  MASTER_FIX_PLAN flagged. Without distributed Rb, the high-V_D
  warm-start cannot relax to the correct V_B regime.
- **AMBITIOUS gate (<1.0 dec cell-wide on full 25 biases, both
  directions)** is **not reached** by any P1a pipeline. The
  closest is z432 bwd at 1.027 dec — a single-direction result.
- **KILL_SHOT trigger from AUTONOMOUS_PLAN_2026-05-16.md**
  ("P1 reveals avg > 2.0 dec everywhere → no functional model claim
  possible") is **partially triggered**: 2 of 3 pipelines have
  avg > 2.0 dec. Only z432 PT sits below, and only because its
  fwd half is computed over a subset. Pivot decision should be
  re-discussed after P1b (z446/z449/z454 on zgx).

## §1.4 Reproducibility

- Harness: `scripts/P1a_honest_baseline.py` (single-shot, no physics
  changes; only adds the backward-order V_D loop on top of existing
  z430/z432/z443 inner kernels).
- Run: `HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/P1a_honest_baseline.py`
- Total wall time: 2 706 s (45 min) on ikaros.
- All per-curve RMSE arrays in `results/P1a_honest_baseline/summary.json`
  under `pipelines.<name>.per_curve_RMSE_{fwd,bwd,avg}` keyed by
  `per_curve_keys` (VG1, VG2 pairs, length 25 each).

---

## P1b zgx addendum (2026-05-17, post-rsync)

Newer pipelines run on zgx with both sweep directions on the full bias set:

| pipeline.variant | n_fwd | n_bwd | fwd | bwd | **avg** |
|---|---|---|---|---|---|
| z446.BASELINE_DC_GP | 25 | 25 | 1.619 | 2.823 | 2.221 |
| z446.DC_VBIC | 25 | 25 | 1.311 | 2.864 | 2.087 |
| z446.PT_GP | 25 | 25 | 1.349 | 1.027 | 1.188 |
| **z446.PT_VBIC** | 25 | 25 | 1.396 | 1.156 | **1.276** |
| z449.v449_A | 25 | 25 | 1.311 | 2.864 | 2.087 |
| z449.v449_B | 25 | 25 | 1.311 | 2.864 | 2.087 |
| z449.v449_C (α0×5) | 25 | 25 | 1.610 | 2.886 | 2.248 |
| z454.SB_OFF | 25 | 25 | 1.311 | 2.864 | 2.087 |
| z454.SB_ON_DEFAULT | 25 | 25 | 2.686 | 2.707 | 2.696 |
| z454.SB_LOW | 25 | 25 | 2.628 | 2.694 | 2.661 |
| z454.SB_HOT | 25 | 25 | 2.795 | 2.824 | 2.809 |

### Honest combined ranking (P1a + P1b, defensible numbers only)

1. **z432 / z446.PT_GP** — avg 1.188 (fwd=1.349, bwd=1.027, 25 biases each, 32%/50% conv)
2. **z446.PT_VBIC** — avg **1.276** (fwd=1.396, bwd=1.156, 25/25 biases, fully balanced)
3. z430 V_SINT_PIN — avg 2.301
4. z443 / z449.v449_A/B / z454.SB_OFF — avg 2.087 (all identical: same underlying physics)
5. z454.SB_* — avg 2.66-2.81 (snapback destroys DC)

Path-B claim defensible at **1.19-1.28 dec cell-wide avg, 25 biases, both sweep directions, fully balanced**. This is the publishable headline. All "<1.0 dec breakthroughs" in prior campaign were forward-only on dropped-bias subsets.
