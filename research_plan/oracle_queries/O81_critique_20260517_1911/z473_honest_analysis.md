# z473 — R_body sweep + V3/V6/V7 triplet test

Date: 2026-05-17. Cell: NX_1p8 calibrated (`snap_Is=4.5192e-12`).
Wall: 410 s (sweep+V3+V6+V7+Mario at R=1e8) + 102 s (retry at 1e7 and 1e6).

## TL;DR

- **R_body=1e7 Ω chosen.** Calibration intact (Id_pk drift **0.007 dec**;
  KILL_SHOT 0.3 dec **not triggered**).
- **V6 (self-reset) FLIPS to PASS** at R_body ≤ 1e7. V_B post-release
  drops from 0.531 V (R=1e8) to **0.274 V** (R=1e7) to 0.034 V (R=1e6).
- **V7 (relaxation oscillation) STILL FAILS** at every tested R_body
  (1e9, 1e8, 1e7, 1e6, 1e5). 0 cycles, no free-running osc.
- **V3 (DC knee) UNCHANGED** — knee=NaN both fwd and bwd. R_body is a
  transient-only knob; V3 is pure DC.
- **Mario shape: 1/5 → 3/5** at R_body=1e7 (gains `t_fall_match` and
  `self_reset_match`). Below 1e7 the leak is too aggressive, t_fall
  falls outside the ±30% band, drops to 2/5.

## Pre-registered gates

| Gate | Criterion | Result |
|------|-----------|--------|
| INFRA | sweep done + V3/V6/V7 rerun on chosen R_body | **PASS** |
| DISCOVERY | V6 PASS AND Id_pk drift < 0.15 dec | **PASS** (R=1e7) |
| AMBITIOUS | V3+V6+V7 ALL PASS AND Mario ≥ 3/5 | **FAIL** (V3, V7 still F) |
| KILL_SHOT | Id_pk drift > 0.3 dec on chosen R_body | **FALSE** (0.007 dec) |

## Step 1–3 — R_body sweep at primary bias (VG1=0.6, VG2=0, Vd=2V, 200 ns pulse)

| R_body | Id_pk (mA) | drift (dec) | τ_decay (ns) | V_B @200 ns post | reset<0.4 V |
|--------|-----------|-------------|--------------|------------------|-------------|
| ∞ (default) | 4.31 | 0.008 | 89.7  | 0.478 | no |
| 1e9 Ω      | 4.31 | 0.008 | 102.6 | 0.468 | no |
| 1e8 Ω      | 4.31 | 0.008 | 166.6 | 0.338 | yes |
| 1e7 Ω      | 4.30 | 0.007 | 52.2  | 0.001 | yes |
| 1e6 Ω      | 4.17 | 0.007 | 6.95  | 0.000 | yes |

Notes:
- Id_pk is essentially flat across all R_body — the leak current is
  ~Vb/R ≈ 60 nA at R=1e7, negligible vs the 4.3 mA snapback current.
- At R=1e6 Id_pk drops slightly (4.17 mA vs 4.23 target = 0.007 dec
  drift). Still well inside the 0.15-dec gate.
- τ_decay non-monotonic with R because of leak-driven re-firing of the
  parasitic NPN during decay. R=1e8 has the longest tail; R≤1e7 the
  leak wins outright.
- **Picker chose R=1e7 (post-hoc):** first R that gives BOTH
  in-band Id_pk AND reset<0.4 V AND passes V6. The original script
  picked R=1e8 (first to satisfy reset+in-band), but V6 failed there;
  the retry script (z473b_retry_lower_rbody.py) walks R=1e7 and 1e6.

## Step 4 — V3 / V6 / V7 on chosen R_body=1e7

| Test | Metric | Gate | Verdict (z473) | (z472 baseline) |
|------|--------|------|----------------|-----------------|
| V3 | V_knee fwd=NaN, bwd=NaN | \|V_knee−1.5\|≤0.3 V | **FAIL** | FAIL |
| V6 | V_B_post_mean=0.274 V, t_reset=40.7 ns | t_reset<100 µs AND V_B_post<0.3 V | **PASS** | FAIL |
| V7 | n_cycles=0, period=NaN | ≥3 cycles AND period∈[100,1000] ns | **FAIL** | FAIL |

Triplet result: **1/3 PASS** (V6 only).

## Step 6 — Mario shape v2 (R_body=1e7)

| metric | our cell (z473) | target | within ±30%? | (z472) |
|--------|----------------|--------|--------------|--------|
| t_rise | 2.9 ns | 26 ns | no (too fast) | no |
| t_fall | 71 ns approx | 76 ns | **yes** | no |
| V_B swing | 0.62 V | 0.5–0.7 V | **yes** | yes |
| self-reset between pulses | yes | yes | **yes** | no |
| free-running osc period | NaN | 430 ns | no | no |

**3/5 metrics match** — up from z472's 1/5. Two new flips:
`t_fall_match` (body leak speeds the post-pulse decay to ~70 ns,
within band) and `self_reset_match` (V_B in the inter-pulse gap now
drains below 0.3 V instead of latching at 0.44 V).

## Why V7 still fails — physics interpretation

R_body provides reset path AFTER V_d drops. During constant V_d=2 V
hold, the parasitic NPN's positive feedback (Id → Iii → I_avl → Vb ↑ →
NPN base drive ↑) sustains itself faster than the body cap can drain
through R=1e7 (τ=C·R=10 fF·10 MΩ=100 ns). Reducing R further (1e6,
τ=10 ns) also fails V7 — at that point the leak is fast enough to
prevent latch formation in the first place, but still does not
*break* the latch periodically. **A linear resistive leak is the
wrong mechanism for relaxation oscillation; what's needed is either
(a) a nonlinear (e.g. Zener-like) leak that switches on above some
V_b threshold, or (b) a time-delayed NPN base resistance.**

The z472 hypothesis that V3, V6, V7 share a single root cause
(body-leak path) is **partially falsified**: V6 and Mario self-reset
are body-leak-bound, V7 is not, and V3 is pure DC (insensitive to
R_body by construction).

## Recommendation: z474

Two paths:

1. **Cheap (lock the win):** accept R_body=1e7 as the published cell
   default. Re-run the full z461 9-test once with it (expect 7/9, up
   from 6/9; +V6, V3/V7 still F). Lock the Mario 3/5 in the paper.
2. **Ambitious (chase V7):** add a *nonlinear* body-leak (threshold-
   gated, via existing `R_body_thresh` field in TransientCfgV2 — set
   thresh≈0.5 V) AND a small NPN base-resistance RBE in series so the
   feedback loop has its own time constant. This is the physically
   correct fix for relaxation oscillation; requires a 2D sweep.

I recommend **option 1 for z474** (no new ambitious moves), with V3
deferred until snapback fwd/bwd hysteresis is re-examined separately.

## Files written

- `rbody_sweep.json` — 5-point sweep data
- `v3v6v7_post_sweep.json` — triplet at R=1e8 (original picker)
- `retry_lower.json` — V6/V7/Mario at R=1e7 and R=1e6 (the real
  results used in this analysis)
- `mario_shape_v2.json` — Mario at R=1e8 (single-run)
- `transient_overlay_with_reset.png` — three-panel plot
- `patch.diff` — recommended z461 config change (R_body=1e7)
- `stdout.log`, `retry_stdout.log`, `run.log` — raw run logs
