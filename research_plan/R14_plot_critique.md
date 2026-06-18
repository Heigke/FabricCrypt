# R-14: Snapback Plot Critique (Oracle O61)

**Date**: 2026-05-13
**Oracle**: OpenAI gpt-5 (vision)
**Packet**: research_plan/oracle_queries/O61_plot_critique/

## User complaint
"They go down and then some snap, they don't look good."

## Oracle verdict (synthesis)

### Per-plot shape match
- **z331 (forced-Vsint, 3-panel + per-Vg2)**: NO qualitative match. Curves fall with Vd then jump in a staircase; no real ramp→peak→snapback. In log view the "snap" is just a small step. Trigger pulled slightly closer to silicon onset than z328.
- **z328_V6 (free Vsint, iv_vg1_0.2/0.4/0.6)**: Non-physical early behavior; current decreases with Vd before knee. Never approaches measured µA ramp.

### Quantitative gaps
- **Amplitude**: model predicts 1e-11…1e-12 A pre-snap; silicon shows 1e-9…1e-7 A → **2–4 decades low**.
- **Trigger Vd**: model ≈2.0–2.7 V; silicon ≈1.4–2.0 V → too late by ~0.5–1 V.
- **Pre-knee slope**: model Id DECREASES with Vd (wrong sign); silicon is flat-to-increasing.
- **Quantization**: visible Vsint/Vb discretization steps in model.

### z331 vs z328_V6
**z331 forced-Vsint is closer** to silicon at onset (shows a knee-like jump). z328_V6 free-Vsint is further off.

### Root cause (visual only)
Body-charging / parasitic NPN feedback is **far too weak**. Impact-ionization current Iii into the body is orders of magnitude small; Vb hardly rises (diagnostics show near-zero or negative Vb), so NPN never properly triggers. This explains late knee, low amplitude, AND the wrong pre-knee slope simultaneously.

## Top single anomaly to fix
**The unphysical pre-knee negative dId/dVd**. Current must rise monotonically with Vd before snapback. Cure should also pull the knee left and raise amplitude toward µA.

## Recommended next experiment
**E_iii_boost**: Sweep impact-ionization coefficient / Iii→Vb coupling gain in pyport. Targets:
1. Pre-knee dId/dVd > 0 (monotonic).
2. Vb reaches >0.6 V before Vd=2.0 V (turn on NPN).
3. Amplitude in 1e-9…1e-7 A pre-snap.
4. Re-test on z331 forced-Vsint topology (the slightly-better baseline).

Also: investigate Vsint/Vb solver discretization — staircase artifacts suggest fixed-point or coarse grid, may need finer Vb resolution or smoothing.
