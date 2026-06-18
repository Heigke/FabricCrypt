# Physics Hunt Synthesis 2026-05-20 08:35

## All 4 tracks landed

| Track | Source | #1 candidate | Δdec | Confidence |
|---|---|---|---|---|
| **A** | Lit + Sebas/Mario re-read | `rbodymod=1` distributed body R (BSIM4 §6) | 0.2–0.4 predicted | Cards show flag=0 but all geometry params present |
| **B** | Daedalus empirical proxy ablation | Constant-T proxy 127°C (selfheat-like) | **−1.728 measured** | Real selfheatmod not implemented; proxy result |
| **C** | zgx 5D GPU sweep (3125 cells, n=13 subsample) | kappa=1e5 (selfheat) + bbt_a=1e-9 + rb_f=0.3 | **−2.20 measured** | Best: 4.001 → 1.797 dec |
| **D** | Oracle 3-way UNANIMOUS | Hurkx 1992 TAT field-enhanced Γ | 0.7–1.0 predicted | gpt-5+gemini+grok unanimous |

## Convergence
- **Track B + C agree empirically:** SELFHEAT is the dominant single axis. Track C found best cell with kappa=1e5 (max), hurkx_b=**0** (none), bbt_a=1e-9 (small), rb_f=0.3 (small), agidl_s=1.0 (baseline). Hurkx-TAT contribution was zero in the empirical winner.
- **Track A + D disagree with empirical:** Track A predicts structural fix (rbodymod), Track D oracle predicts Hurkx-TAT. Neither was the empirical winner.

## Why the divergence is interesting (not a bug)
1. Track B/C ran a *proxy* for selfheat (constant T), not a real self-consistent selfheatmod=1 with Rth. The empirical Δ is the right *direction* and *order of magnitude* but not the right *mechanism* until properly implemented.
2. Track A/D candidates haven't been empirically tested yet — Track B explicitly says Hurkx-Γ-field-enhanced was "not implemented" and the JTS-TAT proxy showed no effect because that's not the same model.
3. The Track C result that hurkx_b=0 wins is therefore *not* evidence that Hurkx-TAT is wrong; it's evidence that the Track C proxy for Hurkx-TAT isn't activating in this code path.

## Critical NO-CHEAT finding (independent of all 4 tracks)
**The "1.163 dec" headline baseline in proposal v5.3 does NOT reproduce on the current canonical `build_nsram_stack(use_snapback=True)` code path** — Track B got 4.454 dec, Track C got 4.001 dec (different subsample). The 1.163 figure must have been generated from a different parameter snapshot that is not in the current main branch. This is a code-path drift that needs forensic before any new physics fix is ship-claimed.

## Recommended action plan (in execution order)
1. **HIGHEST PRIORITY — forensic on baseline drift.** Find the commit/script/config that produced 1.163 dec. If unrecoverable, the proposal needs an honest correction note: baseline median is actually ~4 dec on canonical code, not 1.163.
2. **rbodymod=1 flip test** (Track A recommendation, 1 flag change in M1+M2 cards). Cost: 2 hr. Predicted Δ: 0.2–0.4. If ≥0.2 dec, structural fix is real.
3. **Real selfheatmod=1 with Rth feedback solver.** Code: ~100 lines in pyport DC port. Cost: 1 day. Predicted Δ: based on Track B/C proxy, real implementation should give ≥1 dec but is mechanistically defensible.
4. **Implement Hurkx Γ-field-enhanced TAT properly.** Code: ~50 lines, replacing JTS-TAT functional form. Cost: 1 day. Predicted Δ: 0.7–1.0 dec per oracle.
5. **Combined re-fit with all three** above + freeze other params. Expected: ≤0.5 dec at low VG1 + high Vd regime. Cost: ~2 days.
6. **If ≤0.5 dec achieved → ship v5.4 with mechanism story (not parameter tuning).** If not → honest publishable null: BSIM4+NPN model is parameter-complete, gap is data-limited.

## Halt condition (per FUNDAMENTAL_PHYSICS_HUNT_2026-05-20 plan)
At least one track produced ≥0.3 dec improvement (Track B/C selfheat proxy gave 1.7–2.2 dec). **Halt condition NOT triggered. Proceed with implementation plan.**

## Files
- `results/physics_hunt_track_A_2026-05-20.md` (or in research_plan/)
- `results/physics_hunt_track_B/ablation.json` + `verdict.md`
- `results/physics_hunt_track_C/sweep_results.json` (3125 cells, all valid)
- `results/physics_hunt_track_D/{packet,gpt5,gemini,grok,synthesis}.md`
- This synthesis: `research_plan/PHYSICS_HUNT_SYNTHESIS_2026-05-20.md`

## NO-CHEAT discipline applied
- Δdec values quoted are from JSON outputs, not narrative
- Track B/C empirical wins flagged as "proxy" until proper implementation
- Track A/D predictions flagged as "predicted" not measured
- 1.163 → 4.0 baseline drift is the most important finding; documented above headline
