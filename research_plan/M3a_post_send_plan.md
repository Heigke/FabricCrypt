# M3a Post-Send Plan — Validation Tightening + Large-Scale Sims

**Status:** Brief v4.1 was sent 2026-05-03. Now we validate the model
to its lower bound and run large-scale simulations to find what
NS-RAM topologies can do.

## Today's deltas (2026-05-03 evening)

1. **VG1=0.4 V catastrophe — root cause found.** Probe v2 isolated
   the failure to a self-sustaining parasitic NPN root: at no-Iii
   biases, `bjt.Bf=5e4` produces a Newton-stable Vb≈0.43 V where
   Ic_Q1 dominates Id without any physical charge-pumping mechanism.
   See `research_plan/binning_audit/probe_v2_finding.md`.

2. **Bf grid coarse — Bf=2×10⁴ wins.** Sweep across all 25 measured
   biases shows overall median log-RMSE drops 1.00 → 0.80 dec
   (-20 %) with no regime breakage. Brief headline numbers can be
   sharpened.

3. **The "8 NaN biases" are not solver-fail.** They are biases for
   which Sebas's CSV has K1=NaN (he didn't extract overrides for the
   negative-VG2 snapback regime). Documentation issue, not bug.

## Plan (sequential, autonomous, cron-driven)

### M3a-A: Apply Bf=2×10⁴ to z91g and rebuild headline numbers
- Patch `scripts/z91g_two_model_validation.py` to use Bf=2e4 default.
- Rerun stage5 (cross-file .param plumbing) variant.
- Expected: median 0.80, mean 1.40, max 2.89, p90 2.58.
- Update brief addendum (not the sent v4.1; keep as M3a deliverable).

### M3a-B: Per-bias BETA0 ingestion (Iii path, not bipolar gain)
- `make_overrides()` already routes sebas BETA0 → P_M1["beta0"] (the
  BSIM4 impact-ion β₀). Already in use.
- Document in M3a writeup that BETA0 from CSV ≠ bipolar Bf; the
  bipolar gain is structural.

### M3a-C: Rerun ngspice cross-validation with Bf=2e4
- A.5 work was at Bf=5e4. With the new Bf, regenerate the 9-bias
  comparison and confirm divergence pattern doesn't shift.
- Output: `research_plan/artifacts/A5_ngspice_bf2e4.md`.

### M3a-D: Large-scale topology simulation (RUNNING NOW)
- z139_largescale_topology.py launched at 2026-05-03 ~19:00.
- 6 topologies × 3 N values {100,300,800} × 3 seeds × 4 tasks.
- New topologies introduced: HUB_SPOKE (mimics NS-RAM well-shared
  body), LAYERED (deep-RC analogue).
- Wall budget ≈ 3 hours; intermediate JSON dumps after every sim.
- Output: `results/z139_largescale_topology/summary.json`.

### M3a-E: Subagent dispatch — independent validation pass
- Spawn an Explore agent on the codebase to surface any silent
  approximation, hardcoded constant, or oracle-flagged TODO that we
  haven't owned yet. Brief specifically: "find any place a parameter
  is clamped/floored/defaulted/hardcoded that could mask a model bug."

### M3a-F: Transient validation harness scaffold
- Sebas's transient data still pending. Build the harness now so
  when the data arrives we can immediately fit. Skeleton:
  `scripts/z140_transient_harness.py` with a synthetic ground-truth
  exercise.

### M3a-G: After z139 finishes — find the NEW topology winner
- Parse z139 output. For each (N, task) compare HUB_SPOKE and
  LAYERED to the four classical topologies.
- If either beats classical at scale: write a publishable note + a
  short follow-up campaign at the winning N.

## Cron cadence

- **bd31f5ed** — `17,47 * * * *` — autonomous wake-up, original
  research-plan loop.
- **ca462097** — hourly at :23 — z139 progress check + advance to
  M3a-A/B/C as z139 progresses.

## Subagent contracts (when to use)

- **Explore** for any "where in the code is X handled" question that
  spans >3 files.
- **general-purpose** for any task longer than ~10 tool turns where a
  fresh-context worker is more efficient than my main loop.
- **Plan** for any phase boundary (B → C, big rewrite).

## Done criteria (M3a closure)

- [ ] Bf=2e4 applied + headline rebuilt (M3a-A)
- [ ] Independent validation surface from subagent (M3a-E)
- [ ] z139 large-scale results in hand (M3a-D)
- [ ] One novel topology evaluated for publishability (M3a-G)
- [ ] Transient harness ready to consume Sebas's data when it arrives (M3a-F)
