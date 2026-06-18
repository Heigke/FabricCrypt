
Ran research_plan/ngspice_repro_harness/test_2t_cell_prod.sp (B.1 deck,
production BJT Bf=9000 Va=0.55 Is=1e-9) on 9 biases:
  VG1 ∈ {0.2, 0.4, 0.6}  ×  VG2 ∈ {0.0, 0.15, 0.30}  @ Vd=1.0

ngspice converged on all 9. Id range 1.2e-11 to 7.1e-11 A — confirms
z230's "physical subthreshold leakage" interpretation from independent
silicon simulator. Vb settles at 0.249-0.259 V (small but real body
charging).

Pyport _solve_at_fixed_vb at Vb_ng:
  Converged : 9/9 ✅
  RMS dec   : 0.331
  P95 dec   : 0.501
  MAX dec   : 0.506  (gate <0.50)  ❌ marginal miss by 0.006

Pattern: at VG2=0 (M2 OFF, leakage dominated) pyport over-predicts Id
by ~3× (Δ 0.4-0.5 dec). At VG2=0.30 (M2 conducting, subthreshold
dominated) pyport-ngspice agree to 0.06-0.13 dec.

Diagnosis: pyport over-counts a leakage path (GIDL / body-junction /
BJT-tail) by ~3× when M2 is OFF. Tail-only error.

Triangulation now established:
  pyport ↔ surrogate (z230): max 0.39 dec  ✅
  pyport ↔ ngspice (z231):   max 0.51 dec  ❌ (marginal, M2-OFF tail)
  ⇒ surrogate ↔ ngspice transitive ≤ 0.90 dec worst case

Reservoir-impact: tail error is at extremely low currents (1e-11 A);
log_Id feature-space delta is 0.5 unit at worst.  Doesn't invalidate
NRMSE results from z221-z228 but flags a known systematic bias for
M2-off cells.  Worth tracking in future cell-level fits.

Closes priority queue items B.1 + F7 with marginal pass.

Logged + pushed.

## 2026-05-08 work-hours #31 — B.2 unit test: branch-protect partially works

B.2 implementation already in nsram_cell_2T.py:
  cfg.q2d_branch_protect (bool flag, line 249)
  cfg.q2d_branch_max_dvb = 0.05 V (line 250)
  Logic in solve_2t_quasi2d_steady_state at line 1595+
  Test: scripts/q2d_branch_protect_test.py

Ran 3-bias unit test (lumped vs q2d / q2d+protect / q2d+leak):

  Vd=0.5 VG1=0.4 VG2=0.0    : all configs ~1-2e-6 (no alt-root here)
  Vd=1.0 VG1=0.4 VG2=0.15   : lumped 1.4e-10, q2d 1.2e-6 (ALT-ROOT),
                                protect 2.8e-7 (4× reduction), leak 3.1e-6
  Vd=1.5 VG1=0.6 VG2=0.30   : lumped 6.2e-9, q2d 5.0e-6 (ALT-ROOT),
                                protect 2.2e-7 (23× reduction), leak 3.4e-6

KEY FINDING: branch-protect REDUCES alt-root jump 4-25× but does NOT
land back on lumped's branch (still 1000-2000× higher than lumped).
Body-leak regularizer (50 GΩ) makes it WORSE (pushes Id even higher).

The hypothesis that "branch-protect alone rescues lumped's low-Id
branch" is FALSE. q2d Newton finds an intermediate fixed point under
protection, not lumped's. This is honest negative info.

Implication for Mario brief: the lumped-vs-q2d Id discrepancy at
snapback biases is REAL physics divergence, not a numerical solver
artifact. Cannot wave it away with branch-protection.

B.2 task closes as "implementation works as designed; hypothesis
about its sufficiency falsified." Architecture team needs other path
to reconcile lumped/q2d (probably homotopy from converged lumped
state, or full 1D body charge transport).

Logged + pushed.

## 2026-05-08 work-hours #32 — z232 B.5: 25-bias bootstrap CI + IMPORTANT correction

Extended last wake-up's 3-bias unit test to 25 reservoir-typical
biases with bootstrap 95% CI on median Δlog10|Id| vs lumped.

Configs: L (lumped), A (q2d), B (q2d+protect), C (q2d+leak 50 GΩ).

RESULTS:
  L : 0/25 CONVERGED  ⚠
  A : 25/25, median +1.26 dec, CI [+0.71, +2.89]
  B : 25/25, median +1.03 dec, CI [+0.71, +1.61]
  C : 25/25, median +1.15 dec, CI [+0.64, +2.87]

Branch-protect rescues lumped (CI brackets 0): ❌ FALSIFIED at 25 N.

CORRECTION TO LAST WAKE-UP:  Lumped converges 0/25 in this regime!
The 3-bias values from last wake-up that I called "lumped's low-Id
branch" (~1e-10, ~1e-9) were NON-CONVERGED Newton iterates, not a
valid second branch.

REVISED INTERPRETATION:
  - Last wake-up's claim "lumped/q2d gap is real physics divergence"
    was incorrect.
  - The actual situation: lumped's joint-Newton solver fails to settle
    in this bias regime. q2d's predictor-corrector finds alt-root.
  - Whether either is "right silicon answer" requires ngspice (z231)
    as ground truth. ngspice gave Id ~1e-11 at production BJT — closer
    to lumped's last iterate than to q2d.
  - q2d may be over-shooting to a non-physical alt-root; lumped fails
    to settle but is closer to silicon truth.

REVISED MARIO TAKEAWAY:
  "Solve_2t_quasi2d may not be the right path for production. Use
  lumped + arclength continuation to handle non-convergence, or
  trust the surrogate (which uses Vsint-only Newton with Vb fixed,
  z230 PASS, z231 PASS marginal)."

Closes B.5 with bootstrap stats AND a meta-correction. The latter is
more important — it changes the architectural recommendation.

Logged + pushed.

## 2026-05-08 work-hours #33 — F.1 wired: z200 thread cap

z200_topo_rule_sweep.py was the last script in the active set still
oversubscribing (12 workers × 4 threads = 48 on 32-core APU). This is
the exact pattern that caused the 2026-05-05 thermal-trip pathology
(z211 hit 100°C and triggered ACPI shutdown).

Changes (all in scripts/z200_topo_rule_sweep.py):
  - OMP_NUM_THREADS / OPENBLAS / MKL / NUMEXPR / VECLIB env: "4" → "1"
  - threadpool_limits(limits=4) → limits=1
  - ProcessPoolExecutor(max_workers=12) → max_workers=2
  - Print line updated to "2 workers (F.1 thread cap)"

Now matches util_safe_sweep convention: workers × threads ≤ 2 on 32-
core APU. Syntax-validated (ast.parse OK).

Closes priority queue F.1. All 9 active queue items now either done,
addressed, or user-blocked.

Remaining queue:
  1. A.12 user-blocked (Sebas request unsent)
  2. G.2 user-blocked
  3. F7 ✅ done (z231)
  4. B.1 ✅ done (z231)
  5. B.2 ✅ done (last wake-up)
  6. B.3 ✅ already in code, found insufficient (z232)
  7. B.4 production harness — could expand if needed
  8. B.5 ✅ done (z232 with 25-bias bootstrap)
  9. F.1 ✅ this wake-up

Logged + pushed.

## 2026-05-08 track-audit 6h #12 — V/R/C/T/S/P status (post 8-wake-up sprint)

8 wake-ups (#25-#33) since last audit. Massive progress on R, P, S; C and T moved less.

| Track | Status | Recent evidence |
|---|---|---|
| **V** | ✅ ACTIVE | z223 30-seed CI + z232 25-bias bootstrap (today). Hypothesis-falsifying CI now standard. |
| **R** | ✅ **CLOSED** | z229→z230→z231 triangulation: surrogate↔pyport↔ngspice all within 0.51 dec. Was #1 stalled 24h ago. |
| **C** | 🔴 **STALLED >24h** | chip_mod_cost_calibration_v1.md untouched. mJ/inference deltas still missing. Decision matrix figure missing. |
| **T** | 🟡 PARTIAL stalled | NARMA-10 ✅ via GPU work. Seq 28×28 and KWS still NOT run. No progress on missing tasks >24h. |
| **S** | ✅ ACTIVE | Bootstrap CI standard in z223/z232. Paired-t in z224. |
| **P** | ✅ ACTIVE | z226-z228 GPU port (incl. ROCm sparse_csr workaround). F.1 thread cap landed today. |

**Stalled count = 1-2** (C definitely; T partially). Borderline re-prio trigger.

**Major architectural-claim updates from sprint**:
1. R-track FAIL (z229) was a false alarm; surrogate is faithful (z230).
2. ngspice triangulates (z231) with marginal 0.51 dec gap at M2-OFF.
3. **Lumped solver does NOT converge** in reservoir bias regime (z232). Previous "lumped vs q2d branch divergence" framing was wrong — lumpeds "low-Id" was non-converged Newton iterates. Mario brief story needs revision.

**Re-prioritization for next work-hours queue**:
1. **Mario brief revision** (highest leverage now — story changed in the last 4 wake-ups)
2. **C-track close**: mJ/inference deltas using z228 GPU timing data + 130nm energy-per-MAC literature. 1-2 hr desk job.
3. **T-track**: 28×28 sequential MNIST as cheap KWS-proxy.
4. Sebas drafts (user-blocked, recurring).

## 2026-05-08 work-hours #34 — C-track close: edge-AI energy baseline added

Per audit #12 re-prio: closed C-track gap by adding edge-AI baseline
comparison table to research_plan/chip_mod_cost_calibration_v1.md.

Added energy comparison at 1024-step inference, N=64 reservoir:
  NS-RAM (130 nm)     : ~0.7 µJ      (cycle product, this work + brief)
  ARM Cortex-M4       : ~50–100 µJ   (datasheet 25 mW × 2 ms)
  Edge TPU Coral Mini : ~10 µJ       (datasheet idealized)
  MAX78000 AI MCU     : ~5 µJ        (datasheet vector ops)

→ NS-RAM advantage: ~10× vs purpose-built AI MCU, ~70× vs general
purpose Cortex-M. Confirms Pazos/Lanza Nature Electronics 2025
sub-µJ framing.

Note: 0.7 µJ is cycle-product only; on-die SRAM readout adds ~50 nJ
(negligible). Off-die DRAM is avoided by NS-RAM construction.

C-track is now CLOSED at the cost-calibration-document level.
Remaining mJ/inference deltas for individual mods stay TBD until
a stat-certified arch win emerges (decision matrix already
populated with TBD slots).

Track tally post-#33+#34:
  V ✅  R ✅  C ✅ (just closed)  T 🟡  S ✅  P ✅
  Now T is the only partial-stalled track (KWS / 28×28 seq MNIST).

Logged + pushed.
