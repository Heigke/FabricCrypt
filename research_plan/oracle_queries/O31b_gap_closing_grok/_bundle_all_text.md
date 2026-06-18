# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: plan.md (6955 chars) ===
```
# Gap-Closing Plan — Make Architecture Findings Mario-Actionable

**Premise**: z210/z211 found a candidate architectural win
(`ER_SPARSE + r=2/s=0.3 inhibition + ff + N=256` → best 1.000,
final 0.958). This finding has 5 named gaps that prevent it being
sent to Mario as a chip-mod recommendation. This plan closes all 5
in parallel via 5 tracks (V/R/C/T/S) plus a thermal-track (P) that
unblocks larger sweeps.

Each track has clear "done" criteria and 3-tier cadence (hourly /
daily / weekly).

---

## Track V — Validation power (statistical CI)

**Gap**: 24 test samples × 3 seeds on toy task. Binomial CI ~ ±0.16.

**Done criteria**:
- ≥10 seeds per architecture variant
- ≥240 test samples per task
- Bootstrap 95% CI on accuracy/RMSE for every claim
- Paired-sample test (same seed, same data) for arch vs. baseline

**Concrete artifacts**:
- `scripts/z212_high_power_validation.py` — 10 seeds × 240 test
  samples on the candidate winner. Wall budget ~30 min on 3 workers.
- Bootstrap-CI plot in `figures/z212/`

**Daily cron task**: re-run candidate at 10 seeds; compare CI to
yesterday's; flag drift.

---

## Track R — Realism (silicon-grounded)

**Gap**: NSRAMSurrogate is a 20×20×25 lookup of pyport's I_d. Two
abstraction layers from silicon. The inhibition gain might be a
surrogate artifact.

**Done criteria**:
- Reservoir built using pyport-direct cell evaluations (no surrogate)
  on at least one architecture variant
- ngspice cross-check on 3 network nodes' op-points to verify the
  surrogate doesn't lie about the ridge of operation
- Surrogate-vs-pyport-direct accuracy delta < 5 pp on the same task

**Concrete artifacts**:
- `scripts/z213_pyport_direct_reservoir.py` — bypass surrogate; use
  `solve_2t_steady_state` per cell per timestep on N=32 reservoir
  (small, ~5 min wall, ~2-4 min on GPU once port is ready)
- Comparison figure `figures/z213/surrogate_vs_pyport_direct.pdf`

**Risk**: this is heavy. May need GPU.

---

## Track C — Chip-cost model

**Gap**: "Lateral inhibition" is software W-matrix sign pattern.
Realisation requires per-cell sign-inverter sub-fabric or a separate
inhibitory crossbar. Without area/power model Mario cannot decide.

**Done criteria**:
- Explicit area cost (in µm²) for adding inhibitory connections
  with radius r at 130 nm
- Power cost (mW added per inferenced signal)
- Comparative table: r=0 (baseline) / r=1 / r=2 / r=4 — area delta,
  energy delta, accuracy delta
- A 1-page recommendation Mario can read in 5 minutes

**Concrete artifacts**:
- `research_plan/chip_mod_inhibition_cost_v1.md` with literature-
  cited area numbers (per-cell sign-inverter ≈ 4 transistors ~5 µm²
  in 130 nm; crossbar grows as O(N·r))
- Decision matrix figure `figures/chip_cost_inhibition/decision.pdf`

**Effort**: 1 day desk-research + drafting.

---

## Track T — Task suite (generalisation)

**Gap**: Only Mackey-Glass-vs-sin tested. Brief targets keyword
spotting / edge-AI.

**Done criteria**: ALL three tasks tested at the candidate winner:
1. NARMA-10 regression (NRMSE)
2. Sequential MNIST (row-by-row, 10-class)
3. Speech Commands v2 subset (12-class keyword spotting)

Each task with 10 seeds + 240+ test samples. Each compared against
the same architecture-baseline triplet:
   {ER_SPARSE-only, ER_SPARSE+inhibition, MESH_4N-only}.

**Concrete artifacts**:
- `scripts/z214_narma10.py` — NARMA-10 regression harness
- `scripts/z215_seq_mnist.py` — Sequential MNIST harness
- `scripts/z216_speech_commands.py` — KWS harness (subset of 12-class)
- Combined figure `figures/task_suite/winners_per_task.pdf`

**Effort**: 1-2 weeks of careful implementation; some tasks ride on
existing harnesses, KWS is genuinely new.

---

## Track S — Statistical hardening

**Gap**: 3 seeds, no CI, no cross-architecture statistical test.

**Done criteria**:
- All claims backed by bootstrap 95% CI
- Paired t-test for every "X beats Y" claim
- Reproducibility: code committed, seeds explicit, ngspice version
  pinned

**Concrete artifacts**:
- `scripts/util_bootstrap_ci.py` — shared utility used by V/R/T tracks
- Stats appendix to each track's report

**Effort**: 0.5 day to write utilities; integrated into each track.

---

## Track P — Thermal/compute (UNBLOCKS V, R, T)

**Gap**: 6+ workers on NSRAMSurrogate hits 100°C. ProcessPoolExecutor
+ surrogate work bypasses thread caps. Currently can't run sweeps.

**Done criteria**:
- A surrogate-eval primitive that respects thread caps OR
- A GPU port of the surrogate (gfx1151 ROCm) that runs at <60°C
- A safe-to-run "background sweep" runner with built-in thermal pause
  + auto-resume

**Concrete artifacts**:
- `scripts/util_safe_sweep.py` — wraps ProcessPoolExecutor with
  thermal monitor + pause/resume + per-config wall-time cap
- `scripts/nsram_surrogate_gpu.py` — torch port of the surrogate
  using GPU bilinear interpolation
- Verification: 432-config z211 sweep completes without exceeding
  75°C

**Effort**: 1 day for safe-sweep; 1 day for GPU port (riskier).

---

## Cross-track schedule (week 1-3)

```
Week 1 (immediate):
  Day 1-2: Track P (safe-sweep wrapper)  → unblocks everything
  Day 1:   Track C (chip-cost desk research, parallel)
  Day 2-3: Track V (10-seed validation of z210/z211 candidate)
  Day 3-4: Track S (utilities, plumbing into V results)

Week 2:
  Day 5-7: Track T.1 (NARMA-10) on candidate
  Day 8-9: Track T.2 (Sequential MNIST) on candidate
  Day 10:  Track R.1 (pyport-direct N=32 reservoir comparison)

Week 3:
  Day 11-13: Track T.3 (Speech Commands KWS subset)
  Day 14:    Track R.2 (ngspice cross-check on network nodes)
  Day 15:    Synthesis: Mario-actionable 1-page chip-mod recommendation
```

## Decision gates

- **End of Week 1**: Does 10-seed validation hold the z210/z211 win?
  - Yes (CI ≥ +5pp) → proceed to Week 2 task suite
  - No → declare z210/z211 a lottery artifact; pivot to I.1 hetero-cell
- **End of Week 2**: Does the win generalise across NARMA + Sequential
  MNIST?
  - Yes (≥2 tasks improve) → proceed to KWS as the Mario-headline task
  - No → architecture-specific; rewrite the chip-mod note as
    "task-conditional"
- **End of Week 3**: Is the chip-mod note + cost model clean enough
  to send Mario?
  - Yes → send + cc Sebas
  - No → another oracle-critique round, identify final blockers

---

## Cron strategy update

Add to existing 4 crons:

5. **GPU off-hours**: 02:30 every day. Run heavy compute (Track T,
   Track R) when ambient is coolest. Auto-pause if APU > 65°C.
6. **Oracle feedback**: every 6 hours during active exploration days.
   Brief packet auto-built from latest results. Captures stalls,
   cherry-picking risks, statistical pitfalls early.
7. **Track-progress audit**: every 4 hours during work-hours. Counts
   done-criteria flags per track; flags any track stalled > 24 h.

(Implementation in next wake-up after oracle review of this plan.)

---

*Drafted 2026-05-07 by Eric. Send to oracles for critique before
locking down cron schedule.*

```
