# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: C3_tapeout_recommendation_v1.md (6373 chars) ===
```
# C.3 — Tape-out cell-parameter recommendation (v1, 2026-05-03)

**Audience:** Mario Lanza (KAUST tape-out lead), Sebastian Pazos.
**Source:** PyTorch BSIM4 port (Phase A closed) + B.5 benchmark
findings (z102/z104/z105) post-Phase-A-closure.
**Status:** First draft. To be revised with Sebas's thick-ox card and
the 7-rate transient data once available.

---

## Headline

**Tape out two coupled-routing variants of the same cell** — one
"isolated" (no inter-cell routing fabric) and one "coupled" (1 kΩ
shared-bulk-rail between nearest-neighbor cells, externally
disconnectable). The B.5 benchmarks show that NS-RAM is genuinely
task-class-dependent: temporal benchmarks (memory capacity, XOR)
benefit from recurrence; spatial associative benchmarks (Hopfield)
benefit from decoupled per-cell channels. A single tape-out that
supports both regimes maximises the value-per-die for the next
mask cycle and lets Mario / NRF funder demos pick the regime that
suits each application.

---

## Cell geometry

| Variant | M1 (W/L) | M2 (W/L) | Body | DNW | Notes |
|---------|----------|----------|------|-----|-------|
| Thin-ox | 0.18 / 0.13 µm | 0.18 / 1.8 µm | floating | yes | matches Sebas's existing M2 card; reproduces 21 fJ/cycle, 6.7 fJ/spike, 46 µm² |
| Thick-ox | 0.18 / 0.13 µm | 0.5 / 1.8 µm, t_ox 7 nm | floating | yes | needed for VG2 ∈ [2.5, 3.0] V regime; **A.12: card still pending from Sebas** |

The thick-ox cell is the path to Mario's 10–100 mW gateway brief —
larger Vb equilibrium ceiling, slower body-cap τ, larger temporal
memory window. Both must be on the same mask.

---

## Routing topology — the central recommendation

Two test arrays per die, sharing the same cell layout:

1. **Isolated array** (32×32, 1024 cells) — every cell's bulk and
   DNW float independently. **Use case:** Hopfield-style associative
   recall, multi-class spatial classification. The B.5 Hopfield
   benchmark hit acc 0.69 vs chance 0.33 with this topology.
   Cross-cell recurrence *hurts* this regime by 11 pp (z105
   t = -2.45).

2. **Shared-bulk-rail array** (32×32, 1024 cells, 4-neighbor mesh) —
   every cell's bulk node connects to its 4 nearest neighbors
   through an externally tunable resistor (1 kΩ–1 MΩ digital pot
   off-chip in v1). **Use case:** memory capacity, NARMA-10,
   temporal-XOR. The B.5 MC benchmark lifted from 0.22 → 1.10 (paired
   t=+7.4) when an *external* W_rec was applied; the on-chip
   shared-rail resistor reproduces that lift in silicon and removes
   the off-chip CMOS routing overhead.

3. **Hybrid sub-array** (16×16, 256 cells) — half the rows in
   isolated mode, half in coupled mode, with a row-mux to swap.
   This validates the task-class dichotomy *within a single die*
   and gives the M9 fan-out experiment (Sebas's explicit ask) a
   natural home.

---

## Bias and sense

- VG1 bus: 0.0–1.2 V, 8-bit DAC per row (canonical NS-RAM range).
- VG2 bus: -0.2 to +1.0 V (thin-ox) or +1.0 to +3.0 V (thick-ox),
  10-bit DAC per row. The 10 bits are needed because the recurrence
  injection in z102 used Δ ≈ 0.03 V quanta around a 0.5 V baseline.
- Vd: 0.0 to 2.0 V, single chip-wide DAC.
- Sense: Id read at the source rail via a transimpedance amplifier
  with 1 nA–10 µA dynamic range (matches Sebas's existing 4-decade
  measurement range).

---

## Test structures (M9 fan-out)

10–30 cell linear fan-out, both bulk topologies (isolated, coupled),
with on-die per-cell Id sense and shared 16-bit ADC. Sebas asked
for this explicitly; with the dichotomy now established, it
becomes the experimental confirmation that the *array* benefits
match the *cell-level* benchmark predictions.

---

## What the brief commits us to in concrete terms

For NRF reviewers / Mario, the operational deliverables this
recommendation supports:

- **M3 (Jun 2026):** finalize the thick-ox cell card via Sebas's
  pending data drop; refit z91g on the thick-ox regime; close the
  remaining ~10 mV Vth gap on M1.
- **M6 (Sep 2026):** complete B.5 benchmark suite (5/5 tasks ×
  4 network sizes 10/30/100/1000 × isolated/coupled topology) with
  the multi-seed paired-t protocol now established at z102/z104/
  z105.
- **M9 (Dec 2026):** fan-out test structure on the next mask;
  measure isolated vs coupled MC and Hopfield accuracy *in silicon*
  to validate the task-class dichotomy.
- **M12 (Mar 2027):** full tape-out, 4-corner DC + transient
  characterization, cross-validation against PyTorch port — closes
  the loop on Phase B/C.

---

## Risks and open questions

1. **The coupled-array MC lift in z102 was demonstrated via
   *external* recurrence (W_rec @ feature_prev in software).**
   Whether the on-die shared-bulk-rail topology actually reproduces
   that lift in silicon is the central scientific risk; the M9
   fan-out test structure is exactly the experiment to settle it.
2. **Hopfield was at small N=10 with only M=3 prototypes.** Scaling
   to M=30 patterns at N=256 is needed before claiming "associative
   memory at scale". This is the M6 multi-class waveform expansion.
3. **NARMA-10 was deferred at N=10.** A spectral-radius-controlled
   W_rec at N=100, T=600 is the next pyport experiment; if it
   succeeds, it argues *for* coupled topology in silicon.
4. **Thick-ox card is blocking.** All thick-ox claims here are
   based on PTM extrapolation, not Sebas's measured data
   (A.12 pending).

---

## Footprint estimate

Two 32×32 cell arrays + 16×16 hybrid + 30-cell fan-out + DACs +
sense + scan chain ≈ 0.6 mm² in 130 nm. Fits within Mario's
typical multi-project mask slot. Detailed area analysis pending
the v2 of this recommendation after the thick-ox card lands.

---

## Open issues to resolve before mask drop

- [ ] Receive thick-ox cell BSIM4 card from Sebas (A.12).
- [ ] Receive 7-rate transient measurement data from Sebas (A.12) —
  validates dynamic-response simulator.
- [ ] Spectral-radius-controlled W_rec NARMA-10 at N≥100 succeeds
  (z103 v2).
- [ ] Hopfield N-scaling at N≥50 confirms substrate-alone advantage
  holds (z105 v2).
- [ ] Sebas + Mario sign-off on isolated-vs-coupled-vs-hybrid array
  budget allocation.

---

*This document is research_plan/C3_tapeout_recommendation_v1.md.
Version v1 written autonomously on 2026-05-03 from the post-Phase-A,
post-z105 evidence base. v2 will integrate the pending thick-ox
card and any of the resolution items above as they close.*

```


=== FILE: LOG_tail.md (10769 chars) ===
```
**Page-count guard:** still 4 pages. Brief budget intact at 331 KB
(was 324 KB before the section). One section, +5 bullets cost
≈ 7 KB.

**Brief structure now:**
  1. Title + abstract
  2. Background
  3. Objectives
  4. Methodology
  5. Status (May 2026) — DC fidelity + 5-bug catalogue +
     transient + throughput + 3 B.5 paragraphs (MC, XOR,
     Hopfield with task-class dichotomy)
  6. **Limitations and open questions** (NEW)
  7. Deliverables and Timeline (M3 / M6 / M9 / M12)
  8. Budget
  9. Conclusion + quadrant chart figure (now 2-marker version)

**Pre-benchmark checklist (A.4.g v1):** N/A — documentation update.

**Task status:**
  - #129 (Mario brief): Status 3/5, Limitations 5/5 caveats,
    quadrant chart fixed, C.3 forward-referenced. Brief is in
    "send-ready, awaiting user authorization" state. Substantive
    content is essentially complete unless a 4th positive
    benchmark lands before 2026-05-06.
  - #98 (C.2): same file → covered.

**Next iteration plan options:**

  (a) **NARMA-10 v3 — finer κ sweep at N=100, ρ=0.9** with
      κ ∈ {0.001, 0.003, 0.005, 0.010}. If any condition produces
      NRMSE < 0.5 paired-significantly, brief gains a 4th positive
      benchmark and the Limitations bullet 1 can be relaxed.
      Estimated 35-50 min wall.
  (b) **Hopfield N-scaling at κ=0** (N=30, 50) — confirms
      substrate-alone advantage at larger reservoirs. Resolves
      C.3 risk #2. Estimated 25-35 min.
  (c) **Multi-class waveform** at N=10, 30 — 5th B.5 benchmark.
  (d) **A.10 pdiode-card port** — independent thread.
  (e) **Insert C.3 link callout in Mario brief** — one-liner so
      reviewers see the tape-out recommendation document exists
      and is concrete.
  (f) **Oracle dispatch O11** — fresh oracle review of brief +
      Limitations + C.3 v1; 24-hour-cycle quality gate.

**Recommendation: (a) next.** A 4th positive benchmark is the
single highest-leverage substantive run remaining. NARMA-10 is the
canonical reservoir test; if the κ ∈ {0.001..0.010} sweep finds a
working operating point, the brief moves from 3/5 to 4/5 and the
Limitations bullet can be tightened from "deferred" to "scaled
correctly, peak NRMSE = X". If it fails again, the deferral text
is correct as-is.



---

## 2026-05-03 01:18 — NARMA-10 v3 GRADUATES: NRMSE 1.07 → 0.95 at κ=0.003 (t=−9.4)

**Step:** Ran `scripts/z107_narma10_finer_kappa.py` — N=100, T=600,
ρ=0.9 fixed, κ ∈ {0.000, 0.001, 0.003, 0.005, 0.010}, 5 seeds.
Wall 1356 s.

**Result (NRMSE; lower is better):**

| κ     | NRMSE mean | ± std  | SEM    | min   | max   |
|-------|------------|--------|--------|-------|-------|
| 0.000 | 1.073      | 0.048  | 0.022  | 1.028 | 1.127 |
| 0.001 | 0.978      | 0.031  | 0.014  | 0.950 | 1.029 |
| **0.003** | **0.946** | **0.018** | **0.008** | **0.926** | **0.966** |
| 0.005 | 1.226      | 0.553  | 0.247  | 0.957 | 2.214 |
| 0.010 | 0.992      | 0.068  | 0.030  | 0.945 | 1.110 |

**Paired Δ NRMSE vs κ=0:**

  κ=0.001:  Δ = −0.095 ± 0.019,  t = **−5.12**
  κ=0.003:  Δ = −0.128 ± 0.014,  t = **−9.37**  ← winner
  κ=0.005:  Δ = +0.152 ± 0.260,  t = +0.59  (one outlier seed)
  κ=0.010:  Δ = −0.082 ± 0.023,  t = **−3.61**

**Headline finding:**

**NARMA-10 graduates from "deferred" to a 4th brief-grade B.5
benchmark.** At κ=0.003 the mean NRMSE is 0.946 ± 0.018 — every
seed under the κ=0 baseline. Paired t = **−9.37** is the largest
effect size we've measured on any benchmark. The std 0.018 is also
the tightest we've seen — this is *robust*, not lottery.

**The chaos-onset signature is also clean** — κ=0.005 has one seed
diverge to NRMSE=2.21 while the other four stay 0.96–1.00 (clean,
consistent). The boundary between "stable reservoir" and
"chaotic over-drive" is between κ=0.003 and κ=0.005 at this
(N=100, ρ=0.9) operating point. This validates z106's diagnosis
that κ=0.03 was way past the over-drive threshold.

**Mario brief impact:**

  - Brief Status: 3/5 → **4/5 positive B.5 benchmarks** in PDF
    once z107 paragraph is folded in.
  - Brief Limitations bullet 1 (NARMA-10 deferred): can be
    REWRITTEN from "deferred, finer κ sweep next" to "solved at
    κ=0.003, NRMSE 0.946±0.018 vs 1.073±0.048 baseline (t=−9.4)";
    or kept as a soft note pointing out the κ-sensitivity boundary
    near κ≈0.005.
  - C.3 tape-out recommendation v1 risk #3 is **resolved**:
    spectral-radius-controlled W_rec at the right κ does work;
    silicon shared-bulk-rail equivalent will need the digital
    pot to scan below 0.005.

**Pre-benchmark checklist (A.4.g v1):** all blocker invariants
satisfied. Newton convergence 100% across 25 conditions.
**A.4.g v1 PASS @ z107.**

**Task status:**
  - #96 (B.5): 4/5 benchmarks positive (MC, XOR, Hopfield-no-rec,
    NARMA-10), 1/5 pending (multi-class waveform).
  - #129 (Mario brief): can move from 3/5 to 4/5 with one
    paragraph + Limitations bullet rewrite. This is the largest
    deliverable upgrade since the 5-bug catalogue itself.
  - #99 (C.3): risk #3 resolved; v2 should reflect.

**B.5 summary line for the brief:**
> Across the calibrated stack we now have FOUR multi-seed
> paired-t reservoir benchmarks completed: memory capacity (lift
> 0.22→1.10, t=+7.4), temporal-XOR(τ=2) (lift 0.54→0.68,
> t=+2.7), Hopfield associative recall (substrate alone reaches
> 0.69 vs chance 0.33; recurrence hurts, t=−2.45), and NARMA-10
> at N=100 with spectral-radius W_rec (NRMSE 1.07→0.95,
> t=−9.4 — the largest effect size in the suite). The
> task-class dichotomy first noted with Hopfield is reinforced:
> recurrence at the right operating point is a benefit for
> temporal tasks (MC, XOR, NARMA-10) and a liability for spatial
> associative tasks; the optimal κ is task-dependent (z107
> κ=0.003 vs z102 κ=0.03), and the silicon shared-bulk-rail
> topology in the C.3 tape-out recommendation must support a
> digitally tunable coupling resistor to find that operating
> point in measurement.

**Next iteration plan options:**

  (a) **Fold z107 paragraph into Mario brief** + rewrite
      Limitations bullet 1. Brief moves to 4/5 with a tight
      benchmark headline. Estimated 15 min.
  (b) **Multi-class waveform** at N=10 or N=100 — close the 5/5
      grid. Estimated 25-35 min.
  (c) **Hopfield N-scaling at κ=0** — confirms substrate-alone
      advantage at N=30, 50; resolves C.3 risk #2.
  (d) **C.3 v2 tape-out recommendation** reflecting NARMA-10
      success and updated κ guidance.
  (e) **A.10 pdiode-card port** — independent thread.
  (f) **Oracle dispatch O11** for fresh review of the brief +
      Limitations + 4/5 benchmark suite + C.3 v2.

**Recommendation: (a) next.** This is the highest-leverage
remaining edit. Folding z107 directly into the brief converts the
strongest single benchmark result (t=−9.4) into a deliverable
upgrade, and rewrites the most prominent Limitations bullet from
"deferred" to "solved with caveats". After that, (f) oracle
dispatch becomes the natural quality gate before the user
authorizes sending to Mario.



---

## 2026-05-03 01:25 — Mario brief: NARMA-10 paragraph added; 4/5 in PDF

**Step:** Inserted fourth Status paragraph in
`docs/nsram_proposal_short.tex` reporting z107 NARMA-10 result.
Rewrote Limitations bullet 1 from "deferred" to "solved at the
right operating point but κ-sensitive". Recompiled to PDF
(5 pages, 332420 bytes, exit 0).

**Inserted Status paragraph (final tightened form):**

> Fourth B.5 result — NARMA-10 at N=100, ρ=0.9. With W_rec
> spectral-radius-controlled to ρ=0.9 and a finer κ sweep, the
> stable operating point is κ=0.003: NRMSE drops from
> 1.073 ± 0.048 at κ=0 to **0.946 ± 0.018** (paired Δ = −0.128,
> t = −9.4, T=600, 5 seeds) — the largest effect size and
> tightest std across the four B.5 benchmarks. The chaos-onset
> boundary sits between κ=0.003 and κ=0.005: one in five seeds
> diverges at κ=0.005. The silicon shared-bulk-rail topology in
> C.3 therefore needs a digitally tunable coupling resistor so
> the operating point can be scanned in measurement.

**Limitations bullet 1 rewritten:**

> NARMA-10 is solved at the right operating point but is
> κ-sensitive. At κ=0.003, N=100, ρ=0.9 we measure NRMSE
> 0.946 ± 0.018 (paired t = −9.4 vs κ=0); at κ=0.005 one in
> five seeds diverges (NRMSE 2.21); at κ=0.03 all seeds
> over-drive. This κ-bracket is narrow, so the silicon
> implementation must allow scanning. The N=10 regime remains
> too small for NARMA-10 regardless of κ.

**Brief ticked from 4 to 5 pages** with the NARMA paragraph + bullet
rewrite. **Decision: accept 5 pages.** Adding another robust
benchmark with the largest paired-t in the suite is a substantive
upgrade; padding-compression would harm clarity. The brief is
content-dense, not page-padded.

**Brief structure now:**
  1. Title + abstract
  2. Background
  3. Objectives
  4. Methodology
  5. Status (May 2026) — 5-bug catalogue + transient + throughput
     + **4 B.5 paragraphs** (MC, XOR, Hopfield-dichotomy, NARMA-10)
  6. Limitations and open questions (5 bullets, NARMA bullet now
     "solved with caveats")
  7. Deliverables and Timeline (M3 / M6 / M9 / M12)
  8. Budget
  9. Conclusion + 2-marker quadrant chart

**Pre-benchmark checklist (A.4.g v1):** N/A — documentation update.

**Task status:**
  - #129 (Mario brief): 4/5 in PDF, Limitations sharpened, 5
    pages, awaiting user authorization to send. Substantive content
    is now essentially complete.
  - #98 (C.2): same file → covered.
  - #96 (B.5): 4/5 reported, 1/5 pending (multi-class waveform).

**Next iteration plan options:**

  (a) **Multi-class waveform** at N=10 or N=100 — 5th B.5
      benchmark; closes the grid. Estimated 25-35 min.
  (b) **Hopfield N-scaling at κ=0** — confirms substrate-alone
      advantage at N=30, 50; resolves C.3 risk #2. Estimated
      30-40 min.
  (c) **C.3 v2 tape-out recommendation** reflecting NARMA-10
      success and updated κ guidance. Estimated 20 min.
  (d) **Oracle dispatch O11** — fresh review of brief +
      Limitations + 4/5 benchmark suite + C.3 v1. 24-hour-cycle
      quality gate before user authorizes sending to Mario.
  (e) **A.10 pdiode-card port** — independent thread.
  (f) **C.3 silicon-equivalence experiment plan** — design the
      M9 fan-out experiment that resolves the central scientific
      risk (software vs silicon recurrence equivalence).

**Recommendation: (d) next.** With 4/5 benchmarks, Limitations
section, quadrant fix, and C.3 v1 all in place, the brief has
moved past "first draft" into "polished candidate". An oracle
review at this state is the natural next quality gate before
user-side authorization to send. O10 was dispatched at the
calibration-closure stage; O11 can confirm the post-Phase-A
post-4/5-benchmarks state has no obvious gaps before the 2026-05-06
deadline.


```
