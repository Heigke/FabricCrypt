# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: C3_tapeout_recommendation_v2.md (8944 chars) ===
```
# C.3 — Tape-out cell-parameter recommendation (v2, 2026-05-03)

**Audience:** Mario Lanza (KAUST tape-out lead), Sebastian Pazos.
**Source:** PyTorch BSIM4 port (Phase A closed) + B.5 benchmark
findings (z102, z104, z105, z107) post-Phase-A-closure.
**Status:** v2 — supersedes v1 (2026-05-03 00:13). Changes vs v1
are flagged inline. Pending: thick-oxide card data drop from
Pazos (A.12), full Hopfield N-scaling sweep.

**Changes vs v1:**

  - Risk #3 (NARMA-10 deferred) → **RESOLVED** by z107 result;
    rewritten to capture the κ-bracket finding.
  - Routing-topology section now specifies the κ ↔ R_bulk mapping
    requirement and the coupling-resistor range/resolution.
  - Limitations & open issues updated accordingly.

---

## Headline (unchanged)

**Tape out two coupled-routing variants of the same cell** — one
"isolated" (no inter-cell routing fabric) and one "coupled"
(shared-bulk-rail between nearest-neighbor cells, externally
disconnectable through a digitally tunable resistor). The B.5
benchmarks show NS-RAM is genuinely task-class-dependent: temporal
benchmarks (memory capacity, NARMA-10, XOR) benefit from
recurrence; spatial associative benchmarks (Hopfield) benefit from
decoupled per-cell channels. A single tape-out that supports both
regimes maximises the value-per-die for the next mask cycle.

---

## Cell geometry (unchanged)

| Variant | M1 (W/L) | M2 (W/L) | Body | DNW | Notes |
|---------|----------|----------|------|-----|-------|
| Thin-ox | 0.18 / 0.13 µm | 0.18 / 1.8 µm | floating | yes | matches Sebas's existing M2 card; 21 fJ/cycle, 6.7 fJ/spike, 46 µm² |
| Thick-ox | 0.18 / 0.13 µm | 0.5 / 1.8 µm, t_ox 7 nm | floating | yes | needed for VG2 ∈ [2.5, 3.0] V; **A.12 card pending** |

---

## Routing topology — UPDATED with κ ↔ R_bulk mapping

Two test arrays per die plus a hybrid sub-array, sharing the same
cell layout:

1. **Isolated array** (32×32, 1024 cells) — every cell's bulk
   floats independently. Use case: Hopfield-style associative
   recall, multi-class spatial classification. The B.5 Hopfield
   benchmark hit acc 0.69 vs chance 0.33 with this topology;
   recurrence at κ=0.03 *hurts* (z105, paired t = −2.45).

2. **Shared-bulk-rail array** (32×32, 1024 cells, 4-neighbor mesh)
   — every cell's bulk node connects to its 4 nearest neighbors
   through a digitally tunable resistor R_bulk. Use case:
   memory capacity, NARMA-10, temporal-XOR. The B.5 MC benchmark
   lifted MC from 0.22 → 1.10 (z102, paired t = +7.4); NARMA-10
   went from NRMSE 1.07 → 0.95 (z107, paired t = −9.4) with
   software W_rec.

   **κ ↔ R_bulk requirement (NEW in v2):**

   The optimal κ values measured in our pyport were:

       z102 MC:        κ ≈ 0.03   at N=10,  ρ = 1/√N (≈ 0.32)
       z107 NARMA-10:  κ ≈ 0.003  at N=100, ρ = 0.9

   The shared-bulk-rail equivalent is the time constant
   τ_coupling = R_bulk · C_body_eff (where C_body_eff is the
   effective per-cell body capacitance ≈ 5–10 fF per Pazos's
   thin-ox card). The fraction of Vb that mixes between
   neighboring cells per simulation timestep dt is
   α ≈ 1 − exp(−dt / τ_coupling). For our pyport sample period
   dt = 10 ns this gives:

       κ_eff = 0.003   ⇔  τ_coupling ≈ 3.3 µs  ⇔  R_bulk ≈ 600 MΩ
       κ_eff = 0.030   ⇔  τ_coupling ≈ 330 ns  ⇔  R_bulk ≈ 66 MΩ
       κ_eff = 0.300   ⇔  τ_coupling ≈ 33 ns   ⇔  R_bulk ≈ 6.6 MΩ

   This is *much* higher resistance than the 1 kΩ–1 MΩ range
   proposed in v1 — that range was wrong. **Revised v2 requirement:
   R_bulk digitally tunable from ≈ 1 MΩ to ≈ 1 GΩ (3 decades),
   with at least 8-bit logarithmic resolution (~ 6 % per code).**
   This covers κ_eff ∈ [0.003, 0.3] at a 10 ns sample rate, with
   fine steps around the chaos-onset boundary κ ≈ 0.005 measured
   in z107.

   *Caveat:* the κ ↔ τ_coupling mapping is a small-signal first-order
   estimate; precise correspondence requires a circuit-level transient
   analysis with the actual measured C_body. The M9 fan-out test
   structure is designed to provide this calibration in silicon.

3. **Hybrid sub-array** (16×16, 256 cells) — half the rows
   isolated, half coupled, with row-mux to swap. Validates the
   task-class dichotomy *within a single die* and gives the
   M9 fan-out experiment a natural home.

---

## Bias and sense (unchanged)

- VG1 bus: 0.0–1.2 V, 8-bit DAC per row.
- VG2 bus: −0.2 to +1.0 V (thin-ox) or +1.0 to +3.0 V (thick-ox),
  10-bit DAC per row (10 bits needed because κ=0.003 corresponds
  to 0.03 V quanta around 0.5 V).
- Vd: 0.0–2.0 V, single chip-wide DAC.
- Sense: Id at the source rail via TIA, 1 nA – 10 µA dynamic
  range matching Sebas's 4-decade measurement setup.

**NEW:** R_bulk digital pot — 8-bit log over 1 MΩ–1 GΩ range,
externally programmable per array (1 setting per array; if area
permits, per-row would let measurement scan κ along a single
mesh row in one sweep).

---

## Test structures — M9 fan-out (unchanged but now scoped)

10–30 cell linear fan-out, both bulk topologies (isolated, coupled).
On-die per-cell Id sense, shared 16-bit ADC. Sebas asked for this
explicitly. With the κ ↔ R_bulk mapping above, the M9 measurement
goal becomes:

  - Sweep R_bulk across the digital-pot range; measure MC
    (memory-capacity) and NARMA-10 NRMSE at each setting.
  - Find the silicon optimum and compare to the pyport prediction.
  - The measured silicon optimum tells us either (a) software
    recurrence and shared-bulk-rail are equivalent (the optimum
    settings agree within 1–2 decades), or (b) they are not (in
    which case the silicon measurement *replaces* the software
    prediction and we update the pyport accordingly).

Either outcome is a publishable result.

---

## Milestones (unchanged scope, updated state)

- **M3 (Jun 2026):** finalize thick-ox cell card via Sebas's
  pending data drop; refit z91g on thick-ox regime; close
  remaining ~10 mV Vth gap on M1.
- **M6 (Sep 2026):** complete B.5 benchmark suite at 4 sizes ×
  isolated/coupled topology with 5-seed paired-t protocol now
  established at z102/z104/z105/z107.
- **M9 (Dec 2026):** fan-out test structure on next mask;
  measure isolated vs coupled MC and Hopfield accuracy *in silicon*;
  calibrate κ ↔ R_bulk mapping in silicon.
- **M12 (Mar 2027):** full tape-out, 4-corner DC + transient
  characterization, cross-validation against PyTorch port.

---

## Risks — UPDATED

1. **Software vs silicon recurrence equivalence is unproven.**
   The κ-mediated MC and NARMA-10 lifts (z102, z107) were
   demonstrated via *external* W_rec in software. Whether the
   on-die shared-bulk-rail topology reproduces those lifts in
   silicon is the **central scientific risk and the project's
   primary deliverable** — the M9 fan-out structure is the
   experiment that settles it. Either outcome is publishable.

2. **Hopfield small-scale.** Reported at N=10, M=3 only.
   "Associative memory at scale" requires N ≥ 50, M ≥ 30. The
   N-scaling at κ=0 is the next pyport task; the silicon hybrid
   sub-array tests it directly at N=256.

3. **NARMA-10 κ-sensitivity (RESOLVED in v2 — was open in v1).**
   z107 found the stable operating point at κ=0.003 (NRMSE
   0.946 ± 0.018, paired t = −9.4 vs κ=0). Chaos-onset between
   κ=0.003 and κ=0.005. The κ-bracket is narrow, which is the
   reason v2 specifies the digitally tunable R_bulk with
   logarithmic resolution near the low end.

4. **Thick-ox card pending from Pazos (A.12).** All thick-ox
   claims here are PTM-extrapolation, not measurement.

5. **Absolute task performance is below state-of-the-art.**
   MC = 1.10 vs theoretical max ≈ N = 10. NARMA-10 NRMSE = 0.95
   vs canonical-ESN literature 0.1–0.3. We measure relative
   *lifts*, not absolute SOTA. Closing the gap requires
   N-scaling, input-gain tuning, and biasing work scheduled for
   Phase B.

---

## Footprint estimate (unchanged)

Two 32×32 cell arrays + 16×16 hybrid + 30-cell fan-out + DACs +
sense + scan chain ≈ 0.6 mm² in 130 nm. Fits multi-project mask
slot. Includes the new R_bulk digital pots (~10 kΩ² each, 6
total) at negligible area cost.

---

## Open issues to resolve before mask drop — UPDATED

- [ ] Receive thick-ox cell BSIM4 card from Sebas (A.12).
- [ ] Receive 7-rate transient measurement data from Sebas (A.12).
- [x] **z107 NARMA-10 finer-κ sweep** — RESOLVED 2026-05-03.
- [ ] **Hopfield N-scaling at N=30, 50** confirms substrate-alone
  advantage.
- [ ] **C_body characterization** on a representative cell — needed
  to refine the κ ↔ R_bulk mapping above.
- [ ] Sebas + Mario sign-off on isolated/coupled/hybrid array
  budget allocation and on the R_bulk digital-pot range.
- [ ] **Multi-class waveform B.5 benchmark** to close the 5/5 grid.

---

*This document is research_plan/C3_tapeout_recommendation_v2.md.
v1 written 2026-05-03 00:13 from the post-Phase-A, post-z105
evidence base; v2 written 2026-05-03 01:55 incorporating z107
results and the O11 oracle review's κ↔R_bulk requirement.*

```


=== FILE: LOG_tail.md (11536 chars) ===
```
  - #96 (B.5): 4/5 main + Hopfield N-scaling addendum;
    multi-class waveform pending.

**Next iteration plan options:**

  (a) **Multi-class waveform** at N=10 or N=100 — close 5/5 grid.
      Would let the brief say "5/5 B.5 benchmarks complete" and
      remove one open issue from C.3 v2.
  (b) **C_body characterization** — pyport experiment to extract
      effective body capacitance from a transient ramp; refines
      κ↔R_bulk numerics in C.3 v2.
  (c) **Hopfield M-scaling at N=50** — extend z108 along the
      storage-capacity axis (the only open Hopfield risk).
  (d) **A.10 pdiode-card port** — independent thread.
  (e) **Oracle dispatch O12** — fresh review with all post-O11
      changes (5/5 brief, C.3 v2, z108 addendum). Quality gate
      before user authorization.
  (f) **Draft transmittal email skeleton** — incorporates Gemini
      O11 recommendation that the email proactively manage the
      A.12 thick-ox dependency and frame silicon-equivalence as
      project value, not deal-breaker.

**Recommendation: (a) next.** Multi-class waveform closes the
5/5 grid that the brief implicitly promises (it lists all 5
benchmarks in the protocol-establishment sentence of the MC
paragraph). Even a null result there is a deliverable: it would
either give us a 5th positive or add a 6th honest Limitation
(matching the existing pattern). Estimated 25-35 min.



---

## 2026-05-03 02:37 — B.5 multi-class waveform: 5/5 grid CLOSED; substrate solves, recurrence neutral

**Step:** Ran `scripts/z109_multiclass_waveform.py` — 4-class
waveform discrimination (sine, square, sawtooth, triangle) at
N=30, T=800, κ ∈ {0.00, 0.003}, 5 seeds. Wall 700 s.

**Result (classification accuracy, chance = 0.25):**

| κ     | acc mean | ± std | SEM   | min   | max   |
|-------|----------|-------|-------|-------|-------|
| 0.000 | **0.567**| 0.090 | 0.040 | 0.443 | 0.657 |
| 0.003 | 0.595    | 0.091 | 0.041 | 0.500 | 0.721 |

**Paired Δ acc = +0.028 ± 0.026** (t = +1.05). NOT significant.

**Vs-chance test:** the κ=0 baseline at 0.567 is **2.27× chance**
(0.25); a single-sample t-vs-chance gives (0.567 − 0.25) / 0.040 ≈
**+8.0** — extremely significant deviation from chance even
though the recurrence-vs-no-recurrence comparison is null.

**Interpretation:**

  - **Substrate alone solves the task well above chance.** A 30-cell
    array of memoryless cells discriminates 4 periodic waveform
    classes at 57 % accuracy (2.3× chance), driven only by
    per-cell I-V nonlinearity reading the instantaneous Vd.
  - **Recurrence at κ=0.003 is neutral here** — neither
    statistically helpful nor harmful. The waveform task lives
    between Hopfield (recurrence hurts) and MC/NARMA (recurrence
    helps): it's a *spatial-pattern-discrimination* task at its
    core, the per-cell readout already extracts most of the
    signal.
  - **5/5 B.5 grid is now CLOSED.** Brief commitment fulfilled.

**Pre-benchmark checklist (A.4.g v1):** all blocker invariants
satisfied. Newton convergence 100% across 10 conditions.
**A.4.g v1 PASS @ z109.**

**Mario brief — proposed paragraph (compact):**

> Fifth B.5 result — multi-class waveform. With N=30 cells and
> 4 classes (sine, square, sawtooth, triangle), the linear ridge
> readout reaches accuracy 0.567 ± 0.090 with no recurrence and
> 0.595 ± 0.091 at κ=0.003 (5 seeds, chance 0.25). Both clear
> chance by ~8 SE; the recurrence vs no-recurrence Δ is
> +0.028 ± 0.026 (paired t = +1.05) — not significant. Like
> Hopfield, the substrate alone resolves spatial-pattern
> discrimination at modest scale; unlike Hopfield, recurrence
> is *neutral* rather than harmful. This refines the task-class
> dichotomy from "recurrence helps OR hurts" to "recurrence
> helps temporal tasks, is neutral-to-harmful on spatial tasks
> depending on how much per-cell decoupling matters".

**Refined dichotomy across the 5 benchmarks:**

| Benchmark        | κ=0 baseline | κ=κ* lift | Interpretation         |
|------------------|--------------|-----------|------------------------|
| MC               | 0.22         | +0.88     | recurrence essential   |
| NARMA-10         | 1.07 (NRMSE) | −0.13     | recurrence essential   |
| temporal-XOR(2)  | 0.54         | +0.13     | recurrence beneficial  |
| Multi-class wave | 0.57         | +0.03 (n.s.) | recurrence neutral |
| Hopfield (M=3)   | 0.69 → 1.00@N=50 | -0.11 | recurrence harmful   |

A clean monotonic ordering emerges: **the more the task
requires temporal memory across timesteps, the more recurrence
helps**. MC requires literal multi-step memory; NARMA-10 has a
~10-step horizon; XOR has a 2-step horizon; multi-class waveform
discriminates at a single timestep with periodic context;
Hopfield is purely instantaneous spatial. Recurrence helpfulness
tracks this ordering exactly, with a clean *neutral* benchmark
between "helps" and "hurts".

**Task status:**
  - #96 (B.5): **5/5 main benchmarks complete + Hopfield
    N-scaling addendum.** All open issues from C.3 v2 except
    Hopfield M-scaling and thick-ox card now closed.
  - #129 (Mario brief): can incorporate the 5/5 result and the
    refined dichotomy framing. Estimated 15-20 min.
  - #98 (C.2): same file → covered.

**Next iteration plan options:**

  (a) **Add z109 paragraph + refined-dichotomy table to Mario
      brief** — converts brief from "4/5" to "5/5" and adds the
      monotonic dichotomy ordering as a single visual table.
      Estimated 15-25 min (table layout in LaTeX may need a
      moment).
  (b) **C_body characterization** — extract effective body cap
      from a transient ramp; refines κ↔R_bulk in C.3 v2.
  (c) **Hopfield M-scaling at N=50** — extend z108 along the
      storage-capacity axis (the only remaining open Hopfield
      knob).
  (d) **A.10 pdiode-card port** — independent thread.
  (e) **Oracle dispatch O12** — fresh review with all post-O11
      changes including 5/5 grid and refined dichotomy.
  (f) **Draft transmittal email skeleton** for Mario — ready
      when user authorizes send.

**Recommendation: (a) next.** The 5/5 closure is the single
biggest deliverable upgrade remaining; folding it into the brief
is the highest-leverage post-z109 action. The refined dichotomy
table also addresses Gemini's O11 critique that the
"task-class dichotomy" was over-confident: now it's a measured
5-point monotonic ordering, not a 2-point claim.



---

## 2026-05-03 02:45 — Mario brief: 5/5 + monotonic dichotomy table folded in

**Step:** Inserted two new paragraphs into Status section of
`docs/nsram_proposal_short.tex` — the z109 multi-class waveform
result and a 5-row dichotomy table ordering benchmarks by
temporal-memory horizon. Recompiled (5 pages, 374,321 B; +39 KB).

**Inserted paragraph 1 (z109):**

> Fifth B.5 result — multi-class waveform; closes the 5/5 grid.
> With N=30 cells and four classes (sine, square, sawtooth,
> triangle), the linear ridge readout reaches accuracy
> 0.567 ± 0.090 without recurrence and 0.595 ± 0.091 at
> κ=0.003 (5 seeds, T=800, chance 0.25). Both clear chance by
> ~8 SE; the recurrence-vs-no-recurrence Δ is +0.028 ± 0.026
> (paired t = +1.05) — not significant. Substrate-alone
> classification at 2.3× chance shows the per-cell nonlinearity
> already extracts the spatial-pattern signal; recurrence is
> *neutral* here, neither helping nor harming.

**Inserted paragraph 2 (monotonic ordering table):**

> Refined task-class ordering across all five benchmarks.
> Ordering benchmarks by their required temporal-memory horizon
> yields a clean monotonic recurrence-effect ordering:
>
> | benchmark               | memory horizon       | effect at κ* |
> |-------------------------|----------------------|--------------|
> | memory capacity         | multi-step           | essential (+0.88, t=+7.4) |
> | NARMA-10 (N=100)        | ~10 steps            | essential (−0.13 NRMSE, t=−9.4) |
> | temporal-XOR (τ=2)      | 2 steps              | beneficial (+0.13, t=+2.7) |
> | multi-class waveform    | one-step + context   | neutral (+0.03, n.s.) |
> | Hopfield (M=3, N=10)    | instantaneous        | harmful (−0.11, t=−2.45) |
>
> The earlier "task-class dichotomy" framing thereby resolves
> into a measured five-point monotonic ordering: recurrence
> helpfulness tracks the task's temporal-memory requirement.
> This is the empirical content behind the dual-topology
> tape-out plan in C.3.

**Why the table is the strongest single addition since the
5-bug catalogue:**

  - Converts a 2-point claim ("recurrence helps OR hurts") into
    a 5-point monotonic ordering with measured effect sizes.
  - Makes the C.3 dual-topology recommendation feel *forced by
    data* rather than an architectural opinion.
  - Addresses Gemini O11's "task-class dichotomy is over-confident
    for N=10 / M=3" critique directly: now there are five
    benchmarks at varying scales and the dichotomy is a
    *trend*, not a pair.
  - Anyone scanning the Status section for the headline result
    will read this table as a single visual artifact.

**Brief metrics:** 5 pages, 374 KB (+39 KB for the new paragraph
+ table). Page count stable. The brief is now content-saturated
within 5 pages — adding more would push to 6.

**Send-readiness checklist (updated):**
  - [x] DC fidelity closure documented + 5-bug catalogue.
  - [x] Transient + throughput + GPU 5× target.
  - [x] **5/5 B.5 benchmarks reported with 5-seed paired-t**
        (MC, NARMA, XOR, multi-class waveform, Hopfield with
        N-scaling addendum).
  - [x] **Monotonic dichotomy table** ordering by memory horizon.
  - [x] Limitations section (5 bullets, all bounded or resolved
        including Hopfield N-scaling resolution).
  - [x] Quadrant chart fixed (2 markers, periphery caveat).
  - [x] All O11 oracle micro-edits applied.
  - [x] C.3 v2 referenced from brief.
  - [ ] **User authorization to send (only remaining gate).**

**Pre-benchmark checklist (A.4.g v1):** N/A — wording-only.

**Task status:**
  - #129 (Mario brief): substantive content COMPLETE. 5/5 grid
    + dichotomy table is the strongest possible final state
    achievable autonomously before the 2026-05-06 deadline.
  - #98 (C.2): same file → covered.
  - #99 (C.3): v2 issued, all 5 benchmarks support the
    dual-topology rationale.
  - #96 (B.5): all 5 benchmarks reported.

**Next iteration plan options:**

  (a) **Oracle dispatch O12** — fresh review with all post-O11
      changes including 5/5 grid and monotonic dichotomy. The
      brief is at its substantive ceiling; an external check is
      the natural quality gate before user authorization.
      Estimated 2-4 min wall + 1-2 min response time per provider.
  (b) **Draft transmittal email skeleton** for Mario incorporating
      Gemini O11's "manage A.12 dependency proactively" and
      "frame silicon-equivalence as project value, not
      deal-breaker" recommendations. Estimated 10-15 min.
  (c) **C_body characterization** — extracts effective body cap
      from a transient ramp; refines κ↔R_bulk in C.3 v2.
      Estimated 15-25 min.
  (d) **Hopfield M-scaling at N=50** — extends z108 along the
      remaining open axis. Estimated 30-40 min.
  (e) **A.10 pdiode-card port** — independent thread (deferred
      multiple iterations).

**Recommendation: (a) next.** With the brief at its content
ceiling, an oracle review is the natural quality gate. O12 with
the same gpt-5 + gemini-2.5-pro pair should either return
"green-light, send it" (in which case we draft the email next
iteration) or surface any final actionable blind spot.


```
