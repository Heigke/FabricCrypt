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
