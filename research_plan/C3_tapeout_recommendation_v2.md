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
   thin-ox card). For a 2D mesh where each cell connects to
   `deg` neighbors through R_bulk, the per-step Vb mixing is

       α_total = 1 − exp(−deg · dt / (R_bulk · C_body_eff))

   with deg = 4 for a nearest-neighbor 2D mesh, deg = 1 for a
   single shared common rail. **This 4× factor was missed in v2's
   first numerical pass and is restored here per the O12 oracle
   review.** For dt = 10 ns and C_body_eff in 5–10 fF:

   *4-neighbor mesh (deg = 4):*
       κ_eff = 0.003   ⇔  R_bulk ≈ 1.3–2.7 GΩ
       κ_eff = 0.030   ⇔  R_bulk ≈ 130–260 MΩ
       κ_eff = 0.300   ⇔  R_bulk ≈ 11–22 MΩ

   *single shared rail (deg = 1):*
       κ_eff = 0.003   ⇔  R_bulk ≈ 330–670 MΩ
       κ_eff = 0.030   ⇔  R_bulk ≈ 33–67 MΩ
       κ_eff = 0.300   ⇔  R_bulk ≈ 3.3–6.7 MΩ

   We adopt the **2D 4-neighbor mesh** as the canonical topology
   (it preserves the spatial-locality structure of the W_rec
   matrix used in the pyport sweeps). **Revised v2.2 R_bulk
   requirement (widened from v2.1's 10 MΩ–4 GΩ):** digitally
   tunable from ≈ **1 MΩ to ≈ 100 GΩ** (5 decades), with at
   least 10-bit logarithmic resolution (~ 1.4 % per code).
   This wide range is *deliberately generous*: see Caveat #5
   below; the v2.1 numerical R_bulk values are pyport
   coupling-model values, not silicon predictions. The M9
   characterization will identify the *actual* silicon
   operating point within this range.

   *Caveats (added with the v2.1 fix):*

   1. The α_total formula is a small-signal first-order estimate;
      precise correspondence requires a circuit-level transient
      analysis with the actual measured C_body. The M9 fan-out
      test structure provides this calibration in silicon.

   2. The pyport κ multiplies a *random* W_rec drawn from
      N(0, 1/√N) (or spectral-radius-controlled to ρ=0.9 in
      z107). A uniform 4-neighbor positive-conductance network
      is *not* spectrally equivalent to a random matrix; the M9
      measurement must therefore calibrate the silicon coupling
      against the software prediction (this is exactly the
      project's central scientific deliverable per Risk #1).

   3. At the top of the R_bulk range, off-state subthreshold
      leakage and body-junction diode leakage may set an
      effective floor above the digital-pot's nominal high; the
      mask should budget margin for this and provide an option
      for an external lab-mode pot during the M9 characterization
      phase.

   4. Above α ≈ 0.1 the linearization α ≈ deg·dt/τ used in
      earlier v2 numbers diverges from the exact form; we use
      the exact `1 − exp(...)` here.

   5. **(NEW in v2.2, from z110/z113 findings 2026-05-03)** The
      pyport's *emergent* body-cap time constant lands at
      **τ_body ≈ 2.1–2.4 µs across 10 of 12 sampled (VG1, VG2)
      operating points** (z113; with C_body = 1 fF, R_internal ≈
      2.1 GΩ in the implicit-Euler solver). Two regime-crossing
      outliers at strong-M2-on biases show non-monotonic decay
      and are not captured by a simple exponential fit. The
      uniform 2.1 µs is set by the body-junction diode clamping
      Vb_eq at +0.546 V across most of the bias space; once Vb
      breaks free of that clamp the dynamic changes regime. This
      is three decades slower than Pazos's measured silicon
      **τ_body ≈ 0.7 ns**. The α formula above is therefore a generic
      coupling model; its *numerical* R_bulk values are
      pyport-coupling-model values, not silicon predictions.
      The two physical mechanisms — software VG2 recurrence at
      the dt = 10 ns sample timestep, vs silicon shared-bulk-rail
      Vb diffusion at the body-cap timescale — are physically
      distinct, and only approximately equivalent in the
      small-signal limit. **The M9 fan-out characterization is
      the calibration step that establishes the actual silicon
      mapping, exactly as Risk #1 already states.**

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

**v2.2 (2026-05-03 03:50, after z110 τ_body finding):** R_bulk
digital pot — **10-bit log over 1 MΩ–100 GΩ (5 decades)**, was
8-bit over 10 MΩ–4 GΩ in v2.1. The wider range is justified by
the pyport-vs-silicon τ_body mismatch (z110: pyport ≈ 2 µs at
subthreshold; Pazos paper ≈ 0.7 ns silicon — a 3-decade gap).
Externally programmable per array; per-row programmability
optional if area permits. Optional external lab-mode pot remains
on the BOM for the M9 characterization phase.

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
results and the O11 oracle review's κ↔R_bulk requirement;
v2.1 patch applied 2026-05-03 03:00 fixing the 4-neighbor degree
factor in the κ↔R_bulk derivation per O12 oracle review;
v2.2 patch applied 2026-05-03 04:00 widening R_bulk range
(1 MΩ–100 GΩ) and adding caveat #5 after z110 measured the
pyport-emergent τ_body ≈ 2 µs vs Pazos's 0.7 ns silicon value.*
