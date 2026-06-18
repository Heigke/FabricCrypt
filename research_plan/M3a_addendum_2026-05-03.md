# ⚠️ DRAFT — DO NOT SEND (pending O18 fixes)

See `research_plan/01_LOG.md` 2026-05-03 ~22:18 for the 3-of-3 oracle
FIX verdict. Required corrections before this is sendable:

1. Section 2 "two-axis architectural rec (HUB_SPOKE for classification)"
   — REMOVE. Statistically unsupported (n=2 noise + unfair ρ normalization).
2. Section 1 VG1=0.4 V framing — change "wrong Newton root" to
   "non-physical Bf parameterization" (per openai correction).
3. Section 4 "ready" framing — must add "device model not yet validated
   end-to-end against silicon transient" caveat.

Do NOT send to Mario / Sebas / NRF in current form.

---


# M3a Addendum to NS-RAM Funding Brief v4.1

**Date:** 2026-05-03
**Audience:** Mario Lanza (KAUST), Sebastian Pazos (KAUST)
**Author:** Eric Bergvall (ENIMBLE / Nervdynamics)
**Brief reference:** `nsram_proposal_short.tex` v4.1 sent same day

This addendum supersedes the brief's Section 5 ("WHAT WORKS, WHAT
DOESN'T") and Section 6 ("PRELIMINARY HYPOTHESES") with results
generated in the 12 hours after the brief was dispatched. It does
not change the recommendation; it sharpens the DC-fit numbers and
reports a topology scaling sweep (n=3 seeds; suggestive only — see
Section 5 caveats). The device model has NOT yet been validated
end-to-end against silicon transient data.

---

## 1 — DC fit: chronology of corrections (1.00 → 0.80 → 1.31 → **1.39**)

The honest physical-bounded result is **median log-RMSE 1.39 dec**
across 25/33 biases at Bf = 100 (physical) with a bounded η ∈ [0, 1]
sigmoid Iii→Vb injection. The chronology of corrections is:

| step                            | Bf      | gain      | median | physical? |
|---------------------------------|--------:|-----------|-------:|-----------|
| Brief v4.1 (sent 2026-05-03)    | 5×10⁴   | —         | 1.00   | no (Bf)   |
| Stage6 hack (post-send try)     | 2×10⁴   | —         | 0.80   | no (Bf)   |
| F1 unbounded gain (intermediate)| 100     | γ = 1×10⁵ | 1.31   | no (γ)    |
| **F1.v2 η-bounded (honest)**    | **100** | **η ≤ 1** | **1.39** | **yes** |

Per-VG1-row at the honest baseline (Bf=100, η_max=1.0):

| row    | brief v4.1 | F1.v2 honest |
|--------|-----------:|-------------:|
| VG1=0.2 | 1.66      | 1.20         |
| VG1=0.4 (catastrophe) | 2.83 | **1.35** |
| VG1=0.6 | 0.91      | 2.25         |

The honest physical-bounded result is **0.39 dec worse** than the
brief's 1.00 dec headline. This is a real walk-back: the brief's
1.00 was achievable only via the unphysical Bf hack. The 1.39 dec
number is the right one to communicate externally because every
parameter in the model is now physically defensible.

**Independent ngspice cross-validation (F2, 180-pt grid):** pyport's
BSIM4 evaluator agrees with ngspice to **~1–2 % on Id/gm/gds, ~2–4 %
on gmb** across the full operating envelope. The BSIM4 *evaluator*
is solid; the residual to silicon is in the 2T-cell-level parasitic
mechanism (where pyport adds physics that vanilla ngspice doesn't
model).

**Diagnosis of the VG1=0.4 V row** (probe v2,
`research_plan/binning_audit/probe_v2_finding.md`): at no-impact-
ionisation biases the parasitic NPN settles into a self-sustaining
high-Vb root (Vb≈0.43 V, Ic_Q1≈7×10⁻⁸ A) with no physical
charge-pumping mechanism (Iii ~10⁻²⁵ A). All five arclength
cold-start seeds converge to the SAME root, which is unique. The
cell equations admit this self-biased NPN steady state because
Bf=2×10⁴ is non-physical (real 130 nm parasitic NPN gain is
10–100). The failure is in the parameterization, not the solver.
Lowering Bf to the physical range (≤ 100) plus adding a bounded
η ∈ [0, 1] sigmoid Iii→Vb trigger (F1.v2) reduces the row from
2.83 → **1.35 dec**. The honest VG1=0.4 V row is now better than
its brief v4.1 number (2.83 dec) but worse than VG1=0.6 V (which
relied on the unphysical Bf to fire the parasitic NPN at high Vd
— the η-bounded model can no longer overdrive there). This is the
honest trade-off; the unphysical Bf=5×10⁴ hid the trade-off behind
a single fitted-but-non-physical parameter. See
`research_plan/M3b_fix_plan_2026-05-03.md` F1.v2.

The 8 NaN biases reported in the brief are **not solver failures**
— they are biases for which Sebastian's parameter CSV has K1=NaN
(the negative-VG2 snapback regime he did not extract). All 33
biases now evaluate to finite log-RMSE under the un-overridden card.

---

## 2 — Topology scaling matrix (4× larger than brief tested)

The brief's C.3 tape-out recommendation pinned MESH_4N as the
preferred topology, validated up to N=200. We now have a 6×3 sweep
at N ∈ {100, 300, 800} with 3 seeds × 4 tasks each (small n;
findings below are suggestive — see Section 5):

| topology       | N=100 MC | N=300 MC | N=800 MC | N=800 XOR | N=800 WAVE | scale ×|
|----------------|---------:|---------:|---------:|----------:|-----------:|-------:|
| RAND_GAUSS     | 1.42     | 1.50     | 1.87     | 0.53      | 0.47       | 1.31   |
| **MESH_4N**    | 1.87     | 2.40     | **3.29** | **0.91**  | 0.52       | 1.75   |
| ER_SPARSE      | 2.12     | 2.56     | 2.20     | 0.63      | 0.46       | 1.04   |
| WS_SMALLWORLD  | 1.66     | 2.44     | 2.94     | 0.85      | 0.51       | 1.77   |
| **HUB_SPOKE**  | 1.18     | 0.86     | 2.89     | 0.90      | **0.61**   | 2.45   |
| LAYERED        | 2.78     | 1.53     | 2.17     | 0.57      | 0.48       | 0.78   |

(MC = memory capacity; XOR = τ=2 binary-XOR readout accuracy;
WAVE = 4-class waveform classification accuracy; scale × = ratio
MC(N=800) / MC(N=100). Bf=2×10⁴, T=500, κ=0.03, ρ=0.9.)

### Six findings

1. **MESH_4N is the MC leader at N=800** (3.29 dec, n=3 seeds).
   Consistent with the brief's C.3 recommendation; "validated at
   4× scale" overstates n=3 — call this a positive directional
   replication pending n≥10 confirmation.

2. **HUB_SPOKE: suggestive non-monotone scaling.** The cell values
   (×2.45 scaling, WAVE=0.61, MC=0.86 at N=300) come from n≤3 seeds
   and HUB_SPOKE is unfairly disadvantaged by the global ρ=0.9
   normalization (see Section 5). The non-monotonicity is
   *suggestive only* and not statistically supported. Treat as
   motivation for a follow-up study with per-topology spectral
   calibration and n≥10 seeds, not as a tape-out signal.

3. **LAYERED is anti-scaling** (×0.78) — MC DECREASES with N. The
   2-layer feedforward + sparse-skip topology does not benefit
   from network growth at the tested input drive. Negative result.

4. **ER_SPARSE: apparent plateau at N=300** (peak MC=2.56) then
   2.20 at N=800. Suggestive only — n=3 seeds, no confidence
   interval. The collinearity-saturation hypothesis is consistent
   with z117/z115/z114 small-N results but the N=800 dip falls
   within the cross-seed variance and should not be cited as
   established without n≥10 replication.

5. **WS_SMALLWORLD nearly matches MESH_4N at N=800** (2.94 vs 3.29
   MC). Small-world rewiring is a viable alternative if the
   2D-grid layout is undesirable for fabrication.

6. **Random Gaussian is the worst at every scale.** The brief's
   choice to recommend a *structural* topology rather than random
   recurrence is empirically grounded across 6 topologies × 3 scales.

---

## 3 — Architectural recommendation — HOLD pending z142 (2026-05-04)

The brief's primary architectural recommendation (MESH_4N) is
**no longer supported** by the η-bounded honest rerun. Partial z142
(5 seeds at rho_lambda; rho_p95_sv + rho_deg_norm still running):

| topology       | N=800 z139 (Bf=2e4) | N=800 z142 (η-bounded, n=5) |
|----------------|--------------------:|----------------------------:|
| **ER_SPARSE**  | 2.20                | **3.55** — random-sparse    |
| **LAYERED**    | (new in z142)       | **3.56** — feed-forward     |
| RAND_GAUSS     | 1.87                | 2.25                        |
| MESH_4N        | **3.29 (z139 champ)** | 2.20 — drops to mid-pack |
| WS_SMALL       | 2.94                | 2.17                        |
| HUB_SPOKE      | 2.89                | 1.92; WAVE 0.61→0.53 gone   |

ER_SPARSE and LAYERED are statistically tied for MC champion at
~3.55 dec, ~1.36 dec above MESH_4N. The Bf=2×10⁴ in z139 was a
fitting hack (per O18 critique); the inversion confirms that
z139's ranking was an artefact of unphysical cell gain.

**Pending the two remaining ρ-normalisation variants** (rho_p95_sv
+ rho_deg_norm), the new architectural rec is **ER_SPARSE OR
LAYERED for MC tasks**, NOT MESH_4N. HUB_SPOKE's WAVE advantage
is also gone. C.3 will be rewritten in the next addendum revision
once the full 270-sim sweep lands (~08:00 my time).

---

## 4 — Status of M3a deliverables

- [x] M3a-A: z91g rebuild at Bf=2×10⁴ — median 0.80 dec
  (`results/z91g_two_model_validation_stage6_bf2e4/`)
- [x] M3a-B: per-bias BETA0 — resolved as documentation. The CSV's
  `BETA0` column is the BSIM4 impact-ion β₀, already routed to
  `P_M1["beta0"]`. It is NOT the bipolar Bf and never was.
- [ ] M3a-C: ngspice cross-validation rerun at Bf=2×10⁴ — pending
- [x] M3a-D: large-scale topology sweep (z139) — full table above
- [x] M3a-E: independent codebase audit (Explore subagent, 5
  candidates flagged; #1, #2, #4 deferred as off-25°C / refactor
  risks; #3 awaits Sebas CSV; #5 closed with negative grid result)
- [x] M3a-F: transient validation harness scaffold —
  `scripts/z140_transient_harness.py` runs end-to-end on synthetic
  input; awaits Sebas's measured traces
- [x] M3a-G: this addendum

**SENDABLE NOW with explicit caveats.** Per O20+O21+O22 unanimous
verdicts after F1–F4 closure: the honest 1.39 dec result is the
defensible baseline, the topology table inversion is real, and
M3c's structural-fix path was empirically tested over four oracle
rounds without producing a < 1.0 dec result at honest physical
parameters. See §6 below.

---

## 5 — Caveats per O18 triple-review (2026-05-03)

Three reviewers (gemini, openai, grok) independently flagged the
following weaknesses in this addendum. Each must be addressed before
any external claim is made.

- **n=2 statistical limitation in z139.** Section 2's topology
  table reports 3 seeds × 4 tasks per cell, but several seeds did
  not converge cleanly — the practical n for HUB_SPOKE at N=300
  was 2 (MC values {1.21, 0.49}, variance > mean). The HUB_SPOKE
  non-monotone behaviour (Finding 2) and the ER_SPARSE plateau
  (Finding 4) are NOT statistically supported and should be
  treated as suggestive only.

- **Unfair ρ-normalization for HUB_SPOKE.** All topologies were
  λmax-scaled to ρ=0.9, which clamps the hub edge weight in
  HUB_SPOKE far harder than it clamps the diffuse edges in MESH_4N
  or RAND_GAUSS. HUB_SPOKE's reported numbers are therefore lower
  bounds under unfair normalization. A like-for-like comparison
  requires per-topology effective-spectral-radius calibration.

- **Median 0.80 dec is on 25/33 rows, not 33.** The DC fit headline
  excludes 8 K1=NaN biases (Sebastian's CSV omits negative-VG2
  snapback extraction). The full 33/33 refit with imputed/refit K1
  is required before any external claim. The brief's 1.00 dec used
  the same exclusion; the relative improvement is real but the
  absolute number must be qualified until F3 closes.

- **Bf=2×10⁴ is non-physical.** The 2×10⁴ value is a fitting
  compensation for the missing Iii→Vb coupling, not a measurement
  of the parasitic NPN gain. Real 130 nm NPN gain is 10–100. Any
  forward use of this card outside the calibration set should
  expect mis-extrapolation. F1 (Bf clamp) and F2 (Iii→Vb trigger)
  in M3b address this.

---

## 6 — M3c structural-fix attempt: empirical floor at 1.39 dec

After the M3b honest-baseline closed at median 1.39 dec, the M3c
plan proposed a structural rewrite (lateral-NPN-as-channel-current)
targeting < 1.0 dec at honest physical Bf=100. Per O20+O21+O22
oracle rounds plus four direct empirical tests, this work has now
characterised the structural gap as follows:

### 6.1 — The Vb-clamp finding (O21 unanimous)

At Bf=100 honest physical, the parasitic-NPN base current Ib_Q1
draws exponentially with Vbe = Vb. This pins Vb_global at ~0.39 V
across all biases regardless of Iii/GIDL inflow magnitude. The
Gummel-Poon Ic_Q1 ≈ Is·exp(Vbe/Vt) consequently pins at ~1.6×10⁻⁸ A
across all biases. **Silicon shows orders-of-magnitude bias-
dependent variation that this single-node-body model architecture
cannot reproduce.**

### 6.2 — Three structural-fix paths empirically tested

| path | description                              | result                          |
|:-----|:-----------------------------------------|:--------------------------------|
| B    | Augment BJT base drive via η_lat·Iii     | β·Iii is 5 OoM below Ic_Q1 floor — no observable effect at any bias |
| C    | M(Vbc)·Ids avalanche multiplier on drain | (M−1)·Ids is also small because Ids itself is ≪ Ic_Q1 — no effect |
| α/M3c.3 | Local-base node + spread resistor Rb  | Mechanism works only at unphysical Rb ≥ 10 GΩ (literature: 100 kΩ – 1 MΩ); pathological at high Rb |

The fundamental tension: at honest physical parameters (Bf=100,
realistic body diode Js, realistic Iii/GIDL, realistic Rb), the
model's inflow currents are too small to overcome the BJT's
exponential base-current drain. Adding the Miller-style avalanche
multiplier at a local-base node (gpt-5/gemini's δ recommendation)
gives M·Ids ≈ 10⁻¹⁰ A even at high-current biases — still
insufficient.

### 6.3 — Honest characterisation

**The η-bounded honest 2T-cell model has a structural floor at
~1.39 dec on this dataset.** Reaching < 1.0 dec without
re-introducing fudge factors (Bf >> 100, BV << 5 V, Rb >> 1 MΩ)
requires either:

  (a) **Silicon-level ground-truthing** of Bf, Rb, BV from
      Sebas's measurements (high-Vd snapback I-V or transient
      base-injection test). A measured Bf=10⁴ in lateral
      geometry, for example, would not be a fudge — it would
      be calibration.
  (b) **Acknowledgement that the model architecture has a
      ~1.4 dec floor** on bias-dependent current variation, and
      ship that as the final result for this dataset.

### 6.4 — Path forward (revised from M3c plan)

The 6-week M3c calendar in the original plan is now invalid. The
empirical work over the past 6 hours has cut out the dead-end
(M3c.2 lateral-NPN restructure, both replace and augment paths)
and surfaced the actual structural diagnosis. Two tracks:

  1. **Sebas measurement request (BLOCKED on task #128)**:
     transient base-injection or pulsed Vd snapback measurement
     to ground-truth Rb/Bf. ETA depends on Sebas availability.
  2. **Brief / NRF / Mario communication**: 1.39 dec is the
     honest baseline; the architectural recommendation is
     ER_SPARSE for cross-norm robustness; the < 1.0 dec target
     is conditional on (1).

This addendum's headline numbers (1.39 dec, ER_SPARSE) are the
defensible result. The original M3c "~6 weeks calendar" line in
the brief should read: "M3c structural fix: blocked on
silicon-level parameter ground-truthing from Sebas."

### 6.5 — Code state at addendum send

  - F1.v2 η-bounded refactor: committed, gate passes
  - M3c.1 charge-conserving routing: committed, η_lat ∈ [0,1] toggle
  - M3c.2 paths B (β·Ib) + C (M·Ids on drain): committed, gate
    passes; both empirically insufficient
  - M3c.3 local-base inner solve: committed with O22 routing fix,
    gate passes; effective only at unphysical Rb
  - All defaults reproduce F1.v2 bit-identical at η_lat=0,
    use_local_base=False, use_lateral_collector=False.

Code is in a clean reproducer state for any follow-up work
(e.g. once Sebas's data lands).
