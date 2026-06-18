# NS-RAM Brief v4.8 — DRAFT 2026-05-19

**Supersedes v4.7 (2026-05-19).** Changes vs v4.7:
- §1 baseline corrected: median **1.163 dec** (current pyport LEGACY, fwd+bwd; triode RMSE
  VG1=0.6 = 1.183). The previously-quoted **1.62 dec** came from a different config
  (constant I_par=100 nA injection) and was misleading; it is removed from the headline.
- §2 candidate table updated: **C3 (BSIM4 §10.1 JTS-TAT) added as a second KILLSHOT** alongside
  C1 (Pazos NPN). Both internally falsified 2026-05-19.
- §6 non-claims: explicit "we are NOT claiming Pazos NPN OR BSIM4 JTS-TAT as the missing
  parallel-path."
- §7 Sebas ask: re-ranked — T-sweep > W-scaling > negative-Vd > thick-oxide cell card.
- §8 new: "what 1.163 dec means for Mario" — translation to practical-use terms.

**To:** Mario (cc Sebas). **From:** Eric / ikaros team. **Tone:** technical-plain, honest
negatives, no marketing. Bootstrap CIs where computed; fwd+bwd reported separately.
**Status:** draft, internal review pending, ship target 2026-06-01.

---

## One-page abstract

We have a compact-model port of Sebas's 2T NS-RAM cell. The current pyport LEGACY config
(no parallel-current injection) fits 33-bias DC at **median 1.163 dec fwd+bwd**
(triode VG1=0.6 RMSE = 1.183, fwd 1.151 / bwd 1.168; n=66; source: `results/Pillar_I_C3_jts_tat/verdict.md`).
That is **not** the sub-1.0 dec target. We did not get under 1.0 dec yet.

We spent the week trying to identify a parallel-conduction mechanism that could close the gap.
The headline result is honest-negative, and now applies to **two** textbook candidates:

- **C1 — Pazos parasitic NPN (Gummel-Poon parallel) is FALSIFIED.** NPN-OFF improves median
  fit by **1.19 dec** vs NPN-ON (NPN-ON=4.461, NPN-OFF=3.268, n=66 each). T-coefficient
  300→400 K = **−0.21** dec (gate ≥ +3; predicted band +5 to +7 dec for NPN-dominant);
  result is INDETERMINATE — i.e. NPN turning on has no temperature signature in the data.
  Heatmap over (BJT_IS, BJT_BF): no basin below 2.1 dec.
  Source: `results/Pillar_I_C1_npn_parallel/verdict.md`.
- **C3 — BSIM4 §10.1 JTS-TAT at default parameters is FALSIFIED.** At JTSS=2.5e-7
  (BSIM4 §10.1 default), JTS-ON median = 1.398 dec, JTS-OFF (Is=0) median = 1.163 dec —
  i.e. turning JTS on **worsens** fit by 0.234 dec. T-coefficient 300→400 K = **0.00 dec**
  (target band +0.15..+0.35 dec; closed-form TAT prediction 0.24). The solver suppresses
  V_jct under the simulated bias, killing the TAT current; this is the mechanism by which
  the gate fails.
  Heatmap (JTSS, NJTS) shows a tunable optimum JTSS=1e-6 / NJTS=5 reaching **0.852 dec
  forward-only**, which beats the LEGACY forward-only baseline of 1.151 dec by 0.30 dec.
  However: (a) it fails the ≤1.0 dec gate when computed at the heatmap **best cell** rather
  than the median, (b) JTS-OFF still wins on the joint fwd+bwd median, and (c) the bwd
  branch is not improved. We do not promote JTS-TAT to "the mechanism".
  Source: `results/Pillar_I_C3_jts_tat/verdict.md`.

**Net:** the residual gap from 1.163 dec to the sub-1.0 dec target is consistent with several
mechanisms, but two of the prominent textbook candidates (C1 NPN, C3 JTS-TAT) are now ruled
out as the primary parallel-conduction path. **Discrimination of what remains requires
temperature-sweep + W-scaling experimental data that only Sebas can supply.** Until then we
are running C2 (self-consistent floating-body sub-Vt) and C4 (measurement-artifact /
contact-resistance, not yet tested) in parallel.

What we *can* ship right now: a working torch port with documented **1.163 dec** honest fit;
an exported Verilog-A model (`nsram_cell_2T.va`) with explicit square-law-vs-BSIM4 fidelity
caveat; a topology-zoo phase diagram on a distilled BSIM4 cell (104/120 edge-of-chaos, broadband
NARMA r² up to 0.968 under heterogeneous σ ≥ 0.05 V); an internal-use ML predictor for
VG1 ∈ [0.2, 0.6] interpolation (curve-CV 0.119 dec PASS, LOVO not yet locked); and the I.6
adversarial bias list (3 killshot biases pre-registered 2026-05-19) we ask Sebas to measure.

---

## §1. DC-fit current state (honest)

All numbers are torch pyport 2T NS-RAM, 33-bias DC, fwd+bwd, T=300 K (model-default),
n=66 unless noted. Source files cited per row.

| Variant | Median dec (66) | fwd | bwd | Triode RMSE VG1=0.6 | Source |
|---|---|---|---|---|---|
| **Torch port, LEGACY (no I_par, JTS-OFF, NPN-OFF default)** | **1.163** | 1.151 | 1.168 | 1.183 | `results/Pillar_I_C3_jts_tat/verdict.md` LEGACY block |
| Torch port, NPN ON, LEGACY-driven Ic | 3.618 | 3.072 | 3.736 | 2.382 | `results/Pillar_I_C1_npn_parallel/verdict.md` LEGACY block |
| Torch port, NPN ON, NPN-mode | 4.461 | 4.136 | 4.499 | 2.154 | `results/Pillar_I_C1_npn_parallel/verdict.md` NPN_ON |
| Torch port, NPN OFF (gain=0) | 3.268 | 3.267 | 3.268 | 2.154 | `results/Pillar_I_C1_npn_parallel/verdict.md` NPN_OFF |
| Torch port, JTS ON (BSIM4 §10.1 default JTSS=2.5e-7) | 1.398 | 1.398 | 1.397 | 2.832 | `results/Pillar_I_C3_jts_tat/verdict.md` JTS_ON |
| Torch port, JTS OFF (Is=0) | 1.163 | 1.151 | 1.168 | 1.183 | `results/Pillar_I_C3_jts_tat/verdict.md` JTS_OFF |

Notes:
- LEGACY = the actual default of the current pyport (no parallel injection, no JTS, no NPN).
  This is what the brief from now on calls "baseline".
- The 1.62-dec number from earlier internal notes was a **constant I_par=100 nA injection**
  test config and is not a baseline; it is removed from the brief.
- Pre-registered MASTER_FIX target is < 0.7 dec; campaign gate is < 1.0 dec — neither met.

**LOVO pre-registration:** not yet locked at time of writing. To be locked before any
"new best dec" claim is reported.

---

## §2. Mechanism investigation: candidate physics shortlist

The gap from 1.163 dec → < 1.0 dec is *empirically* not closed by the textbook leakage paths.
After today's analyses we hold a shortlist, not an identification:

| ID | Candidate | Verdict 2026-05-19 |
|----|-----------|---|
| **C1** | Pazos parasitic NPN (Gummel-Poon parallel) | **KILLED.** NPN-OFF beats NPN-ON by **1.19 dec** median; T-coeff **−0.21** dec (predicted +5..+7); heatmap floor 2.1 dec. Source: `results/Pillar_I_C1_npn_parallel/verdict.md`. |
| **C3** | BSIM4 §10.1 JTS-TAT (trap-assisted tunneling) | **KILLED at default.** JTS-ON worsens by 0.234 dec vs JTS-OFF; T-coeff **0.00** dec (target +0.15..+0.35; TAT closed-form 0.24) — solver suppresses V_jct under simulated bias, killing the TAT current. Heatmap optimum (JTSS=1e-6, NJTS=5) reaches **0.852 dec fwd-only**, beats LEGACY fwd by 0.30 dec — but FAILS ≤1.0 gate at best cell on joint metric and bwd not improved. Source: `results/Pillar_I_C3_jts_tat/verdict.md`. |
| C2 | Self-consistent floating-body M1 sub-Vt | **OPEN — in flight** (top priority). |
| C4 | Measurement artifact / contact-resistance floor (~5 MΩ probe leakage) | **OPEN — not yet tested.** Parsimony check planned. |
| — | Well-tap, STI edge-FET, GIDL as *primary* path | Ruled out at diagnostic bias (≥16 dec undershoot, I.4). |

**Explicit caveat:** the residual gap is consistent with multiple mechanisms; pending T-sweep
+ W-scaling experimental data from Sebas to discriminate.

The Pillar-I.6 adversarial-bias list (§4) is the team's recommended minimal additional
measurement set for that discrimination.

---

## §3. Pillar I prior-narrowing (not falsifying gates — per O90 oracle critique)

### §3.1 Forward/backward rectification (I.1)
1/33 biases pass the |log10 R| ≥ 0.5 + p<0.01 + 10/33-bias gate. Verdict: MOSFET-like.
Caveat: protocol measures sweep hysteresis (|Vd| up-down, same polarity), not true ±Vd
rectification. Strict-mode rerun planned (frozen ROI + block-bootstrap + symmetric
three-outcome gate).

### §3.2 VG2 sensitivity at VG1=0.6 (I.3)
Slope d log10|Id|/dVG2 at four Vd probes:

| Vd (V) | slope (dec/V) | 95 % CI | R² |
|---|---|---|---|
| 0.05 | −0.013 | [−0.024, −0.002] | 0.408 |
| 0.10 | −0.021 | [−0.035, −0.003] | 0.476 |
| 0.20 | −0.027 | [−0.050, −0.003] | 0.390 |
| 0.50 | −0.075 | [−0.159, −0.003] | 0.318 |

All four vote well-tap (insensitive to VG2). R² weak. Treat as prior-narrowing.

### §3.3 Temperature scaling prediction (I.4)
At diagnostic bias (VG1=0.6, VG2=−0.05, Vd=0.05), per-candidate from BSIM4 model:

| Candidate | I(T=300 K) (A) | ΔI 300→400 K (dec) |
|---|---|---|
| well-tap | 1.5e−20 | +5.17 |
| STI | 0 (model inactive) | 0 |
| GIDL | ~1e−37 | +0.82 |

All three undershoot measured 250 nA by ≥16 dec at 300 K. Magnitude-killshot for the
standard candidates as primary conductor. Caveat: model-derived priors only, not independent
evidence (per O90).

### §3.4 Geometry (I.2)
From `data/sebas_2026_04_22/2tnsram_simple.asc`:
- M1 (NS-FET, floating body): W=0.36 µm, L=180 nm
- M2 (gate-control, bulk): W=0.36 µm, L=1800 nm
- tox = 4.0 nm (M2 card); 3.3 nm in M1 header comment (inconsistent)
- Process-node ambiguity: slides say "130 nm" and "180 nm CMOS"; LTspice `.param Ln=0.18u`
  is consistent with 180 nm drawn channel; SPICE card filenames say 130 nm. **Request
  one-line confirmation from Sebas.**

---

## §4. Pillar I.6 — Adversarial bias list (HEADLINE deliverable; pre-registered 2026-05-19)

Three killshot biases at maximum pairwise model-prediction divergence (250 nA-class signal),
pre-registered before any Sebas-measurement campaign:

| Rank | VG1 | VG2 | Vd | T (K) | discriminates | max div (dec) |
|---|---|---|---|---|---|---|
| 1 | 0.0 | −0.2 | 1.4 | 400 | well-tap vs STI | 25.73 |
| 2 | 0.3 | −0.2 | 1.4 | 400 | well-tap vs GIDL | 25.73 |
| 3 | 0.0 | +0.6 | 0.40 | 220 | STI vs GIDL | 25.29 |

Any one of these biases, measured on real silicon, falsifies two of three standard candidates.
Caveat: I.4 magnitude killshot already shows none is the *primary* path; these biases
discriminate residual contributions and constrain the missing-physics search space.

---

## §5. Deliverables ready or near-ready

### §5.1 `predictor.pkl` (Pillar V, INTERNAL ENGINEERING SAFETY-NET — not a scientific claim)
- XGBoost + LightGBM + MLP ensemble on 2705 rows, 33-bias DC at T=300 K.
- **Curve-CV (interpolation within VG1 ∈ [0.2, 0.6]):** worst median **0.07 dec** in the
  current campaign run (down from 0.119 in v4.7) — PASS vs 0.5 gate.
- **LOVO (extrapolation to a held-out VG1):** not yet locked; previously 3.06 dec — FAIL vs
  0.5 gate. Only three VG1 values in dataset; LOVO forces wide extrapolation.
- Shippable for interpolation use within the measured envelope; not for extrapolation.
  Engineering, not physics.

### §5.2 `nsram_cell_2T.va` (Pillar IV.1, Verilog-A)
- Skeleton complete, 26/26 begin/end balanced, KCL signs 1:1 with pyport residuals.
- **Fidelity caveat (square-law):** M1/M2 intrinsic Ids uses textbook square-law +
  body-effect + CLM, NOT full BSIM4 (mobility/DIBL/CLM-regions/poly/quantum/moderate-inv
  smoothing). Expected mismatch 0.1–1 decade in Id over Vd sweep. Foundry BSIM4 VA can be
  swapped in by replacing the two intrinsic-Ids blocks; parameter NAMES (U0, VTH0, AGIDL,
  BGIDL, ALPHA0, BJT_IS, BJT_BF, BJT_VAF, snap_BV, JTSS, NJTS, ...) preserved for
  compatibility.
- IV.2 pyport↔ngspice 33-bias diff harness in flight; gate max-RMSE ≤ 1e-4.
- IV.3 Cadence-Spectre lint scheduled.

### §5.3 Pillar III topology zoo phase diagram (PASS both gates)
- 120 conditions (6 topologies × 5 stimuli × 4 heterogeneity levels), N=2048, T=4000, dt=0.005.
- Cell: distilled MLP surrogate of the Pillar-V ensemble (val median |res| = 0.076 dec
  inside VG1 ∈ [0.2, 0.6] envelope).
- **Edge-of-chaos:** 104/120 conditions PASS (Lyapunov in [−0.05, 0.05]).
- **NARMA-30 r²:** max 0.994 (sine+noise, gameable carrier). Honest broadband-stim top-3:
  - WS_SMALL_WORLD + Mackey-Glass, σ=0.10 → 0.968
  - HIERARCHICAL_3LEVEL + Mackey-Glass, σ=0.05 → 0.955
  - BA_SCALE_FREE + Mackey-Glass, σ=0.10 → 0.937
- σ ≥ 0.05 V heterogeneity required to break global sync.
- Caveats: in-sample readout (upper-bound r²); Rosenstein-proxy Lyapunov; runs inside the
  0.07-dec interpolation envelope, NOT at the 1.163-dec full-DC honest gate. Train-test
  rerun planned Days 6–7.

### §5.4 Methods paper outline (honest-negative framing)
Working title: *Pinpointing the parasitic conduction path in a 2T NS-RAM: a multi-candidate
falsification study*.

1. Introduction & device
2. Measurement set & priors (Pillar I.1–I.4)
3. Candidate shortlist; **two internal killshots: C1 NPN, C3 JTS-TAT**
4. Model port & 1.163-dec honest fit (LEGACY pyport)
5. Discriminator design (I.6 adversarial bias list)
6. Engineering deliverables: ML emulator (caveats), Verilog-A (caveats), topology phase diagram
7. What remains and what we ask of the next measurement campaign

---

## §6. What we are **not** claiming

- We are not claiming sub-1.0 dec DC fit. Current best honest baseline: **1.163 dec**.
- **We are not claiming Pazos NPN OR BSIM4 JTS-TAT as the missing parallel-path; both
  have been internally falsified** (sources cited in §1, §2).
- We are not claiming to have identified the mechanism behind the residual gap. We hold a
  shortlist (C2, C4) and an adversarial-discrimination protocol (I.6).
- We are not claiming the Verilog-A export is silicon-accurate. It is topology-and-sign
  accurate; intrinsic Ids is reduced-fidelity square-law.
- We are not claiming the ML predictor extrapolates. It interpolates within VG1 ∈ [0.2, 0.6]
  (curve-CV 0.07 dec); LOVO not yet locked but previously FAIL at 3.06 dec.
- We are not claiming the topology phase-diagram NARMA r² numbers (0.994) represent honest
  reservoir capability — sine+noise is a carrier-latching artefact. Honest broadband NARMA
  r² top is 0.968 (in-sample), expected to drop 5–20 % under proper train-test split.
- We are not claiming SHAP feature importance carries physics meaning given only three VG1
  rows in dataset.

---

## §7. What we ask of Sebas (re-ranked)

In order of discrimination power for the *remaining* candidates (C2, C4) and for the
residual-contribution map (C1, C3 internally killed but still informative):

1. **Temperature sweep** at the 250 nA diagnostic bias (VG1=0.6, VG2=−0.05, Vd=0.05),
   T ∈ {220, 250, 300, 350, 400} K, single device. Discriminates C2 vs the standard
   leakage paths (well-tap/STI/GIDL) via Arrhenius slope; also closes the C3-JTS-TAT
   T-coeff question (model predicted +0.24 dec, we measured 0.00 — silicon T-sweep is the
   tiebreaker).
2. **W-scaling**: one additional geometry, single device with W=0.18 µm and/or W=0.72 µm
   at L=180 nm. Splits well-tap (∝WL) from GIDL (∝W) by factor ~2–4. Independent of
   T-sweep.
3. **Negative-Vd point**: a single |Vd|≈0.5 V measurement at Vd=−0.5 V (true ±Vd, not
   sweep hysteresis) at one or two (VG1, VG2) corners. Falsifies MOSFET-like-only via real
   rectification check; isolates diode-like parallel paths.
4. **Thick-oxide cell card / process-node confirmation**: one-line answer on (a) process
   node 130 vs 180 nm drawn channel, (b) measurement temperature on the existing
   33-bias data, (c) whether 0.18 µm Ln is real-drawn or placeholder, (d) tox 3.3 vs 4.0 nm.

Cost to Sebas of item 4: low (one email). Items 1–3: one measurement session each.
Information value to the campaign: very high.

---

## §8. What 1.163 dec means for Mario (practical-use translation)

A "median dec" of 1.163 means: over the 66 fwd+bwd sweep points, the model's |log10 Id|
differs from measurement by a median of **1.163 decades**. Concretely:

- **Per-current interpretation.** If the measurement says Id = 100 nA at some bias, the
  model is within roughly **×14** at the median worst-case point (10^1.163 ≈ 14.6). At the
  best half of points it is closer; at the worst half it is farther. This is not
  quantitative-accurate.
- **What it is usable for.** Sensitivity-and-direction simulations — "if I change VG2 by
  +0.1 V, which direction and roughly how much does Id move", topology-level reservoir
  experiments (which is exactly what Pillar III uses, with the 0.07-dec predictor as the
  inner cell), and as a sign-and-shape-correct skeleton for the Verilog-A export.
- **What it is NOT usable for.** Quantitative current-budget design (e.g. sizing a
  pull-up resistor against the leakage floor at a specific bias), absolute-power
  predictions, or claims that any single decade-level feature in the simulated curve is
  real silicon physics rather than model error.
- **Bias-dependent caveat.** At VG1=0.2 LEGACY median is 0.631 dec (~×4 current error)
  — usable. At VG1=0.6 LEGACY median is 1.732 dec (~×54 current error) — only for
  qualitative sensitivity. The bias-aware ML predictor (§5.1) is the safer tool inside
  the VG1 ∈ [0.2, 0.6] interpolation envelope (0.07 dec ≈ ×1.18 — quantitative-usable).

The 1.163-dec figure is what we ship as the *honest physics baseline*. The 0.07-dec
predictor is the engineering safety-net for downstream design work that needs
quantitative-grade Id values inside the measured envelope.

---

## §9. Tone note (internal)

This brief is intentionally honest-negative on the model identification. The position is:
*we ran the campaign correctly, we report what we found, two prominent textbook candidates
(C1 NPN, C3 JTS-TAT) are internally falsified, we did not yet identify the mechanism.*
Honest negative results have value — they constrain the search space and prevent the
collaboration from wasting silicon on a wrong hypothesis. The shippable engineering
deliverables (VA, ML emulator, topology phase diagram) hold their value independent of the
DC-fit mystery.
