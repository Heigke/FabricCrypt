# NS-RAM Brief v4.7 — DRAFT 2026-05-19

**To:** Mario (cc Sebas). **From:** Eric / ikaros team. **Tone:** technical-plain, honest negatives,
no marketing. Bootstrap CIs where computed; fwd+bwd reported separately where relevant.
**Status:** draft, internal review pending, ship target 2026-06-01.

---

## One-page abstract

We have a compact-model port of Sebas's 2T NS-RAM cell that fits 33-bias DC at **median 1.62 dec
fwd+bwd** (parallel-path I_par=100 nA constant, torch BSIM4 port w/ NPN ON, 2026-05-18). That is
**not** the sub-1.0 dec target. We did not get under 1.0 dec yet.

We spent the last week trying to *identify* the mechanism behind the 100 nA constant residual.
The headline result is honest-negative:

- **Parasitic NPN as parallel Gummel-Poon path is falsified.** NPN-OFF improves median fit by
  1.19 dec vs NPN-ON. Heatmap basin floor 2.1 dec. The torch port already wires NPN in
  parallel — so 1.62 dec is the **with-NPN-on** number, and NPN is not the mechanism.
- **Standard candidates ruled out as primary conductor.** Well-tap junction at the diagnostic
  bias predicts 1.5e−20 A, 16 decades below the measured 250 nA floor. STI edge-FET and GIDL
  are similarly off (orthogonal model predictions, all undershoot). The dominant 100 nA path
  is not one of the three textbook BSIM4 leakage candidates.
- **Forward/backward sweep asymmetry says MOSFET-like, not diode-like** (1/33 biases pass
  the rectification gate). Caveat: protocol sweeps |Vd| up-then-down, same polarity; this is
  hysteresis evidence, not true ±Vd rectification. Treat as prior-narrowing.
- **VG2 sensitivity at VG1=0.6 (slope d log10|Id|/dVG2)** votes well-tap (all four Vd probes,
  slopes −0.013 to −0.075 dec/V). Treat as prior-narrowing, not a falsifying gate. R² 0.32–0.48.

**Net:** the 100 nA constant residual is consistent with multiple mechanisms; **discrimination
requires temperature-sweep + W-scaling experimental data that only Sebas can supply.** Until
then we are running BSIM4 JTS-TAT (C3) and self-consistent floating-body sub-Vt (C2) fits in
parallel, and parsimony-checking the artifact / contact-resistance hypothesis (C4).

What we *can* ship right now: a working torch port with documented 1.62 dec honest fit; an
exported Verilog-A model (`nsram_cell_2T.va`) with explicit square-law-vs-BSIM4 fidelity
caveat; a topology-zoo phase diagram on a distilled BSIM4 cell (104/120 edge-of-chaos, NARMA
r² up to 0.968 broadband-stimulus held under heterogeneous σ ≥ 0.05 V); an internal-use ML
predictor for VG1 ∈ [0.2, 0.6] interpolation (curve-CV 0.119 dec PASS); and an adversarial
bias list (Pillar I.6) the team would like Sebas to measure to discriminate the candidate set.

---

## §1. DC-fit current state (honest)

| Variant | Median dec (33 biases) | Notes |
|---|---|---|
| Torch port, I_par=100 nA constant, NPN ON | **1.62** | 2026-05-18, fwd+bwd |
| Torch port, NPN OFF (no I_par) | 3.27 | 2026-05-19, C1 killshot control |
| Torch port, NPN ON (no I_par) | 4.46 | 2026-05-19, C1 killshot baseline |
| Vbs-clamped baseline | 2.907 | reference floor |

Best-on-record is 1.62 dec. The pre-registered MASTER_FIX target is < 0.7 dec, the current
campaign gate is < 1.0 dec — neither is met. Honest gap remains open.

**LOVO pre-registration:** not yet locked at time of writing. To be locked before any "new
best dec" claim is reported.

---

## §2. Mechanism investigation: candidate physics shortlist

The 100 nA constant-current residual is *empirically* required to hit 1.62 dec but the physical
origin is not yet identified. After today's analyses we hold a shortlist, not an identification:

| ID | Candidate | Today's verdict |
|----|-----------|---|
| C1 | Parasitic NPN (Gummel-Poon parallel) | **DEAD** — NPN-OFF improves by 1.19 dec; killshot fires. |
| C2 | Self-consistent floating-body M1 sub-Vt | OPEN — to be fit |
| C3 | BSIM4 JTS-TAT (junction trap-assisted-tunneling) | OPEN — in flight (top priority) |
| C4 | Measurement artifact / contact-resistance floor (~5 MΩ probe leakage) | OPEN — parsimony check |
| — | Well-tap, STI edge-FET, GIDL as *primary* path | ruled out at diagnostic bias (≥16 dec undershoot, I.4) |

**Explicit caveat:** the 100 nA constant residual is consistent with multiple mechanisms;
pending T-sweep + W-scaling experimental data from Sebas to discriminate.

The Pillar-I.6 adversarial-bias list (below) is the team's recommended minimal additional
measurement set for that discrimination.

---

## §3. Pillar I prior-narrowing (not falsifying gates — per O90 oracle critique)

### §3.1 Forward/backward rectification (I.1)
1/33 biases pass the |log10 R| ≥ 0.5 + p<0.01 + 10/33-bias gate. Verdict: MOSFET-like.
Caveat: protocol measures sweep hysteresis (|Vd| up-down, same polarity), not true ±Vd
rectification. Strict-mode rerun planned (frozen ROI + block-bootstrap + symmetric three-outcome
gate).

### §3.2 VG2 sensitivity at VG1=0.6 (I.3)
Slope d log10|Id|/dVG2 at four Vd probes:

| Vd (V) | slope (dec/V) | 95 % CI | R² |
|---|---|---|---|
| 0.05 | −0.013 | [−0.024, −0.002] | 0.408 |
| 0.10 | −0.021 | [−0.035, −0.003] | 0.476 |
| 0.20 | −0.027 | [−0.050, −0.003] | 0.390 |
| 0.50 | −0.075 | [−0.159, −0.003] | 0.318 |

All four vote well-tap (insensitive to VG2). R² weak. Caveat: I_par residual not directly
computable (model-without-parallel current trace not in tree); raw |Id| at low Vd is used as
proxy.

### §3.3 Temperature scaling prediction (I.4)
At diagnostic bias (VG1=0.6, VG2=−0.05, Vd=0.05), per-candidate from BSIM4 model:

| Candidate | I(T=300 K) (A) | ΔI 300→400 K (dec) |
|---|---|---|
| well-tap | 1.5e−20 | +5.17 |
| STI | 0 (model inactive) | 0 |
| GIDL | ~1e−37 | +0.82 |

All three undershoot measured 250 nA by ≥16 dec at 300 K. **Magnitude-killshot** for the standard
candidates as primary conductor. Caveat: model-derived priors only; **not independent evidence**
(per O90).

### §3.4 Geometry (I.2)
From `data/sebas_2026_04_22/2tnsram_simple.asc`:
- M1 (NS-FET, floating body): W=0.36 µm, L=180 nm
- M2 (gate-control, bulk): W=0.36 µm, L=1800 nm
- tox = 4.0 nm (M2 card); 3.3 nm in M1 header comment (inconsistent)
- **Process-node ambiguity flagged:** slides say both "130 nm" and "180 nm CMOS" for the
  same device family. LTspice `.param Ln=0.18u` is consistent with 180 nm drawn channel; SPICE
  card filenames say 130 nm. **Request one-line confirmation from Sebas.**

---

## §4. Pillar I.6 — Adversarial bias list (HEADLINE deliverable, requested for Sebas measurement)

Maximum pairwise model-prediction divergence biases (250 nA-class signal):

| Rank | VG1 | VG2 | Vd | T (K) | discriminates | max div (dec) |
|---|---|---|---|---|---|---|
| 1 | 0.0 | −0.2 | 1.4 | 400 | well-tap vs STI | 25.73 |
| 2 | 0.3 | −0.2 | 1.4 | 400 | well-tap vs GIDL | 25.73 |
| 3 | 0.0 | +0.6 | 0.40 | 220 | STI vs GIDL | 25.29 |

Any *one* of these biases measured on real silicon would falsify two of three standard candidates.
Caveat: I.4 magnitude killshot already shows none is the *primary* path; these biases still
discriminate residual contributions and constrain the missing-physics search space.

---

## §5. Deliverables ready or near-ready

### §5.1 `predictor.pkl` (Pillar V, INTERNAL ENGINEERING SAFETY-NET, **not a scientific claim**)
- XGBoost + LightGBM + MLP ensemble on 2705 rows, 33-bias DC at T=300 K.
- **Curve-CV (interpolation within VG1 ∈ [0.2, 0.6]):** worst median 0.119 dec — PASS vs 0.5
  gate.
- **LOVO (extrapolation to a held-out VG1):** worst median 3.06 dec — FAIL vs 0.5 gate. Only
  three VG1 values in dataset; LOVO forces wide extrapolation.
- Shippable for interpolation use within the measured envelope; not for extrapolation. **This
  is engineering, not physics** (per O90).
- SHAP feature importance computed but not presented as physics insight.

### §5.2 `nsram_cell_2T.va` (Pillar IV.1, Verilog-A)
- Skeleton complete, 26/26 begin/end balanced, KCL signs 1:1 with pyport residuals.
- **Fidelity caveat:** M1/M2 intrinsic Ids uses textbook square-law + body-effect + CLM,
  NOT full BSIM4 (mobility/DIBL/CLM-regions/poly/quantum/moderate-inv smoothing/etc.).
  Expected mismatch 0.1–1 decade in Id over Vd sweep. Foundry BSIM4 VA can be dropped in by
  swapping the two intrinsic-Ids blocks; parameter NAMES (U0, VTH0, AGIDL, BGIDL, ALPHA0,
  BJT_IS, BJT_BF, BJT_VAF, snap_BV, etc.) are preserved for compatibility.
- IV.2 pyport↔ngspice 33-bias diff harness in flight today; gate max-RMSE ≤ 1e-4.
- IV.3 Cadence-Spectre lint scheduled.

### §5.3 Pillar III topology zoo phase diagram
- 120 conditions (6 topologies × 5 stimuli × 4 heterogeneity levels), N=2048, T=4000, dt=0.005.
- Cell: distilled MLP surrogate of the Pillar-V ensemble (val median |res| = 0.076 dec inside
  VG1 ∈ [0.2, 0.6] envelope).
- **Edge-of-chaos:** 104/120 conditions PASS (Lyapunov in [−0.05, 0.05]).
- **NARMA-30 r²:** max 0.994 (sine+noise, gameable carrier). Honest broadband-stim top-3:
  - WS_SMALL_WORLD + Mackey-Glass, σ=0.10 → 0.968
  - HIERARCHICAL_3LEVEL + Mackey-Glass, σ=0.05 → 0.955
  - BA_SCALE_FREE + Mackey-Glass, σ=0.10 → 0.937
- σ ≥ 0.05 V heterogeneity required to break global sync; ER and MODULAR underperform WS and
  HIERARCHICAL on broadband stim.
- **Caveats:** in-sample readout (no train-test split — upper-bounded NARMA r²); Rosenstein-proxy
  Lyapunov; runs inside the 0.119-dec interpolation envelope, NOT at the 1.62-dec full-DC honest
  gate. Train-test rerun planned Days 6–7.

### §5.4 Methods paper outline
Honest-negative framing: what we measured, what we ruled out, what remains. NO physics-claim
title. Working title: *Pinpointing the parasitic conduction path in a 2T NS-RAM: a multi-candidate
falsification study*. Outline:

1. Introduction & device
2. Measurement set & priors (Pillar I.1–I.4)
3. Candidate shortlist and the C1 NPN killshot
4. Model port & 1.62-dec honest fit
5. Discriminator design (I.6 adversarial bias list)
6. Engineering deliverables: ML emulator (caveats), Verilog-A (caveats), topology phase diagram
7. What remains and what we ask of the next measurement campaign

---

## §6. What we are **not** claiming

- We are not claiming sub-1.0 dec DC fit. Current best: 1.62 dec.
- We are not claiming to have identified the mechanism behind the 100 nA constant residual.
  We hold a shortlist (C2, C3, C4) and an adversarial-discrimination protocol (I.6).
- We are not claiming the Verilog-A export is silicon-accurate. It is topology-and-sign accurate;
  intrinsic Ids is reduced-fidelity square-law.
- We are not claiming the ML predictor extrapolates. It interpolates within VG1 ∈ [0.2, 0.6];
  worst LOVO extrapolation is 3.06 dec FAIL.
- We are not claiming the topology phase-diagram NARMA r² numbers (0.994) represent honest
  reservoir capability — sine+noise is a carrier-latching artefact. Honest broadband NARMA r²
  top is 0.968 (in-sample), expected to drop 5–20 % with proper train-test split (in progress).
- We are not claiming SHAP feature importance carries physics meaning given only three VG1
  rows.

---

## §7. What we ask of Sebas

In order of discrimination power (highest first):

1. **Temperature sweep** at the 250 nA diagnostic bias (VG1=0.6, VG2=−0.05, Vd=0.05),
   T ∈ {220, 250, 300, 350, 400} K, single device. **Discriminates C2/C3/well-tap/STI/GIDL via
   Arrhenius slope.**
2. **One additional geometry**: a single device with L=0.5 or 1.0 µm at W=0.36 µm. Splits
   well-tap (∝WL) from GIDL (∝W) by factor ~3.
3. **One-line confirmation**: process node 130 vs 180 nm drawn channel; measurement
   temperature on the existing 33-bias data; whether 0.18 µm Ln is real-drawn or placeholder.
4. **The three I.6 adversarial biases measured on the existing geometry** — falsifies two of
   three standard candidates per bias.

Cost to Sebas of items 1+3: low (≈one measurement session and one email reply). Information
value to the campaign: very high.

---

## §8. Tone note (internal)

This brief is intentionally honest-negative on the model identification. The position is:
*we ran the campaign correctly, we report what we found, we did not yet identify the mechanism.*
Honest negative results have value — they constrain the search space and prevent the
collaboration from wasting silicon on a wrong hypothesis. The shippable engineering deliverables
(VA, ML emulator, topology phase diagram) hold their value independent of the DC-fit mystery.
