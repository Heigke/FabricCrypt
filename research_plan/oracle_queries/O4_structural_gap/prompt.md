# Oracle Query O4 — Structural Gap in Snapback Shape

## Setup (terse)

- DUT: 2T NS-RAM cell, Sebas Pazos, 130 nm bulk CMOS.
  - M1 = 130 nm DNWFB nmos (deep-n-well floating-body), gate=VG1, drain=Vd, source=GND, body floats.
  - M2 = 130 nm bulk nmos, gate=VG2, drain=Vd, source=M1.B (P-body of M1), body= (TBD: floating in our port, possibly GND in deck).
  - Parasitic NPN: emitter=GND, base=M1.B (floating P-body), collector=Vd. Bf=5e4, Ikr=0.1, IS scaled by `mbjt`.
  - DNW (vnwell) tied to +2 V via vnwell_Rs (currently 3e9 Ω) → DNW→Pbody body diode forward-biased, supplying base current to NPN.
- Measurements: 33 I_D vs V_d sweeps across (VG1, VG2) bias grid.
  - VG1 ∈ {0.20, 0.40, 0.60} V
  - VG2 ∈ {-0.2 → +0.5} V depending on VG1 branch (see Sebas slide 1 text).
  - V_d swept 0 → 2.0 V (or 3.5 V for some).
- Solver: BSIM4 v4.8.3 PyTorch port + Gummel-Poon NPN + arclength continuation (tol 1e-13, branch detection, adaptive ds), produces I_D(V_d) including hysteretic snapback.

## What we have implemented and verified

- BSIM4 §6.1 impact ionization (Iii via IIMOD branches) — present, tested.
- BSIM4 §6.2 GIDL/GISL — present, tested.
- 4 body diodes (bs/bd of M1+M2, plus DNW→Pbody well-body diode, vnwell=+2 V).
- Gummel-Poon NPN with Bf=5e4, Ikr=0.1, IS×mbjt.
- Per-bias overrides loaded from Sebas's `2Tcell_BSIM_param_DC.csv`: ETAB, K1, ALPHA0, BETA0, NFACTOR, IS, area, mbjt.
- Arclength continuation, vnwell_Rs sweep tooling, two-model evaluation harness.

## What we have NOT verified or implemented (be honest)

- **BSIM4 §10.2 self-heating** (SHMOD, RTH0, CTH0): not in our port at all. M1/M2 cards declare RTH0/CTH0 but we ignore.
- **BSIM4 §6.4 PSCBE second-order substrate term**: present in `compute_dc.py` but not audited end-to-end; PSCBE2=1e-5 in BA cards vs 1e-20 (effectively off) in PTM card, large divergence.
- **RBODYMOD** (distributed body resistance ladder, RBPB/RBPS/RBPD/RBSB/RBDB): set to 0. Cards do not provide values either, so likely fine — but worth confirming.
- **LTSpice transient initialisation**: deck contains `.op 0` and likely a `.tran` with UIC ramp. We do pure DC continuation. If LTSpice's "DC" sweep is actually a slow ramp, snapback may be a dynamic effect rooted in `.tran`.
- **M2.B topology**: we wired M2.B to floating P-body (= M1.B). The .asc deck may have M2.B = GND (ground-referenced bulk). This is the single biggest topology ambiguity.
- **mbjt semantics**: we apply `mbjt` to scale IS of the well-body diode (DNW→Pbody) only. It may instead scale the BJT collector current Bf or the NPN IS. Sebas's CSV has mbjt up to 5; if it gates Bf this is a big deal.
- Whether the BSIM4 IIMOD branch (alpha0 vs alpha1 weighting) matches Sebas's deck exactly across IIMOD=0/1/2.

## Empirical sweep results so far (loss = log10-RMSE in current decades)

| Configuration                                    | median (dec) | p90 (dec) | shape |
|--------------------------------------------------|-------------:|----------:|-------|
| Constant α0×10, vanilla β0 (baseline)           | 0.95         | 1.97      | smooth tongues |
| β0 = 2.0 globally (O3 recommendation)            | 2.99         | —         | worse |
| vnwell_Rs sweep min @ Rs=3e9 Ω (current best v24)| **0.896**    | 2.43      | smooth tongues |

In **all** our configurations: predictions are smooth flat tongues 1e-13 → 1e-7 across all VGs.
A subset of measurements jump ~6 decades to ~1e-3 at high V_d in some bias regimes. We never reproduce this jump.

## Visual evidence (attached)

- `sebas_iv_fits_main.png` — Sebas's actual I-V fit triptych (LTSpice deck output vs measurement). Symbols=measurements, lines=simulations. Multiple VG1 panels showing high-V_d snapback **reproduced** by his fit.
- `our_v24_fit_vs_meas.png` — our v24 result (Rs=3e9 best). 3 panels (VG1 = 0.2/0.4/0.6 V), open circles = measurement, lines = our prediction. Predictions sit at ~1e-12 floor, measurements show late-V_d jumps in some traces, never matched.
- `sebas_param_*.png` — Sebas's per-VG2 parameter curves (NFACTOR, K1, ETAB, BETA0). These are sanity inputs we already use via the CSV.

## Specific questions

**(a) Sanity-check on target.** Looking at `sebas_iv_fits_main.png`: does Sebas's fit ALSO produce smooth tongues that miss the high-I_d jumps, or does it actually reproduce them? If the former, our 0.9 dec residual matches his and we're done. If the latter (which our reading of the plot suggests — his lines clearly bend up at high V_d), what physics in his deck does ours lack to make that jump appear?

**(b) Best-bet structural hypothesis.** Of the following, which most likely explains the missing high-V_d jump:
 1. Self-heating (SHMOD/RTH0/CTH0) creating thermal runaway latch.
 2. RBODYMOD distributed body network changing the V_b dynamics seen by the parasitic NPN.
 3. M2.B = GND topology (vs our M2.B = floating P-body) — fundamentally re-routes the body-current path.
 4. `mbjt` actually gating BJT Bf (not just well-body diode IS), so we're under-weighting collector injection.
 5. LTSpice doing a `.tran` UIC ramp, so snapback is a dynamic / displacement-current effect outside DC scope.

**(c) Measurement artifact?** Could the high-V_d jump be a measurement artifact (instrument compliance, pulsed-IV settling, ESD diode turn-on at the pad) that even Sebas's DC fit doesn't reproduce — making it correctly out-of-scope for a steady-state model?

**(d) Single discriminating experiment.** What ONE env-var or one-line edit on our existing port would best discriminate among the hypotheses above, given we can rerun the 33-curve eval in ~5 min?

## Code attachments

- `nsram_cell_2T.py` — top-level cell wrapping
- `compute_dc.py` — BSIM4 DC core (incl. PSCBE)
- `compute_iimpact_and_gidl.py` — §6.1/§6.2
- `compute_body_diodes.py` — 4 body diodes incl. DNW
- `bjt.py` — Gummel-Poon NPN
- `arclength.py` — continuation solver
- `z91g_two_model_validation.py` — eval harness
- `M1_130DNWFB.txt`, `M2_130bulkNSRAM.txt`, `parasiticBJT.txt` — Sebas's BSIM/BJT cards
- `2tnsram_simple.asc` — Sebas's LTSpice deck (TOPOLOGY GROUND TRUTH)
- `2Tcell_BSIM_param_DC.csv` — per-bias overrides

Please be concrete and structural. The parametric route is exhausted — we need the missing physics or the missing wire.
