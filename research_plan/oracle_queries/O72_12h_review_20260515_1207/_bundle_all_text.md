# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: context.md (16056 chars) ===
```
## 2026-05-15 06:21 — 3h MEP+DS-N tick: ACTIVE ikaros=scripts/z365_perVG1_bbo.py, APU=47C. R-55a KILL-SHOT closed model program. Phase A.3 SURR-V4 done. Apps phase B done (5 surv+5 retract). z374/z375 daedalus apps still in flight.
## 2026-05-15 06:37 — all-night stale
## 2026-05-15 06:47 — :47 tick ACTIVE: scripts/z365_perVG1_bbo.py APU=46C sentinel=alive
## 2026-05-15 06:53 — 2h deep-dive: all 4A/B/C/D done, KILL-SHOT 05:30. No new dispatch.
## 2026-05-15 06:53 — master fix tick: model program CLOSED (R-55a KILL-SHOT 05:30). Apps phase B done. O70 12h oracle in flight. No new launches.
## 2026-05-15 06:55 — 6h oracle critique tick: SKIP (O70 12h oracle already in flight covers same scope, dispatched 06:00)
## 2026-05-15 07:07 — all-night stale
## 2026-05-15 07:13 — topology stale (KILL-SHOT 05:30, program closed)
## 2026-05-15 07:37 — all-night stale
## 2026-05-15 07:47 — :47 tick ACTIVE: scripts/z365_perVG1_bbo.py APU=46C sentinel=alive
## 2026-05-15 08:07 — all-night + topology stale; KILL-SHOT 05:30 closes program
## 2026-05-15 08:37 — all-night stale
## 2026-05-15 08:47 — :47 tick ACTIVE: scripts/z365_perVG1_bbo.py APU=46C sentinel=alive
## 2026-05-15 08:53 — 2h deep-dive: all 4 phases done, KILL-SHOT logged. No dispatch.
## 2026-05-15 09:07 — all-night stale
## 2026-05-15 09:13 — topology stale (program closed)
## 2026-05-15 09:21 — 3h MEP+DS-N tick: model program CLOSED (KILL-SHOT 05:30). Apps phase B done. PLAN CLOSED — 0 AMBITIOUS PASS.
## 2026-05-15 09:37 — all-night stale
## 2026-05-15 09:47 — :47 tick ACTIVE: scripts/z365_perVG1_bbo.py APU=46C sentinel=alive
## 2026-05-15 10:07 — all-night stale
## 2026-05-15 10:13 — topology stale (program closed)
## 2026-05-15 10:37 — all-night stale
## 2026-05-15 10:47 — :47 tick ACTIVE: scripts/z365_perVG1_bbo.py APU=46C sentinel=alive
## 2026-05-15 10:53 — 2h deep-dive: all done, no dispatch
## 2026-05-15 10:53 — master fix stale (program closed)
## 2026-05-15 11:07 — all-night stale
## 2026-05-15 11:13 — topology stale (program closed)
## 2026-05-15 11:37 — all-night stale
## 2026-05-15 11:47 — :47 tick ACTIVE: scripts/z365_perVG1_bbo.py APU=45C sentinel=alive
## 2026-05-15 12:07 — all-night stale
## 2026-05-15 12:13 — topology stale (program closed)
## 2026-05-15 12:21 — 3h MEP+DS-N tick: PLAN CLOSED (already logged 09:21)
## 2026-05-15 12:37 — all-night stale
## 2026-05-15 12:47 — :47 tick ACTIVE: scripts/z365_perVG1_bbo.py APU=46C sentinel=alive
## 2026-05-15 12:53 — 2h deep-dive stale
## 2026-05-15 12:55 — 6h oracle critique: SKIP (O70 12h already in flight, no new activity since)
## 2026-05-15 13:07 — all-night stale
## 2026-05-15 13:13 — topology stale
## 2026-05-15 13:37 — all-night stale
## 2026-05-15 13:47 — :47 tick ACTIVE: scripts/z365_perVG1_bbo.py APU=46C sentinel=alive
## 2026-05-15 14:07 — all-night stale
## 2026-05-15 14:13 — topology stale (program closed 05:30)

## 2026-05-15 14:20 — SEVEN GAPS PLAN DISPATCHED
Plan: research_plan/SEVEN_GAPS_PLAN_2026-05-15.md
3 subagents parallel:
- A: Track 1 (snapback S1-S4 incl. body-strap diagnostic) + Track 2 (transient data audit)
- B: Track 3 (input tuning) + Track 4 (temperature integration)
- C: Tracks 5/6/7 (noise + aging + wafer variation)
Pre-reg gates per sub-step. Honest verification each step.
Oracle O71 critique after all tracks complete.
NO-CHEAT. Running across ikaros + daedalus.

## 2026-05-15 15:10 — Subagent C: Tracks 5/6/7 COMPLETE (honest)
Scripts: scripts/seven_gaps_2026_05_15/{N1,N2,N3,AG1,AG2,AG3,WV1,WV3}*.py
- N1 1/f noise PASS: alpha={1.01, 1.01, 0.98} for K_f x{0.1,1,10} (all in [0.8,1.2])
- N2 RTN PASS 3/4: tau_emp/tau_target = {0.97, 1.00, 1.54, 7.30} (100ms tau too long vs T_S=5s)
- N3 Bayes real-noise FAIL: KL(one_f)=2.26, KL(mixed)=5.10 vs Gauss=0.10. Realistic substrate noise has long-range correlations -> MH chain biased; quality WORSE than i.i.d. Gaussian. Honest negative result.
- AG1 HCI FAIL gate: A_HCI=1e-15 with n=3 yields dVth<1e-19 mV even at 300uA/1e4s. Spec'd coeff is ~10 orders of magnitude too small for realistic silicon HCI.
- AG2 NBTI PASS: tau=1w sweep gives 1-week retention 36.8% (in [30%,70%] gate). tau=1h/1d collapse.
- AG3 retention DEGENERATE: KL invariant under amplitude scaling (standardization cancels amp_factor). Need bit-floor model not amp model; current AG2 form doesn't break RNG quality.
- WV1 audit: 149 CSVs, ONE unique device (2vHCa-2). SINGLE_DEVICE -> WV3 path.
- WV3 sigma sweep: NO breaking point up to 100mV (acc 0.93-0.94). UNEXPECTED: sigma=0 reservoir ALSO fails (acc=0.22, below chance) - homogeneous reservoir cannot classify; heterogeneity is required, not a defect.
Plots: results/{N1_1f_noise,N2_RTN,N3_bayes_realnoise,AG1_HCI,AG2_NBTI,AG3_retention,WV1_audit,WV3_sensitivity}/
Defensible: N1, N2, AG2, WV1. Open: N3 (correlation bias), AG1 (coeff calibration), AG3 (model form), WV3 (need real DS-N10 not proxy).

## 2026-05-15 — Track 1 (Snapback) S1+S3 + Track 2 (Transient) T1
- S1 (z377 body-strap diagnostic, VG1=0.6, VG2=0.20, R-46 best params): forcing Vb ∈ {0.0..0.8} V, 1D Newton on Vsint only, 11 conditions x 81 Vd-points, conv 78-81/81.
- KEY RESULT: at Vd=1.5V, Ids(Vb=0) = 3.0e-12 A; Ids(Vb=0.8) = 1.05e-6 A → **5.5 dec jump**. Measured = 2.07e-5 A (Vb=0.8 still 1.3 dec under, so true snapped basin sits at Vb ≈ 0.85-0.9 V).
- DECISION GATE: SOLVER_ISSUE (jump >= 2 dec threshold). BSIM4 Ids(Vbs) physics IS in the model; 2D Newton fails to find high-Vb basin. → S2 (continuation/homotopy) is the right next step. S4 (empirical fold) NOT triggered.
- Plot: results/z377_body_strap_diagnostic/{ids_vs_vd_perVb.png, summary.json}.
- S3 lit dive: 7 papers cataloged in research_plan/snapback_literature_2026-05-15.md. Key: Nanotech-2008 TechConnect paper validates quasi-2D body topology already in nsram_cell_2T.py (_residuals_quasi2d, Plan-A). LVTSCR paper supports iii_gain continuation for S2. No Mario-Lanza-specific 2T-NS-RAM snapback paper found (only KAUST faculty page).
- T1 transient audit: ZERO transient data exists. The `tdata` column in 33 IV-CSVs is slow-DC acquisition timestamp (290 ms/step), not a pulse trace. data/sebas_2026_05_02/ has only PNG + fit-JSON + p-diode card. Zoom screenshots are IV plots, no oscilloscope traces.
- T2 SKIPPED (no transient data); T3 triggered → HARD-BLOCK transient validation pending Sebas pulsed-Id(t) measurements. Inventory: research_plan/T1_transient_data_audit_2026-05-15.md.
- S4 NOT executed (gate=S2; physics is fine, no empirical fold needed).
- Next: implement S2 arc-length / iii_gain continuation in solve_2t_with_homotopy. Lit references TechConnect 2008 + LVTSCR for arc-length past limit-point.

## 2026-05-15 14:35 — MAJOR REVERSAL: S1 BODY-STRAP shows SOLVER not PHYSICS
At Vd=1.5V VG1=0.6: Ids(Vb=0)=3e-12 → Ids(Vb=0.8)=1.05e-6A = 5.5 DEC jump.
BSIM4 fold IS in pyport — 2D Newton just couldn't find high-Vb basin.
KILL-SHOT verdict reversed. Next: S2 continuation/homotopy solver (arc-length per LVTSCR lit).
T1: ZERO transient data exists. T3 hard-block confirmed — request from Sebas.
S3: 7 papers cited, BIMOS LIF + BSIM3+VBIC TLP ESD most relevant.

## 2026-05-15 14:30 — Tracks 3+4 (I1-I3, TM1-TM3) executed
I1 VG2 sensitivity at Vb=0.3: **FAIL** (frac_responsive=0%, autonomous-flat). Best responsive (Vb=0.7, VG1=0.51, VG2=0.585) sens=5.1% Inet=-71nA. plot: results/I1_vg2_sensitivity/sensitivity_heatmap.png
I2 DS-N10 sine tuned: TUNED collapses to chance (acc=18.9%, rate_ratio=0) BASELINE acc=97.8% rate_ratio=5.0× — **GATE FAIL** (50× needed; ~5×observed). plot: results/I2_dsn10_tuned/sine_acc.png
I3 Bayes RNG substrate: **PASS** KL=0.00339≤0.005 gate. Better than Voss(0.00808) and numpy(0.00511). plot: results/I3_bayes_tuned/posterior.png
TM1 BSIM4 T-coefs wired: **PASS**. G1 identity at 25°C bit-exact. G2 ΔVth(85°C)=-22.0mV vs predicted -21.3mV from kt1. Already-wired (dc.py:317, temp.py:222/229/252).
TM2 RMSE vs T sweep: **PASS** monotonic 25→100°C, no NaN. ΔRMSE/10°C≈0.125 dec. RMSE(25)=1.07→RMSE(100)=1.90. plot: results/TM2_t_sweep/rmse_vs_T.png
TM3 DS-N10 at 85°C: **PASS (ROBUST)** drop=1.1pp<5pp. acc(27°C)=97.8%, acc(85°C)=96.7%. plot: results/TM3_dsn10_hot/acc_vs_T.png
NET: 4/6 PASS (I3, TM1, TM2, TM3); 2/6 FAIL (I1@Vb=0.3 autonomous, I2 tuned-regime degenerate spike dynamics).

## 2026-05-15 14:55 — Track B DONE (4/6 PASS)
TM1/TM2/TM3 thermal all PASS — model is THERMALLY ROBUST (-1.1pp at 85°C).
I3 substrate-RNG PASS (KL=0.00339).
I1/I2 reveal STRUCTURAL trade-off: responsive regime ⊕ spike-generating regime (not both).
All 3 overnight tracks now complete. Ready for O71 synthesis.
## 2026-05-15 15:07 — all-night stale (Seven Gaps complete; awaiting O71 dispatch)
## 2026-05-15 15:47 — :47 tick ACTIVE: scripts/z365_perVG1_bbo.py APU=47C sentinel=alive
## 2026-05-15 15:53 — 2h deep-dive: all done, no dispatch
## 2026-05-15 15:55 — master fix tick: S2 arc-length continuation dispatched (z378). S1 already reversed KILL-SHOT. Awaiting S2 result.

## 2026-05-15 — z378 S2 arc-length solver: PRE-REGISTERED GATES
Script: scripts/z378_arc_length_solver.py; outputs results/z378_arc_length_solver/{summary.json, snapback_arc_length_compare.png, run.log}.
Inputs: R-46 per-VG1 best (z365 eval 94) across all 33 Sebas IV-CSVs (VG1∈{0.2,0.4,0.6}, VG2 sweep).
Compares baseline forward_2t (cold-start 2D Newton + warm-start cascade) vs forward_2t_arclength_grad (pseudo-arc-length continuation, already implemented in nsram/bsim4_port/arclength.py).
Gates (pre-registered, frozen before running):
- INFRA: zero NaN across 33 biases AND wall < 30 min
- DISCOVERY: cell-wide median RMSE < 0.85 dec AND max forward-jump in arc-result > 0.5 dec for at least one VG1
- AMBITIOUS: cell-wide median RMSE < 0.5 dec AND VG1=0.6 arc forward-jump > 1.5 dec
- KILL-SHOT: arc-length converges but no fold detected anywhere AND all jumps < 0.5 dec -> returns the question to physics (S4 empirical fold or different topology)

## 2026-05-15 16:05 — S2 KEY REFINEMENT
Arc-length converges but traces SAME low-Vb branch as cold-Newton (zero folds 0/3 biases).
High-Vb basin is DISCONNECTED, not fold-connected at R-46 params.
S1's "solver issue" interpretation refined: it's solver SELECTING wrong basin, not failing to navigate fold.
This IS bistability = the NS-RAM memory mechanism!
Next: (a) iii_gain homotopy or (b) two-branch search with hot Vb=0.8 init.

## 2026-05-15 16:20 — z378 full sweep DONE — KILL-SHOT confirmed
33/33 biases: arc-length traces identical curve to baseline, 0 folds anywhere.
Bit-identical Ids (±1e-5 dec). Continuation-along-Vd CAN'T reach disconnected root.

PHANTOM HYPOTHESIS (subagent (c)): S1 only solved 1D-on-Vsint with Vb=0.8 PINNED.
If R_B(Vb=0.8, ...) ≠ 0, then no real 2D root exists there → S1's "5.5 dec" was phantom.

S2a/S2b in flight will determine which:
- If S2b hot-init Newton ON FULL 2D converges and gives high Ids → bistable, S1 was real
- If S2b hot-init diverges or relaxes back to low Vb → S1 was phantom

Phase verdict pending S2a/S2b completion (~30-45 min).
## 2026-05-15 16:37 — all-night stale (S2a/S2b/S2c in flight, O71 dispatched)
## 2026-05-15 16:43 — topology tick: solver-strategy phase active (S2a/b/c + O71)

## 2026-05-15 16:50 — S2a HUGE FINDING: THREE basins, not two
S2a iii=10→0.90 homotopy: ALL 3 biases end at Vb=+1.18V, Ids=5.9μA flat.
This is a NON-PHYSICAL saturation root (well-diode runaway), NOT the fold.
S1's "Vb=0.8 → 5.5 dec" was a PHANTOM — Newton converged to this saturation attractor.

Real basin structure:
  - low-Vb ≈ 0 (sub-threshold, normal cold start)
  - INTERMEDIATE Vb ≈ 0.3-0.7V (real snapback fold — UNREACHED by any solver)
  - high-Vb ≈ +1.18V (non-physical saturation, ill-conditioned)

Implication: S2b needs Vb_init ∈ [0.4, 0.6] to find real fold. Vb_init=0.8 = phantom basin.
Vb-clamp at e.g. 1.0V may be needed to stop solver running to +1.18 saturation.

## 2026-05-15 17:00 — S2b KILL-SHOT — bistability NOT BSIM4-native
3/3 biases: hot Vb=0.8 init relaxes to identical low-Vb root (1.97/1.06/0.81 dec same).
Hot fold = 0.03-0.14 dec << 0.5 gate. 27-36 of 81-89 points marked "hot" but spurious.

Multi-strategy KILL-SHOT now:
  S2 arc-length: 0 folds (continuation can't reach disconnect)
  S2a iii homotopy: found non-physical saturation runaway (Vb=+1.18, Ids flat 5.9μA)
  S2b multi-init: hot init relaxes back to cold root

CONFIRMS: bistability is NOT BSIM4-native at R-46. Needs architectural mechanism
(explicit state variable / discrete latch). S1's "5.5 dec jump" definitively phantom.

S2c PTC still in flight — only remaining test. Awaiting O71 oracle synthesis.

## 2026-05-15 17:10 — S3 PHYSICS HYPOTHESIS BATTERY DISPATCHED
3 parallel subagents on ikaros (PTC z381 also running):
- S3-A: thermal static T-sweep + self-heating feedback (Rth)
- S3-B: vnwell + etab + max-aggressive combined sweep
- S3-C: lateral NPN reorientation (M1.D ↔ M2.S via shared body)
Each pre-reg gates: DISCOVERY if any config gives VG1=0.6 fold > 0.5 dec.
If ALL 3 KILL-SHOT → bistability not in BSIM4-pyport at all, regardless of topology tweak.

## 2026-05-15 17:18 — S3-A thermal KILL-SHOT + CRITICAL STRUCTURAL INSIGHT
T=150°C: max fold 0.000 dec. Self-heating Rth=500: dT=1mK (nW dissipation).
Both KILL-SHOT cleanly.

KEY INSIGHT: pyport curve is MONOTONE, data is S-shaped (2.2 dec fold).
Thermal cannot create fold from monotone — only shifts curve up.
Missing physics must REDUCE Ids with Vd (NDR — negative differential resistance).
Candidates: avalanche-quench, bistable latch, breakdown-then-pinch.

H1+H2+H5 ruled out (3/5). Awaiting H3 (vnwell) + H4 (etab) + S2c PTC + O71 oracle.
## 2026-05-15 17:21 — 3h MEP+DS-N tick: 3 active py jobs, APU=51C. S3 NDR investigation in flight (H3/H4/PTC/O71).

## 2026-05-15 17:28 — S3-B vnwell+etab KEY FINDING: mechanism exists, MIS-SCALED in VG1
Diag: etab IS wired (Vth shifts 265mV over Vbs[0,1]).
Combined clamp-off + etab=20: fold APPEARS at VG1=0.2 (2.76 dec) and VG1=0.4 (3.33 dec)!
BUT VG1=0.6 still flat (0.025 dec). INVERTED vs physics (hi VG1 should = stronger fold).

INTERPRETATION: pyport fold mechanism EXISTS but scales wrong way in VG1.
Something CLAMPS regenerative loop at high VG1 (where it should run hottest).
Missing: VG1-rising-regenerative-gain (BJT β·M product or VG1-gated avalanche multiplier).

H3+H4 individually FAIL. Combined gives partial fold at low-VG1 only.
4/5 hypotheses tested. Awaiting S2c PTC + O71.

## 2026-05-15 17:34 — 6h track audit
Phase A MEP: done. Phase B (DS-N): 5 surv + 5 retract done.
Phase C oracle: O67/O68/O69/O70 dispatched; O71 in flight.
Model program: 7 topology fixes failed, 4/5 NDR hypotheses ruled out (H1/H2/H5 plus partial H3/H4).
KEY finding: fold mechanism exists at low VG1 with clamp-off+etab=20, but INVERTED in VG1.
Current focus: S3-D (Vdsat/R_S/BETA0/VG1-dep) investigating root of VG1-inversion.
## 2026-05-15 17:37 — all-night stale
## 2026-05-15 17:47 — :47 tick: 3 python scripts active, APU=51C sentinel=alive

## 2026-05-15 18:00 — S3-D BREAKTHROUGH: VG1-inversion root cause = Vsint pump
At VG1=0.4/0.6, Vsint pumps to 0.19-0.23V → Vgs_M1 in subthreshold → Ids 100× lower.
NOT BSIM4 IIMOD (exp branch healthy at all VG1). NOT topology. Operating point pathology.
Test C beta0=10: VG1=0.6 fold=1.91 dec (MATCHES MEAS!) but breaks VG1=0.2 (branch flip).
Bistability hidden in (Vsint, Vb) space — Newton picks wrong branch at high VG1.

NEXT: bisect Bf × m2_body_gnd × pdiode at VG1=0.6 to identify what pumps Vsint.
Real fix: add Vsint clamp or wire body-pdiode properly to break source-follower lockup.
## 2026-05-15 18:07 — all-night stale
## 2026-05-15 18:13 — topology tick: S3-D root cause found (Vsint pump). Awaiting human decision on S3-E bisection.
## 2026-05-15 18:37 — all-night stale
## 2026-05-15 18:47 — :47 tick: 3 active py, APU=49C sentinel=alive
## 2026-05-15 18:53 — 2h deep-dive stale
## 2026-05-15 19:07 — all-night stale
## 2026-05-15 19:13 — topology tick: S3-D Vsint-pump root cause logged. PTC z381 still running biases 2-3.
## 2026-05-15 19:37 — all-night stale
## 2026-05-15 19:47 — :47 tick: 3 active py, APU=49C sentinel=alive

```
