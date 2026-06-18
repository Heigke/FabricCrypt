
**R-53 (Two-stage cascaded: Ids*=sqrt(M) + body += (sqrt(M)-1)|Ids|)**
  Same sweep. INFRA=PASS, DISCOVERY=PASS, AMBITIOUS=FAIL.
  Best: bv_ids=6V → cell-wide=0.96247 dec (Δ = −0.0025 dec vs baseline). Best of the three variants.
  per_VG1 at best: {0.20:1.7744, 0.40:1.1583, 0.60:0.8625}. fold06=1.6134.
  Improvement concentrated at VG1=0.40 (Δ=−0.003). VG1=0.60 essentially unchanged.

**R-54 (NPN base resistor RB_npn, bjt_emitter_to_gnd path)**
  Sweep ∈ {0,100,1k,10k,100k} Ω. PARTIAL RESULT — RB=0 completed (baseline-equivalent: cell=0.96501).
  RB=100 stalled in Newton solver >14 min wallclock with no per-curve progress; killed at 34min total.
  RB=1k/10k/100k skipped to honor 90-min budget — same code path would stall.
  Root cause: prelim BJT call gives _Ib_est on saturated NPN; resulting Vbe_eff = Vb − Ib*RB swings strongly negative, destabilising the outer (Vsint, Vb) Newton iteration.
  Fix would need: full inner base-node Newton solve (~M3c.3 local-base pattern), not 1-shot fixed-point.
  Gates: INFRA=FAIL (only RB=0 ran), DISCOVERY=trivially PASS at RB=0, AMBITIOUS=FAIL.

**Cross-variant verdict**: None of R-52/53/54 produced any meaningful fold improvement.
All three sit at the 0.965 dec floor that R-46 per-VG1 already achieved. The 2-3 dec
measured snapback fold is still NOT reproduced. fold06≈1.61 is the natural decade-spread
of the data trace (post-peak drop), not a model-side snapback fold; the model
monotonically rises with Vd in all variants.

**Conclusion**: The regenerative loop oracles flagged (Mii × floating-body NPN) needs the
FULL topology rebuild R-55a (D3 zener G1→B, BVPar = 3.5−1.5·V_G1, M3 BSS145 G2→B,
BJT VA=100, IS=5e-9, Bf=10000) per O69 + R-55 Zoom synthesis. Marginal topology
tweaks on top of current Mii=0 backbone cannot close the loop.

Plots:
  results/z375_topology_variants/R52_ids_multiplier/best_per_VG1.png
  results/z375_topology_variants/R53_two_stage/best_per_VG1.png
  results/z375_topology_variants/R54_npn_base_resistor/best_per_VG1.png (partial)
Summaries:
  results/z375_topology_variants/{R52,R53,R54}*/summary.json
Code changes: nsram/nsram/bsim4_port/nsram_cell_2T.py
  + cfg flags: use_ids_multiplier, bv_ids, n_ids, use_two_stage_avl, use_base_resistor, RB_npn
  + R-52/R-53 logic in _residuals after m2 eval (modifies m1["Ids"])
  + R-54 logic in BJT block (prelim compute_bjt for Ib estimate)

## 2026-05-15 05:25 — R-52/53/54 ALL FAILED
Best: R-53 0.96247 (Δ=−0.0025 from 0.965). No model fold reproduction.
R-54 stalled with RB≥100 (solver instability — needs M3c.3 local-base Newton).
Confirms O69: open Mii×floating-body loop can't be closed via marginal tweaks.
NEXT: R-55a full topology rebuild — D3 zener + M3 BSS145 + BJT VA=100/IS=5e-9/Bf=10000 + BVPar(VG1).
6 consecutive topology fixes have failed (R-43/45/47/49/52/53). R-55a is the
last designed approach before full retraction.

## 2026-05-15 05:30 — R-55a KILL-SHOT TRIGGERED
All 5 R-55a elements (D3 zener + BVPar(VG1) + nbvPar(VG1) + M3 BSS145 + BJT canonical) implemented and tested individually + ALL-ON.

KILL-SHOT gate: ALL-ON does NO better than baseline on snapback fold (jump≈0.03 dec, model still monotone-rising; measured fold 2.2-2.9 dec at knee).

INDIVIDUAL:  d3=0.03  bvpar=0.03  nbv=0.03  m3=0.00 (destabilized fit)
ALL-ON:      cell=4.083, jump=0.00 (M3 BSS145 dominates, destabilizes solver)

7 consecutive topology rewrites have failed (R-43/45/47/49/52/53/55a).
Per O69 agreement: this is the retract signal for the pyport→snapback program.

The pyport simulator cannot reproduce Sebas's 2-3 dec BJT snapback fold by
adding topology elements. The "0.965 dec fit" is sub-threshold curve-fitting.

Honest status: model fit program FAILED. Application program survives (5 niche claims).
## 2026-05-15 05:37 — all-night stale (R-55a KILL-SHOT triggered 05:30; awaiting human decision)
## 2026-05-15 05:47 — :47 tick ACTIVE: scripts/z365_perVG1_bbo.py APU=47C sentinel=alive
## 2026-05-15 06:07 — all-night stale (KILL-SHOT 05:30; O70 12h oracle dispatched 06:00)

## 2026-05-15 06:13 — topology tick
ALERT: R-55a KILL-SHOT gate crossed 05:30. Topology rebuild program ENDED per O69 pact.
7 consecutive failures (R-43/45/47/49/52/53/55a). No new R-phase dispatched.
Awaiting O70 12h oracle synthesis + human decision (retract report vs continue apps).
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
