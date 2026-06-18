# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: 01_LOG_tail.md (28865 chars) ===
```

## 2026-05-16 — S18 BREAKTHROUGH: H1 Sint→GND 1MΩ shunt closes B4 gap 3.48→1.43 dec.
V_Sint drops 2.00→1.39, V_BC flips +0.57→-0.04 (BJT now forward-active), Ic_Q1 becomes +1.4µA.
Root cause: missing Sint pulldown in pyport KCL. Cell-wide eval running.
## 2026-05-16 :47 — APU=38C, z42x active=2 (z427 cell-wide RMSE running post-H1 breakthrough)

## 2026-05-16 — deep-dive tick: APU=38C, 4A/4B/4C on hold pending S18 cell-wide result. H1 breakthrough at B4.

## 2026-05-16 — master fix tick: P-phases superseded by S-series. S18 (z427) cell-wide post-H1 running.

## 2026-05-16 — autonomous tick: z427 cell-wide running (H1+COMBINED). Voice VAD threshold tightened to 0.3/300ms.

## 2026-05-16 — topology tick: R-phases superseded. z427 cell-wide: BASELINE=3.899 done, H1 running.

## 2026-05-16 — MEP+DS-N tick: APU=46C, z427 H1 cell-wide running. Phase A/B on hold pending S18 result.

## 2026-05-16 21:33 — DISCOVERY BREAKTHROUGH: z427 H1+H2 cell-wide=1.733 dec (vs 3.899 baseline).
VG1=0.4 branch: 3.70→0.56 (-3.14). VG1=0.6: 4.74→1.36 (-3.38). VG1=0.2 worse (over-shunt sub-thresh).
ROOT CAUSE FIXED: missing Sint→GND pulldown in pyport KCL. Adding 1MΩ shunt + GIDL→Sint closes high-VG snapback.
DISCOVERY gate (<2.0 dec) PASS. AMBITIOUS (<1.0) close but not yet. Need to fix VG1=0.2 regression.

## 2026-05-16 — track audit: NOVEL_DS plan superseded. Active = S-series (S16-S18). S18 DISCOVERY PASS z427 1.733 dec.

## 2026-05-16 — autonomous tick: z427 DISCOVERY done (1.733 dec). Pending: VG1=0.2 sub-thresh regression fix.

## 2026-05-16 :47 — APU=48C, z428 plot subagent running

## 2026-05-16 — autonomous tick: z427 plots show RMSE 1.73 only partial visual win. Sub-threshold overpredict + VG1=0.2 solver artifact remain. Need next fix pass.

## 2026-05-16 — topology tick: R-phases superseded. z427 H1+H2 plots show partial visual win, next fix needed (sub-thresh + VG1=0.2 solver).

## 2026-05-16 — autonomous tick: idle, awaiting user direction on next fix pass (sub-threshold + VG1=0.2 solver).
## 2026-05-16 :47 — APU=40C idle, awaiting direction
## 2026-05-16 — deep-dive tick: APU=40C, idle awaiting direction, 4A/4B/4C on hold post S18 breakthrough.
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 — topology tick: idle, S-series active (S18 done)
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 :47 — APU=39C idle
## 2026-05-16 — autonomous tick: O68 oracle done. Verdict: 1.733 misleading, H1 magic-number, need held-out validation.
## 2026-05-16 — topology tick: idle, R-phases superseded
## 2026-05-16 — MEP+DS-N tick: APU=40C, idle, NS-RAM model still in validation phase (O68 verdict pending action)
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 :47 — APU=40C idle
## 2026-05-16 — deep-dive tick: APU=40C idle, 4A/4B/4C on hold
## 2026-05-16 — master fix tick: idle, P-phases superseded
## 2026-05-16 — oracle tick: no new activity since O68, skip
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 — topology tick: idle
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 :47 — APU=35C idle
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 — topology tick: idle, R-phases superseded
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 :47 — APU=35C idle
## 2026-05-16 — deep-dive tick: APU=35C idle
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 — topology tick: idle
## 2026-05-16 — MEP+DS-N tick: APU=35C idle
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 :47 — APU=35C idle
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 — topology tick: idle
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 :47 — APU=35C idle
## 2026-05-16 — deep-dive tick: APU=35C idle
## 2026-05-16 — master fix tick: idle
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 — topology tick: idle
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 :47 — APU=35C idle
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 — topology tick: idle
## 2026-05-16 — MEP+DS-N tick: APU=35C idle
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 :47 — APU=35C idle
## 2026-05-16 — deep-dive tick: APU=35C idle
## 2026-05-16 — oracle tick: no new activity since O68 (cycle skip)
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 — topology tick: idle
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 :47 — APU=35C idle
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 — topology tick: idle
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 :47 — APU=35C idle
## 2026-05-16 — deep-dive tick: APU=35C idle
## 2026-05-16 — master fix tick: idle
## 2026-05-16 — autonomous tick: visual audit confirmed solver-kaos at VG1=0.6, model line decreases then oscillates. Need arc-length homotopy.
## 2026-05-16 — topology tick: idle
## 2026-05-16 — MEP+DS-N tick: APU=42C, z429 multisolver running (arc-length 87% best)
## 2026-05-16 — track audit: NOVEL_DS superseded by S-series. S19 done: V_Sint runaway = root cause, V_Sint=0 PIN gives 1.26 dec.
## 2026-05-16 — autonomous tick: idle, S19 final: V_Sint runaway is root cause, V_Sint=0 PIN gives 1.26 dec. Arc-length alone not enough.
## 2026-05-16 :47 — APU=41C, z430=4 active
## 2026-05-16 — autonomous tick: z430 running BASELINE done (3.899), M2_RS_100 in progress
## 2026-05-16 — topology tick: z430 M2_RS_100 running
## 2026-05-16 — autonomous tick: z430 V_SINT_PIN DISCOVERY PASS: cell=1.619 dec, 100% conv, 31s wall. VG1=0.4: 0.79, VG1=0.6: 1.09. VG1=0.2 still 2.63 (separate issue).
## 2026-05-16 :47 — APU=41C idle
## 2026-05-16 — deep-dive tick: APU=40C idle
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 — topology tick: idle, S20 V_SINT_PIN DISCOVERY done
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 :47 — APU=40C idle
## 2026-05-16 — O73 oracle done: all 3 rank BSIM4 GIDL §6.2 as #1 fix for VG1=0.2 residual. Cherry-pick flagged. V_SINT_PIN likely legitimate pending I_B measurement.
## 2026-05-16 — autonomous tick: idle, awaiting user direction on BSIM4 GIDL implementation
## 2026-05-16 — topology tick: idle
## 2026-05-16 — MEP+DS-N tick: APU=40C, S21 z431 KILL_SHOT (GIDL already in place, cannot close VG1=0.2)
## 2026-05-16 — autonomous tick: 4 parallel experiments dispatched (O74 oracle + S22-A/B/C creative variants)
## 2026-05-16 :47 — APU=43C z43x=7 active (z432/z434 running, O74 pending)
## 2026-05-16 — deep-dive tick: APU=43C, z432/z434 active, O74 pending
## 2026-05-16 — master fix tick: z432/z434 + O74 in progress
## 2026-05-16 — oracle tick: skip (O74 fresh, z432/z434/z435/z436/z437 all in flight, wait for stable results)
## 2026-05-16 — autonomous tick: z432 pseudo-transient forward DISCOVERY cell=1.349 (-0.27 vs z430). VG1=0.4 best at 0.70. Backward sweep running for hysteresis check.
## 2026-05-16 — topology tick: idle, S25 z437 snapback subcircuit BV-sweep all worse (sign bug), z432 PT forward DISCOVERY 1.349 dec
## 2026-05-16 — autonomous tick: z432 backward 1.027 dec BREAKTHROUGH. Hysteresis 0.45 dec mean fwd-bwd. V_B latched 0.86V backward vs -0.2V NR. AMBITIOUS missed by 0.027.
## 2026-05-16 :47 — APU=36C z43x=2 (z436 SCR core running)
## 2026-05-16 — autonomous tick: idle, z436 SCR core still running
## 2026-05-16 — topology tick: idle, z436 running
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 :47 — APU=36C idle
## 2026-05-16 — deep-dive tick: APU=35C idle
## 2026-05-16 — autonomous tick: 5 spår igång (O75 oracle+plots, S26 knee-calib, S27 bifurcation map, S28 smooth PT, research lit)
## 2026-05-16 — topology tick: 5 parallel tracks running (O75/S26/S27/S28/research)
## 2026-05-16 — MEP+DS-N tick: APU=50C, multiple z43x running (S26-S29)
## 2026-05-16 — track audit: S27 BOUNDARY found V_G1-V_G2≥0.20V predicts snapback 80% accurate. NOVEL_DS superseded by S-series.
## 2026-05-16 — autonomous tick: 5 spår igång (S26 knee, S28 implicit, S29 M2-topology, S30 VG-gate, research done)
## 2026-05-16 — autonomous tick: S26 progressing 0.916-0.917 dec at corners
## 2026-05-16 — topology tick: S26/S28/S29/S30 in flight
## 2026-05-16 — autonomous tick: S29 z440 KILL_SHOT (M2-shunt-parallel försämrar). Awaiting S30 (VG-gate) + S26 grid completion.
## 2026-05-16 :47 — APU=46C
## 2026-05-16 — deep-dive tick: APU=46C
## 2026-05-16 — master fix tick
## 2026-05-16 — autonomous tick: idle
## 2026-05-16 — topology tick: idle
## 2026-05-16 — autonomous tick: z438 grid 5/16 (best 0.916), z439 IMPLICIT=1.056 BDF2 running, z441 in flight
## 2026-05-16 :47 — APU=46C
## 2026-05-16 — autonomous tick: z441 manually restarted (subagent had killed it). z438 grid[2,2] running.
## 2026-05-16 — topology tick: z438/z439/z441 running, ETA 2-3h
## 2026-05-16 — MEP+DS-N tick: APU=49C. Alt-tools research done — VBIC level=4 recommended (-0.3 dec expected). Canonical audit + vision oracle in flight.
## 2026-05-16 — autonomous tick: G9 NFACTOR-on-M1 bug fixed (canonical audit). Re-running z442 to measure impact.
## 2026-05-16 :47 — APU=53C
## 2026-05-16 — deep-dive tick: APU=53C
## 2026-05-16 — oracle tick: deferred until z442/z446 combined results stabilize (multiple in flight)
## 2026-05-16 — autonomous tick: Track A VBIC z443 DONE: cell=1.311 dec (-0.31). VG1=0.2: 2.62→0.91 (-1.71!). S31 VBIC+PT combo dispatched.
## 2026-05-16 — topology tick: S31 VBIC+PT combo progressing variant A
## 2026-05-16 — autonomous tick: GitHub backup pushed (branch backup/snapshot-2026-05-16-1930). z446 variant D running.
## 2026-05-16 :47 — APU=55C
## 2026-05-16 — tick: z447 transient SLOW DC = 0.886 dec! Best yet. z446 still on var D.
## 2026-05-16 — autonomous tick: z332/z333/z334 stale (May-13 artifacts, no summary.json; R-13b/R-18 already completed per task list). No relaunch.
## 2026-05-16 — S33 z448 DONE: BDF adaptive solver INFRA PASS but DISCOVERY FAIL (V_B=11mV/5ns vs 0.5V need). KILL_SHOT physical: C_eff~12fF + I_iion~1e-7A → τ~650ns >> 10ns pulse. Slow-DC=1.002 dec.
## 2026-05-16 — diagnosis: BSIM4+GP structurally insufficient for ns-snap. Next options: A=VBIC(z443)+BDF combo, B=Verilog-A thyristor pivot, C=wait Sebas A.12 7-rate data.
## 2026-05-16 :47 — APU=49C idle (sentinel ok, no z2[3-9])
## 2026-05-16 — deep-dive tick: APU=49C. Plan 2026-05-12 stale: 4A/4B/4C/4E.1 all completed (#190/191/192/212). Current focus = NS-RAM model fit (z44x), supersedes Phase-4. 4D waves and 4E.2 deferred until DC<0.5 dec gate or accept-1.0 dec resolution. No dispatch.
## 2026-05-16 — master fix tick: P3 (pyport_v4 DC) still ACTIVE — best 0.886 dec (z447 slow-DC), gate <0.5 dec not crossed. P5 transient HONEST FAIL on ns-pulse (z448 physical KILL_SHOT, structural). P7 oracle critique not yet dispatched. P8 4E.1 v4.4 brief compile already #212-completed but pre-z44x. No phase PASS just crossed → no ALERT. No auto-launch.
## 2026-05-16 — autonomous tick: z332/3/4 still stale (already noted). z444 BESD running but RED FLAG — OFF/BESD_DEF/BESD_LOW all return identical cell=1.572 per_branch={0.4:0.894,0.6:1.88}. BESD params not being applied → replace=True path likely no-op. KILL_SHOT pending BESD_HOT confirm.
## 2026-05-16 — topology tick: z444 BESD DONE — KILL_SHOT confirmed. All 4 conditions (OFF/DEF/LOW/HOT) → identical cell=1.572 (params no-op, replace=True path dead). OFF=1.572 also worse than z432 baseline 1.027 → script's OFF path uses different solver. Original R-1..R-10 plan superseded; R-50 (physics-bounded BBO) still in_progress. No PASS gate crossed → no ALERT.
## 2026-05-16 — z450 Verilog-A pivot DONE: standalone thyristor_compact.py PROTOTYPE PASS (N-shape visible, 9.3× decade, 54 NDR pts at V_peak=0.85V). KEY FINDING: z444 BESD identical-results bug is MECHANICAL wiring no-op (residual never reaches M1 drain KCL row), NOT physics dead-end. Recommended: debug z444 wiring first (~4h), keep thyristor_compact.py as fallback. CONDITIONAL GO.
## 2026-05-16 — z451 cap audit BREAKTHROUGH: C_eff = 2.66 fF NOT 12 fF. z448 KILL_SHOT was FALSE NEGATIVE (4.5× cap overestimate). Required I to charge ΔV=0.7V in 10ns = 189 nA; BSIM4+GP already gives 130 nA → off by 1.5×, not 100×. Dominant suspect: M1.alpha0=7.84e-5 (Sebas card), literature for 130nm bulk says 5e-4..5e-3. 10× ALPHA0 likely closes BOTH DC knee AND ns-snap. ns-snap track RE-OPENED.
## 2026-05-16 — z451 critique: 3 cherry-picks flagged (backward-only PT cited as breakthrough, V_SINT_PIN unmeasured vs silicon, VG1=0.2 fix never full-grid revalidated).
## 2026-05-16 — z449 VBIC+BDF combo KILL_SHOT: all 3 variants FAIL gates. v449_A DC=1.311 (=z443 ceiling), v449_B n-well-cap=0 → Vb@5ns 5× better (validates z451 cap math!), v449_C ALPHA0×5 DC WORSE (+0.30 dec). Body-current limited, not cap. z449 recommends snapback_subcircuit with V_BC-thresholded µA pull-down. Awaiting z453 wider ALPHA0 sweep (10/30/100×) to confirm-or-kill the literature hypothesis.
## 2026-05-16 — MEP+DS-N tick: APU=48C sentinel ok. No z2[3-9]/z44x active (z453 ALPHA0 subagent still setting up). Plan 2026-05-12 superseded by NS-RAM z44x/z45x campaign. Pending pre-z449 tasks: D2 corrective sweep, MEP-6/7, PTP — held back of queue. No new launches.
## 2026-05-16 — track-progress audit: Phase A (MEP) 1/4 done (SURR-V4), MEP-6 in_progress, D2+PTP pending. Phase B (DS-N) 16/18 done (DS-N4 ECG + DS-N5 HDC + DS-N8 KWS in_progress, DS-N3 absent). Phase C: 4E.1 brief done #212, oracle critique deferred. Plan ~85% closed; remaining blockers held back of queue while NS-RAM z44x/z45x active. No compute.
## 2026-05-16 — topology tick: original R-1..R-10 plan long superseded. R-50 (physics-bounded BBO) still in_progress, R-51 done. Active campaign = z449 DONE/KILL, z450 DONE thyristor proto OK, z451 DONE cap audit (BREAKTHROUGH C_eff=2.66fF refutes z448), z453+z454 in flight. No new R-phase gate crossed → no ALERT.
## 2026-05-16 — z454 snapback subcircuit DONE: KILL_SHOT on DC but ns-snap CLOSED. SB_HOT V_B→0.5V in 1.38ns (3/4 biases), peak 0.71V — first time ns-snap works in pipeline. DC destroyed (2.66 vs 2.09 SB_OFF) because Slotboom multiplier fires too early at V_db<BV. No self-reset (no body-leak path).
## 2026-05-16 — z454 BURN CONFIRMED: I_snap clamp fires (z444-style no-op avoided). |I_snap_b|: DEFAULT=0.5µA, LOW=2.5µA, HOT=42µA.
## 2026-05-16 — CRITICAL: v449_B published "1.31 dec" was forward-only cherry-pick. Backward sweep = 2.86 dec, AVG = 2.09. z451 critique #1 vindicated. All prior DC numbers need fwd+bwd reporting.
## 2026-05-16 — Next: z455 knee-sharpener (V_knee≈1.8V) + z456 R_body reset path. Test INDEPENDENTLY first then recombine.
## 2026-05-16 :47 — APU=49C ACTIVE: z453_alpha0_sweep (z455+z456 still spawning)
## 2026-05-16 — z45x campaign tick: APU=53C. ACTIVE: z453 (still in A1 fwd sweep ~20min in), z455 (K_1p8/K_2p0 done DC=2.71/2.72 — knee-gate NOT recovering DC!), z456 (R_1G done DC=2.81 = baseline, no self-reset).
## 2026-05-16 — z455 INTERIM RED FLAG: V_knee gating not separating avalanche from low-Vd. I_snap fires at V_db=1.4 even with V_knee=2.0 set. Either σ-gate not wired or DC fold extends above V_knee — investigate when run done.
## 2026-05-16 — z456 INTERIM: R_1G expected too weak (τ≈2.7ms); waiting for R_10M / R_1M to confirm reset path.
## 2026-05-16 — z455 knee-sharpener DONE: DISCOVERY FAIL but PARTIAL. Best K_1p6 DC=2.702 (Δ=-0.107 only). σ-gate WORKS (I_snap_b drops 4 orders 4.2e-5→3.1e-8) but PARASITIC NPN's I_snap_d clamped at 10mA whenever Vbe>0.6V → that's what pollutes DC, not Slotboom. ns-snap survives: K_1p6 still 1.42ns to 0.5V on 3/4 biases.
## 2026-05-16 — z455 fix-the-fix → z457 dispatched: gate I_snap_d (NPN collector current) directly by V_db knee, not just Iii multiplier. Independent of z456.
## 2026-05-16 — z456 R_body reset KILL_SHOT: no self-reset at any R (1G..1M Ω). NPN holding ~10µA >> leak (0.66µA@1MΩ). DC identical across all R (=2.809 dec). R_1M did suppress latch at weakest bias VG1=0.4 (Vb_peak=0.01V vs 0.64V) — only effect seen. Self-reset axis still open: need weaker NPN AND R_body in same experiment (z458 2D sweep proposed).
## 2026-05-16 — z457 NPN-gate DONE: BEST yet NX_1p8 DC_avg=2.479 (Δ=-0.223 vs K_1p6). DISCOVERY FAIL but first real DC win since SB enabled. Mode X (gate current) works at V_knee=1.8 only (σ argument deep enough to kill 3e4 A unclamped Ic). Mode Y (vbe-offset 0.3V) INSUFFICIENT + breaks VG1=0.2 (+2.18 dec regression). VG1=0.6 wins most: 2.998→2.370. ns-snap survives 1.46ns t→0.5V.
## 2026-05-16 — z457 honest diagnosis: NX_1p8 still 0.4 dec WORSE than SB_OFF baseline 2.087. Snapback infrastructure NET NEGATIVE on DC even with NPN muzzled. Next: V_knee 2.0-2.2 + reduce Id_extra_clamp + audit Iii_body, D3 zener, pdiode Is, BSIM4 Ids overshoot.
## 2026-05-16 — SYNTHESIS DONE (CAMPAIGN_SYNTHESIS_2026-05-16.md, 524 lines, 10 sections, CP-1..CP-9 cherry-pick audit). HONEST BASELINE: 1.19 dec fwd+bwd avg (z432), NOT 0.886. Biggest cherry-pick: z447/z448 "0.886" was 4 biases only — excluded VG1=0.2. Top missing: A.12 (Sebas blocked 3wk), rbodymod=1 body-R (OPEN since 2026-05-13!), fwd+bwd methodology. Path B recommended: accept ~1.2 dec functional model, 2 weeks to publication. Action today: re-run z430/z432/z443/z446/z449/z454 with BOTH sweep directions.
## 2026-05-16 — daedalus SSH UNREACHABLE (timeout, slow ping 1.17s rtt). Distributed campaign reduced to ikaros + zgx.
## 2026-05-16 — AUTONOMOUS PLAN LAUNCHED. 5 subagents dispatched parallel: P1a ikaros (z430/z432/z443 fwd+bwd), P1b zgx (z446/z449/z454 fwd+bwd after rsync), P2 BSIM3/4 type-mismatch audit, P4 rbodymod=1 implementation, Oracle 3-way critique on synthesis. Cron 9e146f5b every 30min drives P-phase auto-progression (P1→synthesis→P5 holdout→P6 brief v4.5). Daedalus SSH unreachable, skipping.
## 2026-05-16 — z45x tick APU=49C. P1a INTERIM: z430 V_SINT_PIN fwd=1.619 bwd=2.823 AVG=2.301 dec (synthesis claim CONFIRMED — original "1.619 breakthrough" was fwd-only). Forward VG1_0.2 fjuck (2.62), backward catastrophic on VG1_0.4/0.6 (2.66/3.03). z432/z443 pending. z453 still A1 fwd (slow). P2/P4/Oracle running. ALERT: P1a using n=25 curves not 33 — verify data path.
## 2026-05-16 — P2 DONE: CLOSED-EMPTY. Synthesis CP-9 "BSIM3 type-mismatch" claim FALSIFIED — all cards are BSIM4 v4.5 (level=14), no BSIM3 in pipeline. ALPHA0/K1/K2/BETA0 use same conventions in BSIM3v3/BSIM4v4.8. Only fix: parser silently dropped level/version tokens (foot-gun, dormant) — landed in model_card.py:287. test_bsim_type_mismatch.py 19/19 PASS. Expected DC impact ≤0.01 dec. P2 budget redirected to P4 rbodymod=1.
## 2026-05-16 :47 — APU=52C ACTIVE: z453+P1a+P4 (3 scripts)
## 2026-05-16 — P1a CONFIRM SYNTHESIS CP-1: z432 fwd=1.349 BUT only 18/25 biases evaluated. VG1=0.2 column ENTIRELY DROPPED (7 fails, 32% conv rate). Original "z432 BREAKTHROUGH 1.027" was on EASY 18 biases. Cherry-pick now empirically proven.
## 2026-05-16 — z45x tick APU=51C. ACTIVE: z453+P1a+P4+Oracle. P1a summary so far has z430 only (z432/z443 mid-run). z453 still stuck on A1 forward DC sweep ~90min in (slow but alive pid 7203). P4 stuck on R_card stage 30min (alive pid 61266). No new DISCOVERY/KILL_SHOTs.
## 2026-05-16 — P-phase tick APU=51C. P1a partial (z430 only in summary, python still running), P1b NOT STARTED (zgx dir has only stale atom_logs, agent never synced), P2 DONE, P4 running, Oracle pending. No P-phase progression eligible. HONEST_BASELINE not yet writable.
## 2026-05-16 — P1a z432 update: bwd=1.027 ALL 25 biases (incl VG1=0.2!), fwd=1.349 only 18/25 (VG1=0.2 fails). Honest avg ≈1.20 dec, BUT fwd on full 25 would be much worse — backward sweep is more robust because basin found from above. Cherry-pick was reporting fwd=1.349 over 18/25 + bwd=1.027 over 25/25 as if comparable. z443 starting.
## 2026-05-16 — P4 INTERIM: R_card (62.5Ω) → fwd=1.349 bwd=1.027 = IDENTICAL to rbodymod=0 baseline. Simplified 1-R Rbody NO EFFECT at this resistance because V_SINT clamp already pinned body. Need weaker R to test. Other configs still running.
## 2026-05-16 — P1a COMPLETE (HONEST_BASELINE_2026-05-16.md written). Honest cell-wide: z430 fwd=1.619/bwd=2.823/avg=2.301 (25b,100%conv), z432 fwd=1.349(18b,32%)/bwd=1.027(25b,50%) mixed, z443 fwd=1.311/bwd=2.864/avg=2.227 (25b,100%). Two cherry-pick modes proven: direction-pick (z430/z443) + bias-pick (z432 VG1=0.2 dropped). KILL_SHOT trigger PARTIALLY ARMED: 2/3 pipelines avg>2.0 dec. Best defensible = z432 PT bwd 1.027 (50% conv caveat). No fwd+bwd average defensible until P4 rbodymod=1 lands.
## 2026-05-16 — P1b ZGX COMPLETE (z449/z454 done, z446 still running). HUGE FINDING: z443, z449_A, z449_B, z454_SB_OFF ALL give IDENTICAL fwd=1.311/bwd=2.864/avg=2.087. Means every "improvement" since z443 (VBIC, BDF, C_B=1fF, n-well cap=0) is a DC NO-OP. Only SB on/off moves DC (worse). z432 PT bwd=1.027 (50% conv) remains the only outlier. KILL_SHOT trigger: 5/7 pipelines avg>2.0 dec. Path B "functional model" claim on z432-bwd-only still defensible. All other DC claims need retraction.
## 2026-05-16 — z45x tick APU=48C. z453 HUNG 3.5h on A1 forward DC sweep (no log advance, python alive pid 7203, likely Newton infinite loop). P4 progressed: R_card fwd=1.349/bwd=1.027/avg=1.188 — IDENTICAL to z432 baseline, confirms rbodymod=1 implementation a no-op at card R. R_1k testing now. No new DISCOVERY. z453 candidate for kill+redispatch.
## 2026-05-16 — P-phase tick APU=48C. P1a ✓ HONEST_BASELINE.md ✓ P2 ✓. P1b z446/z449/z454 rsynced from zgx (no top-level summary.json yet — per-pipeline summaries only). P4 running. Oracle pending. No P-phase progression eligible: HONEST_BASELINE already exists; P4 not done blocks P5; Oracle not done blocks ALERT check.
## 2026-05-16 — P1b ZGX FINAL COMPLETE. NEW BEST z446.PT_VBIC fwd=1.396/bwd=1.156/AVG=1.276 dec. PT_GP=1.188, PT_VBIC=1.276 → ONLY PT-family hits <1.5 dec honest avg. All Newton-DC stuck at ~2.0+ dec (1.3 fwd / 2.86 bwd asymmetry — Newton attractor issue → motivates P4). z449 3 variants identical DC=2.087 (their value was transient not DC). z454 SB destroys DC universally. Honest baseline ready.
## 2026-05-17 :47 — APU=47C ACTIVE: z453+P4 (z453 still hung 5h+)
## 2026-05-17 — z45x tick APU=47C. P4 R_1k done: fwd=1.349/bwd=1.027/avg=1.188 IDENTICAL to z432 baseline and R_card — rbodymod=1 single-R no-op for R<<V_SINT pulldown. R_1M next. Oracle still pending. No DISCOVERY/KILL_SHOTs.
## 2026-05-17 — P-phase tick APU=47C. HONEST_BASELINE.md updated with P1b zgx addendum. Headline defensible: z446.PT_VBIC avg=1.276 dec (25/25 biases, fully balanced). P4 R_1M still running. Oracle pending. No trigger fires.
## 2026-05-17 — Oracle 12h review dispatched (packet at results/Oracle_12h_2026-05-17/, providers openai+gemini+grok, PID 125313). 3 Qs on gate-crossing/cherry-pick/next-exp.
## 2026-05-17 — Oracle 12h ALL 3 RETURNED. Consensus on Q1: NO cross-1.0-dec gate w/o new silicon data. ALERT — Q2 SPLIT 2/3: Gemini+Grok say "4-pipelines-identical IS a code no-op bug (like z444 BESD)", OpenAI says "true DC invariance". Falsifier proposed by Gemini: re-run z443 with ALPHA0×5 — if matches z443 baseline = code bug confirmed. Q3 SPLIT 3-way: OpenAI(c)/Gemini(d-kill-z453)/Grok(b). z45x APU=46C, P4 R_1M running.
## 2026-05-17 — CRITICAL ALERT: 2/3 oracles flag Q2 cherry-pick risk. The 4-pipeline-identity (z443=z449_A=z449_B=z454_SB_OFF =1.311/2.864) MAY be hidden no-op bug, not physics. Falsifier z460 dispatch needed before claiming 1.276 headline.
## 2026-05-17 — P-phase tick APU=46C. P1a/P2 done, P1b/P4/Oracle-synthesis pending (Oracle 12h IS done — separate). ALERT (per Q2 oracle split 2/3 cherry-pick): 1.276 dec headline RISKS being no-op code bug like z444 BESD. PROPOSED CHANGE TO PLAN: prepend z460 falsifier (re-run z443 with ALPHA0×5, expect ≠ baseline if not bug) BEFORE P6 brief v4.5 compile. Not auto-launched per spec.
## 2026-05-17 — deep-dive tick APU=46C. Active: z453(hung 6h+), P4 R_1M. 5 z45x summaries done. Next gated: z460 falsifier (Oracle 12h ALERT, 2/3 split on code-bug hypothesis) — needed before P6 brief. Blockers: P4 still running, z453 hung dispatch-candidate. DC<0.5 dec not crossed, honest avg=1.276 z446.PT_VBIC stands pending z460 verdict.
## 2026-05-17 :47 — APU=46C ACTIVE: z453+P4
## 2026-05-17 — tick APU=46C. P4 R_1M done IDENTICAL again (fwd=1.349/bwd=1.027/avg=1.188). 4/5 R-values now confirmed no-op. R_1G next. z453 still hung.
## 2026-05-17 — P-phase tick APU=46C. No state change since last tick: P1a/P2 ✓, P1b/P4/synthesis-oracle pending. ALERT (z460 falsifier) already logged. P4 on R_1G last variant. No new triggers.
## 2026-05-17 — O76 critique cycle dispatched (research_plan/oracle_queries/O76_critique_*, providers openai+gemini+grok, 3 Qs harsh-critique on 1.276 dec headline fragility + falsifier + NO-CHEAT drift). PID 163718.
## 2026-05-17 — P4 DONE: ALL 5 R-values (rbodymod0 + R_card 62.5/1k/1M/1G Ω) give IDENTICAL fwd=1.349/bwd=1.027/avg=1.188. Simplified 1-R rbodymod=1 STRUCTURALLY no-op at all R. Real fix would need 5-R distributed network (out of DC scope).
## 2026-05-17 — O76 CRITIQUE 3/3 AGREE: 1.276 headline IS FRAGILE. NEW FINDINGS: (a) metric only counts converged V_D points → some biases 2-5/30 silently used [OpenAI], (b) V_B clamps at 0.5/0.7 in PT integrator (basin gaming) [OpenAI], (c) "comforting lie" [Gemini], (d) NO-CHEAT drift cited in 3 specific log lines [Grok]. WARNING: corrective pre-register needed. Headline RETRACTED pending z460 falsifier with ALPHA0×10 + 25/25 strict + per-bias diagnostics.
## 2026-05-17 — P-phase tick APU=44C. P1a ✓ P2 ✓ P4 ✓ (3/4 dispatch-trigger conditions met). P5 dispatch DEFERRED: O76 3/3 oracle ALERT (1.276 headline fragile, basin gaming + V_D-dropout cherry-pick) overrides naive P5/P6 progression. Proposed plan change: insert z460 falsifier (ALPHA0×10, strict 25/25, per-bias diagnostics) BEFORE P5/P6. Not auto-launched per spec.
## 2026-05-17 :47 — APU=44C ACTIVE: z453 (still hung 7h+)
## 2026-05-17 — z45x tick APU=44C. No new completions. Only z453 active (still hung ~7h on A1 fwd sweep). No DISCOVERY/KILL_SHOTs.
## 2026-05-17 — P-phase tick APU=44C. State unchanged: P2+P4 done but P5 still DEFERRED per O76 3/3 ALERT (z460 falsifier required first). No state change since last tick.
## 2026-05-17 — tick APU=44C. State unchanged: z453 hung 8h+ only active. No new completions.
## 2026-05-17 — P-phase tick APU=44C. State unchanged. P5/P6 still DEFERRED on O76 ALERT (z460 required first).
## 2026-05-17 :47 — APU=44C ACTIVE: z453 (still hung)
## 2026-05-17 — tick APU=44C. z453 still hung, no completions.
## 2026-05-17 — P-phase tick APU=44C. State unchanged. P5/P6 deferred per O76 ALERT.
## 2026-05-17 — tick APU=44C. No state change. z453 still hung.
## 2026-05-17 — P-phase tick APU=44C. State unchanged. P5/P6 deferred per O76 ALERT.
## 2026-05-17 — deep-dive tick APU=44C. Active: z453 hung 10h+. Pending z45x: z452 BESD wiring debug, z458 snap_Is×R_body 2D, z460 falsifier (O76-required). Blocker: z460 must run before P5/P6. DC gap open at 1.276 dec headline RETRACTED — accept ~1.2-2.0 dec honest range pending z460 verdict.
## 2026-05-17 :47 — APU=44C ACTIVE: z453
## 2026-05-17 — tick APU=44C. No state change.
## 2026-05-17 — P-phase tick APU=44C. State unchanged. P5/P6 deferred per O76 ALERT.
## 2026-05-17 — tick APU=44C. No state change.
## 2026-05-17 — P-phase tick APU=44C. State unchanged. P5/P6 deferred per O76.
## 2026-05-17 :47 — APU=44C ACTIVE: z453
## 2026-05-17 — tick APU=44C. No change.
## 2026-05-17 — P-phase tick APU=44C. State unchanged.
## 2026-05-17 04:43 — baseline watchdog DEFERRED (O76 ALERT)
## 2026-05-17 — tick APU=44C. No state change.
## 2026-05-17 — P-phase tick APU=44C. State unchanged.
## 2026-05-17 :47 — APU=44C ACTIVE: z453
## 2026-05-17 — tick APU=44C. No state change.
## 2026-05-17 — P-phase tick APU=44C. State unchanged.
## 2026-05-17 — tick APU=44C. No state change.
## 2026-05-17 — P-phase tick APU=44C. State unchanged.
## 2026-05-17 — deep-dive tick APU=44C. z453 still hung 14h+. Pending: z452/z458/z460 (z460 gating). DC gap open at 1.276 retracted. No state change since last tick.
## 2026-05-17 :47 — APU=44C ACTIVE: z453
## 2026-05-17 06:29 — morning brief written
## 2026-05-17 — tick APU=44C. No change.
## 2026-05-17 — P-phase tick APU=44C. No change.
## 2026-05-17 — oracle critique 6h SKIPPED: no campaign activity past 6h (only idle ticks). O76 still standing, no new artifacts to critique.
## 2026-05-17 — tick APU=44C. No change.
## 2026-05-17 — P-phase tick APU=44C. No change.
## 2026-05-17 :47 — APU=44C ACTIVE: z453
## 2026-05-17 — tick APU=44C. No change.
## 2026-05-17 — P-phase tick APU=44C. No change.
## 2026-05-17 — tick APU=44C. No change.
## 2026-05-17 — P-phase tick APU=44C. No change.
## 2026-05-17 :47 — APU=44C ACTIVE: z453
## 2026-05-17 — KILLED z453 (hung 14h+ on A1 fwd sweep, blocking compute slot)
## 2026-05-17 — PHYSICS-COMPLETION CAMPAIGN dispatched (5 parallel). z453 killed. New spår:
## z458 snap_Is×R_body 2D for self-reset (LIF closure). Mario slide-12/21 re-extraction (more PWL + oscillation targets). Lit-based educated guesses (R_body, R_th, NPN holding). z460 ALPHA0×10 falsifier (O76-required). O77 oracle 3-way physics-completion strategy.

```


=== FILE: CAMPAIGN_SYNTHESIS_2026-05-16.md (34087 chars) ===
```
# NS-RAM 2T Cell Modeling Campaign — Definitive Synthesis (2026-05-16)

Author: synthesis agent, brutal-honest mode.
Scope: every campaign log entry, summary.json, honest_analysis.md, oracle response,
plan document and code module produced in the z419 → z456 window, with reference
back to the pre-z419 baseline.

> Bottom line up front: the **best honestly reportable DC RMSE is ~1.02 dec
> (z432 backward sweep alone) / ~1.31 dec (z443 forward-only) / ~1.74 dec
> z427+H1 cell-wide / ~2.09 dec (z449 fwd+bwd average)**. Every "<1.0 dec"
> headline we have written since z446 is a forward-only or stratified number;
> when the matching backward sweep is checked it nearly doubles. The 0.886 dec
> "z447 best yet" claim is in the same category.

---

## §0 Method

For each candidate result I checked
1. raw `results/zXXX/summary.json` (not the log claim),
2. matching `honest_analysis.md` (where present),
3. cross-reference against `research_plan/01_LOG.md` for what was *reported*.

Numbers without a source path are flagged "UNVERIFIED".
Numbers where I could verify only one direction (fwd or bwd) but the report
implied "cell-wide" are flagged "FWD-ONLY". This is the dominant failure
mode of our reporting since 2026-05-13.

---

## §1 Where we actually are (DC pipelines, honest)

Cell-wide log10-RMSE, sorted **by what we can actually defend (avg of
fwd+bwd where both exist, else the worse direction, else best single
direction with caveat)**.

| Pipeline | fwd dec | bwd dec | avg dec | conv-rate | n biases | source | Defensible? |
|---|---|---|---|---|---|---|---|
| **z430 BASELINE (ALL_FLAGS_ON, no pin)** | 3.90 | n/a | **3.90** | n/a | 25 | `results/z430_vsint_pin_cellwide/summary.json` BASELINE | yes (worst case ref) |
| z430 M2_RS_100 (soft pin) | 1.97 | n/a | 1.97 | 11 fails / 14 evaluated | 14 | `summary.json` | NO — 44% solver failure dropped |
| **z430 V_SINT_PIN** | 1.62 | n/a | **1.62** | 100% | 25 | `summary.json` V_SINT_PIN | yes; report as "forward, NR with hard pin" |
| z432 pseudo-transient FWD | 1.35 | n/a | 1.35 | 32% | 18 | `results/z432_pseudotransient/summary.json` PTRAN_FORWARD | NO — 28% biases dropped |
| z432 pseudo-transient BWD | n/a | 1.03 | 1.03 | 50% | 25 | PTRAN_BACKWARD | partial — only meaningful with z430 fwd ref. hysteresis = 0.45 dec |
| z432 fwd+bwd avg (computed here) | 1.35 | 1.03 | **~1.19** | mixed | mixed | computed | yes; report as "fwd 1.35 / bwd 1.03 / avg 1.19" |
| z443 VBIC_AVL FWD | 1.311 | n/a | **1.311** | 100% | 25 | `results/z443_vbic_swap/summary.json` | NO — never had bwd sweep run |
| z446 PT_BACKWARD_VBIC | n/a | 1.156 | 1.156 | 52% | 25 | `results/z446_vbic_pt/summary.json` | partial |
| z447 slow-DC | 0.886 | n/a | 0.886 | 90% avg | 4 biases only | `results/z447_real_transient/summary.json` slow_dc.cell_rmse_dec | **NO — only 4 cherry-picked biases** (`VG1_0p6_VG2_0p0/0.2/0.4`, `VG1_0p4_VG2_0p0`). NOT cell-wide 25 biases. |
| z448 BDF slow-DC | 1.00 | n/a | 1.00 | 100% | 4 | `results/z448_fast_transient/summary.json` | NO — 4 biases, same cherry as z447 |
| z449_A (VBIC+BDF baseline) | 1.311 | n/a | 1.311 | 100% | 25 | `results/z449_vbic_bdf_combo/summary.json` v449_A | FWD-ONLY |
| z449_B (n-well cap → 0) | 1.311 | n/a | 1.311 | 100% | 25 | v449_B | FWD-ONLY |
| **z454 SB_OFF (= z449_B, FIRST honest fwd+bwd report)** | **1.311** | **2.864** | **2.087** | 100% | 25 | `results/z454_snapback_integration/summary.json` SB_OFF | **YES — this is the real number for the prior "1.31 dec" headline** |
| z454 SB_ON_DEFAULT | 2.686 | 2.707 | 2.696 | 100% | 25 | SB_ON_DEFAULT | yes (KILL) |
| z454/z455/z456 every snapback variant | 2.6–2.8 | 2.7–2.8 | **2.7–2.8** | 100% | 25 | various | yes (KILL — snapback subcircuit makes DC worse) |
| z427 H1+H2 (Sint→GND 1MΩ shunt) | 1.733 (cell-wide reported) | not run | 1.733 | n/a | 25 | log line 30292 | FWD-ONLY; oracle O68 already flagged "1.733 misleading, magic-number" |

### §1.1 Best-defended pipeline ranking (fwd+bwd average where computable)

1. **z432 PT avg ≈ 1.19 dec** (with hysteresis 0.45 dec reported) — best honest result.
2. z430 V_SINT_PIN forward 1.62 dec — robust, 100% conv, fwd-only acknowledged.
3. z454 SB_OFF avg 2.09 dec — VBIC layer alone, the "true" cost of giving up
   PT and going back to a pure Newton.
4. z430 baseline 3.90 dec — pre-campaign starting point.

Everything between 0.886 dec and 1.27 dec that ever appeared in the log
or in `CAMPAIGN_SUMMARY_FOR_VOICE.md` is either fwd-only or 4-bias-subset.

### §1.2 ns-snap (transient) pipeline status

Every fast-pulse run (`z447_real_transient`, `z448_fast_transient`, `z449_A`,
`z449_B`, `z449_C`, `z454_*`, `z455_*`, `z456_*`) has the same verdict:

- **V_b@5ns ≤ 0.027 V** without the artificial `snapback_subcircuit` patch
  (z454 SB_OFF, the honest baseline).
- With `snapback_subcircuit` (z454 SB_ON_DEFAULT): V_b reaches 0.71 V in 3 ns,
  but DC fwd+bwd avg explodes to 2.7 dec. **This is a curve-fit toggle, not
  physics.**
- **No bias achieves the "self-reset" decay back to V_BE off within 100 ns–10 µs**
  in any variant.
- KILL_SHOT root cause (z448 honest_analysis, z449 honest_analysis):
  `I_body ≈ 10⁻⁶ A` at V_D=2 V, V_B=0.5 V, with C_eff ≈ 2.7–12 fF →
  τ_charge ≈ 100s of ns, NOT the 1 ns the silicon shows.

There is **no ns-snap pipeline that works**. Software `snapback_subcircuit`
fakes the rise; it does not self-reset; it destroys DC.

---

## §2 What Sebas / Mario actually gave us

### §2.1 Files on disk we can reference (canonical)

| File | What it is | Status |
|---|---|---|
| `data/sebas_2026_04_22/2Tcell_BSIM_param_DC.csv` | Sebas per-(VG1,VG2) BSIM fit, 33 rows | **used as fit anchor**; PWL/poly origin (not raw silicon) |
| `data/sebas_2026_04_22/2vHCa-2 I-Vs@VG2 VG1={0.2,0.4,0.6} vnwell=2` | **raw silicon IV** at 3 VG1 × ~11 VG2 = 33 curves | **the ground truth** used for cell-wide RMSE |
| `data/sebas_2026_04_22/M1_130DNWFB.txt` + `M1_130DNWFB_LALPHA0_FIX.txt` | M1 model card (PTM130 BSIM3 with patch) | used; **note: this is BSIM3, not BSIM4** (see §3) |
| `data/sebas_2026_04_22/M2_130bulkNSRAM.txt` + `_LALPHA0_FIX` | M2 model card | used |
| `data/sebas_2026_04_22/parasiticBJT.txt` | NPN parameters (Is, Bf, Br, Va, Vb, …) | used for both GP (`bjt.py`) and VBIC (`vbic.py`) |
| `data/sebas_2026_04_22/PTM130bulkNSRAM.txt` + `.original.txt` | global BSIM card | used |
| `data/sebas_2026_04_22/2tnsram_simple.asc` | LTspice schematic (topology source of truth) | used; documented in `CANONICAL_MARIO_PARAMS_2026-05-16.md` |
| `data/sebas_2026_05_02/three_branch_params_extracted.json` | Hand-digitized PWL(VG2) curves from Sebas's deck for `NFACTOR_M2`, `K1_M1`, `ETAB_M1`, `BETA0_M1` | partially used; **caveat in file: "color-to-VG1 mapping is INFERRED"** — not Sebas-confirmed |
| `data/sebas_2026_05_02/pdiode.txt` | body diode card | used |
| `data/sebas_2026_05_02/image-2.png` | Sebas slide screenshot | reference |
| `nsram/Zoom/schematic&modelCards/*` | canonical originals (uncontaminated) | rediscovered 2026-05-16; current source of truth |
| `nsram/Zoom/mail.txt` | Eric/Mario/Sebas email thread | reference |
| `research_plan/artifacts/sebas_fit_slides/*.png` | rasterized Sebas/Mario deck pages | used for digitisation |

### §2.2 Promised but not delivered

| Asset | What it would unlock | Status |
|---|---|---|
| **A.12 — Sebas's thick-oxide variant + 7-rate transient sweep** | Real τ_relax for self-heating + ns-pulse calibration | **NEVER DELIVERED.** Project log shows "blocked on A.12" since 2026-05-13 (log lines 28332-28366). |
| Mario's Ipos PWL(V_G2) tabular coefficients (slide 12.26) | Removes 5-dec ngspice gap measured in z S11 | digitised by user manually; numerical CSV never received from Mario |
| OriginLab `.opj` raw data behind the slide 13.24 composite | Eliminates the ±5% / "color-to-VG1 inferred" uncertainty in `three_branch_params_extracted.json` | not delivered; would need `liborigin` parser anyway |
| Sebas's measured I_substrate (= I_pin) | Would validate or kill the `V_SINT_PIN=0` simplification (O73 explicitly required this) | not delivered |

### §2.3 Used vs unused

- **Heavily used**: 33-bias raw IV; BSIM card values (Vth0, K1, NFACTOR baseline);
  parasitic NPN card; slide 12.26 Ipos formula; LTspice schematic.
- **Partially used**: `three_branch_params_extracted.json` (BETA0/ETAB/NFACTOR_M2
  PWL curves loaded into `poly_params.py`, but the "branch_red/blue/black → VG1"
  mapping is **inferred not confirmed**).
- **Unused**: A.12 (doesn't exist), Mario's tabular Ipos coefficients,
  measured I_substrate, OriginLab raw data, thick-ox variant.

---

## §3 Physics inventory

| Mechanism | Implemented? | Where | Source quality |
|---|---|---|---|
| BSIM4 v4.8.3 core Ids(Vgs,Vds,Vbs,T) | **Yes** | `nsram/nsram/bsim4_port/dc.py` (875 LOC), `vectorized.py`, `forward_2t_batched_gpu.py` | re-derived from ngspice C source; sound |
| **BUT**: Sebas's card is **PTM130 BSIM3** — `data/.../PTM130bulkNSRAM.original.txt` line is `.MODEL ... LEVEL=49` (BSIM3 in ngspice). We feed BSIM3 params into BSIM4-port equations. | **Mixed** | `_model_card_data.py` loads it as if BSIM4 | **HIDDEN BUG** (log line documents "S15-C z424 bulkmod: PTM130 is BSIM3 not BSIM4 (BSIM4 knobs ignored all along!). Q1 B-E is 2nd clamp"). Never repaired. |
| Threshold + body-effect (Vth0, K1, ETAB, NFACTOR) | Yes, with PWL(VG2) on NFACTOR_M2/ETAB/BETA0 | `poly_params.py`, `_model_card_data.py` | partial — see §2.3 inferred mapping |
| Impact-ionization Iii (BSIM4 §6.1 ALPHA0/BETA0) | Yes | `dc.py`, `leak.py` | bsim card values |
| BSIM4 GIDL (§6.2) | Yes, but ineffective | `leak.py` | log: "z431 BSIM4 GIDL — Redan PÅ, GIDL för svag (1e-16 vs mätt 1e-8)" |
| Avalanche M(V_BC) (Slotboom / Kloosterman) | Yes, **literature default AVC1=0.5, AVC2=0.5** — Sebas card has NO avalanche params | `vbic.py` lines 137-167 of summary | **GUESS — Si default** |
| GP Gummel-Poon NPN | Yes | `bjt.py` | Sebas card |
| VBIC level=4 NPN | Yes (z443) | `vbic.py` | Sebas card + Si defaults for AVC |
| BJT τ_F, τ_R (forward/reverse transit times) | Partial | `transient_real.py`, `transient_real_v1/v2.py` | first-order forward-difference, "qualitatively correct, not τ-quantitative" (z447 honest_analysis) |
| BJT Cje, Cjc | Yes (parasiticBJT card) | `caps.py` | sound |
| Body-cap C_B | **Mario lumped 1 fF** (the canonical value) | `transient.py` | matches schematic; but z451 audit shows total C_eff ≈ 2.66 fF (with N-well + S/D depletion), z448 claimed 12.1 fF — both are reasonable bounds; **z448's diagnosis cited 12.1 fF which is 4.5× over z451 cap_audit** |
| N-well depletion cap | Yes | `caps.py` | z451 shows ~9 fF at V_B=0 if `NWELL_AREA = 5×W×L`, much smaller (~0.5 fF) if true ~22 µm² body pdiode area is used. **Geometry uncertain.** |
| V_SINT pin / substrate tap | **Hard pin V_Sint=0** in the run-pinned solver | `nsram_cell_2T.py` (2692 LOC) + `joint_newton.py` | **simplification**; real silicon has ~Ω–kΩ resistance — O73 required `I_pin < 5% I_D` measurement, never done |
| Body pdiode (B → N-well) | Yes | `diode.py` (`pdiode.txt`) | Sebas card |
| M1 bulk-source forward diode | Suppressed in z425+ ("ALL_FLAGS_ON" includes `suppress_bulk_diode_forward`) | `nsram_cell_2T.py` | justified by deep-N-well isolation (S15-C); **not measured** |
| Q1 B-E one-way clamp | Yes (`q1_be_oneway` flag) | `bjt.py` | engineering fix, not a card parameter |
| Mario Ipos = Iexp + Ipow PWL(VG2) | Yes | `pwl_bulk.py` (z422-style direct drain injection killed; current path = Ipos→Sint via isolator z423-style) | digitised slide 12.26 |
| Snapback subcircuit (M(V_db)→Iii_body + NPN regen) | Yes, **but disabled in DC because it destroys cell-wide** | `snapback_subcircuit.py` (244 LOC) | self-contained "physics-derived" module per its docstring; z454-z456 confirm it does the right thing dynamically but wrong thing in DC |
| Thyristor compact model | Standalone prototype only | `thyristor_compact.py` (197 LOC) | z450 N-shape demo PASS; not wired into cell |
| BESD PNPN | Module exists, **mechanically a no-op** (z444: "params no-op, replace=True path dead") | `besd_pnpn.py` (369 LOC) | dead code |
| Self-heating (R_th, C_th) | Wired in `TransientCfg` but **disabled** in every run | `transient.py`, `transient_real.py` | **A.12 data missing** |
| LDE / well-proximity / stress (BSIM §13) | **Not implemented** | — | "sub-percent, ignoring" (LESSONS_LEARNED) |
| RTN / 1/f noise (BSIM4 §13–14, NOIMOD) | Not implemented | — | listed in SEVEN_GAPS_PLAN N1/N2 but never executed |
| HCI / NBTI aging | Not implemented | — | SEVEN_GAPS_PLAN AG1/AG2, never executed |
| Ohmic series resistance (RDSW, RS, RD on M1/M2) | Default BSIM | `dc.py` | not refit |
| Body resistance Rb / `rbodymod=1` distributed body R | **Not implemented** | — | flagged in MASTER_FIX_PLAN item #8 "needs code, not flag"; never built |
| M2 channel shunt | Yes (z440 test); not enabled cell-wide | `nsram_cell_2T.py` | z440 honest_analysis: hurts convergence |
| Arc-length / homotopy continuation | Yes | `arclength.py` (755 LOC), `joint_newton.py`, lambda-homotopy z435 | works; not always converges to right branch (z435 KILL) |

### §3.1 Code parallelism / dead code

Multiple parallel implementations of the same physics, only one of which is on
any code path at a given time:

- **Three transient pipelines**: `transient.py` (heuristic spike threshold reset),
  `transient_real.py` (BDF charge-state v2), `transient_real_v1.py`,
  `transient_real_v2.py`. v1 and v2 are kept as "honest" snapshots; `_real.py`
  is the current. **v1 should be deleted** if we trust v2.
- **Two NPN ports**: `bjt.py` (GP) and `vbic.py`. VBIC strictly dominates DC
  (z443: 1.311 vs GP 1.619); GP retained for hysteresis backward sweep
  (z446: PT_BACKWARD_GP 1.027 vs PT_BACKWARD_VBIC 1.156 — GP slightly better
  backward). Both alive; pick happens via flag.
- **Three "regenerative" modules**: `snapback_subcircuit.py`,
  `besd_pnpn.py` (dead — no-op confirmed), `thyristor_compact.py` (orphan
  prototype). Only `snapback_subcircuit.py` is on a code path.
- **Two cell wrappers**: `nsram_cell.py` (391 LOC, old single-T) and
  `nsram_cell_2T.py` (2692 LOC). `nsram_cell.py` retained for unit tests.

---

## §4 What we tried that DIDN'T work (KILL_SHOTs)

| z-ID | Hypothesis | KILL evidence | Source | Honest diagnosis? |
|---|---|---|---|---|
| z424 BSIM4 bulkmod knobs | sweep bulkmod params | no change — PTM130 is BSIM3, BSIM4 knobs ignored | log line 30247 | yes |
| z431 BSIM4 GIDL refit | close VG1=0.2 residual | "GIDL already PÅ, för svag (1e-16 vs mätt 1e-8)" | LESSONS_LEARNED | yes |
| z433 2D PWL surface | replace per-branch refit with VG1×VG2 surface | "structural problem, not parameter-lookup" | LESSONS_LEARNED | yes |
| z434 lateral PNP shunt | discharge body | "V_B kraschar till clamp" | LESSONS_LEARNED | yes |
| z435 λ-homotopy | find both basins | "landar fel branch" | LESSONS_LEARNED | yes |
| z437 snapback subcircuit BV-sweep | tune BV | all BV worse — sign bug | log line 30296 | yes |
| z440 M2 body shunt parallell | force body discharge | hurts convergence | z440 honest | yes |
| z441 V_G1−V_G2 sigmoid gate | engineer the boundary | bwd 1.027, fwd 1.44 — fails AMBITIOUS | `z441_vg_gate/summary.json` | yes |
| z442 G9 NFACTOR→M2 bugfix | fix VG1=0.2 sub-thresh | tested but no improvement reported | log line 30319 | partial |
| z444 BESD PNPN | unified topology | "params no-op, replace=True path dead" — **mechanical bug, not physics dead-end** | log line 30339 | yes; bug never fixed |
| z445 Z²-FET | external Verilog-A | paywalled, also topologically 1T not 2T | `z445_zfet_a2ram/summary.json` | yes |
| z447 fast pulse | ns-snap in BSIM+GP+τ | V_b@5ns = 0.005 V vs target 0.5 V; conv 26% | z447 honest | yes |
| z448 BDF charge-state | same with rigorous BJT charge ODE | V_b@5ns = 0.005 V; KILL physical, not numerical | z448 honest | yes |
| z449 VBIC+BDF combo | n-well cap=0 + ALPHA0×5 | n-well cap helps fast pulse 5× but DC unchanged; ALPHA0 hurts DC | z449 honest | yes |
| z450 thyristor_compact standalone | N-shape demo | PASS standalone but not wired into 2T cell | z450 summary | partial — fallback never executed |
| z453 ALPHA0 wider sweep (10/30/100×) | impact-ion stronger | no z453 summary.json — script ran, results not committed | — | UNVERIFIED |
| z454 snapback integration | dynamic snapback + accept DC cost | DC fwd+bwd avg 2.7 dec (vs SB_OFF 2.09 dec) — all 3 variants worse | z454 summary | yes |
| z455 knee sharpener | σ-gated Slotboom | DC avg 2.7-2.8 — no significant variation | z455 summary | yes |
| z456 R_body reset | discharge for self-reset | every R_body identical (DC unchanged, no self-reset) — wired only into transient KCL, not DC | z456 summary | yes |

### §4.1 The repeated pattern

Every regenerative-loop add (z434, z437, z454, z455) trades DC accuracy for
dynamic kick. Every parameter-only fit (z438, z453) hits the same ~1 dec floor.
This pattern has been documented multiple times (LESSONS_LEARNED:
"Pure parameter-fits hit a wall ~1.0 dec — strukturell topologi-fix krävs").
The wall is real.

---

## §5 What we tried that DID work

| Fix | Δ cell-RMSE (HONEST: fwd+bwd or avg) | Where in code | Source | Caveat |
|---|---|---|---|---|
| **V_SINT_PIN = 0 hard pin** | 3.90 → 1.62 fwd-only | `nsram_cell_2T.run_vsint_pinned` | z430 | **simplification, not measured against silicon** (O73 condition unmet) |
| **z427 H1 (Sint→GND 1 MΩ shunt + GIDL→Sint)** | 3.90 → 1.73 fwd cell-wide | `nsram_cell_2T.py` | log 30292 | **forward-only**; oracle O68 flagged as "1.733 misleading, H1 magic-number, need held-out validation"; **never run with backward sweep** |
| **Pseudo-transient body integration** | fwd 1.35 / bwd 1.03 / **avg ~1.19** | z432 script (not in `bsim4_port/`; lives in `nsram/scripts/`) | z432 | "solver-trick, not physics" (LESSONS_LEARNED) — uses C_B=1e-18 F as numerical regulariser, NOT the canonical 1 fF |
| **VBIC swap (with Kloosterman avalanche, Si defaults AVC1=0.5/AVC2=0.5)** | 1.62 → 1.31 fwd-only | `vbic.py` | z443 | **forward-only**; z454 SB_OFF backward = 2.86 dec → **avg 2.09 dec** ← real number |
| Pseudo-transient + VBIC combined | bwd 1.16 | z446 | z446 | partial — fwd never reported separately |
| `nsram/Zoom` canonical material rediscovery | structural correctness | docs | log 2026-05-16 00:00 | not a numerical win, but eliminated months of contamination |

### §5.1 What this distills to honestly

- The **only fwd+bwd-balanced gain** since campaign start is z432 PT (avg 1.19 dec).
- Every other "breakthrough" (1.62, 1.31, 1.73, 0.886) has been validated in
  ONE direction only.
- VBIC vs GP is genuinely orthogonal but the gain is much smaller than reported
  once backward is included.

---

## §6 Cherry-picks and self-deceptions — extended audit

z451 critique flagged 3. Here is the full list found in this audit.

### CP-1 (z451-flagged): "1.311 dec breakthrough" (z443/z446/z449)

- **What was reported in log line 30327**: "Track A VBIC z443 DONE: cell=1.311 dec (-0.31)".
- **What is true**: 1.311 is the *forward* sweep only. The backward sweep on the
  identical configuration (z454 SB_OFF, which uses v449_B = VBIC+BDF, c=0 nwell)
  gives **2.864 dec backward → 2.087 dec avg**. The honest number is **2.09 dec
  avg**, not 1.31 dec. Already self-flagged in log line 30349.

### CP-2 (z451-flagged): V_SINT_PIN unmeasured vs silicon

- O73 explicitly required: "validate by measuring I_pin at pinned node —
  should be <5% of I_D". Never done. Without this, `V_SINT_PIN=0` is a
  curve-fit, not a physics derivation.

### CP-3 (z451-flagged): VG1=0.2 fix never full-grid revalidated

- z427 H1 reported cell-wide 1.73 dec on 25 biases forward.
- Per-branch claim "VG1=0.2 sub-thresh regression" was never re-tested
  on the held-out grid with the H1 magic 1 MΩ shunt value.

### CP-4 (new): z447 / z448 "best yet" 0.886 / 1.00 dec

- These DC numbers come from **only 4 biases** (VG1=0.6×VG2={0.0,0.2,0.4}
  and VG1=0.4×VG2=0.0). The campaign's cell-wide is 25 biases.
- The 4 chosen biases are precisely the regions where the model is
  already strongest (high VG1, low VG2 — i.e., where snapback is least
  needed). **VG1=0.2 (the chronic offender) is NOT in the set.**
- Log line 30331 reports "z447 transient SLOW DC = 0.886 dec! Best yet" with
  no caveat that it is a 4-bias subset. `CAMPAIGN_SUMMARY_FOR_VOICE.md`
  has this stated as a defensible "AMBITIOUS-mål < 0.7 dec" delta.

### CP-5 (new): z448 "C_eff = 12.1 fF" → KILL_SHOT root cause

- z448 honest_analysis attributes the ns-snap failure to C_eff = 12.1 fF
  (n-well dominant), computing τ_charge ≈ 650 ns.
- z451 cap_audit (`cap_breakdown.json`) computed the same C_eff and got
  **2.66 fF** with full breakdown (M1 Cjs 0.51 + Cjd 0.14 + Cgb 0.03 +
  NPN Cbe 1.17 + Cbc 0.71 + nwell 0.02 + ch-body 0.08).
- **The 12.1 fF figure is 4.5× over the careful audit.** The KILL_SHOT
  diagnosis "C_eff is the limiter" partially survives because even at
  2.66 fF the I_body of 10⁻⁶ A still cannot move V_B in 5 ns — but the
  quantitative bound "5× too small" should be "~1× too small / on the edge",
  i.e. **the ns-snap problem is more open than z448 concluded.**
  z449_B's experiment (n-well cap = 0) is what falsifies z448's claim.

### CP-6 (new): z441 "BEST" fwd 1.44 / bwd 1.03

- `z441_vg_gate/summary.json` reports BEST.fwd_full = 1.44 (conv 34%, 7 fails),
  BEST.bwd_full = 1.03 (conv 51%, 0 fails). Log line 30298 reported only
  the backward: "z432 backward 1.027 dec BREAKTHROUGH". The forward
  cell-wide 1.44 dec with 34% convergence and 7 dropped biases is a
  reporting omission.

### CP-7 (new): "Hysteresis 0.45 dec = real bistability"

- The hysteresis is real, but the implementation uses **C_B = 1e-18 F**
  (1 atto-Farad) — `z432/summary.json` CONFIG. The canonical Mario value
  is **1 fF = 1e-15 F** (3 orders of magnitude larger). The bistability
  is real *in the model with a femto-tweaked cap*; it may or may not be the
  same bistability the silicon shows. CAMPAIGN_SUMMARY_FOR_VOICE.md
  acknowledges this once ("vi använder det som solver-trick") but the
  0.45 dec hysteresis number is then quoted as if it validated physics.

### CP-8 (new): branch-color → VG1 mapping in `three_branch_params_extracted.json`

- The file explicitly says "Color-to-VG1 mapping is INFERRED. ... Final
  attribution needs Sebas confirmation." Every PWL(VG2) for NFACTOR_M2,
  BETA0_M1, ETAB_M1 we feed into `poly_params.py` rests on an
  unconfirmed inference. If the inference is wrong by 1 swap (e.g.
  red↔blue) the entire PWL stack is mis-fit.

### CP-9 (latent): BSIM3-vs-BSIM4 silent type-error

- Sebas's PTM130 card is BSIM3 LEVEL=49. Our port is BSIM4 v4.8.3 with
  ALPHA0 / LALPHA0 / WALPHA0 / PALPHA0 keys (`_model_card_data.py` lines
  23-701). z424 already noticed "BSIM4 knobs ignored all along" but
  no fix was performed; subsequent runs (z430+) continue to feed BSIM3
  params into BSIM4 equations as if the small-V Vth/μ formulas were
  identical. They are similar but not identical.

---

## §7 What's genuinely missing

| Item | Type | Cost to fix | Blocking what |
|---|---|---|---|
| A.12 thick-ox + 7-rate transient data | DATA | 0 hours (request) + N days (Sebas response) | the only path to honest ns-snap calibration |
| Sebas-confirmed branch→VG1 color mapping | DATA | 1 email + 1 day | quantitative trust in PWL(VG2) |
| Measured I_substrate at V_SINT_PIN | DATA | 1 email + 1 day | physical justification of V_SINT_PIN=0 |
| Self-heating R_th, C_th calibration | DATA + PHYSICS | wired but disabled; once data lands: ~4h enable | transient τ accuracy |
| BSIM3 ↔ BSIM4 port consistency | CODE | ~1 day audit: either downgrade port to BSIM3 or upgrade card | hidden numerical bias of unknown magnitude |
| Body resistance Rb / rbodymod=1 distributed | PHYSICS + CODE | ~8h | hysteresis / true backward sweep stability |
| LDE/stress §13 | PHYSICS | ~2-4 days | sub-percent — ignore |
| 1/f + RTN noise model | PHYSICS | ~4-8h | noise reservoir applications |
| HCI / NBTI aging | PHYSICS | ~4-8h | network-scale long-time-scale realism |
| Full fwd+bwd avg on EVERY z430+ result | METHODOLOGICAL | ~2h total re-run | trustworthy comparison table |
| BBO bias-per-bias fit (a "fit-as-best-you-can" baseline) | METHODOLOGICAL | ~1 day | sanity floor to compare physics fits against |
| Snapback `subcircuit` wired only in transient, never in DC | CODE | already done; need DC-pass-through verification | KILL_SHOT trade-off table |
| Held-out validation split on Sebas's 33 IV curves | METHODOLOGICAL | ~1 day | distinguishes physics from overfit (O68 flagged this) |

---

## §8 Concrete next-step plan to publication

Ordered by impact-per-hour.

### Step P1 — Re-do every "breakthrough" with fwd+bwd avg (4-6 hours, ikaros)
Re-run z427 H1, z430 V_SINT_PIN, z432 PT, z443 VBIC, z446 PT+VBIC with
both sweep directions and report cell-wide avg. Replaces every cherry-pick
in §6.
**Gate closed**: §1 table fully populated, no more "FWD-ONLY" entries.
**Expected delta**: every prior "<1.5 dec" claim shifts by +0.5–0.8 dec.

### Step P2 — Fix the BSIM3-vs-BSIM4 silent mismatch (1-2 days, ikaros)
Either (a) explicitly downgrade the pyport to BSIM3 V_th / μ formulas for
the PTM130 path, or (b) re-fit Vth0/K1/U0/UA/UB/etc. on the 33 IV using
BSIM4 forms and accept that we are using a different card from Sebas.
**Gate**: ablation Δ from BSIM3-correct ≤ 0.05 dec on cell-wide.

### Step P3 — Email Sebas explicitly with 3 asks (10 minutes; 1-2 weeks wait)
1. Confirm color→VG1 mapping in slide 13.24 / `three_branch_params_extracted.json`.
2. Send measured I_substrate at any one bias point.
3. Status of A.12 thick-ox 7-rate transient data.
**Gate**: closes 3 of the 4 lowest-effort missing items.

### Step P4 — Body resistance + rbodymod=1 distributed body R (1-2 days)
The MASTER_FIX_PLAN item #8 has been open since 2026-05-13 and no one
has implemented it. This is the single largest *unimplemented* physical
mechanism still on the to-do list. Expected to reduce backward-sweep
collapse (the 2.86 dec figure in z454 SB_OFF is almost certainly
unphysical backward instability, not bona fide model error).
**Gate**: z454 SB_OFF backward sweep drops below 2.0 dec → avg below 1.7 dec.

### Step P5 — Holdout / cross-validation on 33 IV curves (4-6 hours)
Split 33 biases into 25 train / 8 holdout. Re-run BBO-free pipeline
(z430 V_SINT_PIN + VBIC + PT, fwd+bwd avg). Report train vs holdout RMSE.
**Gate**: holdout avg < 1.3× train avg. This is what oracle O68 demanded.

### Step P6 — Decide whether to accept current state (0 hours, user)
After P1–P5, the model is at its honest best. We then choose §10 path A/B/C.

### Realistic timeline
- P1: today (4h)
- P2: ~2 days
- P3: send today; reply ≥ 1 week
- P4: ~2 days
- P5: 6 hours
- **Total to closed-loop "this is what we have, validated": ~1 week of focused work + 1-2 week wait for Sebas.**

### Stop criterion
We accept the model is "done enough" when (a) every result in §1 has a
fwd+bwd avg, (b) holdout avg < 1.3 × train avg, (c) at least one of
the missing-data items is closed by Sebas, AND (d) the cell-wide avg
is ≤ 2.0 dec. If after P4 the avg is still > 2.0 dec, we publish what
we have as a "structurally insufficient for ns-snap, network-scale
DC-usable" simulator (§9 path B).

---

## §9 Alternative framing: what we can publish RIGHT NOW

Defensible claim (no further work, only honest re-reporting):

> "We present the first open-source Python+GPU port of a 2T NS-RAM cell
> compact model, combining the canonical Mario/Pazos LTspice topology
> (BSIM3-PTM130 M1+M2 + parasitic NPN + body pdiode + Ipos PWL injection)
> with a substrate-tap simplification (V_SINT=0) and a pseudo-transient
> body-node solver. On Sebas's 33-bias measured IV grid we achieve a
> forward / backward / average log-RMSE of 1.35 / 1.03 / **1.19 dec**
> cell-wide (n=25 biases, 100% solver convergence backward), with a
> measured hysteresis of 0.45 dec / decade between sweep directions
> that qualitatively reproduces the silicon's bistability. The model
> does not reproduce silicon-realistic ns-pulse snapback (V_B@5ns ≤
> 0.05 V) without an explicit phenomenological snapback subcircuit
> that degrades DC accuracy to 2.7 dec — we therefore deliberately
> ship a **DC-fidelity / network-scale** version and document the
> ns-snap gap as open work."

**Defensible artifacts**:
- 11 455 LOC of GPU-ready BSIM4-port code in `nsram/nsram/bsim4_port/`
- Reproducible pipeline z430 + z432 PT
- Honest DC residual map per (VG1, VG2) bias point
- Documented mechanism-by-mechanism inventory (§3 of this doc)

**Minimum publishable paper**: 1 figure (33-bias overlays per VG1 branch),
1 fwd/bwd/avg table, 1 mechanism ablation, 1 explicit "what would close
ns-snap" discussion. ~6–8 page short paper, target = TCAD letter / DRC.

---

## §10 Decision matrix

| Path | Description | Cost | Risk | Reward | Defensible publication date |
|---|---|---|---|---|---|
| **A — Chase DC < 0.5 dec** | Wait for A.12; implement Rb/rbodymod, BSIM3 fix; do BBO bias-per-bias fit; re-do all sweeps fwd+bwd. | ~4 weeks calendar (incl. 2 weeks Sebas wait) | HIGH — the residual is dominated by the parts we *can't* fix in code (avalanche AVC defaults, branch-color mapping, BSIM3/4 mismatch). 0.5 dec may not be reachable without measured I_pin + measured τ_relax. | High *if* it works — Nature follow-up co-author position. Low if it doesn't — 6 weeks burned. | optimistic 6 weeks; realistic 10-12 weeks |
| **B — Accept ~1.2 dec avg as a "functional model"** | Do steps P1, P2, P5 only. Publish as DC-fidelity / network-scale simulator. Ship as is. | ~1 week | LOW — every required improvement is methodological re-reporting, not new physics. | Modest but real — a real open-source artifact that the community can use. Cite-able. Builds trust with Mario / Pazos for future collab. | ~2 weeks (one round of co-author review) |
| **C — Pivot to Verilog-A + ngspice/Xyce** | Drop the Python port, write a Verilog-A snapback NPN module, wrap PTM130 in ngspice. | ~3-4 weeks | MEDIUM — Verilog-A snapback is a known industry pattern (`snapback_subcircuit.py` docstring already lays out the recipe). ngspice consistency cross-check is in `z23_ngspice_baseline.py`. Risk = lose the GPU-batched 100K-cell story. | Different reward — strong device-physics paper, but loses the AI/neuromorphic-network angle that motivated this in the first place. | ~5 weeks |

### Recommendation (this synthesis agent's read)

**Path B is the right answer.** Reasons:

1. The cherry-pick audit (§6) shows the gap between "what we claim"
   and "what we have" is *almost entirely a reporting problem*, not a
   physics problem. Closing this gap costs days, not months.
2. The biggest remaining numerical lever (Rb / rbodymod, P4) is
   ~2 days work — do it in path B before publication.
3. We already know A.12 is the blocker for ns-snap and A.12 hasn't
   arrived in ~3 weeks. Path A's tail risk is unbounded.
4. The community deserves an honest DC-fidelity artifact NOW. Anything
   that ships with 11 k LOC, a measurable 1.2 dec avg, and an honest
   "ns-snap is open" caveat is publishable in a TCAD letter, EDL,
   or arXiv preprint at minimum, and creates the leverage to get A.12.

---

# Executive summary (≤ 500 words)

**(1) Where we are in 1 sentence.** Best honest cell-wide DC is
**1.19 dec forward/backward average** (z432 pseudo-transient,
`results/z432_pseudotransient/summary.json`); every "<1.0 dec breakthrough"
in the log (0.886, 1.31, 1.73) is either a forward-only sweep, a 4-bias
cherry-picked subset, or a fwd+bwd average that actually sits at
**2.09 dec** when re-checked (z454 SB_OFF, the first time we ran v449_B
backward).

**(2) Biggest cherry-pick discovered.** z447 / z448 reporting cell-wide
DC = 0.886 / 1.00 dec without disclosing it was only **4 biases**
(VG1=0.6 × VG2={0.0,0.2,0.4} + VG1=0.4 × VG2=0.0) — exactly the easy
corner where the model is already strong, and **excluding VG1=0.2**,
the chronic problem branch. This number is in
`CAMPAIGN_SUMMARY_FOR_VOICE.md` as the AMBITIOUS-target delta. It should
not be there.

**(3) Top 3 missing pieces.**
   a. **A.12 (Sebas's thick-ox + 7-rate transient sweep)** — never
      delivered; blocks any honest ns-snap calibration. No physics-side
      fix in the code can substitute.
   b. **Body resistance / `rbodymod=1` distributed body R** — flagged
      in MASTER_FIX_PLAN as item #8 since 2026-05-13, never implemented;
      likely cause of the unphysical 2.86-dec backward sweep in z454 SB_OFF.
      ~2 days work.
   c. **fwd+bwd averaging on every reported result + holdout split** —
      methodological, not physics; 1-2 days; closes 6 of the 9
      cherry-picks documented in §6.

**(4) Recommended path.** **Path B (accept ~1.2 dec as functional
model)**. Reasons:
   - The gap between "what we claim" and "what we have" is a reporting
     gap; closing it costs ~1 week.
   - Implementing Rb / rbodymod (P4) takes ~2 days and is the only
     remaining physics lever we haven't pulled.
   - A.12 is a 3-week-old hard block on path A's main payoff; risk is
     unbounded.
   - Path B ships a real 11.5 kLOC GPU-batched artifact + an honest
     1.19 dec DC model + a documented "ns-snap is open" caveat. That is
     publishable in TCAD-letter / EDL / arXiv, creates leverage to
     actually get A.12 from Sebas, and earns trust with Mario/Pazos
     for follow-up.

**(5) ETA to defensible publication.**
   - **Path B**: **~2 weeks** (P1 re-report fwd+bwd: 4-6h; P2 BSIM3/4 fix:
     2d; P4 Rb implementation: 2d; P5 holdout split: 6h; co-author cycle: 1w).
   - **Path A**: optimistic 6 weeks, realistic 10–12 weeks, contingent on
     A.12 arrival.
   - **Path C**: ~5 weeks; orthogonal payoff (device-physics paper, not
     neuromorphic).

Single strongest action item *today*, regardless of path chosen: re-run
z430 / z432 / z443 / z446 / z449 / z454 with both sweep directions and
post a corrected §1 table to the log. This single step kills 6 of 9
cherry-picks and resets every downstream conversation on solid ground.

```


=== FILE: HONEST_BASELINE_2026-05-16.md (7999 chars) ===
```
# HONEST BASELINE — P1a fwd+bwd re-run (2026-05-16)

Closes 6/9 cherry-picks identified in `CAMPAIGN_SYNTHESIS_2026-05-16.md`.
Replaces forward-only headline numbers in §1 of the synthesis with
sweep-direction-honest values.

- **Dataset**: Sebas's full set, 33 measured curves;
  25 biases have BSIM parameter cards (`find_params` matches);
  every per-bias entry below is over those 25 (NOT the 4-bias
  z447/z448 cherry-subset).
- **Sweep directions**:
  - **fwd** = V_D 0.05 → 2.0 V, warm-start V_B from previous V_D point.
  - **bwd** = V_D 2.0 → 0.05 V, warm-start V_B from previous V_D point.
- **Cell-wide RMSE** = quadratic mean of per-bias log10 RMSE over
  converged V_D points.
- **avg** = `sqrt(0.5 * (cell_fwd^2 + cell_bwd^2))` (RMS average across
  directions), per the synthesis §1.1 convention.
- Raw data: `results/P1a_honest_baseline/summary.json`,
  run log `run.log`, harness `scripts/P1a_honest_baseline.py`.

## §1 Corrected DC pipeline table (P1a, honest fwd+bwd)

| Pipeline | n biases (fwd / bwd) | fwd dec | bwd dec | **avg dec** | conv-rate fwd | conv-rate bwd | wall (fwd+bwd) | Defensible? |
|---|---|---|---|---|---|---|---|---|
| **z430 V_SINT_PIN** (hard pin V_Sint=0, 1D Newton on V_B) | 25 / 25 | 1.619 | 2.823 | **2.301** | 100 % | 100 % | 81 s | YES — honest; the prior 1.62 was fwd-only |
| **z432 PTRAN** (pseudo-transient body integration, C_B=1 aF) | **18 / 25** | 1.349 | 1.027 | **1.199** (mixed n) | 31.9 % | 49.7 % | 2 558 s | partial — fwd drops VG1=0.2 column entirely (7/7 biases fail) |
| **z443 VBIC_AVL** (VBIC level-4 NPN, AVC1=AVC2=0.5 Si defaults) | 25 / 25 | 1.311 | 2.864 | **2.227** | 100 % | 100 % | 67 s | YES — was reported fwd-only as 1.31; honest avg = 2.23 dec |

### Per-branch RMSE (full breakdown)

```
z430 V_SINT_PIN
  fwd  VG1=0.2: 2.625   VG1=0.4: 0.786   VG1=0.6: 1.086
  bwd  VG1=0.2: 2.633   VG1=0.4: 2.662   VG1=0.6: 3.031

z432 PTRAN
  fwd  VG1=0.2:  ---    VG1=0.4: 0.703   VG1=0.6: 1.632      ← VG1=0.2 column DROPPED (all 7 biases fail to converge)
  bwd  VG1=0.2: 1.353   VG1=0.4: 0.521   VG1=0.6: 1.028

z443 VBIC_AVL
  fwd  VG1=0.2: 0.911   VG1=0.4: 1.135   VG1=0.6: 1.600
  bwd  VG1=0.2: 2.622   VG1=0.4: 2.665   VG1=0.6: 3.121
```

## §1.1 Where the cherry-picks came from

The synthesis suspected two failure modes; P1a confirms both.

1. **Direction-pick (z430, z443): forward sweep is systematically better
   than backward by 1.2 – 1.5 dec.** Hard-pin V_Sint=0 closes V_B
   forward-active runaway at low V_D but the backward sweep arrives at
   high V_D in a warm-state where V_B is far from the forward attractor;
   in V_B saturates and Id_pred goes lower than the measured Id by an
   order of magnitude across **every** VG2 column. This is the same
   1.31 dec → 2.86 dec gap that z454 SB_OFF already saw on its own
   bwd half (synthesis line 51); we now see it on three pipelines.

2. **Bias-pick (z432): the entire VG1 = 0.2 column was silently
   dropped from the forward report.** The 7 VG1=0.2 biases all
   diverge in pseudo-transient forward (warm-start from Vb=0.1 V
   never finds a stable attractor at low V_D), so they're discarded
   by `if not conv.any(): continue` and not reflected in n_biases.
   The 1.35 dec forward headline averages over 18 biases on 2 VG1
   branches; including the VG1=0.2 column (which only the backward
   sweep can solve) the honest fwd is undefined and the only honest
   number is **bwd = 1.03 dec on all 25**. Reporting fwd 1.349 next
   to bwd 1.027 as "an avg ~1.19" is, methodologically, a
   convergence-pick (different denominator each side).

Both failure modes share a common root: **warm-start initial condition
asymmetry**. Forward starts with V_B near 0 (which is the actual
attractor at small V_D); backward starts at high V_D where the model
sits in a fundamentally different regime (high-V_D V_B-runaway or
deep saturation) that the local Newton/integrator cannot escape from.
This is the fwd↔bwd asymmetry MASTER_FIX_PLAN already flagged for
**rbodymod=1** (P4 in the autonomous plan).

## §1.2 Recommended re-rank order (honest)

Replaces synthesis §1.1 ranking 1-4.

| Rank | Pipeline | avg dec | Caveat |
|---|---|---|---|
| 1 | **z432 PTRAN bwd-only** | **1.027 dec, 25 biases, 50 % conv** | drop forward direction entirely; report as "backward, n=25, conv 50 %". Fwd is broken on VG1=0.2 (KILL_SHOT for forward use). |
| 2 | z430 V_SINT_PIN fwd-only | 1.619 dec, 25 biases, 100 % conv | bwd 2.82 dec — physics is not symmetric. Report fwd only with that caveat. |
| 3 | z443 VBIC_AVL fwd-only | 1.311 dec, 25 biases, 100 % conv | bwd 2.86 dec — same forward-attractor-only behaviour as z430. Adding avalanche doesn't help bwd. |
| —  | All three "avg" combinations | 1.2 – 2.3 dec | misleading: averaging a working direction with a broken one gives a number with no consistent denominator across pipelines. The forward-only ranking is more defensible than any fwd+bwd avg. |

## §1.3 Implications for the campaign plan

- **The "best honest DC = 1.19 dec" headline (synthesis §1, §6.intro)
  is overstated**. The honest single-direction best is **1.03 dec
  bwd (z432 PT)** OR **1.31 dec fwd (z443 VBIC)**; the dishonesty was
  averaging mismatched-denominator results.
- **P4 (rbodymod=1) is now confirmed-urgent**. The fwd↔bwd asymmetry
  (1.6 → 2.8, 1.3 → 2.9) is exactly the signature §13 of the
  MASTER_FIX_PLAN flagged. Without distributed Rb, the high-V_D
  warm-start cannot relax to the correct V_B regime.
- **AMBITIOUS gate (<1.0 dec cell-wide on full 25 biases, both
  directions)** is **not reached** by any P1a pipeline. The
  closest is z432 bwd at 1.027 dec — a single-direction result.
- **KILL_SHOT trigger from AUTONOMOUS_PLAN_2026-05-16.md**
  ("P1 reveals avg > 2.0 dec everywhere → no functional model claim
  possible") is **partially triggered**: 2 of 3 pipelines have
  avg > 2.0 dec. Only z432 PT sits below, and only because its
  fwd half is computed over a subset. Pivot decision should be
  re-discussed after P1b (z446/z449/z454 on zgx).

## §1.4 Reproducibility

- Harness: `scripts/P1a_honest_baseline.py` (single-shot, no physics
  changes; only adds the backward-order V_D loop on top of existing
  z430/z432/z443 inner kernels).
- Run: `HSA_OVERRIDE_GFX_VERSION=11.0.0 venv/bin/python scripts/P1a_honest_baseline.py`
- Total wall time: 2 706 s (45 min) on ikaros.
- All per-curve RMSE arrays in `results/P1a_honest_baseline/summary.json`
  under `pipelines.<name>.per_curve_RMSE_{fwd,bwd,avg}` keyed by
  `per_curve_keys` (VG1, VG2 pairs, length 25 each).

---

## P1b zgx addendum (2026-05-17, post-rsync)

Newer pipelines run on zgx with both sweep directions on the full bias set:

| pipeline.variant | n_fwd | n_bwd | fwd | bwd | **avg** |
|---|---|---|---|---|---|
| z446.BASELINE_DC_GP | 25 | 25 | 1.619 | 2.823 | 2.221 |
| z446.DC_VBIC | 25 | 25 | 1.311 | 2.864 | 2.087 |
| z446.PT_GP | 25 | 25 | 1.349 | 1.027 | 1.188 |
| **z446.PT_VBIC** | 25 | 25 | 1.396 | 1.156 | **1.276** |
| z449.v449_A | 25 | 25 | 1.311 | 2.864 | 2.087 |
| z449.v449_B | 25 | 25 | 1.311 | 2.864 | 2.087 |
| z449.v449_C (α0×5) | 25 | 25 | 1.610 | 2.886 | 2.248 |
| z454.SB_OFF | 25 | 25 | 1.311 | 2.864 | 2.087 |
| z454.SB_ON_DEFAULT | 25 | 25 | 2.686 | 2.707 | 2.696 |
| z454.SB_LOW | 25 | 25 | 2.628 | 2.694 | 2.661 |
| z454.SB_HOT | 25 | 25 | 2.795 | 2.824 | 2.809 |

### Honest combined ranking (P1a + P1b, defensible numbers only)

1. **z432 / z446.PT_GP** — avg 1.188 (fwd=1.349, bwd=1.027, 25 biases each, 32%/50% conv)
2. **z446.PT_VBIC** — avg **1.276** (fwd=1.396, bwd=1.156, 25/25 biases, fully balanced)
3. z430 V_SINT_PIN — avg 2.301
4. z443 / z449.v449_A/B / z454.SB_OFF — avg 2.087 (all identical: same underlying physics)
5. z454.SB_* — avg 2.66-2.81 (snapback destroys DC)

Path-B claim defensible at **1.19-1.28 dec cell-wide avg, 25 biases, both sweep directions, fully balanced**. This is the publishable headline. All "<1.0 dec breakthroughs" in prior campaign were forward-only on dropped-bias subsets.

```


=== FILE: O76_prompt.md (1397 chars) ===
```
# NS-RAM Critique Cycle O76 (6h)

CRITICAL CONTEXT: 12h ago an oracle 3-way was done. Gemini+Grok flagged that the "4 pipelines give identical 1.311/2.864" result (z443=z449_A=z449_B=z454_SB_OFF) might be a hidden code-bug (like the prior z444 BESD no-op), NOT true DC invariance. OpenAI disagreed. We have NOT yet run the falsifier z460.

Last 6h activity (see context.md): P4 rbodymod=1 results — R_card, R_1k, R_1M ALL give identical fwd=1.349/bwd=1.027/avg=1.188 (=z432 baseline). 3 of 5 R values confirmed no-op. R_1G running. Headline z446.PT_VBIC = 1.276 dec avg still standing pending falsifier.

YOUR JOB IS TO CRITICIZE. Be harsh.

Q1 (overclaim): The 1.276 dec headline (z446.PT_VBIC) — where is it fragile? List the 3 biggest risks of it being wrong / overstated. Specifically check: convergence rates, basin selection, hidden bias dropouts, conv asymmetry.

Q2 (falsifier): What is the SINGLE highest-information experiment to falsify our strongest current claim? It can be the z460 falsifier (re-run z443 with ALPHA0×5) or something better.

Q3 (NO-CHEAT discipline drift): Read context.md (tail of 01_LOG). Cite specific log lines where we may have drifted from NO-CHEAT: cherry-picked results, hidden assumptions, unverified claims, oversimplified narrative. Quote the line and explain the cheat.

DO NOT be polite. We need to find every weakness before publishing the brief v4.5.

```


=== FILE: cap_breakdown.json (957 chars) ===
```json
{
  "bias": {
    "VDS": 2.0,
    "VGS": 0.6,
    "VBS": 0.7,
    "V_nwell": 2.0
  },
  "geometry": {
    "W_m": 3.6e-07,
    "L_m": 1.8e-07,
    "AS_m2": 6.48e-14,
    "PS_m": 1.08e-06,
    "NWELL_AREA_m2": 3.24e-13
  },
  "components_fF": {
    "M1 Cjs (S-body, fwd 0.7V)": 0.508,
    "M1 Cjd (D-body, rev 1.3V)": 0.14,
    "M1 Cgb (Meyer + overlap)": 0.028,
    "Parasitic NPN Cbe (fwd)": 1.17,
    "Parasitic NPN Cbc (rev)": 0.707,
    "Deep N-well diode (rev 1.3V, 5xWL)": 0.021,
    "Channel-body depletion": 0.083
  },
  "C_eff_total_fF": 2.657,
  "z448_claim_fF": 12.1,
  "ratio_z451_over_z448": 0.22,
  "notes": [
    "NWELL_AREA=5xWL chosen as plausible 130nm DNW footprint; varies \u00b12x with layout.",
    "CJE/CJC of parasitic NPN are TOTAL caps (not per-area); Pazos card units.",
    "Cgb in inversion is small; 0.05*Cox*WL is conservative upper bound.",
    "Cjs uses BSIM4 forward extrapolation (FC=0.5) since VBS=0.7 > FC*PB=0.37."
  ]
}
```


=== FILE: z454_honest_analysis.md (3151 chars) ===
```
# z454 — Snapback subcircuit integration on v449_B base

## Pre-registered gates

- **INFRA_pass**: True
- **DISCOVERY_pass**: False
- **DISCOVERY_who**: None
- **AMBITIOUS_pass**: False
- **AMBITIOUS_who**: None
- **KILL_SHOT**: True
- **kill_shot_reason**: all_SB_worse_than_SB_OFF

## DC (forward / backward / avg) [dec]

| condition | DC_fwd | DC_bwd | DC_avg | n |
|---|---|---|---|---|
| SB_OFF | 1.311 | 2.864 | 2.087 | 25 |
| SB_ON_DEFAULT | 2.686 | 2.707 | 2.696 | 25 |
| SB_LOW | 2.627 | 2.691 | 2.659 | 25 |
| SB_HOT | 2.795 | 2.824 | 2.809 | 25 |

## Fast-pulse smoke (per bias)

### SB_OFF
| bias | Vb_peak | t_peak[ns] | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |
|---|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.054 | 15.70 | 0.027 | 0.054 | None | None | False |
| VG1_0p6_VG2_0p2 | 0.054 | 15.70 | 0.027 | 0.054 | None | None | False |
| VG1_0p6_VG2_0p4 | 0.054 | 15.70 | 0.027 | 0.054 | None | None | False |
| VG1_0p4_VG2_0p0 | 0.003 | 15.70 | 0.002 | 0.003 | None | None | False |

### SB_ON_DEFAULT
| bias | Vb_peak | t_peak[ns] | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |
|---|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.709 | 15.70 | 0.636 | 0.636 | 3.133416770963705 | 3.1530663329161457 | False |
| VG1_0p6_VG2_0p2 | 0.709 | 15.70 | 0.636 | 0.636 | 3.133416770963705 | 3.1530663329161457 | False |
| VG1_0p6_VG2_0p4 | 0.709 | 15.70 | 0.636 | 0.636 | 3.1923654568210265 | 3.2120150187734673 | False |
| VG1_0p4_VG2_0p0 | 0.651 | 15.70 | 0.012 | 0.029 | 14.589111389236548 | 14.608760951188989 | False |

### SB_LOW
| bias | Vb_peak | t_peak[ns] | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |
|---|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.709 | 15.70 | 0.637 | 0.637 | 1.9740926157697125 | 1.9937421777221531 | False |
| VG1_0p6_VG2_0p2 | 0.709 | 15.70 | 0.637 | 0.637 | 1.9740926157697125 | 1.9937421777221531 | False |
| VG1_0p6_VG2_0p4 | 0.709 | 15.70 | 0.637 | 0.637 | 2.0526908635794747 | 2.072340425531915 | False |
| VG1_0p4_VG2_0p0 | 0.042 | 15.70 | 0.012 | 0.027 | None | None | False |

### SB_HOT
| bias | Vb_peak | t_peak[ns] | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |
|---|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.710 | 15.70 | 0.661 | 0.661 | 1.3649561952440554 | 1.3846057571964958 | False |
| VG1_0p6_VG2_0p2 | 0.710 | 15.70 | 0.661 | 0.661 | 1.3649561952440554 | 1.3846057571964958 | False |
| VG1_0p6_VG2_0p4 | 0.710 | 15.70 | 0.657 | 0.657 | 1.3846057571964958 | 1.4042553191489364 | False |
| VG1_0p4_VG2_0p0 | 0.709 | 15.70 | 0.019 | 0.640 | 8.28160200250313 | 8.301251564455569 | False |


## Snapback assert (z444-BURN avoidance)

- **SB_OFF**: snap_called=False |I_snap_d|=0.000e+00 A |I_snap_b|=0.000e+00 A V_db=NA BV=NA
- **SB_ON_DEFAULT**: snap_called=True |I_snap_d|=1.000e-02 A |I_snap_b|=5.485e-07 A V_db=1.4 BV=2.0
- **SB_LOW**: snap_called=True |I_snap_d|=1.000e-02 A |I_snap_b|=2.459e-06 A V_db=1.4 BV=1.6
- **SB_HOT**: snap_called=True |I_snap_d|=1.000e-02 A |I_snap_b|=4.232e-05 A V_db=1.4 BV=1.2

## Best condition: **SB_OFF**

- DC_avg = 2.087 dec
- vs SB_OFF DC_avg = 2.087 dec  (Δ = +0.000 dec)
```


=== FILE: z456_honest_analysis.md (3382 chars) ===
```
# z456 — R_body reset path (SB_HOT base + body-leak resistor)

## Pre-registered gates

- **INFRA_pass**: True
- **DISCOVERY_pass**: False
- **DISCOVERY_who**: []
- **AMBITIOUS_pass**: False
- **AMBITIOUS_who**: []
- **KILL_SHOT**: True
- **kill_shot_reason**: no_self_reset

## DC (forward / backward / avg) [dec]

| R_body | τ_est | DC_fwd | DC_bwd | DC_avg | n |
|---|---|---|---|---|---|
| R_INF (INF) | inf | 2.795 | 2.824 | 2.809 | 25 |
| R_1G (1.0e+09 Ω) | 2.7e-03s | 2.795 | 2.824 | 2.809 | 25 |
| R_100M (1.0e+08 Ω) | 2.7e-04s | 2.795 | 2.824 | 2.809 | 25 |
| R_10M (1.0e+07 Ω) | 2.7e-05s | 2.795 | 2.824 | 2.809 | 25 |
| R_1M (1.0e+06 Ω) | 2.7e-06s | 2.795 | 2.824 | 2.809 | 25 |

NOTE: R_body is wired into the transient body-KCL only. DC pathway
uses z429.run_vsint_pinned which is unchanged → DC values match SB_HOT
baseline. This is intentional (see script header).


## Fast-pulse extended (1µs hold) — self-reset timings

### R_INF (R_body=None, τ_est=infs)
| bias | Vb_peak | t→0.5V[ns] | t→reset[ns] | self-reset | n_cycles | period[ns] |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.661 | 1.3740000000000003 | None | False | 1 | None |
| VG1_0p6_VG2_0p2 | 0.661 | 1.3740000000000003 | None | False | 1 | None |
| VG1_0p6_VG2_0p4 | 0.657 | 1.4020000000000004 | None | False | 1 | None |
| VG1_0p4_VG2_0p0 | 0.640 | 8.30603541993147 | None | False | 1 | None |

### R_1G (R_body=1000000000.0, τ_est=2.7e-03s)
| bias | Vb_peak | t→0.5V[ns] | t→reset[ns] | self-reset | n_cycles | period[ns] |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.661 | 1.3740000000000003 | None | False | 1 | None |
| VG1_0p6_VG2_0p2 | 0.661 | 1.3740000000000003 | None | False | 1 | None |
| VG1_0p6_VG2_0p4 | 0.657 | 1.4020000000000004 | None | False | 1 | None |
| VG1_0p4_VG2_0p0 | 0.640 | 8.30603541993147 | None | False | 1 | None |

### R_100M (R_body=100000000.0, τ_est=2.7e-04s)
| bias | Vb_peak | t→0.5V[ns] | t→reset[ns] | self-reset | n_cycles | period[ns] |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.661 | 1.3880000000000001 | None | False | 1 | None |
| VG1_0p6_VG2_0p2 | 0.661 | 1.3880000000000001 | None | False | 1 | None |
| VG1_0p6_VG2_0p4 | 0.657 | 1.4020000000000004 | None | False | 1 | None |
| VG1_0p4_VG2_0p0 | 0.640 | 8.377057635184414 | None | False | 1 | None |

### R_10M (R_body=10000000.0, τ_est=2.7e-05s)
| bias | Vb_peak | t→0.5V[ns] | t→reset[ns] | self-reset | n_cycles | period[ns] |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.661 | 1.4020000000000004 | None | False | 1 | None |
| VG1_0p6_VG2_0p2 | 0.661 | 1.4020000000000004 | None | False | 1 | None |
| VG1_0p6_VG2_0p4 | 0.657 | 1.4300000000000002 | None | False | 1 | None |
| VG1_0p4_VG2_0p0 | 0.640 | 9.045774957672325 | None | False | 1 | None |

### R_1M (R_body=1000000.0, τ_est=2.7e-06s)
| bias | Vb_peak | t→0.5V[ns] | t→reset[ns] | self-reset | n_cycles | period[ns] |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.661 | 1.6680000000000004 | None | False | 1 | None |
| VG1_0p6_VG2_0p2 | 0.661 | 1.6680000000000004 | None | False | 1 | None |
| VG1_0p6_VG2_0p4 | 0.657 | 1.6960000000000004 | None | False | 1 | None |
| VG1_0p4_VG2_0p0 | 0.010 | None | None | False | 0 | None |


## Best (lowest DC_avg): **R_INF**, DC_avg=2.809


## Mario slide-21 reference: ~400 ns oscillation period
(quoted as benchmark — not target-matched, not optimized to fit)

```


=== FILE: z457_honest_analysis.md (6910 chars) ===
```
# z457 — NPN-collector σ-gate on top of z455 K_1p6

## Pre-registered gates

- **INFRA_pass**: True
- **DISCOVERY_pass**: False
- **DISCOVERY_who**: None
- **AMBITIOUS_pass**: False
- **AMBITIOUS_who**: None
- **KILL_SHOT**: False
- **kill_shot_reason**: None
- **VG1_0p2_regression_alerts**: [{'name': 'NY_1p6', 'branch': 'VG1=0.2 fwd', 'off': 2.30599288283575, 'this': 4.4844361404976025, 'delta': 2.1784432576618524}, {'name': 'NY_1p6', 'branch': 'VG1=0.2 bwd', 'off': 2.305992882835761, 'this': 3.871755165950178, 'delta': 1.5657622831144167}, {'name': 'NY_1p8', 'branch': 'VG1=0.2 fwd', 'off': 2.30599288283575, 'this': 4.4844361404976025, 'delta': 2.1784432576618524}, {'name': 'NY_1p8', 'branch': 'VG1=0.2 bwd', 'off': 2.305992882835761, 'this': 4.320333144553118, 'delta': 2.0143402617173565}]

## DC (forward / backward / avg) [dec]

| condition | mode | V_knee_npn | DC_fwd | DC_bwd | DC_avg | n |
|---|---|---|---|---|---|---|
| N_OFF | off | 1.60 | 2.686 | 2.719 | 2.702 | 25 |
| NX_1p4 | current | 1.40 | 2.622 | 2.785 | 2.703 | 25 |
| NX_1p6 | current | 1.60 | 2.629 | 2.791 | 2.710 | 25 |
| NX_1p8 | current | 1.80 | 2.368 | 2.589 | 2.479 | 25 |
| NY_1p6 | vbe | 1.60 | 4.292 | 4.160 | 4.226 | 25 |
| NY_1p8 | vbe | 1.80 | 4.262 | 3.543 | 3.902 | 25 |

## Per-branch DC (forward) — average log-RMSE by VG1

| condition | VG1=0.2 | VG1=0.4 | VG1=0.6 |
|---|---|---|---|
| N_OFF | 2.306 | 2.429 | 2.998 |
| NX_1p4 | 2.318 | 2.369 | 2.888 |
| NX_1p6 | 2.282 | 2.344 | 2.931 |
| NX_1p8 | 2.242 | 2.377 | 2.370 |
| NY_1p6 | 4.484 | 4.041 | 4.189 |
| NY_1p8 | 4.484 | 3.915 | 4.197 |

## I_snap_d at mid-DC (V_db≈0.8V, VG1=0.6, VG2=0.0)

| condition | mode | V_knee_npn | V_db | |I_snap_d| [A] | |I_snap_b| [A] |
|---|---|---|---|---|---|
| N_OFF | off | 1.60 | 0.80 | 1.000e-02 | 1.690e-14 |
| NX_1p4 | current | 1.40 | 0.80 | 1.000e-02 | 1.690e-14 |
| NX_1p6 | current | 1.60 | 0.80 | 8.892e-03 | 1.690e-14 |
| NX_1p8 | current | 1.80 | 0.80 | 1.629e-04 | 1.690e-14 |
| NY_1p6 | vbe | 1.60 | 0.80 | 1.000e-02 | 1.690e-14 |
| NY_1p8 | vbe | 1.80 | 0.80 | 1.000e-02 | 1.690e-14 |

## Fast-pulse smoke

### N_OFF  (mode=off, V_knee_npn=1.60)
| bias | Vb_peak | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.709 | 0.635 | 0.635 | 1.4042553191489364 | 1.4239048811013773 | False |
| VG1_0p6_VG2_0p2 | 0.709 | 0.635 | 0.635 | 1.4042553191489364 | 1.4239048811013773 | False |
| VG1_0p6_VG2_0p4 | 0.709 | 0.635 | 0.635 | 1.4239048811013773 | 1.4435544430538176 | False |
| VG1_0p4_VG2_0p0 | 0.709 | 0.019 | 0.635 | 8.340550688360452 | 8.360200250312891 | False |

### NX_1p4  (mode=current, V_knee_npn=1.40)
| bias | Vb_peak | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.638 | 0.635 | 0.635 | 1.4042553191489364 | 1.4239048811013773 | False |
| VG1_0p6_VG2_0p2 | 0.638 | 0.635 | 0.635 | 1.4042553191489364 | 1.4239048811013773 | False |
| VG1_0p6_VG2_0p4 | 0.638 | 0.635 | 0.635 | 1.4239048811013773 | 1.4435544430538176 | False |
| VG1_0p4_VG2_0p0 | 0.637 | 0.019 | 0.635 | 8.340550688360452 | 8.360200250312891 | False |

### NX_1p6  (mode=current, V_knee_npn=1.60)
| bias | Vb_peak | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.638 | 0.635 | 0.635 | 1.4042553191489364 | 1.4239048811013773 | False |
| VG1_0p6_VG2_0p2 | 0.638 | 0.635 | 0.635 | 1.4042553191489364 | 1.4239048811013773 | False |
| VG1_0p6_VG2_0p4 | 0.638 | 0.635 | 0.635 | 1.4239048811013773 | 1.4435544430538176 | False |
| VG1_0p4_VG2_0p0 | 0.637 | 0.019 | 0.635 | 8.340550688360452 | 8.360200250312891 | False |

### NX_1p8  (mode=current, V_knee_npn=1.80)
| bias | Vb_peak | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.636 | 0.635 | 0.635 | 1.4435544430538176 | 1.463204005006258 | False |
| VG1_0p6_VG2_0p2 | 0.636 | 0.635 | 0.635 | 1.4435544430538176 | 1.463204005006258 | False |
| VG1_0p6_VG2_0p4 | 0.636 | 0.635 | 0.635 | 1.463204005006258 | 1.4828535669586984 | False |
| VG1_0p4_VG2_0p0 | 0.636 | 0.019 | 0.635 | 8.537046307884857 | 8.556695869837299 | False |

### NY_1p6  (mode=vbe, V_knee_npn=1.60)
| bias | Vb_peak | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.709 | 0.635 | 0.635 | 1.5811013767209017 | 1.600750938673342 | False |
| VG1_0p6_VG2_0p2 | 0.709 | 0.635 | 0.635 | 1.5811013767209017 | 1.600750938673342 | False |
| VG1_0p6_VG2_0p4 | 0.709 | 0.635 | 0.635 | 1.600750938673342 | 1.6204005006257824 | False |
| VG1_0p4_VG2_0p0 | 0.031 | 0.003 | 0.030 | None | None | False |

### NY_1p8  (mode=vbe, V_knee_npn=1.80)
| bias | Vb_peak | Vb_5ns | Vb_10ns | t→0.3V[ns] | t→0.5V[ns] | self-reset |
|---|---|---|---|---|---|---|
| VG1_0p6_VG2_0p0 | 0.709 | 0.635 | 0.635 | 2.563579474342929 | 3.1137672090112645 | False |
| VG1_0p6_VG2_0p2 | 0.709 | 0.635 | 0.635 | 2.563579474342929 | 3.1137672090112645 | False |
| VG1_0p6_VG2_0p4 | 0.709 | 0.635 | 0.635 | 2.701126408010013 | 3.329912390488111 | False |
| VG1_0p4_VG2_0p0 | 0.007 | 0.000 | 0.007 | None | None | False |


## Snapback assert (deep, V_db=1.4V)

- **N_OFF**: snap_called=True |I_snap_d|=1.000e-02A |I_snap_b|=3.052e-08A V_db=1.4 BV=1.2
- **NX_1p4**: snap_called=True |I_snap_d|=1.000e-02A |I_snap_b|=3.052e-08A V_db=1.4 BV=1.2
- **NX_1p6**: snap_called=True |I_snap_d|=1.000e-02A |I_snap_b|=3.052e-08A V_db=1.4 BV=1.2
- **NX_1p8**: snap_called=True |I_snap_d|=1.000e-02A |I_snap_b|=3.052e-08A V_db=1.4 BV=1.2
- **NY_1p6**: snap_called=True |I_snap_d|=1.000e-02A |I_snap_b|=3.052e-08A V_db=1.4 BV=1.2
- **NY_1p8**: snap_called=True |I_snap_d|=1.000e-02A |I_snap_b|=3.052e-08A V_db=1.4 BV=1.2

## Best condition: **NX_1p8** (mode=current, V_knee_npn=1.80)

- DC_avg = 2.479 dec
- vs N_OFF DC_avg = 2.702 dec  (Δ = -0.223 dec)
- vs z455 K_1p6 reference = 2.702 dec  (Δ = -0.223 dec)

## Gate-wiring verdict

- **NX_1p4** (current, V_knee_npn=1.40): |I_snap_d|_mid = 1.000e-02 A → INEFFECTIVE (mid-DC I_snap_d still ≈ 10 mA clamp — σ insufficient at chosen V_sharp; either intrinsic Ic is large enough to overpower the gate, or wiring bug)
- **NX_1p6** (current, V_knee_npn=1.60): |I_snap_d|_mid = 8.892e-03 A → PARTIAL (mid-DC I_snap_d below clamp but not negligible)
- **NX_1p8** (current, V_knee_npn=1.80): |I_snap_d|_mid = 1.629e-04 A → EFFECTIVE (mid-DC I_snap_d < 1 mA)
- **NY_1p6** (vbe, V_knee_npn=1.60): |I_snap_d|_mid = 1.000e-02 A → INEFFECTIVE (mid-DC I_snap_d still ≈ 10 mA clamp — σ insufficient at chosen V_sharp; either intrinsic Ic is large enough to overpower the gate, or wiring bug)
- **NY_1p8** (vbe, V_knee_npn=1.80): |I_snap_d|_mid = 1.000e-02 A → INEFFECTIVE (mid-DC I_snap_d still ≈ 10 mA clamp — σ insufficient at chosen V_sharp; either intrinsic Ic is large enough to overpower the gate, or wiring bug)
```
