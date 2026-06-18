# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: context.md (26783 chars) ===
```

## 2026-05-16 — autonomous tick: campaign past R-1..R-10 (rebuild plan superseded)
Active phase: S15-trio post-mortem. Sebas data not fittable with PTM130 BSIM3 + Q1 BJT topology.
No R-phase gate active; pivots pending (BSIMSOI / 2D-PWL / accept residual). User decision-block.

## 2026-05-16 — S16 dispatched: Ideal Floating Body pyport rebuild
Physical fixes only, NO shortcuts:
  A) BSIM4 v4.8.3 eqns (PTM130 BSIM3 → BSIM4 mapping)
  B) Remove M1 bulk-source forward diode (leak #1, deep-N-well isolation justified)
  C) Q1 B-E one-way (leak #2, sigmoid rectifier above 0.7V)
  D) Mario Ipos PWL(V_G1) injected at body
  E) Sebas's per-bias BSIM fits as anchor
Pre-reg gates: AMBITIOUS = cell<1.0 dec AND ablation ≥0.3 dec per flag.
Budget 3h. Output z425_ideal_floating_body.

## 2026-05-16 — autonomous tick: APU=45C, S16 (z425 ideal floating body) running via subagent
Campaign on physics rebuild pivot; MEP+DS-N phases on hold pending S16 outcome.

## 2026-05-16 — autonomous tick: APU=47C, S16 (z425) still running, cron z332-z334 stale

## 2026-05-16 :47 — idle APU=47C sent=3 (S16 subagent active, no z-script proc)

## 2026-05-16 — autonomous deep-dive tick: APU=47C, S16 subagent still running (2 files written). 4A/4B/4C on hold pending S16.

## 2026-05-16 — oracle tick deferred: z425_finish still running, results not finalized
Skip oracle dispatch until ablation complete. No-cheat: critique needs finalized data.

## 2026-05-16 — autonomous tick: z425_finish python proc active (S16 ablation in progress). z332-z334 cron stale.

## 2026-05-16 — topology tick: R-1..R-10 superseded by S16 z425. ALL_FLAGS_ON variant still running.

## 2026-05-16 — autonomous tick: S17 found BJT bug (vbe_thresh=0.7→0.35, gate Iec symmetric).
Applied to bjt.py. z425 re-running with fix (finish_v2.log). Expect cell<2 dec.
Cron z332/z334 still stale.

## 2026-05-16 :47 — APU=50C sent=3 z42x=6 (z425 v2-run BJT-fix active)

[2026-05-15T18:08:16Z] VOICE_DECISION: Stoppa alla pågående processer omedelbart.

## 2026-05-16 — autonomous tick: APU=48C, S18 (z427 V_Sint fix) running. Voice OpenAI Realtime working w/ interrupt. Cron z332-z334 stale.

## 2026-05-16 — topology tick: R-phases superseded by S-series. S18 z427 (V_Sint runaway fix) running, 4 hypotheses.

[2026-05-15T18:14:49Z] VOICE_DECISION: Höj spänningen på V_G1 till 0,7 volt och sänk V_D till 0,4 volt.

[2026-05-15T18:14:56Z] VOICE_DECISION: Sänk strömbegränsningen till 50 mikroampere.

[2026-05-15T18:19:09Z] VOICE_DECISION: Stoppa alla pågående processer omedelbart.

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

```


=== FILE: p1a_summary.json (14023 chars) ===
```json
{
  "pipelines": {
    "z430_V_SINT_PIN": {
      "pipeline": "z430_V_SINT_PIN",
      "n_curves_fwd": 25,
      "n_curves_bwd": 25,
      "per_curve_keys": [
        {
          "VG1": 0.2,
          "VG2": -0.2
        },
        {
          "VG1": 0.2,
          "VG2": -0.15
        },
        {
          "VG1": 0.2,
          "VG2": -0.1
        },
        {
          "VG1": 0.2,
          "VG2": -0.05
        },
        {
          "VG1": 0.2,
          "VG2": 0.0
        },
        {
          "VG1": 0.2,
          "VG2": 0.05
        },
        {
          "VG1": 0.2,
          "VG2": 0.1
        },
        {
          "VG1": 0.4,
          "VG2": 0.0
        },
        {
          "VG1": 0.4,
          "VG2": 0.05
        },
        {
          "VG1": 0.4,
          "VG2": 0.1
        },
        {
          "VG1": 0.4,
          "VG2": 0.15
        },
        {
          "VG1": 0.4,
          "VG2": 0.2
        },
        {
          "VG1": 0.4,
          "VG2": 0.25
        },
        {
          "VG1": 0.4,
          "VG2": 0.3
        },
        {
          "VG1": 0.6,
          "VG2": 0.0
        },
        {
          "VG1": 0.6,
          "VG2": 0.05
        },
        {
          "VG1": 0.6,
          "VG2": 0.1
        },
        {
          "VG1": 0.6,
          "VG2": 0.15
        },
        {
          "VG1": 0.6,
          "VG2": 0.2
        },
        {
          "VG1": 0.6,
          "VG2": 0.25
        },
        {
          "VG1": 0.6,
          "VG2": 0.3
        },
        {
          "VG1": 0.6,
          "VG2": 0.35
        },
        {
          "VG1": 0.6,
          "VG2": 0.4
        },
        {
          "VG1": 0.6,
          "VG2": 0.45
        },
        {
          "VG1": 0.6,
          "VG2": 0.5
        }
      ],
      "per_curve_RMSE_fwd": [
        2.995569539321151,
        2.9639070064225668,
        2.9147643174034887,
        2.807599313725496,
        2.5804064013403742,
        2.23221185969724,
        1.5621284451505901,
        0.8596464015912328,
        0.8401947112911373,
        0.8139994105517246,
        0.804985725218632,
        0.750440024505717,
        0.719154855747766,
        0.6990105481912723,
        1.2634930244094238,
        1.2373117498381738,
        1.211874604549978,
        1.173496643796118,
        1.1490827431059196,
        1.1093301362671464,
        1.0536020320845048,
        0.9889970579162992,
        0.945860992888704,
        0.8969622921385639,
        0.8074263149801115
      ],
      "per_curve_RMSE_bwd": [
        2.9888987789399177,
        2.970803572241825,
        2.920217662019734,
        2.830634403368255,
        2.6026733307027263,
        2.2474950696001805,
        1.5548137731757288,
        3.109770862712883,
        3.0353902546973996,
        2.913234631490086,
        2.8352487196806773,
        2.538165167370736,
        2.2119397541599586,
        1.6862665463393607,
        3.3135221636796652,
        3.3028960015340307,
        3.2837640904332237,
        3.1750351939867834,
        3.2036222913236774,
        3.08327499819576,
        3.064888870789839,
        2.9568844791671225,
        2.7822441182806297,
        2.6281689510847155,
        2.4012868203198416
      ],
      "per_curve_RMSE_avg": [
        2.9922360180688674,
        2.9673572929094307,
        2.9174922638784,
        2.819140386001009,
        2.5915637810431758,
        2.239866499866692,
        1.5584754005767216,
        2.281410392098141,
        2.227051992118004,
        2.138870152497778,
        2.0840630173121792,
        1.8715558563978605,
        1.6446673193352521,
        1.290757648008707,
        2.507572905413484,
        2.493999034005135,
        2.475050167044287,
        2.3935269850218366,
        2.406614524178082,
        2.3170237445621433,
        2.2916829004362724,
        2.204686032489266,
        2.0779238859466873,
        1.9636462752955532,
        1.791384331678177
      ],
      "per_branch_RMSE_fwd": {
        "VG1_0.2": 2.6245587058145876,
        "VG1_0.4": 0.7859912604242465,
        "VG1_0.6": 1.0855839638811928
      },
      "per_branch_RMSE_bwd": {
        "VG1_0.2": 2.6333558307165665,
        "VG1_0.4": 2.6615863931026627,
        "VG1_0.6": 3.031251303082154
      },
      "cell_wide_fwd": 1.6187161900853293,
      "cell_wide_bwd": 2.822789857574503,
      "cell_wide_avg": 2.3009111982071198,
      "convergence_rate_fwd": 1.0,
      "convergence_rate_bwd": 1.0,
      "fails_fwd": 0,
      "fails_bwd": 0,
      "wall_sec_fwd": 31.0,
      "wall_sec_bwd": 50.6
    },
    "z432_PTRAN": {
      "pipeline": "z432_PTRAN",
      "n_curves_fwd": 18,
      "n_curves_bwd": 25,
      "per_curve_keys": [
        {
          "VG1": 0.2,
          "VG2": -0.2
        },
        {
          "VG1": 0.2,
          "VG2": -0.15
        },
        {
          "VG1": 0.2,
          "VG2": -0.1
        },
        {
          "VG1": 0.2,
          "VG2": -0.05
        },
        {
          "VG1": 0.2,
          "VG2": 0.0
        },
        {
          "VG1": 0.2,
          "VG2": 0.05
        },
        {
          "VG1": 0.2,
          "VG2": 0.1
        },
        {
          "VG1": 0.4,
          "VG2": 0.0
        },
        {
          "VG1": 0.4,
          "VG2": 0.05
        },
        {
          "VG1": 0.4,
          "VG2": 0.1
        },
        {
          "VG1": 0.4,
          "VG2": 0.15
        },
        {
          "VG1": 0.4,
          "VG2": 0.2
        },
        {
          "VG1": 0.4,
          "VG2": 0.25
        },
        {
          "VG1": 0.4,
          "VG2": 0.3
        },
        {
          "VG1": 0.6,
          "VG2": 0.0
        },
        {
          "VG1": 0.6,
          "VG2": 0.05
        },
        {
          "VG1": 0.6,
          "VG2": 0.1
        },
        {
          "VG1": 0.6,
          "VG2": 0.15
        },
        {
          "VG1": 0.6,
          "VG2": 0.2
        },
        {
          "VG1": 0.6,
          "VG2": 0.25
        },
        {
          "VG1": 0.6,
          "VG2": 0.3
        },
        {
          "VG1": 0.6,
          "VG2": 0.35
        },
        {
          "VG1": 0.6,
          "VG2": 0.4
        },
        {
          "VG1": 0.6,
          "VG2": 0.45
        },
        {
          "VG1": 0.6,
          "VG2": 0.5
        }
      ],
      "per_curve_RMSE_fwd": [
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        0.9196197800516875,
        0.8343104259776877,
        0.7144715536109328,
        0.6446062728143475,
        0.5803447256098961,
        0.5867937930404864,
        0.5611158893229279,
        1.8402102675514411,
        1.779977960067933,
        1.7205063498640427,
        1.7537357391410728,
        1.572622297641623,
        1.682579117273603,
        1.4984201583712147,
        1.4973077024974752,
        1.51370956622811,
        1.5200942236788408,
        1.5204721839077153
      ],
      "per_curve_RMSE_bwd": [
        0.9499149802233882,
        1.1221798062682964,
        1.4248055914882694,
        1.4966201322860386,
        1.4821598316504208,
        1.4909728571550804,
        1.4040828480844791,
        0.6669498852002764,
        0.6078268636614734,
        0.5251232312786993,
        0.4764391499541844,
        0.4485420106940696,
        0.44362074037302063,
        0.43293959659819203,
        1.2179941198564106,
        1.16970403267042,
        1.1215482000685422,
        1.0107935606662104,
        1.0000200886984898,
        0.9408486584041135,
        0.9373968451238335,
        0.9366251394277448,
        0.9690228089754666,
        0.9809419605648078,
        0.98087359295478
      ],
      "per_curve_RMSE_avg": [
        0.9499149802233882,
        1.1221798062682964,
        1.4248055914882694,
        1.4966201322860386,
        1.4821598316504208,
        1.4909728571550804,
        1.4040828480844791,
        0.8032816097829503,
        0.7299066320713952,
        0.6269864468024019,
        0.5667942795056558,
        0.5186472480890574,
        0.5201568594375874,
        0.5011425623279595,
        1.5604299895872515,
        1.5060758716567109,
        1.4522417954664917,
        1.431309307072542,
        1.3177976075331435,
        1.3631340146709783,
        1.2497951464662782,
        1.2488388622569186,
        1.2708898172550593,
        1.2792446167286589,
        1.2794429778995853
      ],
      "per_branch_RMSE_fwd": {
        "VG1_0.4": 0.7034343579866809,
        "VG1_0.6": 1.6319445170904723
      },
      "per_branch_RMSE_bwd": {
        "VG1_0.2": 1.3534514215128182,
        "VG1_0.4": 0.5213239064623348,
        "VG1_0.6": 1.028497244436865
      },
      "cell_wide_fwd": 1.34906163370139,
      "cell_wide_bwd": 1.026861976331113,
      "cell_wide_avg": 1.198835436988685,
      "convergence_rate_fwd": 0.31851851851851853,
      "convergence_rate_bwd": 0.49733333333333335,
      "fails_fwd": 7,
      "fails_bwd": 0,
      "wall_sec_fwd": 1434.0,
      "wall_sec_bwd": 1123.9
    },
    "z443_VBIC_AVL": {
      "pipeline": "z443_VBIC_AVL",
      "n_curves_fwd": 25,
      "n_curves_bwd": 25,
      "per_curve_keys": [
        {
          "VG1": 0.2,
          "VG2": -0.2
        },
        {
          "VG1": 0.2,
          "VG2": -0.15
        },
        {
          "VG1": 0.2,
          "VG2": -0.1
        },
        {
          "VG1": 0.2,
          "VG2": -0.05
        },
        {
          "VG1": 0.2,
          "VG2": 0.0
        },
        {
          "VG1": 0.2,
          "VG2": 0.05
        },
        {
          "VG1": 0.2,
          "VG2": 0.1
        },
        {
          "VG1": 0.4,
          "VG2": 0.0
        },
        {
          "VG1": 0.4,
          "VG2": 0.05
        },
        {
          "VG1": 0.4,
          "VG2": 0.1
        },
        {
          "VG1": 0.4,
          "VG2": 0.15
        },
        {
          "VG1": 0.4,
          "VG2": 0.2
        },
        {
          "VG1": 0.4,
          "VG2": 0.25
        },
        {
          "VG1": 0.4,
          "VG2": 0.3
        },
        {
          "VG1": 0.6,
          "VG2": 0.0
        },
        {
          "VG1": 0.6,
          "VG2": 0.05
        },
        {
          "VG1": 0.6,
          "VG2": 0.1
        },
        {
          "VG1": 0.6,
          "VG2": 0.15
        },
        {
          "VG1": 0.6,
          "VG2": 0.2
        },
        {
          "VG1": 0.6,
          "VG2": 0.25
        },
        {
          "VG1": 0.6,
          "VG2": 0.3
        },
        {
          "VG1": 0.6,
          "VG2": 0.35
        },
        {
          "VG1": 0.6,
          "VG2": 0.4
        },
        {
          "VG1": 0.6,
          "VG2": 0.45
        },
        {
          "VG1": 0.6,
          "VG2": 0.5
        }
      ],
      "per_curve_RMSE_fwd": [
        0.9961746947923242,
        0.9638130403611452,
        0.953037440683144,
        0.9346170084362722,
        0.9216065401655901,
        0.8506199672800442,
        0.7274046179242989,
        1.3953493496263452,
        1.3401647993057377,
        1.268283011999761,
        1.228947730847812,
        1.0730290744309365,
        0.8464431948598645,
        0.5393453837801758,
        1.8001619425390598,
        1.773745623168738,
        1.7478139514175215,
        1.7073849902718534,
        1.6828185776706863,
        1.6388473805680264,
        1.5795526080324263,
        1.5050343797575105,
        1.4401234763862156,
        1.3724814090920314,
        1.2457626753815572
      ],
      "per_curve_RMSE_bwd": [
        2.983658971666357,
        2.963463441174354,
        2.912437374690717,
        2.8216002936351177,
        2.590278846267137,
        2.228495641006743,
        1.5270016342418753,
        3.103100778925262,
        3.0384293472611574,
        2.9169586968729293,
        2.8396674825565604,
        2.545987261301781,
        2.2187942379163847,
        1.6997139881558445,
        3.393965034961184,
        3.37998368544684,
        3.3661514877639602,
        3.2532729518132295,
        3.288046573576449,
        3.1639508153629117,
        3.153050755958788,
        3.0481353647496103,
        2.879570390450275,
        2.7480729256699985,
        2.5319434460108385
      ],
      "per_curve_RMSE_avg": [
        2.2242509842360105,
        2.2035257139351856,
        2.166860842882998,
        2.1017774822159634,
        1.944081160380785,
        1.6866753022916945,
        1.1960040696324894,
        2.405850603390538,
        2.348200863171212,
        2.249132036785502,
        2.187924123072496,
        1.9536430750376406,
        1.679213201522595,
        1.2609363751866371,
        2.716577044545679,
        2.699116823110576,
        2.6819609847409462,
        2.5979750195108866,
        2.6118124583669307,
        2.519563999864889,
        2.493663520926021,
        2.403773875189731,
        2.276607263099706,
        2.1720508768331963,
        1.9953273236703615
      ],
      "per_branch_RMSE_fwd": {
        "VG1_0.2": 0.9106606186802844,
        "VG1_0.4": 1.1351932963274642,
        "VG1_0.6": 1.5995503114003287
      },
      "per_branch_RMSE_bwd": {
        "VG1_0.2": 2.6223088904709706,
        "VG1_0.4": 2.6653269282094967,
        "VG1_0.6": 3.1212950468688705
      },
      "cell_wide_fwd": 1.3110292027686277,
      "cell_wide_bwd": 2.863778003439945,
      "cell_wide_avg": 2.2271082173413377,
      "convergence_rate_fwd": 1.0,
      "convergence_rate_bwd": 1.0,
      "fails_fwd": 0,
      "fails_bwd": 0,
      "wall_sec_fwd": 27.7,
      "wall_sec_bwd": 38.6
    }
  },
  "total_wall_sec": 2705.7,
  "notes": {
    "dataset": "Sebas 33-curve set (find_params hits ~25 biases)",
    "fwd_direction": "V_D 0.05 -> 2.0 V (warm-start V_B from prev)",
    "bwd_direction": "V_D 2.0 -> 0.05 V (warm-start V_B from prev)",
    "rmse": "log10-RMSE over converged V_D points, then quadratic mean across biases",
    "cell_wide_avg": "sqrt(0.5*(cell_fwd^2 + cell_bwd^2)), quadratic mean"
  }
}
```


=== FILE: z446_p1b_summary.json (44151 chars) ===
```json
{
  "pipeline": "z446_vbic_pt",
  "phase": "P1b_honest_baseline",
  "host": "zgx",
  "n_curves_total": 33,
  "variants": {
    "BASELINE_DC_GP": {
      "kind": "dc",
      "flags": {},
      "forward": {
        "cell_rmse_dec": 1.618716190085353,
        "n": 25,
        "fails": 0,
        "wall_sec": 16.779592752456665,
        "per_bias": [
          {
            "VG1": 0.2,
            "VG2": -0.05,
            "log_rmse": 2.807599313725496,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.6037114065549327
          },
          {
            "VG1": 0.2,
            "VG2": -0.1,
            "log_rmse": 2.9147643174034887,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.563979579442372
          },
          {
            "VG1": 0.2,
            "VG2": -0.15,
            "log_rmse": 2.9639070064228905,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5693190308805337
          },
          {
            "VG1": 0.2,
            "VG2": -0.2,
            "log_rmse": 2.995569539321151,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5036165447758312
          },
          {
            "VG1": 0.2,
            "VG2": 0.0,
            "log_rmse": 2.5804064013403742,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5813706794352139
          },
          {
            "VG1": 0.2,
            "VG2": 0.05,
            "log_rmse": 2.2322118596972396,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5230928273993422
          },
          {
            "VG1": 0.2,
            "VG2": 0.1,
            "log_rmse": 1.5621284451505901,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.790764775453128
          },
          {
            "VG1": 0.4,
            "VG2": 0.0,
            "log_rmse": 0.8596464015912328,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.653583006295542
          },
          {
            "VG1": 0.4,
            "VG2": 0.05,
            "log_rmse": 0.8401947112911373,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.6524283192197121
          },
          {
            "VG1": 0.4,
            "VG2": 0.1,
            "log_rmse": 0.8139994105517245,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.6512516111713054
          },
          {
            "VG1": 0.4,
            "VG2": 0.15,
            "log_rmse": 0.804985725218632,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.6500794580627289
          },
          {
            "VG1": 0.4,
            "VG2": 0.2,
            "log_rmse": 0.7504400245057168,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.6489218990294232
          },
          {
            "VG1": 0.4,
            "VG2": 0.25,
            "log_rmse": 0.7191548557477662,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.6477445453869607
          },
          {
            "VG1": 0.4,
            "VG2": 0.3,
            "log_rmse": 0.6990105481912723,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.6465813416154915
          },
          {
            "VG1": 0.6,
            "VG2": 0.0,
            "log_rmse": 1.2634930244094238,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.6873764146896337
          },
          {
            "VG1": 0.6,
            "VG2": 0.05,
            "log_rmse": 1.2373117498381738,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.6873733084288105
          },
          {
            "VG1": 0.6,
            "VG2": 0.1,
            "log_rmse": 1.2118746045499778,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.6873888394144266
          },
          {
            "VG1": 0.6,
            "VG2": 0.15,
            "log_rmse": 1.173496643796118,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.6873779678081015
          },
          {
            "VG1": 0.6,
            "VG2": 0.2,
            "log_rmse": 1.1490827431059196,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.6873764146896337
          },
          {
            "VG1": 0.6,
            "VG2": 0.25,
            "log_rmse": 1.1093301362671464,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.6873717552864553
          },
          {
            "VG1": 0.6,
            "VG2": 0.3,
            "log_rmse": 1.0536020320845048,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.6873764146896337
          },
          {
            "VG1": 0.6,
            "VG2": 0.35,
            "log_rmse": 0.9889970579162991,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.6873810740211496
          },
          {
            "VG1": 0.6,
            "VG2": 0.4,
            "log_rmse": 0.945860992888704,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.6838265845040884
          },
          {
            "VG1": 0.6,
            "VG2": 0.45,
            "log_rmse": 0.8969622921385639,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.6825872153642083
          },
          {
            "VG1": 0.6,
            "VG2": 0.5,
            "log_rmse": 0.8074263149801115,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.682597844802064
          }
        ],
        "per_branch": {
          "VG1_0.2": 2.62455870581464,
          "VG1_0.4": 0.7859912604242463,
          "VG1_0.6": 1.0855839638811928
        },
        "vb_max_overall": 0.790764775453128
      },
      "backward": {
        "cell_rmse_dec": 2.822789857574503,
        "n": 25,
        "fails": 0,
        "wall_sec": 27.217610120773315,
        "per_bias": [
          {
            "VG1": 0.2,
            "VG2": -0.05,
            "log_rmse": 2.8306344033682556,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.4971333978621015
          },
          {
            "VG1": 0.2,
            "VG2": -0.1,
            "log_rmse": 2.920217662019734,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.49881584973916404
          },
          {
            "VG1": 0.2,
            "VG2": -0.15,
            "log_rmse": 2.970803572241825,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.500960436735374
          },
          {
            "VG1": 0.2,
            "VG2": -0.2,
            "log_rmse": 2.9888987789399177,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.4874517950486789
          },
          {
            "VG1": 0.2,
            "VG2": 0.0,
            "log_rmse": 2.6026733307027263,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.49337414790147527
          },
          {
            "VG1": 0.2,
            "VG2": 0.05,
            "log_rmse": 2.2474950696001805,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5097714064310412
          },
          {
            "VG1": 0.2,
            "VG2": 0.1,
            "log_rmse": 1.5548137731757288,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5076715893665589
          },
          {
            "VG1": 0.4,
            "VG2": 0.0,
            "log_rmse": 3.1097708627128826,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.4698648263886137
          },
          {
            "VG1": 0.4,
            "VG2": 0.05,
            "log_rmse": 3.0353902546974,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.4698642881371262
          },
          {
            "VG1": 0.4,
            "VG2": 0.1,
            "log_rmse": 2.913234631490086,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.4698597640352819
          },
          {
            "VG1": 0.4,
            "VG2": 0.15,
            "log_rmse": 2.8352487196806773,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.46986179467102135
          },
          {
            "VG1": 0.4,
            "VG2": 0.2,
            "log_rmse": 2.538165167370736,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.47399504206579596
          },
          {
            "VG1": 0.4,
            "VG2": 0.25,
            "log_rmse": 2.2119397541599586,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.46985587289751946
          },
          {
            "VG1": 0.4,
            "VG2": 0.3,
            "log_rmse": 1.6862665463393607,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.47398232858389217
          },
          {
            "VG1": 0.6,
            "VG2": 0.0,
            "log_rmse": 3.3135221636796657,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.47736420956272335
          },
          {
            "VG1": 0.6,
            "VG2": 0.05,
            "log_rmse": 3.3028960015340307,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.47737527463002916
          },
          {
            "VG1": 0.6,
            "VG2": 0.1,
            "log_rmse": 3.2837640904332237,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.4773757893035371
          },
          {
            "VG1": 0.6,
            "VG2": 0.15,
            "log_rmse": 3.1750351939867834,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.47739920875262215
          },
          {
            "VG1": 0.6,
            "VG2": 0.2,
            "log_rmse": 3.2036222913236774,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.47736832670481383
          },
          {
            "VG1": 0.6,
            "VG2": 0.25,
            "log_rmse": 3.0832749981957606,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.4773727012881009
          },
          {
            "VG1": 0.6,
            "VG2": 0.3,
            "log_rmse": 3.0648888707898396,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.47735674752156537
          },
          {
            "VG1": 0.6,
            "VG2": 0.35,
            "log_rmse": 2.9568844791671225,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.4773690986811312
          },
          {
            "VG1": 0.6,
            "VG2": 0.4,
            "log_rmse": 2.7822441182806297,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.4773568288448931
          },
          {
            "VG1": 0.6,
            "VG2": 0.45,
            "log_rmse": 2.628168951084715,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.4905376398000805
          },
          {
            "VG1": 0.6,
            "VG2": 0.5,
            "log_rmse": 2.4012868203198416,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.4905440109569477
          }
        ],
        "per_branch": {
          "VG1_0.2": 2.6333558307165665,
          "VG1_0.4": 2.6615863931026627,
          "VG1_0.6": 3.031251303082154
        },
        "vb_max_overall": 0.5097714064310412
      },
      "avg_cell_rmse_dec": 2.220753023829928
    },
    "DC_VBIC": {
      "kind": "dc",
      "flags": {
        "use_vbic_for_q1": true,
        "vbic_AVC1": 0.5,
        "vbic_AVC2": 0.5
      },
      "forward": {
        "cell_rmse_dec": 1.3110292027686277,
        "n": 25,
        "fails": 0,
        "wall_sec": 15.039406538009644,
        "per_bias": [
          {
            "VG1": 0.2,
            "VG2": -0.05,
            "log_rmse": 0.9346170084362722,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5773660977214775
          },
          {
            "VG1": 0.2,
            "VG2": -0.1,
            "log_rmse": 0.9530374406831439,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5789680304136198
          },
          {
            "VG1": 0.2,
            "VG2": -0.15,
            "log_rmse": 0.9638130403611453,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.58057108814622
          },
          {
            "VG1": 0.2,
            "VG2": -0.2,
            "log_rmse": 0.996174694792324,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5821758849102289
          },
          {
            "VG1": 0.2,
            "VG2": 0.0,
            "log_rmse": 0.9216065401655898,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5663476858242638
          },
          {
            "VG1": 0.2,
            "VG2": 0.05,
            "log_rmse": 0.8506199672800442,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5554487550208759
          },
          {
            "VG1": 0.2,
            "VG2": 0.1,
            "log_rmse": 0.727404617924299,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5507789402888231
          },
          {
            "VG1": 0.4,
            "VG2": 0.0,
            "log_rmse": 1.3953493496263452,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.4711556418098316
          },
          {
            "VG1": 0.4,
            "VG2": 0.05,
            "log_rmse": 1.340164799305738,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.47045628085409147
          },
          {
            "VG1": 0.4,
            "VG2": 0.1,
            "log_rmse": 1.2682830119997612,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.4697549658408738
          },
          {
            "VG1": 0.4,
            "VG2": 0.15,
            "log_rmse": 1.228947730847812,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.46906996331020284
          },
          {
            "VG1": 0.4,
            "VG2": 0.2,
            "log_rmse": 1.0730290744309365,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.4684079109417015
          },
          {
            "VG1": 0.4,
            "VG2": 0.25,
            "log_rmse": 0.8464431948598645,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.4677452030990922
          },
          {
            "VG1": 0.4,
            "VG2": 0.3,
            "log_rmse": 0.5393453837801759,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.46710472141434656
          },
          {
            "VG1": 0.6,
            "VG2": 0.0,
            "log_rmse": 1.8001619425390598,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5154456857892937
          },
          {
            "VG1": 0.6,
            "VG2": 0.05,
            "log_rmse": 1.7737456231687383,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5154434838272332
          },
          {
            "VG1": 0.6,
            "VG2": 0.1,
            "log_rmse": 1.7478139514175215,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5154544934235054
          },
          {
            "VG1": 0.6,
            "VG2": 0.15,
            "log_rmse": 1.7073849902718534,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.515446786762298
          },
          {
            "VG1": 0.6,
            "VG2": 0.2,
            "log_rmse": 1.6828185776706865,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5154456857892937
          },
          {
            "VG1": 0.6,
            "VG2": 0.25,
            "log_rmse": 1.6388473805680264,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5154423828381769
          },
          {
            "VG1": 0.6,
            "VG2": 0.3,
            "log_rmse": 1.5795526080324265,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5154456857892937
          },
          {
            "VG1": 0.6,
            "VG2": 0.35,
            "log_rmse": 1.5050343797575105,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5154489886922542
          },
          {
            "VG1": 0.6,
            "VG2": 0.4,
            "log_rmse": 1.4401234763862154,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5127256372262112
          },
          {
            "VG1": 0.6,
            "VG2": 0.45,
            "log_rmse": 1.3724814090920314,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5118530642624964
          },
          {
            "VG1": 0.6,
            "VG2": 0.5,
            "log_rmse": 1.2457626753815572,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.5118603754394763
          }
        ],
        "per_branch": {
          "VG1_0.2": 0.9106606186802843,
          "VG1_0.4": 1.1351932963274642,
          "VG1_0.6": 1.5995503114003287
        },
        "vb_max_overall": 0.5821758849102289
      },
      "backward": {
        "cell_rmse_dec": 2.863778003439945,
        "n": 25,
        "fails": 0,
        "wall_sec": 21.033201456069946,
        "per_bias": [
          {
            "VG1": 0.2,
            "VG2": -0.05,
            "log_rmse": 2.8216002936351177,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.3921617718825713
          },
          {
            "VG1": 0.2,
            "VG2": -0.1,
            "log_rmse": 2.9124373746907173,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.38333428508672845
          },
          {
            "VG1": 0.2,
            "VG2": -0.15,
            "log_rmse": 2.963463441174354,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.3850026625336777
          },
          {
            "VG1": 0.2,
            "VG2": -0.2,
            "log_rmse": 2.983658971666357,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.3770660460044958
          },
          {
            "VG1": 0.2,
            "VG2": 0.0,
            "log_rmse": 2.5902788462671364,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.3939600613456581
          },
          {
            "VG1": 0.2,
            "VG2": 0.05,
            "log_rmse": 2.2284956410067434,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.38807360834732146
          },
          {
            "VG1": 0.2,
            "VG2": 0.1,
            "log_rmse": 1.5270016342418753,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.38667920710272113
          },
          {
            "VG1": 0.4,
            "VG2": 0.0,
            "log_rmse": 3.103100778925262,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.3420152988706079
          },
          {
            "VG1": 0.4,
            "VG2": 0.05,
            "log_rmse": 3.038429347261158,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.3420114752279528
          },
          {
            "VG1": 0.4,
            "VG2": 0.1,
            "log_rmse": 2.9169586968729293,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.3420011092223839
          },
          {
            "VG1": 0.4,
            "VG2": 0.15,
            "log_rmse": 2.8396674825565604,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.34199855020593245
          },
          {
            "VG1": 0.4,
            "VG2": 0.2,
            "log_rmse": 2.545987261301781,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.3419993729624483
          },
          {
            "VG1": 0.4,
            "VG2": 0.25,
            "log_rmse": 2.2187942379163847,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.3419892500702572
          },
          {
            "VG1": 0.4,
            "VG2": 0.3,
            "log_rmse": 1.6997139881558447,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.3419891154344591
          },
          {
            "VG1": 0.6,
            "VG2": 0.0,
            "log_rmse": 3.393965034961184,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.35913849689685023
          },
          {
            "VG1": 0.6,
            "VG2": 0.05,
            "log_rmse": 3.37998368544684,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.35915309676949425
          },
          {
            "VG1": 0.6,
            "VG2": 0.1,
            "log_rmse": 3.3661514877639602,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.3591553428225052
          },
          {
            "VG1": 0.6,
            "VG2": 0.15,
            "log_rmse": 3.2532729518132295,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.37267719675200894
          },
          {
            "VG1": 0.6,
            "VG2": 0.2,
            "log_rmse": 3.2880465735764486,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.35914663924638746
          },
          {
            "VG1": 0.6,
            "VG2": 0.25,
            "log_rmse": 3.1639508153629117,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.3726642639779303
          },
          {
            "VG1": 0.6,
            "VG2": 0.3,
            "log_rmse": 3.1530507559587884,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.3591413046357307
          },
          {
            "VG1": 0.6,
            "VG2": 0.35,
            "log_rmse": 3.0481353647496103,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.359148604598464
          },
          {
            "VG1": 0.6,
            "VG2": 0.4,
            "log_rmse": 2.879570390450275,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.3591380008876446
          },
          {
            "VG1": 0.6,
            "VG2": 0.45,
            "log_rmse": 2.7480729256699985,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.35913949295278563
          },
          {
            "VG1": 0.6,
            "VG2": 0.5,
            "log_rmse": 2.5319434460108385,
            "n_conv": 30,
            "n_pts": 30,
            "vb_max": 0.35914622916980016
          }
        ],
        "per_branch": {
          "VG1_0.2": 2.6223088904709706,
          "VG1_0.4": 2.665326928209497,
          "VG1_0.6": 3.1212950468688705
        },
        "vb_max_overall": 0.3939600613456581
      },
      "avg_cell_rmse_dec": 2.0874036031042866
    },
    "PT_GP": {
      "kind": "pt",
      "flags": {},
      "forward": {
        "cell_rmse_dec": 1.3490616337013903,
        "n": 18,
        "fails": 7,
        "wall_sec": 778.2727203369141,
        "per_bias": [
          {
            "VG1": 0.4,
            "VG2": 0.0,
            "log_rmse": 0.9196197800516875,
            "n_conv": 10,
            "n_pts": 30,
            "vb_max": 0.6758002700099793
          },
          {
            "VG1": 0.4,
            "VG2": 0.05,
            "log_rmse": 0.8343104259776877,
            "n_conv": 10,
            "n_pts": 30,
            "vb_max": 0.6433306705391314
          },
          {
            "VG1": 0.4,
            "VG2": 0.1,
            "log_rmse": 0.7144715536109327,
            "n_conv": 10,
            "n_pts": 30,
            "vb_max": 0.6451574323915106
          },
          {
            "VG1": 0.4,
            "VG2": 0.15,
            "log_rmse": 0.6446062728143475,
            "n_conv": 10,
            "n_pts": 30,
            "vb_max": 0.6849163017866597
          },
          {
            "VG1": 0.4,
            "VG2": 0.2,
            "log_rmse": 0.580344725609896,
            "n_conv": 11,
            "n_pts": 30,
            "vb_max": 0.6218378341214907
          },
          {
            "VG1": 0.4,
            "VG2": 0.25,
            "log_rmse": 0.5867937930404864,
            "n_conv": 11,
            "n_pts": 30,
            "vb_max": 0.6551418164394375
          },
          {
            "VG1": 0.4,
            "VG2": 0.3,
            "log_rmse": 0.5611158893229277,
            "n_conv": 11,
            "n_pts": 30,
            "vb_max": 0.6051031586506772
          },
          {
            "VG1": 0.6,
            "VG2": 0.0,
            "log_rmse": 1.8402102675514411,
            "n_conv": 9,
            "n_pts": 30,
            "vb_max": 0.6502597491774896
          },
          {
            "VG1": 0.6,
            "VG2": 0.05,
            "log_rmse": 1.7799779600679333,
            "n_conv": 9,
            "n_pts": 30,
            "vb_max": 0.6502338247389104
          },
          {
            "VG1": 0.6,
            "VG2": 0.1,
            "log_rmse": 1.7205063498640427,
            "n_conv": 9,
            "n_pts": 30,
            "vb_max": 0.6506898369460821
          },
          {
            "VG1": 0.6,
            "VG2": 0.15,
            "log_rmse": 1.7537357391410728,
            "n_conv": 9,
            "n_pts": 30,
            "vb_max": 0.6830034601476239
          },
          {
            "VG1": 0.6,
            "VG2": 0.2,
            "log_rmse": 1.572622297641623,
            "n_conv": 9,
            "n_pts": 30,
            "vb_max": 0.650855588902394
          },
          {
            "VG1": 0.6,
            "VG2": 0.25,
            "log_rmse": 1.6825791172736035,
            "n_conv": 9,
            "n_pts": 30,
            "vb_max": 0.7059859533245416
          },
          {
            "VG1": 0.6,
            "VG2": 0.3,
            "log_rmse": 1.4984201583712147,
            "n_conv": 9,
            "n_pts": 30,
            "vb_max": 0.6502856761889657
          },
          {
            "VG1": 0.6,
            "VG2": 0.35,
            "log_rmse": 1.4973077024974755,
            "n_conv": 9,
            "n_pts": 30,
            "vb_max": 0.6501819835795064
          },
          {
            "VG1": 0.6,
            "VG2": 0.4,
            "log_rmse": 1.51370956622811,
            "n_conv": 9,
            "n_pts": 30,
            "vb_max": 0.6340485813423294
          },
          {
            "VG1": 0.6,
            "VG2": 0.45,
            "log_rmse": 1.5200942236788408,
            "n_conv": 9,
            "n_pts": 30,
            "vb_max": 0.6342418424232154
          },
          {
            "VG1": 0.6,
            "VG2": 0.5,
            "log_rmse": 1.520472183907716,
            "n_conv": 9,
            "n_pts": 30,
            "vb_max": 0.6342482927824415
          }
        ],
        "per_branch": {
          "VG1_0.4": 0.7034343579866807,
          "VG1_0.6": 1.6319445170904727
        },
        "vb_max_overall": 0.7059859533245416
      },
      "backward": {
        "cell_rmse_dec": 1.0268618987992062,
        "n": 25,
        "fails": 0,
        "wall_sec": 610.2200109958649,
        "per_bias": [
          {
            "VG1": 0.2,
            "VG2": -0.05,
            "log_rmse": 1.4966201322860386,
            "n_conv": 6,
            "n_pts": 30,
            "vb_max": 0.8616779440932819
          },
          {
            "VG1": 0.2,
            "VG2": -0.1,
            "log_rmse": 1.42480559148827,
            "n_conv": 5,
            "n_pts": 30,
            "vb_max": 0.8221362745571222
          },
          {
            "VG1": 0.2,
            "VG2": -0.15,
            "log_rmse": 1.1221798062682968,
            "n_conv": 6,
            "n_pts": 30,
            "vb_max": 0.7783673453883552
          },
          {
            "VG1": 0.2,
            "VG2": -0.2,
            "log_rmse": 0.9499149802233882,
            "n_conv": 6,
            "n_pts": 30,
            "vb_max": 0.8265134610579729
          },
          {
            "VG1": 0.2,
            "VG2": 0.0,
            "log_rmse": 1.482159831610416,
            "n_conv": 7,
            "n_pts": 30,
            "vb_max": 0.7668796816822167
          },
          {
            "VG1": 0.2,
            "VG2": 0.05,
            "log_rmse": 1.4909728571550804,
            "n_conv": 8,
            "n_pts": 30,
            "vb_max": 0.7663054627678797
          },
          {
            "VG1": 0.2,
            "VG2": 0.1,
            "log_rmse": 1.4040828480844794,
            "n_conv": 9,
            "n_pts": 30,
            "vb_max": 0.8117703257866278
          },
          {
            "VG1": 0.4,
            "VG2": 0.0,
            "log_rmse": 0.6669498852002764,
            "n_conv": 20,
            "n_pts": 30,
            "vb_max": 0.7000000000000001
          },
          {
            "VG1": 0.4,
            "VG2": 0.05,
            "log_rmse": 0.6078268636614734,
            "n_conv": 20,
            "n_pts": 30,
            "vb_max": 0.7000000000000001
          },
          {
            "VG1": 0.4,
            "VG2": 0.1,
            "log_rmse": 0.5251232312786994,
            "n_conv": 20,
            "n_pts": 30,
            "vb_max": 0.7000000000000001
          },
          {
            "VG1": 0.4,
            "VG2": 0.15,
            "log_rmse": 0.4764391499541844,
            "n_conv": 20,
            "n_pts": 30,
            "vb_max": 0.6879254740920717
          },
          {
            "VG1": 0.4,
            "VG2": 0.2,
            "log_rmse": 0.4485375733958696,
            "n_conv": 20,
            "n_pts": 30,
            "vb_max": 0.6
          },
          {
            "VG1": 0.4,
            "VG2": 0.25,
            "log_rmse": 0.44362074037302063,
            "n_conv": 21,
            "n_pts": 30,
            "vb_max": 0.6
          },
          {
            "VG1": 0.4,
            "VG2": 0.3,
            "log_rmse": 0.4329395965981923,
            "n_conv": 20,
            "n_pts": 30,
            "vb_max": 0.6
          },
          {
            "VG1": 0.6,
            "VG2": 0.0,
            "log_rmse": 1.2179941198564106,
            "n_conv": 17,
            "n_pts": 30,
            "vb_max": 0.7000000000000001
          },
          {
            "VG1": 0.6,
            "VG2": 0.05,
            "log_rmse": 1.16970403267042,
            "n_conv": 17,
            "n_pts": 30,
            "vb_max": 0.7000000000000001
          },
          {
            "VG1": 0.6,
            "VG2": 0.1,
            "log_rmse": 1.1215482000685422,
            "n_conv": 17,
            "n_pts": 30,
            "vb_max": 0.7000000000000001
          },
          {
            "VG1": 0.6,
            "VG2": 0.15,
            "log_rmse": 1.0107935606662102,
            "n_conv": 16,
            "n_pts": 30,
            "vb_max": 0.7000000000000001
          },
          {
            "VG1": 0.6,
            "VG2": 0.2,
            "log_rmse": 1.0000200886984896,
            "n_conv": 17,
            "n_pts": 30,
            "vb_max": 0.7000000000000001
          },
          {
            "VG1": 0.6,
            "VG2": 0.25,
            "log_rmse": 0.9408486584041135,
            "n_conv": 16,
            "n_pts": 30,
            "vb_max": 0.7000000000000001
          },
          {
            "VG1": 0.6,
            "VG2": 0.3,
            "log_rmse": 0.9373968451238335,
            "n_conv": 17,
            "n_pts": 30,
            "vb_max": 0.7000000000000001
          },
          {
            "VG1": 0.6,
            "VG2": 0.35,
            "log_rmse": 0.9366251394277448,
            "n_conv": 17,
            "n_pts": 30,
            "vb_max": 0.7000000000000001
          },
          {
            "VG1": 0.6,
            "VG2": 0.4,
            "log_rmse": 0.9690228089754666,
            "n_conv": 17,
            "n_pts": 30,
            "vb_max": 0.7000000000000001
          },
          {
            "VG1": 0.6,
            "VG2": 0.45,
            "log_rmse": 0.9809419605648078,
            "n_conv": 17,
            "n_pts": 30,
            "vb_max": 0.7000000000000001
          },
          {
            "VG1": 0.6,
            "VG2": 0.5,
            "log_rmse": 0.9808735929547802,
            "n_conv": 17,
            "n_pts": 30,
            "vb_max": 0.7000000000000001
          }
        ],
        "per_branch": {
          "VG1_0.2": 1.3534514215065598,
          "VG1_0.4": 0.5213233610635837,
          "VG1_0.6": 1.0284972444368647
        },
        "vb_max_overall": 0.8616779440932819
      },
      "avg_cell_rmse_dec": 1.1879617662502984
    },
    "PT_VBIC": {
      "kind": "pt",
      "flags": {
        "use_vbic_for_q1": true,
        "vbic_AVC1": 0.5,
        "vbic_AVC2": 0.5
      },
      "forward": {
        "cell_rmse_dec": 1.3964517427496117,
        "n": 25,
        "fails": 0,
        "wall_sec": 752.3965406417847,
        "per_bias": [
          {
            "VG1": 0.2,
            "VG2": -0.05,
            "log_rmse": 1.0260718484273448,
            "n_conv": 3,
            "n_pts": 30,
            "vb_max": 0.5627618350903223
          },
          {
            "VG1": 0.2,
            "VG2": -0.1,
            "log_rmse": 1.1704376743558282,
            "n_conv": 3,
            "n_pts": 30,
            "vb_max": 0.6255966361369806
          },
          {
            "VG1": 0.2,
            "VG2": -0.15,
            "log_rmse": 1.140755467953402,
            "n_conv": 2,
            "n_pts": 30,
            "vb_max": 0.5855201394757783
          },
          {
            "VG1": 0.2,
            "VG2": -0.2,
            "log_rmse": 1.163196869895531,
            "n_conv": 3,
            "n_pts": 30,
            "vb_max": 0.6276698202073587
          },
          {
            "VG1": 0.2,
            "VG2": 0.0,
            "log_rmse": 1.0996359229989978,
            "n_conv": 4,
            "n_pts": 30,
            "vb_max": 0.5378486167311154
          },
          {
            "VG1": 0.2,
            "VG2": 0.05,
            "log_rmse": 0.8734742544617146,
            "n_conv": 4,
            "n_pts": 30,
            "vb_max": 0.5207522651680613
          },
          {
            "VG1": 0.2,
            "VG2": 0.1,
            "log_rmse": 0.914490805041991,
            "n_conv": 5,
            "n_pts": 30,
            "vb_max": 0.5277178871973311
          },
          {
            "VG1": 0.4,
            "VG2": 0.0,
            "log_rmse": 1.3903314580137722,
            "n_conv": 12,
            "n_pts": 30,
            "vb_max": 0.4947305997971263
          },
          {
            "VG1": 0.4,
            "VG2": 0.05,
            "log_rmse": 1.225674584697732,
            "n_conv": 12,
            "n_pts": 30,
            "vb_max": 0.4915329813485698
          },
          {
            "VG1": 0.4,
            "VG2": 0.1,
            "log_rmse": 0.9869469858100441,
            "n_conv": 12,
            "n_pts": 30,
            "vb_max": 0.49012152009982607
          },
          {
            "VG1": 0.4,
            "VG2": 0.15,
            "log_rmse": 0.8201242568391712,
            "n_conv": 12,
            "n_pts": 30,
            "vb_max": 0.49116702633703935
          },
          {
            "VG1": 0.4,
            "VG2": 0.2,
            "log_rmse": 0.1626311758684523,
            "n_conv": 13,
            "n_pts": 30,
            "vb_max": 0.48943846986383605
          },
          {
            "VG1": 0.4,
            "VG2": 0.25,
            "log_rmse": 0.16031614770948,
            "n_conv": 12,
            "n_pts": 30,
            "vb_max": 0.4877506788961188
          },
          {
            "VG1": 0.4,
            "VG2": 0.3,
            "log_rmse": 0.17280653893934686,
            "n_conv": 13,
            "n_pts": 30,
            "vb_max": 0.4977376815124556
          },
          {
            "VG1": 0.6,
            "VG2": 0.0,
            "log_rmse": 2.0685766191087898,
            "n_conv": 10,
            "n_pts": 30,
            "vb_max": 0.503713660887177
          },
          {
            "VG1": 0.6,
            "VG2": 0.05,
            "log_rmse": 1.9990038394495813,
            "n_conv": 10,
            "n_pts": 30,
            "vb_max": 0.5036579251901557
          },
          {
            "VG1": 0.6,
            "VG2": 0.1,
            "log_rmse": 1.9290131873778358,
            "n_conv": 10,
            "n_pts": 30,
            "vb_max": 0.5061421974943567
          },
          {
            "VG1": 0.6,
            "VG2": 0.15,
            "log_rmse": 1.874640258735047,
            "n_conv": 10,
            "n_pts": 30,
            "vb_max": 0.4706027036301727
          },
          {
            "VG1": 0.6,
            "VG2": 0.2,
            "log_rmse": 1.7496628662427345,
            "n_conv": 10,
            "n_pts": 30,
            "vb_max": 0.5061324516364137
          },
          {
            "VG1": 0.6,
            "VG2": 0.25,
            "log_rmse": 1.780472190960012,
            "n_conv": 10,
            "n_pts": 30,
            "vb_max": 0.4715659769913227
          },
          {
            "VG1": 0.6,
            "VG2": 0.3,
            "log_rmse": 1.6545621043677954,
            "n_conv": 10,
            "n_pts": 30,
            "vb_max": 0.503825147664126
          },
          {
            "VG1": 0.6,
            "VG2": 0.35,
            "log_rmse": 1.6531517447812574,
            "n_conv": 10,
            "n_pts": 30,
            "vb_max": 0.503713660887177
          },
          {
            "VG1": 0.6,
            "VG2": 0.4,
            "log_rmse": 1.676363392835906,
            "n_conv": 10,
            "n_pts": 30,
            "vb_max": 0.5484790982512754
          },
          {
            "VG1": 0.6,
            "VG2": 0.45,
            "log_rmse": 1.6849550410852956,
            "n_conv": 10,
            "n_pts": 30,
            "vb_max": 0.5363036537951157
          },
          {
            "VG1": 0.6,
            "VG2": 0.5,
            "log_rmse": 1.6854850455012624,
            "n_conv": 10,
            "n_pts": 30,
            "vb_max": 0.5359146677410217
          }
        ],
        "per_branch": {
          "VG1_0.2": 1.0613702499498727,
          "VG1_0.4": 0.8589013053774117,
          "VG1_0.6": 1.8015752936781584
        },
        "vb_max_overall": 0.6276698202073587
      },
      "backward": {
        "cell_rmse_dec": 1.1556908486330506,
        "n": 25,
        "fails": 0,
        "wall_sec": 636.3331356048584,
        "per_bias": [
          {
            "VG1": 0.2,
            "VG2": -0.05,
            "log_rmse": 0.8019034655788825,
            "n_conv": 7,
            "n_pts": 30,
            "vb_max": 0.5616770270621212
          },
          {
            "VG1": 0.2,
            "VG2": -0.1,
            "log_rmse": 0.8763587883602828,
            "n_conv": 8,
            "n_pts": 30,
            "vb_max": 0.6221351650879109
          },
          {
            "VG1": 0.2,
            "VG2": -0.15,
            "log_rmse": 0.8294799823852335,
            "n_conv": 7,
            "n_pts": 30,
            "vb_max": 0.5783661972940104
          },
          {
            "VG1": 0.2,
            "VG2": -0.2,
            "log_rmse": 0.9430210537040529,
            "n_conv": 7,
            "n_pts": 30,
            "vb_max": 0.6265132888502717
          },
          {
            "VG1": 0.2,
            "VG2": 0.0,
            "log_rmse": 0.888428860917711,
            "n_conv": 8,
            "n_pts": 30,
            "vb_max": 0.6055876680137436
          },
          {
            "VG1": 0.2,
            "VG2": 0.05,
            "log_rmse": 0.7312161219425032,
            "n_conv": 9,
            "n_pts": 30,
            "vb_max": 0.5661290484565056
          },
          {
            "VG1": 0.2,
            "VG2": 0.1,
            "log_rmse": 0.7930033058116612,
            "n_conv": 9,
            "n_pts": 30,
            "vb_max": 0.537255690075619
          },
          {
            "VG1": 0.4,
            "VG2": 0.0,
            "log_rmse": 1.0800549560513586,
            "n_conv": 21,
            "n_pts": 30,
            "vb_max": 0.49999999999999994
          },
          {
            "VG1": 0.4,
            "VG2": 0.05,
            "log_rmse": 0.959780906889141,
            "n_conv": 21,
            "n_pts": 30,
            "vb_max": 0.49999999999999994
          },
          {
            "VG1": 0.4,
            "VG2": 0.1,
            "log_rmse": 0.78742521211563,
            "n_conv": 21,
            "n_pts": 30,
            "vb_max": 0.49999999999999994
          },
          {
            "VG1": 0.4,
            "VG2": 0.15,
            "log_rmse": 0.6700910223372958,
            "n_conv": 21,
            "n_pts": 30,
            "vb_max": 0.49999999999999994
          },
          {
            "VG1": 0.4,
            "VG2": 0.2,
            "log_rmse": 0.2752374677254606,
            "n_conv": 21,
            "n_pts": 30,
            "vb_max": 0.49999999999999994
          },
          {
            "VG1": 0.4,
            "VG2": 0.25,
            "log_rmse": 0.28483759186582697,
            "n_conv": 21,
            "n_pts": 30,
            "vb_max": 0.49999999999999994
          },
          {
            "VG1": 0.4,
            "VG2": 0.3,
            "log_rmse": 0.2820342095734921,
            "n_conv": 21,
            "n_pts": 30,
            "vb_max": 0.49999999999999994
          },
          {
            "VG1": 0.6,
            "VG2": 0.0,
            "log_rmse": 1.7081821095493954,
            "n_conv": 17,
            "n_pts": 30,
            "vb_max": 0.49999999999999994
          },
          {
            "VG1": 0.6,
            "VG2": 0.05,
            "log_rmse": 1.658795333666035,
            "n_conv": 17,
            "n_pts": 30,
            "vb_max": 0.49999999999999994
          },
          {
            "VG1": 0.6,
            "VG2": 0.1,
            "log_rmse": 1.6093023341000352,
            "n_conv": 17,
            "n_pts": 30,
            "vb_max": 0.49999999999999994
          },
          {
            "VG1": 0.6,
            "VG2": 0.15,
            "log_rmse": 1.4956890356053993,
            "n_conv": 16,
            "n_pts": 30,
            "vb_max": 0.49999999999999994
          },
          {
            "VG1": 0.6,
            "VG2": 0.2,
            "log_rmse": 1.483769966033678,
            "n_conv": 17,
            "n_pts": 30,
            "vb_max": 0.49999999999999994
          },
          {
            "VG1": 0.6,
            "VG2": 0.25,
            "log_rmse": 1.4224634391103141,
            "n_conv": 16,
            "n_pts": 30,
            "vb_max": 0.49999999999999994
          },
          {
            "VG1": 0.6,
            "VG2": 0.3,
            "log_rmse": 1.417899986031581,
            "n_conv": 17,
            "n_pts": 30,
            "vb_max": 0.49999999999999994
          },
          {
            "VG1": 0.6,
            "VG2": 0.35,
            "log_rmse": 1.416995963621209,
            "n_conv": 17,
            "n_pts": 30,
            "vb_max": 0.49999999999999994
          },
          {
            "VG1": 0.6,
            "VG2": 0.4,
            "log_rmse": 1.4442917243362,
            "n_conv": 17,
            "n_pts": 30,
            "vb_max": 0.49999999999999994
          },
          {
            "VG1": 0.6,
            "VG2": 0.45,
            "log_rmse": 1.4543807514201357,
            "n_conv": 17,
            "n_pts": 30,
            "vb_max": 0.49999999999999994
          },
          {
            "VG1": 0.6,
            "VG2": 0.5,
            "log_rmse": 1.4543739330938517,
            "n_conv": 17,
            "n_pts": 30,
            "vb_max": 0.49999999999999994
          }
        ],
        "per_branch": {
          "VG1_0.2": 0.8401714743243983,
          "VG1_0.4": 0.6962337408167731,
          "VG1_0.6": 1.5092480591194308
        },
        "vb_max_overall": 0.6265132888502717
      },
      "avg_cell_rmse_dec": 1.2760712956913312
    }
  },
  "wall_sec_total": 2857.3
}
```
