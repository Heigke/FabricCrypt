0.965 dec fit = sub-threshold curve-fitting only, NOT BJT fold physics.
→ R-49 dbd-avalanche directly targets this. The fit-quality breakthrough
   is illusory until R-49 reproduces the 2-3 dec fold jump.

## 2026-05-15 01:00 — z373 LIF demo (HONEST CRITICAL FINDING #2)
Cells spike (325Hz, 100%) but AUTONOMOUS, NOT INPUT-DRIVEN.
W_in=0 control: 332.9 Hz; Poisson 200 Hz drive: 325.2 Hz (input SLOWS slightly).
LUT Inet ≈ 170pA nearly constant over Vd[0.25,1.5V] at tested VG2.
CV(ISI)=0.04 (delta at 3ms refractory) — NOT Poisson-like LIF.
DS-N10 sine class 97.8% MAY have used different VG2 regime — needs audit.
Cells are intrinsic oscillators in this LUT regime. Input-coupling requires VG2 in [0.55,0.65].

## 2026-05-15 01:25 — R-49 BUGFIX
Bug: avalanche seed was m1["Ibd"] (~1e-18A reverse bias) → M×Ibd≈0 → no effect.
Fix: seed = m1["Ids"] (actual channel current, BSIM4 IIMOD standard).
Restart PID 2182783. Pre-registered gate: if any BV gives perVG1 cell < 0.85 dec
AND VG1=0.20 fold > 1 dec → R-49 PASS (real snapback physics). Else: another retract.

## 2026-05-15 01:35 — z374 UCI-HAR smoke test (CRITICAL WARNING)
N=1024 seed=0: linear_ridge_raw=0.962, rp_ridge=0.962, nsram_vdbit=0.760.
NS-RAM HDC 83.86% locked claim is 20pp BELOW trivial sklearn linear ridge.
UCI-HAR is too easy — wrong benchmark for proving NS-RAM advantage.
V_d-as-bit modulation IS load-bearing (ablation collapses to 16.8%).
Full n=10 seeds running but trajectory clear: NS-RAM not winning this.
Need harder benchmark where linear methods fail (high-D, nonlinear, temporal).
Candidates: DVS-Gesture, full GoogleSpeech, MIT-BIH 48-record ECG.

## 2026-05-15 01:45 — all-night cron tick (z332/3/4 stale)
Active: z368 R-49 bugfixed (PID 2182786, 12min, 99.9%), z365 (121/240 evals min 0.965),
z370 phys-BBO (1h42m), z374 UCI-HAR pubgrade on daedalus (6min).
2 honest negatives ikväll: snapback fold missing, UCI-HAR baselines crush NS-RAM at trivial 96.2%.
APU=50°C OK.
ACTIVE=claude-08d7-cwd APU=50C sentinel=alive
## 2026-05-15 01:47 — :47 tick ACTIVE: scripts/z365_perVG1_bbo.py, APU=51C, sentinel=alive

## 2026-05-15 01:50 — all-night tick (stale refs)
z332/z333/z334 not in current plan. Current: z368 R-49 bugfixed running.

## 2026-05-15 02:00 — R-49 BUGFIXED COMPLETE — FAILED
z368 28min. Best global BV=6.0V: cell=1.1907 (vs baseline 1.1915, Δ=-0.0008 noise).
Best perVG1 BV=6.0V: cell=0.9601 (vs 0.9650, Δ=-0.005 noise).
ALL 5 physics gates FAIL. BV chose unphysical 6.0V edge.
VG1=0.60 fold unchanged. Avalanche-into-body topology insufficient.
3rd consecutive failed topology fix (R-47 subdiode, R-49 dbd, both fail).
NEXT: snapback fold MAY require fundamentally different physics — not body
injection but lateral NPN parasitic in different orientation.
Or: a Multiplicative TERM on Ids itself, not body injection.

## 2026-05-15 02:13 — topology rebuild tick
ALERT: R-49 (dbd avalanche, bugfixed) just CROSSED gate as FAIL (all 5 physics gates).
R-50 phys-BBO still running. 3 consecutive topology rewrites failed (R-43/R-47/R-49).
Next R-phase NOT dispatched — needs human design decision: lateral NPN reorientation
or multiplicative Ids-rather-than-body-injection. NO auto-launch.
## 2026-05-15 02:15 — all-night tick (stale)
z332/3/4 obsolete. R-49 FAILED 02:00. Waiting on z365 (BBO), z370, z374 daedalus.
## 2026-05-15 02:47 — :47 tick ACTIVE: scripts/z365_perVG1_bbo.py, APU=49C, sentinel=alive

## 2026-05-15 02:50 — 2h deep-dive tick
APU=49°C. 4A/4B/4C all completed. 4D oracle O68 dispatched 23:11 (in flight). 4E v4.4 done (#212).
No new dispatch — R-49 FAILED 02:00; needs human design decision.
Active: z365 (121+/240 evals min 0.965), z370, z374 daedalus UCI-HAR.

## 2026-05-15 02:53 — master fix campaign tick
Active P-phase: model topology rebuild. R-49 FAILED gate 02:00.
3 consecutive failed (R-43/R-47/R-49). NEXT not dispatched (human design needed).
Surrogate-side: z365 BBO 121/240 evals at 0.965 floor. z374 UCI-HAR daedalus running.
ALERT: snapback fold remains unreproduced; "0.965 fit" is sub-threshold curve-fit only.
## 2026-05-15 02:55 — all-night tick (stale z332/3/4)
## 2026-05-15 03:13 — topology tick: R-49 FAILED (logged 02:00). Next R-phase awaits human design. z365/z370/z374 still running.
## 2026-05-15 03:21 — 3h MEP+DS-N tick: ACTIVE ikaros=scripts/z365_perVG1_bbo.py, daedalus=PID=$!, APU=48C, sentinel=alive

## 2026-05-15 03:23 — 6h track audit
Phase A MEP: A.3 done, A.1/A.4 pending, A.2 in_progress. 1/4 done.
Phase B novel DS: 10+ DS-N completed (5 surv, 5 retract). DS-N4/N5 still in_progress.
Phase C oracle: O67 in_progress, O68 dispatched 23:11. Brief v4.4 done.
Topology rebuild: 3 consecutive fails (R-43/R-47/R-49). Snapback unreproduced.
Surviving claims: DS-N10, N11, N14, N15, N16. UCI-HAR z374 running (digital crushes).
## 2026-05-15 03:37 — all-night stale tick (z332/3/4 obsolete; z365/z370/z374 still running)
## 2026-05-15 03:47 — :47 tick ACTIVE: scripts/z365_perVG1_bbo.py APU=48C sentinel=alive
## 2026-05-15 04:07 — all-night stale tick
## 2026-05-15 04:13 — topology tick: R-49 FAILED. Next phase awaits human design. z365/z370/z374 active.

## 2026-05-15 04:20 — OVERNIGHT PLAN dispatched (4 subagents parallel)
T1: R-52/53/54 fundamental topology variants (ikaros, sequential, 90min)
T1b: R-55 Zoom folder + Mario/Sebas deep dive (subagent, 30min)
T2: z375 DVS-Gesture 128x128 production on daedalus GPU (3h)
T3: O69 oracle aggressive falsification (15min)
T4: continues — z365 BBO, z370 phys-BBO, z374 UCI-HAR daedalus
Plan: research_plan/OVERNIGHT_PLAN_2026-05-15.md
NO-CHEAT: pre-registered gates, retract on KILL-SHOT.

## 2026-05-15 04:30 — R-55 BREAKTHROUGH FINDING
5 missing elements found in nsram_zenodo Sebas reference:
1. D3 zener G1→B missing (we attacked body↔drain path only)
2. BVPar = 3.5 − 1.5·V_G1 (we use constant nbv=7) — controls fold sharpness
3. M3 BSS145 body-bias G2→B (we fitted with NFACTOR polynomial)
4. BJT params mismatch: VA=100 (us 0.55, 180× too low!), IS=5e-9 (us 1e-9), Bf=10000 (us 9000)
5. etab sign-flip M1(+1.8) vs M2(-0.087) — textbook DIBL collapse snapback

R-55a: port full NeuronSubCirc.asc topology (D3 + M3 + corrected Q1).
Existing 3 failed fixes attacked the wrong path. New direction unlocked.

## 2026-05-14 22:40 — O69 oracle synthesis (openai+gemini+grok)
PKT: research_plan/oracle_queries/O69_falsification_20260514_223516/
Q1 ROOT CAUSE — 3/3 agree (a) MISSING PHYSICS, specifically: regenerative loop between impact-ionization Mii(Vds,T) and floating-body Vbe NOT closed. OpenAI: "you never closed the actual snapback loop … Add the damn lateral NPN with Mii(Vds,T) and Rb". Gemini: "I_impact = (M-1)*I_ds … positive feedback. Latch-up. Your R-49 multiplies a tiny reverse-bias diode current. Physics multiplies the MAIN CHANNEL current." Grok: "explicit Qb dynamics … ΔV_th = q·ΔQb/Cox".
Q1 KILL-EXPT — converge on body-strap test: clamp Body→Source via low-R shunt (or grounded body contact). Fold collapses → confirms missing NPN/kink; fold persists → simulator bug or measurement artifact (retract snapback claim).
Q2 BENCHMARK — 3/3 name the SAME three datasets: DVS128 Gesture (tonic), MIT-BIH Arrhythmia (wfdb/physionet), Google Speech Commands (torchaudio). Linear <80%, NS-RAM primitives (threshold + multi-τ leak + body memory) map natively.
Q3 KILL-SHOT — convergence on DVS128 Gesture as the program-level test. Gemini: NS-RAM reservoir must ≥ digital LIF reservoir of same size. OpenAI: ≥85% acc AND ≥5× energy advantage over int8 MCU baseline. Grok: ≥85% + <10% energy of MobileNet; else "full retraction".
RETRACTION VERDICT — Gemini: "not a research program; salvage operation … 24-hour window is not for finding a new direction, it's for writing the retraction." Grok: "Retract now — your 'NS-RAM substrate is useful' is already falsified horseshit". OpenAI: not yet — gives one-shot kill via NPN+Mii fix THEN kill-shot benchmark.
ACTIONABLE (overnight): (1) "Cheater" multiplicative test — `I_body += gain*Ids`, sweep gain 1e-5..1e-1; if any value reproduces 2-3 dec fold → topology confirmed, implement proper Mii(Vds,T). (2) Body-strap re-measurement on Sebas data (or check if any existing trace has body-grounded variant). (3) Download DVS128 + MIT-BIH tonight; pre-register kill-shot before any tuning.
## 2026-05-15 04:37 — all-night stale tick (R-55 finding logged 04:30)

## 2026-05-15 04:43 — O69 ORACLE CONVERGENCE (3/3 agree)
Root cause: regenerative IIM × floating-body loop NEVER CLOSED.
R-49 wrong direction confirmed: should multiply Ids, not reverse-bias diode.
Kill-shot benchmark: DVS128 Gesture (unanimous). Currently running z375.
Retract-leaning: 2/3 (Gemini bluntest), OpenAI gives final R-52+R-55a window.
Two independent sources (R-55 Zoom + O69) converge on: M(V_db)·Ids + VA=100 + D3 zener.
R-52 implementation now exactly matches what oracles prescribe.
## 2026-05-15 04:47 — :47 tick ACTIVE: scripts/z365_perVG1_bbo.py APU=50C sentinel=alive
## 2026-05-15 04:53 — 2h deep-dive tick: 4A/B/C done, 4D O69 done 04:43, 4E v4.4 done. No new dispatch — R-52/53/54 + DVS-Gesture in flight.
## 2026-05-15 05:07 — all-night stale (z332/3/4 obsolete; overnight plan running R-52/3/4, R-55a pending, z375 DVS-Gesture)

## 2026-05-15 05:13 — topology tick
ACTIVE: R-52 (M·Ids multiplicative) running on ikaros via subagent.
R-49 FAILED (logged 02:00). R-55 found 5 missing physics elements 04:30.
O69 oracle CONVERGENCE 04:43: R-52 is exact right fix per all 3 oracles.
DVS-Gesture kill-shot benchmark running on daedalus.
No new dispatch — waiting on R-52 and z375 results.

## 2026-05-14 23:14 — R-52 / R-53 / R-54 topology variants COMPLETE (z375)
Script: scripts/z375_topology_R52_R53_R54.py — three new variants, eval on 33-curve fit with R-46 per-VG1 best params (eval 94 of z365 BBO).

**R-52 (M(V_db) directly on Ids_M1, multiplicative)**
  Sweep bv_ids ∈ {6,8,10,12,15} V, n_ids=4. INFRA=PASS, DISCOVERY=PASS (barely), AMBITIOUS=FAIL.
  Best: bv_ids=6V → cell-wide=0.96493 dec (vs baseline 0.96501). Δ = −0.00008 dec.
  per_VG1 at best: {0.20:1.7744, 0.40:1.1613, 0.60:0.8625}. fold06=1.6125.
  VERDICT: variant has near-zero effect. Multiplying Ids by M(V_db) at the residual stage gets re-equilibrated by Vb shifting in the Newton loop — body voltage adapts so net Ids ≈ same.

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
