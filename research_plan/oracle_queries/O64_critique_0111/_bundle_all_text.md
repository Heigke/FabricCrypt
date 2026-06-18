# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: context.md (12537 chars) ===
```
  Igidl_M1: ng=1.96e-24, py=1.96e-20 (4 dec too LARGE)
  Ib_Q1 SIGN-FLIPPED

ROOT CAUSE: BJT Vbe wired wrong.
  ngspice deck: Q1 vsint vb 0 (collector=Sint, EMITTER=GND) → Vbe=Vb=0.267
  pyport: Vbe = Vb - Vsint = -0.115 (REVERSE biased → OFF)
  D1 was applied to emitter wire but Ic_Q1 compute still uses Vbe=Vb-Vsint.
  Fix: cfg.bjt_emitter_to_gnd=True OR rewrite Vbe = Vb - 0 in BJT term.

R-20 dispatched to verify + disable eta_sigmoid choke.
This is the structural KCL bug R-17 oracles flagged.

## 2026-05-13 23:13 cron — R-phase status
Active: R-13b (z332 conv=False, structural blocker explains), R-19 (probe+M2.B audit),
R-20 (BJT Vbe + eta_sigmoid disable). R-3..R-7 of TOPOLOGY_REBUILD_PLAN superseded —
real blocker is BJT Vbe miswire (not param recipe). ALERT: R-21 fix-and-refit not yet
dispatched; waiting on R-20 to confirm Vbe rewrite raises Ic by ~6 OoM before refit.

## 2026-05-13 23:18 — R-10/z326 returned (solver-only insufficient)
S1/S2 FAIL infra; S3 (body_pdiode_to=gnd) PASS infra (Vb 2.0→1.969) but
cell-wide 3.43 dec (worse than z304's 0.99). Pulling Vb down trades
trivial-undershoot for activated-overshoot (+4 dec at VG1=0.2). Confirms
R-15/R-18/R-17: need PHYSICS fix (BJT Vbe miswire), not solver re-init.

## 2026-05-13 23:25 — R-21/z337 BJT-fix refit DONE: median 4.16 dec (worse, expected)

cfg.bjt_emitter_to_gnd=True + Vbc=Vb-Vsint patch in nsram_cell_2T.py L530.
33/33 curves valid, no NaN, no crashes.
  cell-wide median: 4.155 dec
  per VG1: 0.20→3.26  0.40→3.88  0.60→5.51
  baselines: z304=0.99 z313=3.01 z326=3.43 z334=7.05
Pattern matches z326 prediction: fixing BJT trades trivial-undershoot for
activated-overshoot (β=10914 now fires hard, but alpha0/Bf/VAF/lat_BV
were tuned for the BROKEN BJT). Next: R-22 BBO refit of (alpha0, Bf, VAF,
lat_BV, body_pdiode_Rs) WITH bjt_emitter_to_gnd=True locked.

## 2026-05-13 22:43 cron — z338 BBO running 14/60 evals
Best so far cost=3.46 (eval 4: a0=3.2e-5 Bf=950 Va=0.48 Is=1.3e-9 BV=4.28 Rs=5.4e9)
on 9-bias subset. All evals 3.4-4.9 — landscape stays away from sub-1.0
dec basin so far. ETA ~30-40min. z332/z333 completed earlier; gate paths
done. No new dispatch needed; let DE finish then assess.
22:50 idle-check: ACTIVE: z338_bjt_fix_bbo APU=44C sentinel=alive

## 2026-05-13 22:59 deep-dive cron — DEFER 4D (model under repair)
APU=44C OK. 4A/4B/4C tasks all marked completed (#190/191/192) — workflow
would trigger 4D oracle wave. DEFER: since R-15/R-17/R-18/R-20 found
BJT Vbe miswire (z337 4.16 dec post-fix, z338 BBO 19/60 evals best
3.46), critiquing pre-fix benchmark results would be stale. 4D will
run AFTER R-22 BBO yields cell-wide dec < 0.95 PASS gate.

## 2026-05-13 23:13 cron — R-22 z338 36/60 evals
DE landscape floor sticks ~3.43 dec (eval 21 best). 5-param BBO (a0, Bf,
Va, Is, BV, Rs) not finding sub-1.0 basin despite BJT-fix applied.
Suggests additional structural blocker beyond BJT topology. ETA ~20min.
z332/z333/z334 all closed via R-13b/R-15/R-18; gate paths done.
No new dispatch — let DE finish before next physics probe (eta_sigmoid
disable, Iii path post-fix, lat_BV true off, etc).

## 2026-05-13 23:18 cron — R-22 active (z338 DE)
38/60 evals, floor 3.43 dec (cost from eval 21). No gate cross yet.
R-1..R-10 superseded by R-13..R-22 (BJT topology fix arc). R-22 PASS
gate (cell-med <0.95) NOT crossed. ALERT: post-R-22 path requires
new physics probe (eta_sigmoid disable + Iii routing post-fix).

## 2026-05-13 23:43 cron — R-22 z338 eval 62/~60 budget
DE polish phase exceeded nominal 60-eval budget; best stays 3.425 (eval 21).
Process at 1h13m, timeout 5400s = 30min headroom. z332/z333/z334 closed.
No fresh dispatch. After DE returns: apply best params, refit 33 curves,
log z338 final dec. If still >0.95, next probe: eta_sigmoid full disable +
Iii path post-fix audit (R-23).
23:50 idle-check: ACTIVE: scripts/z338_bjt_fix_bbo.py APU=44C sentinel=2

## 2026-05-13 23:58 — R-22 z338 BBO timeout-killed at eval 76

Best cost = 3.425 dec (eval 21): a0=1.6e-5, Bf=2605, Va=0.36, Is=3.3e-10,
BV=4.02, Rs=5.5e6. Top 5 all 3.42-3.55 — landscape floor is REAL.
5-param BBO + bjt_emitter_to_gnd=True + Vbc=Vb-Vsint cannot recover z304's
0.99 dec. STRUCTURAL DIFFERENCE between z304 cfg and z338 cfg is the bug.
Candidates: use_well_diode, body_pdiode_to, eta_sigmoid, m2_body_gnd,
use_lateral_collector, vnwell_Rs. R-23: cfg-diff audit + selective revert.

## 2026-05-14 00:13 cron — R-23 z339 cfg-diff + O63 oracle dispatch in flight
z339: 6 cfg variants × 33 curves running on ikaros. O63 oracle 3-way
(openai+gemini+grok) on 12h gap-closing review — openai uploading now.
No new dispatch. Awaiting both to land before next physics probe.

## 2026-05-14 00:18 cron — R-23 (z339) variant A_baseline running
O63 oracle dispatch in flight on openai still. R-1..R-10 superseded by
R-13..R-23 arc. No gate crossed; R-23 will declare which cfg flag matters.

## 2026-05-14 00:30 — O63 oracle 3-way UNANIMOUS

Q1 (z304 = 0.99 spurious?): YES (3/3). Compensation for broken BJT.
  → goal shift: stop chasing z304, find new sub-1 basin with correct BJT.
Q2 (structural blocker at 3.43 floor): eta_sigmoid (3/3 PRIMARY).
  Secondary: use_lateral_collector (gpt-5+grok), body_pdiode_to (gpt-5+grok).
Q3 (highest-value experiment):
  gpt-5: ngspice OP handover (separate solver from physics)
  gemini: eta_sigmoid disable + 20-eval BBO (surgical)
  grok: cfg-diff R-23 (in flight covers all candidates) - PRIORITY 1
R-23 z339 already covers eta_sigmoid disable variant — good alignment.

## 2026-05-14 01:00 — R-23 z339 cfg-diff DEFINITIVE

Ranking (best→worst) at z338 best params:
  A_baseline_z338        4.447 dec  ← BEST (current cfg)
  C_eta_sigmoid_on       4.579 dec  (Q2 oracle prediction FALSIFIED)
  D_m2_body_vb           5.022 dec
  E_pdiode_gnd           5.779 dec
  F_all_revert           6.229 dec
  B_emitter_default      6.238 dec  (reverting BJT fix → +1.8 dec)

KEY FINDINGS:
1. BJT fix (R-20) is correct — reverting costs 1.8 dec
2. eta_sigmoid OFF (z338 default) is BETTER than ON (against oracle Q2)
3. NO cfg toggle breaks 3.4 dec plateau
4. Plateau is structural — beyond current pyport architecture
5. z304's 0.99 confirmed spurious (per O63 Q1) — true sub-1 basin needs
   either deeper physics rewrite OR ngspice handover.

Next: R-24 ngspice OP handover test (gpt-5's Q3 #1) — separate
solver/basin from physics. If pyport with ngspice OPs still ≥2 dec,
proves arch limit. Defer until human input on rewrite scope.

## 2026-05-14 01:10 MEP+DS-N cron — model-fix arc active, novel-DS deferred
APU=42C  sentinel=2w  ACTIVE: z340_ngspice_handover (R-24 launching)
NOVEL_DS_PLAN Phase A/B steps DEFERRED while R-13..R-24 resolves BJT
plateau diagnosis. Thermal OK. No new DS-N launches.

## 2026-05-14 01:14 — R-24 z340 ngspice handover: REWRITE REQUIRED

Pyport evaluated at EXACT ngspice OPs (no solving):
  Line A (eta_sigmoid ON):  4.116 dec
  Line B (eta_sigmoid OFF): 4.116 dec  (identical — eta_sigmoid is no-op)
  Per-VG1: 0.20→1.82  0.40→3.98  0.60→5.48 dec

VERDICT per gpt-5 O63 rules: both ≥2.0 → architecture missing physics.
Solver/basin NOT the blocker. Pyport _residuals topology is wrong.

Error scales monotonically with VG1 — high-gate-overdrive regime where
M1 turns on hard. R-25: per-term component decomposition at VG1=0.60
to identify which path (Ids_M1, Ic_Q1, Ic_lat, Ic_avalanche, Igidl,
Ibd_M1) is wrong by how many dec.

## 2026-05-14 02:00 — R-25 z341 component decomp: PYPORT IS NOT THE OFFENDER

At VG1=0.60, VG2=0.20, 10 Vd points, forced-node ngspice OP vs pyport
_residuals at same (Vsint, Vb):

  Total Id agreement py↔ng: ≤0.24 dec at ALL points
  Both py and ng vs measured: -4.8 to -6.5 dec (missing 5 decades)
  Per-term py↔ng: Ic_Q1=0.00, Ib_Q1=0.02, Ids_M1=3.5 (but Ids_M1≪Ic_Q1)
  Best single-term swap (Ids_M1 → ngspice): +0.19 dec only

KEY: ngspice `@m1[isub]` = 0 at every Vd ⇒ BSIM4 card Iii is OFF / dead
at L=0.13 µm sizes. With Iii=0, Vb only gets the well-diode drive
(~0.26 V), giving Ic_Q1 ~1e-11. Reality needs Vb→0.6 V → Ic ~1e-5.

VERDICT (R-25): pyport architecture is NOT the blocker — neither is the
solver. The IIMOD path in M1_130DNWFB.txt is dead. R-26 priority: probe
the M1 BSIM4 card. Enable/raise alpha0,alpha1,beta0 (IIMOD>=1) and
verify ngspice `@m1[isub]` becomes non-zero in saturation. If card has
IIMOD=0, this explains 5 dec of "pyport plateau" — it was never pyport.

Files: results/z341_component_decomp/{per_term_diff.json, verdict.md,
decomp_plot_VG1_0.60_VG2_0.20.png}

## 2026-05-14 01:30 — R-25 z341 OVERTHROWS R-24: Sebas card IIMOD bug

Per-term diff at VG1=0.60 ngspice OPs:
  Pyport TOTAL Id matches ngspice TOTAL Id to ≤0.24 dec EVERY Vd.
  Both 4.8-6.5 dec BELOW silicon. PYPORT ARCH IS NOT THE BUG.

ngspice @m1[isub] = 0 at every Vd → BSIM4 IIMOD inactive at L=0.13µm
With Iii=0 → Vb only 0.26V → Q1 saturates 1e-11 vs silicon 4e-5 (5 dec gap)

R-22 BBO swept alpha0 [7.84e-7, 7.84e-2] (max 1000× default) = Iii 5e-14.
Need 10⁻⁵ A → 9 dec MORE. Search space was way too narrow.

R-26: aggressive ALPHA0/ALPHA1/BETA0 sweep + check if @m1[isub] activates
at higher params + check what other IIMOD knobs the M1 card needs.
The "architecture rewrite" was a false alarm — fix the card.

## 2026-05-14 01:42 — R-26 z342 ROOT CAUSE: lalpha0 length cancellation

M1 card values (data/sebas_2026_04_22/M1_130DNWFB.txt):
  alpha0=7.84e-5, lalpha0=-9.84e-12, alpha1=0, beta0=19, lbeta0=-9.5e-7
At L=0.13µm: lalpha0 cancels 97% → effective alpha0 ≈ 2.7e-6 (not 7.84e-5)

BSIM4 v4.5: no IIMOD level/bjtoff flag. Only the length-cancellation.
Sebas's card was extracted at long-channel; min-geometry kills IIMOD.

Sweep ngspice with patched card at VG1=0.6, VG2=0.20, Vd=2.0:
  default:                  Vb=0     @m1[isub]=0       Id=2.07e-12
  lalpha0=0:                Vb=0.003 @m1[isub]=3.18e-12 Id=2.07e-12
  alpha0×10 + lalpha0=0:    Vb=0.790 @m1[isub]=4.16e-7  Id=2.63e-8 ✓
  silicon target:                                          Id=4.05e-5

Recommendation: set lalpha0=0, alpha0=7.84e-4 in M1 card → +3 dec recovery.
Remaining 2-3 dec = M2 series + Q1 Bf saturation (next phase).

## 2026-05-14 00:48 cron — R-27/z343 IIMOD-fix refit in flight
Patched M1 card (lalpha0=0, alpha0×10) applied. Pyport cell-wide refit
running with bjt_emitter_to_gnd=True + z338 best Bf params. Expected
+3 dec recovery (R-26 prediction). z332/z333/z334 closed. No new
dispatch. Awaiting z343 median_dec to compare vs z337 (4.16) baseline.

## 2026-05-14 01:08 — R-27 z343: 3.99 dec (only -0.17 vs z337 baseline)

Patched M1 card (lalpha0=0, alpha0=7.84e-4) did NOT flow through pyport.
Per-VG1: 0.20→3.10  0.40→3.87  0.60→5.77 dec. Pattern unchanged.

Possible causes (subagent flagged):
1. pyport size-dep scaling silently undoes lalpha0=0 patch
2. body_pdiode_Rs=5.48e6 bleeds Iii before Vb lifts
3. Q1 Va=0.36 saturates collector
4. M2 card still has lalpha0 cancellation

R-28: instrument pyport Iii_M1 at flagship bias with patched card →
verify alpha0_eff. Then sweep body_pdiode_Rs and patch M2 card too.
00:50 idle-check: APU=42C sentinel=2w

## 2026-05-14 01:18 deep-dive cron — DEFER 4D (model card patch arc active)
APU=42C OK. 4A/4B/4C tasks completed (#190-192). Workflow → 4D oracle wave.
DEFER: R-26 found Sebas card IIMOD bug (lalpha0 length cancellation),
R-27 patch attempt 3.99 dec, R-28 patch-flow verify in flight.
4D will run AFTER model card arc resolves to sub-1 dec.

## 2026-05-14 01:18 cron — MASTER_FIX_PLAN P-phase status
P1-P7 superseded by R-13..R-28 arc (BJT topology + card patch).
P3 pyport_v4 gate (DC<0.7 dec): NOT crossed. Current best 3.43 (z338).
R-28 patch-flow verify + M2 card + Rs sweep in flight. No P-gate crossing.
ALERT: if R-28 yields sub-1.0 dec, log "P3 PASS triggered" before P4 dispatch.

## 2026-05-14 01:35 — R-28 patch-flow VERIFIED + Rs sweep partial

R-28A: pyport DOES see patch. Iii went 5.54e-17 → 5.54e-16 (exactly ×10
matching alpha0×10). alpha0_eff_pyport=7.84e-4 ✓ confirmed.

BUT at OP (Vsint=0.382, Vb=0.267, Vd=2.0): Ids_M1=1.2e-14 (subthreshold
regime). Iii ∝ Idsa_Vdseff. Iii stays tiny because PYPORT's BSIM4 channel
gives Ids 3 dec below ngspice (R-25 finding: py=4e-15 vs ng=1.5e-11).

R-28C dual_card_refit (process died at Rs=1e9): Rs=1e8 → 4.61 dec
(WORSE than z343 Rs=5.5e6 → 3.99). Raising Rs hurts.

REAL upstream bug: pyport BSIM4 Ids_M1 is 3 dec too low. Card patch
correct but downstream of channel. R-29: probe pyport BSIM4 channel
evaluator vs ngspice at same OP. Likely an Idsa_Vdseff calculation
discrepancy (Vdseff effective Vds smoothing).

```
