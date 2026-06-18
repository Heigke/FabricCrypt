# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: context.md (10512 chars) ===
```
No shared WARNING. Regular push.

## 2026-05-11 hourly check-in :17 — idle, launched small-N sanity (O40 follow-up)
APU 41°C, sentinel alive PID 9161, no z-script running. Launched z251
(NS-RAM vs ESN at N ∈ {30, 50, 100}). O40 non-blocking follow-up to
test if small-N regime has any NS-RAM niche the matrix missed.

## 2026-05-11 z251 small-N sanity: NO NICHE
N=30: NS-RAM 0.733 vs ESN 0.585 (ESN). N=50: 0.696 vs 0.576 (ESN).
N=100: 0.693 vs 0.572 (ESN). 0/3 strict NS-RAM wins. Grok's small-N
niche hypothesis falsified. Matrix now 14 cells: 0 NS-RAM wins, 11
ESN wins, 3 ties. Last residual O40 follow-up closed. Pushing.

## 2026-05-11 resource audit 00:03 — all GREEN
disk 49% (3.7TB total), mem 3.8/31GB (12%), APU 41°C, sentinel PID 9161
alive, results 52GB stable (no growth this cycle), no /tmp logs >1MB.
No alert. Mario v4.3 brief locked; user-side critical path ~15min.

## 2026-05-11 hourly check-in :17 — idle, launched ESN-fairness sweep
APU 35°C, sentinel alive, no z-script running. Launched z252 (ESN
hyperparam sweep on NARMA-10 N=200: sr×leak×gain = 4×3×3 = 36 configs,
n=5). O40 optional follow-up to check if NS-RAM beats any detuned ESN.

## 2026-05-11 z252 ESN-fairness sweep: ESN was UNDER-tuned at default
36 ESN configs (sr×leak×gain) on NARMA-10 N=200, 5 seeds.
Best ESN: NRMSE 0.461 (sr=1.1, leak=0.6, gain=0.3). Default ESN (z243):
0.563. NS-RAM: 0.612. NS-RAM beats detuned ESN at 17/36 configs;
ESN beats NS-RAM at 19/36. The "ESN-baseline-might-be-over-tuned"
concern from O40 (openai+grok) goes the OPPOSITE way: ESN was actually
under-tuned. A well-tuned ESN beats NS-RAM by 0.15 NRMSE, not 0.05.
Mario brief citing 0.563 is the fairest (default ESN); brief stays
honest. ESN-fairness optic permanently closed.

## 2026-05-11 hourly check-in :17 — idle, launched STEP E NS-RAM sweep
APU 34°C, sentinel alive, idle. Launched z253 (NS-RAM hyperparam sweep
on NARMA-10 N=200: g_VG2 × leak × dt = 27 configs, n=5). Mirror of
z252 ESN sweep. Compare best NS-RAM vs default 0.612 and best ESN 0.461.

## 2026-05-11 STEP E z253 NS-RAM hyperparam: ❌ MATRIX DEFINITIVELY CLOSED
27 configs (g_VG2 × leak × dt) on NARMA-10 N=200, 5 seeds. Best NS-RAM
tune NRMSE 0.646 (g_VG2=0.10, leak=0.30, dt=5e-7) — WORSE than default
0.612. 0/27 better than default. 0/27 beat best-tuned ESN 0.461. ESN
gap grows default-vs-default 0.05 → best-vs-best 0.15. No further
compute moves the brief; v4.3 is unconditionally final.

## 2026-05-11 daily synthesis 02:43 #41
APU 34°C, sentinel alive (PID 9161). Massive 24h sprint completed:
  - V_G2 continuum closed (z244b + z246, both honest FAIL per NO-CHEAT)
  - NS-RAM-vs-ESN matrix: 14 head-to-head cells, 0 NS-RAM wins, 11 ESN wins, 3 ties
  - ESN-fairness sweep (z252): default ESN was UNDER-tuned, gap actually grows
  - NS-RAM hyperparam sweep (z253): 0/27 beat default 0.612; tuning gives no headroom
  - O40 oracle 3/3 consensus: send brief now, no more compute changes the framing
  - MARIO_SEND_DECISION.md written

🚩 Blocked >5 days: sebas_silicon_characterisation_request.md (6d unsent),
sebas_thick_ox_request_addendum.md (4d unsent), mario_update_note_v2_draft.md
(2d unsent post-z243 revision).

No milestone flips. Mario v4.3 brief locked unconditionally final.
Critical path = user 15-min email-actions.

Proposal subrepo already in sync (last commit d068732 just pushed z253).

## 2026-05-11 hourly check-in :17 — idle, no queued action
APU 35°C, sentinel alive, no z-script running. All explicit plan steps
(VG2 1+3, NEXT_DIRECTION A-E, O40 follow-ups) executed. Per NO-CHEAT
discipline: not manufacturing new compute. Critical path = user emails.

## 2026-05-11 GPU off-hours 03:23 — z254 30-seed publication polish
NARMA-5/NARMA-20/MC bumped from 5 to 30 seeds. NARMA-20 flipped from
"tie" to ESN strict (NS-RAM 0.981 [0.95,1.02] vs ESN 0.853 [0.78,0.92]).
All 3 cells now ESN strict wins with tight CIs. Matrix tally updates
to 14 cells: 0 NS-RAM wins, 12 ESN wins, 1 tie (only MG h=12 remains).
APU peak <60°C (no thermal events). Mario v4.3 unchanged.

## 2026-05-11 hourly check-in :17 — idle, no queued action
APU 34°C, sentinel PID 9161 alive, no z-script running. All explicit
plan steps + 30-seed publication polish complete. Per NO-CHEAT:
no manufactured compute. Critical path = user 15-min email-actions.

## 2026-05-11 baseline watchdog 04:43 — PASS (4th consecutive day)
ER_SPARSE/ff/N=256/seed=0/1ep: best 0.500 final 0.125. Exact match.
Diff 0.000 ≤ 0.10. NO regression after ~30 new scripts (z244-z254) +
new plans + docs + cron rebuilds since ground truth. APU 34°C.

## 2026-05-11 hourly check-in :17 — idle, no queued action
APU <50°C, sentinel alive, no z-script running. Plan closed, baseline
watchdog PASS (4 consecutive days). Per NO-CHEAT: no manufactured work.

## 2026-05-11 hourly check-in :17 — idle (5th consecutive)
APU 34°C, sentinel alive, no z-script. Plan closed. Critical path = user.

## 2026-05-11 morning brief 06:29 #42
Overnight 8h: 10 entries. Overnight-launcher cron 68911a4c is DELETED
(removed 2026-05-10 as stale); no work from it. Active overnight runs:
z253 NS-RAM hyperparam sweep (FAIL, 0/27 better than default 0.612)
and z254 30-seed publication polish (NARMA-20 flipped tie → ESN strict).
APU peak <60°C, no thermal events. Daily synth #41 + resource audit +
baseline watchdog (PASS 4 days) + 4 idle hourly check-ins. Pending: 0
queued compute. Critical path: user 15-min email-actions (Sebas main
unsent 7d, Sebas thick-ox 4d, Mario v2 2d).

## 2026-05-11 hourly check-in :17 — idle (6th consecutive)
APU 34°C, sentinel alive, no z-script. Plan closed. No-op per discipline.

## 2026-05-11 hourly check-in :17 — idle (7th consecutive)
APU 34°C, sentinel alive, no z-script. No-op per discipline.

## 2026-05-11 hourly check-in :17 — idle (8th consecutive)
APU 35°C, sentinel alive, no z-script. No-op per discipline.

## 2026-05-11 track-audit 6h #19 — V/R/C/T/S/P (post-matrix-closure, brief locked)

Since #18: z243 (already pre-#18) → ESN matrix STEP A-E complete (z247-z253)
→ z251 small-N + z252 ESN-fairness + z253 NS-RAM hyperparam sweeps all FAIL
honestly → z254 30-seed polish flipped NARMA-20 tie to ESN strict win
→ MARIO_SEND_DECISION.md written → O40 oracle 3/3 consensus on closure
→ 8 consecutive idle hourly check-ins.

| Track | Status | Δ since #18 |
|---|---|---|
| **V** | ✅ HELD | Power-saturated; z254 bumped 3 matrix cells to n=30. |
| **R** | ✅ CLOSED | (no change) |
| **C** | ✅ CLOSED | (no change) |
| **T** | ✅ MAXIMAL | 14 head-to-head cells + 36+27 hyperparam configs all run. |
| **S** | ✅ HELD | NO-CHEAT principle codified; pre-reg gates honoured. |
| **P** | ✅ HELD | APU peaks <60°C overnight; baseline watchdog PASS 4 days. |

**Stalled count = 0** (seventh consecutive audit at zero stalled), but
the project is now COMPUTE-CLOSED: no work that meaningfully advances
the brief remains. Idle is the correct state.

The actual stalled item is HUMAN-side, not in the audit tracks:
  - Sebas main: 7 days unsent (well past 5-day flag)
  - Sebas thick-ox: 4 days unsent
  - Mario v2: 2 days unsent post-z243 revision

Recommendation: consider stretching the hourly :17 cron to every 4h
until user redirects or sends. The 8 consecutive idle no-ops generate
log noise without information.

Logged.

## WEEKLY REVIEW 2026-05-11 (Mon, week of 05-04 → 05-11)

**Done this week** (compute, in order):
- M3b corrections — Mario brief v4.0 → v4.1 honest walkback (1.39 dec at Bf=100,
  η≤1; ER_SPARSE wins MC at honest cell — NOT MESH_4N).
- V_G2 continuum bridge hypothesis: z244 + z244b — both gates FAILED honestly
  (5× hysteresis vs 100× pre-reg; no monotone decay max-at-fastest). Plan CLOSED.
- z246 mixed-mode fabric: best-mix f=0.25 edges pure-floating by 0.006 NRMSE,
  below 0.016 margin. FAIL.
- NS-RAM-vs-ESN matrix STEPS A-E (z247–z253): 14 head-to-head cells (NARMA-5/10/
  20, MC, N∈{100,200,500,1000} scaling, MG h∈{6,12}, small-N{30,50,100}) +
  36 ESN-fairness + 27 NS-RAM hyperparam configs. Result: 0 NS-RAM wins, 12 ESN
  wins, 1 marginal tie at MG h=12 ESN-side. Best detuned ESN 0.461 still beats
  NS-RAM 0.612.
- z254 30-seed publication-polish: flipped NARMA-20 tie → ESN strict win.
- Mario brief v4.3 locked stateless (no z-numbers/oracle tags). onepager.tex +
  packaged zip `nsram_proposal_full.zip` (1.3 MB) ready for manual upload.
- 4 consecutive baseline-watchdog PASS (best 0.5 / final 0.125, no drift).
- 8 consecutive disciplined idle hourly :17 (post-closure no-op pattern).

**In-flight**: nothing — compute plan reached natural closure 05-09.

**Blocked (HUMAN-side, not in audit tracks)**:
- Sebas main email: **7 days unsent** (past 5-day flag) — blocks thick-ox cell
  card + 7-rate transient data (task #128) which gates A.4 transient validation.
- Sebas thick-ox addendum: 4 days unsent.
- Mario v2 update note: 2 days unsent.
- A.4 transient (task #90) BLOCKED on Sebas data.
- A.6 Robert Julia cross-val (task #91) BLOCKED on Robert.

**Top priority next week**: USER sends the three queued emails. Without Sebas
silicon data (B_f transient, thick-ox card) all remaining brief refinements are
post-hoc; no compute path advances the headlines.

**Risk escalation**: Sebas email aged past 5-day flag; if no reply in another
week the M3b track will need a "freeze v4.3 indefinitely" decision instead of
a "wait for silicon data" hold.

**Re-plan needed?** No. B.5 fired last week; full matrix run + brief lock
already closed it. Next research direction needs USER buy-in (pivot to NS-RAM
non-reservoir primitive: PUF, stateful trigger, chaotic oscillator).

## 2026-05-11 09:45 — GIT REPO CORRUPTION DETECTED (push blocked)

`git fsck` reports several empty objects (zero-byte files dated Mar 14):
  - fc/914f4f6c... ab/fe3ca8... 40/89bcab9...
This is unclean-shutdown damage, not a code/research issue. Weekly review
is safely written to disk (01_LOG.md). NOT auto-fixing — repo repair is
user-action territory (potential options: `git fsck --lost-found`, fetch
missing objects from Heigke/NSRAM remote, or `git clone` fresh and
re-apply diff). Surfacing to user via PushNotification.

## 2026-05-11 hourly check-in :17 — idle (9th consecutive); VG2_CONTINUUM_PLAN CLOSED, no pending steps

## 2026-05-11 hourly check-in :17 — idle (10th consecutive); plan CLOSED, blocked on user-side email sends

## 2026-05-11 V_G2/ESN-matrix wake-up — all STEPS A-E CLOSED (z247-z254), STEP F synthesis done (NSRAM_VS_ESN_FINDINGS.md, MARIO_SEND_DECISION.md). Zero cells passed → Mario v4.3 stays final. Nothing to pick.

```
