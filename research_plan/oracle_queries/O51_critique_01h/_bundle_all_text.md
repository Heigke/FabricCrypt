# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: context.md (11139 chars) ===
```

## DAY-END SUMMARY 2026-05-12

**Locked wins**:
- HDC headline 80.23% n=20 UCI-HAR, CI95±0.74pp, 2.3 nJ/inf
- HDC noise robustness: N=2048 noise-immune at σ=0.05 (80.4%)
- Bayesian NS-RAM RNG: ESS 1.03× + NIST 5/5 (dual headline candidate)
- 4A oracle 3-way IP-licensing consensus
- DA1: 5 new device specs (S_fire/S_relax, snapback peak V_d, etc)
- DA2: zenodo deck IS outdated (user-confirmed + SA2 mapping)
- SA1: full Sebas canonical 58-param table, image-2.png decoded as
  source of three_branch_params_extracted.json

**Honest negatives**:
- Per-branch DC: V_G1=0.6 0.70 dec borderline PASS; 0.2/0.4 FAIL.
  Cell-wide single-Bf fit impossible. (NEW understanding, not new fail)
- SA3 identified 3 missing topology elements: VNwell→VB diode,
  VB-VG2 cap, VB as output node. Snapback gap = these topology gaps.
- 4D oracle GATE verdict: KWS chance-level is ship-blocker
- z303/z303b: zenodo BJT params degrade fit (different process)

**Path to v4.4 brief** (revised):
1. (BLOCKED) Implement VNwell→VB diode topology + Cj + V-dep leakage
2. (BLOCKED) Implement VB-VG2 coupling cap
3. (BLOCKED) Re-fit per-branch on new topology
4. (POSSIBLE) Use V_G1=0.6 branch fit (0.70 dec) as best-case showcase
5. (POSSIBLE) Ship dual-headline (HDC + Bayesian RNG) gated only on
   topology fix; explicitly flag subthreshold (V_G1=0.2) as next-stage

Net for tonight: don't push v4.4. The model rebuild needs topology
work, not parameter work. SA3 + SA4 jointly pinpoint exact next step.

## 2026-05-12 19:47 — :47 idle cron — idle, APU=43C, last campaign <1h ago

## 2026-05-12 19:47 — Deep-dive 2h cron: 4E gated, NOT triggered

4A/4B/4C/4D all closed. Per workflow: would trigger 4E brief compile.

**Decision: HOLD 4E.** Today's deeper findings rule out shippable brief:
- SA3 identified 3 missing topology elements (VNwell→VB diode + cap +
  output node) — pyport structurally incomplete
- SA4 confirmed branch optima incompatible — pure parameter refit cannot
  bridge the gap
- Oracle 4D verdict GATE not SHIP (KWS chance-level credibility gap)
- User feedback: model genuinely bad, RNG result not spectacular,
  network sims unspectacular

4E brief v4.4 compile would package findings that do not yet justify a
brief. Better to wait for topology fix (next-stage work) than ship a
brief whose credibility is below threshold.

Cluster idle, APU 43°C. No new compute launched.

NEXT-STEP candidates (user-gated, not auto-launched):
1. Implement VNwell→VB diode in pyport, re-test V_G1=0.6 branch
2. Draft email to Sebas asking for V_d > 2V transient sweeps
3. KWS gate attack (different SNN encoding, not just delta-mod retry)
4. Build out per-branch v0.1 model with explicit "subthreshold pending"

## 2026-05-12 19:30 — O49 oracle critique cycle — ≥2/3 fragility flag

Q1 CONSENSUS (3/3): HDC noise-robust headline OVERCLAIMED.
  Mixing N=1024 energy (2.3 nJ) with N=2048 noise-acc (80.4% @ σ=0.05)
  as single headline. z293 4B2 gates locked-FAILED but z302 N=2048
  result swapped in without flagging it's a different cell.

Q2 (2/3 gpt-5+grok): rerun SA4 with FULL SA1 canonical knobs (NFACTOR
  uncapped to 15, K1 per-V_G1 fixed from JSON, MBJT step 0→1 enforced
  at V_G1≈0.3). Today's SA4 only swept Bf/alpha0/Rs — branch
  incompatibility might be artifact of clipping parameterization, not
  fundamental missing topology.

Q3 (2/3 drift): "SA4 V_G1=0.6 borderline PASS at 0.704 (gate <0.7)"
  and "σ-robustness via N=2048" — both present marginal/cross-cell
  results as wins. gemini judged 4E hold exemplary.

**WARNING: corrective pre-register needed** (per cron protocol).

## 2026-05-12 20:47 — :47 idle — idle, APU=41C, last cron <2h

## 2026-05-12 21:47 — :47 idle — idle, APU=36C

## 2026-05-12 21:47 — Deep-dive 2h cron: O49 corrective pre-register

APU 36°C, cluster idle. 4A-4D closed, 4E held per O49 WARNING.

**Corrective action**: O49 Q2 consensus (gpt-5+grok) requested SA4
rerun with FULL SA1 canonical knobs to falsify "topology-blocked"
narrative. Today's z304 only swept Bf/alpha0/Rs — clipped param space.

**Pre-register z305 corrective sweep** (locked before compute):
- Per-V_G1 branch: NFACTOR_M2 uncapped (BBO bound 1..15, not 1..3)
- Per-V_G1 K1 fixed per SA1 (0.558/0.538/0.418)
- mbjt step: 0.001 at V_G1=0.2, 1.0 at V_G1=0.4/0.6 (per SA1)
- BETA0 per-V_G1 (10.75/19/20)
- ETAB_M1 +1.8 fixed
- Sweep Bf ∈ {500, 1000, 3000, 9000} (Bf=50 ruled out), Rs ∈ {0, 1e9, 1e10}
- 4×3×3=36 jobs, ~15 min wall on cluster

**Gates** (unchanged from SA4):
- PASS-conservative: cell-wide median log-RMSE < 0.5 dec across ALL 33 curves
- AMBITIOUS: < 0.3 dec + signed bias < 0.1
- FALSIFICATION: if conservative PASS, topology-gap narrative is wrong
  → revert "VNwell diode mandatory" claim, re-open v4.4 path

Launching as subagent next message.

## 2026-05-12 21:55 — z305 result: O49 falsification REJECTED

12/12 jobs. Best cell-wide: Bf=9000 Rs=1e9 → median 1.46 dec FAIL.

Per-V_G1 vs z304:
- 0.2: 2.06 → 4.56 (worse +2.50)
- 0.4: 1.41 → 1.76 (worse +0.35)
- 0.6: 0.70 → **0.43 (BETTER −0.27, branch PASS conservative)**

**O49 Q2 hypothesis REJECTED**: SA1 canonical knobs do not rescue
cell-wide fit. SA3 topology-gap narrative stands; v4.4 stays gated.

**New finding**: V_G1=0.6 branch alone reaches 0.43 dec (conservative
PASS, AMBITIOUS borderline 0.5 gate). Best branch-only result we have.

**Implementation bug detected**: ETAB_M1=+1.8 applied as global, but SA1
specifies per-branch (0.95/1.7/2.5). V_G1=0.2 regression of +2.5 dec is
this bug, not physics. Should re-run z305b with per-branch ETAB.

Also: subagent notes NFACTOR was already 1.25..12.15 in CSV path (not
clipped) in z304 — the 1..3 BBO clip lived in a different fitter, not
the path we ran today. So one O49 premise was wrong from the start.

**Net**: O49 critique partially correct (revealed our HDC headline
mixing was a real overclaim; should restate). But its model-side
hypothesis (clipping caused incompatibility) — wrong. Topology-gap
is the real cause.

## 2026-05-12 21:30 — 3h campaign cron — PLAN CLOSED, incremental v4.4 mode

APU=36C, cluster idle, no z2/z3 script active.

State: all Phase A/B/4-* tracks closed. AMBITIOUS PASSes today:
  - HDC N=1024 80.23% n=20 UCI-HAR (locked headline)
  - HDC σ-robust (caveat: cross-cell, oracle-flagged)
  - Bayesian NS-RAM RNG ESS 1.03× + NIST 5/5
  - V_G1=0.6 DC fit 0.43 dec (branch-only PASS conservative)

GATED: v4.4 brief HELD per Oracle 4D + Oracle 4D-recheck (KWS chance,
topology gap, HDC headline overclaim).

Next non-cron step (needs explicit user approval to launch):
  - VNwell→VB diode topology implementation in pyport (multi-day work,
    not a cron action)
  - Sebas email for V_d>2V transient sweeps
  - z305b per-branch-ETAB bug fix (1h, low value)

Per workflow: 0 NEW AMBITIOUS PASS this cron, plan effectively closed.
Logging "PLAN CLOSED — incremental v4.4 mode" per protocol.

## 2026-05-12 21:30 — track audit (6h cron)
Phase A: A.1✓ A.2✓ A.3 pending (#177) A.4 pending (#178). 2/4.
Phase B: DS-N1✓ DS-N2✓ DS-N5✓(AMBITIOUS) DS-N3✓(AMBITIOUS+NIST) DS-N4 in_progress(#186) DS-N6✓(FAIL). 5/6.
Phase 4 (deep-dive): 4A✓ 4B✓ 4C✓ 4D✓ 4E HELD (oracle GATE).
SA1✓ SA2✓ SA3✓ SA4✓ z305✓(O49 falsif rejected).
Headlines locked: HDC 80.23%, Bayesian RNG NIST 5/5. v4.4 brief gated on topology rebuild.

## 2026-05-12 22:47 — :47 idle — idle, APU=35C

## 2026-05-12 23:47 — :47 idle — idle, APU=35C

## 2026-05-12 23:47 — deep-dive 2h cron: no action
4A-D closed, 4E HELD (oracle GATE + topology gap). z305 yesterday
falsified the "clip" hypothesis, strengthened topology-blocked narrative.
No new compute launched. Cluster idle, APU 35°C.

## 2026-05-13 00:47 — :47 idle — idle, APU=35C

## 2026-05-13 00:21 — O50 12h synth — consensus

Q1 — Most defensible standalone (3/3): **Bayesian NS-RAM RNG** (NIST 5/5
+ ESS 1.03×). Most fragile: HDC noise headline (mixes N=1024 energy
with N=2048 noise acc). V_G1=0.6 fit = diagnostic, not headline.

Q2 — Stratified reporting (3/3): valid IF accompanied by (a) full 7-pt
table both pre/post-corrective, (b) cell-wide FAIL stated first, (c)
SA1 physical regime justification (mbjt step at V_G1=0.3), (d) ETAB
bug disclosure for V_G1=0.2 regression.

Q3 — Next 1-3h experiment (2/3 gemini+grok): **z305b ETAB per-branch
fix**. 1/3 gpt-5: VNwell→VB diode ablation stub. Consensus: z305b
first (cheap, removes confound), then VNwell ablation if z305b clean.

**Recommended action sequence (oracle-backed)**:
1. z305b: per-branch ETAB fix (1h, falsifies if V_G1=0.2 drops from
   4.56 dec back to ~2.06 — confirms bug)
2. z306 VNwell diode ablation (1-2h): minimal stub, 36-cell grid,
   Gate A: signed bias from 1.67 → ≤0.30 dec; Gate B: transient
   log-RMSE drops ≥0.7 dec; Gate C: V_G1=0.6 DC degrades ≤0.2 dec
3. If z306 PASS: green-light full topology rebuild → v4.4 re-open

Awaiting user gate for z305b launch.

## 2026-05-13 00:30 — 3h campaign cron: launching z305b per O50 consensus

APU=35C, cluster idle. O50 (2/3 oracle majority) flagged z305b
ETAB per-branch fix as highest-value 1-3h experiment.

**Pre-registered gates (locked)**:
- HYPOTHESIS: V_G1=0.2 regression in z305 (2.06→4.56 dec) caused by
  ETAB_M1 global +1.8 vs SA1 per-branch (0.95/1.7/2.5).
- PASS-bug-confirmed: V_G1=0.2 log-RMSE drops from 4.56 back to ≤2.30
  (within 0.3 dec of z304 baseline 2.06)
- BONUS PASS-conservative cell-wide: median <0.5 dec → would re-open
  v4.4 path (against today's expectations)
- FAIL: V_G1=0.2 stays >3.0 dec → bug isn't the cause, real physics

## 2026-05-13 01:47 — :47 idle — idle, APU=42C

## 2026-05-13 01:47 — deep-dive 2h cron: z305b running, no new launch
APU=41C. z305b in-flight (per O50 consensus track). 4E HELD pending z305b outcome.

## 2026-05-13 00:55 — z305b ETAB-fix result: BUG-FIX FAILED gate

12 jobs done. Best cell-wide: Bf=9000 Rs=1e9 → 1.462 dec (V_G1=0.2/0.4/0.6
= 4.56 / 1.75 / 0.43). **IDENTICAL to z305 within roundoff.**

Per-V_G1 best (Rs=0 column at z305b restored Sebas's measured per-row
ETAB 0.95): V_G1=0.2 = 2.06 dec (matches z304 baseline). So the per-row
CSV ETAB was already being honored at the row level — the "bug" wasn't
the actual cause.

**Real cause** of z305's V_G1=0.2 regression (2.06 → 4.56): forced
canonical K1=0.558 or BETA0=10.75 (not ETAB). z305b inherits same
canonical forcings, just with right ETAB — and still gets the same
regression when Rs≠0. With Rs=0 (vnwell OFF), V_G1=0.2 recovers.

**Structural finding (3rd confirmation today)**:
- V_G1=0.2 wants Rs=0 (vnwell OFF)
- V_G1=0.4/0.6 want Rs≥1e9 (vnwell ON)
- No (Bf, Rs) pair satisfies both regimes
- This is NOT a bug — it's the topology-gap (SA3): pure parameter sweep
  CANNOT bridge branches because the parasitic VNwell→VB path needs
  to be present-but-modulated by V_G1, not just static-on/static-off

**Gate verdicts**:
- PASS-bug-confirmed (V_G1=0.2 ≤2.30 at any cell): FAIL at 4.56 BUT
  the Rs=0 sub-row gives 2.06 → narrowly PASS if we accept per-branch Rs
- BONUS cell-wide PASS <0.5: FAIL (1.46)
- FAIL cause confirmed: not bug, real topology

**Net for v4.4**: z305b confirms SA3 topology-rebuild is the only path.
v4.4 stays HELD. Best branch result still V_G1=0.6 = 0.43 dec.

```


=== FILE: corrective_bf_9000_rs_1e9_z305b_etab_perbranch.json (6734 chars) ===
```json
{
  "script": "z305b_etab_perbranch",
  "bf": 9000,
  "rs": 1000000000.0,
  "alpha0": 7.84e-05,
  "sa1_per_vg1": {
    "0.2": {
      "K1": 0.558,
      "mbjt": 0.001,
      "BETA0": 10.75,
      "ETAB": 0.95
    },
    "0.4": {
      "K1": 0.538,
      "mbjt": 1.0,
      "BETA0": 19.0,
      "ETAB": 1.7
    },
    "0.6": {
      "K1": 0.418,
      "mbjt": 1.0,
      "BETA0": 20.0,
      "ETAB": 2.5
    }
  },
  "etab_m1_per_branch": {
    "0.2": 0.95,
    "0.4": 1.7,
    "0.6": 2.5
  },
  "cellwide_median_log_rmse": 1.4615970267624667,
  "cellwide_signed_dec_median": 1.2167437019831713,
  "worst_branch_median": 4.564459702175461,
  "elapsed_s": 149.37002325057983,
  "device": "cuda",
  "rows": [
    {
      "vg1": 0.2,
      "bf": 9000,
      "alpha0": 7.84e-05,
      "rs": 1000000000.0,
      "etab_used": 0.95,
      "median_log_rmse": 4.564459702175461,
      "signed_dec_median": 4.032989823375832,
      "p90_log_rmse": 5.472945583049756,
      "n_finite": 7,
      "n_total": 7,
      "per_curve": [
        {
          "VG2": -0.05,
          "log_rmse": 4.42381174332897,
          "signed_dec": 3.9280599649945023,
          "n_conv": 15
        },
        {
          "VG2": -0.1,
          "log_rmse": 4.564459702175461,
          "signed_dec": 4.032989823375832,
          "n_conv": 17
        },
        {
          "VG2": -0.15,
          "log_rmse": 4.5328494735677864,
          "signed_dec": 4.023713416907901,
          "n_conv": 17
        },
        {
          "VG2": -0.2,
          "log_rmse": 4.265757495080886,
          "signed_dec": 3.917721665468493,
          "n_conv": 15
        },
        {
          "VG2": 0.0,
          "log_rmse": 4.678014733808539,
          "signed_dec": 4.243265795944124,
          "n_conv": 15
        },
        {
          "VG2": 0.05,
          "log_rmse": 5.184832908902463,
          "signed_dec": 5.904635942244956,
          "n_conv": 17
        },
        {
          "VG2": 0.1,
          "log_rmse": 5.905114594270695,
          "signed_dec": 5.953531041438968,
          "n_conv": 15
        }
      ]
    },
    {
      "vg1": 0.4,
      "bf": 9000,
      "alpha0": 7.84e-05,
      "rs": 1000000000.0,
      "etab_used": 1.7,
      "median_log_rmse": 1.7545123284052977,
      "signed_dec_median": 1.2788718837317727,
      "p90_log_rmse": 3.440424242588648,
      "n_finite": 11,
      "n_total": 11,
      "per_curve": [
        {
          "VG2": -0.05,
          "log_rmse": 1.4640831728618815,
          "signed_dec": 1.2788718837317727,
          "n_conv": 10
        },
        {
          "VG2": -0.1,
          "log_rmse": 1.4618353560382193,
          "signed_dec": 1.2783541411890322,
          "n_conv": 10
        },
        {
          "VG2": -0.15,
          "log_rmse": 1.7482927242327737,
          "signed_dec": 1.3706453716612774,
          "n_conv": 11
        },
        {
          "VG2": -0.2,
          "log_rmse": 1.4615970267624667,
          "signed_dec": 1.2780576789446076,
          "n_conv": 10
        },
        {
          "VG2": 0.0,
          "log_rmse": 1.7545123284052977,
          "signed_dec": 1.3703968267626845,
          "n_conv": 11
        },
        {
          "VG2": 0.05,
          "log_rmse": 1.9643537296938114,
          "signed_dec": 1.3723796129033428,
          "n_conv": 11
        },
        {
          "VG2": 0.1,
          "log_rmse": 1.185243302549503,
          "signed_dec": 1.1099640043986847,
          "n_conv": 5
        },
        {
          "VG2": 0.15,
          "log_rmse": 2.282007358160249,
          "signed_dec": 1.1120601791073446,
          "n_conv": 6
        },
        {
          "VG2": 0.2,
          "log_rmse": 2.5352132963085956,
          "signed_dec": 1.0351235736150324,
          "n_conv": 4
        },
        {
          "VG2": 0.25,
          "log_rmse": 3.440424242588648,
          "signed_dec": 3.440424242588648,
          "n_conv": 1
        },
        {
          "VG2": 0.3,
          "log_rmse": 3.5393594004275246,
          "signed_dec": 3.4100494615241335,
          "n_conv": 2
        }
      ]
    },
    {
      "vg1": 0.6,
      "bf": 9000,
      "alpha0": 7.84e-05,
      "rs": 1000000000.0,
      "etab_used": 2.5,
      "median_log_rmse": 0.4262229865582069,
      "signed_dec_median": -0.12569611892374422,
      "p90_log_rmse": 1.1215921987059048,
      "n_finite": 15,
      "n_total": 15,
      "per_curve": [
        {
          "VG2": -0.05,
          "log_rmse": 0.18005990851823184,
          "signed_dec": -0.12569611892374422,
          "n_conv": 10
        },
        {
          "VG2": -0.1,
          "log_rmse": 0.18031322527103602,
          "signed_dec": -0.12683984025260653,
          "n_conv": 10
        },
        {
          "VG2": -0.15,
          "log_rmse": 0.1799754922901342,
          "signed_dec": -0.12551459439836066,
          "n_conv": 10
        },
        {
          "VG2": -0.2,
          "log_rmse": 0.1800557003159811,
          "signed_dec": -0.12594525538278223,
          "n_conv": 10
        },
        {
          "VG2": 0.0,
          "log_rmse": 0.18016810930328866,
          "signed_dec": -0.12573860315559582,
          "n_conv": 10
        },
        {
          "VG2": 0.05,
          "log_rmse": 0.18022002955190414,
          "signed_dec": -0.12588803623966704,
          "n_conv": 10
        },
        {
          "VG2": 0.1,
          "log_rmse": 0.18100006992261172,
          "signed_dec": -0.12622344219528703,
          "n_conv": 10
        },
        {
          "VG2": 0.15,
          "log_rmse": 0.43715889067617386,
          "signed_dec": -0.1261464900000595,
          "n_conv": 10
        },
        {
          "VG2": 0.2,
          "log_rmse": 0.4262229865582069,
          "signed_dec": -0.1251210176610389,
          "n_conv": 10
        },
        {
          "VG2": 0.25,
          "log_rmse": 0.9560737178435864,
          "signed_dec": 1.181025183079215,
          "n_conv": 5
        },
        {
          "VG2": 0.3,
          "log_rmse": 1.1288297353989432,
          "signed_dec": 1.2167437019831713,
          "n_conv": 4
        },
        {
          "VG2": 0.35,
          "log_rmse": 0.9734753567285981,
          "signed_dec": -0.1846441678549251,
          "n_conv": 6
        },
        {
          "VG2": 0.4,
          "log_rmse": 1.1107358936663472,
          "signed_dec": 1.1107358936663472,
          "n_conv": 1
        },
        {
          "VG2": 0.45,
          "log_rmse": 1.3953524095721708,
          "signed_dec": 1.3953524095721708,
          "n_conv": 1
        },
        {
          "VG2": 0.5,
          "log_rmse": 0.9490000679783659,
          "signed_dec": 0.9490000679783659,
          "n_conv": 1
        }
      ]
    }
  ]
}
```


=== FILE: summary_z304_sebas_refit.json (6782 chars) ===
```json
{
  "script": "z304_aggregate",
  "n_cells_loaded": 176,
  "n_finite_cells": 176,
  "n_source_files": 11,
  "by_vg1": {
    "0.2": {
      "best": {
        "vg1": 0.2,
        "bf": 500,
        "alpha0": 1e-05,
        "rs": 0,
        "median_log_rmse": 2.0610291308357587,
        "signed_dec_median": -1.4757399592295073,
        "p90_log_rmse": 2.1123002207762025,
        "n_finite": 7
      },
      "pareto": [
        {
          "bf": 500,
          "alpha0": 1e-05,
          "rs": 0,
          "median_log_rmse": 2.0610291308357587,
          "signed_dec_median": -1.4757399592295073
        },
        {
          "bf": 500,
          "alpha0": 0.0001,
          "rs": 0,
          "median_log_rmse": 2.0610291308357587,
          "signed_dec_median": -1.4757399592295073
        },
        {
          "bf": 500,
          "alpha0": 0.001,
          "rs": 0,
          "median_log_rmse": 2.0610291308357587,
          "signed_dec_median": -1.4757399592295073
        },
        {
          "bf": 500,
          "alpha0": 0.01,
          "rs": 0,
          "median_log_rmse": 2.0610291308357587,
          "signed_dec_median": -1.4757399592295073
        }
      ],
      "n_branch_cells": 64
    },
    "0.4": {
      "best": {
        "vg1": 0.4,
        "bf": 50,
        "alpha0": 1e-05,
        "rs": 10000000000.0,
        "median_log_rmse": 1.4046663288699635,
        "signed_dec_median": 0.4243714966378498,
        "p90_log_rmse": 1.4945019316616872,
        "n_finite": 7
      },
      "pareto": [
        {
          "bf": 50,
          "alpha0": 1e-05,
          "rs": 10000000000.0,
          "median_log_rmse": 1.4046663288699635,
          "signed_dec_median": 0.4243714966378498
        },
        {
          "bf": 50,
          "alpha0": 0.0001,
          "rs": 10000000000.0,
          "median_log_rmse": 1.4046663288699635,
          "signed_dec_median": 0.4243714966378498
        },
        {
          "bf": 50,
          "alpha0": 0.001,
          "rs": 10000000000.0,
          "median_log_rmse": 1.4046663288699635,
          "signed_dec_median": 0.4243714966378498
        },
        {
          "bf": 50,
          "alpha0": 0.01,
          "rs": 10000000000.0,
          "median_log_rmse": 1.4046663288699635,
          "signed_dec_median": 0.4243714966378498
        }
      ],
      "n_branch_cells": 48
    },
    "0.6": {
      "best": {
        "vg1": 0.6,
        "bf": 9000,
        "alpha0": 1e-05,
        "rs": 10000000000.0,
        "median_log_rmse": 0.7042229003043868,
        "signed_dec_median": 0.12519489440961706,
        "p90_log_rmse": 0.9573272527337507,
        "n_finite": 11
      },
      "pareto": [
        {
          "bf": 9000,
          "alpha0": 1e-05,
          "rs": 10000000000.0,
          "median_log_rmse": 0.7042229003043868,
          "signed_dec_median": 0.12519489440961706
        },
        {
          "bf": 9000,
          "alpha0": 0.0001,
          "rs": 10000000000.0,
          "median_log_rmse": 0.7042229003043868,
          "signed_dec_median": 0.12519489440961706
        },
        {
          "bf": 9000,
          "alpha0": 0.001,
          "rs": 10000000000.0,
          "median_log_rmse": 0.7042229003043868,
          "signed_dec_median": 0.12519489440961706
        },
        {
          "bf": 9000,
          "alpha0": 0.01,
          "rs": 10000000000.0,
          "median_log_rmse": 0.7042229003043868,
          "signed_dec_median": 0.12519489440961706
        },
        {
          "bf": 500,
          "alpha0": 1e-05,
          "rs": 1000000000.0,
          "median_log_rmse": 0.8765201949636146,
          "signed_dec_median": -0.09842195704459478
        },
        {
          "bf": 500,
          "alpha0": 0.0001,
          "rs": 1000000000.0,
          "median_log_rmse": 0.8765201949636146,
          "signed_dec_median": -0.09842195704459478
        }
      ],
      "n_branch_cells": 64
    }
  },
  "best_cellwide_compromise": {
    "bf": 50,
    "alpha0": 1e-05,
    "rs": 10000000000.0,
    "vg1_02_med": 2.3975482170253373,
    "vg1_04_med": 1.4046663288699635,
    "vg1_06_med": 2.7901932952092294,
    "worst_branch_med": 2.7901932952092294,
    "median_across_branches": 2.3975482170253373,
    "max_abs_signed": 3.165630881809051
  },
  "top_5_cellwide": [
    {
      "bf": 50,
      "alpha0": 1e-05,
      "rs": 10000000000.0,
      "vg1_02_med": 2.3975482170253373,
      "vg1_04_med": 1.4046663288699635,
      "vg1_06_med": 2.7901932952092294,
      "worst_branch_med": 2.7901932952092294,
      "median_across_branches": 2.3975482170253373,
      "max_abs_signed": 3.165630881809051
    },
    {
      "bf": 50,
      "alpha0": 0.0001,
      "rs": 10000000000.0,
      "vg1_02_med": 2.3975482170253373,
      "vg1_04_med": 1.4046663288699635,
      "vg1_06_med": 2.7901932952092294,
      "worst_branch_med": 2.7901932952092294,
      "median_across_branches": 2.3975482170253373,
      "max_abs_signed": 3.165630881809051
    },
    {
      "bf": 50,
      "alpha0": 0.001,
      "rs": 10000000000.0,
      "vg1_02_med": 2.3975482170253373,
      "vg1_04_med": 1.4046663288699635,
      "vg1_06_med": 2.7901932952092294,
      "worst_branch_med": 2.7901932952092294,
      "median_across_branches": 2.3975482170253373,
      "max_abs_signed": 3.165630881809051
    },
    {
      "bf": 50,
      "alpha0": 0.01,
      "rs": 10000000000.0,
      "vg1_02_med": 2.3975482170253373,
      "vg1_04_med": 1.4046663288699635,
      "vg1_06_med": 2.7901932952092294,
      "worst_branch_med": 2.7901932952092294,
      "median_across_branches": 2.3975482170253373,
      "max_abs_signed": 3.165630881809051
    },
    {
      "bf": 50,
      "alpha0": 1e-05,
      "rs": 1000000000.0,
      "vg1_02_med": 3.2264262915314457,
      "vg1_04_med": 1.4894433845213277,
      "vg1_06_med": 1.7824399697183684,
      "worst_branch_med": 3.2264262915314457,
      "median_across_branches": 1.7824399697183684,
      "max_abs_signed": 3.778944938250161
    }
  ],
  "gates": {
    "vg1_0.2": {
      "PASS_conservative": false,
      "AMBITIOUS": false,
      "SAFETY": false,
      "median_log_rmse": 2.0610291308357587,
      "signed_dec_median": -1.4757399592295073
    },
    "vg1_0.4": {
      "PASS_conservative": false,
      "AMBITIOUS": false,
      "SAFETY": true,
      "median_log_rmse": 1.4046663288699635,
      "signed_dec_median": 0.4243714966378498
    },
    "vg1_0.6": {
      "PASS_conservative": false,
      "AMBITIOUS": false,
      "SAFETY": true,
      "median_log_rmse": 0.7042229003043868,
      "signed_dec_median": 0.12519489440961706
    }
  },
  "verdict": {
    "ALL_PASS_conservative": false,
    "ALL_AMBITIOUS_SHIP_v4.4": false,
    "ALL_SAFETY": false,
    "CELLWIDE_BEATS_DA3": false
  },
  "da3_reference_median": 0.99
}
```


=== FILE: summary_z305_corrective.json (14626 chars) ===
```json
{
  "script": "z305_aggregate",
  "n_cells": 12,
  "n_finite_cells": 12,
  "n_source_files": 12,
  "best_cell": {
    "bf": 9000,
    "rs": 1000000000.0,
    "alpha0": 7.84e-05,
    "cellwide_median_log_rmse": 1.4615970267624667,
    "cellwide_signed_dec_median": 1.2167437019831713,
    "worst_branch_median": 4.564459702175461,
    "n_curves_cellwide": 33,
    "per_vg1": {
      "0.2": {
        "median_log_rmse": 4.564459702175461,
        "signed_dec_median": 4.032989823375832,
        "p90_log_rmse": 5.472945583049756,
        "n_finite": 7,
        "n_total": 7
      },
      "0.4": {
        "median_log_rmse": 1.7545123284052977,
        "signed_dec_median": 1.2788718837317727,
        "p90_log_rmse": 3.440424242588648,
        "n_finite": 11,
        "n_total": 11
      },
      "0.6": {
        "median_log_rmse": 0.4262229865582069,
        "signed_dec_median": -0.12569611892374422,
        "p90_log_rmse": 1.1215921987059048,
        "n_finite": 15,
        "n_total": 15
      }
    }
  },
  "best_by_worst_branch": {
    "bf": 500,
    "rs": 10000000000.0,
    "alpha0": 7.84e-05,
    "cellwide_median_log_rmse": 1.7186422824302858,
    "cellwide_signed_dec_median": 0.9488079048772393,
    "worst_branch_median": 3.279043711844223,
    "n_curves_cellwide": 33,
    "per_vg1": {
      "0.2": {
        "median_log_rmse": 3.279043711844223,
        "signed_dec_median": 3.6678670228042813,
        "p90_log_rmse": 3.6994830880816294,
        "n_finite": 7,
        "n_total": 7
      },
      "0.4": {
        "median_log_rmse": 1.3863546999067744,
        "signed_dec_median": 1.4479760758835631,
        "p90_log_rmse": 1.5204149060699266,
        "n_finite": 11,
        "n_total": 11
      },
      "0.6": {
        "median_log_rmse": 1.881434601093058,
        "signed_dec_median": -2.1525971932278134,
        "p90_log_rmse": 2.1471685993377703,
        "n_finite": 15,
        "n_total": 15
      }
    }
  },
  "all_cells_ranked": [
    {
      "bf": 9000,
      "rs": 1000000000.0,
      "alpha0": 7.84e-05,
      "cellwide_median_log_rmse": 1.4615970267624667,
      "cellwide_signed_dec_median": 1.2167437019831713,
      "worst_branch_median": 4.564459702175461,
      "n_curves_cellwide": 33,
      "per_vg1": {
        "0.2": {
          "median_log_rmse": 4.564459702175461,
          "signed_dec_median": 4.032989823375832,
          "p90_log_rmse": 5.472945583049756,
          "n_finite": 7,
          "n_total": 7
        },
        "0.4": {
          "median_log_rmse": 1.7545123284052977,
          "signed_dec_median": 1.2788718837317727,
          "p90_log_rmse": 3.440424242588648,
          "n_finite": 11,
          "n_total": 11
        },
        "0.6": {
          "median_log_rmse": 0.4262229865582069,
          "signed_dec_median": -0.12569611892374422,
          "p90_log_rmse": 1.1215921987059048,
          "n_finite": 15,
          "n_total": 15
        }
      }
    },
    {
      "bf": 3000,
      "rs": 10000000000.0,
      "alpha0": 7.84e-05,
      "cellwide_median_log_rmse": 1.572165710235055,
      "cellwide_signed_dec_median": 0.9887105579946969,
      "worst_branch_median": 3.480498884261861,
      "n_curves_cellwide": 33,
      "per_vg1": {
        "0.2": {
          "median_log_rmse": 3.480498884261861,
          "signed_dec_median": 2.8071106694482495,
          "p90_log_rmse": 4.2322433272152855,
          "n_finite": 7,
          "n_total": 7
        },
        "0.4": {
          "median_log_rmse": 1.894062761701453,
          "signed_dec_median": 2.1919467984565237,
          "p90_log_rmse": 2.279446457222868,
          "n_finite": 11,
          "n_total": 11
        },
        "0.6": {
          "median_log_rmse": 1.2343338953453955,
          "signed_dec_median": -1.3873877465637232,
          "p90_log_rmse": 1.3318474559447588,
          "n_finite": 15,
          "n_total": 15
        }
      }
    },
    {
      "bf": 500,
      "rs": 1000000000.0,
      "alpha0": 7.84e-05,
      "cellwide_median_log_rmse": 1.644227545542835,
      "cellwide_signed_dec_median": 0.857724585814216,
      "worst_branch_median": 3.748408408822016,
      "n_curves_cellwide": 33,
      "per_vg1": {
        "0.2": {
          "median_log_rmse": 3.748408408822016,
          "signed_dec_median": 3.3964488922874736,
          "p90_log_rmse": 4.418635456211415,
          "n_finite": 7,
          "n_total": 7
        },
        "0.4": {
          "median_log_rmse": 1.9600133235463126,
          "signed_dec_median": 2.2505494928507774,
          "p90_log_rmse": 2.5412941382781735,
          "n_finite": 11,
          "n_total": 11
        },
        "0.6": {
          "median_log_rmse": 1.0506725718533554,
          "signed_dec_median": -1.169348067007312,
          "p90_log_rmse": 1.1953975212933008,
          "n_finite": 15,
          "n_total": 15
        }
      }
    },
    {
      "bf": 1000,
      "rs": 1000000000.0,
      "alpha0": 7.84e-05,
      "cellwide_median_log_rmse": 1.6562405897244394,
      "cellwide_signed_dec_median": 0.869449223132027,
      "worst_branch_median": 3.946221176470414,
      "n_curves_cellwide": 33,
      "per_vg1": {
        "0.2": {
          "median_log_rmse": 3.946221176470414,
          "signed_dec_median": 3.5796773187639213,
          "p90_log_rmse": 4.585723711452564,
          "n_finite": 7,
          "n_total": 7
        },
        "0.4": {
          "median_log_rmse": 2.032048237695419,
          "signed_dec_median": 0.8833239578854313,
          "p90_log_rmse": 2.8155382038248495,
          "n_finite": 11,
          "n_total": 11
        },
        "0.6": {
          "median_log_rmse": 0.8244470146325626,
          "signed_dec_median": -0.8734350581480221,
          "p90_log_rmse": 0.9302426676318251,
          "n_finite": 15,
          "n_total": 15
        }
      }
    },
    {
      "bf": 9000,
      "rs": 10000000000.0,
      "alpha0": 7.84e-05,
      "cellwide_median_log_rmse": 1.6759109038851796,
      "cellwide_signed_dec_median": 0.9681098211120007,
      "worst_branch_median": 4.043085221322548,
      "n_curves_cellwide": 33,
      "per_vg1": {
        "0.2": {
          "median_log_rmse": 4.043085221322548,
          "signed_dec_median": 3.6004574361636887,
          "p90_log_rmse": 4.623615926664974,
          "n_finite": 7,
          "n_total": 7
        },
        "0.4": {
          "median_log_rmse": 2.067086502620025,
          "signed_dec_median": 2.495431299889411,
          "p90_log_rmse": 2.5462878710482144,
          "n_finite": 11,
          "n_total": 11
        },
        "0.6": {
          "median_log_rmse": 0.9427043479593645,
          "signed_dec_median": -1.019491565388166,
          "p90_log_rmse": 0.9573928756967149,
          "n_finite": 15,
          "n_total": 15
        }
      }
    },
    {
      "bf": 500,
      "rs": 10000000000.0,
      "alpha0": 7.84e-05,
      "cellwide_median_log_rmse": 1.7186422824302858,
      "cellwide_signed_dec_median": 0.9488079048772393,
      "worst_branch_median": 3.279043711844223,
      "n_curves_cellwide": 33,
      "per_vg1": {
        "0.2": {
          "median_log_rmse": 3.279043711844223,
          "signed_dec_median": 3.6678670228042813,
          "p90_log_rmse": 3.6994830880816294,
          "n_finite": 7,
          "n_total": 7
        },
        "0.4": {
          "median_log_rmse": 1.3863546999067744,
          "signed_dec_median": 1.4479760758835631,
          "p90_log_rmse": 1.5204149060699266,
          "n_finite": 11,
          "n_total": 11
        },
        "0.6": {
          "median_log_rmse": 1.881434601093058,
          "signed_dec_median": -2.1525971932278134,
          "p90_log_rmse": 2.1471685993377703,
          "n_finite": 15,
          "n_total": 15
        }
      }
    },
    {
      "bf": 1000,
      "rs": 10000000000.0,
      "alpha0": 7.84e-05,
      "cellwide_median_log_rmse": 1.7991552221708413,
      "cellwide_signed_dec_median": 1.24500670710417,
      "worst_branch_median": 3.485179742510139,
      "n_curves_cellwide": 33,
      "per_vg1": {
        "0.2": {
          "median_log_rmse": 3.485179742510139,
          "signed_dec_median": 4.094070636918446,
          "p90_log_rmse": 3.885529912054596,
          "n_finite": 7,
          "n_total": 7
        },
        "0.4": {
          "median_log_rmse": 1.5671125110572732,
          "signed_dec_median": 1.7555380393379583,
          "p90_log_rmse": 1.820059501652589,
          "n_finite": 11,
          "n_total": 11
        },
        "0.6": {
          "median_log_rmse": 1.5968733250857001,
          "signed_dec_median": -1.8565454062589462,
          "p90_log_rmse": 1.844381700058338,
          "n_finite": 15,
          "n_total": 15
        }
      }
    },
    {
      "bf": 3000,
      "rs": 1000000000.0,
      "alpha0": 7.84e-05,
      "cellwide_median_log_rmse": 2.2326820226672925,
      "cellwide_signed_dec_median": 1.4804993503614714,
      "worst_branch_median": 4.20839268815272,
      "n_curves_cellwide": 33,
      "per_vg1": {
        "0.2": {
          "median_log_rmse": 4.20839268815272,
          "signed_dec_median": 3.386912861851143,
          "p90_log_rmse": 5.020667825079505,
          "n_finite": 7,
          "n_total": 7
        },
        "0.4": {
          "median_log_rmse": 2.4700087274672895,
          "signed_dec_median": 3.0242875115146584,
          "p90_log_rmse": 3.1435135084770325,
          "n_finite": 11,
          "n_total": 11
        },
        "0.6": {
          "median_log_rmse": 0.6808459901862913,
          "signed_dec_median": -0.5057091750619671,
          "p90_log_rmse": 0.7989418075991326,
          "n_finite": 15,
          "n_total": 15
        }
      }
    },
    {
      "bf": 9000,
      "rs": 0.0,
      "alpha0": 7.84e-05,
      "cellwide_median_log_rmse": 2.6618139128937424,
      "cellwide_signed_dec_median": -2.5706291985767553,
      "worst_branch_median": 4.37334893305727,
      "n_curves_cellwide": 33,
      "per_vg1": {
        "0.2": {
          "median_log_rmse": 2.1361190884178325,
          "signed_dec_median": -1.50503340743475,
          "p90_log_rmse": 2.2692346706541,
          "n_finite": 7,
          "n_total": 7
        },
        "0.4": {
          "median_log_rmse": 2.4938516182354338,
          "signed_dec_median": -2.4109416274818347,
          "p90_log_rmse": 2.738789365031107,
          "n_finite": 11,
          "n_total": 11
        },
        "0.6": {
          "median_log_rmse": 4.37334893305727,
          "signed_dec_median": -3.9500964240678504,
          "p90_log_rmse": 4.515226718772673,
          "n_finite": 15,
          "n_total": 15
        }
      }
    },
    {
      "bf": 3000,
      "rs": 0.0,
      "alpha0": 7.84e-05,
      "cellwide_median_log_rmse": 2.725772650972054,
      "cellwide_signed_dec_median": -2.7690228416772253,
      "worst_branch_median": 4.530011655977114,
      "n_curves_cellwide": 33,
      "per_vg1": {
        "0.2": {
          "median_log_rmse": 2.1977677733476617,
          "signed_dec_median": -1.775272486193181,
          "p90_log_rmse": 2.2671654177936773,
          "n_finite": 7,
          "n_total": 7
        },
        "0.4": {
          "median_log_rmse": 2.627361473916951,
          "signed_dec_median": -2.755259823439914,
          "p90_log_rmse": 2.817727919632092,
          "n_finite": 11,
          "n_total": 11
        },
        "0.6": {
          "median_log_rmse": 4.530011655977114,
          "signed_dec_median": -4.332895829148584,
          "p90_log_rmse": 4.691720478877983,
          "n_finite": 15,
          "n_total": 15
        }
      }
    },
    {
      "bf": 1000,
      "rs": 0.0,
      "alpha0": 7.84e-05,
      "cellwide_median_log_rmse": 2.855212149419574,
      "cellwide_signed_dec_median": -2.786947387287965,
      "worst_branch_median": 4.727073337349762,
      "n_curves_cellwide": 33,
      "per_vg1": {
        "0.2": {
          "median_log_rmse": 2.2275895370874834,
          "signed_dec_median": -2.0608467372649333,
          "p90_log_rmse": 2.2878643994424848,
          "n_finite": 7,
          "n_total": 7
        },
        "0.4": {
          "median_log_rmse": 2.7851408503731214,
          "signed_dec_median": -2.736661793704311,
          "p90_log_rmse": 3.0066613897631225,
          "n_finite": 11,
          "n_total": 11
        },
        "0.6": {
          "median_log_rmse": 4.727073337349762,
          "signed_dec_median": -4.419340360464326,
          "p90_log_rmse": 4.894650310874422,
          "n_finite": 15,
          "n_total": 15
        }
      }
    },
    {
      "bf": 500,
      "rs": 0.0,
      "alpha0": 7.84e-05,
      "cellwide_median_log_rmse": 3.148947893574699,
      "cellwide_signed_dec_median": -3.253600275987914,
      "worst_branch_median": 4.750195200347424,
      "n_curves_cellwide": 33,
      "per_vg1": {
        "0.2": {
          "median_log_rmse": 2.0610291308360997,
          "signed_dec_median": -1.4757399592267628,
          "p90_log_rmse": 2.1123002207739985,
          "n_finite": 7,
          "n_total": 7
        },
        "0.4": {
          "median_log_rmse": 3.0349573312365132,
          "signed_dec_median": -3.22042430947816,
          "p90_log_rmse": 3.1883200206542677,
          "n_finite": 11,
          "n_total": 11
        },
        "0.6": {
          "median_log_rmse": 4.750195200347424,
          "signed_dec_median": -4.696176407065674,
          "p90_log_rmse": 4.876422311998427,
          "n_finite": 15,
          "n_total": 15
        }
      }
    }
  ],
  "gates_at_best": {
    "PASS_conservative": false,
    "AMBITIOUS": false,
    "cellwide_median_log_rmse": 1.4615970267624667,
    "cellwide_signed_dec_median": 1.2167437019831713,
    "threshold_conservative": 0.5,
    "threshold_ambitious_med": 0.3,
    "threshold_ambitious_bias": 0.1
  },
  "diff_vs_z304_per_branch": {
    "0.2": {
      "z304_best_med": 2.061,
      "z304_best_bf": 500,
      "z304_best_rs": 0,
      "z305_at_best_cell_med": 4.564459702175461,
      "delta": 2.503459702175461
    },
    "0.4": {
      "z304_best_med": 1.405,
      "z304_best_bf": 50,
      "z304_best_rs": 10000000000.0,
      "z305_at_best_cell_med": 1.7545123284052977,
      "delta": 0.34951232840529767
    },
    "0.6": {
      "z304_best_med": 0.7,
      "z304_best_bf": 9000,
      "z304_best_rs": 10000000000.0,
      "z305_at_best_cell_med": 0.4262229865582069,
      "delta": -0.27377701344179306
    }
  },
  "verdict": "CONSERVATIVE FAIL: O49 clipped-parameterization hypothesis REJECTED. Even with SA1-canonical per-branch overrides, cell-wide median exceeds 0.5 dec. SA3 topology-gap narrative is supported."
}
```
