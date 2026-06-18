# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: context.md (14967 chars) ===
```
branch. The 0.965 dec headline is part-physics, part-overfit. Honest
write-up should report 1.19 dec (mean-param baseline) as the "no
per-VG1 cheating" floor, with the 0.965 quoted as "engineering fit
with caveats — VG1=0.6 branch needs topology change, not parameters".

R-arc closed: 1.131 dec global floor is real; 0.965 per-VG1 is half
real / half over-parameterization.

## 2026-05-14 21:15 — R-48 MIXED VERDICT

Per-VG1 params split:
  Bf, iii_body_gain: monotonic ✓ — REAL physics (M2 body-effect Vth shift)
  log10(vnwell_Rs): non-monotonic, VG1=0.6 4 OoM off → CURVE-FIT artifact

LOOCV: VG1=0.20 PHYSICS_PASS, VG1=0.40 PHYSICS_PASS,
       VG1=0.60 CURVE_FIT (ratio 2.37)

Honest accounting:
  1.131 global-knob floor (R-43/45/47)
  1.192 mean-of-triples (no per-VG1)
  0.965 R-46 with per-VG1 fit
  0.23 dec gain: ~half real physics, ~half curve-fit at VG1=0.6

VG1=0.6 needs topology change (n-well clamp path), not parameter fitting.
Honest claim: 1.19 dec baseline; 0.965 with VG1=0.6 caveat.

## 2026-05-14 21:13 cron — 8 spår aktiva
R-46 done (0.965 dec). R-48 mixed verdict. DS-N7c Memory Palace dead.
DS-N11 partial PASS (Lorenz vs LSTM). Active: R-48 done, R-49 dbd avalanche,
DS-N7d Method-of-Loci right, DS-N8 KWS, DS-N10 reservoir, DS-N12 STDP, O67.

## 2026-05-14 21:25 — DS-N7d FINAL DEATH CERTIFICATE for Memory Palace

Method-of-Loci with proper STDP-pretrained substrate FAILS WORSE than random:
  NS-RAM real STDP (full cue): 0.537
  NS-RAM random pretrain:      0.814 (+27.7pp BETTER)
  Digital MLP:                 0.543

Wave propagation real (12→3200 cells visible) but DESTROYS per-key
discrimination — every key writes to overlapping wavefronts → M matrix
loses per-item signatures.

Memory Palace dead in 4 architectures (A1/A2/A3/D7d). Not framing
problem — physics: additive linear substrate with smearing dynamics
cannot implement sparse content-addressable kernel.

PERMANENT RETRACTION: NS-RAM as memory architecture (any form). 
What remains viable:
  - Temporal filter / reservoir node (DS-N11 Lorenz win)
  - Cell-level energy density (4 fJ, 3.3 bits)
  - DC fit (1.19 honest baseline)
17:51 idle-check: APU=81C
17:52 idle-check: 4 heavy + 6 booting subagents, APU=81C, monitor close
17:53 machine-balance WARN: ikaros 4 heavy + 6 dispatched all going to ikaros; daedalus SSH blocked; zgx only DS-N8

## 2026-05-14 21:45 — R-51 M2 audit: M2 IS CLEAN

binunit=2 ✓ (R-37 fix global), lalpha0 cancellation only 7% (M2 Leff=1.78µm
10× M1), lpe0/toxe correct. Even with LALPHA0_FIX, M2 Iii at flagship is
19 decades below M1 → M2 can't move body balance.

Confirms: VG1=0.6 residual is GENUINE TOPOLOGY ISSUE.
R-49 (drain-body avalanche, currently running) is correct direction.

## 2026-05-14 21:55 — DS-N16 SECOND SURVIVAL: 5G equalizer pitch

BER tracks fp32 LMS within 1pp across SNR sweep.
Energy: NS-RAM 2.76 pJ vs fp32 LMS 237.8 pJ → 86× reduction
Energy vs int8 LMS: 6.7× (not 100× AMBITIOUS)

Commercial pitch: 5G UE PUSCH receiver, Wi-Fi 7 OFDMA equalizer.
Concrete: ~85% modem MAC energy savings.

Honest: NS-RAM converges 2.3× slower (1800 vs 770 symbols).
Energy from literature (not measured).

Two surviving claims now:
  DS-N11 predictive coding (Lorenz vs LSTM, 2000× energy)
  DS-N16 adaptive equalizer (5G/Wi-Fi, 86× vs fp32)
Plus DC fit 1.19 dec honest, cell-level 4 fJ/3.3 bit density.

## 2026-05-14 22:00 — DS-N14 MIXED: energy PASS, detection FAIL

KWS gate: recall 0.095, precision 0.508, wake 5.1%, 19.5× power save
ECG PVC: recall 0.098, precision 0.48, wake 6.6%, 17.2× power save

Energy story defensible — Mario's canonical 10-100× pitch matches.
Detection story FAIL — gate fires near chance, saves energy by missing.

To make defensible: STDP plasticity + multi-tau temporal pooling +
within-record cross-val for ECG.

3 surviving claims:
  DS-N11 predictive coding (Lorenz)
  DS-N16 5G/Wi-Fi equalizer (86×)
  DS-N14 edge cascade ENERGY only (20× save, detection TBD)

## 2026-05-14 22:30 — DS-N13 topology zoo DONE

Best: lattice2D 0.0865 NRMSE (uniform-degree wins).
Worst: ER 0.224 (heavy-tailed hubs → avalanche overdrive).
LSTM baseline 0.0642 (beats NS-RAM 26% accuracy, costs 600× more energy).

Design principle: NS-RAM reservoirs must use bounded-degree
topologies (lattice/ring/SW). Avoid ER/scale-free (hubs kill).

This is useful negative finding for Mario/Sebas: topology matters,
not all reservoirs work equally.

## 2026-05-14 22:40 — DS-N15 4th SURVIVING CLAIM

NIST 3/15 raw (expected for 1/f). KL=0.00147 = 3.1× BETTER than numpy.
Energy: 1 fJ/bit ensemble vs digital ~10 pJ/bit → 10,000× advantage.

Defensible commercial pitch: "Edge Bayesian inference TRNG"
- Trillion-sample MCMC at sub-µW
- All HW-RNGs use whitener (standard), NS-RAM same with <0.1 pJ overhead
- Sweet spot: probabilistic inference / Bayes nets / particle filters

4 surviving claims:
  1. DS-N11 predictive coding (Lorenz, 2000× energy vs LSTM)
  2. DS-N16 5G/Wi-Fi equalizer (86× vs fp32 LMS)
  3. DS-N14 edge cascade ENERGY (20× vs always-on)
  4. DS-N15 stochastic RNG for Bayesian (10,000× vs digital)
## 2026-05-14 22:43 cron — 8 spår live, 4 surviving claims captured
## 2026-05-14 22:48 cron — R-49 dbd avalanche + R-50 physics BBO in flight

## 2026-05-14 22:50 — DS-N12 RETRACT: STDP ECG fails completely

NS-RAM acc 0.554 vs digital STDP 0.725 (-17pp).
NS-RAM+STDP bit-identical to NS-RAM-frozen → STDP DOES NOTHING (Vb saturates).
NS-RAM 3× more energy than digital STDP.

5th retraction. NS-RAM substrate too fast for QRS-scale STDP.
Surviving claims: DS-N11/N14e/N15/N16.

## 2026-05-14 23:00 — DS-N10 5th SURVIVING: sine-frequency classification

Task-specific results:
  Mackey-Glass: Random wins (trivial 1-step AR)
  NARMA: Random wins (predicts mean)
  Memory capacity: LIF wins (more linear)
  Sine class 4-way: NS-RAM 97.8% vs LIF 86.1% vs Random 24.4% ← CLEAR WIN

NS-RAM bistable phase-lock = frequency-discrimination specialist.
Loses on linear/regression tasks. Wins on phase/frequency tasks.

5 surviving claims with 5 distinct strengths now identified.

## 2026-05-14 23:10 — wake-up status
ACTIVE: z365 BBO (5h14m), z370 phys-BBO (42m), z371 GPU blitz (13m). APU=70°C OK.
DS-N18 done: 1000 params, 100% valid Id, ~57s/curve.
DS-N17 still running (zgx mass-scale). DS-N10 already logged (sine class win).

## 2026-05-14 23:25 — autonomous tick
APU=83°C (close to 85°C pause threshold — NO new launches).
Active: z371 GPU blitz (21m), z365 BBO (5h22m), z370 phys-BBO (50m).
Stale loop refs (z332/z333/z334) — not relevant; campaign moved to R-49/R-50/DS-N17-18.
Waiting on in-flight, no new spawns until APU<60°C.
## 2026-05-14 23:47 — :47 idle check
ACTIVE: z365 BBO + z368 dbd-avalanche + z370 + z371. APU=79°C. Sentinel PID 2720 alive.

## 2026-05-14 23:50 — 2h deep-dive tick
APU=68°C (above 60 but below 85). 9 python scripts active. No new dispatch.
4A/4B/4C all completed. 4D/4E gated on R-49 + R-50 + DS-N17/18 results.
Inflight: R-49 dbd-avalanche, R-50 phys-BBO, DS-N17, DS-N18 (26/33), z365 BBO.

## 2026-05-14 23:55 — DS-N18 DONE (z371 GPU blitz)
NEW GLOBAL-KNOB FLOOR: 1.0484 dec (was 1.131 from R-43 DE).
184/1000 beat R-43. Top-10 plateau 1.048-1.054. 0/1000 < 0.95.
DE in R-43/R-45/R-47 was stuck in local basin — GPU Sobol/LHS dense coverage found true floor.
Best params PHYSICAL: Bf=26538, Va=2.046, Is=5.92e-11, vnwell_Rs=1.11e8, iii_body_gain=0.796.
CONFIRMS: R-46 per-VG1 0.965 break is STRUCTURAL (branch-specific topology required).
→ R-49 dbd-avalanche direction validated.
Outputs: results/z371_gpu_blitz/.

## 2026-05-15 00:00 — DS-N17 DONE (zgx, 25 min)
INFRA PASS: all 4 sweeps complete @ N=1M in <140s/task.
Layered topology: monotonic NRMSE 0.737→0.747 from 10K→1M (best scaling).
ER topology: 0.873→0.749 (14% drop).
NO NS-RAM win over LSTM at scale (LSTM 0.08 vs best NS-RAM 0.747 on Mackey-Glass).
DS-N11 Lorenz win (0.044) used predictive-coding head — ridge readout (DS-N17) loses.
Clarification: NS-RAM scaling-ready, but architectural win is head-specific.

## 2026-05-15 00:10 — SURR-V4 DONE on daedalus (88s!)
100K 5D surrogate (V_G1,V_G2,V_d,V_Nwell,V_b) — 94.6% converged.
Daedalus GPU 24× faster than expected. 1M extrapolated = 15 min.
Pulled NPZ back to ikaros. Task #177 → completed.
Note: Daedalus IP now .40 (was .37), use daedalus.local mDNS.

## 2026-05-15 00:25 — autonomous tick
APU=47°C. z368 R-49 progress: baseline reproduced (perVG1=0.965, global=1.19) confirms R-46/R-50 floor.
Now sweeping BV in [8,12]V for dbd avalanche. Resumed stopped z365/z370 (got SIGSTOP somehow).

## 2026-05-15 00:38 — all-night cron tick
Stale refs z332/z333/z334 — campaign moved to R-49/R-50/SURR-V4 stack.
Active: z365 (5h51m), z368 R-49 dbd-avalanche (BV sweep), z370, O68 oracle.
SURR-V4 100K daedalus done (88s). Status dashboard built.

## 2026-05-15 00:42 — topology rebuild tick
R-phase: R-4/R-7 (BBO + topology fix) ACTIVE via z368 R-49 dbd-avalanche.
z368 progress: baseline confirmed (perVG1=0.965, global=1.19); BV=8.0V done, BV=10.0V in progress.
ALERT: R-5/R-8 (LIF + network re-run on v5 surrogate) NOT YET dispatched.
SNAPBACK + LIF dynamics NOT YET demonstrated — user request received.

## 2026-05-15 00:55 — z372 snapback demo (HONEST CRITICAL FINDING)
SNAPBACK FOLD MISSING ON 3/3 BIASES.
VG1=0.2: model fold 0.06 vs measured 2.94 dec
VG1=0.4: model fold 0.05 vs measured 2.32 dec
VG1=0.6: model fold 0.02 vs measured 2.20 dec
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

```


=== FILE: z371_gpu_blitz_summary.json (25407 chars) ===
```json
{
  "script": "z371_gpu_param_blitz",
  "task": "DS-N18",
  "device": "cuda",
  "gpu_name": "AMD Radeon 8060S",
  "N_samples": 1000,
  "n_sobol": 500,
  "n_lhs": 500,
  "n_curves": 33,
  "param_bounds_R50": {
    "Bf": [
      50.0,
      50000.0
    ],
    "Va": [
      0.5,
      3.0
    ],
    "Is": [
      1e-13,
      1e-07
    ],
    "vnwell_Rs": [
      100000.0,
      1000000000.0
    ],
    "iii_body_gain": [
      0.1,
      1.0
    ]
  },
  "patches_active": [
    "R-20 BJT Vbc",
    "R-29 Vth/tox",
    "R-37 binunit",
    "R-41 body_pdiode_to=vnwell + use_well_diode=True",
    "R-45 cfg.vnwell=2.0 frozen",
    "z371 tensorized bjt.Bf/Va/Is + cfg.iii_body_gain/vnwell_Rs"
  ],
  "elapsed_total_s": 1898.4244430065155,
  "elapsed_loop_s": 1891.3051862716675,
  "wall_time_min": 31.640407383441925,
  "best": {
    "rank": 0,
    "idx": 389,
    "params": {
      "Bf": 26538.323750765994,
      "Va": 2.045733422972262,
      "Is": 5.923521560752862e-11,
      "vnwell_Rs": 110853623.41670267,
      "iii_body_gain": 0.7956871120259166
    },
    "cell_wide_median_dec": 1.0484038518967627,
    "n_valid_curves": 33,
    "n_total_curves": 33,
    "sampler": "sobol"
  },
  "top_10": [
    {
      "rank": 0,
      "idx": 389,
      "params": {
        "Bf": 26538.323750765994,
        "Va": 2.045733422972262,
        "Is": 5.923521560752862e-11,
        "vnwell_Rs": 110853623.41670267,
        "iii_body_gain": 0.7956871120259166
      },
      "cell_wide_median_dec": 1.0484038518967627,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 1,
      "idx": 273,
      "params": {
        "Bf": 40514.997707959265,
        "Va": 2.7225432726554573,
        "Is": 3.9092693028614466e-10,
        "vnwell_Rs": 40887230.612988524,
        "iii_body_gain": 0.13327379021793606
      },
      "cell_wide_median_dec": 1.0498971334130516,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 2,
      "idx": 507,
      "params": {
        "Bf": 40553.06009695619,
        "Va": 2.4673043739418796,
        "Is": 1.0222573059796736e-10,
        "vnwell_Rs": 100468424.56263985,
        "iii_body_gain": 0.37922513178931294
      },
      "cell_wide_median_dec": 1.0508907762307922,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 3,
      "idx": 35,
      "params": {
        "Bf": 6167.019081581384,
        "Va": 2.7008371050469577,
        "Is": 1.1131642780006126e-09,
        "vnwell_Rs": 19477009.46597463,
        "iii_body_gain": 0.5250671631656587
      },
      "cell_wide_median_dec": 1.0511313974103333,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 4,
      "idx": 27,
      "params": {
        "Bf": 9341.284273425117,
        "Va": 2.2890839371830225,
        "Is": 1.1271575659002016e-10,
        "vnwell_Rs": 100329964.0800084,
        "iii_body_gain": 0.6287194843403995
      },
      "cell_wide_median_dec": 1.051240680017914,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 5,
      "idx": 842,
      "params": {
        "Bf": 14845.57911152552,
        "Va": 2.9836158943291164,
        "Is": 1.806774821299751e-10,
        "vnwell_Rs": 115585270.30270365,
        "iii_body_gain": 0.813102015853623
      },
      "cell_wide_median_dec": 1.0524630218070194,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 6,
      "idx": 961,
      "params": {
        "Bf": 19623.645931865998,
        "Va": 2.762230413334966,
        "Is": 1.8895044790785945e-10,
        "vnwell_Rs": 55800294.50583728,
        "iii_body_gain": 0.17918554291761124
      },
      "cell_wide_median_dec": 1.053117936597255,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 7,
      "idx": 907,
      "params": {
        "Bf": 28049.074593251906,
        "Va": 2.690083340669287,
        "Is": 9.355239106813877e-09,
        "vnwell_Rs": 3158933.7217614073,
        "iii_body_gain": 0.19639258018667652
      },
      "cell_wide_median_dec": 1.053934918902325,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 8,
      "idx": 594,
      "params": {
        "Bf": 48741.97894145558,
        "Va": 2.1037424661972377,
        "Is": 1.3793043662674143e-11,
        "vnwell_Rs": 169028871.18020725,
        "iii_body_gain": 0.9780587712332381
      },
      "cell_wide_median_dec": 1.054043773629596,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 9,
      "idx": 812,
      "params": {
        "Bf": 13957.753483308652,
        "Va": 2.50386145203475,
        "Is": 9.161532467760041e-10,
        "vnwell_Rs": 19950571.761149496,
        "iii_body_gain": 0.6922301395382107
      },
      "cell_wide_median_dec": 1.0543615008070957,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    }
  ],
  "top_50": [
    {
      "rank": 0,
      "idx": 389,
      "params": {
        "Bf": 26538.323750765994,
        "Va": 2.045733422972262,
        "Is": 5.923521560752862e-11,
        "vnwell_Rs": 110853623.41670267,
        "iii_body_gain": 0.7956871120259166
      },
      "cell_wide_median_dec": 1.0484038518967627,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 1,
      "idx": 273,
      "params": {
        "Bf": 40514.997707959265,
        "Va": 2.7225432726554573,
        "Is": 3.9092693028614466e-10,
        "vnwell_Rs": 40887230.612988524,
        "iii_body_gain": 0.13327379021793606
      },
      "cell_wide_median_dec": 1.0498971334130516,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 2,
      "idx": 507,
      "params": {
        "Bf": 40553.06009695619,
        "Va": 2.4673043739418796,
        "Is": 1.0222573059796736e-10,
        "vnwell_Rs": 100468424.56263985,
        "iii_body_gain": 0.37922513178931294
      },
      "cell_wide_median_dec": 1.0508907762307922,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 3,
      "idx": 35,
      "params": {
        "Bf": 6167.019081581384,
        "Va": 2.7008371050469577,
        "Is": 1.1131642780006126e-09,
        "vnwell_Rs": 19477009.46597463,
        "iii_body_gain": 0.5250671631656587
      },
      "cell_wide_median_dec": 1.0511313974103333,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 4,
      "idx": 27,
      "params": {
        "Bf": 9341.284273425117,
        "Va": 2.2890839371830225,
        "Is": 1.1271575659002016e-10,
        "vnwell_Rs": 100329964.0800084,
        "iii_body_gain": 0.6287194843403995
      },
      "cell_wide_median_dec": 1.051240680017914,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 5,
      "idx": 842,
      "params": {
        "Bf": 14845.57911152552,
        "Va": 2.9836158943291164,
        "Is": 1.806774821299751e-10,
        "vnwell_Rs": 115585270.30270365,
        "iii_body_gain": 0.813102015853623
      },
      "cell_wide_median_dec": 1.0524630218070194,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 6,
      "idx": 961,
      "params": {
        "Bf": 19623.645931865998,
        "Va": 2.762230413334966,
        "Is": 1.8895044790785945e-10,
        "vnwell_Rs": 55800294.50583728,
        "iii_body_gain": 0.17918554291761124
      },
      "cell_wide_median_dec": 1.053117936597255,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 7,
      "idx": 907,
      "params": {
        "Bf": 28049.074593251906,
        "Va": 2.690083340669287,
        "Is": 9.355239106813877e-09,
        "vnwell_Rs": 3158933.7217614073,
        "iii_body_gain": 0.19639258018667652
      },
      "cell_wide_median_dec": 1.053934918902325,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 8,
      "idx": 594,
      "params": {
        "Bf": 48741.97894145558,
        "Va": 2.1037424661972377,
        "Is": 1.3793043662674143e-11,
        "vnwell_Rs": 169028871.18020725,
        "iii_body_gain": 0.9780587712332381
      },
      "cell_wide_median_dec": 1.054043773629596,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 9,
      "idx": 812,
      "params": {
        "Bf": 13957.753483308652,
        "Va": 2.50386145203475,
        "Is": 9.161532467760041e-10,
        "vnwell_Rs": 19950571.761149496,
        "iii_body_gain": 0.6922301395382107
      },
      "cell_wide_median_dec": 1.0543615008070957,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 10,
      "idx": 439,
      "params": {
        "Bf": 16984.031221503392,
        "Va": 2.5944320899434388,
        "Is": 1.0161813706774335e-10,
        "vnwell_Rs": 72130828.12079895,
        "iii_body_gain": 0.5014674127101898
      },
      "cell_wide_median_dec": 1.0551956518477572,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 11,
      "idx": 133,
      "params": {
        "Bf": 26380.994962202385,
        "Va": 2.577159204054624,
        "Is": 1.8096605793636971e-09,
        "vnwell_Rs": 11254197.704337038,
        "iii_body_gain": 0.11295578451827169
      },
      "cell_wide_median_dec": 1.056650816029877,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 12,
      "idx": 869,
      "params": {
        "Bf": 29769.104198832836,
        "Va": 2.5125242890482915,
        "Is": 2.8122276872612416e-09,
        "vnwell_Rs": 7743996.693780111,
        "iii_body_gain": 0.9912235171151658
      },
      "cell_wide_median_dec": 1.056805183648659,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 13,
      "idx": 423,
      "params": {
        "Bf": 15126.542581571266,
        "Va": 2.005143886897713,
        "Is": 9.948350877567547e-09,
        "vnwell_Rs": 2695891.669474298,
        "iii_body_gain": 0.2880574038252235
      },
      "cell_wide_median_dec": 1.0571341281816329,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 14,
      "idx": 291,
      "params": {
        "Bf": 5985.179321561009,
        "Va": 2.1762777883559465,
        "Is": 4.0535885316179184e-11,
        "vnwell_Rs": 198600275.4908173,
        "iii_body_gain": 0.7702411610633135
      },
      "cell_wide_median_dec": 1.0575716010861034,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 15,
      "idx": 63,
      "params": {
        "Bf": 22881.521416874602,
        "Va": 2.0025235223583877,
        "Is": 3.0708597765474067e-10,
        "vnwell_Rs": 78460135.68752506,
        "iii_body_gain": 0.8829300950281322
      },
      "cell_wide_median_dec": 1.057579923021189,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 16,
      "idx": 81,
      "params": {
        "Bf": 39846.8271820806,
        "Va": 2.8509450489655137,
        "Is": 5.113575822495824e-09,
        "vnwell_Rs": 4402739.86401279,
        "iii_body_gain": 0.49460418717935684
      },
      "cell_wide_median_dec": 1.0583295853950663,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 17,
      "idx": 616,
      "params": {
        "Bf": 12842.52043356836,
        "Va": 2.8548531150108327,
        "Is": 3.8408496909003634e-09,
        "vnwell_Rs": 14312190.294474198,
        "iii_body_gain": 0.6723369147280153
      },
      "cell_wide_median_dec": 1.0583527055816493,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 18,
      "idx": 668,
      "params": {
        "Bf": 48115.110022354784,
        "Va": 1.7871351318484006,
        "Is": 1.3872313186814437e-10,
        "vnwell_Rs": 125777298.36741965,
        "iii_body_gain": 0.5066040802641264
      },
      "cell_wide_median_dec": 1.0588799968333262,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 19,
      "idx": 153,
      "params": {
        "Bf": 46226.50492498651,
        "Va": 2.181714665610343,
        "Is": 2.102567153228935e-10,
        "vnwell_Rs": 43024825.75580981,
        "iii_body_gain": 0.6834307597018778
      },
      "cell_wide_median_dec": 1.0594791666316212,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 20,
      "idx": 473,
      "params": {
        "Bf": 45363.59829986468,
        "Va": 2.366420852020383,
        "Is": 1.4628841174720171e-08,
        "vnwell_Rs": 4009113.354543635,
        "iii_body_gain": 0.8760329646989703
      },
      "cell_wide_median_dec": 1.0597684779817358,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 21,
      "idx": 959,
      "params": {
        "Bf": 34507.5725650503,
        "Va": 2.548237511169151,
        "Is": 5.418317881010655e-09,
        "vnwell_Rs": 11648928.335211692,
        "iii_body_gain": 0.6967835734072532
      },
      "cell_wide_median_dec": 1.0599067302555767,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 22,
      "idx": 74,
      "params": {
        "Bf": 29655.05857169628,
        "Va": 1.282588787842542,
        "Is": 2.1145732792265e-09,
        "vnwell_Rs": 9145276.092141306,
        "iii_body_gain": 0.898362027015537
      },
      "cell_wide_median_dec": 1.0603270390395212,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 23,
      "idx": 929,
      "params": {
        "Bf": 31064.61715605223,
        "Va": 2.0975449318420996,
        "Is": 7.869325269879729e-11,
        "vnwell_Rs": 74249867.7485526,
        "iii_body_gain": 0.4650812737330797
      },
      "cell_wide_median_dec": 1.0603739202885771,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 24,
      "idx": 247,
      "params": {
        "Bf": 16119.789298856631,
        "Va": 2.4269742909818888,
        "Is": 8.312694331575352e-09,
        "vnwell_Rs": 7771425.749127782,
        "iii_body_gain": 0.14033572217449547
      },
      "cell_wide_median_dec": 1.0604438194874712,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 25,
      "idx": 976,
      "params": {
        "Bf": 14259.551600478508,
        "Va": 1.3421306850071273,
        "Is": 3.4498109095552604e-09,
        "vnwell_Rs": 5987792.424809878,
        "iii_body_gain": 0.9450930813974819
      },
      "cell_wide_median_dec": 1.0606817774409032,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 26,
      "idx": 341,
      "params": {
        "Bf": 30320.083723869175,
        "Va": 2.733126051723957,
        "Is": 3.58913809643504e-09,
        "vnwell_Rs": 13104911.595756445,
        "iii_body_gain": 0.8258185720071196
      },
      "cell_wide_median_dec": 1.0609253072039868,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 27,
      "idx": 718,
      "params": {
        "Bf": 12803.606632281884,
        "Va": 2.3231536446312258,
        "Is": 1.2540705604243147e-09,
        "vnwell_Rs": 27340654.510164075,
        "iii_body_gain": 0.8102510706011868
      },
      "cell_wide_median_dec": 1.0625703974348955,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 28,
      "idx": 562,
      "params": {
        "Bf": 42504.88231701961,
        "Va": 2.600162945544282,
        "Is": 3.818807193955121e-10,
        "vnwell_Rs": 56968804.81936837,
        "iii_body_gain": 0.6418804325584335
      },
      "cell_wide_median_dec": 1.0626654827096074,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 29,
      "idx": 505,
      "params": {
        "Bf": 23236.111567813598,
        "Va": 2.0922341846770482,
        "Is": 3.816158551769374e-11,
        "vnwell_Rs": 95501937.64572757,
        "iii_body_gain": 0.9032823145458088
      },
      "cell_wide_median_dec": 1.0632753656527518,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 30,
      "idx": 691,
      "params": {
        "Bf": 42622.8559085179,
        "Va": 2.1592073548218327,
        "Is": 1.1460601417196955e-08,
        "vnwell_Rs": 4400378.826111249,
        "iii_body_gain": 0.8457377274492527
      },
      "cell_wide_median_dec": 1.063731955141033,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 31,
      "idx": 792,
      "params": {
        "Bf": 30203.207182573373,
        "Va": 2.8867476539710184,
        "Is": 4.821268841674965e-09,
        "vnwell_Rs": 8661019.189658247,
        "iii_body_gain": 0.32193553453737683
      },
      "cell_wide_median_dec": 1.0643849906691505,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 32,
      "idx": 644,
      "params": {
        "Bf": 49285.97738392462,
        "Va": 1.7639439806298183,
        "Is": 5.010286523168241e-11,
        "vnwell_Rs": 181805865.36448947,
        "iii_body_gain": 0.2404123189479558
      },
      "cell_wide_median_dec": 1.0644570504843993,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 33,
      "idx": 893,
      "params": {
        "Bf": 29920.634550240888,
        "Va": 1.6442280456149616,
        "Is": 1.302101838701463e-11,
        "vnwell_Rs": 146400244.1660674,
        "iii_body_gain": 0.7282951651692442
      },
      "cell_wide_median_dec": 1.0651436672608485,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 34,
      "idx": 383,
      "params": {
        "Bf": 22212.777769984677,
        "Va": 1.856887654401362,
        "Is": 4.239601012522911e-09,
        "vnwell_Rs": 7323369.979855846,
        "iii_body_gain": 0.6905686501413584
      },
      "cell_wide_median_dec": 1.0657729268609615,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 35,
      "idx": 321,
      "params": {
        "Bf": 40885.31863987446,
        "Va": 2.9035515286959708,
        "Is": 1.636054773608827e-11,
        "vnwell_Rs": 123476505.07807381,
        "iii_body_gain": 0.9481326624751091
      },
      "cell_wide_median_dec": 1.0661508134490603,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 36,
      "idx": 755,
      "params": {
        "Bf": 9354.760782565205,
        "Va": 2.873930089741817,
        "Is": 1.665031812578384e-08,
        "vnwell_Rs": 2618309.0179373473,
        "iii_body_gain": 0.5030711474830507
      },
      "cell_wide_median_dec": 1.0661714413998167,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 37,
      "idx": 469,
      "params": {
        "Bf": 30082.731751119718,
        "Va": 1.8648014143109322,
        "Is": 1.4447883359122829e-09,
        "vnwell_Rs": 31791734.915467203,
        "iii_body_gain": 0.31681073643267155
      },
      "cell_wide_median_dec": 1.0664968974797446,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 38,
      "idx": 385,
      "params": {
        "Bf": 41834.79216997512,
        "Va": 2.3812526925466955,
        "Is": 3.0108024513634244e-09,
        "vnwell_Rs": 5231855.30111538,
        "iii_body_gain": 0.7675999311730266
      },
      "cell_wide_median_dec": 1.0667390816535343,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 39,
      "idx": 509,
      "params": {
        "Bf": 35257.487997015356,
        "Va": 1.3086337914203072,
        "Is": 1.574844902686554e-10,
        "vnwell_Rs": 76323241.3448061,
        "iii_body_gain": 0.907235032711455
      },
      "cell_wide_median_dec": 1.0671784930777117,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 40,
      "idx": 93,
      "params": {
        "Bf": 36367.48986202292,
        "Va": 2.4178372947499156,
        "Is": 7.764124323314462e-10,
        "vnwell_Rs": 55351558.924836226,
        "iii_body_gain": 0.6103338702581823
      },
      "cell_wide_median_dec": 1.067588471270199,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 41,
      "idx": 834,
      "params": {
        "Bf": 39730.84309083737,
        "Va": 2.4325547814852193,
        "Is": 1.0018657659391792e-08,
        "vnwell_Rs": 7295025.663739738,
        "iii_body_gain": 0.5637704759634643
      },
      "cell_wide_median_dec": 1.0676078111968497,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 42,
      "idx": 352,
      "params": {
        "Bf": 24291.22149287723,
        "Va": 1.6379211768507957,
        "Is": 8.221930175712263e-11,
        "vnwell_Rs": 158266598.20989922,
        "iii_body_gain": 0.370104212872684
      },
      "cell_wide_median_dec": 1.0678997699921322,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 43,
      "idx": 770,
      "params": {
        "Bf": 37663.03465277365,
        "Va": 1.9173135590025758,
        "Is": 8.118370378413123e-11,
        "vnwell_Rs": 121080437.7565297,
        "iii_body_gain": 0.21909565954977828
      },
      "cell_wide_median_dec": 1.067906264873948,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 44,
      "idx": 316,
      "params": {
        "Bf": 3903.6636328324676,
        "Va": 1.3381093991920352,
        "Is": 1.0626057296094613e-08,
        "vnwell_Rs": 4298183.387841805,
        "iii_body_gain": 0.40073690395802264
      },
      "cell_wide_median_dec": 1.0682040985667498,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 45,
      "idx": 846,
      "params": {
        "Bf": 3000.745411549392,
        "Va": 2.07304074051652,
        "Is": 1.8515383120520144e-09,
        "vnwell_Rs": 16001776.001469744,
        "iii_body_gain": 0.3236754815985188
      },
      "cell_wide_median_dec": 1.0686766365102771,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 46,
      "idx": 410,
      "params": {
        "Bf": 27057.352838059887,
        "Va": 1.4398049488663673,
        "Is": 7.290101054663817e-09,
        "vnwell_Rs": 7853300.792246538,
        "iii_body_gain": 0.2645462978631258
      },
      "cell_wide_median_dec": 1.0695372855629965,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "sobol"
    },
    {
      "rank": 47,
      "idx": 905,
      "params": {
        "Bf": 33016.70980959711,
        "Va": 1.8914245641078449,
        "Is": 5.549799971814544e-10,
        "vnwell_Rs": 65444648.10301104,
        "iii_body_gain": 0.6445575254072303
      },
      "cell_wide_median_dec": 1.0699403531005496,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 48,
      "idx": 939,
      "params": {
        "Bf": 29858.884939687574,
        "Va": 2.1764692206185083,
        "Is": 5.290443350834142e-11,
        "vnwell_Rs": 210876027.65237927,
        "iii_body_gain": 0.35856233886377653
      },
      "cell_wide_median_dec": 1.070053125746726,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    },
    {
      "rank": 49,
      "idx": 965,
      "params": {
        "Bf": 19155.587077326018,
        "Va": 1.315865389718535,
        "Is": 4.53074078243299e-09,
        "vnwell_Rs": 7874131.922110021,
        "iii_body_gain": 0.9819238604928427
      },
      "cell_wide_median_dec": 1.0702124085634688,
      "n_valid_curves": 33,
      "n_total_curves": 33,
      "sampler": "lhs"
    }
  ],
  "baselines": {
    "z363_R43_global_floor": 1.1306581736187744,
    "z365_R46_perVG1": 0.965,
    "ngspice_target_aspiration": 0.27
  },
  "stats": {
    "median_dec": 1.3841625311103107,
    "p10_dec": 1.1001958028855299,
    "p25_dec": 1.170968440075225,
    "min_dec": 1.0484038518967627,
    "max_dec": 7.2461430580234065,
    "n_below_R43_floor": 184,
    "n_below_0p95": 0,
    "n_below_0p50": 0
  },
  "gates": {
    "INFRA_under_30min": false,
    "DISCOVERY_beat_R43": true,
    "BREAKTHROUGH_under_0p95": false
  }
}
```


=== FILE: z372_snapback_demo_summary.json (2188 chars) ===
```json
{
  "script": "z372_snapback_demo",
  "task": "Compare pyport model vs Sebas measured snapback @ VG2=+0.20 for VG1 in {0.2,0.4,0.6}",
  "params_source": "R-46 z365 per-VG1 BBO best (cell-wide median = 0.965 dec)",
  "x_best_R46": [
    1889.88,
    1.8447,
    9.1722,
    1092.27,
    1.5152,
    9.8983,
    417.63,
    0.9036,
    6.7846
  ],
  "results_per_bias": [
    {
      "VG1": 0.2,
      "VG2": 0.1,
      "file": "StandardIV_HH_2vHCa-2_VG2=0.10_VG=0.2(1)_03-33-55PM.csv",
      "params": {
        "Bf": 1889.88,
        "iii_body_gain": 1.8447,
        "vnwell_Rs": 1486620098.5286007
      },
      "rmse_dec": 1.9697764851973347,
      "n_valid_points": 89,
      "Vth_measured_V": null,
      "V_snapback_knee_V": 2.0003,
      "measured_jump_dec": 2.9384075319933425,
      "model_jump_dec": 0.06404175505583964,
      "snapback_reproduced": false
    },
    {
      "VG1": 0.4,
      "VG2": 0.2,
      "file": "StandardIV_HH_2vHCa-2_VG2=0.20_VG=0.4(1)_03-39-29PM.csv",
      "params": {
        "Bf": 1092.27,
        "iii_body_gain": 1.5152,
        "vnwell_Rs": 7912249981.34314
      },
      "rmse_dec": 1.0642496027489,
      "n_valid_points": 81,
      "Vth_measured_V": 1.55052,
      "V_snapback_knee_V": 1.55052,
      "measured_jump_dec": 2.3160563345939975,
      "model_jump_dec": 0.046577844765955945,
      "snapback_reproduced": false
    },
    {
      "VG1": 0.6,
      "VG2": 0.2,
      "file": "StandardIV_HH_2vHCa-2_VG2=0.20_VG=0.6(1)_03-45-46PM.csv",
      "params": {
        "Bf": 417.63,
        "iii_body_gain": 0.9036,
        "vnwell_Rs": 6089757.5146773
      },
      "rmse_dec": 0.8100924917706103,
      "n_valid_points": 81,
      "Vth_measured_V": 1.15031,
      "V_snapback_knee_V": 1.15031,
      "measured_jump_dec": 2.2016349538033824,
      "model_jump_dec": 0.0242535976052487,
      "snapback_reproduced": false
    }
  ],
  "snapback_reproduced_count": 0,
  "total_biases": 3,
  "rmse_summary_dec": [
    1.9697764851973347,
    1.0642496027489,
    0.8100924917706103
  ],
  "verdict": "pyport vs Sebas \u2014 snapback FOLD MISSING on 3/3 biases (VG2=+0.20). Sub-threshold OK; post-knee shape wrong. RMSE=1.97/1.06/0.81 dec."
}
```
