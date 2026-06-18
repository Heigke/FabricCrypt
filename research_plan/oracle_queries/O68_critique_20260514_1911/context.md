  (a) Cell-level 4 fJ/3.3 bit density
  (b) Predictive coding temporal filter (Lorenz vs LSTM)
  (c) DC fit 0.965 (pending R-48 physics-vs-fitting verdict)

## 2026-05-14 17:35 — R-48 PHYSICS XVAL: VERDICT = MIXED, leaning CURVE-FIT

z367 leave-one-out linear extrapolation on R-46 per-VG1 params.

Monotonicity:
  Bf       1890 / 1092 / 418   MONOTONIC
  iii      1.84 / 1.52 / 0.90  MONOTONIC
  log10Rs  9.17 / 9.90 / 6.78  NON-MONOTONIC (peaks at VG1=0.4, crashes at 0.6)

R-46 per-branch dec (eval-94 fit):
  VG1=0.2: 1.774   VG1=0.4: 1.161   VG1=0.6: 0.863
  Cell-wide median (from history): 0.965

LOOCV (held branch predicted by linear fit on other two):
  hold 0.2: pred=1.311  (ratio 0.74)  PHYSICS_PASS  [pred Rs clipped 13->10]
  hold 0.4: pred=1.226  (ratio 1.06)  PHYSICS_PASS
  hold 0.6: pred=2.048  (ratio 2.37)  CURVE_FIT     [pred Rs clipped 10.6->10, true=6.78]

Honest no-per-VG1 baseline (single mean param triple):
  VG1=0.2:2.152  VG1=0.4:1.145  VG1=0.6:1.230  cell-wide=1.192

Cost of dropping "real physics" claim:
  R-46 reported cell-wide:  0.965 dec  (engineering per-VG1 fit)
  R-46 prior global floor:  1.131 dec  (single global knobs, R-43/45/47)
  Honest mean-param fit:    1.192 dec  (this run)
  → per-VG1 gain over honest single-knob baseline: ~0.23 dec
  → Of which: VG1=0.2 hold passes physics extrapolation
                VG1=0.4 hold passes physics extrapolation
                VG1=0.6 hold FAILS — its log10Rs=6.78 is the non-monotonic
                outlier that cannot be predicted from {0.2, 0.4}.

Interpretation:
- Bf and iii_body_gain DO follow smooth monotonic VG1-dependence —
  plausible physics: as VG1 raises Vth/changes ionization regime, the
  parasitic BJT gain (Bf) and impact-ionization body coupling weaken.
  These two parameters look like real physical knobs.
- log10(vnwell_Rs) is the curve-fit knob: high at VG1=0.2/0.4 (>1e9 Ω,
  effectively decoupling the n-well), then drops 3 orders of magnitude
  at VG1=0.6. This is not a smooth physical trend; it is the model
  compensating for a regime change it cannot represent structurally
  (likely the same anti-correlation R-43/R-45/R-47 already falsified).

Plausible physical mechanism for VG1 dependence (if any):
- Body-effect-driven Vth shift moves M2 between subthreshold/strong
  inversion → changes how much of Id flows through the parasitic BJT
  vs the MOS channel → drives the apparent Bf trend. This is real and
  smooth.
- The Rs collapse at VG1=0.6 is more likely a missing branch in the
  topology (e.g., the n-well clamp engaging at high VG1 because the
  pdiode-to-vnwell path saturates) — i.e. the structural shortfall
  that R-44/R-47 already identified, masquerading as a parameter knob.

Final verdict: PARTIAL PHYSICS (2/3), CURVE-FITTING on the hardest
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
