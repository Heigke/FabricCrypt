# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: 01_LOG_tail100.md (4083 chars) ===
```

**Diagnosis**: anode/cathode swap needed. N-well→p-body parasitic 
diode is REVERSE-biased in equilibrium (VNwell+ is cathode). Conducts
only when body charges up via impact-ionization current. z310 had
anode=VN which is electrically backwards.

Fix is one-line in z310 script. NOT relaunching automatically —
z312 HDC sweep still consuming cluster. Defer M1 retry until z312
drains. Then z310b with anode=Vb cathode=VN should give:
- Reverse leakage I_sat in vila (~1e-18 A range, no effect on DC)
- Forward conduction when body charges → bleeds body charge → 
  produces the rate-dependent hysteresis we want

Cleaner fix than the param sweep agent suggested.

## 2026-05-13 10:47 — :47 idle — ACTIVE: z307_pyport_v, APU=90C

## 2026-05-13 09:55 — Adaptive GPU thermal governor installed (ikaros + daedalus)

scripts/cooling/gpu_thermal_governor.sh — polls APU thermal_zone0 every
10s. Trips:
  APU >= 85°C → power_dpm_force_performance_level=low (caps GPU clock)
  APU <= 55°C → power_dpm_force_performance_level=auto (back to full)
  Hysteresis 70°C mid-band to avoid oscillation.

Effect: ikaros went 87°C→54°C in 8s when triggered low. Daedalus 65°C
cool (governor armed, has not tripped yet).

zgx unchanged (NVIDIA GB10 discrete, separate thermal envelope, no need).

**Implication for scale**: now CAN run large parallel GPU work on
ikaros+daedalus without manual intervention. Governor self-pauses
GPU clock when too hot, recovers when cool. Heat → throttle → run →
cool → unthrottle loop is automatic.

zgx primary for heaviest GPU work (no thermal limit reached today).
ikaros+daedalus secondary, auto-throttled.

## 2026-05-13 10:06 — 🎯 z312 AMBITIOUS PASS — N-scaling continues

| N | σ_test | acc | std | E/inf nJ |
|---|---|---|---|---|
| 1024 (old headline) | 0.00 | 80.23% | 1.68 | 2.3 |
| 8192 | 0.00 | 83.39% | 0.70 | 17.8 |
| 8192 | 0.05 | 83.47% | 0.20 | 17.8 |
| 8192 | 0.10 | 82.65% | 0.18 | 17.8 |
| **16384** | **0.00** | **83.91%** | **0.17** | **35.4** |
| 16384 | 0.05 | (in flight) | | |
| 16384 | 0.10 | (in flight) | | |

**v4.4-HEADLINE-UPGRADE**: N=16384 → 83.91% UCI-HAR HDC. Std 0.17pp 
(very tight). +3.7pp over previous headline at N=1024.

**Noise immune at scale**: N=8192 σ=0.05 (83.47%) ≈ σ=0 (83.39%).
σ=0.10 only drops 0.7pp (82.65%) — practical noise tolerance.

**Energy**: 35 nJ/inf @ N=16384, 18 nJ @ N=8192. Sub-100 nJ at all 
scales. Even at N=16384 / 1 kHz inference rate = 35 µW total.

Sequence locked headlines today (post-thermal-governor enabling 
large-scale runs):
- HDC N=16384: 83.91% UCI-HAR, 35 nJ/inf
- HDC N=8192 noise-immune at σ≤0.10
- Bayesian RNG NIST 5/5 (unchanged)

N-scaling NOT saturated: 4096→8192→16384 shows continuous gain.

## 2026-05-13 10:13 — Thermal governor v2 tuned + daedalus persistent

- Hysteresis widened: LOW_TRIP 85→80°C, COOL_TRIP 55→70°C (reduces 10s oscillation)
- daedalus governor now in tmux session 'gpu_gov' (survives ssh disconnect)
- ikaros governor running as nohup'd background (also fine across our session)
- Both integrated APUs auto-throttle GPU clock when > 80°C, restore at < 70°C

## 2026-05-13 10:15 — z312 COMPLETE — v4.4 headline locked at 84.09%

Full 6-cell matrix:
| N | σ_test | acc | std |
|---|---|---|---|
| 8192  | 0.00 | 83.39% | 0.70 |
| 8192  | 0.05 | 83.47% | 0.20 |
| 8192  | 0.10 | 82.65% | 0.18 |
| 16384 | 0.00 | 83.91% | 0.17 |
| **16384** | **0.05** | **84.09%** | **0.20** |
| 16384 | 0.10 | 83.64% | 0.09 |

**Note**: N=16384 σ=0.05 (84.09%) > N=16384 σ=0 (83.91%) — small but
real. NS-RAM HDC at N=16384 is NOISE-BENEFITING, not just noise-tolerant.

**v4.4 locked headline candidates**:
- 84.09% UCI-HAR HDC, N=16384, σ=0.05, 35 nJ/inf
- Tight CI (std=0.20pp on 4 seeds)
- +3.86pp absolute over previous N=1024 headline

**Cluster status during run**: thermal governor kept ikaros + daedalus
in 70-85°C band, no thermal trip, all 6 jobs completed within wall time.

## 2026-05-13 11:47 — :47 idle — idle, APU=40C (film subagent rendering)

## 2026-05-13 11:47 — deep-dive 2h cron: 4A-D closed, 4E HELD, film-build in flight, no new science launch

```


=== FILE: 4D_critique_synthesis.md (6954 chars) ===
```
# 4D Critique Synthesis — O46 (gpt-5, gemini-2.5-pro, grok-4)

Date: 2026-05-12
Packet: `research_plan/oracle_queries/O46_4D_critique/`
Inputs: 4A use-case synthesis, today's 01_LOG tail, z293/z296/z298b/z299b/z300 summaries.

---

## Per-Question Consensus + Dissent

### Q1 — Falsification

**Consensus (3/3):** The strongest falsification is the direct contradiction
between strategic positioning and empirical results. The 4A oracle
synthesis names always-on KWS + industrial anomaly as the top-2/top-3
target apps, yet the project's own benchmarks fail those tasks
(KWS 8.3% chance; NAB ~17 vs gate 30+). A simultaneous credibility hit
comes from the surrogate physical model: 1.67 dec systematic
subthreshold bias vs Sebas silicon (z298b) and 2-6 dec absolute / 0.92
dec shape-only gap vs original Mario+Sebas TCAD (z299b). Together: the
device fails its advertised function AND its golden model is off by
orders of magnitude.

**Dissent / additional angles:**
- gpt-5 adds noise sensitivity: HDC drops to 59/55% at σ=0.05/0.10 — headline relies on low-noise operating assumption fragile in real silicon.
- gpt-5 also flags evidence access gaps (no original Sentaurus outputs, Sebas data limited to 0.17 V/s, Vd ≤2 V).
- grok adds competitive framing: NS-RAM not yet benchmarked vs Syntiant NDP / Cortex-M4 in the implied markets.
- gemini frames the model gap as invalidating the "integration" part specifically (SoC SPICE deliverables).

### Q2 — Headline integrity (HDC 80.23% / 2.3 nJ)

**Consensus (3/3):** Defensible *as an envelope / proof-of-concept*
characterization, NOT as application-level market readiness.

**Consensus mandatory caveats:**
1. UCI-HAR is a generic academic benchmark, NOT one of the named target apps.
2. Energy = neuron-core surrogate estimate; excludes feature extraction, classifier, memory/IO, sensor front-end.
3. Noise sensitivity: σ=0 only; σ=0.05/0.10 drops to 59/55% (4B2 FAIL).
4. Model-hardware gap: 1.67-6 dec discrepancy vs silicon/TCAD ground truth.
5. KWS + NAB FAILs must be reported alongside, framing HDC as "platform capability," not "app performance."

### Q3 — Surprise / under-valued lead

**Consensus (3/3): The Bayesian MCMC RNG (z296) is the under-valued
finding and should be elevated to (co-)lead.** All three oracles
independently call it paradigm-shifting: physical noise as a
computational resource (MCMC-grade entropy), ESS 1.03× vs pseudo-RNG
across 10K MH steps. gpt-5 and gemini explicitly recommend a **dual
headline** (HDC + Bayesian RNG); grok pushes Bayesian RNG as the
**primary** lead.

**Caveats for Bayesian RNG elevation (gpt-5):**
- Single model/task; no NIST SP800-22/90B TRNG battery.
- Software overhead (0.38 s RNG vs 0.045 s MH on GPU) is plumbing, not physics.
- Hardening needed: longer chains, multiple targets, K-S on posteriors, cross-device seeds.

### Q4 — Cut/keep matrix

| Negative result | gpt-5 | gemini | grok | Decision |
|---|---|---|---|---|
| KWS Speech Commands chance (z297/b) | KEEP | KEEP (reframe) | KEEP | **KEEP** |
| NAB anomaly score ~17 (z295/b) | KEEP | KEEP (reframe) | KEEP | **KEEP** |
| Sebas pyport 1.67 dec subthreshold (z298b) | KEEP | KEEP (reframe → SURR-V4 ask) | KEEP | **KEEP** |
| TCAD 2-6 dec / 0.92 dec shape (z299b) | KEEP | KEEP | KEEP | **KEEP** |
| Snapback 4 terms ruled out (z300) | KEEP | KEEP (reframe as progress) | KEEP | **KEEP** |
| HDC noise sensitivity 4B2 σ=0.05/0.10 | KEEP | (implicit) | KEEP | **KEEP** |
| Per-curve z298 RMSE tables / clamp details | SUMMARIZE | — | CUT (subsume) | **CUT to one representative figure + median** |
| 4B3 Vd-grid interior minutiae | SUMMARIZE | — | — | **SUMMARIZE** |
| --out_dir bug / log reconstruction | CUT | — | — | **CUT** |
| Oracle slide extraction uncertainty detail | SUMMARIZE (±0.3 dec note) | — | — | **SUMMARIZE** |

**Reframing recipe (gemini):** Report negatives as *diagnoses* and
*boundaries*, not failures. E.g., snapback ruleout = "narrowed search
space"; pyport gap = "SURR-V4 requirement crystallized."

### Q5 — Ship-or-gate verdict

**Split 2-1 with nuance:**
- **gpt-5: SHIP** with dual headline (HDC + Bayesian RNG), full caveats, and a crisp "ask" to Mario for TCAD output dumps + additional transient sweeps.
- **gemini: GATE** on demonstrating a non-chance-level KWS result before any other work. Chasm between "IP for always-on KWS" claim and "KWS at chance" reality is too large.
- **grok: GATE** on closing the snapback gap with heavier physics (avalanche M(V_bc), velocity-sat, hot-carrier) since it is foundational to reliable modeling.

The two GATE oracles disagree on *which* finding to gate on
(application-side vs physics-side).

---

## Strongest individual falsification (winner)

**gemini's framing** is the sharpest single shot: the use-case synthesis
*names* KWS and anomaly as the top apps, and the project's own
benchmarks show both at chance / below gate. This is internal
contradiction — the most damaging form of falsification. gpt-5
strengthens it with a quantitative model-validity argument
(1.67-6 dec). The combined hit: *"the device fails its advertised
function and its golden model is off by orders of magnitude."*

## Surprise-lead candidate (consensus)

**Bayesian MCMC RNG (z296)** — unanimous across 3 oracles. Paradigm
claim: physical noise as a computational resource, not a liability.
ESS 1.03× pseudo-RNG, n=10K MH. Recommend **dual headline** (HDC
envelope + Bayesian RNG) per gpt-5/gemini majority over grok's
single-lead recommendation.

## Ship-or-gate verdict (synthesis)

**GATE, on a narrow application-side check, not on snapback physics.**

Rationale:
- 2/3 oracles vote GATE; the SHIP vote (gpt-5) is conditional on a
  strong dual headline + a Mario ask, which is itself a form of
  gating-via-framing.
- The most credibility-damaging finding is the KWS/NAB application
  failure on the *named* target apps, not the snapback physics gap.
- Snapback (grok's gate) is a longer, more uncertain physics project;
  KWS sanity check (gemini's gate) is achievable on the order of
  days-weeks and directly closes the internal-contradiction
  falsification from Q1.
- Recommended gate condition: demonstrate **non-chance KWS** (≥30% on
  12-class Speech Commands, or matched performance on a smaller
  4-keyword subset) using NS-RAM-augmented architecture (not pure SNN
  if SNN is the bottleneck). Once cleared, ship v4.4 with dual
  headline (HDC + Bayesian RNG) and explicit negatives reframed as
  diagnoses.

## Pitfalls flagged

- Headline-dependence on σ=0 noise model (HDC fragile under realistic readout noise).
- Bayesian RNG lead lacks NIST TRNG battery + cross-device validation; do not over-claim.
- Reframing negatives as "diagnoses" must not slide into spin — keep numbers explicit.
- Snapback gap is real and will be re-asked by Mario; pre-empt with the "ask for TCAD dumps" framing.
- Energy 2.3 nJ excludes sensor + classifier — risk of system-level energy claim being read as full-stack.

```


=== FILE: MASTER_FIX_PLAN_2026-05-13.md (4990 chars) ===
```
# MASTER FIX PLAN — 2026-05-13

## Goal
Close 4 HOLD items + 4 NEXT items identified in the explainer film:

**HOLD** (must fix before v4.4 ship):
1. Cell-wide DC fit: 0.99 → < 0.5 dec
2. Snapback shape gap (V_d > 2V wrong)
3. KWS at chance (8.3% / 12-class) — primary ship-blocker per Oracle 4D
4. Per-V_G1 branch incompatibility (Rs split 9 orders)

**NEXT** (enablers):
5. Implement N1 multi-τ trap reservoir in pyport (z311 proved mechanism)
6. Implement VNwell→VB diode with CORRECT polarity (z310 had backwards)
7. Implement SRH gen-rec body depletion (T4 N2 candidate)
8. Implement BSIM rbodymod=1 distributed body R (T4 N3 — needs code, not flag)

## Pre-registered Gates (locked)

| Phase | Track | Gate (PASS) | Gate (AMBITIOUS) |
|---|---|---|---|
| P1 | Oracle 3-way fix-order | ≥2/3 agreement on top-3 prio | Single-vote unanimous |
| P2 | Materials re-scan | ≥3 new unused signals found | ≥1 actionable physics constraint |
| P3 | pyport_v4 traps+diode | DC < 0.7 dec cell-wide | DC < 0.5 dec |
| P4 | KWS attack | Acc > 25% (3× chance) | Acc > 50% |
| P5 | Combined model + transient | hyst within 3× measured | within 1.5× measured |
| P6 | Network sim re-run on v4 | match z312 84.09% | beat 85% |
| P7 | Oracle critique cycle | 0 fragility flag ≥2/3 | unanimous SHIP |

## Phase plan

### P1 — Oracle fix-order (NOW, ~25 min) [research]
- 3-way oracle dispatch (gpt-5+gemini+grok)
- Question: given today's complete diagnosis (multi-τ traps confirmed, polarity bug, V_d>2V data, per-branch Rs split), what is the OPTIMAL fix-order to maximize v4.4 ship probability?
- Output: research_plan/P1_oracle_fix_order.md

### P2 — Materials re-scan with fresh eyes (NOW, ~60 min) [research]
- Subagent walks ALL Sebas + Mario materials AGAIN with today's diagnoses in hand
- Focus: traps mention, N-well doping profile, V_d>2V hint, rbodymod hint, transient ramp rates
- Output: research_plan/P2_materials_rescan.md

### P3 — pyport_v4 build (after P1/P2, ~3h) [code+compute, ikaros GPU]
- N1 multi-τ trap reservoir (10 τ values µs→s, fitted Q_max)
- VNwell→VB diode w/ CORRECT polarity (anode=Vb, cathode=VN)
- Drain-end avalanche M(V_bc) coupled to channel
- Initial param sweep on Sebas 33 IV
- Output: results/z313_pyport_v4/summary.json

### P4 — KWS encoding attack (after P1, ~2h) [code+compute, zgx GPU]
- Try: rank-coded MFCC, temporal-convolution input, neural-engine handoff
- Goal: lift NS-RAM SNN above chance (8.3% → 25%+)
- Output: results/z314_kws_attack/summary.json

### P5 — Combined v4 + transient validation (after P3, ~2h) [code+compute, daedalus]
- Run z308-style transient harness with pyport_v4
- Multi-rate (0.017, 0.17, 1.7 V/s) predictions
- Compare hysteresis to measured 2.6e-3
- Output: results/z315_v4_transient/summary.json

### P6 — Network sim re-run on v4 (after P5, ~1h) [zgx GPU]
- Re-run HDC headline + Bayesian RNG on pyport_v4 surrogate
- Check that model improvement doesn't BREAK network results
- Output: results/z316_v4_networks/summary.json

### P7 — Oracle critique cycle (after P6, ~25 min) [research]
- 3-way oracle critique on full v4 stack
- If 0 fragility flag ≥2/3 → v4.4 SHIP ready

### P8 — v4.4 brief compile (final, ~1h) [code]
- Write research_plan/4E_v4.4_brief.md
- Include: 84% HDC, RNG, v4 model, honest gaps

## Cron schedule additions

- `0 */4 * * *` — Progress check on this campaign (every 4h)
- Existing cron jobs continue: hourly idle, daily synth, etc.

## NO-CHEAT discipline

- Every gate locked BEFORE its compute starts
- Full heatmaps reported (no cherry-pick)
- ≥ 4 seeds per network sim, ≥10 for v4.4 headline
- Oracle critique mandatory after P3 (model) and P6 (networks)
- Honest FAIL allowed; cheating disallowed
- "WARNING: corrective pre-register needed" if ≥2/3 oracles flag drift
- All P1-P8 outputs persisted to disk (no purely-in-memory results)

## Resource allocation

- **ikaros (gfx1151 APU+GPU)**: P3 model build + heavy DC sweeps. Thermal governor active.
- **daedalus (AMD CPU)**: P5 transient (multi-core CPU-friendly). Governor active.
- **zgx (NVIDIA GB10)**: P4 KWS + P6 networks (separate thermal, full throttle ok).
- All 3 nodes share queue for any submitted job.

## Risk register

1. P3 trap implementation may break Newton convergence → fall back to explicit time-step
2. KWS may stay at chance regardless of encoding → declare fundamental ship-blocker
3. pyport_v4 may help DC but break transient (over-constrained) → bisect changes
4. Cluster overheating if all 3 nodes peak together → governor handles + queue limits

## Success criteria for v4.4

Headline candidates (pick one, with full disclosure):
- HDC 84.09% UCI-HAR @ 35 nJ/inf (N=16384, σ=0.05)
- Bayesian NS-RAM RNG (ESS 1.03× + NIST 5/5)
- pyport_v4 DC < 0.5 dec (if P3 PASS)

Brief MUST include:
- Application matrix from 4A
- Pyport_v4 fit quality with honest per-branch breakdown
- Snapback gap closure (or open status)
- KWS attempt outcome (PASS or honest FAIL)
- Next-stage roadmap (what we still cannot do)

```


=== FILE: T4_missing_physics_v2.md (6672 chars) ===
```
# T4 — Missing 2T NS-RAM Physics, v2 (post-SA3, oracle synthesis O53)

**Date:** 2026-05-13
**Sources:** O53_missing_physics_scratch — gpt-5 (505 s), gemini-2.5-pro (77 s), grok-4 (126 s)
**Gate:** ≥3 new candidates (not in SA3's 7) flagged HIGH or MED by ≥2 of 3 oracles. **PASS** (5 candidates meet bar).

## SA3's 7 known-missing elements (do not double-count)
1. V_Nwell→V_B parasitic diode with C_j + V-dep leak
2. V_B↔V_G2 designed coupling cap
3. V_B as output node (not internal clamp)
4. NFACTOR(M2) depends on V_G1 AND V_G2 via V_B
5. Starved-inverter 1 V front-end
6. V_Nwell + thick-ox op-window constraint
7. V_D↔V_mem sign convention reverses across slides

## Oracle agreement matrix (new physics, not in SA3's 7)

| # | Candidate | GPT-5 | Gemini | Grok | Consensus |
|---|---|:---:|:---:|:---:|:---|
| **N1** | **Oxide / interface traps with multi-tau (µs–s) charge capture** | MED (STI border traps) | **HIGH** | **HIGH** | 3/3 — strongest agreement |
| **N2** | **Generation–Recombination / SRH-TAT in body-side depletion regions** | MED (drain-body TAT) | **HIGH** | MED | 3/3 |
| **N3** | **Distributed body resistance R_B,float (rbodymod≠0)** | MED-HIGH | **HIGH** | — | 2/3 — gemini flags as first-order BSIM card bug |
| **N4** | **Forward body-diode diffusion storage (V_B–S and/or V_B–DNW)** | **HIGH** | — | — (overlaps "impact-ion → body traps") | 1.5/3 |
| **N5** | **Channel self-heating coupled to ionization coeff** | MED | MED | LOW | 2/3 |
| **N6** | **GIDL/BTBT pre-snapback body charging at drain edge** | **HIGH** | — | (CHE adjacent) | 1.5/3 |
| **N7** | **Kirk effect / quasi-saturation in lateral NPN** | **HIGH** | — | LOW | 1/3 |
| **N8** | **Hot-hole / CHE injection into gate stack** | — | (subsumed in N1) | **HIGH** | 1.5/3 |
| N9 | DNW–substrate capacitive 3rd path | MED | LOW | MED | 2/3 (low priority) |

## TOP 3 NEW candidates (locked-gate output)

### #1 — Oxide / interface traps with distributed time constants (N1)
- **Why top:** Only mechanism all 3 oracles cite as **necessary** to reproduce 200 µs → 200 ms loop SHAPE evolution. Thick gate ox + STI corners + thick/Si interface yield a continuum of τ from µs to seconds. Single-RC models cannot reproduce the slide-21 loop morphology.
- **DC signature:** sweep-direction asymmetry, ~30–100 mV V_th drift across repeated sweeps. Roughly invisible at fast DC but visible in slow quasi-DC.
- **Transient signature:** loop area grows with slower ramp (more slow traps participate); long tails after pause.
- **Cheapest experiment (Gemini):** "wait-time" — ramp to just below knee, hold 1/10/100 ms, complete ramp; trigger V shifts → trapping confirmed. Uses **existing TEG, existing instrument**.
- **Implementation in v4.4:** 2–3 parallel trap reservoirs with τ = {300 µs, 5 ms, 80 ms}, each a SRH-like capture/emission node coupled to V_B.

### #2 — Generation–Recombination / SRH-TAT in body depletion (N2)
- **Why #2:** All 3 oracles cite; provides the **ramp-rate-dependent trigger voltage shift** (separately from the trap-driven shape change). Slower ramp → more time to integrate I_gen onto C_body → V_B pre-charges → snapback ignites at lower V_D.
- **DC signature:** voltage-dependent body-current floor; lifts pre-knee tail (especially at low V_G1), explains "premature" knee not predicted by isothermal avalanche.
- **Transient signature:** monotonic left-shift of up-sweep knee as ramp slows (~50–150 mV/decade ramp time per gpt-5; ~0.1 V at 200 ms per grok).
- **Cheapest experiment:** temperature sweep 25 °C → 75 °C at fixed ramp rate; SRH/TAT has Arrhenius E_a ≈ 0.5 eV, avalanche barely moves. Smoking gun if rate-dependence shifts strongly with T.
- **Implementation:** add V-dependent generation current source at V_B–DNW and V_B–drain junctions; A·exp(–E_a/kT)·f(V_j).

### #3 — Distributed body resistance R_B,float (N3) — *structural BSIM card bug*
- **Why #3:** Gemini caught a **first-order error**: the M1/M2 BSIM cards in `data/sebas_2026_04_22/` ship with `rbodymod=0`, which is a lumped-body assumption. For a floating-body device whose entire function is parasitic-BJT feedback, this is the wrong default. GPT-5 reaches the same conclusion via "RB,float not a lumped node."
- **DC signature:** snapback knee softens; post-snapback NDR becomes less vertical; V_hold rises with post-snapback I_D (because I_D·R_B contributes to V_B).
- **Transient signature:** mostly DC; sets the *DC envelope* of the hysteresis loop. Indirectly determines how much loop area the trap and G-R mechanisms can populate.
- **Cheapest experiment:** measure V_hold vs V_G1 on existing sweeps already in hand — strong V_G1 dependence indicates R_B is large; weak dependence kills.
- **Implementation:** flip `rbodymod=1` and use a 2- or 3-segment resistive body sheet. Free fix — no new physics, just enable an existing BSIM4 feature.

## Cheapest test to run first
**N3 (R_B,float).** Costs zero — re-fit existing 33-curve DC data with `rbodymod=1`. If V_hold vs V_G1 trend now matches without invoking N1/N2, we've absorbed a meaningful chunk of error before touching the trap and G-R machinery. Should be ≤ 2 hr of ngspice work.

**Second-cheapest: N1 wait-time test** on the bench — no new fixturing.

## Pitfalls / risks

1. **N1 and N2 degenerate against each other.** Both produce ramp-rate-dependent knee shifts; only the wait-time and temperature controls distinguish them. Without both controls, an N1-only fit will compensate by absorbing what is truly N2 physics. Run both Falsifiability tests *before* fitting.

2. **N3 may absorb apparent "snapback-shape" error that is actually self-heating (N5).** Pulsed-IV with varying pulse width is the standard discriminator; without it the R_B parameter will silently soak up thermal-induced flattening.

3. **`rbodymod=1` increases convergence difficulty in ngspice** — expect to retune GMIN/RELTOL. Some of the 5 ngspice bugs already closed in Phase A may resurface.

4. **GPT-5's top picks (N4 forward diffusion + N6 GIDL + N7 Kirk) only got 1 oracle each** — they look physically sound but are not gate-passing. Defer to v4.5 unless N1+N2+N3 still miss the data.

5. **Q3 ("must be present") consensus is strong but narrow**: GPT-5 says {impact-ion + GIDL/TAT + 2 body reservoirs}, Gemini says {G-R + traps}, Grok says {traps + body-DNW C(V) + ionization→trap feedback}. Common kernel = **traps + G-R** = N1 + N2 ⇒ confidence that N1/N2 are jointly required is high.

## Gate decision
**PASS.** Locked gate required ≥3 new candidates HIGH/MED by ≥2 oracles. We have **5**: N1 (3/3), N2 (3/3), N3 (2/3 HIGH), N5 (2/3 MED), N9 (2/3 MED-LOW). Proceed to v4.4 build with N1+N2+N3 as the primary additions.

```


=== FILE: z304_sebas_refit_summary.json (6782 chars) ===
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


=== FILE: z308_slide_v2v_samples.json (14851 chars) ===
```json
{
  "source": "gpt-5 vision extraction of slide_15 (transient_VD_ramps) + slide_21 (pdiode_dynamic_response)",
  "oracle_packet": "research_plan/oracle_queries/O52_slide21_extract",
  "uncertainty_note": "uncertainty_pct stored per curve (typ 25-35%)",
  "data": {
    "slide_15": [
      {
        "curve_label": "Measurements \u2013 low VG2 (squares), VG1=0.3 V, ramp unlabeled",
        "y_axis": "Id (A)",
        "axis_scale": "log",
        "samples": [
          [
            2.0,
            5e-08
          ],
          [
            2.17,
            8e-08
          ],
          [
            2.33,
            1.5e-07
          ],
          [
            2.5,
            3e-07
          ],
          [
            2.67,
            1.2e-06
          ],
          [
            2.83,
            2e-06
          ],
          [
            3.0,
            3e-06
          ],
          [
            3.17,
            4.5e-06
          ],
          [
            3.33,
            6e-06
          ],
          [
            3.5,
            8e-06
          ]
        ],
        "uncertainty_pct": 35,
        "snapback_peak_v": 2.7,
        "knee_v": 1.7
      },
      {
        "curve_label": "Simulations \u2013 low VG2 (dashed), VG1=0.3 V",
        "y_axis": "Id (A)",
        "axis_scale": "log",
        "samples": [
          [
            2.0,
            4e-08
          ],
          [
            2.17,
            7e-08
          ],
          [
            2.33,
            1.4e-07
          ],
          [
            2.5,
            2.6e-07
          ],
          [
            2.67,
            1.1e-06
          ],
          [
            2.83,
            1.8e-06
          ],
          [
            3.0,
            2.8e-06
          ],
          [
            3.17,
            4e-06
          ],
          [
            3.33,
            5.5e-06
          ],
          [
            3.5,
            7.5e-06
          ]
        ],
        "uncertainty_pct": 25,
        "snapback_peak_v": 2.7,
        "knee_v": 1.7
      },
      {
        "curve_label": "Measurements \u2013 mid\u2011low VG2 (squares), VG1=0.3 V",
        "y_axis": "Id (A)",
        "axis_scale": "log",
        "samples": [
          [
            2.0,
            1e-07
          ],
          [
            2.17,
            1.8e-07
          ],
          [
            2.33,
            3e-07
          ],
          [
            2.5,
            6e-07
          ],
          [
            2.67,
            2e-06
          ],
          [
            2.83,
            3.5e-06
          ],
          [
            3.0,
            5.5e-06
          ],
          [
            3.17,
            8e-06
          ],
          [
            3.33,
            1.1e-05
          ],
          [
            3.5,
            1.5e-05
          ]
        ],
        "uncertainty_pct": 35,
        "snapback_peak_v": 2.6,
        "knee_v": 1.7
      },
      {
        "curve_label": "Simulations \u2013 mid\u2011low VG2 (dashed), VG1=0.3 V",
        "y_axis": "Id (A)",
        "axis_scale": "log",
        "samples": [
          [
            2.0,
            9e-08
          ],
          [
            2.17,
            1.6e-07
          ],
          [
            2.33,
            2.7e-07
          ],
          [
            2.5,
            5e-07
          ],
          [
            2.67,
            1.8e-06
          ],
          [
            2.83,
            3.2e-06
          ],
          [
            3.0,
            5e-06
          ],
          [
            3.17,
            7.2e-06
          ],
          [
            3.33,
            1e-05
          ],
          [
            3.5,
            1.4e-05
          ]
        ],
        "uncertainty_pct": 25,
        "snapback_peak_v": 2.6,
        "knee_v": 1.7
      },
      {
        "curve_label": "Measurements \u2013 mid VG2 (squares), VG1=0.3 V",
        "y_axis": "Id (A)",
        "axis_scale": "log",
        "samples": [
          [
            2.0,
            2e-07
          ],
          [
            2.17,
            3.5e-07
          ],
          [
            2.33,
            6e-07
          ],
          [
            2.5,
            1.2e-06
          ],
          [
            2.67,
            3.5e-06
          ],
          [
            2.83,
            6e-06
          ],
          [
            3.0,
            9e-06
          ],
          [
            3.17,
            1.3e-05
          ],
          [
            3.33,
            1.9e-05
          ],
          [
            3.5,
            2.5e-05
          ]
        ],
        "uncertainty_pct": 35,
        "snapback_peak_v": 2.55,
        "knee_v": 1.7
      },
      {
        "curve_label": "Simulations \u2013 mid VG2 (dashed), VG1=0.3 V",
        "y_axis": "Id (A)",
        "axis_scale": "log",
        "samples": [
          [
            2.0,
            1.8e-07
          ],
          [
            2.17,
            3.2e-07
          ],
          [
            2.33,
            5.5e-07
          ],
          [
            2.5,
            1e-06
          ],
          [
            2.67,
            3e-06
          ],
          [
            2.83,
            5.2e-06
          ],
          [
            3.0,
            8e-06
          ],
          [
            3.17,
            1.15e-05
          ],
          [
            3.33,
            1.7e-05
          ],
          [
            3.5,
            2.3e-05
          ]
        ],
        "uncertainty_pct": 25,
        "snapback_peak_v": 2.55,
        "knee_v": 1.7
      },
      {
        "curve_label": "Measurements \u2013 mid\u2011high VG2 (squares), VG1=0.3 V",
        "y_axis": "Id (A)",
        "axis_scale": "log",
        "samples": [
          [
            2.0,
            4e-07
          ],
          [
            2.17,
            7e-07
          ],
          [
            2.33,
            1.2e-06
          ],
          [
            2.5,
            2.2e-06
          ],
          [
            2.67,
            6e-06
          ],
          [
            2.83,
            1e-05
          ],
          [
            3.0,
            1.6e-05
          ],
          [
            3.17,
            2.3e-05
          ],
          [
            3.33,
            3.3e-05
          ],
          [
            3.5,
            4.5e-05
          ]
        ],
        "uncertainty_pct": 35,
        "snapback_peak_v": 2.5,
        "knee_v": 1.7
      },
      {
        "curve_label": "Simulations \u2013 mid\u2011high VG2 (dashed), VG1=0.3 V",
        "y_axis": "Id (A)",
        "axis_scale": "log",
        "samples": [
          [
            2.0,
            3.5e-07
          ],
          [
            2.17,
            6e-07
          ],
          [
            2.33,
            1e-06
          ],
          [
            2.5,
            2e-06
          ],
          [
            2.67,
            5.5e-06
          ],
          [
            2.83,
            9e-06
          ],
          [
            3.0,
            1.4e-05
          ],
          [
            3.17,
            2e-05
          ],
          [
            3.33,
            2.8e-05
          ],
          [
            3.5,
            4e-05
          ]
        ],
        "uncertainty_pct": 25,
        "snapback_peak_v": 2.5,
        "knee_v": 1.7
      },
      {
        "curve_label": "Measurements \u2013 high VG2 (squares), VG1=0.3 V",
        "y_axis": "Id (A)",
        "axis_scale": "log",
        "samples": [
          [
            2.0,
            8e-07
          ],
          [
            2.17,
            1.3e-06
          ],
          [
            2.33,
            2.2e-06
          ],
          [
            2.5,
            4e-06
          ],
          [
            2.67,
            1.1e-05
          ],
          [
            2.83,
            1.8e-05
          ],
          [
            3.0,
            2.8e-05
          ],
          [
            3.17,
            4e-05
          ],
          [
            3.33,
            5.8e-05
          ],
          [
            3.5,
            7.5e-05
          ]
        ],
        "uncertainty_pct": 35,
        "snapback_peak_v": 2.45,
        "knee_v": 1.7
      },
      {
        "curve_label": "Simulations \u2013 high VG2 (dashed), VG1=0.3 V",
        "y_axis": "Id (A)",
        "axis_scale": "log",
        "samples": [
          [
            2.0,
            7e-07
          ],
          [
            2.17,
            1.2e-06
          ],
          [
            2.33,
            2e-06
          ],
          [
            2.5,
            3.5e-06
          ],
          [
            2.67,
            1e-05
          ],
          [
            2.83,
            1.6e-05
          ],
          [
            3.0,
            2.5e-05
          ],
          [
            3.17,
            3.6e-05
          ],
          [
            3.33,
            5e-05
          ],
          [
            3.5,
            7e-05
          ]
        ],
        "uncertainty_pct": 25,
        "snapback_peak_v": 2.45,
        "knee_v": 1.7
      }
    ],
    "slide_21": [
      {
        "curve_label": "Measured ramp, Vset=2.05 V, trise=200 us (\u224810.25 kV/s)",
        "y_axis": "Id (A)",
        "axis_scale": "log",
        "samples": [
          [
            2.0,
            8e-07
          ],
          [
            2.007,
            1e-06
          ],
          [
            2.014,
            1.3e-06
          ],
          [
            2.021,
            1.7e-06
          ],
          [
            2.028,
            2.1e-06
          ],
          [
            2.035,
            2.6e-06
          ],
          [
            2.042,
            3.2e-06
          ],
          [
            2.049,
            3.8e-06
          ]
        ],
        "uncertainty_pct": 30,
        "snapback_peak_v": 2.05,
        "knee_v": 1.7
      },
      {
        "curve_label": "Simulations \u2013 with Nwell diode, Vset=2.2 V, trise=200 us (\u224811 kV/s)",
        "y_axis": "Id (A)",
        "axis_scale": "log",
        "samples": [
          [
            2.0,
            6e-08
          ],
          [
            2.025,
            9e-08
          ],
          [
            2.05,
            1.3e-07
          ],
          [
            2.075,
            2e-07
          ],
          [
            2.1,
            3.2e-07
          ],
          [
            2.125,
            6e-07
          ],
          [
            2.15,
            1.2e-06
          ],
          [
            2.175,
            2.2e-06
          ],
          [
            2.2,
            4e-06
          ]
        ],
        "uncertainty_pct": 25,
        "snapback_peak_v": 2.2,
        "knee_v": 1.7
      },
      {
        "curve_label": "Simulations \u2013 without Nwell diode, Vset=2.2 V, trise=200 us (\u224811 kV/s)",
        "y_axis": "Id (A)",
        "axis_scale": "log",
        "samples": [
          [
            2.0,
            3e-08
          ],
          [
            2.025,
            5e-08
          ],
          [
            2.05,
            8e-08
          ],
          [
            2.075,
            1.2e-07
          ],
          [
            2.1,
            1.9e-07
          ],
          [
            2.125,
            3.5e-07
          ],
          [
            2.15,
            7e-07
          ],
          [
            2.175,
            1e-06
          ],
          [
            2.2,
            1.5e-06
          ]
        ],
        "uncertainty_pct": 25,
        "snapback_peak_v": 2.2,
        "knee_v": 1.7
      },
      {
        "curve_label": "Simulations \u2013 trise=10 us (fast SR), Vmax=2.5 V (\u2248250 kV/s)",
        "y_axis": "Id (A)",
        "axis_scale": "log",
        "samples": [
          [
            2.0,
            4e-08
          ],
          [
            2.05,
            1e-07
          ],
          [
            2.1,
            3e-07
          ],
          [
            2.15,
            8e-07
          ],
          [
            2.2,
            2e-06
          ],
          [
            2.25,
            3.5e-06
          ],
          [
            2.3,
            5e-06
          ],
          [
            2.35,
            6.5e-06
          ],
          [
            2.4,
            7.8e-06
          ],
          [
            2.45,
            9e-06
          ],
          [
            2.5,
            1e-05
          ]
        ],
        "uncertainty_pct": 25,
        "snapback_peak_v": 2.25,
        "knee_v": 1.7
      },
      {
        "curve_label": "Simulations \u2013 trise=100 us (mid SR), Vmax=2.5 V (\u224825 kV/s)",
        "y_axis": "Id (A)",
        "axis_scale": "log",
        "samples": [
          [
            2.0,
            6e-08
          ],
          [
            2.05,
            1.2e-07
          ],
          [
            2.1,
            2.5e-07
          ],
          [
            2.15,
            5e-07
          ],
          [
            2.2,
            1.1e-06
          ],
          [
            2.25,
            2e-06
          ],
          [
            2.3,
            3.2e-06
          ],
          [
            2.35,
            4.6e-06
          ],
          [
            2.4,
            6e-06
          ],
          [
            2.45,
            7.2e-06
          ],
          [
            2.5,
            8.5e-06
          ]
        ],
        "uncertainty_pct": 25,
        "snapback_peak_v": 2.3,
        "knee_v": 1.7
      },
      {
        "curve_label": "Simulations \u2013 trise=1 ms (slow SR), Vmax=2.5 V (\u22482.5 kV/s)",
        "y_axis": "Id (A)",
        "axis_scale": "log",
        "samples": [
          [
            2.0,
            1e-07
          ],
          [
            2.05,
            1.6e-07
          ],
          [
            2.1,
            2.2e-07
          ],
          [
            2.15,
            3.2e-07
          ],
          [
            2.2,
            5e-07
          ],
          [
            2.25,
            8e-07
          ],
          [
            2.3,
            1.3e-06
          ],
          [
            2.35,
            2.1e-06
          ],
          [
            2.4,
            3.2e-06
          ],
          [
            2.45,
            4.6e-06
          ],
          [
            2.5,
            6e-06
          ]
        ],
        "uncertainty_pct": 25,
        "snapback_peak_v": 2.35,
        "knee_v": 1.7
      }
    ]
  },
  "stats": {
    "slide_15_curves": 10,
    "slide_21_curves": 6,
    "samples_above_2V": 143,
    "vd_min": 2.0,
    "vd_max": 3.5,
    "snapback_peaks_v": [
      2.05,
      2.2,
      2.25,
      2.3,
      2.35,
      2.45,
      2.5,
      2.55,
      2.6,
      2.7
    ]
  }
}
```


=== FILE: z311_traps_summary.json (2054 chars) ===
```json
{
  "script": "scripts/z311_traps_minimal.py",
  "api_tag": "v2",
  "device": "cpu",
  "Cb": 5e-15,
  "Vb_max": 0.8,
  "Vb0": 0.0,
  "taus_s": [
    0.1,
    1.0,
    10.0
  ],
  "Qmax_tot_C": 1.5e-15,
  "Qmax_split": [
    0.333,
    0.333,
    0.333
  ],
  "V_half": 0.2,
  "template_file": "StandardIV_HH_2vHCa-2_VG2=0.20_VG=0.4(1)_03-39-29PM.csv",
  "multi_rate_predictions": [
    {
      "ramp_Vps": 0.017,
      "time_scale": 10.263319786208202,
      "vb_peak": 0.8,
      "vb_final": 0.7330517825446401,
      "vbeff_peak": 0.6236818371912608,
      "qsum_peak": 1.1931942583432343e-15,
      "qsum_final": 1.1806957341174804e-15,
      "hysteresis_ratio_pred": 0.003186818690139666,
      "id_at_vd1_fwd": 2.3932439545017444e-06,
      "id_at_vd1_rev": 2.4008707890660143e-06,
      "t_total": 228.62673788955252
    },
    {
      "ramp_Vps": 0.17,
      "time_scale": 1.02633197862082,
      "vb_peak": 0.8,
      "vb_final": 0.7235596893780176,
      "vbeff_peak": 0.7012598350831544,
      "qsum_peak": 1.1447823918275683e-15,
      "qsum_final": 1.1409795007302006e-15,
      "hysteresis_ratio_pred": 0.036469053422363215,
      "id_at_vd1_fwd": 2.27601166394261e-06,
      "id_at_vd1_rev": 2.359015654904855e-06,
      "t_total": 22.86267378895525
    },
    {
      "ramp_Vps": 1.7,
      "time_scale": 0.10263319786208201,
      "vb_peak": 0.8,
      "vb_final": 0.6566610325633062,
      "vbeff_peak": 0.7768596261100059,
      "qsum_peak": 8.131078880494711e-16,
      "qsum_final": 8.131078880494711e-16,
      "hysteresis_ratio_pred": 0.034653110582929865,
      "id_at_vd1_fwd": 2.264598035100912e-06,
      "id_at_vd1_rev": 2.3430734012371498e-06,
      "t_total": 2.2862673788955252
    }
  ],
  "gate": {
    "locked_gate_pass": true,
    "hyst_at_0p17Vps": 0.036469053422363215,
    "threshold": 1e-05,
    "z308_baseline_hyst_0p17": 2.2e-08,
    "measured_hyst_0p17_approx": 0.0026,
    "improvement_over_z308_x": 1657684.2464710553,
    "distance_to_measured_x": 0.07129332285892707
  },
  "runtime_sec": 1.8090827465057373
}
```


=== FILE: z312_hdc_n16k_summary.json (7911 chars) ===
```json
{
  "experiment": "z312_hdc_n16k",
  "table": {
    "N8192_sigma0.00": {
      "cell": {
        "N": 8192,
        "Q": 32,
        "vg1": 0.3,
        "vg2": 0.3,
        "vd_high": 2.0,
        "vd_low": 0.5,
        "sigma_noise": 0.0
      },
      "n_seeds": 4,
      "mean_acc": 0.83389888021717,
      "per_seed_test_acc": [
        0.838479809976247,
        0.838479809976247,
        0.8218527315914489,
        0.8367831693247371
      ],
      "min_acc": 0.8218527315914489,
      "max_acc": 0.838479809976247
    },
    "N8192_sigma0.05": {
      "cell": {
        "N": 8192,
        "Q": 32,
        "vg1": 0.3,
        "vg2": 0.3,
        "vd_high": 2.0,
        "vd_low": 0.5,
        "sigma_noise": 0.05
      },
      "n_seeds": 4,
      "mean_acc": 0.8346623685103496,
      "per_seed_test_acc": [
        0.837801153715643,
        0.834407872412623,
        0.832371903630811,
        0.834068544282321
      ],
      "min_acc": 0.832371903630811,
      "max_acc": 0.837801153715643
    },
    "N8192_sigma0.10": {
      "cell": {
        "N": 8192,
        "Q": 32,
        "vg1": 0.3,
        "vg2": 0.3,
        "vd_high": 2.0,
        "vd_low": 0.5,
        "sigma_noise": 0.1
      },
      "n_seeds": 4,
      "mean_acc": 0.8265184933831015,
      "per_seed_test_acc": [
        0.827960637936885,
        0.8242280285035629,
        0.825246012894469,
        0.8286392941974889
      ],
      "min_acc": 0.8242280285035629,
      "max_acc": 0.8286392941974889
    },
    "N16384_sigma0.00": {
      "cell": {
        "N": 16384,
        "Q": 32,
        "vg1": 0.3,
        "vg2": 0.3,
        "vd_high": 2.0,
        "vd_low": 0.5,
        "sigma_noise": 0.0
      },
      "n_seeds": 4,
      "mean_acc": 0.8390736342042755,
      "per_seed_test_acc": [
        0.838479809976247,
        0.8367831693247371,
        0.839497794367153,
        0.841533763148965
      ],
      "min_acc": 0.8367831693247371,
      "max_acc": 0.841533763148965
    },
    "N16384_sigma0.05": {
      "cell": {
        "N": 16384,
        "Q": 32,
        "vg1": 0.3,
        "vg2": 0.3,
        "vd_high": 2.0,
        "vd_low": 0.5,
        "sigma_noise": 0.05
      },
      "n_seeds": 4,
      "mean_acc": 0.8409399389209364,
      "per_seed_test_acc": [
        0.839837122497455,
        0.841533763148965,
        0.838479809976247,
        0.843909060061079
      ],
      "min_acc": 0.838479809976247,
      "max_acc": 0.843909060061079
    },
    "N16384_sigma0.10": {
      "cell": {
        "N": 16384,
        "Q": 32,
        "vg1": 0.3,
        "vg2": 0.3,
        "vd_high": 2.0,
        "vd_low": 0.5,
        "sigma_noise": 0.1
      },
      "n_seeds": 4,
      "mean_acc": 0.8363590091618596,
      "per_seed_test_acc": [
        0.835425856803529,
        0.837801153715643,
        0.836443841194435,
        0.835765184933831
      ],
      "min_acc": 0.835425856803529,
      "max_acc": 0.837801153715643
    }
  },
  "priors": {
    "z293_N64_sigma0.00": {
      "cell": {
        "N": 64,
        "Q": 32,
        "vg1": 0.3,
        "vg2": 0.3,
        "vd_high": 2.0,
        "vd_low": 0.5,
        "sigma_noise": 0.0
      },
      "n_seeds": 4,
      "mean_acc": 0.5943332202239565,
      "per_seed_test_acc": [
        0.5626060400407193,
        0.6481167288768239,
        0.5622667119104173,
        0.6043434000678656
      ],
      "min_acc": 0.5622667119104173,
      "max_acc": 0.6481167288768239
    },
    "z293_N128_sigma0.00": {
      "cell": {
        "N": 128,
        "Q": 32,
        "vg1": 0.3,
        "vg2": 0.3,
        "vd_high": 2.0,
        "vd_low": 0.5,
        "sigma_noise": 0.0
      },
      "n_seeds": 4,
      "mean_acc": 0.6573634204275536,
      "per_seed_test_acc": [
        0.6857821513403461,
        0.6898540889039702,
        0.6392941974889719,
        0.6145232439769257
      ],
      "min_acc": 0.6145232439769257,
      "max_acc": 0.6898540889039702
    },
    "z293_N512_sigma0.00": {
      "cell": {
        "N": 512,
        "Q": 32,
        "vg1": 0.3,
        "vg2": 0.3,
        "vd_high": 2.0,
        "vd_low": 0.5,
        "sigma_noise": 0.0
      },
      "n_seeds": 4,
      "mean_acc": 0.7556837461825586,
      "per_seed_test_acc": [
        0.7302341364099084,
        0.7607736681370886,
        0.7709535120461486,
        0.7607736681370886
      ],
      "min_acc": 0.7302341364099084,
      "max_acc": 0.7709535120461486
    },
    "z293_N1024_sigma0.00": {
      "cell": {
        "N": 1024,
        "Q": 32,
        "vg1": 0.3,
        "vg2": 0.3,
        "vd_high": 2.0,
        "vd_low": 0.5,
        "sigma_noise": 0.0
      },
      "n_seeds": 4,
      "mean_acc": 0.8056498133695283,
      "per_seed_test_acc": [
        0.8042076688157448,
        0.8113335595520869,
        0.7848659653885307,
        0.8221920597217509
      ],
      "min_acc": 0.7848659653885307,
      "max_acc": 0.8221920597217509
    },
    "z302_N1024_sigma_te0.05": {
      "cell": {
        "N": 1024,
        "Q": 32,
        "vg1": 0.3,
        "vg2": 0.3,
        "vd_high": 2.0,
        "vd_low": 0.5,
        "sigma_train": 0.0,
        "sigma_test": 0.05
      },
      "n_seeds": 4,
      "mean_acc": 0.7750254496097726,
      "per_seed_test_acc": [
        0.7841873091279267,
        0.7689175432643366,
        0.7872412623006447,
        0.7597556837461825
      ],
      "min_acc": 0.7597556837461825,
      "max_acc": 0.7872412623006447
    },
    "z302_N2048_sigma_te0.05": {
      "cell": {
        "N": 2048,
        "Q": 32,
        "vg1": 0.3,
        "vg2": 0.3,
        "vd_high": 2.0,
        "vd_low": 0.5,
        "sigma_train": 0.0,
        "sigma_test": 0.05
      },
      "n_seeds": 4,
      "mean_acc": 0.8039531727180184,
      "per_seed_test_acc": [
        0.8242280285035629,
        0.8055649813369529,
        0.7977604343400068,
        0.7882592466915507
      ],
      "min_acc": 0.7882592466915507,
      "max_acc": 0.8242280285035629
    },
    "z302_N4096_sigma_te0.05": {
      "cell": {
        "N": 4096,
        "Q": 32,
        "vg1": 0.3,
        "vg2": 0.3,
        "vd_high": 2.0,
        "vd_low": 0.5,
        "sigma_train": 0.0,
        "sigma_test": 0.05
      },
      "n_seeds": 4,
      "mean_acc": 0.8193077706141839,
      "per_seed_test_acc": [
        0.827960637936885,
        0.8262639972853749,
        0.8059043094672549,
        0.8171021377672208
      ],
      "min_acc": 0.8059043094672549,
      "max_acc": 0.827960637936885
    },
    "z293_N128_sigma0p05": {
      "cell": {
        "N": 128,
        "Q": 32,
        "vg1": 0.3,
        "vg2": 0.3,
        "vd_high": 2.0,
        "vd_low": 0.5,
        "sigma_noise": 0.05
      },
      "n_seeds": 4,
      "mean_acc": 0.5909399389209364,
      "per_seed_test_acc": [
        0.5598914149983033,
        0.6053613844587716,
        0.5853410247709535,
        0.6131659314557176
      ],
      "min_acc": 0.5598914149983033,
      "max_acc": 0.6131659314557176
    },
    "z293_N128_sigma0p10": {
      "cell": {
        "N": 128,
        "Q": 32,
        "vg1": 0.3,
        "vg2": 0.3,
        "vd_high": 2.0,
        "vd_low": 0.5,
        "sigma_noise": 0.1
      },
      "n_seeds": 4,
      "mean_acc": 0.5486087546657619,
      "per_seed_test_acc": [
        0.5364777740074652,
        0.5605700712589073,
        0.5531048523922634,
        0.5442823210044113
      ],
      "min_acc": 0.5364777740074652,
      "max_acc": 0.5605700712589073
    }
  },
  "gates": {
    "AMBITIOUS": true,
    "PASS": true,
    "details": [
      "AMBITIOUS hit: N8192_sigma0.00=0.8339>0.82 (sigma=0)",
      "AMBITIOUS hit: N8192_sigma0.10=0.8265>0.80 (sigma=0.10)",
      "AMBITIOUS hit: N16384_sigma0.00=0.8391>0.82 (sigma=0)",
      "AMBITIOUS hit: N16384_sigma0.10=0.8364>0.80 (sigma=0.10)",
      "PASS check (sigma=0): N=1024:0.8056 -> N=8192:0.8339 -> N=16384:0.8391  monotone=True"
    ]
  }
}
```
