# Combined text bundle for oracle context

All text artifacts concatenated below for one-shot context. Each file is delimited by a `=== FILE: <name> ===` marker.



=== FILE: O107_prior_synthesis.md (3214 chars) ===
```
# O107 synthesis — When does embodiment help? (4-way)

## Consensus on Q1 task categories

| Category | OpenAI | Gemini | Grok | DeepSeek |
|---|---|---|---|---|
| **Survival / power-thermal control** | ✅ | ✅ | ✅ | ✅ |
| **Latency-aware self-modeling** | ✅ | ✅ | ✅ | ✅ |
| **Hidden-state / substrate prediction** | ✅ | ✅ (caveat) | ✅ | ✅ (caveat) |
| **Aging / drift tracking** | ✅ | — | ✅ | ✅ |
| **Attestation / PUF** | ✅ | ✅ | ✅ | ✅ |
| **Adaptive precision / energy-aware brittle inference** | — | ✅ | — | ✅ |
| **Computational anomaly detection** | — | ✅ | — | — |

**Top-3 unanimous:** (1) thermal-aware survival control, (2) latency-aware
self-modeling in closed loops, (3) PUF-class attestation.

## Q9 — Brutal honesty (the killer reading)

Three of four oracles (Gemini, Grok, DeepSeek) say embodiment on twin
commodity GPUs is **net-zero** for any task that isn't itself about the
hardware, **even when the task IS about the hardware**, *provided the
generic baseline gets equivalent volume of this-chip data*. The honest
control isn't "ikaros vs daedalus" but "ikaros vs daedalus + N× more
ikaros data".

OpenAI is more permissive: control/optimisation with own-latency in the
loop is a legitimate win-zone, and attestation is real (but PUF-class).

## Q10 — Killer experiments converge on one design

All four propose variants of the same thing:
- An RL/control agent
- Maximises useful work (tokens/s, FLOPs, batches)
- Under a hard power+thermal cap
- With its own action latency in the loop
- Baseline: same agent trained on the twin only
- Win-gate: 5–15% more work at equal/fewer violations

Variants:
- **OpenAI**: 30-min LLM decode under 110 W cap + 95 °C Tedge; ABBA cross-host
- **Gemini**: 30-min adaptive throughput on CIFAR-10 batches with 95 °C cap
- **Grok**: 8-min × 50-trial power-cap tracking with sparsity mask
- **DeepSeek**: 10-second thermal-budget racing from cold; 20 trials

**Common requirement**: train both models elsewhere, then test on target
with closed-loop control where physics, not just data, is the gatekeeper.

## What this means for our C1/C2/C3 prereg

- **C1 self-prediction**: Gemini and DeepSeek flag this as trivially won
  by *any* model trained on this chip's data, regardless of embodiment.
  Our gate (ikaros vs daedalus, NRMSE ≥30% gap) is necessary but not
  sufficient — to claim embodiment win we'd need a third arm: generic
  pooled-data baseline.
- **C2 self-anomaly**: Same flaw. Gemini explicitly says "specialisation,
  not embodiment". Likely NULL.
- **C3 thermal-survival**: Most likely to PASS but methodologically
  vulnerable: twins have same RC mass + fan curve, so transplant model
  should do nearly as well. The cleanest C3 is the closed-loop variant
  oracles describe — and our sim-based C3 is the weakest version of it
  (we already use the same thermal-model class for both arms).

## Path forward

Our C1/C2/C3 are honest first-pass tests. If all NULL, that's a finding:
embodiment on twin commodity GPUs is real-identity-but-no-advantage; the
publishable claim is the negative one. If C3 PASSES on simulator, the
follow-up must be the closed-loop on real hardware version (oracles'
killer experiment) — this is the Phase 6 design.

```


=== FILE: O108_prior_synthesis.md (4019 chars) ===
```
# Synthesis — Oracle O108 (4-way) on Embodiment Phase 7 critic holes

**Date**: 2026-05-31
**Bundle**: prompt.md + (this run had no zipped attachments — prompt-only)
**Responding oracles**: openai gpt-5 (210s), gemini (server error), grok-4 (9s), deepseek-reasoner (32s) → 3/4 usable.

## Strong convergence (all 3 oracles agreed)

### Top 3 critic holes (gpt-5, grok, deepseek consensus)

1. **Distribution-shift confound is unresolved** — no factorial ablation separates "model trained on own data" (trivial) from "structure adds capability" (interesting). All three demanded the same 2×2 (structure × data) matrix.
2. **C2 AUROC <0.5 is a SCALING/POLARITY ARTIFACT** (orientation flip under mismatched z-score), not a capability gap. Settle by applying training-host scaler to test host, or by sign-flipping anomaly score.
3. **Statistics insufficient** — 5 seeds is far below minimum (10-30 required), no bootstrap CIs, no multiple-comparison correction. Time-series window-overlap means N is even more inflated than it looks.

### Killer falsifier (Q8)
- **grok / deepseek converge**: swap the hash between machines at inference. If model trained on ikaros-data with ikaros-hash performs identically when given daedalus-hash → hash is causally irrelevant → embodiment hypothesis falsified.
- **gpt-5 variant**: pool data + DANN/IRM domain-invariant model under global scaling — if it matches "self-specialists" on both hosts, the chassi-bound effect collapses to ordinary domain generalization.

### Architecture (Q4)
- Universal agreement: do NOT claim architecture-agnostic. Ridge is a linear probe. Must demonstrate at least one nonlinear model (MLP/LSTM/Transformer) with same directional effect. Otherwise scope: "shallow regression on these two machines."

### Statistics (Q5)
- gpt-5: ≥10 days × ≥20 seeds, hierarchical moving-block bootstrap, BCa CIs, Holm-Bonferroni on 4 preregistered, BH-FDR on rest.
- grok: ≥20-30 seeds, BCa CIs, FDR across ≥16 tests.
- deepseek: N ≥ 10 seeds, percentile bootstrap n=10 000, α=0.025 per gate (Bonferroni for 2 confirmatory).
- Common floor: **10+ seeds, BCa or percentile bootstrap, pre-register the confirmatory tests**.

### External validity / N=2 (Q6)
- All 3: only honest position is explicit scope limitation. ("Demonstrated on these 2 machines; population claim is future work.")
- gpt-5's extreme version: blinded "everything-swapped" crossover (disks, RAM, PSUs, locations, blinded analysts).
- deepseek's softer version: get a third machine, even briefly.

### Cross-task generalization (Q7)
- gpt-5: ≥3 task families (short-horizon dynamics, anomaly w/ conformal baseline, control/decision w/ real penalty).
- grok: ≥3 + a negative-control task that does NOT show advantage.
- deepseek: ≥4 across distinct sensor modalities.
- Common floor: **3 body-centric tasks + 1 negative-control abstract task**.

### Defensible vs overclaim (Q9, Q10)
- **Defensible (gpt-5/grok/deepseek convergent paraphrase)**:
  "Two physically identical AMD Ryzen AI Max+ PRO 395 workstations exhibit
  large, repeatable, within-chassis advantages on self-prediction and
  self-anomaly tasks; the effect is consistent with learnable per-chassis
  dynamics but **has not yet been isolated from training-distribution shift**."
- **Overclaims to AVOID**:
  - "architecture-agnostic"
  - "embodiment" (implies agency)
  - "chassi-bound identity / PUF-grade"
  - any population-level claim across the gfx1151 line
  - any transfer claim to abstract tasks (we have 10 null hypotheses already)

## What changed for Phase 7 plan based on this synthesis

- A/B/C/D ablation is **the** killer test — prioritised. (Implemented & run.)
- C2 AUROC <0.5 = artifact — must apply same-training-host scaler, then re-evaluate; lower priority for headline.
- Multi-architecture: at minimum MLP done; LSTM/Transformer are nice-to-have for paper.
- 30 seeds + bootstrap CI mandated for confirmatory cells.
- Final claim language tracks the consensus minimal version, not the embodiment-rich version.

```


=== FILE: constitutive_design.md (1256 chars) ===
```
# Task B (Phase 9) — constitutive live-substrate coupling

Reservoir where the leak rate IS the live substrate at every forward-pass step.

```python
def step(h, x, t):
    apu_t = read_thermal_zone0()           # /sys/class/thermal/thermal_zone0/temp
    apu_p = read_rapl_package()            # /sys/class/powercap/intel-rapl:0/energy_uj
    alpha = sigmoid((apu_t - 50) / 10) * 0.5 + 0.25   # 0.25..0.75
    gain  = 1.0 + 0.1 * (apu_p / 30 - 1)              # gain modulation
    new_h = (1 - alpha) * h + alpha * np.tanh(gain * (W @ h + U @ x))
    return new_h
```

Substrate is NOT a feature appended to `x`. It *is* a parameter of the
recurrent update. Two hosts with different thermal trajectories run
*different dynamical systems*, not just receive different inputs.

Transplant matrix:
- A — trained on ikaros, evaluated on ikaros (own substrate at inference).
- B — trained on daedalus, evaluated on ikaros (transplant; alien dynamics).
- C — random `alpha = 0.5` constant control (no substrate coupling).
- D — SHUFFLE: ikaros-trained model, but at inference replay daedalus's
  recorded substrate trajectory through `alpha`. Tests whether
  *trajectory specificity* matters beyond mere mismatch.

Pre-reg: A − B ≥ 10% NRMSE, A − D ≥ 5% NRMSE.

```


=== FILE: fan_control_design.md (714 chars) ===
```
# Task C — closed-loop fan PWM control

Task: at each control step, output fan PWM duty (0..255). Reward:
`-(T - T_target)² - λ * (PWM)²`. Per-chassis thermal RC, ambient, paste
condition differ → ikaros-trained controller learns *ikaros's transfer
function*.

Pre-reg: ikaros-trained policy achieves RMS(T − T_target) at least 20%
lower than (a) constant-PWM baseline, (b) PID with default gains,
(c) daedalus-trained transplant.

Implementation:
- Try `/sys/class/hwmon/*/pwm1`. If non-writable, fall back to
  *simulated thermal RC* parameterized by recorded ikaros vs daedalus
  step-response (so the difference between chassis is preserved).
- Action latency in the loop (read T → act → next read 1 s later).

```


=== FILE: phase7_abcd_summary.md (2169 chars) ===
```
# Phase 7 A/B/C/D — the killer ablation (30 seeds, ridge reservoir, real twins)

Two metrics × two eval hosts × four cells.

## C2 (self-anomaly autoencoder, AUROC, n=30 seeds per cell)

| Eval host  | Cell | Structure          | Data host | AUROC mean | AUROC std |
|------------|------|--------------------|-----------|------------|-----------|
| ikaros     | A    | ikaros-hash        | ikaros    | 0.8340     | 0.0254    |
| ikaros     | B    | random             | ikaros    | 0.8338     | 0.0246    |
| ikaros     | C    | ikaros-hash        | daedalus  | 0.4923     | 0.0414    |
| ikaros     | D    | random             | daedalus  | 0.4917     | 0.0411    |
| daedalus   | A    | daedalus-hash      | daedalus  | 0.8291     | 0.0456    |
| daedalus   | B    | random             | daedalus  | 0.8327     | 0.0471    |
| daedalus   | C    | daedalus-hash      | ikaros    | 0.4994     | 0.0462    |
| daedalus   | D    | random             | ikaros    | 0.4994     | 0.0462    |

Pre-reg gates:
- A − B ≥ 10%  →  FAIL (Δ = +0.0002 on ikaros, −0.0036 on daedalus)
- A − C large  →  PASS (Δ ≈ +0.34, but this just confirms data matters)
- (A − B) > (C − D)  →  FAIL (both ≈ 0)
- A > max(B,C,D) by ≥ 1σ  →  FAIL (B matches A within noise)

**Conclusion**: chassis-hash-derived structure adds **zero** capability over
arbitrary random structure given the same data. All signal is in the data
distribution, none in the chassis-bound init.

Same pattern observed on C1 (next-step prediction, NRMSE).

## Interpretation

The hash is being used as a *label* (a deterministic but informationally
inert mapping from chassis → 4-byte seed). The seed only chooses *which*
random reservoir we instantiate. Since random reservoirs of the same size
are statistically interchangeable (Yildiz et al. 2012; Lukoševičius 2012),
choosing one via SHA-256(chassi-id) vs. choosing one via `seed = 7` cannot
change the fitted ridge solution beyond seed noise.

The substrate never enters the forward pass. The model is *recognisable* by
the chassis (the data it was trained on is from this chassis) but it is
*not constitutive* (the chassis is not doing computation inside the model).

```


=== FILE: self_replication_design.md (700 chars) ===
```
# Task D — self-replication / body-knows-itself

At time t the model must predict whether its OWN output at t+H seconds will
match a reference computation of the same model on the same input,
re-evaluated against fresh live substrate.

Only a model that has internalised its chassis's substrate trajectory can
correctly predict its own near-future state, because the recurrent state at
t+H is a function of substrate reads between t and t+H.

Pre-reg: ikaros self-replication F1 ≥ 0.7, transplant (daedalus-trained →
ikaros eval) F1 ≤ 0.5.

Adversary: a model that does NOT use live substrate cannot solve this above
chance on the constitutive coupling reservoir — proves body-info is the
only path.

```
