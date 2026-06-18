# O109 Synthesis — Coupling × Model × Benchmark

**Date:** 2026-05-31  •  **Providers:** OpenAI (gpt-5), Gemini (2.5-pro), Grok (grok-4-latest), DeepSeek (reasoner)

## 1. Diagnosis of the A−B null (Phase 7 ABCD)

**Unanimous verdict: (d) all three causes, dominated by (a) decorative coupling.**

| Cause                                    | OpenAI | Gemini | Grok | DeepSeek | Mean |
|------------------------------------------|--------|--------|------|----------|------|
| (a) hash coupling decorative             | 0.60   | 0.60   | 0.45 | ~0.50    | 0.54 |
| (c) benchmark doesn't require body-info  | 0.15   | 0.30   | 0.40 | ~0.30    | 0.29 |
| (b) ridge ESN too universal              | 0.25   | 0.10   | 0.15 | ~0.20    | 0.17 |

All four agree: the hash never enters the forward pass, so it is a label not a computation; random ESN matrices of the same class are statistically equivalent (Yıldız et al. 2012); ridge readout washes out any weak structural prior; next-step prediction is solvable from data alone.

## 2. Constitutive coupling design — consensus

The substrate must parameterize the **recurrent update equation itself**, not be appended to x. Four near-identical pseudocode sketches:

```
α(t) = sigmoid((T_apu(t) - T_ref)/T_range)   # leak rate
ρ(t) or γ(t) = f(P_pkg(t))                    # spectral radius / gain
h_{t+1} = (1-α)·h + α·tanh( γ·(W·h + U·x) )
```

- **Signals**: APU temp (`thermal_zone0`, 10–50 Hz), package power (RAPL, 1–10 Hz), optionally fan RPM, clocks.
- **Confounds (all four oracles flagged)**: sensor drift → use EMA-differenced values; aliasing → low-pass with τ ≈ 1–3 s; self-heating from sysfs reads; replay alignment (must time-align ikaros/daedalus traces); training-time scaling must NOT be re-fit on eval host.
- **Pre-reg gates**: A−B ≥ 10% NRMSE, A−D ≥ 5%, A−C ≥ 5%.

## 3. Architecture ranking (consensus best → worst)

1. **Continuous-time RNN with substrate as parameter** (substrate sets the vector field)
2. **Neural ODE** (substrate as forcing function — Chen et al. 2018)
3. **Spiking NN** (real-time integration naturally couples to live sensors)
4. **Predictive coding / energy-based** (substrate modulates landscape)
5. **LSTM** (gates modulable, but discrete time)
6. **Transformer** (adaptive layer-norm / attention temperature — possible but weak)
7. **MLP** (only if every weight is substrate-parametric)
8. **Ridge ESN** ← current null result; pure readout dominates

Grok added Hopfield (4th) and contrastive DAE (last); all agree ridge ESN is bottom-tier for measurable chassis-binding.

## 4. Benchmarks where body-info is the ONLY PATH

Unanimous trio (with literature anchors):

1. **Closed-loop fan/thermal control** — per-chassis RC time constant, fan curve, paste condition determine the unique optimal policy (Pfeifer & Bongard 2006; Hauser et al. 2011 morphological computation).
2. **Self-replication / self-prediction** — only a model coupled to its own substrate trajectory can predict its own future output under live sensor influence.
3. **Survival race / thermal-budget decoding** — maximise tokens before sensor hits 95 °C; requires internalised fan curve + heat-spreader lag (Ha & Schmidhuber 2018; recent thermal-constrained LLM serving).

## 5. Sharpest critical test

**Constitutive reservoir (α, ρ driven by live T/P) on 30-second thermal-budget survival task.**
- Pre-reg: own-chassis (A) achieves ≥12% more tokens than transplant (B) at equal thermal violations (Cohen's d > 0.8, n=40, BCa 95% CI).
- Negative controls: C constant-α=0.5, D shuffled alien trajectory.

## 6. Brutal honesty: recoverable or falsified?

**RECOVERABLE — but the *hash-coupling* hypothesis is falsified.** Identity-via-decoration is dead. The constitutive hypothesis (substrate ∈ forward pass) has not yet been tested at a benchmark that requires it. All four oracles say: ~3 more aligned experiments (constitutive coupling + body-required task + appropriate architecture) before declaring exhaustive falsification.

## 7. Strongest defensible paper claim (current data)

**Negative methodological result**: "Random structural priors (hash-derived ESN seeds) yield no measurable chassis-binding advantage over arbitrary random seeds on standard sensor-prediction benchmarks, demonstrating that identity-via-initialization is decorative; live-substrate coupling and embodiment-requiring tasks are necessary conditions." This is a valid null-result paper; the constitutive follow-up determines whether it stays a null or becomes positive.
