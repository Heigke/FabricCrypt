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
