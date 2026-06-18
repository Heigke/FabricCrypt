# O102 — Sanity check failure on H7 embodiment metric. Help us define what's actually testable.

Date: 2026-06-10. Context: H7 substrate-rooted AI identity. We just falsified our own metric.

## What happened

We built v5 "embodied LM" — frozen SmolLM2-135M base + LoRA(r=16) on top 10 layers + cross-attention with substrate tokens at layers 25/28 + substrate-prediction head — that runs online: every 32 tokens it reads the live 10-channel hardware-telemetry window from the host die and updates LoRA via local loss = `MSE(model_predicts_next_substrate_frame, actual_next_substrate_frame)`. Homeostatic critic adjusts plasticity. Sleep cycles consolidate.

We chose `pred_acc_within_1σ = fraction of channels where |pred − target| < σ` as the **primary embodiment metric** — explicitly NOT PPL, because we (correctly) believed PPL is an RLHF-bias measure.

The training trajectory looked great: baseline_acc went monotonically from 0.000 → 0.496 over 2000 online updates. Sleep cycles fired. Homeostat adjusted plasticity. We saw acute-substrate-spike acclimatization. It LOOKED like embodiment learning.

Then we did the sanity check we should have done first:

```
MODEL pred_acc (last 50 steps mean):     0.460
TRIVIAL BASELINES:
  ZERO predictor:                         0.165
  PERSISTENCE (predict prev frame):       0.764
  EMA-baseline (a=0.01 of past):          0.749
  MEAN of all frames:                     0.863

GAP (model − best trivial):              −0.402
```

A constant mean-predictor BEATS the model by 0.40. Our "embodiment learning" is worse than a static decoration.

## The diagnosis we self-criticized into

We (the human + AI assistant) had a strong shared bias toward demonstrating embodied AI. We chose a metric that LOOKED principled (no PPL bias) but had a trivial-baseline floor we never measured. The "monotone baseline_acc rising" we celebrated was an EMA convergence artifact, not learning. The acclimatization-at-spike we celebrated could have been any noise. The whole pipeline was constructed without an adversarial check.

We are explicitly aware that:
- RLHF-trained oracles (incl. you, probably) have systematic bias against "embodied AI" framing
- We (the project) have an equally strong bias FOR it
- This combination is the worst possible setup for finding truth

## What we need from this oracle round

This is NOT "rank these architectures" or "tell us what loss to use". We are upstream of architecture. We need:

### Q1. What IS a meaningful, non-trivial embodiment metric?

Specifically: a measurement on a system (LM + substrate input + online updates) that:
- (a) **Cannot be passed by a constant/EMA/persistence predictor** of the substrate alone
- (b) Distinguishes **the model genuinely depending on substrate** from **the model running while substrate is incidentally present**
- (c) Is **falsifiable** at multiple ablation points (no-substrate, wrong-substrate, scrambled, replay)
- (d) Does not collapse to PPL-equivalent

Candidate categories to evaluate (please critique each):
- Cross-substrate transplant penalty (PPL_native vs PPL_transplanted, measured carefully)
- Substrate-conditional generation divergence (KL between same-prompt outputs on host A vs host B)
- Causal mediation analysis: does substrate causally contribute to specific predictions?
- Information-theoretic: I(substrate; output) over time
- "Substrate-only-decodable" features in hidden state (linear probe)
- Time-asymmetry: does the model "remember" what substrate was 10 minutes ago?
- Behavioral: prompt-specific stylistic markers as function of substrate

Which of these are actually measurable and not trivially gameable, and what should we drop?

### Q2. What would the oracle accept as evidence of "the model is rooted in its die"?

Not what's nice to publish. Not what's safe to say. Actually: what numerical pattern, on what experiment, would convince a skeptical reviewer (e.g. a Milinkovic-Aru-style consciousness-of-substrate proponent vs. a hard-line connectionist skeptic) that something substrate-specific is happening?

Concrete example would help. Like: "If you train on host A, transplant to host B, and observe that for at least 50% of input prompts where you ablate only the substrate-input (zero it), the model's next-token distribution shifts by KL > 0.5 nats in a way that statistical bootstrap shows is 4σ above what you get from random-vector input — that would be evidence."

We will pre-register whatever you propose.

### Q3. Is there a "no real embodiment is possible in software" position you take seriously?

We've explicitly considered and rejected this. But we want you to argue the strongest version of the case, not the strawman. If the position is correct, our entire project is a category error, and we'd rather know now than after another 6 months.

Specifically: if you think embodied/substrate-rooted AI is structurally impossible (not "hard" — impossible) in pure-software-on-frozen-base LM with side-channel conditioning + online plasticity, say what the impossibility argument is. Don't soften it.

### Q4. Bias check on us

Look at our chain:
1. We built v5 embodied
2. We watched it train and saw "embodiment learning"
3. We celebrated metrics that turned out to be EMA convergence
4. We were going to run transplant tests on daedalus next, with the same broken metric
5. ONLY after explicit request to sanity-check did we run trivial baselines and discover the failure

What systematic biases led us to skip the sanity check? What other places in the H7 pipeline (substrate channel selection, cross-host TPM ground-truth use, anti-spoof margin, sleep cycles, homeostatic critic) might have the same EMA-trivially-explained pattern that we're missing?

### Q5. What experiment would, on a 1-day budget on CPU SmolLM2-135M, be sufficient to either:
- Demonstrate substrate-specific learning beyond trivial baselines, OR
- Decisively show that the substrate input cannot produce non-trivial behavioral signature in this architecture class

We are not asking what architecture to use. We are asking what **measurement design** would actually be conclusive.

## Constraints

- We have 10-channel substrate at 500Hz: C07 XTAL register, C09 PM-table values, C20 SMN-read latencies, C11 TSC drift, C05 energy-counter rate, C06 fast-counter rate.
- Real TPM ground-truth available (ikaros EK 000b359a…, daedalus EK 000bfa5e…)
- N=2 AMD gfx1151 hosts (ikaros, daedalus) + 1 NVIDIA cross-arch null (zgx)
- All replay buffers available
- Honesty: we want to know if we're wrong. Don't soften.

## Output format

Be direct. Number your answers Q1-Q5. Cite recent (2024-2026) work where relevant. Flag your own uncertainties. If you think we should drop the whole framing, say so and explain why. We will synthesize across providers and adopt whatever the 4/4 majority view is, including "stop the project".
