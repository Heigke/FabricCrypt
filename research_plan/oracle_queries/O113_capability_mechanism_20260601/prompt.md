# O113 — Adversarial Search for Real Capability-Gain Mechanisms in Commodity-Chip Embodied AI

You are being asked to be **adversarial and creative**. We have spent months building
a per-die identity signature on commodity AMD Strix Halo APUs (gfx1151). We can
fingerprint a specific physical chip with 100% LOO classification accuracy from a
290-dim vector built out of TSC offsets, cache-line ping-pong, DRAM refresh timing,
syscall latency, and NVMe physics. So far so good.

The hard claim — and the one we are stuck on — is **CAPABILITY GAIN**: that
giving an AI access to its host chip's idiosyncratic physics makes the AI
*better at some task*, not merely *bonded to that chip*.

Phase 15 ran five pre-registered tests of capability gain. **All five were NULL**
on the pre-registered gate (full SUMMARY attached). We do not believe this
closes the question. We believe we are asking the wrong question, measuring the
wrong axis, or missing a mechanism. Your job is to tell us what we missed.

## Attached artifacts

- `phase15_SUMMARY.json` — the five NULL pre-registered tests (E1–E5), with means + CIs + verdicts.
- `phase14b_summary.txt` — the prior smaller embodied-tiny benchmark (T1 fail, **T2 PASS** anomaly detection +0.49 AUROC, **T3 PASS** twin-paradox host-ID +50pp, T4 fail substrate-aware completion 1.06× vs 1.2× prereg).
- `phase13_signature_v2_slim.json` — the 290-dim signature_v2 with 100% LOO host classification, drift p95=0.19, constitutive A vs C swap gate PASS (57.9% NRMSE inflation when you transplant a wrong-chip signature into a chip-conditioned model).

## The 5 verified per-chip signals (replicated 24h apart, two physical machines)

| # | Probe | inter-machine KS-D | intra-machine KS-D | ratio |
|---|---|---|---|---|
| 1 | nanosleep(0) latency distribution | 0.7224 | 0.0152 | ~47× |
| 2 | sched_yield latency distribution  | 0.9931 | 0.0222 | ~45× |
| 3 | inter-core cache-line ping-pong p50 | 0.9118 | small | huge |
| 4 | RDTSC offset between same-package cores | 0.91 | small | huge |
| 5 | DRAM refresh-window timing pattern | ~0.9 | small | huge |

Both machines: identical AMD Strix Halo SKU, microcode `0xb70001e`, same kernel,
same git tree. The signals separate the dies; software ABI cannot.

## Phase 15 (capability-gain tests) — the five NULLs

1. **E1 free_entropy_reg**: Use chip-jitter as a *regularizer* (replace dropout-style
   noise) on noisy classification. 30 seeds. Embodied −0.35pp vs matched synthetic
   (CI excludes zero on the wrong side). **Chip jitter is a WORSE regularizer than
   uniform-random with matched marginal stats.** Hypothesised reason: chip jitter
   has temporal autocorrelation, which is the wrong noise spectrum for L2 / dropout.

2. **E2 attention_bias_dram**: Use per-token DRAM access latency as an attention
   prior on a text-classification transformer. 30 seeds. Embodied +0.14pp vs
   random bias, CI spans zero. **DRAM latency at this token granularity is not
   correlated with anything semantically useful.**

3. **E3 thermal_inference_budget**: When chip ≥58°C, skip layer-4 of the model
   for cheaper inference at iso-accuracy. 12 seeds. clean-seed QPS ratio = 1.055
   (CI [0.982, 1.116]). Pre-reg was 1.15. **Mechanism works directionally
   (12% gain when active) but cold-start noise wrecks the population CI.**

4. **E4 predictive_scheduling**: Predict per-request latency from chip state
   features to reorder a queue. 12 seeds. Predictor↔oracle Pearson = 0.04 (chance).
   2.84% improvement is generic anti-arrival-order benefit, not embodiment.
   **Chip state at request granularity does not predict per-request latency.**

5. **E5 per_machine_finetune**: Inject the 290-dim chip vector as conditioning
   into LM finetuning on a per-user corpus, hoping it helps per-machine
   personalisation. 25 seeds. Embodied PPL = 75.65, vanilla 75.04, matched-random 76.28.
   **Embodied beats random conditioning (CI [0.60, 1.05] excludes zero) but ANY
   conditioning hurts vs no conditioning.** Chip info is information, but the
   architecture is a net loss.

**Across 109 seeds, zero experiments cleared their pre-registered computational
gain gate. We need to know what we missed.**

---

## QUESTIONS — please answer ALL, numbered. Be adversarial. Push hard.

### A. Reframing capability gain

1. We measured "capability" as static-benchmark improvement (accuracy / PPL on
   classification, text). Is this the right frame for embodied AI? Argue for
   alternatives and rank them by how much we should expect chip access to help:
   - **Robustness gain** (perturbation / adversary / distribution shift)
   - **Adaptive gain** (varying conditions in time)
   - **Personalisation gain** (user / machine fit)
   - **Trust / sovereignty** (provable HW-binding, unfakeable provenance)
   - **Energy efficiency** (matched-flops less power, or matched-power more useful work)
   - Others we have not named

2. **What is the smallest example in the literature where access to physical
   substrate genuinely IMPROVED a learning algorithm**, not just "ran the algorithm"?
   We want concrete recent (2023–2026) citations:
   - Physical / photonic / memristor reservoir computing — beyond ESN-on-silicon
   - NS-RAM analog persistence (we have our own paper there; what else?)
   - Pfeifer / Iida morphological computation — what specifically OFFLOADED?
   - Sparks / Boahen physical neural systems
   - Tegmark physics-informed nets
   - Friston / Rao FEP / active inference and *body-as-prior*
   - 2025–2026 SOTA we may have missed entirely

3. **The embodied-cognition reframe.** Embodied cognition's actual claim isn't
   "body makes brain better at chess" — it's "body OFFLOADS computation the
   brain would otherwise do". Where on a commodity AMD APU can the chip
   *genuinely offload work* from an AI? Be concrete:
   - Free entropy (E1 said no for regularization — for what tasks could it be yes?)
   - Free attention bias (E2 said no for DRAM/text — for what architectures yes?)
   - Free memory (chip state as part of model state — RAG memory? streaming?)
   - Free metric (chip latency as task-difficulty proxy — for adaptive computation?)
   - Free random (Monte-Carlo, particle filters, MCMC — does chip RNG help?)
   - Free clock / scheduler (event timing in continual learning?)
   - Free thermal envelope (joint energy-accuracy optimisation across time)
   - Free serial number (federated learning identity? sybil resistance?)

4. **Brainstorm 10 NEW mechanisms** that could plausibly give real capability
   gain on a commodity AMD APU specifically. For each, give:
   - Mechanism in one line
   - Task it would help
   - Why we should expect a gain (a priori)
   - Cheapest falsifiable pre-reg gate
   - Risk of being another null

5. **Pick the best 3 mechanisms to try next**, with concrete pre-reg gate
   suggestions (effect size, n_seeds, threshold, CI requirement) that are
   STRICTER than what we ran in Phase 15. If you cannot propose 3 that you
   yourself would bet on, say so explicitly — that itself is informative.

### B. Hardware-specific opportunities (AMD Strix Halo / RDNA 3.5 / Zen 5)

6. What specific AMD APU features should we be exploiting that we are not?
   - WMMA / MFMA instruction analog timing variation across CUs
   - Per-CU latency variation (RDNA 3.5 has 40 CUs)
   - Infinity Fabric routing variance between CCDs
   - Shared L3 / Infinity Cache contention patterns
   - SMU / SMN telemetry as continuous physical signal (we have access)
   - HSA queue scheduler stochasticity
   - HBM-style unified memory access pattern as compute primitive

### C. Skepticism check (be honest)

7. **Quantify**: Bayesian P(genuine capability gain exists on commodity APU |
   Phase 15 null on 5 well-motivated mechanisms). State your prior and your
   posterior. Show your reasoning. If the answer is <0.3, recommend we pivot
   to identity-only framing.

8. **Steelman the null**: Is the "embodiment makes AI more capable" hypothesis
   FUNDAMENTALLY flawed at commodity-silicon scale, or have we just been
   unlucky with task choice? Give your strongest argument for each side.

9. **Compare to NS-RAM**: that paper showed real compute gain via genuine
   analog physics in a custom device. What is the gap between "custom analog
   device" and "commodity APU jitter"? Is the gap unbridgeable in principle,
   or is it a matter of finding the right interface?

10. **Honest verdict**: If you had 1 month of our time, would you spend it on
    (a) one more capability-gain mechanism, (b) deepening the identity / 
    unfakeable-provenance story (which DOES work, 100% LOO), or (c) abandoning
    the commodity-APU angle and going to FPGA / NS-RAM where the physics is
    custom? Justify in 5 lines.

---

Be merciless. We can take it. We would rather hear "you are chasing a phantom"
than waste another month.
