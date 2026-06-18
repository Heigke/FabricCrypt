# Oracle query O108 — Embodiment Phase 7 critic holes (4-way)

You are one of four oracles (GPT-5, Gemini 2.5 Pro, Grok-4, DeepSeek-R) being
asked **adversarially** to find what's missing before publication. Be terse
(≤700 words). HOSTILE / FALSIFYING tone preferred — find what's broken, don't
validate. Cite relevant literature where applicable (PUF, embodiment, OOD
detection, ML reproducibility).

## Background

We have spent a campaign building "chassi-bound model identity" on twin
**AMD Ryzen AI Max+ PRO 395** chassis (HP Z2 mini G1a — physically identical
hardware: ikaros, daedalus).

**Phase 3 (structure-binding)**: A 256-bit fingerprint over DMI + hwmon enum
+ PCI topology + per-CPU + thermal stats. Gates G1–G4 all pass on both
machines (cross-host distance large, intra-host stable across reboots,
remeasure stable, fingerprint causally driven by chassi not env).

**Phase 5 (body-centric task advantage)**:
- C1 (self-substrate-prediction, ridge over 100-step history of 5 channels):
  ikaros-trained NRMSE 0.140, ikaros model evaluated on daedalus = 73.49
  (524× worse). Pre-reg WIN gate (≥30% improvement self-vs-transplant): **PASS**.
- C2 (self-anomaly autoencoder): self AUROC 0.870, transplant AUROC 0.484
  (BELOW chance), generic-untrained 0.793. Pre-reg gate (≥10 pp): **PASS**.
- C3 (thermal-survival scheduling sim): NULL — simulator too generous, both
  policies survive without trips.

**Hypotheses V4+V5 abstract-task transfers (10 hypotheses)**: ALL NULL.
The embodiment advantage only shows on body-centric tasks (predicting one's
own substrate trajectory or detecting anomalies in it).

## Critic holes we know about

1. **Distribution-shift confound**: C1/C2 transplant fails could be just
   "model trained on data drawn from distribution A and tested on distribution
   B" rather than chassi-bound STRUCTURE adding capability. We have no A/B
   ablation that separates structure effect from data effect.
2. **C2 AUROC <0.5 is suspicious** — should be 0.5 (chance) under pure
   distribution shift; <0.5 implies the anomaly direction is inverted on
   the foreign host. Could be artifact of feature-scaling mismatch.
3. **Single architecture**: only ridge / tiny MLP autoencoder tested.
4. **Statistical power**: 5 seeds, no bootstrap CIs, no multiple-comparison
   correction across our many tests.
5. **N=2 machines** → no claim of generality across the gfx1151 population.
6. **Signal underutilization**: 32+ HW catalog mechanisms exist but the
   structure hash uses only ~10. Have not built max-bandwidth signature.

## Questions

Answer each numbered question explicitly:

1. Given the above, what critic holes do we still MISS? List the top 3
   that must be closed before publication.
2. Distribution-shift critique: how do we conclusively separate "model
   trained on own data wins" (trivial) from "chassi-bound structure adds
   capability beyond data" (interesting)? Propose the exact ablation
   matrix you would demand from a referee position.
3. C2 AUROC <0.5 on transplant — capability gap or distribution-shift
   artifact? How to distinguish? What experiment would settle it?
4. Multi-architecture: must we test embodiment with MLP / LSTM /
   Transformer to claim it's not ridge-specific? Or is "architecture
   agnostic" overclaim we should avoid?
5. Statistical robustness: minimum N seeds for defensible claim?
   Bonferroni correction needed across how many tests? What CI method?
6. External validity: only 2 machines. What is the ONE argument that
   silences the critic if we can't get a third machine?
7. Cross-task generalization: how many body-centric tasks must show
   advantage to claim it's "general" embodiment (not specific to
   C1/C2)?
8. The killer falsifier we haven't run yet — what is it? Be specific.
9. Brutal honesty: given all evidence, what's the BEST defensible claim?
   What's the OVER-CLAIM we should AVOID?
10. Headline single-sentence claim for the paper abstract that survives
    adversarial critique?

Synthesis target: one paragraph per question, brutal, falsifying,
cite-where-relevant.
