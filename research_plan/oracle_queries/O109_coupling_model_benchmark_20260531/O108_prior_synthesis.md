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
