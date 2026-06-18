# O106 — Performance Advantage Hunt on Envelope-Keyed Neural Computation

## Context (very short)
We built an envelope-keyed reservoir-computing setup on an AMD gfx1151 APU (ikaros)
and a remote AMD machine (daedalus). A 23-feature host "envelope" (power
profile, thermal RC, per-core latency / freq stats) is SHA256-hashed into the
structure of a 128-neuron sparse reservoir along 5 axes:
  1. sparse adjacency mask (~25% density)
  2. per-neuron activation choice (5-way)
  3. recurrent update permutation
  4. per-neuron weight scale
  5. per-neuron leak rate
A ridge readout is trained on NARMA-10. We then "transplant" the trained weights
to the other machine's structure (re-derived from THAT machine's envelope) and
measure NRMSE degradation.

## Results (Phase A-D so far)
- **Phase A1**: cross-machine envelope cos_dist = 0.01238
- **Phase A2**: workload-invariant on same chassis (143× separation)
- **Phase C gates passed (3/4)**:
  - G1 (ikaros self) NRMSE = 0.61
  - G2 (daedalus transplant) NRMSE = 368 → **607× degradation factor**
  - G3 (random envelope) NRMSE = 489 → **807×**
- **Phase C5 FAIL**: envelope-keyed structure was **5.7% WORSE** than vanilla
  deterministic baseline on the same task. The envelope BINDS the model to the
  chassis tightly, but does NOT make the model better.
- **Phase D1 (deeper envelope)**: expanding 23 → 165 envelope features
  (deterministic polynomial/log/product feature expansion) gives factor
  632× (vs 693× for 23-feat) → **NEUTRAL**: more bits ≠ tighter binding
- **Phase D2 (single-axis ablation)**: ranked by binding contribution:
  1. perm 920×  (recurrent update permutation alone gives biggest binding)
  2. mask 481×
  3. acts 464×
  4. weight_scale 97×
  5. leak 70×
- **Phase D3, E, F still running.**

## Phase F: WHAT WE NEED FROM YOU
We want to make the envelope-keyed model PERFORM BETTER than a generic
baseline. Pre-registered C5 WIN gate: envelope-keyed config beats baseline by
≥10% on at least one task AND a random-envelope must do worse (constructive
falsifier — proves the win is envelope-attributed, not from extra structural
randomness).

### Six hypotheses we are about to test (or have implemented)
1. **Envelope-tuned sparsity**: derive target reservoir density from host
   power profile → optimal compute/memory ratio for THIS chip
2. **Substrate as natural dropout**: per-chip RTN/jitter pattern hashed into
   permanent dropout mask → free regularization that beats SGD-noise dropout
3. **Envelope-adaptive learning rate**: clock-stability bits → optimal LR
   schedule per chassis
4. **Per-position weight scaling from envelope**: per-CU latency variance →
   per-neuron weight scale matched to which CU runs which neuron
5. **Live envelope as noise schedule for variational inference**: chassi
   thermal noise IS the random sample, free entropy source
6. **Envelope-determined attention sparsity**: per-chip cache topology →
   attention pattern that matches data locality

### Direct questions
1. Of the 6 hypotheses, which is **most likely** to give measurable improvement
   on a small reservoir-computing benchmark (NARMA-10, Mackey-Glass-17, memory
   capacity)? Rank them. Brutal honesty.

2. Any **2024-2026 papers** that show **per-chip-tuned neural networks
   outperforming a generic baseline**? Specifically: per-die, per-chassi, or
   per-instance hardware fingerprint used as a structural prior, with
   measured accuracy or efficiency advantage. Include arXiv IDs / DOIs.

3. Has anyone trained models that **EXPLICITLY exploit silicon-specific
   compute irregularities** — per-CU latency, per-die FP-rounding behavior,
   chassis thermal RC — for accuracy or energy efficiency wins?
   (We are not asking about hardware-aware NAS in the usual sense — we want
   per-individual-die specialization, not per-architecture-class.)

4. What is the **cleanest experimental design** to demonstrate
   "envelope-keyed model BETTER than baseline" with a **constructive
   falsifier** (random-envelope must be worse)? Specifically:
   - What baseline should we use? (Vanilla determinisitic, or
     hyperparameter-matched random structure?)
   - What task class is most likely to show an envelope advantage?
   - How many seeds / replications to claim significance?

5. **Brutal honesty**: is "envelope binding = performance advantage" a fool's
   errand on a commodity x86 GPU?
   - We can BIND a model to a chassis (607-920× NRMSE degradation on
     transplant — completely uncontroversial result).
   - We cannot YET make the bound model BETTER than vanilla — only TIGHTER to
     the substrate. Is there a fundamental information-theoretic reason this
     should be impossible? Cite arguments.

### Additional context: D2 result
The single most-binding structural axis is **recurrent permutation** (920× by
itself). Suggests the model is binding via "where state ends up in the readout
vector" rather than via WHICH connections exist or HOW MUCH they weigh. Does
this change your answer about where to find the advantage?

## Output we want
- Markdown synthesis ≤ 800 words
- Direct ranking of the 6 hypotheses with one-sentence justification each
- 5-10 specific paper citations (arXiv ID + 1-line summary)
- Concrete recommended experiment(s) we should run next (≤ 5)
- A clear yes/no answer to question 5 (with reasoning)
