# O101 — Oracle Synthesis: Cross-Attack A1+A3 Breakthrough Review

**Date**: 2026-05-30
**Providers**: OpenAI (gpt-5), Gemini (gemini-2.5-pro), Grok (grok-4-latest), DeepSeek (deepseek-reasoner)
**Wall time**: ~3.5 min total

## Headline

**Consensus**: confound is the most likely explanation (P=55-70%).
**Specific confound** flagged by multiple oracles: **per-host `hash(host)`-seeded
spatial pattern + dual-loss = closed-world classifier on a deterministic software
artifact**. The "heavy tails" are likely workload/daemon noise, not silicon.

## Q7 — Probability estimates

| Oracle    | P(novel) | P(known-mislabeled) | P(confound) |
|-----------|----------|---------------------|-------------|
| OpenAI    | 0.35     | 0.10                | 0.55        |
| Gemini    | 0.25     | 0.05                | 0.70        |
| Grok      | 0.10     | 0.25                | 0.65        |
| DeepSeek  | 0.20     | 0.10                | 0.70        |
| **Mean**  | **0.23** | **0.13**            | **0.65**    |

## Q5 — Ranking of falsifiers

| Test                         | OpenAI | Gemini | Grok  | DeepSeek | Borda |
|------------------------------|--------|--------|-------|----------|-------|
| (a) Same-machine reboot      | 4      | 2      | 2     | 1        | 9     |
| (b) Tails-only swap          | 2      | 3      | 1     | 2        | 8     |
| (c) Third-twin replication   | 1      | 4      | 3     | 3        | 11    |
| (d) Indep re-implementation  | 5      | 5      | 5     | 5        | 20    |
| (e) Stale-data ablation      | 3      | 1      | 4     | 4        | 12    |

(Lower Borda = stronger. Ranks are oracle's strength-rank, summed.)

**Top 2 falsifiers by consensus**: (b) tails-only swap and (a) reboot test.
**Lowest priority**: (d) independent re-impl (good hygiene, weak falsifier).

## Specific confound mechanisms named

- **OpenAI**: "closed-world leakage: you trained the task head against daedalus
  features" — the contrastive loss is bound to the same negative class evaluated.
- **Gemini**: "stable software artifact of the operating system's state, not a
  primitive of the silicon."
- **Grok**: "SW-matched control is too weak (Gaussian draws preserve none of the
  heavy-tail marginals)."
- **DeepSeek**: "Spatial pattern artifact (per-host hash seeding) explains the
  effect, not silicon binding."

The DeepSeek and Gemini calls are sharpest: the per-host `hash(host)`-derived
spatial vector inside `HeavyTailSubstrate` is fully deterministic from the
string "ikaros" / "daedalus" and survives any tail-swap. The tails are mostly
decorative.

## Strongest claim ALLOWED (post-survival)

If all falsifiers pass: "Contrastive training on host-collected heavy-tail
latency streams produces readout weights whose NARMA performance degrades more
on a second host's streams than on a Gaussian surrogate matched only in
mean/variance."

NOT allowed: "constitutive silicon binding", "die-unique physical entropy",
generalisation beyond these two gfx1151 APUs and these four channels.

## Decision for Stage 2 pipeline

Given consensus P(confound)=0.65 AND the named mechanism (spatial seeding) being
directly testable in <2 min:

1. **F1 (tails-only swap) FIRST** — directly probes the named confound.
2. **F2 (stale-data) SECOND** — directly probes the OS-state-artifact hypothesis.
3. **F3 (independent reimpl) THIRD** — rules out implementation bug.
4. **F4 (reboot) ONLY IF F1+F2+F3 all keep z > 2.0** — oracle consensus ranks it
   2nd-3rd, but reboot is the costliest test and only diagnostic if earlier
   falsifiers haven't already killed the claim. **OpenAI explicitly ranks it
   LAST** ("least diagnostic").

If F1 or F2 collapses z, F4 reboot is moot — the confound is already named.

## ONE-experiment recommendation (each oracle)

- **OpenAI**: third-twin on minos + tails-only swap.
- **Gemini**: stale-data ablation.
- **Grok**: tails-only swap with first-four-moment matching.
- **DeepSeek**: same-machine reboot.

No unanimity. But (b) tails-only is the cheapest & most mechanistic; OpenAI and
Grok both name it. **Stage 2 runs F1 first.**
