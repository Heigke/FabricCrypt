# O101 — Cross-Attack A1+A3 Breakthrough Falsification Review (2026-05-30)

## Context

Two weeks of identity-bearing-computation attacks all collapsed under SHUFFLE control
(structure-bound, not device-bound). See O100 literature hunt: NO published commodity-
userspace constitutive HW-binding breakthrough exists.

Two days ago we ran **cross-attack A1+A3**: combine the contrastive dual-objective loss
(A1) with heavy-tail RTN/jitter substrate features (A3). Result on twin AMD Ryzen AI
Max+ 395 / gfx1151 (ikaros, daedalus):

- λ=1.0 contrastive weight
- Δ_HW = 8.5 (own-substrate vs other-substrate NRMSE delta on real device streams)
- Δ_SW-matched (SHUFFLE control, same first/second moments) = 0.6
- z = **5.74 → CONSTITUTIVE** by our pre-registered threshold

Critically, single-attack regimes:
- A1 alone (contrastive on Gaussian features): SHUFFLE Δ ≈ HW Δ (no binding)
- A3 alone (heavy-tail features, MSE loss): SHUFFLE Δ ≈ HW Δ (no binding)
- A1+A3 cross: 14× separation

Attachments (`_bundle_all_text.md` has all of these inline):
- `IDENTITY_ATTACK_1_3_2026-05-30.md` — full report
- `IDENTITY_LITERATURE_HUNT_2026-05-30.md` — O100 literature negative result
- `A1_results.json`, `A3_tail_stats.json`, `A13_cross.json` — numbers
- `A1_contrastive.py`, `A3_heavy_tail_transplant.py`, `A13_cross.py` — code

## HOSTILE QUESTIONS — answer in order, be brutal

1. Our cross-attack A1+A3 gave **z=5.74 CONSTITUTIVE** on commodity AMD APU userspace.
   Literature (O100) says this shouldn't work. **What's the ONE methodological confound
   most likely to explain this away?**

2. Is contrastive dual-objective loss the actual mechanism, or does it artificially
   inflate z by amplifying *any* structured feature difference?

3. Heavy-tail RTN/jitter — could the per-device heavy-tail signature actually be a
   **workload-artifact** (background process noise unique to each machine's daemon mix)
   rather than silicon-bound?

4. SHUFFLE Δ ≈ HW Δ in the HT-only regime. Why does adding contrastive loss flip this
   so dramatically? Is the dual loss simply training a classifier that the shuffle no
   longer fools?

5. What's the strongest falsification test we should run BEFORE writing a paper?
   **Rank these:**
   (a) Same-machine reboot test (z still high after reboot?)
   (b) Tails-only swap (hold spatial fixed, swap only tail-stats)
   (c) Third-twin replication (need minos online OR borrow another machine)
   (d) Independent re-implementation in different framework
   (e) Stale-data ablation (replay archived streams from days ago — does z hold?)

6. If our finding survives all 5 falsifiers, what's the strongest claim we can make
   and what would be unjustified?

7. The literature hunt found NO commodity-userspace constitutive transplant breakage.
   Our z=5.74 either (a) genuinely novel, (b) reproducing something known but
   mislabeled, (c) confound. **Probability estimate for each?**

## Output format

For each question: a short verdict line, then ≤8 sentences of reasoning. End with a
bullet list of "if you only run ONE more experiment before publication, run THIS".
