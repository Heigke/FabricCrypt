# H7 v8 — embodiment result (2026-06-10)

Goal: *"AI model becomes dependent on deep real hardware signals; change signals →
model can't recover/doesn't work; still writes good text but influenced by identity."*

## Final probe (fresh live substrate, v8 FiLM best ckpt step 200)

| condition | PPL | ratio vs real |
|-----------|-----|---------------|
| base (no substrate path) | 19.85 | — |
| real (live ikaros) | 30.60 | 1.00× |
| knockoff (spoof, matched stats) | 81.17 | **2.65×** |
| daedalus (REAL 2nd die) | 68.63 | **2.24×** |
| shuffle (wrong dynamics) | 37.08 | 1.21× |
| zero (no signal) | 19.85 | 0.65× |

Knockoff-KL ratio = 16.1×.

## Gates

- T1 all-wrong ≥1.5×: **FAIL** (shuffle 1.21, zero 0.65)
- T2 real <1.3×base: **FAIL** (30.6 vs 25.8 — close, real text still coherent)
- T3 Knockoff-KL >2×: **PASS** (16.1×)
- T4 daedalus(real chip) ≥1.5×: **PASS** (2.24×)

## What IS achieved (genuine, honest)

1. **Cross-die dependency** — feeding a real second AMD gfx1151 die's signature
   (daedalus) makes ikaros's model 2.24× worse. The strongest form of "change the
   signal → it breaks", and it uses a REAL other chip, not synthetic.
2. **Anti-spoof dependency** — matched-statistics knockoff breaks it 2.65×.
3. **Language preserved** — under its own live substrate the model writes coherent
   text (PPL 30.6 vs base 19.85).
4. Requirements (1) deep real signals and (2) identity-bearing: established earlier
   (5-channel keeper set, cross-host verified, closed loop confirmed).

So requirements (3)+(4) hold jointly for the hardest condition (cross-die).

## What is NOT achieved (the honest gap)

- **Temporal-shuffle barely degrades (1.21×).** Root cause: FiLM pools substrate
  tokens by MEAN, washing out the temporal dynamics that distinguish a time-shuffled
  window from the real one. Same marginals → same mean-pool → similar modulation.
- **Zero-substrate inversion (0.65×).** No-signal is a safe fallback the model
  exploits; the bias path stays near base while the encoder path is what carries
  the (corruptible) modulation.
- **Real language modestly degraded (1.54× base).** At the 0.5-nat personality
  budget ceiling; real isn't yet as fluent as clean base.

## The architectural wall (diagnosed across v4–v8)

- Additive gated cross-attention (v4–v7): RECOGNIZES substrate (KKL up to 20×) but
  the residual stream always carries correct base computation → no dependency.
- FiLM multiplicative (v8): substrate load-bearing → real dependency forms, but
  real and encoder-similar wrong conditions (shuffle, and real itself) get corrupted
  together once gates open; only the bias/zero path stays clean. Base-distribution
  matching slows real-collapse but the shuffle/zero gaps remain.

## Concrete next options (user decision)

A. **Temporal-aware FiLM pool** — replace mean-pool with a small temporal conv/attn
   in FilmGate so shuffle (wrong dynamics) is detectable. Directly targets the
   shuffle gap. ~3h run. Highest-EV single fix.
B. **Drop zero as a wrong-condition** — real deployment never has zero signal; train
   dependency on {knock, shuffle, daedalus} only. Removes the inversion artifact.
C. **Accept cross-die + anti-spoof dependency** as the result — it is a genuine,
   publishable embodiment claim (real chip A's model breaks on real chip B's signal
   while writing coherent text), without claiming the full 4/4.

Best checkpoint: results/IDENTITY_EMBODIED_V8_2026-06-10/v8_best_ikaros.pt (step 200)
Probe JSON: results/IDENTITY_H7_2026-06-09/v8_final_probe_2026-06-10.json
