# Knockoff-KL on v5 ikaros checkpoint — verdict

## Numbers

- median KL(real_i || knockoff_i)   = **3.81e-7 nats**
- median KL(real_i || real_j, i≠j)  = 2.74e-7 nats
- ratio                              = **1.39×**
- q90 D_rk = 1.05e-6 ; q90 D_rr = 1.39e-6  (q90 *favors* D_rr — substrate-variance ≈ knockoff-variance at the tail)
- mean D_rk = 5.26e-7 ; mean D_rr = 5.41e-7  (mean essentially identical)
- gate αs at step 2000: **0.047, 0.056**

## Verdict: MARGINAL but de facto FAIL

The architecture is functioning: the v5 model produces slightly different output distributions under REAL vs KNOCKOFF substrate, and this difference is slightly larger than two REAL windows produce against each other (ratio 1.39×).

However the absolute scale is ~1e-7 nats — 6 orders of magnitude below the ~0.1 nat threshold where you'd say "two distributions are meaningfully different". The substrate is doing essentially nothing.

Combined with mean D_rk ≈ mean D_rr and q90 going the wrong way, this is **not embodiment** at the output level.

## Root cause: closed gates

Gate α at 0.047 / 0.056 means `tanh(α) ≈ α`, so the cross-attention contribution is ~5% of identity. Over 2000 update steps the embodied-prediction loss never drove α up because — as we now know — the loss was EMA-trivially explained and gave no learning signal.

## Plan

This makes the verdict from the closed-loop and EMA work conclusive: **v5's loss was the wrong objective**. We need a new training loss whose gradient *forces α open* only when substrate carries actionable information for output.

**v6 training loss candidate (pre-register):**

```
L_total = L_lm  +  λ_em · max(0, τ_em − KL(P(·|S_real) || P(·|S_knock)))
                                + γ · KL(P(·|S_real) || P_base)        # language anchor
```

- L_lm: standard next-token CE on a held-out corpus (e.g. Wikitext) under real S — keeps language alive
- Hinge on Knockoff-KL: actively push real-vs-knockoff KL ABOVE a threshold τ_em
- KL-to-base anchor: prevent collapse of language by limiting drift from frozen base when S is real

This makes embodiment a **training objective**, not an emergent property. Honest framing: we are *engineering* substrate-conditioning, not discovering it. That's fine — the falsification claim becomes "after training, the model behaviorally depends on substrate beyond first/second-order statistics", which the Knockoff-KL ratio measures cleanly.

## Bias check

Am I salvaging the project by changing the loss? Yes, but the change is principled: Knockoff-KL is the metric oracles independently proposed in O102, it's not trivially gameable by EMA convergence, and the architecture can already (weakly) express the discrimination. The honest alternative is to declare "embodiment in pure software with side-channel conditioning doesn't work" — but with KL ratio 1.39× existing already at α=0.05, the path to ratio ≥ 5× with α=0.5 looks open. Worth one more architectural attempt.

If v6 trained with this loss still gives Knockoff-KL ratio < 2× after a reasonable training budget, the embodiment hypothesis is falsified for this architecture class. That is the pre-registered kill condition.
