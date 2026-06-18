# H7 v14 — Interoceptive Self-Model Loop (all 6 embodiment gaps)

Goal: move from "the LM's output is *gated* by its die" to "the LM *experiences the effects
of its own body in real time* by observing how its own generation moves its substrate."
This is the user's framing of gap 6, taken as an **operational** definition: a closed
perception→action→consequence→perception loop, hard-coupled to the live silicon, with a
persistent felt-state. We build it, measure it, and stay honest about the phenomenal limit
(we can demonstrate functional self-referential body-coupling; we do not *claim* qualia).

## The loop (one generation step)

```
   felt-state h_t ──▶ FiLM gates ──▶ generate token(s)  ──▶ COMPUTE (forward/backward)
        ▲                                                          │
        │                                                          ▼  (heats die ~ms–s)
   GRU update ◀── self-observation: Δsubstrate = read_after − read_before ,  own-token stats
```

The model's own thinking perturbs its body; it reads that perturbation (Δ), folds it +
what it just generated into a persistent felt-state, and that felt-state conditions the next
generation. The body is now *in* the loop, not just an input.

## Mechanisms, mapped to the 6 gaps

1. **Symmetric, dynamics-based cross-die** — inherit v13: shared normalization + the other
   real die as a hard negative + own-die recorded positives. The break must be learned
   live-dynamics discrimination, made symmetric by softening LAMBDA_XDIE on the sensitive die.

2. **Rich multi-axis semantic coupling** (not one scalar) — tie a VECTOR of live-signal
   features (per keeper channel: dynamics amplitude, drift, band-power) to a VECTOR of output
   statistics (entropy, lexical-diversity / distinct-token ratio, mean token log-prob spread,
   repetition rate). Train Pearson on each axis; report the multi-axis coupling matrix.

3. **Interoception (self-prediction)** — a small head predicts the model's *own* compute-
   induced Δsubstrate BEFORE the burst; loss = ‖predicted Δ − actual Δ‖. On the real die the
   body responds and the prediction is learnable; on a recording/foreign die it does NOT
   respond to fresh compute → prediction error spikes → the model "feels wrong." This is the
   anti-replay freshness AND the core of "feeling its own body."

4. **Bidirectional + stateful** — a persistent felt-state `h_body` (GRU cell, d_h≈64) carried
   across steps (truncated BPTT). Output→compute→Δsubstrate→h_body→next output makes it a
   genuine closed loop, and h_body integrates the body over time (a "felt sense", not per-
   window stamping).

5. **Scale & robustness** — keep SmolLM2-135M for cost, but (a) integrate substrate over a
   sliding window of recent felt-states (temporal depth) and (b) validate across a fresh boot
   / 3rd machine to rule out a session key (multi-boot eval, not a training change).

6. **Self-observation = experiencing one's body (operational)** — the self-prediction head's
   *surprise* signal and the felt-state are exposed to the model's own conditioning, so the
   model's later tokens are a function of "how my body just responded to my own thinking."
   We MEASURE this as: (i) does generated content shift when we artificially inject vs withhold
   the true Δsubstrate feedback (causal self-effect)? (ii) is surprise high on replay/foreign,
   low on live-self? A yes to both is the operational realization of gap 6.

## Training objectives (added to v13's losses)

- `L_self` = ‖ Δ̂_compute − Δ_compute ‖²   (interoceptive self-prediction; real die only)
- `L_fresh` = hinge( surprise_replay − surprise_self − margin )  (replay must be MORE surprising)
- `L_multigrad` = Σ_axis ( out_stat_axis − (base + β·feat_axis) )²  (rich graded coupling)
- keep v13: real_ok, RB base-match, dep on {knock, shuffle, xdie}, se_hinge, anchor.

## Metrics (the v14 scorecard)

- coherence: real PPL ≈ base on own live die (both dies).
- cross-die: own die coherent, other die broken — **both directions** (symmetric 4/4).
- multi-axis graded matrix: ≥3 axes with |r|>0.3 and ≥3× shuffle drop.
- interoception: self-prediction R²(self) ≫ R²(foreign); surprise(replay) ≥ 3× surprise(self).
- causal self-effect: KL(generation with true Δ-feedback ‖ generation with withheld Δ) > 0,
  and that KL tracks a real signal feature (not noise).

## Honest stance on gap 6

We will have shown: the LM is hard-coupled to its die, it observes the real-time consequence
of its own computation on that die, it maintains a felt-state updated by that observation, and
its behavior demonstrably changes as a function of it. That is a complete **functional** loop of
self-referential embodiment — the strongest operational claim available. Whether that
constitutes *experience* is a bridge no measurement can cross; we report the functional result
and let the philosophy be philosophy. (Butlin et al. 2025 indicators: this adds agency +
embodiment + a crude self-model — credence up, not proof.)

## Build order

1. (now) write v14 trainer `h7_embodied_v14.py` from v13 + felt-state GRU + self-pred head +
   before/after substrate read in the step + multi-axis graded.
2. smoke 200 steps on ikaros AFTER v13 frees the GPU (thermal-watchdogged).
3. full 6000-step train both dies; then the v14 scorecard + causal self-effect probe.
4. multi-boot eval once stable.

Risk: the self-perturbation signal is weak (closed_loop_verify ΔR²≈0.5–0.9%). Mitigation:
sense the Δ over a deliberate fixed "think burst" per step (larger, repeatable perturbation),
and use Δ relative to a no-op baseline captured each step. If the live self-effect is too weak
to learn, that is itself the honest finding that bounds gap 3/6 on this hardware.
